#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — 批量运行多个实验

支持两种模式:
1. 笛卡尔积: 指定 models × methods × tracks 的所有组合
2. 实验配置: 从 experiment YAML 读取具体组合

用法示例:
    # 跑所有 Track A 方法 × 所有模型
    python scripts/run_all.py --include_tracks A

    # 使用实验配置
    python scripts/run_all.py --experiment configs/experiments/quick_test.yaml

    # 跳过已有结果
    python scripts/run_all.py --include_tracks A --resume
"""

import sys
import argparse
import itertools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import registry
from src.config import load_global_config, load_experiment_config, get_project_root
from src.runner import run_experiment


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PTQ Benchmark: 批量运行多个实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === 笛卡尔积模式 ===
    parser.add_argument("--include_models", nargs="+", default=None,
                        help="要包含的模型列表（默认: 全部）")
    parser.add_argument("--exclude_models", nargs="+", default=None,
                        help="要排除的模型列表")
    parser.add_argument("--include_methods", nargs="+", default=None,
                        help="要包含的方法列表（默认: 全部）")
    parser.add_argument("--exclude_methods", nargs="+", default=None,
                        help="要排除的方法列表")
    parser.add_argument("--include_tracks", nargs="+", default=None,
                        choices=["A", "B", "C"],
                        help="要包含的赛道列表（默认: 全部）")

    # === 实验配置模式 ===
    parser.add_argument("--experiment", type=str, default=None,
                        help="实验配置 YAML 路径")

    # === 运行控制 ===
    parser.add_argument("--max_jobs", type=int, default=1,
                        help="最大并发数（当前仅支持顺序执行=1）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已有结果文件的实验")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="结果输出目录")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印计划运行的实验列表，不实际运行")

    return parser.parse_args()


def discover_available_configs(project_root: Path) -> dict:
    """
    扫描 configs/ 目录，发现所有可用的模型、方法配置。

    参数:
        project_root: 项目根目录

    返回:
        dict: {"models": [...], "methods": [...]}
    """
    models_dir = project_root / "configs" / "models"
    methods_dir = project_root / "configs" / "methods"

    models = sorted([
        f.stem for f in models_dir.glob("*.yaml") if f.stem != "README"
    ])
    methods = sorted([
        f.stem for f in methods_dir.glob("*.yaml") if f.stem != "README"
    ])

    return {"models": models, "methods": methods}


def check_existing_results(output_dir: Path, model: str, method: str, track: str) -> bool:
    """
    检查是否已有该实验的结果文件。

    参数:
        output_dir: 结果目录
        model: 模型名
        method: 方法名
        track: 赛道名

    返回:
        bool: 是否存在结果
    """
    pattern = f"*__{model}__{method}__track{track.upper()}.json"
    return bool(list(output_dir.glob(pattern)))


def main():
    """主入口函数。"""
    args = parse_args()
    project_root = get_project_root()

    # 自动发现并注册所有量化方法
    registry.auto_discover()

    # ================================================================
    # 构建实验组合列表
    # ================================================================
    if args.experiment:
        # 从 experiment YAML 读取
        exp_config = load_experiment_config(args.experiment, project_root)
        models = exp_config.get("models", [])
        methods = exp_config.get("methods", [])
        tracks = exp_config.get("tracks", ["A"])
    else:
        # 从 configs/ 目录发现
        available = discover_available_configs(project_root)
        models = args.include_models or available["models"]
        methods = args.include_methods or available["methods"]
        tracks = args.include_tracks or ["A", "B", "C"]

    # 应用排除过滤
    if args.exclude_models:
        models = [m for m in models if m not in args.exclude_models]
    if args.exclude_methods:
        methods = [m for m in methods if m not in args.exclude_methods]

    # 构建笛卡尔积，并过滤不兼容的组合（方法不支持的 Track 跳过）
    experiments = []
    for model, method, track in itertools.product(models, methods, tracks):
        # 检查方法是否支持该赛道
        try:
            method_cls = registry.get(method)
            supported = getattr(method_cls, "supported_tracks", [])
            if track.upper() not in [t.upper() for t in supported]:
                continue
        except KeyError:
            continue
        experiments.append((model, method, track))

    # ================================================================
    # 输出计划
    # ================================================================
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results"

    print("=" * 60)
    print(f"🚀 PTQ Benchmark: run_all")
    print(f"   模型: {models}")
    print(f"   方法: {methods}")
    print(f"   赛道: {tracks}")
    print(f"   总实验数: {len(experiments)}")
    print(f"   输出目录: {output_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n🔍 [DRY RUN] 计划运行的实验:")
        for i, (model, method, track) in enumerate(experiments, 1):
            skip = ""
            if args.resume and check_existing_results(output_dir, model, method, track):
                skip = " [跳过: 已有结果]"
            print(f"  {i}. {model} × {method} × Track {track}{skip}")
        print(f"\n总计: {len(experiments)} 个实验")
        return

    # ================================================================
    # 顺序执行实验
    # ================================================================
    completed = 0
    skipped = 0
    failed = 0

    for i, (model, method, track) in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"📋 实验 {i}/{len(experiments)}: {model} × {method} × Track {track}")
        print(f"{'='*60}")

        # resume 模式: 跳过已有结果
        if args.resume and check_existing_results(output_dir, model, method, track):
            print(f"⏭️  跳过: 已有结果文件")
            skipped += 1
            continue

        try:
            cli_args_str = f"python scripts/run_all.py {' '.join(sys.argv[1:])}"
            result = run_experiment(
                model_name=model,
                method_name=method,
                track=track,
                output_dir=output_dir,
                cli_args_str=cli_args_str,
                script_name="scripts/run_all.py",
            )
            if result:
                completed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            failed += 1

    # ================================================================
    # 汇总
    # ================================================================
    print(f"\n{'='*60}")
    print(f"📊 批量运行汇总:")
    print(f"   ✅ 完成: {completed}")
    print(f"   ⏭️  跳过: {skipped}")
    print(f"   ❌ 失败: {failed}")
    print(f"   总计: {len(experiments)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
