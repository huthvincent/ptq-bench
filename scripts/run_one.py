#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_one.py — 运行单个实验

运行一个 model × method × track 组合的完整实验流程：
加载模型 → 量化 → 评测 → 保存结果

用法示例:
    python scripts/run_one.py --model llama3.1-8b --method gptq --track A
    python scripts/run_one.py --model llama3.1-8b --method fp16 --track A --dry_run

所有参数都可以通过 CLI 覆盖 YAML 配置中的默认值。
"""

import sys
import argparse
from pathlib import Path

# 将项目根目录加入 Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import registry
from src.runner import run_experiment
from src.config import dump_config, load_global_config, load_model_config, load_method_config, load_track_config, merge_configs, resolve_paths


def parse_args():
    """
    解析命令行参数。

    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="PTQ Benchmark: 运行单个实验（model × method × track）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 跑 Track A: Llama-3.1-8B + GPTQ
  python scripts/run_one.py --model llama3.1-8b --method gptq --track A

  # 只打印配置不运行
  python scripts/run_one.py --model llama3.1-8b --method gptq --track A --dry_run

  # 覆盖量化参数
  python scripts/run_one.py --model llama3.1-8b --method gptq --track A --w_bits 3 --group_size 64

  # 指定输出目录
  python scripts/run_one.py --model llama3.1-8b --method fp16 --track A --output_dir results/test
        """,
    )

    # === 核心选择项 ===
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称，对应 configs/models/{name}.yaml")
    parser.add_argument("--method", type=str, required=True,
                        help="量化方法名称，对应 configs/methods/{name}.yaml")
    parser.add_argument("--track", type=str, required=True, choices=["A", "B", "C"],
                        help="赛道: A (W4A16), B (W8A8), C (KV Cache)")

    # === 量化参数覆盖 ===
    quant_group = parser.add_argument_group("量化参数覆盖")
    quant_group.add_argument("--w_bits", type=int, help="权重量化位数")
    quant_group.add_argument("--group_size", type=int, help="量化分组大小")
    quant_group.add_argument("--scheme", type=str, choices=["symmetric", "asymmetric"],
                             help="量化方案")
    quant_group.add_argument("--a_bits", type=int, help="激活量化位数 (Track B)")
    quant_group.add_argument("--smoothquant_alpha", type=float,
                             help="SmoothQuant 迁移强度 (Track B)")

    # === 校准参数覆盖 ===
    calib_group = parser.add_argument_group("校准参数覆盖")
    calib_group.add_argument("--calib_dataset", type=str, choices=["wikitext2", "c4"],
                             help="校准数据集")
    calib_group.add_argument("--num_samples", type=int, help="校准样本数")
    calib_group.add_argument("--seq_len", type=int, help="校准序列长度")
    calib_group.add_argument("--seed", type=int, help="随机种子")

    # === 输出控制 ===
    out_group = parser.add_argument_group("输出控制")
    out_group.add_argument("--exp_name", type=str, help="实验名称（可选）")
    out_group.add_argument("--output_dir", type=str, default=None,
                           help="结果输出目录 (默认: results/)")
    out_group.add_argument("--dry_run", action="store_true",
                           help="只打印合并后的配置，不实际运行")
    out_group.add_argument("--print_config", action="store_true",
                           help="运行前打印最终配置")

    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> dict:
    """
    将 CLI 参数转换为配置覆盖字典。

    只包含用户显式指定的参数（非 None）。

    参数:
        args: 解析后的命令行参数

    返回:
        dict: 配置覆盖字典
    """
    overrides = {}

    # 量化参数覆盖
    weight_overrides = {}
    if args.w_bits is not None:
        weight_overrides["w_bits"] = args.w_bits
    if args.group_size is not None:
        weight_overrides["group_size"] = args.group_size
    if args.scheme is not None:
        weight_overrides["scheme"] = args.scheme
    if weight_overrides:
        overrides["weight"] = weight_overrides

    # 激活参数覆盖
    activation_overrides = {}
    if args.a_bits is not None:
        activation_overrides["a_bits"] = args.a_bits
    if args.smoothquant_alpha is not None:
        activation_overrides["smoothquant_alpha"] = args.smoothquant_alpha
    if activation_overrides:
        overrides["activation"] = activation_overrides

    # 校准参数覆盖
    calib_overrides = {}
    if args.calib_dataset is not None:
        calib_overrides["dataset"] = args.calib_dataset
    if args.num_samples is not None:
        calib_overrides["num_samples"] = args.num_samples
    if args.seq_len is not None:
        calib_overrides["seq_len"] = args.seq_len
    if args.seed is not None:
        calib_overrides["seed"] = args.seed
        overrides.setdefault("common_hyperparams", {})["seed"] = args.seed
    if calib_overrides:
        overrides["calibration"] = calib_overrides

    return overrides


def main():
    """主入口函数。"""
    args = parse_args()

    # 自动发现并注册所有量化方法
    registry.auto_discover()

    print("=" * 60)
    print(f"🚀 PTQ Benchmark: run_one")
    print(f"   模型: {args.model}")
    print(f"   方法: {args.method}")
    print(f"   赛道: Track {args.track}")
    print("=" * 60)

    # 构建 CLI 覆盖
    cli_overrides = build_cli_overrides(args)

    # 构建完整的 CLI 参数字符串（用于记录）
    cli_args_str = " ".join(sys.argv)

    # 如果需要打印配置
    if args.print_config and not args.dry_run:
        project_root = Path(__file__).resolve().parent.parent
        global_config = load_global_config(project_root)
        model_config = load_model_config(args.model, project_root)
        method_config = load_method_config(args.method, project_root)
        track_config = load_track_config(args.track, project_root)
        overrides_with_track = {"track": args.track.upper()}
        overrides_with_track.update(cli_overrides)
        merged = merge_configs(global_config, model_config, method_config, track_config, overrides_with_track)
        merged = resolve_paths(merged, project_root)
        print("\n📋 最终合并配置:")
        print("-" * 40)
        print(dump_config(merged))
        print("-" * 40)

    # 运行实验
    results = run_experiment(
        model_name=args.model,
        method_name=args.method,
        track=args.track,
        cli_overrides=cli_overrides,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        cli_args_str=cli_args_str,
        script_name="scripts/run_one.py",
    )

    if results and not args.dry_run:
        print("\n✅ 实验完成!")
    elif args.dry_run:
        print("\n🔍 [DRY RUN] 完成")


if __name__ == "__main__":
    main()
