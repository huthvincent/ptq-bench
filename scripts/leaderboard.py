#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leaderboard.py — 从 results/ 目录汇总生成排行榜

扫描所有 .json 结果文件，按 Track 分组，
生成 results/leaderboard.md 排行榜，
并更新 summary.md 的 "当前最好结果" 区域。

用法:
    python scripts/leaderboard.py
    python scripts/leaderboard.py --results_dir results/ --top_k 3
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="PTQ Benchmark: 生成排行榜")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="结果文件目录 (默认: results/)")
    parser.add_argument("--output", type=str, default="results/leaderboard.md",
                        help="排行榜输出文件路径")
    parser.add_argument("--top_k", type=int, default=5,
                        help="每个 Track 每个模型展示 top-k 结果")
    parser.add_argument("--update_summary", action="store_true", default=True,
                        help="同时更新 summary.md 的最好结果区域")
    return parser.parse_args()


def load_all_results(results_dir: Path) -> list[dict]:
    """
    加载 results/ 目录下所有 .json 结果文件。

    参数:
        results_dir: 结果文件目录

    返回:
        list[dict]: 所有结果数据列表
    """
    results = []
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name == "leaderboard.json":
            continue  # 跳过排行榜自身
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = json_file.name
            results.append(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  跳过损坏的文件 {json_file.name}: {e}")
    return results


def extract_key_metrics(result: dict) -> dict:
    """
    从结果数据中提取关键指标。

    参数:
        result: 单个实验结果字典

    返回:
        dict: 关键指标摘要
    """
    meta = result.get("meta", {})
    results_data = result.get("results", {})

    metrics = {
        "model": meta.get("model", "?"),
        "method": meta.get("method", "?"),
        "track": meta.get("track", "?"),
        "timestamp": meta.get("timestamp", ""),
        "source_file": result.get("_source_file", ""),
    }

    # PPL
    ppl_data = results_data.get("ppl", {})
    for dataset, vals in ppl_data.items():
        if isinstance(vals, dict) and "ppl" in vals:
            metrics[f"ppl_{dataset}"] = vals["ppl"]

    # lm-eval 平均准确率
    lm_eval = results_data.get("lm_eval", {})
    if "_avg_accuracy" in lm_eval:
        metrics["avg_accuracy"] = lm_eval["_avg_accuracy"]

    # 各任务分数
    for task, task_res in lm_eval.items():
        if task.startswith("_") or not isinstance(task_res, dict):
            continue
        for key in ("acc,none", "acc_norm,none", "exact_match,none"):
            if key in task_res:
                metrics[f"lm_{task}"] = task_res[key]
                break

    # 系统指标
    sys_metrics = results_data.get("system_metrics", {})
    if "vram_peak_mb" in sys_metrics:
        metrics["vram_peak_mb"] = sys_metrics["vram_peak_mb"]

    # 警告标记
    warnings = result.get("warnings", [])
    metrics["has_warnings"] = len(warnings) > 0

    return metrics


def generate_leaderboard(all_results: list[dict], top_k: int = 5) -> str:
    """
    生成排行榜 Markdown 内容。

    参数:
        all_results: 所有结果数据
        top_k: 每个 Track 每个模型展示前 k 名

    返回:
        str: Markdown 格式的排行榜
    """
    # 提取指标
    all_metrics = [extract_key_metrics(r) for r in all_results]

    # 按 Track 分组
    by_track = defaultdict(list)
    for m in all_metrics:
        by_track[m["track"]].append(m)

    lines = []
    lines.append("# 📊 PTQ Benchmark 排行榜")
    lines.append("")
    lines.append(f"*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append(f"总实验数: {len(all_metrics)}")
    lines.append("")

    for track in sorted(by_track.keys()):
        track_data = by_track[track]
        lines.append(f"## Track {track}")
        lines.append("")

        # 按模型分组
        by_model = defaultdict(list)
        for m in track_data:
            by_model[m["model"]].append(m)

        for model_name in sorted(by_model.keys()):
            model_data = by_model[model_name]

            # 按 avg_accuracy 排序（降序），如果没有则按 PPL 排序（升序）
            def sort_key(m):
                if "avg_accuracy" in m:
                    return -m["avg_accuracy"]  # 负号使降序
                if "ppl_wikitext2" in m:
                    return m["ppl_wikitext2"]  # PPL 越低越好
                return float("inf")

            model_data.sort(key=sort_key)
            model_data = model_data[:top_k]

            lines.append(f"### {model_name}")
            lines.append("")

            # 生成表格
            header = "| 排名 | 方法 | PPL (WikiText-2) | Avg Accuracy | VRAM (MB) | ⚠️ | 结果文件 |"
            separator = "|------|------|-----------------|-------------|-----------|---|---------|"
            lines.append(header)
            lines.append(separator)

            for rank, m in enumerate(model_data, 1):
                ppl = m.get("ppl_wikitext2", "-")
                if isinstance(ppl, float):
                    ppl = f"{ppl:.2f}"
                acc = m.get("avg_accuracy", "-")
                if isinstance(acc, float):
                    acc = f"{acc:.4f}"
                vram = m.get("vram_peak_mb", "-")
                if isinstance(vram, float):
                    vram = f"{vram:.0f}"
                warn = "⚠️" if m.get("has_warnings") else ""
                source = m.get("source_file", "")
                lines.append(f"| {rank} | {m['method']} | {ppl} | {acc} | {vram} | {warn} | {source} |")

            lines.append("")

    if not by_track:
        lines.append("*暂无实验结果。请先运行 `bash scripts/run_one.sh` 生成结果。*")
        lines.append("")

    return "\n".join(lines)


def generate_summary_snippet(all_results: list[dict]) -> str:
    """
    生成用于 summary.md 的最好结果摘要。

    参数:
        all_results: 所有结果数据

    返回:
        str: 摘要文本
    """
    all_metrics = [extract_key_metrics(r) for r in all_results]

    lines = []
    lines.append("### 当前最好结果")
    lines.append("")
    lines.append(f"*更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    by_track = defaultdict(list)
    for m in all_metrics:
        by_track[m["track"]].append(m)

    for track in sorted(by_track.keys()):
        track_data = by_track[track]

        def sort_key(m):
            if "avg_accuracy" in m:
                return -m["avg_accuracy"]
            if "ppl_wikitext2" in m:
                return m["ppl_wikitext2"]
            return float("inf")

        track_data.sort(key=sort_key)
        if track_data:
            best = track_data[0]
            ppl = best.get("ppl_wikitext2", "N/A")
            acc = best.get("avg_accuracy", "N/A")
            lines.append(f"- **Track {track}** 最佳: {best['method']} on {best['model']} (PPL={ppl}, Acc={acc})")

    lines.append("")
    lines.append("详细排行榜见 [results/leaderboard.md](results/leaderboard.md)")

    return "\n".join(lines)


def main():
    """主入口函数。"""
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    print(f"📊 扫描结果目录: {results_dir}")
    all_results = load_all_results(results_dir)
    print(f"   找到 {len(all_results)} 个结果文件")

    # 生成排行榜
    leaderboard_md = generate_leaderboard(all_results, top_k=args.top_k)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(leaderboard_md)
    print(f"✅ 排行榜已生成: {output_path}")

    # 更新 summary.md
    if args.update_summary and all_results:
        snippet = generate_summary_snippet(all_results)
        print(f"\n📋 summary.md 最好结果摘要:")
        print(snippet)
        print("\n💡 请手动将以上内容更新到 summary.md 的 '当前最好结果' 区域")


if __name__ == "__main__":
    main()
