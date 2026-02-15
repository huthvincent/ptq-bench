# -*- coding: utf-8 -*-
"""
结果写入器

负责将实验结果写入 .md（人类阅读） 和 .json（机器解析） 两种格式。

命名规则：
    YYYYMMDD_HHMMSS__{model}__{method}__{track}.md
    YYYYMMDD_HHMMSS__{model}__{method}__{track}.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.env_info import collect_env_info, format_env_info


def generate_result_filename(model_name: str, method_name: str, track: str) -> str:
    """
    生成结果文件名（不含扩展名）。

    格式: YYYYMMDD_HHMMSS__{model}__{method}__{track}

    参数:
        model_name: 模型名称
        method_name: 方法名称
        track: 赛道名称

    返回:
        str: 文件名（不含扩展名和目录路径）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}__{model_name}__{method_name}__track{track.upper()}"


def write_results(
    results: dict,
    config: dict,
    output_dir: str | Path,
    cli_args: str = "",
    script_name: str = "",
    warnings: list[str] | None = None,
) -> tuple[Path, Path]:
    """
    将实验结果写入 .md 和 .json 文件。

    参数:
        results: 评测结果字典（来自 evaluator.evaluate()）
        config: 合并后的完整配置字典
        output_dir: 输出目录路径
        cli_args: 完整的命令行参数字符串
        script_name: 运行的脚本名称
        warnings: 运行过程中的警告信息列表

    返回:
        tuple[Path, Path]: (md_path, json_path) 两个文件的路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.get("model", {}).get("name", "unknown")
    method_name = config.get("name", "unknown")
    track = config.get("track", "?")

    basename = generate_result_filename(model_name, method_name, track)
    md_path = output_dir / f"{basename}.md"
    json_path = output_dir / f"{basename}.json"

    # 收集环境信息
    env_info = collect_env_info()

    # --- 写入 JSON ---
    json_data = {
        "meta": {
            "filename": basename,
            "timestamp": datetime.now().isoformat(),
            "script": script_name,
            "cli_args": cli_args,
            "model": model_name,
            "method": method_name,
            "track": track,
        },
        "config": config,
        "results": results,
        "env": env_info,
        "warnings": warnings or [],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)

    # --- 写入 Markdown ---
    md_content = _render_markdown(json_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n📄 结果已保存:")
    print(f"   MD:   {md_path}")
    print(f"   JSON: {json_path}")

    return md_path, json_path


def _render_markdown(data: dict) -> str:
    """
    将结果数据渲染为 Markdown 格式。

    参数:
        data: 包含 meta、config、results、env 的完整数据字典

    返回:
        str: 格式化的 Markdown 内容
    """
    meta = data["meta"]
    config = data["config"]
    results = data["results"]
    env_info = data["env"]
    warnings = data.get("warnings", [])

    lines = []

    # === 标题 ===
    lines.append(f"# 实验结果: {meta['model']} + {meta['method']} (Track {meta['track']})")
    lines.append("")

    # === 警告 ===
    if warnings:
        for w in warnings:
            lines.append(f"> ⚠️ **警告**: {w}")
        lines.append("")

    # === 运行信息 ===
    lines.append("## 运行信息")
    lines.append("")
    lines.append(f"- **运行时间**: {meta.get('timestamp', 'N/A')}")
    lines.append(f"- **脚本**: `{meta.get('script', 'N/A')}`")
    lines.append(f"- **完整 CLI 参数**:")
    lines.append(f"  ```")
    lines.append(f"  {meta.get('cli_args', 'N/A')}")
    lines.append(f"  ```")
    lines.append("")

    # === 数据集信息 ===
    lines.append("## 数据集")
    lines.append("")
    calib = config.get("calibration", config.get("default_calibration", {}))
    lines.append(f"- **校准数据集**: {calib.get('dataset', 'N/A')}")
    lines.append(f"- **校准样本数**: {calib.get('num_samples', 'N/A')}")
    lines.append(f"- **校准序列长度**: {calib.get('seq_len', 'N/A')}")

    eval_config = config.get("eval", config.get("default_eval", {}))
    core = eval_config.get("core_quality", {})
    lines.append(f"- **PPL 评测数据集**: {', '.join(core.get('ppl_datasets', []))}")
    lines.append(f"- **lm-eval 任务**: {', '.join(core.get('lm_eval_tasks', []))}")
    lines.append("")

    # === 量化参数 ===
    lines.append("## 量化参数")
    lines.append("")
    lines.append(f"- **方法**: {meta['method']}")
    lines.append(f"- **赛道**: Track {meta['track']}")
    for key in ("weight", "activation", "kv"):
        if key in config:
            params = config[key]
            lines.append(f"- **{key}**: {json.dumps(params, ensure_ascii=False)}")
    lines.append(f"- **Seed**: {config.get('common_hyperparams', {}).get('seed', 'N/A')}")
    lines.append("")

    # === PPL 结果 ===
    ppl_results = results.get("ppl", {})
    if ppl_results:
        lines.append("## PPL 结果")
        lines.append("")
        lines.append("| 数据集 | PPL | NLL |")
        lines.append("|--------|-----|-----|")
        for dataset, vals in ppl_results.items():
            if isinstance(vals, dict):
                ppl_val = vals.get("ppl", "N/A")
                nll_val = vals.get("nll", "N/A")
                lines.append(f"| {dataset} | {ppl_val} | {nll_val} |")
        lines.append("")

    # === lm-eval 结果 ===
    lm_eval_results = results.get("lm_eval", {})
    if lm_eval_results and "error" not in lm_eval_results:
        lines.append("## lm-eval 任务结果")
        lines.append("")
        lines.append("| 任务 | 指标 | 分数 |")
        lines.append("|------|------|------|")
        for task_name, task_res in lm_eval_results.items():
            if task_name.startswith("_"):
                continue  # 跳过 _avg_accuracy 等元字段
            if isinstance(task_res, dict):
                for metric, score in task_res.items():
                    lines.append(f"| {task_name} | {metric} | {score} |")
        avg_acc = lm_eval_results.get("_avg_accuracy")
        if avg_acc is not None:
            lines.append(f"\n**平均准确率**: {avg_acc}")
        lines.append("")

    # === 系统指标 ===
    sys_metrics = results.get("system_metrics", {})
    if sys_metrics:
        lines.append("## 系统指标")
        lines.append("")
        vram = sys_metrics.get("vram_peak_mb")
        if vram:
            lines.append(f"- **VRAM 峰值**: {vram} MB")
        eval_time = results.get("eval_time_seconds")
        if eval_time:
            lines.append(f"- **评测总耗时**: {eval_time} 秒")
        lines.append("")

    # === 环境信息 ===
    lines.append(format_env_info(env_info))
    lines.append("")

    return "\n".join(lines)
