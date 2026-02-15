# -*- coding: utf-8 -*-
"""
评测引擎

负责对量化后的模型进行评测，包括：
- PPL (Perplexity) 评测：在 WikiText-2 / C4 上计算困惑度
- lm-eval-harness 任务评测：MMLU、GSM8K、HellaSwag 等
- 系统指标评测（Phase 2）：TTFT、吞吐量、VRAM 峰值

主要入口函数：
- evaluate(): 根据配置运行全部评测
- evaluate_ppl(): 只跑 PPL
- evaluate_lm_eval(): 只跑 lm-eval 任务
"""

import time
import json
from typing import Any
from pathlib import Path


def evaluate(model: Any, tokenizer: Any, config: dict) -> dict:
    """
    根据配置运行全部评测。

    这是评测的主入口函数，会根据配置中的 eval 设置
    依次运行 PPL 评测和 lm-eval 任务评测。

    参数:
        model: 量化后（或 FP16 baseline）的模型
        tokenizer: 对应的 tokenizer
        config: 合并后的完整配置字典

    返回:
        dict: 评测结果字典，包含 ppl、lm_eval_results、system_metrics 等
    """
    results = {
        "ppl": {},
        "lm_eval": {},
        "system_metrics": {},
        "eval_time_seconds": 0,
    }

    start_time = time.time()

    # --- PPL 评测 ---
    eval_config = config.get("eval", config.get("default_eval", {}))
    # 兼容两种嵌套结构:
    # 1. default_eval 风格: eval.core_quality.ppl_datasets
    # 2. track 风格: eval.ppl_datasets (平铺)
    core_quality = eval_config.get("core_quality", {})
    ppl_datasets = core_quality.get("ppl_datasets", eval_config.get("ppl_datasets", ["wikitext2"]))

    for dataset_name in ppl_datasets:
        print(f"\n📊 评测 PPL: {dataset_name}")
        ppl = evaluate_ppl(model, tokenizer, dataset_name, config)
        results["ppl"][dataset_name] = ppl

    # --- lm-eval 任务评测 ---
    lm_eval_tasks = core_quality.get("lm_eval_tasks", eval_config.get("lm_eval_tasks", []))
    if lm_eval_tasks:
        print(f"\n📊 评测 lm-eval 任务: {', '.join(lm_eval_tasks)}")
        lm_eval_results = evaluate_lm_eval(model, tokenizer, lm_eval_tasks, config)
        results["lm_eval"] = lm_eval_results

    # --- 系统指标（Phase 2）---
    system_config = eval_config.get("system_metrics", {})
    if isinstance(system_config, dict) and system_config.get("enabled", False):
        print("\n📊 系统指标评测（Phase 2 功能）")
        results["system_metrics"] = {"status": "phase2_not_implemented"}

    # --- VRAM 峰值（如果 torch 可用）---
    try:
        import torch
        if torch.cuda.is_available():
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            results["system_metrics"]["vram_peak_mb"] = round(vram_peak_mb, 1)
            print(f"📊 VRAM 峰值: {vram_peak_mb:.1f} MB")
    except ImportError:
        pass

    results["eval_time_seconds"] = round(time.time() - start_time, 1)
    return results


def evaluate_ppl(model: Any, tokenizer: Any, dataset_name: str, config: dict) -> dict:
    """
    在指定数据集上计算模型的 PPL (Perplexity)。

    使用标准的 sliding window 方法计算 PPL：
    将测试文本 tokenize 后，按 max_seq_len 滑动窗口计算 NLL，
    最终取 exp(avg_nll) 作为 PPL。

    参数:
        model: 模型
        tokenizer: tokenizer
        dataset_name: 数据集名称（"wikitext2" 或 "c4"）
        config: 配置字典

    返回:
        dict: {"ppl": float, "nll": float, "num_tokens": int}
    """
    max_seq_len = config.get("common_hyperparams", {}).get("max_seq_len", 2048)

    try:
        import torch
        from datasets import load_dataset

        # 加载数据集
        if dataset_name == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(dataset["text"])
        elif dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            # C4 很大，只取前 256 条
            texts = []
            for i, item in enumerate(dataset):
                if i >= 256:
                    break
                texts.append(item["text"])
            text = "\n\n".join(texts)
        else:
            print(f"⚠️  未知数据集 {dataset_name}，跳过 PPL 评测")
            return {"ppl": None, "error": f"unknown dataset: {dataset_name}"}

        # Tokenize
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(model.device)
        seq_len = input_ids.size(1)

        print(f"  数据集 token 数: {seq_len}")

        # Sliding window PPL 计算
        nlls = []
        stride = max_seq_len // 2  # 使用 50% 重叠
        for begin in range(0, seq_len - max_seq_len, stride):
            end = begin + max_seq_len
            input_chunk = input_ids[:, begin:end]
            target_chunk = input_chunk.clone()

            # 只计算非重叠部分的 loss（除了第一个窗口）
            if begin > 0:
                target_chunk[:, :stride] = -100

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_chunk)
                nll = outputs.loss.item()
                nlls.append(nll)

        import math
        avg_nll = sum(nlls) / len(nlls) if nlls else float("inf")
        ppl = math.exp(avg_nll)

        print(f"  PPL: {ppl:.2f}")
        return {"ppl": round(ppl, 4), "nll": round(avg_nll, 6), "num_windows": len(nlls)}

    except ImportError as e:
        print(f"⚠️  PPL 评测依赖缺失: {e}")
        return {"ppl": None, "error": str(e)}
    except Exception as e:
        print(f"❌ PPL 评测出错: {e}")
        return {"ppl": None, "error": str(e)}


def evaluate_lm_eval(model: Any, tokenizer: Any, tasks: list[str], config: dict) -> dict:
    """
    使用 lm-evaluation-harness 评测模型在多个任务上的表现。

    参数:
        model: 模型
        tokenizer: tokenizer
        tasks: 任务名称列表，如 ["mmlu", "hellaswag", "gsm8k"]
        config: 配置字典

    返回:
        dict: 每个任务的评测结果
    """
    fewshot_map = config.get("common_hyperparams", {}).get("eval_default_fewshot", {})

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        print(f"  使用 lm-eval-harness v{lm_eval.__version__}")

        # 包装模型为 lm-eval 格式
        lm = HFLM(pretrained=model, tokenizer=tokenizer)

        # 构建任务参数
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=None,  # 使用各任务默认值
            batch_size="auto",
        )

        # 提取结果
        task_results = {}
        for task_name, task_result in results.get("results", {}).items():
            task_results[task_name] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in task_result.items()
            }

        # 计算平均准确率
        accuracies = []
        for task_name, res in task_results.items():
            # lm-eval 的结果键名可能是 acc, acc_norm, exact_match 等
            for key in ("acc,none", "acc_norm,none", "exact_match,none"):
                if key in res:
                    accuracies.append(res[key])
                    break

        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            task_results["_avg_accuracy"] = round(avg_acc, 4)
            print(f"  平均准确率: {avg_acc:.4f}")

        return task_results

    except ImportError as e:
        print(f"⚠️  lm-eval-harness 未安装: {e}")
        print("    请安装: pip install lm-eval")
        return {"error": str(e)}
    except Exception as e:
        print(f"❌ lm-eval 评测出错: {e}")
        return {"error": str(e)}
