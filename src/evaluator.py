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

    # --- KV Cache 压力测试 ---
    kv_stress = eval_config.get("kv_stress_test", {})
    if kv_stress.get("enabled", False):
        results["kv_stress_test"] = {}

        # Passkey Retrieval
        pk_config = kv_stress.get("passkey_retrieval", {})
        if pk_config.get("enabled", True):
            print("\n🔑 评测 Passkey Retrieval")
            pk_result = evaluate_passkey_retrieval(model, tokenizer, pk_config)
            results["kv_stress_test"]["passkey_retrieval"] = pk_result

        # Generation PPL
        gp_config = kv_stress.get("generation_ppl", {})
        if gp_config.get("enabled", True):
            print("\n📝 评测 Generation PPL")
            gp_result = evaluate_generation_ppl(model, tokenizer, gp_config)
            results["kv_stress_test"]["generation_ppl"] = gp_result

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
        elif dataset_name == "longbench":
            # 长上下文 PPL：使用 pg19 长书籍文本
            # pg19-test: 100 条长书籍，每条 ~250K chars
            dataset = load_dataset("emozilla/pg19-test", split="test")
            # 取前 10 条长文档拼接（足够产生大量 token）
            texts = []
            for i, item in enumerate(dataset):
                if i >= 10:
                    break
                texts.append(item["text"])
            text = "\n\n".join(texts)
            # 使用更大的 max_seq_len
            long_ctx = config.get("eval", {}).get("long_context", {})
            if long_ctx.get("enabled", False):
                max_seq_len = long_ctx.get("max_seq_len", 32768)
                print(f"  📏 长上下文: 使用 max_seq_len={max_seq_len}")
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


def evaluate_passkey_retrieval(model: Any, tokenizer: Any, config: dict) -> dict:
    """
    Passkey Retrieval 评测 — KV Cache 压力测试。

    在长上下文中随机位置插入一个 5 位数字密钥，
    用 model.generate() 生成回答，检查是否能精确还原。

    KV Cache 会在 prefill 阶段增长到 context_length，
    然后在 decode 阶段继续增长 — 量化误差持续积累。
    """
    import torch
    import random

    num_keys = config.get("num_keys", 20)
    context_length = config.get("context_length", 2048)
    depths = config.get("depths", [0.1, 0.25, 0.5, 0.75])
    seed = config.get("seed", 42)

    random.seed(seed)
    torch.manual_seed(seed)

    # 填充句 — 用重复的无意义句子填充上下文
    filler = "The grass is green. The sky is blue. The sun is yellow. Today is a beautiful day. "

    results_by_depth = {str(d): {"correct": 0, "total": 0} for d in depths}
    all_results = []

    for trial in range(num_keys):
        passkey = str(random.randint(10000, 99999))

        for depth in depths:
            # 构建上下文: filler + passkey + filler + question
            question = f"\nWhat is the passkey? The passkey is "
            passkey_sentence = f"\nThe passkey to remember is {passkey}. Remember it.\n"

            # 估算 filler token 数以达到目标 context_length
            q_tokens = len(tokenizer.encode(question, add_special_tokens=False))
            pk_tokens = len(tokenizer.encode(passkey_sentence, add_special_tokens=False))
            filler_unit_tokens = len(tokenizer.encode(filler, add_special_tokens=False))

            target_filler_tokens = context_length - q_tokens - pk_tokens
            if target_filler_tokens <= 0:
                target_filler_tokens = 512

            # 在 depth 位置插入 passkey
            filler_before_tokens = int(target_filler_tokens * depth)
            filler_after_tokens = target_filler_tokens - filler_before_tokens

            repeats_before = max(1, filler_before_tokens // filler_unit_tokens)
            repeats_after = max(1, filler_after_tokens // filler_unit_tokens)

            text = (filler * repeats_before) + passkey_sentence + (filler * repeats_after) + question

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=context_length)
            input_ids = inputs["input_ids"].to(model.device)
            actual_len = input_ids.size(1)

            # Generate — KV Cache 在这里持续增长
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 提取生成的 token
            generated_ids = output_ids[0, input_ids.size(1):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 检查是否包含 passkey
            is_correct = passkey in generated_text
            depth_key = str(depth)
            results_by_depth[depth_key]["total"] += 1
            if is_correct:
                results_by_depth[depth_key]["correct"] += 1

            all_results.append({
                "trial": trial,
                "depth": depth,
                "passkey": passkey,
                "generated": generated_text[:50],
                "correct": is_correct,
                "context_tokens": actual_len,
            })

    # 汇总
    total_correct = sum(r["correct"] for r in results_by_depth.values())
    total_tests = sum(r["total"] for r in results_by_depth.values())
    overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0

    # 按 depth 计算准确率
    depth_accuracy = {}
    for d, r in results_by_depth.items():
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0.0
        depth_accuracy[d] = round(acc, 4)
        print(f"  深度 {d}: {r['correct']}/{r['total']} ({acc:.1%})")

    print(f"  总体准确率: {total_correct}/{total_tests} ({overall_accuracy:.1%})")

    return {
        "accuracy": round(overall_accuracy, 4),
        "depth_accuracy": depth_accuracy,
        "total_correct": total_correct,
        "total_tests": total_tests,
        "context_length": context_length,
        "num_keys": num_keys,
        "details": all_results[:10],  # 只保存前 10 条详情
    }


def evaluate_generation_ppl(model: Any, tokenizer: Any, config: dict) -> dict:
    """
    Generation PPL 评测 — KV Cache 压力测试。

    用长 prompt 调用 model.generate() 生成文本,
    然后用 teacher-forcing 计算生成部分的 PPL。

    KV Cache 从 prompt_length 持续增长到 prompt_length + gen_length,
    量化误差随 KV Cache 增长而累积。
    """
    import torch
    import math
    from datasets import load_dataset

    num_prompts = config.get("num_prompts", 5)
    prompt_length = config.get("prompt_length", 1500)
    gen_length = config.get("gen_length", 512)

    # 加载 PG19 长文本
    try:
        dataset = load_dataset("emozilla/pg19-test", split="test")
    except Exception as e:
        print(f"  ❌ 加载 PG19 失败: {e}")
        return {"gen_ppl": None, "error": str(e)}

    gen_ppls = []
    gen_texts_info = []

    for i in range(min(num_prompts, len(dataset))):
        text = dataset[i]["text"]

        # Tokenize 完整文本
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(full_ids) < prompt_length + gen_length:
            print(f"  ⚠️  样本 {i} 太短 ({len(full_ids)} tokens), 跳过")
            continue

        # 截取 prompt
        prompt_ids = torch.tensor([full_ids[:prompt_length]], device=model.device)
        # 参考续写 (用于计算 PPL)
        reference_ids = full_ids[prompt_length:prompt_length + gen_length]

        # Step 1: Generate — KV Cache 持续增长
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=gen_length,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, prompt_length:]
        generated_text = tokenizer.decode(generated_ids[:50], skip_special_tokens=True)

        # Step 2: 计算生成文本的 PPL (teacher-forcing)
        # 用完整序列 (prompt + reference) 做一次 forward, 只计算 reference 部分的 NLL
        full_input = torch.tensor(
            [full_ids[:prompt_length + gen_length]], device=model.device
        )
        labels = full_input.clone()
        labels[:, :prompt_length] = -100  # 只计算 reference 部分的 loss

        with torch.no_grad():
            outputs = model(full_input, labels=labels)
            nll = outputs.loss.item()

        ppl = math.exp(nll)
        gen_ppls.append(ppl)

        gen_texts_info.append({
            "sample": i,
            "prompt_tokens": prompt_length,
            "gen_tokens": len(generated_ids),
            "gen_ppl": round(ppl, 4),
            "generated_preview": generated_text[:100],
        })
        print(f"  样本 {i}: gen_ppl={ppl:.4f}, gen_tokens={len(generated_ids)}")

    if not gen_ppls:
        return {"gen_ppl": None, "error": "no valid samples"}

    avg_ppl = sum(gen_ppls) / len(gen_ppls)
    print(f"  平均 Generation PPL: {avg_ppl:.4f}")

    return {
        "gen_ppl": round(avg_ppl, 4),
        "num_samples": len(gen_ppls),
        "prompt_length": prompt_length,
        "gen_length": gen_length,
        "per_sample": gen_texts_info,
    }

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
