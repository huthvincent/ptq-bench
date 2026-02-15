# -*- coding: utf-8 -*-
"""
运行控制器

组装完整的 "加载模型 → 量化 → 评测 → 保存结果" 流水线。
是 run_one.py 的核心引擎。

流程:
1. 加载并合并配置
2. 加载模型和 tokenizer
3. 准备校准数据（如果方法需要）
4. 调用量化方法
5. 运行评测
6. 保存结果
"""

import sys
import time
from pathlib import Path
from typing import Any

from src import config as cfg
from src import registry
from src.evaluator import evaluate
from src.result_writer import write_results
from src.env_info import collect_env_info


def run_experiment(
    model_name: str,
    method_name: str,
    track: str,
    cli_overrides: dict | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    cli_args_str: str = "",
    script_name: str = "",
) -> dict | None:
    """
    运行一个完整的实验：加载 → 量化 → 评测 → 保存结果。

    参数:
        model_name: 模型名称（对应 configs/models/{name}.yaml）
        method_name: 方法名称（对应 configs/methods/{name}.yaml）
        track: 赛道名称（A / B / C）
        cli_overrides: CLI 参数覆盖字典
        output_dir: 结果输出目录
        dry_run: 只打印配置不运行
        cli_args_str: 完整命令行字符串（用于记录）
        script_name: 脚本名称（用于记录）

    返回:
        dict: 实验结果字典，dry_run 模式返回 None
    """
    project_root = cfg.get_project_root()

    # ================================================================
    # 1. 加载并合并配置
    # ================================================================
    print("=" * 60)
    print(f"📋 加载配置: model={model_name}, method={method_name}, track={track}")
    print("=" * 60)

    global_config = cfg.load_global_config(project_root)
    model_config = cfg.load_model_config(model_name, project_root)
    method_config = cfg.load_method_config(method_name, project_root)
    track_config = cfg.load_track_config(track, project_root)

    # 注入 track 名称到覆盖中
    overrides = {"track": track.upper()}
    if cli_overrides:
        overrides.update(cli_overrides)

    merged = cfg.merge_configs(global_config, model_config, method_config, track_config, overrides)
    merged = cfg.resolve_paths(merged, project_root)

    # 验证配置
    errors = cfg.validate_config(merged)
    if errors:
        print(f"❌ 配置验证失败:")
        for e in errors:
            print(f"   - {e}")
        return None

    # ================================================================
    # dry_run 模式：只打印配置
    # ================================================================
    if dry_run:
        print("\n🔍 [DRY RUN] 合并后的最终配置:")
        print("-" * 40)
        print(cfg.dump_config(merged))
        print("-" * 40)
        print("🔍 [DRY RUN] 配置检查完毕，不执行实际运行")
        return None

    # ================================================================
    # 2. 检查方法是否支持该赛道
    # ================================================================
    method_cls = registry.get(method_name)
    supported = getattr(method_cls, "supported_tracks", [])
    if track.upper() not in [t.upper() for t in supported]:
        print(f"❌ 方法 {method_name} 不支持 Track {track}")
        print(f"   支持的赛道: {supported}")
        return None

    # ================================================================
    # 3. 加载模型和 tokenizer
    # ================================================================
    print(f"\n📦 加载模型: {model_config.get('model_id', 'unknown')}")
    model, tokenizer = _load_model(model_config, merged)

    # ================================================================
    # 4. 准备校准数据
    # ================================================================
    method_instance = method_cls(merged)
    calib_data = None
    if method_instance.requires_calibration():
        print(f"\n📦 准备校准数据")
        calib_data = _prepare_calibration_data(tokenizer, merged)

    # ================================================================
    # 5. 执行量化
    # ================================================================
    print(f"\n⚡ 执行量化: {method_name}")
    start_quant = time.time()
    warnings = []

    try:
        model = method_instance.quantize(model, tokenizer, calib_data)
    except Exception as e:
        print(f"❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    quant_time = time.time() - start_quant
    print(f"⏱️  量化耗时: {quant_time:.1f} 秒")

    # ================================================================
    # 6. 运行评测
    # ================================================================
    print(f"\n📊 开始评测")
    results = evaluate(model, tokenizer, merged)
    results["quant_time_seconds"] = round(quant_time, 1)

    # ================================================================
    # 7. 保存结果
    # ================================================================
    if output_dir is None:
        output_dir = merged.get("paths", {}).get("results_root", "results")
    output_dir = Path(output_dir)

    md_path, json_path = write_results(
        results=results,
        config=merged,
        output_dir=output_dir,
        cli_args=cli_args_str,
        script_name=script_name,
        warnings=warnings,
    )

    return results


def _load_model(model_config: dict, merged_config: dict):
    """
    加载 HuggingFace 模型和 tokenizer。

    参数:
        model_config: 模型配置字典
        merged_config: 合并后的完整配置

    返回:
        tuple: (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = model_config["model_id"]
        dtype_str = model_config.get("dtype", merged_config.get("common_hyperparams", {}).get("dtype", "float16"))
        dtype = getattr(torch, dtype_str, torch.float16)

        trust_remote_code = model_config.get("trust_remote_code", False)
        revision = model_config.get("revision", None)
        attn_impl = model_config.get("attn_implementation", None)
        cache_dir = merged_config.get("paths", {}).get("model_cache_dir", None)

        # 加载 tokenizer
        tokenizer_id = model_config.get("tokenizer_id") or model_id
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=cache_dir,
        )

        # 加载模型
        model_kwargs = model_config.get("model_kwargs", {})
        load_kwargs = {
            "dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "cache_dir": cache_dir,
            **model_kwargs,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        print(f"  ✅ 模型已加载: {model_id} ({dtype_str})")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        return model, tokenizer

    except ImportError as e:
        print(f"❌ 模型加载依赖缺失: {e}")
        print("   请安装: pip install transformers torch")
        raise
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def _prepare_calibration_data(tokenizer: Any, config: dict) -> list:
    """
    准备校准数据。

    从配置中读取校准参数，下载/加载校准数据集，
    tokenize 并 pack 成固定长度的 token blocks。

    参数:
        tokenizer: tokenizer
        config: 合并后的配置

    返回:
        list: 校准数据列表（每个元素是一个 dict，包含 input_ids）
    """
    calib_config = config.get("calibration", config.get("default_calibration", {}))
    dataset_name = calib_config.get("dataset", "wikitext2")
    num_samples = calib_config.get("num_samples", 128)
    seq_len = calib_config.get("seq_len", 2048)
    seed = calib_config.get("seed", 42)

    print(f"  校准数据集: {dataset_name}, 样本数: {num_samples}, 序列长度: {seq_len}, seed: {seed}")

    try:
        import torch
        from datasets import load_dataset
        import random

        random.seed(seed)

        # 加载数据集
        if dataset_name == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            all_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
        elif dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples * 2:  # 多取一些以确保够用
                    break
                texts.append(item["text"])
            all_text = "\n\n".join(texts)
        else:
            raise ValueError(f"不支持的校准数据集: {dataset_name}")

        # Tokenize
        encodings = tokenizer(all_text, return_tensors="pt")
        all_ids = encodings.input_ids[0]
        total_tokens = len(all_ids)

        print(f"  总 token 数: {total_tokens}")

        # 随机采样固定长度的 token blocks
        calib_data = []
        max_start = total_tokens - seq_len
        if max_start <= 0:
            print(f"⚠️ 文本总 token 数 ({total_tokens}) 小于 seq_len ({seq_len})")
            calib_data.append({"input_ids": all_ids[:seq_len].unsqueeze(0)})
        else:
            starts = random.sample(range(max_start), min(num_samples, max_start))
            for s in starts:
                chunk = all_ids[s:s + seq_len].unsqueeze(0)
                calib_data.append({"input_ids": chunk})

        print(f"  已生成 {len(calib_data)} 个校准样本")
        return calib_data

    except ImportError as e:
        print(f"⚠️  校准数据准备依赖缺失: {e}")
        return []
    except Exception as e:
        print(f"⚠️  校准数据准备失败: {e}")
        return []
