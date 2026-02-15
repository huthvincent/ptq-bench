#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py — 准备校准数据

下载并预处理校准数据集（WikiText-2 / C4），
将 tokenized + packed 的 token blocks 保存到 data/processed/，
确保不同量化方法使用完全相同的校准数据。

用法:
    python scripts/prepare_data.py --dataset wikitext2 --model llama3.1-8b
    python scripts/prepare_data.py --dataset c4 --num_samples 512 --seq_len 2048

注意: Phase 1 中大多数方法内部自己处理校准数据，
      此脚本在需要严格控制校准数据一致性时使用。
"""

import sys
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="PTQ Benchmark: 准备校准数据")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4"],
                        help="校准数据集")
    parser.add_argument("--model", type=str, default="llama3.1-8b",
                        help="用哪个模型的 tokenizer")
    parser.add_argument("--num_samples", type=int, default=128,
                        help="校准样本数")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="每个样本的序列长度")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_dir", type=str, default="/data2/zhu11/quant_source/data/processed/calibration",
                        help="输出目录（默认: 共享数据目录）")
    return parser.parse_args()


def main():
    """主入口函数。"""
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 准备校准数据")
    print(f"   数据集: {args.dataset}")
    print(f"   模型 tokenizer: {args.model}")
    print(f"   样本数: {args.num_samples}")
    print(f"   序列长度: {args.seq_len}")
    print(f"   随机种子: {args.seed}")
    print(f"   输出目录: {output_dir}")

    try:
        from src.config import load_model_config
        model_config = load_model_config(args.model, PROJECT_ROOT)
        model_id = model_config["model_id"]

        from transformers import AutoTokenizer
        print(f"\n📦 加载 tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        from src.runner import _prepare_calibration_data
        config = {
            "calibration": {
                "dataset": args.dataset,
                "num_samples": args.num_samples,
                "seq_len": args.seq_len,
                "seed": args.seed,
            }
        }

        calib_data = _prepare_calibration_data(tokenizer, config)

        # 保存
        import torch
        output_name = f"{args.dataset}_{args.num_samples}x{args.seq_len}_seed{args.seed}"
        output_file = output_dir / f"{output_name}.pt"
        torch.save(calib_data, output_file)
        print(f"\n✅ 校准数据已保存: {output_file}")

        # 保存元数据
        meta = {
            "dataset": args.dataset,
            "model_tokenizer": args.model,
            "model_id": model_id,
            "num_samples": len(calib_data),
            "seq_len": args.seq_len,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
            "file": str(output_file),
        }
        meta_dir = Path("/data2/zhu11/quant_source/data/meta")
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / f"{output_name}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"✅ 元数据已保存: {meta_file}")

    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("   请安装: pip install transformers torch datasets")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
