#!/bin/bash
# ==============================================================================
# prepare_data.sh — 准备校准数据的启动脚本
#
# 用法: bash scripts/prepare_data.sh
# ==============================================================================

set -euo pipefail

# === 配置 ===
DATASET="wikitext2"                # 数据集: wikitext2 / c4
MODEL="llama3.1-8b"                # 使用哪个模型的 tokenizer
NUM_SAMPLES=128                    # 校准样本数
SEQ_LEN=2048                       # 序列长度
SEED=42                            # 随机种子

# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CMD="python scripts/prepare_data.py --dataset $DATASET --model $MODEL --num_samples $NUM_SAMPLES --seq_len $SEQ_LEN --seed $SEED"

echo "=========================================="
echo "📦 准备校准数据:"
echo "   $CMD"
echo "=========================================="

eval $CMD
