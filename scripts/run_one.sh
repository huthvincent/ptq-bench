#!/bin/bash
# ==============================================================================
# run_one.sh — 运行单个实验的启动脚本
#
# 用法: bash scripts/run_one.sh
# 修改下面的变量即可配置实验参数
# ==============================================================================

set -euo pipefail

# === 实验配置（修改这些变量） ===
MODEL="llama3.1-8b"          # 模型名 (对应 configs/models/*.yaml)
METHOD="gptq"                # 方法名 (对应 configs/methods/*.yaml)
TRACK="A"                    # 赛道: A (W4A16) / B (W8A8) / C (KV Cache)

# === 可选覆盖参数（取消注释以启用） ===
# W_BITS=4                   # 权重量化位数
# GROUP_SIZE=128             # 量化分组大小
# SCHEME="symmetric"         # 量化方案: symmetric / asymmetric
# CALIB_DATASET="wikitext2"  # 校准数据集: wikitext2 / c4
# NUM_SAMPLES=128            # 校准样本数
# SEQ_LEN=2048               # 校准序列长度
# SEED=42                    # 随机种子

# === 输出控制 ===
OUTPUT_DIR=""                # 留空则使用默认 results/ 目录
DRY_RUN=false                # 设为 true 则只打印配置不运行
PRINT_CONFIG=false           # 设为 true 则运行前打印最终配置

# ==============================================================================
# 以下代码通常不需要修改
# ==============================================================================

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 构建命令
CMD="python scripts/run_one.py --model $MODEL --method $METHOD --track $TRACK"

# 添加可选参数
[[ -n "${W_BITS:-}" ]] && CMD="$CMD --w_bits $W_BITS"
[[ -n "${GROUP_SIZE:-}" ]] && CMD="$CMD --group_size $GROUP_SIZE"
[[ -n "${SCHEME:-}" ]] && CMD="$CMD --scheme $SCHEME"
[[ -n "${CALIB_DATASET:-}" ]] && CMD="$CMD --calib_dataset $CALIB_DATASET"
[[ -n "${NUM_SAMPLES:-}" ]] && CMD="$CMD --num_samples $NUM_SAMPLES"
[[ -n "${SEQ_LEN:-}" ]] && CMD="$CMD --seq_len $SEQ_LEN"
[[ -n "${SEED:-}" ]] && CMD="$CMD --seed $SEED"
[[ -n "${OUTPUT_DIR}" ]] && CMD="$CMD --output_dir $OUTPUT_DIR"
[[ "$DRY_RUN" == "true" ]] && CMD="$CMD --dry_run"
[[ "$PRINT_CONFIG" == "true" ]] && CMD="$CMD --print_config"

echo "=========================================="
echo "🚀 运行命令:"
echo "   $CMD"
echo "=========================================="

eval $CMD
