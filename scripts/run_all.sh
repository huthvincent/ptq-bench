#!/bin/bash
# ==============================================================================
# run_all.sh — 批量运行多个实验的启动脚本
#
# 用法: bash scripts/run_all.sh
# 修改下面的变量即可配置批量运行参数
# ==============================================================================

set -euo pipefail

# === 运行模式（二选一） ===

# 模式 1: 使用实验配置文件
EXPERIMENT="configs/experiments/quick_test.yaml"

# 模式 2: 手动指定（取消注释并清空 EXPERIMENT）
# EXPERIMENT=""
# INCLUDE_MODELS="llama3.1-8b mistral-7b"
# INCLUDE_METHODS="fp16 gptq awq"
# INCLUDE_TRACKS="A B"
# EXCLUDE_MODELS=""
# EXCLUDE_METHODS=""

# === 运行控制 ===
RESUME=false                       # 跳过已有结果的实验
DRY_RUN=false                      # 只打印计划不运行
OUTPUT_DIR=""                      # 留空使用默认 results/

# ==============================================================================
# 以下代码通常不需要修改
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CMD="python scripts/run_all.py"

if [[ -n "${EXPERIMENT}" ]]; then
    CMD="$CMD --experiment $EXPERIMENT"
else
    [[ -n "${INCLUDE_MODELS:-}" ]] && CMD="$CMD --include_models $INCLUDE_MODELS"
    [[ -n "${INCLUDE_METHODS:-}" ]] && CMD="$CMD --include_methods $INCLUDE_METHODS"
    [[ -n "${INCLUDE_TRACKS:-}" ]] && CMD="$CMD --include_tracks $INCLUDE_TRACKS"
    [[ -n "${EXCLUDE_MODELS:-}" ]] && CMD="$CMD --exclude_models $EXCLUDE_MODELS"
    [[ -n "${EXCLUDE_METHODS:-}" ]] && CMD="$CMD --exclude_methods $EXCLUDE_METHODS"
fi

[[ -n "${OUTPUT_DIR}" ]] && CMD="$CMD --output_dir $OUTPUT_DIR"
[[ "$RESUME" == "true" ]] && CMD="$CMD --resume"
[[ "$DRY_RUN" == "true" ]] && CMD="$CMD --dry_run"

echo "=========================================="
echo "🚀 运行命令:"
echo "   $CMD"
echo "=========================================="

eval $CMD
