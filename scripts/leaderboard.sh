#!/bin/bash
# ==============================================================================
# leaderboard.sh — 生成排行榜的启动脚本
#
# 用法: bash scripts/leaderboard.sh
# ==============================================================================

set -euo pipefail

# === 配置 ===
RESULTS_DIR="results"              # 结果文件目录
OUTPUT="results/leaderboard.md"    # 排行榜输出路径
TOP_K=5                            # 每个 Track 每个模型展示前 k 名

# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CMD="python scripts/leaderboard.py --results_dir $RESULTS_DIR --output $OUTPUT --top_k $TOP_K --update_summary"

echo "=========================================="
echo "📊 生成排行榜:"
echo "   $CMD"
echo "=========================================="

eval $CMD
