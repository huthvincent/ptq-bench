# plots/ — 可视化图表

## 用途
存放由脚本自动生成的可视化图表。

## 命名规则
```
YYYYMMDD_HHMM__{model}__{method}__{track}__{metric}.png
```

示例：
```
20260216_2100__all_models__all_methods__trackC__ppl_comparison.png
```

## 图表来源
- 由 `scripts/leaderboard.py --plot` 生成（待实现）
- 格式：PNG（默认）/ PDF（`--format pdf`）

## 注意事项
- 只存代码生成的图，不存手工图
- 图表可以嵌入到 `results/leaderboard.md` 和 `summary.md` 中
