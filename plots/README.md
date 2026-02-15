# plots/ — 可视化图表

## 用途
存放由脚本自动生成的可视化图表。**只存代码生成的图，不存手工图。**

## 命名规则
```
YYYYMMDD_HHMM__{model}__{method}__{track}__{metric}.png
```

示例：
```
20260214_1530__llama3.1-8b__all_methods__trackA__ppl_comparison.png
20260214_1530__all_models__gptq__trackA__accuracy_bar.png
```

## 图表来源
- 由 `scripts/leaderboard.py --plot` 生成（Phase 2 实现）
- 格式：PNG（默认） / PDF（`--format pdf`）

## 注意事项
- 每张图的生成参数记录在同名 `.meta.json` 文件中（Phase 2）
- 同一组数据重新生成的图会覆盖（按相同命名规则）
- 图表可以嵌入到 `results/leaderboard.md` 和 `summary.md` 中
