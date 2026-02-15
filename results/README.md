# results/ — 实验结果

## 用途
存放所有实验运行的结果文件。每次 `run_one.py` 生成一对 `.md` + `.json` 文件。

## 命名规则
```
YYYYMMDD_HHMMSS__{model}__{method}__{track}.md
YYYYMMDD_HHMMSS__{model}__{method}__{track}.json
```

示例：
```
20260214_153022__llama3.1-8b__gptq__trackA.md
20260214_153022__llama3.1-8b__gptq__trackA.json
```

## 文件格式

### `.md` 文件 — 人类阅读
包含以下固定区块：
- 运行时间
- CLI 参数（可复制）
- 数据集信息
- 方法与量化参数
- 评测结果表格（PPL、lm-eval tasks、system metrics）
- 环境信息（GPU、CUDA、git hash 等）

### `.json` 文件 — 机器解析
与 `.md` 包含相同数据，用于 `leaderboard.py` 自动汇总。

### `leaderboard.md` — 排行榜
由 `leaderboard.py` 自动生成，按 Track 分组汇总所有实验结果。

## 注意事项
- 不要手动修改 `.json` 文件
- `run_all.py --resume` 会跳过已有结果文件的实验
- 建议定期将重要结果备份或提交到 git
