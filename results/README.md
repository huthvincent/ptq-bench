# results/ — 实验结果

## 用途
存放所有实验结果。每次 `run_one.py` 生成一对 `.md` + `.json` 文件。

## 命名规则
```
YYYYMMDD_HHMMSS__{model}__{method}__{track}.md
YYYYMMDD_HHMMSS__{model}__{method}__{track}.json
```

示例：
```
20260216_164726__qwen2.5-7b__fp16__trackC.md
20260216_164726__qwen2.5-7b__fp16__trackC.json
```

## 文件格式

### `.md` — 人类阅读
包含：运行时间、CLI 参数、量化参数、PPL 表、lm-eval 任务表、VRAM 峰值、环境信息

### `.json` — 机器解析
与 `.md` 相同数据，用于 `leaderboard.py` 自动汇总

### `leaderboard.md` — 排行榜
由 `leaderboard.py` 或手动汇总生成，按 Track 分组，包含所有模型×方法的对比。

## 已有实验（截至 2026-02-16）

### Track A (Qwen2.5-7B)
- FP16 / RTN / AWQ

### Track C (Qwen2.5-7B + Mistral-7B)
- FP16 / FORGE / KIVI / KVQuant（各两个模型，共 8 组结果）
- 含 WikiText-2 PPL + PG19 长上下文 PPL (4K) + lm-eval 准确率

## 注意事项
- 不要手动修改 `.json` 文件
- `run_all.py --resume` 会跳过已有结果文件的实验
