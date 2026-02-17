# configs/tracks/ — 赛道配置

## 用途
定义三条 benchmark 赛道（Track），每条限定量化类型和评测组合。

## 赛道一览

| Track | 文件 | 量化类型 | 评测内容 |
|:-----:|------|---------|---------|
| A | `track_a.yaml` | W4A16 | WikiText-2 PPL + MMLU/HellaSwag/Winogrande/ARC/PIQA/GSM8K |
| B | `track_b.yaml` | W8A8 | WikiText-2 PPL + lm-eval tasks |
| C | `track_c.yaml` | KV Cache | WikiText-2 PPL + PG19 PPL (4K) + MMLU/HellaSwag/Winogrande |

## Track C 特性

Track C 的 `track_c.yaml` 包含 `long_context` 配置：
- `enabled: true` — 启用长上下文评测
- `max_seq_len: 4096` — PG19 长文本上的滑动窗口 PPL
- `tasks: ["longbench"]` — 使用 PG19-test 数据集

## 注意事项
- 每个方法在其 YAML 里声明自己支持哪些 Track
- `run_one.py --track A` 会自动只加载支持 Track A 的方法
- Track C 的评测比 Track A 多一个 PG19 长上下文 PPL
