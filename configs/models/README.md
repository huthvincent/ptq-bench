# configs/models/ — 模型配置

## 用途
每个 YAML 文件定义一个 LLM 模型的加载参数。新增模型 **只需加一个 YAML** 即可。

## 当前模型

| 文件 | 模型 | 参数量 | HuggingFace ID |
|------|------|:------:|----------------|
| `qwen2.5-7b.yaml` | Qwen2.5-7B | 7.62B | `Qwen/Qwen2.5-7B` |
| `mistral-7b.yaml` | Mistral-7B-v0.3 | 7.25B | `mistralai/Mistral-7B-v0.3` |
| `llama3.1-8b.yaml` | Llama-3.1-8B | 8.03B | `meta-llama/Llama-3.1-8B` |

## 必填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | 模型简称（用于结果命名） | `qwen2.5-7b` |
| `model_id` | HuggingFace 模型 ID | `Qwen/Qwen2.5-7B` |
| `dtype` | 加载精度 | `bfloat16` |
| `max_seq_len` | 最大序列长度 | `131072` |
| `trust_remote_code` | 是否信任远程代码 | `false` |

## 可选字段

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `revision` | 模型 commit hash | `null` |
| `tokenizer_id` | 自定义 tokenizer | `null` |
| `adapter` | 特殊适配器 | `null` |
| `model_kwargs` | 传给 `from_pretrained()` 的额外参数 | `{}` |
| `attn_implementation` | attention 实现方式 | `null` |

## 新增模型教程

1. 复制 `qwen2.5-7b.yaml`，重命名
2. 修改 `model_id`
3. 设置 `dtype`（推荐新模型 `bfloat16`）
4. 设置 `max_seq_len`
5. `python scripts/run_one.py --model 你的模型 --method fp16 --track A --dry_run` 验证

## 命名规范
- 文件名：`{模型系列}-{参数量}.yaml`，全小写
- 示例：`qwen2.5-7b.yaml`、`mistral-7b.yaml`
