# configs/models/ — 模型配置

## 用途
每个 YAML 文件定义一个 LLM 模型的加载参数。新增模型 **只需加一个 YAML** 即可。

## 必填字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | 模型简称（用于结果命名） | `llama3.1-8b` |
| `model_id` | HuggingFace 模型 ID | `meta-llama/Llama-3.1-8B` |
| `dtype` | 默认加载精度 | `float16` / `bfloat16` |
| `max_seq_len` | 模型支持的最大序列长度 | `131072` |
| `trust_remote_code` | 是否需要信任远程代码 | `false` |

## 可选字段

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `revision` | HuggingFace 模型 commit hash | `null`（使用最新） |
| `tokenizer_id` | 自定义 tokenizer（如果与模型不同） | `null`（与 model_id 相同） |
| `adapter` | 特殊模型适配器名称（极少用） | `null` |
| `model_kwargs` | 传给 `from_pretrained()` 的额外参数 | `{}` |
| `attn_implementation` | attention 实现方式 | `null`（自动选择） |

## 新增模型教程

1. 复制 `llama3.1-8b.yaml`，重命名为 `你的模型名.yaml`
2. 修改 `model_id` 为 HuggingFace 上的模型路径
3. 调整 `dtype`（推荐：新模型用 `bfloat16`，旧模型用 `float16`）
4. 设置 `max_seq_len`（从模型文档查）
5. 如果模型需要 `trust_remote_code: true`，务必注明原因
6. 运行 `python scripts/run_one.py --model 你的模型名 --method fp16 --track A --dry_run` 验证配置

## 命名规范
- 文件名：`{模型系列}-{参数量}.yaml`，全小写，用连字符
- 示例：`llama3.1-8b.yaml`、`mistral-7b.yaml`、`qwen2.5-7b.yaml`

## 注意事项
- 部分模型需要预先通过 `huggingface-cli login` 获取访问权限（如 Llama 系列）
- 70B+ 模型在单卡 H200 (141GB) 上 FP16 可能 OOM，建议先跑量化版本
