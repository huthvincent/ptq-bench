# configs/ — 配置体系

## 目录结构

```
configs/
├── config.yaml          # 全局基础配置（路径、种子、默认评测设置）
├── models/              # 模型配置（每个 LLM 一个 YAML）
│   ├── qwen2.5-7b.yaml
│   ├── mistral-7b.yaml
│   └── llama3.1-8b.yaml
├── methods/             # 量化方法配置（每个方法一个 YAML）
│   ├── fp16.yaml        # FP16 baseline
│   ├── rtn.yaml         # RTN (W4A16)
│   ├── gptq.yaml        # GPTQ (W4A16)
│   ├── awq.yaml         # AWQ (W4A16)
│   ├── smoothquant.yaml # SmoothQuant (W8A8)
│   ├── forge.yaml       # FORGE (KV SVD 压缩, chunk_size=16)
│   ├── kivi.yaml        # KIVI (KV INT2, residual_length=32)
│   └── kvquant.yaml     # KVQuant (KV INT2 + outlier, residual_length=32)
├── tracks/              # 赛道配置 (A/B/C)
│   ├── track_a.yaml     # W4A16
│   ├── track_b.yaml     # W8A8
│   └── track_c.yaml     # KV Cache (含 long_context 评测)
└── experiments/         # 实验组合配置（run_all.py 使用）
```

## 配置优先级

**CLI 参数 > experiment YAML > method YAML > track YAML > config.yaml**（从高到低）

## 使用方式

### 新增模型
在 `models/` 里加一个 YAML，设置 `model_id`、`dtype`、`max_seq_len`。详见 `models/README.md`。

### 新增量化方法
1. 在 `methods/` 加一个 YAML（默认参数）
2. 在 `src/methods/` 加一个 wrapper（继承 `BaseQuantMethod`）
3. 详见 `methods/README.md`

### 修改评测
Track C 的 `track_c.yaml` 支持启用 `long_context` 评测（PG19, max_seq_len=4096）。

## 注意事项
- 所有路径支持相对路径和绝对路径
- 修改配置后用 `--dry_run` 验证合并后的最终配置
