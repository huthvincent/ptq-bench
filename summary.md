# PTQ Benchmark — 项目概览

## 目标

构建标准化的 LLM Post-Training Quantization (PTQ) 基准测试框架，覆盖三条赛道：

- **Track A**: Weight-only W4A16（只量化权重到 4-bit）
- **Track B**: W8A8（权重和激活都量化到 8-bit）
- **Track C**: KV Cache Compression（量化/压缩 KV 缓存以支持长上下文）

**不包含** 任何 finetune、QLoRA、SFT、RLHF 方法。

## 支持的模型

| 模型 | 参数量 | HuggingFace ID |
|------|:------:|----------------|
| Qwen2.5-7B | 7.62B | `Qwen/Qwen2.5-7B` |
| Mistral-7B-v0.3 | 7.25B | `mistralai/Mistral-7B-v0.3` |
| Llama-3.1-8B | 8.03B | `meta-llama/Llama-3.1-8B` |

## 方法与状态

### Track A (W4A16)

| 方法 | 状态 | 备注 |
|------|:----:|------|
| FP16 (baseline) | ✅ | — |
| RTN | ✅ | 基础量化方法 |
| GPTQ | ⚠️ | auto-gptq 库兼容性问题 |
| AWQ | ✅ | 使用预量化模型 |

### Track B (W8A8)

| 方法 | 状态 | 备注 |
|------|:----:|------|
| FP16 (baseline) | ✅ | — |
| SmoothQuant | ✅ | — |

### Track C (KV Cache)

| 方法 | 状态 | 备注 |
|------|:----:|------|
| FP16 (baseline) | ✅ | 不压缩 KV |
| FORGE | ✅ | 动态秩 SVD 压缩 |
| KIVI | ✅ | INT2 per-channel Key + per-token Value |
| KVQuant | ✅ | INT2 + outlier 隔离 |

## 评测数据集

| 数据集 | 用途 | 来源 |
|--------|------|------|
| WikiText-2 | 校准 + PPL 评测 | HuggingFace `wikitext` |
| PG19-test | 长上下文 PPL 评测 | HuggingFace `emozilla/pg19-test` |
| lm-eval tasks | 准确率评测 | MMLU, HellaSwag, Winogrande 等 |

## 当前最佳结果

### Track A (Qwen2.5-7B)

| 方法 | PPL ↓ | Avg Acc ↑ |
|------|:-----:|:---------:|
| FP16 | 6.16 | 0.7351 |
| AWQ | 6.91 | 0.7233 |
| RTN | 7.27 | 0.7098 |

### Track C (residual=32, chunk=16, max_seq_len=4096)

| 模型 | 方法 | WikiText-2 | PG19 (4K) | Avg Acc |
|------|------|:---:|:---:|:---:|
| Qwen2.5-7B | FP16 | 6.16 | 11.401 | 0.7372 |
| | FORGE/KIVI/KVQuant | 6.16 | 11.401 | 0.7372 |
| Mistral-7B | FP16 | 4.79 | 8.264 | 0.6131 |
| | FORGE/KIVI/KVQuant | 4.79 | 8.26 | 0.6131 |

> 所有 KV Cache 方法在极端压缩设置下仍然 **完全无损**。

详见 [results/leaderboard.md](results/leaderboard.md)

## 项目导航

| 内容 | 位置 |
|------|------|
| 配置体系 | `configs/` — 全局配置、模型/方法/赛道 YAML |
| 脚本 | `scripts/` — run_one, run_all, leaderboard |
| 结果 | `results/` — 每次实验的 .md + .json |
| 图表 | `plots/` — 自动生成的可视化图表 |
| 核心代码 | `src/` — 配置加载、方法注册、评测引擎 |
| 数据 | `data/` — 数据集缓存与元数据 |
