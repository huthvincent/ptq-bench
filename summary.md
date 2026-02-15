# PTQ Benchmark — 项目概览 (Summary)

## 目标 (Goal)

构建一个标准的 LLM Post-Training Quantization (PTQ) Benchmark 框架，覆盖三条赛道：

- **Track A**: Weight-only W4A16（只量化权重到 4-bit）
- **Track B**: W8A8（权重和激活都量化到 8-bit）
- **Track C**: KV Cache Quantization（量化 KV 缓存以支持长上下文）

**不包含** 任何 finetune、QLoRA、SFT、RLHF 方法。

## 关键数据集 (Key Datasets)

| 数据集 | 用途 | 来源 | 配置 |
|--------|------|------|------|
| WikiText-2 | 校准 + PPL 评测 | HuggingFace `wikitext` | `wikitext-2-raw-v1` |
| C4 | 校准（可选） | HuggingFace `allenai/c4` | `en` |
| lm-eval tasks | 任务评测 | lm-evaluation-harness | MMLU, GSM8K, HellaSwag 等 |
| LongBench | 长上下文评测 (Phase 2) | HuggingFace `THUDM/LongBench` | — |

## 方法 (Methods)

### Track A (W4A16)
- ✅ FP16 (baseline)
- ✅ RTN (baseline)
- ✅ GPTQ
- ✅ AWQ
- 🔜 OmniQuant (Phase 2)
- 🔜 SpQR (Phase 2)

### Track B (W8A8)
- ✅ FP16 (baseline)
- ✅ SmoothQuant

### Track C (KV Cache)
- ✅ FP16 (baseline, 不量化 KV)
- 🔜 KIVI (Phase 2)
- 🔜 KVQuant (Phase 2)

## 当前最好结果 (Current Best Results)

*尚未运行实验。请先执行 `bash scripts/run_one.sh` 并 `bash scripts/leaderboard.sh` 生成排行榜。*

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
| 工作日志 | `daily.md` |
