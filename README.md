# PTQ LLM Quantization Benchmark

A standardized benchmarking framework for **Post-Training Quantization (PTQ)** of Large Language Models.

## Highlights

- **PTQ only** — No finetuning, QLoRA, SFT, or RLHF
- **3 Tracks**: Weight-only (W4A16), Weight+Activation (W8A8), KV Cache Compression
- **8 Methods**: FP16, RTN, GPTQ, AWQ, SmoothQuant, FORGE, KIVI, KVQuant
- **3 Models**: Qwen2.5-7B, Mistral-7B-v0.3, Llama-3.1-8B
- **Extensible**: Add a model = 1 YAML, Add a method = 1 YAML + 1 Python wrapper
- **Reproducible**: Every run auto-records full config, environment, git hash, GPU info

## Quick Start

### 0. Environment Setup

```bash
conda create -n ptq-bench python=3.12 -y
conda activate ptq-bench

pip install torch transformers datasets accelerate safetensors
pip install lm-eval   # evaluation harness
```

### 1. Run a Single Experiment

```bash
# Track A: Qwen2.5-7B + RTN (weight-only W4A16)
python scripts/run_one.py --model qwen2.5-7b --method rtn --track A

# Track C: Mistral-7B + KIVI (KV cache quantization)
python scripts/run_one.py --model mistral-7b --method kivi --track C

# Dry run (print config only):
python scripts/run_one.py --model qwen2.5-7b --method gptq --track A --dry_run
```

### 2. Run Batch Experiments

```bash
python scripts/run_all.py \
    --include_models qwen2.5-7b mistral-7b \
    --include_methods fp16 forge kivi kvquant \
    --include_tracks C
```

### 3. Generate Leaderboard

```bash
python scripts/leaderboard.py --results_dir results/
# Output: results/leaderboard.md
```

## Tracks

| Track | Quantization Type | Methods | Description |
|:-----:|-------------------|---------|-------------|
| **A** | W4A16 (Weight-only) | RTN, GPTQ, AWQ | Quantize weights to 4-bit, activations stay FP16 |
| **B** | W8A8 | SmoothQuant | Quantize both weights and activations to 8-bit |
| **C** | KV Cache | FORGE, KIVI, KVQuant | Compress KV cache for long-context inference |

## Supported Methods

| Method | Track | Library | Status |
|--------|:-----:|---------|:------:|
| FP16 (Baseline) | A, B, C | transformers | ✅ |
| RTN (Round-To-Nearest) | A | transformers | ✅ |
| GPTQ | A | auto-gptq | ⚠️ Blocked (library compat) |
| AWQ | A | autoawq | ✅ |
| SmoothQuant | B | smoothquant | ✅ |
| FORGE (SVD KV compression) | C | transformers | ✅ |
| KIVI (INT2 per-ch/per-tok) | C | transformers | ✅ |
| KVQuant (INT2 + outlier) | C | transformers | ✅ |

## Latest Results

### Track A — Weight-Only W4A16 (Qwen2.5-7B)

| Method | PPL (WikiText-2) ↓ | Avg Accuracy ↑ | PPL Δ | Acc Δ |
|--------|:---:|:---:|:---:|:---:|
| FP16 (baseline) | 6.16 | 0.7351 | — | — |
| AWQ (W4) | 6.91 | 0.7233 | +0.75 | -1.18% |
| RTN (W4) | 7.27 | 0.7098 | +1.11 | -2.53% |

### Track C — KV Cache Compression (residual=32, chunk=16)

| Model | Method | WikiText-2 PPL | PG19 PPL (4K) | Avg Accuracy |
|-------|--------|:-:|:-:|:-:|
| **Qwen2.5-7B** | FP16 | 6.16 | 11.401 | 0.7372 |
| | FORGE / KIVI / KVQuant | 6.16 | 11.401 | 0.7372 |
| **Mistral-7B** | FP16 | 4.79 | 8.264 | 0.6131 |
| | FORGE / KIVI / KVQuant | 4.79 | 8.26 | 0.6131 |

> **Key Finding**: All KV cache compression methods show **zero degradation** even with aggressive settings (residual_length=32, chunk_size=16) and up to 4096-token context.

Full results: [results/leaderboard.md](results/leaderboard.md)

## Project Structure

```
ptq-bench/
├── README.md              # This file
├── summary.md             # Project overview (Chinese)
├── configs/               # Configuration system
│   ├── config.yaml        # Global config (paths, seed, defaults)
│   ├── models/            # Model configs (1 YAML per LLM)
│   ├── methods/           # Method configs (1 YAML per method)
│   ├── tracks/            # Track configs (A / B / C)
│   └── experiments/       # Experiment combos for batch runs
├── scripts/               # Executable scripts
│   ├── run_one.py         # Single experiment
│   ├── run_all.py         # Batch experiments
│   ├── leaderboard.py     # Generate leaderboard
│   └── prepare_data.py    # Data preparation
├── src/                   # Core source code
│   ├── config.py          # Config loading & merging
│   ├── registry.py        # Method auto-discovery
│   ├── runner.py          # Experiment orchestrator
│   ├── evaluator.py       # PPL + lm-eval engine
│   ├── result_writer.py   # .md + .json output
│   ├── env_info.py        # Environment capture
│   └── methods/           # Quantization wrappers
│       ├── fp16.py        # FP16 baseline
│       ├── rtn.py         # RTN
│       ├── gptq.py        # GPTQ
│       ├── awq.py         # AWQ
│       ├── smoothquant.py # SmoothQuant
│       ├── forge.py       # FORGE (SVD KV compression)
│       ├── kivi.py        # KIVI (INT2 KV quantization)
│       └── kvquant.py     # KVQuant (INT2 + outlier isolation)
├── results/               # Experiment results (.md + .json)
├── plots/                 # Auto-generated visualizations
├── data/                  # Dataset pointers (data stored externally)
└── docs/                  # Extended documentation
```

## Adding a New Model

Just add a YAML in `configs/models/`:

```yaml
name: "your-model"
model_id: "org/Model-Name"
dtype: "bfloat16"
max_seq_len: 131072
trust_remote_code: false
```

See [configs/models/README.md](configs/models/README.md) for details.

## Adding a New Method

1. Add a YAML in `configs/methods/` (default hyperparams)
2. Add a wrapper in `src/methods/` (inherit `BaseQuantMethod`, implement `quantize()`)

See [configs/methods/README.md](configs/methods/README.md) for details.

## Result Format

Each experiment generates a pair of files:
- `results/YYYYMMDD_HHMMSS__{model}__{method}__{track}.md` — Human-readable
- `results/YYYYMMDD_HHMMSS__{model}__{method}__{track}.json` — Machine-parseable

Both contain: full CLI args, dataset info, quantization params, PPL/accuracy tables, VRAM peak, environment info.

## Environment

| Item | Version |
|------|---------|
| GPU | NVIDIA H200 NVL 141GB |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.1.0 |
| lm-eval | 0.4.11 |
| Python | 3.13 |
