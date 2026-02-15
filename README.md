# PTQ LLM Quantization Benchmark

一个专注于 **Post-Training Quantization (PTQ)** 的 LLM 量化基准测试框架。

## 目标

- **只做 PTQ**（不含 finetune/QLoRA/SFT/RLHF）
- **三条赛道**: Track A (W4A16)、Track B (W8A8)、Track C (KV Cache Quant)
- **易扩展**: 新增模型只需加 YAML，新增方法只需一个 wrapper + YAML
- **强可复现**: 每次实验自动记录完整参数、环境、数据版本

## 快速开始

### 0. 环境准备

```bash
# 创建 conda 环境（首次）
conda create -n ptq-bench python=3.12 -y
conda activate ptq-bench

# 安装依赖
pip install torch transformers datasets accelerate safetensors
pip install auto-gptq autoawq     # Track A 方法
pip install lm-eval                # 评测框架
```

### 1. 准备校准数据

```bash
bash scripts/prepare_data.sh
```

### 2. 跑 Track A: Llama-3.1-8B + GPTQ

```bash
python scripts/run_one.py --model llama3.1-8b --method gptq --track A

# 或者只看配置不运行:
python scripts/run_one.py --model llama3.1-8b --method gptq --track A --dry_run
```

### 3. 跑 Track C: Llama-3.1-8B + FP16 baseline (KV 不量化)

```bash
python scripts/run_one.py --model llama3.1-8b --method fp16 --track C
```

### 4. 批量跑: 两个模型 × 两种方法 × Track A

```bash
python scripts/run_all.py \
    --include_models llama3.1-8b mistral-7b \
    --include_methods fp16 gptq \
    --include_tracks A

# 或者使用实验配置:
python scripts/run_all.py --experiment configs/experiments/quick_test.yaml
```

### 5. 生成排行榜

```bash
bash scripts/leaderboard.sh
# 结果在 results/leaderboard.md
```

## 目录结构

```
ptq-bench/
├── README.md              # 本文件
├── daily.md               # 每日工作日志
├── summary.md             # 项目概览
├── configs/               # 配置体系
│   ├── config.yaml        # 全局配置
│   ├── models/            # 模型 YAML（每个 LLM 一个）
│   ├── methods/           # 量化方法 YAML
│   ├── tracks/            # 赛道 YAML (A/B/C)
│   └── experiments/       # 实验组合 YAML
├── scripts/               # 可执行脚本
│   ├── run_one.py/sh      # 单个实验
│   ├── run_all.py/sh      # 批量实验
│   ├── leaderboard.py/sh  # 排行榜生成
│   └── prepare_data.py/sh # 数据准备
├── src/                   # 核心代码
│   ├── config.py          # 配置加载
│   ├── registry.py        # 方法注册表
│   ├── runner.py          # 运行控制器
│   ├── evaluator.py       # 评测引擎
│   ├── result_writer.py   # 结果写入器
│   ├── env_info.py        # 环境信息收集
│   └── methods/           # 量化方法 wrapper
├── results/               # 实验结果
├── plots/                 # 可视化图表
├── data/                  # 数据集
└── docs/                  # 扩展文档
```

## Track 说明

| Track | 量化类型 | 代表方法 | 描述 |
|-------|---------|---------|------|
| **A** | W4A16 | RTN, GPTQ, AWQ | 只量化权重到 4-bit |
| **B** | W8A8 | SmoothQuant | 权重+激活都量化到 8-bit |
| **C** | KV Cache | KIVI, KVQuant | 量化 KV Cache 节省长上下文显存 |

## 支持的方法

| 方法 | Track | 库 | 状态 |
|------|-------|-----|------|
| FP16 (Baseline) | A, B, C | transformers | ✅ |
| RTN | A | transformers | ✅ |
| GPTQ | A | auto-gptq | ✅ |
| AWQ | A | autoawq | ✅ |
| SmoothQuant | B | smoothquant | ✅ |
| OmniQuant | A | — | 🔜 Phase 2 |
| SpQR | A | — | 🔜 Phase 2 |
| KIVI | C | — | 🔜 Phase 2 |
| KVQuant | C | — | 🔜 Phase 2 |

## 新增模型

只需在 `configs/models/` 加一个 YAML 文件：

```yaml
name: "qwen2.5-7b"
model_id: "Qwen/Qwen2.5-7B"
dtype: "bfloat16"
max_seq_len: 131072
trust_remote_code: false
```

详见 `configs/models/README.md`。

## 新增量化方法

1. 在 `configs/methods/` 加一个 YAML（默认参数）
2. 在 `src/methods/` 加一个 wrapper（继承 `BaseQuantMethod`）

详见 `configs/methods/README.md`。

## 结果格式

每次实验生成一对文件：
- `results/YYYYMMDD_HHMMSS__{model}__{method}__{track}.md` — 人类阅读
- `results/YYYYMMDD_HHMMSS__{model}__{method}__{track}.json` — 机器解析

包含：完整 CLI 参数、数据集版本、量化参数、PPL 表、lm-eval 任务表、VRAM 峰值、环境信息。
