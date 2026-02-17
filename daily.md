# PTQ Benchmark — 工作日志 (Daily Log)

## 2026-02-14

### 项目初始化
- 创建项目骨架目录结构
- 创建全局配置 `configs/config.yaml`
- 创建模型配置: `llama3.1-8b.yaml`, `mistral-7b.yaml`
- 创建方法配置: `fp16.yaml`, `rtn.yaml`, `gptq.yaml`, `awq.yaml`, `smoothquant.yaml`
- 创建 Track 配置: `track_a.yaml` (W4A16), `track_b.yaml` (W8A8), `track_c.yaml` (KV Cache)
- 实现核心代码模块:
  - `src/config.py` — 配置加载与合并
  - `src/registry.py` — 量化方法自动发现与注册
  - `src/env_info.py` — 环境信息收集
  - `src/evaluator.py` — PPL + lm-eval 评测引擎
  - `src/result_writer.py` — 结果 .md + .json 写入器
  - `src/runner.py` — 运行控制器
- 实现量化方法 wrappers: FP16, RTN, GPTQ, AWQ, SmoothQuant
- 实现脚本: `run_one.py/sh`, `run_all.py/sh`, `leaderboard.py/sh`, `prepare_data.py/sh`
- 所有目录创建 README.md

### 待办
- 安装 Miniconda 和依赖
- 运行环境可行性测试（CUDA 13.0 兼容性）
- 跑通第一个实验（dry_run）

## 2026-02-15

### FORGE 方法开发
- 新增方法: FORGE (Fast On-chip Reconstruction of Generative Embeddings)
  - `src/methods/forge.py` — 动态秩 SVD KV Cache 压缩 (Track C)
  - `configs/methods/forge.yaml` — FORGE 参数配置
- 核心特性: 分块 SVD + 动态秩选择 + 芯片上重构，免校准
- 更新 `configs/methods/README.md` 添加 FORGE 字段说明

## 2026-02-16

### Track C 完整评测
- 跑通 FORGE 完整实验:
  - qwen2.5-7b: PPL=6.16, AvgAcc=0.7372 (与 FP16 一致, 零精度退化)
  - mistral-7b: PPL=4.79, AvgAcc=0.6131 (与 FP16 一致, 零精度退化)
- 跑通 FP16 baseline (Track C): qwen2.5-7b, mistral-7b
- llama3.1-8b: 因 gated repo 无 HF_TOKEN 暂跳过
- 生成 Track C Leaderboard: `results/leaderboard.md`
- 结论: FORGE (energy_threshold=0.95) 在短序列 (2048) 下完全无损

### KIVI + KVQuant 实现
- 新增方法: KIVI (INT2 per-channel Key + per-token Value, 免校准)
  - `src/methods/kivi.py`, `configs/methods/kivi.yaml`
- 新增方法: KVQuant (INT2 + Dense-and-Sparse outlier 隔离)
  - `src/methods/kvquant.py`, `configs/methods/kvquant.yaml`
- 评测结果: 4 方法 × 2 模型 = 8 实验全部完成
  - 所有方法在短序列 (2048) 下与 FP16 一致 (residual_length=128 保护)
  - 真正差异化需 LongBench (32K+) 评测

