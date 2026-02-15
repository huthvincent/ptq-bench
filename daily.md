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
