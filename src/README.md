# src/ — 核心代码

## 用途
项目的 Python 源码包，包含配置加载、量化方法注册、运行控制、评测引擎等核心模块。

## 目录结构

```
src/
├── __init__.py          # 包初始化
├── config.py            # 配置加载与合并（YAML → dict, 支持 CLI 覆盖）
├── registry.py          # 量化方法注册表（自动发现 methods/ 下的所有方法）
├── runner.py            # 运行控制器（加载模型 → 量化 → 评测 → 保存结果）
├── evaluator.py         # 评测引擎（PPL 计算、长上下文 PPL、lm-eval 调用）
├── env_info.py          # 环境信息收集（GPU、CUDA、git hash、包版本等）
├── result_writer.py     # 结果写入器（生成 .md + .json 结果文件）
└── methods/             # 量化方法 wrapper
    ├── __init__.py
    ├── base.py           # 量化方法基类 BaseQuantMethod
    ├── fp16.py           # FP16 baseline（不量化）
    ├── rtn.py            # RTN (Round-To-Nearest)
    ├── gptq.py           # GPTQ wrapper
    ├── awq.py            # AWQ wrapper
    ├── smoothquant.py    # SmoothQuant wrapper
    ├── forge.py          # FORGE (动态秩 SVD KV 压缩)
    ├── kivi.py           # KIVI (INT2 per-ch/per-tok KV 量化)
    └── kvquant.py        # KVQuant (INT2 + outlier 隔离 KV 量化)
```

## 评测引擎 (`evaluator.py`)

支持三种评测：
1. **PPL 评测**: WikiText-2 (标准) + PG19 (长上下文, max_seq_len=4096)
2. **lm-eval 任务**: MMLU, HellaSwag, Winogrande 等（基于 lm-evaluation-harness v0.4.11）
3. **系统指标**: VRAM 峰值（自动记录）

## KV Cache 方法实现

Track C 的 FORGE / KIVI / KVQuant 采用 **monkey-patch** 方式注入模型的 Attention 层，
通过 `forward()` hook 在推理时动态压缩 KV Cache。无需修改模型权重或重新训练。

## 关键设计
- **配置优先级**: CLI 参数 > experiment YAML > method YAML > track YAML > config.yaml
- **自动发现**: `methods/` 下的模块被 `registry.py` 自动扫描并注册
- **可复现**: 每次运行自动记录完整配置、环境、git hash
