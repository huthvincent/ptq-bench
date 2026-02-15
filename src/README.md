# src/ — 核心代码

## 用途
项目的 Python 源码包，包含配置加载、量化方法注册、运行控制、评测引擎等核心模块。

## 目录结构

```
src/
├── __init__.py          # 包初始化
├── config.py            # 配置加载与合并（YAML → Python dataclass）
├── registry.py          # 量化方法注册表（自动发现 methods/ 下的所有方法）
├── runner.py            # 运行控制器（组装 model + method + eval 的主流程）
├── evaluator.py         # 评测引擎（PPL 计算、lm-eval-harness 调用）
├── env_info.py          # 环境信息收集（GPU、CUDA、git hash、包版本等）
├── result_writer.py     # 结果写入器（生成 .md + .json 结果文件）
└── methods/             # 量化方法 wrapper
    ├── __init__.py
    ├── base.py           # 量化方法基类 BaseQuantMethod
    ├── fp16.py           # FP16 baseline（不量化）
    ├── rtn.py            # RTN (Round-To-Nearest)
    ├── gptq.py           # GPTQ wrapper
    ├── awq.py            # AWQ wrapper
    └── smoothquant.py    # SmoothQuant wrapper
```

## 新增方法快速指南

1. 在 `methods/` 创建 `你的方法.py`
2. 继承 `BaseQuantMethod`，实现 `quantize()` 方法
3. 在 `configs/methods/` 添加对应 YAML
4. 方法会被 `registry.py` 自动发现并注册

详细教程见 `configs/methods/README.md`。

## 关键设计
- **配置优先级**: CLI 参数 > experiment YAML > method YAML > 全局 config.yaml
- **自动发现**: `methods/` 下的模块会被 `registry.py` 自动扫描并注册
- **可复现**: 每次运行自动记录完整配置和环境信息
