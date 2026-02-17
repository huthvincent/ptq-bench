# Vibe Coding Guide — PTQ Benchmark

> **本文档面向所有参与开发的人或 AI**。
> 新增方法、模型、指标或修改任何代码前，**必须通读本文档**。

---

## 0. 核心理念

1. **PTQ Only** — 只做 Post-Training Quantization，不含 finetune/QLoRA/SFT/RLHF
2. **加方法 = 1个 .py + 1个 .yaml** — 不允许破坏这个极简约定
3. **加模型 = 1个 .yaml** — 绝大多数模型只需加一个 YAML 即可跑通
4. **强可复现** — 每次实验自动记录脚本名、完整参数、环境信息、结果表格

---

## 1. 目录结构 (不可更改)

```
ptq-bench/
├── vibe_coding.md          # 本文档 — 开发规范 (你正在读的)
├── README.md               # 项目介绍 + 最小运行示例
├── daily.md                # 工作日志 (按日期分块)
├── summary.md              # 项目概览 (含当前最好结果)
├── configs/                # 所有配置 YAML
│   ├── config.yaml         # 全局基础配置 (env/paths/hyperparams)
│   ├── models/             # 模型 YAML    (1 模型 = 1 文件)
│   ├── methods/            # 方法 YAML    (1 方法 = 1 文件)
│   ├── tracks/             # 赛道 YAML    (A/B/C)
│   └── experiments/        # 可选: 预定义实验组合
├── scripts/                # 运行脚本
│   ├── run_one.py / .sh    # 跑单个实验
│   ├── run_all.py / .sh    # 跑多个实验组合
│   ├── leaderboard.py / .sh# 生成排行榜
│   └── prepare_data.py/.sh # 校准数据准备
├── src/                    # 核心源码
│   ├── config.py           # 配置加载/合并
│   ├── registry.py         # 方法注册表 (自动发现)
│   ├── runner.py           # 实验流水线 (加载→量化→评测→保存)
│   ├── evaluator.py        # 评测引擎 (PPL + lm-eval)
│   ├── result_writer.py    # 结果写入 (MD + JSON)
│   ├── env_info.py         # 环境信息收集
│   └── methods/            # 量化方法实现
│       ├── base.py         # 基类 BaseQuantMethod (不可修改)
│       ├── fp16.py         # FP16 baseline
│       ├── rtn.py          # RTN (Round-To-Nearest)
│       ├── awq.py          # AWQ
│       ├── gptq.py         # GPTQ
│       └── smoothquant.py  # SmoothQuant (Track B)
├── results/                # 实验结果 (每次运行生成 .md + .json)
│   └── leaderboard.md      # 排行榜 (由 leaderboard.py 生成)
├── data/                   # 数据集 (raw/processed/meta)
├── plots/                  # 图表 (由脚本生成)
└── docs/                   # 补充文档
```

### 1.1 铁律

- **每个子目录都有 `README.md`**，说明用途、文件索引、使用方法
- **不得新增顶层目录**，除非有充分理由并更新本文档
- **不得修改 `base.py`**，所有方法必须通过继承 + `@register` 接入

---

## 2. 语言与编码规范

### 2.1 语言

| 场景 | 语言 |
|------|------|
| README / summary / daily / 结果文件 | **中文** |
| 代码注释 (函数 docstring、行注释) | **中文** |
| 变量名 / 函数名 / 类名 | **英文** (标准 Python 命名) |
| YAML key | **英文** (snake_case) |
| 控制台输出 (print) | 中文 + emoji 状态图标 |

### 2.2 Python 文件头

每个 `.py` 文件必须以如下格式开头:

```python
# -*- coding: utf-8 -*-
"""
模块英文简称 — 中文一行描述

详细说明 (中文)，包括:
- 这个模块做什么
- 核心实现方式
"""
```

### 2.3 函数 docstring

```python
def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
    """
    中文一句话描述。

    参数:
        model: 说明
        tokenizer: 说明
        calib_data: 说明

    返回:
        Any: 说明
    """
```

### 2.4 控制台输出约定

```
📋  信息/配置
📦  加载中
⚡  执行量化
📊  评测/统计
✅  成功
❌  失败
⚠️   警告
⏱️   耗时
```

---

## 3. 新增量化方法 (最重要的场景)

### 3.1 步骤 (严格按此顺序)

1. **创建** `src/methods/{方法名}.py` — **单文件**，不得拆分多个文件
2. **创建** `configs/methods/{方法名}.yaml`
3. **(可选)** 更新 `configs/models/*.yaml` 添加 `pretrained_quant_models` 条目
4. **更新** `configs/methods/README.md`

### 3.2 方法 Python 文件模板

文件名: `src/methods/{方法名}.py` (全小写，与 YAML 同名)

```python
# -*- coding: utf-8 -*-
"""
XXX — 中文一行描述

详细说明实现方式。
"""

import torch
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


@register("{方法名}")           # ← 与 YAML 的 name 字段一致
class XXXMethod(BaseQuantMethod):
    """方法描述。"""

    supported_tracks = ["A"]    # ← 声明支持的赛道列表

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        """
        执行量化。

        参数:
            model: 原始 FP16/BF16 模型
            tokenizer: tokenizer
            calib_data: 校准数据 (如果 calibration.required=true)

        返回:
            Any: 量化后的模型
        """
        # 1. 从 self.config 读取配置
        w_bits = self.config.get("weight", {}).get("w_bits", 4)
        group_size = self.config.get("weight", {}).get("group_size", 128)

        # 2. 执行量化逻辑
        ...

        # 3. 返回量化后的模型
        return model
```

### 3.3 方法 YAML 模板

文件名: `configs/methods/{方法名}.yaml`

```yaml
# ==============================================================================
# XXX — 中文一行描述
# ==============================================================================

name: "{方法名}"                     # 必填，与 @register 参数一致
display_name: "XXX (全名)"          # 可选，用于排行榜显示
supported_tracks: ["A"]             # 必填
library: "transformers"             # 使用的推理库

weight:
  w_bits: 4
  group_size: 128
  granularity: "per_group"
  scheme: "symmetric"

calibration:
  required: true                    # 是否需要校准数据
```

### 3.4 注册机制 (自动发现)

`src/registry.py` 的 `auto_discover()` 会自动扫描 `src/methods/` 下所有 `.py` 文件并导入。
只要你的类用了 `@register("方法名")`，它就会自动被注册，**不需要手动 import**。

### 3.5 严禁事项

- ❌ 一个方法拆成多个文件 (如 `spark.py` + `spark_cache.py`)
- ❌ 修改 `base.py` 的接口
- ❌ 在 `quantize()` 里 swallow exception — 量化失败必须抛异常
- ❌ 在方法文件中 `print` 不带 emoji 前缀

---

## 4. 新增模型

### 4.1 步骤

1. **创建** `configs/models/{模型名}.yaml`
2. **更新** `configs/models/README.md`

### 4.2 模型 YAML 模板

```yaml
name: "llama3.1-8b"                  # 简短名，用于命令行 --model 参数
model_id: "meta-llama/Llama-3.1-8B"  # HuggingFace Hub 完整 ID

dtype: "bfloat16"                    # 推荐精度
max_seq_len: 131072                  # 模型支持的最大长度
trust_remote_code: false

# 可选
revision: null                       # 指定 HF commit hash (可复现)
tokenizer_id: null                   # 自定义 tokenizer (默认用模型自带)
adapter: null                        # 特殊结构需要 adapter 时填写
model_kwargs: {}                     # 额外传给 from_pretrained 的参数

# 预量化模型 (用于加载 HuggingFace 上已量化好的模型)
pretrained_quant_models:
  awq: "xxx/xxx-AWQ"                 # 可选
  gptq: "xxx/xxx-GPTQ-Int4"          # 可选
```

### 4.3 命名规范

模型 YAML 文件名用 **小写短横线**，与 `name` 字段一致:
- ✅ `llama3.1-8b.yaml` → `name: "llama3.1-8b"`
- ✅ `qwen2.5-7b.yaml` → `name: "qwen2.5-7b"`
- ❌ `Llama_3.1_8B.yaml`

---

## 5. 新增评测指标

### 5.1 PPL 数据集

在 `src/evaluator.py` 的 `evaluate_ppl()` 中添加 dataset 分支:

```python
elif dataset_name == "your_dataset":
    dataset = load_dataset(...)
    text = ...
```

并在 Track YAML (`configs/tracks/track_x.yaml`) 的 `eval.ppl_datasets` 中添加。

### 5.2 lm-eval 任务

直接在 Track YAML 的 `eval.lm_eval_tasks` 中添加 lm-eval-harness 支持的任务名即可。

---

## 6. 命名规范总表

| 对象 | 命名规则 | 示例 |
|------|----------|------|
| 方法 Python 文件 | `src/methods/{name}.py` 小写 | `awq.py`, `rtn.py` |
| 方法 YAML | `configs/methods/{name}.yaml` | `awq.yaml`, `rtn.yaml` |
| 模型 YAML | `configs/models/{name}.yaml` 小写短横线 | `qwen2.5-7b.yaml` |
| Track YAML | `configs/tracks/track_{a/b/c}.yaml` | `track_a.yaml` |
| 结果文件 | `YYYYMMDD_HHMMSS__{model}__{method}__{track}.md/.json` | `20260214_102553__qwen2.5-7b__fp16__trackA.json` |
| `@register` 名 | 与方法 YAML 的 `name` 字段完全一致 | `@register("awq")` |
| 类名 | `{Name}Method` 大驼峰 | `AWQMethod`, `RTNMethod` |

---

## 7. 结果文件规范

每次 `run_one.py` 运行生成一组 `.md` + `.json`:

### 7.1 命名

```
YYYYMMDD_HHMMSS__{model}__{method}__{track}.md
YYYYMMDD_HHMMSS__{model}__{method}__{track}.json
```

### 7.2 JSON 结构 (机器可读，供 leaderboard.py 解析)

```json
{
  "config": { "model": "...", "method": "...", "track": "..." },
  "results": {
    "ppl": { "wikitext2": { "ppl": 6.16 } },
    "lm_eval": { "mmlu": { "acc,none": 0.72 }, "_avg_accuracy": 0.73 },
    "system_metrics": { "vram_peak_mb": 18366.7 }
  },
  "quant_time_seconds": 0.1,
  "environment": { "gpu": "...", "torch": "...", "transformers": "..." }
}
```

### 7.3 MD 结构 (人类可读)

必须包含: 运行时间、CLI 参数、数据集、量化配置、完整指标表格、环境信息。

---

## 8. 赛道 (Tracks)

| Track | 约束 | 代表方法 |
|-------|------|----------|
| **A** | Weight-only W4A16 | FP16, RTN, GPTQ, AWQ, OmniQuant, SpQR |
| **B** | W8A8 (权重+激活) | FP16, SmoothQuant |
| **C** | KV Cache 量化 | FP16, KIVI, KVQuant |

方法的 `supported_tracks` 必须准确声明。runner 会在运行前校验。

---

## 9. 配置优先级 (从低到高)

```
configs/config.yaml (全局默认)
  ↓ 被覆盖
configs/models/{model}.yaml
  ↓ 被覆盖
configs/methods/{method}.yaml
  ↓ 被覆盖
configs/tracks/track_{track}.yaml
  ↓ 被覆盖
CLI 参数 (--override key=value)
```

---

## 10. 可复现性检查清单

每次实验结果 **必须** 自动记录:

- [ ] Git commit hash
- [ ] conda 环境名 + 关键包版本 (torch, transformers, lm-eval)
- [ ] GPU 型号 + CUDA 版本
- [ ] HF 模型 revision
- [ ] 完整 CLI 参数 (可复制)
- [ ] 校准数据集版本 + seed
- [ ] 评测数据集版本

如果发生 fallback (OOM / kernel 降级), **必须** 在结果 MD 中标注 `⚠️ fallback detected`。

---

## 11. 开发流程

### 做任何修改前

1. 读本文档
2. 检查现有代码模式 (看 `rtn.py` 和 `rtn.yaml` 作为参考实现)
3. 遵循单文件约定

### 修改后

1. 更新对应目录的 `README.md`
2. 更新 `daily.md` 记录改动
3. 跑 `--dry_run` 验证配置正确
4. 跑最小实验验证功能
5. 更新 `summary.md` (如果结果有变化)
6. 提交到 Git

---

## 12. 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `方法 'xxx' 未注册` | `@register` 名与 YAML `name` 不一致 | 检查两者是否完全相同 |
| `方法 xxx 不支持 Track Y` | `supported_tracks` 未声明该 Track | 在类中添加 |
| PPL 没有退化 | 量化没有真正生效 | 检查 `quantize()` 是否修改了权重/KV |
| 结果 JSON 解析失败 | 字段名不一致 | 检查 `result_writer.py` 输出格式 |
| 模型下载失败 | Gated model 需要 HF_TOKEN | 设置 `export HF_TOKEN=xxx` |
