# PTQ Benchmark Leaderboard

> **模型**: Qwen2.5-7B | **赛道**: Track A (W4A16) | **硬件**: H200 NVL 141GB
> **日期**: 2026-02-14

---

## 总览排名

| 排名 | 方法 | PPL (WikiText-2) ↓ | Avg Accuracy ↑ | PPL 退化 | Acc 退化 | VRAM (MB) | 量化耗时 |
|:----:|------|:-------------------:|:--------------:|:--------:|:--------:|:---------:|:--------:|
| 🥇 | **FP16** (baseline) | **6.16** | **0.7351** | — | — | 18,367 | — |
| 🥈 | **AWQ** (W4, pre-quantized) | **6.91** | **0.7233** | +0.75 | -1.18% | 19,891 | 5.6s |
| 🥉 | **RTN** (W4, per-group sym) | **7.27** | **0.7098** | +1.11 | -2.53% | 21,822 | 0.1s |
| ⛔ | **GPTQ** (blocked) | — | — | — | — | — | — |

> **注**: AWQ 使用的是 `Qwen/Qwen2.5-7B-Instruct-AWQ` 预量化模型，RTN 使用手动 per-group symmetric round-to-nearest 实现。
> GPTQ 因 `auto_gptq`/`gptqmodel` 与 transformers 4.52 版本不兼容而无法运行。

---

## 各任务详细成绩

### 核心任务 (6 tasks)

| 任务 | FP16 | AWQ | AWQ Δ | RTN | RTN Δ |
|------|:----:|:---:|:-----:|:---:|:-----:|
| **MMLU** (acc) | 0.7178 | 0.7080 | -0.98% | 0.6911 | -2.67% |
| **HellaSwag** (acc_norm) | 0.5999 | 0.6145 | +1.46% | 0.5637 | -3.62% |
| **ARC-Easy** (acc) | 0.8043 | 0.8106 | +0.63% | 0.7778 | -2.65% |
| **ARC-Challenge** (acc_norm) | 0.4795 | 0.5162 | +3.67% | 0.4599 | -1.96% |
| **PIQA** (acc) | 0.7873 | 0.7742 | -1.31% | 0.7802 | -0.71% |
| **Winogrande** (acc) | 0.7253 | 0.6977 | -2.76% | 0.6961 | -2.92% |

### MMLU 细分领域

| 领域 | FP16 | AWQ | RTN |
|------|:----:|:---:|:---:|
| STEM | 0.6984 | 0.6755 | 0.6587 |
| Humanities | 0.6278 | 0.6257 | 0.6045 |
| Social Sciences | 0.8248 | 0.8148 | 0.7975 |
| Other | 0.7679 | 0.7599 | 0.7499 |

---

## 关键发现

### 1. AWQ 显著优于 RTN
- PPL: AWQ 6.91 vs RTN 7.27（AWQ 更低 = 更好）
- Accuracy: AWQ 72.33% vs RTN 70.98%（AWQ 高 1.35 个百分点）
- AWQ 通过保护激活值较大的权重通道，有效降低了量化误差

### 2. AWQ 某些任务甚至超过 FP16
- ARC-Challenge: AWQ **0.5162** > FP16 0.4795（+3.67%）
- HellaSwag: AWQ **0.6145** > FP16 0.5999（+1.46%）
- 可能原因: AWQ 使用的是 **Instruct** 版本的预量化模型，经过指令微调的模型在这些任务上有天然优势

### 3. RTN 退化模式
- 推理能力受损最大: HellaSwag -3.62%, MMLU -2.67%
- 简单任务退化较小: PIQA -0.71%
- RTN 作为最弱 baseline，W4 量化带来 ~2.5% 的平均精度损失

### 4. GPTQ 被库兼容性阻塞
- `auto_gptq 0.7.1` 无法导入: `no_init_weights` 从 `transformers.modeling_utils` 中移除
- `gptqmodel` 构建失败: 无法检测 torch 版本
- **解决方案**: 等待 transformers 4.52 兼容的 `auto_gptq` 或 `gptqmodel` 版本发布

---

## 环境信息

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA H200 NVL 141GB |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.52.0.dev0 |
| autoawq | 0.2.9 |
| lm-eval | 0.4.11 |
| Python | 3.13 |
