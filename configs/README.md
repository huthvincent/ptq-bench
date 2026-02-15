# configs/ — 配置体系

## 目录结构

```
configs/
├── config.yaml          # 全局基础配置（环境、路径、通用超参）
├── models/              # 模型配置（每个 LLM 一个 YAML）
├── methods/             # 量化方法配置（每个方法一个 YAML，含默认校准参数）
├── tracks/              # 赛道配置（Track A/B/C，含评测组合）
└── experiments/         # 实验组合配置（可选：定义具体的 model×method×track 组合）
```

## 使用方式

### 全局配置 `config.yaml`
- 包含 conda 环境名称、Python/CUDA/PyTorch 版本等环境信息
- 存放统一路径（data_root、results_root 等）
- 通用超参（seed、max_seq_len、dtype 等）

### 新增模型
1. 在 `models/` 里复制一个现有 YAML（如 `llama3.1-8b.yaml`）
2. 修改 `model_id`（HuggingFace 模型名）、`dtype`、`max_seq_len` 等字段
3. 90% 情况下只改 YAML 即可运行，无需写代码
4. 详见 `models/README.md`

### 新增量化方法
1. 在 `methods/` 加一个 YAML（默认超参 + 支持的 Track）
2. 在 `src/methods/` 加一个 wrapper Python 文件
3. 在 wrapper 里继承 `BaseQuantMethod` 并实现 `quantize()` 方法
4. 详见 `methods/README.md`

### 新增 Track
- 通常不需要新增 Track，A/B/C 已覆盖主流 PTQ 场景
- 如需新增，在 `tracks/` 加一个 YAML 并定义评测任务集

## 注意事项
- CLI 参数 > experiment YAML > method YAML > track YAML > config.yaml（优先级从高到低）
- 所有路径支持相对路径（相对于项目根目录）和绝对路径
- 修改配置后建议用 `--dry_run` 检查合并后的最终配置
