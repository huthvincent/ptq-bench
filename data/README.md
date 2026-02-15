# data/ — 数据集说明（指向共享目录）

## ⚠️ 注意：实际数据存储在共享目录

实际的数据集存储在项目外的共享目录：
```
/data2/zhu11/quant_source/data/
├── raw/               # HuggingFace datasets 原始数据
├── processed/         # 预处理后的数据（校准 token blocks 等）
└── meta/              # 数据版本与 hash 记录
```

**为什么不放在项目内？**
1. 方便 ptq-bench 上传 GitHub（数据太大不适合提交）
2. 未来的 in-training quant 项目可以共享同一份数据
3. HuggingFace 模型缓存也在 `/data2/zhu11/quant_source/models/`

## 路径配置

在 `configs/config.yaml` 中配置：
```yaml
paths:
  data_root: "/data2/zhu11/quant_source/data"
  model_cache_dir: "/data2/zhu11/quant_source/models"
```

## 使用的数据集

| 数据集 | 用途 | 来源 |
|--------|------|------|
| WikiText-2 | 校准 / PPL 评测 | HuggingFace `wikitext` (`wikitext-2-raw-v1`) |
| C4 | 校准（可选） | HuggingFace `allenai/c4` (`en`) |
| lm-eval tasks | 评测 | lm-evaluation-harness 内置 |
| LongBench | 长上下文评测（Phase 2） | HuggingFace `THUDM/LongBench` |

## 注意事项
- 此目录下不存放实际数据文件（只有本 README）
- 通过 `HF_HOME` 环境变量或 `config.yaml` 控制缓存位置
- 详见 `/data2/zhu11/quant_source/README.md`
