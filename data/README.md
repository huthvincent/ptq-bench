# data/ — 数据集说明（指向共享目录）

## ⚠️ 注意：实际数据存储在共享目录

实际的数据集存储在项目外的共享目录：
```
/data2/zhu11/quant_source/data/
├── raw/               # HuggingFace datasets 原始数据
├── processed/         # 预处理后的数据
└── meta/              # 数据版本与 hash 记录
```

**为什么不放在项目内？**
1. 方便 GitHub 管理（数据太大不适合提交）
2. HuggingFace 模型缓存也在 `/data2/zhu11/quant_source/models/`

## 路径配置

在 `configs/config.yaml` 中：
```yaml
paths:
  data_root: "/data2/zhu11/quant_source/data"
  model_cache_dir: "/data2/zhu11/quant_source/models"
```

## 评测数据集

| 数据集 | 用途 | 来源 |
|--------|------|------|
| WikiText-2 | 校准 + PPL 评测 | HuggingFace `wikitext` (`wikitext-2-raw-v1`) |
| PG19-test | 长上下文 PPL 评测 (max_seq_len=4096) | HuggingFace `emozilla/pg19-test` |
| lm-eval tasks | 准确率评测 | lm-evaluation-harness 内置 |

## 注意事项
- 此目录下不存放实际数据文件
- 通过 `HF_HOME` 环境变量或 `config.yaml` 控制缓存位置
