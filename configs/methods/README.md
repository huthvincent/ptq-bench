# configs/methods/ — 量化方法配置

## 用途
每个 YAML 文件定义一个量化方法的默认参数。新增方法需要：
1. 在此目录加一个 YAML（定义默认参数）
2. 在 `src/methods/` 加一个 wrapper 文件（实现量化逻辑）

## 当前方法

| 方法 | Track | 关键参数 | 状态 |
|------|:-----:|----------|:----:|
| `fp16.yaml` | A,B,C | 无量化 | ✅ |
| `rtn.yaml` | A | w_bits=4, group_size=128 | ✅ |
| `gptq.yaml` | A | w_bits=4, group_size=128 | ⚠️ 库阻塞 |
| `awq.yaml` | A | w_bits=4, pre-quantized | ✅ |
| `smoothquant.yaml` | B | w_bits=8, a_bits=8 | ✅ |
| `forge.yaml` | C | chunk_size=16, energy=0.95 | ✅ |
| `kivi.yaml` | C | k/v_bits=2, residual=32 | ✅ |
| `kvquant.yaml` | C | k/v_bits=2, residual=32, outlier | ✅ |

## 必填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | 方法简称 | `gptq` |
| `display_name` | 展示名 | `GPTQ` |
| `supported_tracks` | 支持的 Track | `["A"]` |
| `library` | 依赖库 | `auto-gptq` |
| `wrapper` | wrapper 模块名 | `gptq` |

## Track C 关键参数

### KIVI / KVQuant
- `k_bits` / `v_bits`: Key/Value 量化位数（当前 2-bit）
- `residual_length`: 保留 FP16 的最近 token 数（当前 32）

### FORGE
- `chunk_size`: SVD 分块大小（当前 16）
- `energy_threshold`: SVD 能量保留阈值（0.95）
- `min_rank` / `max_rank`: 动态秩范围

## 新增方法教程

1. 复制现有 YAML，修改所有字段
2. 在 `src/methods/` 创建 `你的方法.py`，继承 `BaseQuantMethod`
3. 实现 `quantize(model, tokenizer, calib_data, config)` 方法
4. 运行 `--dry_run` 验证

## 命名规范
- 文件名：方法名全小写，与 `name` 字段一致
- 示例：`gptq.yaml`、`forge.yaml`、`kvquant.yaml`
