# configs/methods/ — 量化方法配置

## 用途
每个 YAML 文件定义一个量化方法的默认参数。新增方法需要：
1. 在此目录加一个 YAML（定义默认参数）
2. 在 `src/methods/` 加一个 wrapper 文件（实现量化逻辑）

## 必填字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | 方法简称 | `gptq` |
| `display_name` | 方法展示名 | `GPTQ` |
| `supported_tracks` | 支持的 Track 列表 | `["A"]` |
| `library` | 依赖的 Python 库 | `auto-gptq` |
| `wrapper` | 对应的 wrapper 模块名 | `gptq` |

## 量化参数字段（按 Track）

### Track A (W4A16) 常用字段
- `w_bits`: 权重量化位数 (4)
- `group_size`: 分组大小 (128)
- `granularity`: 量化粒度 (per_group / per_channel)
- `scheme`: 量化方案 (symmetric / asymmetric)

### Track B (W8A8) 常用字段
- `w_bits`: 权重位数 (8)
- `a_bits`: 激活位数 (8)
- `smoothquant_alpha`: SmoothQuant 迁移强度 (0.5)

### Track C (KV quant) 常用字段
- `k_bits`: Key 缓存位数
- `v_bits`: Value 缓存位数
- `kv_group_size`: KV 缓存分组大小

## 新增方法教程

1. 复制 `gptq.yaml`，修改所有字段
2. 在 `src/methods/` 创建 `你的方法名.py`，继承 `BaseQuantMethod`
3. 实现 `quantize(model, tokenizer, calib_data, config)` 方法
4. 运行 `python scripts/run_one.py --method 你的方法名 --model llama3.1-8b --track A --dry_run` 验证

## 命名规范
- 文件名：方法名全小写，与 `name` 字段一致
- 示例：`gptq.yaml`、`awq.yaml`、`smoothquant.yaml`
