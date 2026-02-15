# configs/tracks/ — 赛道配置

## 用途
定义三条 benchmark 赛道（Track），每条赛道限定量化类型和评测组合。

## 当前赛道

| Track | 量化类型 | 描述 |
|-------|---------|------|
| A | W4A16 (Weight-only) | 只量化权重到 4-bit，激活保持 FP16 |
| B | W8A8 | 权重和激活都量化到 8-bit |
| C | KV Cache Quant | 量化 KV Cache 以节省长上下文显存 |

## 文件列表
- `track_a.yaml` — Weight-only W4A16 赛道
- `track_b.yaml` — W8A8 赛道
- `track_c.yaml` — KV Cache 量化赛道

## 注意事项
- 每个方法在其 YAML 里声明自己支持哪些 Track
- `run_one.py --track A` 会自动只加载支持 Track A 的方法
- Track C 的长上下文评测是 Phase 2 功能
