# scripts/ — 可执行脚本

## 用途
包含项目的所有可执行脚本。每个 Python 脚本都配套一个 `.sh` 启动脚本。

## 核心脚本

| 脚本 | 用途 | 配套 .sh |
|------|------|----------|
| `run_one.py` | 运行单个实验（一个 model × method × track） | `run_one.sh` |
| `run_all.py` | 批量运行多个实验组合 | `run_all.sh` |
| `leaderboard.py` | 从 results/ 汇总生成排行榜 | `leaderboard.sh` |
| `prepare_data.py` | 准备校准数据 | `prepare_data.sh` |

## 使用示例

### 单个实验
```bash
# Track C: Qwen2.5-7B + KIVI
python scripts/run_one.py --model qwen2.5-7b --method kivi --track C

# Dry run (只打印配置):
python scripts/run_one.py --model qwen2.5-7b --method kivi --track C --dry_run
```

### 批量实验
```bash
python scripts/run_all.py \
    --include_models qwen2.5-7b mistral-7b \
    --include_methods fp16 forge kivi kvquant \
    --include_tracks C
```

### 生成排行榜
```bash
python scripts/leaderboard.py --results_dir results/
```

## 注意事项
- `.sh` 脚本里的变量集中在顶部，修改参数只需改变量值
- 所有脚本支持 `--dry_run` 参数
- 运行前确保已激活 `ptq-bench` conda 环境
