# scripts/ — 可执行脚本

## 用途
包含项目的所有可执行脚本。每个 Python 脚本都配套一个 `.sh` 启动脚本。

## 核心脚本

| 脚本 | 用途 | 配套 .sh |
|------|------|----------|
| `run_one.py` | 运行单个实验（一个 model × method × track） | `run_one.sh` |
| `run_all.py` | 批量运行多个实验组合 | `run_all.sh` |
| `leaderboard.py` | 从 results/ 汇总生成排行榜 | `leaderboard.sh` |
| `prepare_data.py` | 准备校准数据（Phase 2） | `prepare_data.sh` |

## 使用方式

### 跑单个实验
```bash
bash scripts/run_one.sh
# 或者直接用 Python:
python scripts/run_one.py --model llama3.1-8b --method gptq --track A
```

### 批量实验
```bash
bash scripts/run_all.sh
# 或者:
python scripts/run_all.py --experiment configs/experiments/quick_test.yaml
```

### 生成排行榜
```bash
bash scripts/leaderboard.sh
# 或者:
python scripts/leaderboard.py --results_dir results/
```

## 注意事项
- `.sh` 脚本里的变量都集中在顶部，修改参数只需改变量值
- 所有脚本支持 `--dry_run` 参数（只打印配置，不实际运行）
- 运行前确保已激活正确的 conda 环境
