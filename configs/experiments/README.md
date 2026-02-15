# configs/experiments/ — 实验组合配置

## 用途
定义具体的 model × method × track 实验组合，方便一键跑批量实验。

## 使用方式
```bash
# 使用实验配置运行
python scripts/run_all.py --experiment configs/experiments/quick_test.yaml
```

## 命名规范
- `quick_test.yaml` — 用于快速验证的最小实验组合
- `full_sweep.yaml` — 全量笛卡尔积
- `track_a_all.yaml` — Track A 所有方法

## 注意事项
- experiment YAML 的优先级低于 CLI 参数
- 可以用 `--exclude_models` / `--exclude_methods` 进一步过滤
