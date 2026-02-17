# configs/experiments/ — 实验组合配置

## 用途
定义具体的 model × method × track 实验组合，方便一键批量运行。

## 使用方式
```bash
python scripts/run_all.py --experiment configs/experiments/quick_test.yaml
```

## 命名规范
- `quick_test.yaml` — 用于快速验证的最小组合
- `full_sweep.yaml` — 全量笛卡尔积
- `track_a_all.yaml` — Track A 所有方法
- `track_c_all.yaml` — Track C 所有方法 (含长上下文)

## 注意事项
- experiment YAML 的优先级低于 CLI 参数
- 可以用 `--exclude_models` / `--exclude_methods` 进一步过滤
