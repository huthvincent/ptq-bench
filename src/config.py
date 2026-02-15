# -*- coding: utf-8 -*-
"""
配置加载与合并模块

负责从 YAML 文件加载配置，并按优先级合并：
CLI 参数 > experiment YAML > method YAML > track YAML > 全局 config.yaml

主要功能：
- load_global_config(): 加载全局配置
- load_model_config(): 加载模型配置
- load_method_config(): 加载方法配置
- load_track_config(): 加载赛道配置
- merge_configs(): 按优先级合并所有配置
"""

import os
import yaml
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional


def get_project_root() -> Path:
    """
    获取项目根目录路径。

    通过向上查找 configs/config.yaml 来确定项目根目录。
    如果找不到，回退到当前工作目录。

    返回:
        Path: 项目根目录的绝对路径
    """
    current = Path(__file__).resolve().parent.parent
    if (current / "configs" / "config.yaml").exists():
        return current
    # 回退：从当前工作目录查找
    cwd = Path.cwd()
    if (cwd / "configs" / "config.yaml").exists():
        return cwd
    return cwd


def load_yaml(path: str | Path) -> dict:
    """
    加载一个 YAML 文件并返回字典。

    参数:
        path: YAML 文件的路径（支持字符串或 Path 对象）

    返回:
        dict: 解析后的 YAML 内容

    异常:
        FileNotFoundError: 文件不存在时抛出
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """
    深度合并两个字典。override 中的值会覆盖 base 中的同名键。

    对于嵌套字典，递归合并而不是直接替换。
    对于列表和标量值，override 直接替换 base。

    参数:
        base: 基础字典（低优先级）
        override: 覆盖字典（高优先级）

    返回:
        dict: 合并后的新字典（不修改原字典）
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_global_config(project_root: Path | None = None) -> dict:
    """
    加载全局配置 configs/config.yaml。

    参数:
        project_root: 项目根目录，None 时自动检测

    返回:
        dict: 全局配置字典
    """
    if project_root is None:
        project_root = get_project_root()
    return load_yaml(project_root / "configs" / "config.yaml")


def load_model_config(model_name: str, project_root: Path | None = None) -> dict:
    """
    加载模型配置 configs/models/{model_name}.yaml。

    参数:
        model_name: 模型名称（不带 .yaml 后缀）
        project_root: 项目根目录

    返回:
        dict: 模型配置字典
    """
    if project_root is None:
        project_root = get_project_root()
    path = project_root / "configs" / "models" / f"{model_name}.yaml"
    return load_yaml(path)


def load_method_config(method_name: str, project_root: Path | None = None) -> dict:
    """
    加载量化方法配置 configs/methods/{method_name}.yaml。

    参数:
        method_name: 方法名称（不带 .yaml 后缀）
        project_root: 项目根目录

    返回:
        dict: 方法配置字典
    """
    if project_root is None:
        project_root = get_project_root()
    path = project_root / "configs" / "methods" / f"{method_name}.yaml"
    return load_yaml(path)


def load_track_config(track_name: str, project_root: Path | None = None) -> dict:
    """
    加载赛道配置 configs/tracks/track_{track_name.lower()}.yaml。

    参数:
        track_name: 赛道名称（A / B / C，不区分大小写）
        project_root: 项目根目录

    返回:
        dict: 赛道配置字典
    """
    if project_root is None:
        project_root = get_project_root()
    path = project_root / "configs" / "tracks" / f"track_{track_name.lower()}.yaml"
    return load_yaml(path)


def load_experiment_config(experiment_path: str, project_root: Path | None = None) -> dict:
    """
    加载实验组合配置。

    参数:
        experiment_path: 实验配置文件路径（相对于项目根目录或绝对路径）
        project_root: 项目根目录

    返回:
        dict: 实验配置字典
    """
    if project_root is None:
        project_root = get_project_root()
    path = Path(experiment_path)
    if not path.is_absolute():
        path = project_root / path
    return load_yaml(path)


def merge_configs(
    global_config: dict,
    model_config: dict,
    method_config: dict,
    track_config: dict,
    cli_overrides: dict | None = None,
) -> dict:
    """
    按优先级合并所有配置层。

    合并顺序（低 → 高优先级）：
    1. 全局配置 (config.yaml)
    2. Track 配置 (track_X.yaml)
    3. 方法配置 (method.yaml)
    4. CLI 参数覆盖

    模型配置单独存放在 merged["model"] 下，不参与覆盖。

    参数:
        global_config: 全局配置
        model_config: 模型配置
        method_config: 方法配置
        track_config: 赛道配置
        cli_overrides: CLI 参数覆盖（可选）

    返回:
        dict: 合并后的最终配置
    """
    # 按优先级依次合并
    merged = copy.deepcopy(global_config)
    merged = deep_merge(merged, track_config)
    merged = deep_merge(merged, method_config)

    # 模型配置放在独立 key 下
    merged["model"] = copy.deepcopy(model_config)

    # CLI 覆盖优先级最高
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)

    return merged


def resolve_paths(config: dict, project_root: Path | None = None) -> dict:
    """
    将配置中的相对路径解析为绝对路径。

    参数:
        config: 配置字典
        project_root: 项目根目录

    返回:
        dict: 路径已解析的配置字典
    """
    if project_root is None:
        project_root = get_project_root()

    config = copy.deepcopy(config)
    if "paths" in config:
        for key, value in config["paths"].items():
            if value and isinstance(value, str) and not Path(value).is_absolute():
                config["paths"][key] = str(project_root / value)
    return config


def dump_config(config: dict) -> str:
    """
    将配置字典转换为可读的 YAML 字符串。

    参数:
        config: 配置字典

    返回:
        str: 格式化的 YAML 字符串
    """
    return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)


def validate_config(config: dict) -> list[str]:
    """
    验证合并后的配置是否包含必要字段。

    参数:
        config: 合并后的配置字典

    返回:
        list[str]: 验证错误列表（空列表表示验证通过）
    """
    errors = []

    # 检查模型配置
    if "model" not in config:
        errors.append("缺少 model 配置")
    elif "model_id" not in config.get("model", {}):
        errors.append("model 配置中缺少 model_id")

    # 检查方法配置
    if "name" not in config:
        errors.append("缺少 method name")

    return errors
