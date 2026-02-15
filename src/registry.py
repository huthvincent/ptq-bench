# -*- coding: utf-8 -*-
"""
量化方法注册表

负责自动发现并注册 src/methods/ 目录下的所有量化方法。
采用简单的字典注册模式，而不是复杂的 metaclass 或 entry_points。

主要功能：
- register(): 注册一个量化方法类
- get(): 获取已注册的方法类
- list_methods(): 列出所有已注册方法
- auto_discover(): 自动扫描并导入 methods/ 目录下的所有模块
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Type

# 全局注册表：方法名 → 方法类
_REGISTRY: dict[str, Type] = {}


def register(name: str):
    """
    方法注册装饰器。

    用法:
        @register("gptq")
        class GPTQMethod(BaseQuantMethod):
            ...

    参数:
        name: 方法名称（与 configs/methods/ 下的 YAML 文件名对应）

    返回:
        装饰器函数
    """
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"方法 '{name}' 已注册，不能重复注册。已注册的类: {_REGISTRY[name].__name__}")
        _REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return decorator


def get(name: str):
    """
    获取已注册的量化方法类。

    参数:
        name: 方法名称

    返回:
        Type: 对应的方法类

    异常:
        KeyError: 方法未注册时抛出
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"方法 '{name}' 未注册。可用方法: {available}")
    return _REGISTRY[name]


def list_methods() -> list[str]:
    """
    列出所有已注册的方法名称。

    返回:
        list[str]: 已注册的方法名称列表（按字母排序）
    """
    return sorted(_REGISTRY.keys())


def get_methods_for_track(track: str) -> list[str]:
    """
    获取支持指定赛道的所有方法名称。

    参数:
        track: 赛道名称（A / B / C）

    返回:
        list[str]: 支持该赛道的方法名称列表
    """
    result = []
    for name, cls in _REGISTRY.items():
        supported = getattr(cls, "supported_tracks", [])
        if track.upper() in [t.upper() for t in supported]:
            result.append(name)
    return sorted(result)


def auto_discover():
    """
    自动扫描 src/methods/ 目录下的所有 Python 模块并导入。

    导入时，模块内用 @register 装饰器标注的类会自动注册到全局注册表中。
    跳过 __init__.py 和 base.py（基类模块）。
    """
    methods_dir = Path(__file__).resolve().parent / "methods"
    if not methods_dir.exists():
        return

    package_name = "src.methods"

    for module_info in pkgutil.iter_modules([str(methods_dir)]):
        # 跳过 __init__ 和 base 模块
        if module_info.name in ("__init__", "base"):
            continue
        module_name = f"{package_name}.{module_info.name}"
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            # 某些方法的依赖可能未安装，打印警告但不中断
            print(f"⚠️  跳过方法模块 {module_info.name}: 导入失败 ({e})")
        except Exception as e:
            print(f"⚠️  跳过方法模块 {module_info.name}: {e}")
