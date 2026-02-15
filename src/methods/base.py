# -*- coding: utf-8 -*-
"""
量化方法基类

定义所有量化方法必须实现的接口。
新增方法时，继承 BaseQuantMethod 并实现 quantize() 方法即可。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseQuantMethod(ABC):
    """
    所有量化方法的基类。

    新增一个量化方法需要：
    1. 继承此类
    2. 用 @register("方法名") 装饰器注册
    3. 实现 quantize() 方法
    4. 可选实现 load_quantized() 方法

    属性:
        name (str): 方法名称（由 registry 自动设置）
        supported_tracks (list[str]): 支持的赛道列表, 如 ["A", "B"]
    """

    # 子类必须声明支持哪些赛道
    supported_tracks: list[str] = []

    def __init__(self, config: dict):
        """
        初始化量化方法。

        参数:
            config: 合并后的完整配置字典
        """
        self.config = config

    @abstractmethod
    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        """
        执行量化操作。

        这是核心方法，每个量化算法必须实现。

        参数:
            model: 原始（FP16/BF16）HuggingFace 模型
            tokenizer: 对应的 tokenizer
            calib_data: 校准数据（如果方法需要校准的话），格式为 token ids 列表

        返回:
            Any: 量化后的模型（可以是原始模型或新模型对象）
        """
        raise NotImplementedError

    def load_quantized(self, model_path: str, config: dict) -> Any:
        """
        加载已保存的量化模型。（可选实现）

        参数:
            model_path: 量化模型保存路径
            config: 配置字典

        返回:
            Any: 加载的量化模型
        """
        raise NotImplementedError(f"{self.__class__.__name__} 不支持加载已保存的量化模型")

    def get_quant_spec(self) -> dict:
        """
        返回当前量化配置的摘要（用于结果记录）。

        返回:
            dict: 量化规格摘要
        """
        spec = {
            "method": getattr(self, "_registry_name", self.__class__.__name__),
            "supported_tracks": self.supported_tracks,
        }
        # 从 config 中提取量化参数
        for key in ("weight", "activation", "kv"):
            if key in self.config:
                spec[key] = self.config[key]
        return spec

    def requires_calibration(self) -> bool:
        """
        判断该方法是否需要校准数据。

        返回:
            bool: 是否需要校准
        """
        calib_config = self.config.get("calibration", {})
        return calib_config.get("required", True)
