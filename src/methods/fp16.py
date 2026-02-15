# -*- coding: utf-8 -*-
"""
FP16 Baseline — 不做任何量化

作为参考基准，直接加载原始模型进行评测。
所有其他方法的结果都与 FP16 baseline 做对比。
"""

from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


@register("fp16")
class FP16Method(BaseQuantMethod):
    """
    FP16 / BF16 baseline，不做任何量化。

    直接返回原始模型，用于建立参考基准。
    """

    supported_tracks = ["A", "B", "C"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        """
        不执行量化，直接返回原始模型。

        参数:
            model: 原始模型
            tokenizer: tokenizer（未使用）
            calib_data: 校准数据（未使用）

        返回:
            Any: 原始模型（不做修改）
        """
        print("📋 FP16 Baseline: 不执行量化，直接使用原始模型")
        return model

    def requires_calibration(self) -> bool:
        """FP16 不需要校准数据。"""
        return False

    def get_quant_spec(self) -> dict:
        """返回 FP16 的量化规格（无量化）。"""
        return {
            "method": "fp16",
            "quantize": False,
            "description": "原始 FP16/BF16 精度，无量化",
        }
