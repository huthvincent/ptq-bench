# -*- coding: utf-8 -*-
"""
RTN (Round-To-Nearest) — 最简单的权重量化 baseline

直接对权重做 per-group round-to-nearest 对称量化，
不使用校准数据优化。这是量化的最弱 baseline。

实现方式: 手动对每个线性层的权重做 per-group symmetric 量化。
"""

import torch
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


def _quantize_tensor_rtn(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """
    对单个权重张量执行 per-group symmetric round-to-nearest 量化。

    参数:
        weight: 形状 [out_features, in_features] 的权重
        bits: 量化位宽
        group_size: 每组的元素数

    返回:
        torch.Tensor: 模拟量化 (simulate quantize) 后的权重
    """
    orig_shape = weight.shape
    orig_dtype = weight.dtype

    # 展平为 [out, in] 然后按 group_size 分组
    w = weight.float().reshape(-1, group_size)

    # 对称量化: scale = max(|w|) / (2^(bits-1) - 1)
    qmax = (1 << (bits - 1)) - 1
    scales = w.abs().amax(dim=1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-10)

    # 量化再反量化 (simulate quantize)
    w_q = (w / scales).round().clamp(-qmax, qmax)
    w_deq = w_q * scales

    return w_deq.reshape(orig_shape).to(orig_dtype)


@register("rtn")
class RTNMethod(BaseQuantMethod):
    """
    RTN 量化方法。

    通过手动对每个线性层权重做 per-group symmetric round-to-nearest 量化。
    不使用校准数据，纯粹基于权重分布来量化。
    """

    supported_tracks = ["A"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        """
        执行 RTN 量化。

        遍历模型所有线性层，对权重做 per-group RTN 量化。
        使用 simulate quantize (伪量化): 量化后立即反量化回 float。

        参数:
            model: 原始模型
            tokenizer: tokenizer（未使用）
            calib_data: 校准数据（RTN 不使用）

        返回:
            Any: 伪量化后的模型
        """
        w_bits = self.config.get("weight", {}).get("w_bits", 4)
        group_size = self.config.get("weight", {}).get("group_size", 128)

        print(f"📋 RTN: W{w_bits} group_size={group_size}")
        print(f"📋 RTN: 手动 per-group symmetric round-to-nearest")

        n_quantized = 0
        n_skipped = 0

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    w = module.weight.data
                    # 跳过太小的权重 (如 lm_head 如果 vocab 不能被 group_size 整除)
                    if w.numel() % group_size != 0:
                        n_skipped += 1
                        continue
                    module.weight.data = _quantize_tensor_rtn(w, w_bits, group_size)
                    n_quantized += 1

        print(f"✅ RTN 量化完成: {n_quantized} 个线性层量化, {n_skipped} 个跳过")
        return model
