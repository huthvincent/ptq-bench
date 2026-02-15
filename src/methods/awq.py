# -*- coding: utf-8 -*-
"""
AWQ — Activation-aware Weight Quantization

加载 HuggingFace Hub 上的预量化 AWQ 模型进行评测。
使用 autoawq 的 from_quantized() 加载预量化模型。
"""

from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


@register("awq")
class AWQMethod(BaseQuantMethod):
    """
    AWQ 量化方法 wrapper。

    使用 autoawq 的 from_quantized() 加载预量化 AWQ 模型。
    """

    supported_tracks = ["A"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        """
        加载预量化的 AWQ 模型。

        参数:
            model: 原始模型（将被释放）
            tokenizer: tokenizer
            calib_data: 校准数据（预量化模型不需要）

        返回:
            Any: 预量化的 AWQ 模型
        """
        import torch
        from awq import AutoAWQForCausalLM

        model_id = self.config.get("model", {}).get("model_id", "")
        pretrained_quant = self.config.get("model", {}).get("pretrained_quant_models", {})
        awq_model_id = pretrained_quant.get("awq", model_id + "-AWQ")

        cache_dir = self.config.get("paths", {}).get("model_cache_dir", None)
        trust_remote_code = self.config.get("model", {}).get("trust_remote_code", False)

        print(f"📋 AWQ: 加载预量化模型: {awq_model_id}")

        # 释放原始模型
        del model
        torch.cuda.empty_cache()

        # 使用 autoawq 的 from_quantized 加载
        awq_model = AutoAWQForCausalLM.from_quantized(
            awq_model_id,
            fuse_layers=False,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )

        print(f"✅ AWQ 预量化模型加载完成: {awq_model_id}")
        print(f"   参数量: {sum(p.numel() for p in awq_model.model.parameters()) / 1e9:.2f}B")

        # 返回底层的 transformers 模型
        return awq_model.model
