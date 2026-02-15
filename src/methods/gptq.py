# -*- coding: utf-8 -*-
"""
GPTQ — 基于 Hessian 信息的权重量化方法

加载 HuggingFace Hub 上的预量化 GPTQ 模型进行评测。
由于 auto_gptq 与 transformers 4.52 存在兼容性问题
（no_init_weights 导入失败），因此使用预量化模型。
"""

from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


@register("gptq")
class GPTQMethod(BaseQuantMethod):
    """
    GPTQ 量化方法 wrapper。

    加载 HuggingFace Hub 上的预量化 GPTQ 模型。
    """

    supported_tracks = ["A"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        """
        加载预量化的 GPTQ 模型。

        参数:
            model: 原始模型（将被释放）
            tokenizer: tokenizer
            calib_data: 校准数据（预量化模型不需要）

        返回:
            Any: 预量化的 GPTQ 模型
        """
        import torch

        model_id = self.config.get("model", {}).get("model_id", "")
        pretrained_quant = self.config.get("model", {}).get("pretrained_quant_models", {})
        gptq_model_id = self.config.get("weight", {}).get("pretrained_model_id", None)
        if gptq_model_id is None:
            gptq_model_id = pretrained_quant.get("gptq", model_id + "-GPTQ-Int4")

        cache_dir = self.config.get("paths", {}).get("model_cache_dir", None)
        trust_remote_code = self.config.get("model", {}).get("trust_remote_code", False)

        print(f"📋 GPTQ: 加载预量化模型: {gptq_model_id}")

        # 释放原始模型
        del model
        torch.cuda.empty_cache()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_model = AutoModelForCausalLM.from_pretrained(
            gptq_model_id,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )

        # 重新加载 tokenizer
        new_tokenizer = AutoTokenizer.from_pretrained(
            gptq_model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )

        self._new_tokenizer = new_tokenizer

        print(f"✅ GPTQ 预量化模型加载完成: {gptq_model_id}")
        print(f"   参数量: {sum(p.numel() for p in quant_model.parameters()) / 1e9:.2f}B")
        return quant_model
