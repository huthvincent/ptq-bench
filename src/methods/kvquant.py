# -*- coding: utf-8 -*-
"""
KVQuant — Sensitivity-Weighted KV Cache Quantization with Outlier Isolation

核心思路:
- Key Cache: per-channel 量化 (沿 channel 维度计算统计量)
- Value Cache: per-token 量化 (沿 token 维度计算统计量)
- Dense-and-Sparse: 每个向量隔离 top-k 异常值，单独 FP16 存储
- 反量化时: 低精度 dense 部分 + FP16 sparse 异常值

修复说明 (2026-02-17):
  从 monkey-patch attention forward 迁移到自定义 CacheLayer。

参考: Coleman Hooper et al., "KVQuant: Towards 10 Million Context Length
LLM Inference with KV Cache Quantization", 2024.
"""

import torch
import torch.nn as nn
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any
from transformers.cache_utils import DynamicLayer, DynamicCache


# ──────────────────────────────────────────────
# Dense-and-Sparse 量化/反量化
# ──────────────────────────────────────────────

def _quantize_with_outliers(tensor: torch.Tensor, bits: int, quant_dim: int,
                            num_outliers: int = 1) -> dict:
    """
    Dense-and-Sparse 量化: 隔离异常值后量化。

    步骤:
    1. 沿每个向量找 top-k 绝对值最大的元素 (sparse outliers)
    2. 将 outlier 位置置零
    3. 对剩余部分做均匀量化 (dense)
    4. 分别存储 dense (INT) 和 sparse (FP16)
    """
    dtype = tensor.dtype
    qmin = 0
    qmax = (1 << bits) - 1

    outlier_vals = None
    outlier_idx = None

    if num_outliers > 0:
        abs_tensor = tensor.abs()
        _, outlier_idx = abs_tensor.topk(num_outliers, dim=-1)
        outlier_vals = torch.gather(tensor, dim=-1, index=outlier_idx)
        dense_tensor = tensor.clone()
        dense_tensor.scatter_(dim=-1, index=outlier_idx, value=0.0)
    else:
        dense_tensor = tensor

    t_min = dense_tensor.amin(dim=quant_dim, keepdim=True)
    t_max = dense_tensor.amax(dim=quant_dim, keepdim=True)
    t_range = (t_max - t_min).clamp(min=1e-8)

    scale = t_range / qmax
    zero_point = t_min

    q = ((dense_tensor - zero_point) / scale).round().clamp(qmin, qmax).to(torch.uint8)

    return {
        "q": q, "scale": scale, "zero_point": zero_point,
        "outlier_values": outlier_vals, "outlier_indices": outlier_idx,
    }


def _dequantize_with_outliers(qdata: dict, dtype: torch.dtype) -> torch.Tensor:
    """反量化: dense 部分 + sparse outlier 加回。"""
    result = qdata["q"].to(dtype) * qdata["scale"] + qdata["zero_point"]
    if qdata["outlier_values"] is not None:
        result.scatter_(dim=-1, index=qdata["outlier_indices"], src=qdata["outlier_values"])
    return result


# ──────────────────────────────────────────────
# KVQuantCacheLayer — 子类化 DynamicLayer
# ──────────────────────────────────────────────

class KVQuantCacheLayer(DynamicLayer):
    """
    KVQuant 量化的 Cache Layer。

    Key: per-channel 量化 + outlier 隔离
    Value: per-token 量化 + outlier 隔离
    最近 residual_length 个 token 保持 FP16。
    """

    def __init__(self, key_bits: int = 2, value_bits: int = 2,
                 num_outliers: int = 1, residual_length: int = 128,
                 layer_idx: int = 0):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.num_outliers = num_outliers
        self.residual_length = residual_length
        self.layer_idx = layer_idx
        self.cumulative_length = 0

        self._quantized_key: dict | None = None
        self._quantized_value: dict | None = None
        self._residual_key: torch.Tensor | None = None
        self._residual_value: torch.Tensor | None = None
        self._quantize_count = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        dtype = key_states.dtype

        # 累积完整 KV
        if self._quantized_key is not None:
            dequant_key = _dequantize_with_outliers(self._quantized_key, dtype)
            dequant_value = _dequantize_with_outliers(self._quantized_value, dtype)
            parts_key = [dequant_key]
            parts_value = [dequant_value]
            if self._residual_key is not None and self._residual_key.numel() > 0:
                parts_key.append(self._residual_key)
                parts_value.append(self._residual_value)
            parts_key.append(key_states)
            parts_value.append(value_states)
            full_key = torch.cat(parts_key, dim=-2)
            full_value = torch.cat(parts_value, dim=-2)
        else:
            if self.keys.numel() > 0:
                full_key = torch.cat([self.keys, key_states], dim=-2)
                full_value = torch.cat([self.values, value_states], dim=-2)
            else:
                full_key = key_states
                full_value = value_states

        seq_len = full_key.shape[-2]

        if seq_len <= self.residual_length:
            self.keys = full_key
            self.values = full_value
            self._quantized_key = None
            self._quantized_value = None
            self._residual_key = None
            self._residual_value = None
            return full_key, full_value

        # 分离历史 + 残差
        split_point = seq_len - self.residual_length
        hist_key = full_key[:, :, :split_point, :].contiguous()
        hist_value = full_value[:, :, :split_point, :].contiguous()
        self._residual_key = full_key[:, :, split_point:, :].contiguous()
        self._residual_value = full_value[:, :, split_point:, :].contiguous()

        # 量化 (Key per-channel dim=2, Value per-token dim=3) + outlier 隔离
        self._quantized_key = _quantize_with_outliers(
            hist_key, self.key_bits, quant_dim=2, num_outliers=self.num_outliers
        )
        self._quantized_value = _quantize_with_outliers(
            hist_value, self.value_bits, quant_dim=3, num_outliers=self.num_outliers
        )
        self._quantize_count += 1

        self.keys = torch.tensor([], dtype=dtype, device=key_states.device)
        self.values = torch.tensor([], dtype=dtype, device=key_states.device)

        # 重构返回
        dequant_key = _dequantize_with_outliers(self._quantized_key, dtype)
        dequant_value = _dequantize_with_outliers(self._quantized_value, dtype)
        return_key = torch.cat([dequant_key, self._residual_key], dim=-2)
        return_value = torch.cat([dequant_value, self._residual_value], dim=-2)

        return return_key, return_value

    def get_seq_length(self) -> int:
        return self.cumulative_length


class KVQuantQuantizedCache(DynamicCache):
    """用 KVQuantCacheLayer 替代 DynamicLayer 的 DynamicCache。"""

    def __init__(self, key_bits: int = 2, value_bits: int = 2,
                 num_outliers: int = 1, residual_length: int = 128,
                 num_layers: int = 32, **kwargs):
        layers = [
            KVQuantCacheLayer(
                key_bits=key_bits,
                value_bits=value_bits,
                num_outliers=num_outliers,
                residual_length=residual_length,
                layer_idx=i,
            )
            for i in range(num_layers)
        ]
        from transformers.cache_utils import Cache
        Cache.__init__(self, layers=layers)


def _inject_kvquant_cache(model: nn.Module, kvq_config: dict):
    """Monkey-patch model.generate() 使其使用 KVQuantQuantizedCache。"""
    key_bits = kvq_config["key_bits"]
    value_bits = kvq_config["value_bits"]
    num_outliers = kvq_config["num_outliers"]
    residual_length = kvq_config["residual_length"]
    num_layers = model.config.num_hidden_layers

    original_generate = model.generate

    def patched_generate(*args, **kwargs):
        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            cache = KVQuantQuantizedCache(
                key_bits=key_bits,
                value_bits=value_bits,
                num_outliers=num_outliers,
                residual_length=residual_length,
                num_layers=num_layers,
            )
            kwargs["past_key_values"] = cache

        result = original_generate(*args, **kwargs)

        if not hasattr(model, "_kvquant_stats_printed"):
            model._kvquant_stats_printed = True
            if isinstance(kwargs.get("past_key_values"), KVQuantQuantizedCache):
                layer = kwargs["past_key_values"].layers[0]
                if isinstance(layer, KVQuantCacheLayer) and layer._quantize_count > 0:
                    print(f"  📊 KVQuant 量化确认: layer 0 量化了 {layer._quantize_count} 次, "
                          f"累计 {layer.cumulative_length} tokens")
                else:
                    print("  ⚠️ KVQuant: 量化未触发")

        return result

    model.generate = patched_generate
    print(f"  ✅ KVQuant Cache 注入完成: {num_layers} 层, "
          f"INT{key_bits}/INT{value_bits}, outliers={num_outliers}, residual={residual_length}")


@register("kvquant")
class KVQuantMethod(BaseQuantMethod):
    """KVQuant — Sensitivity-Weighted KV Cache Quantization with Outlier Isolation."""

    supported_tracks = ["C"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        kv_config = self.config.get("kv", {})
        key_bits = kv_config.get("key_bits", 2)
        value_bits = kv_config.get("value_bits", 2)
        num_outliers = kv_config.get("num_outliers", 1)
        residual_length = kv_config.get("residual_length", 128)

        print(f"📋 KVQuant: key_bits={key_bits}, value_bits={value_bits}")
        print(f"📋 KVQuant: num_outliers={num_outliers}, residual_length={residual_length}")
        print(f"📋 KVQuant: per-channel Key + per-token Value + Dense-and-Sparse outlier 隔离")

        kvq_config = {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "num_outliers": num_outliers,
            "residual_length": residual_length,
        }

        _inject_kvquant_cache(model, kvq_config)
        return model
