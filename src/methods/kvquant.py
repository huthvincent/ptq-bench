# -*- coding: utf-8 -*-
"""
KVQuant — Sensitivity-Weighted KV Cache Quantization with Outlier Isolation

核心思路:
- Key Cache: per-channel 量化 (沿 channel 维度计算统计量)
- Value Cache: per-token 量化 (沿 token 维度计算统计量)
- Dense-and-Sparse: 每个向量隔离 top-k 异常值，单独 FP16 存储
- 反量化时: 低精度 dense 部分 + FP16 sparse 异常值
- 需要校准数据计算最优 scale (本实现用 min/max 均匀量化简化)

参考: Coleman Hooper et al., "KVQuant: Towards 10 Million Context Length
LLM Inference with KV Cache Quantization", 2024.
"""

import torch
import torch.nn as nn
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


def _quantize_with_outliers(tensor: torch.Tensor, bits: int, quant_dim: int,
                            num_outliers: int = 1) -> dict:
    """
    Dense-and-Sparse 量化: 隔离异常值后量化。

    步骤:
    1. 沿每个向量找 top-k 绝对值最大的元素 (sparse outliers)
    2. 将 outlier 位置置零
    3. 对剩余部分做均匀量化 (dense)
    4. 分别存储 dense (INT) 和 sparse (FP16)

    参数:
        tensor: [B, H, T, D]
        bits: 量化位宽
        quant_dim: 量化维度 (Key: dim=2 per-channel, Value: dim=3 per-token)
        num_outliers: 每个向量隔离的异常值个数

    返回:
        dict: {q, scale, zero_point, outlier_values, outlier_indices}
    """
    dtype = tensor.dtype
    B, H, T, D = tensor.shape
    qmin = 0
    qmax = (1 << bits) - 1

    if num_outliers > 0:
        # 找异常值: 沿 "非量化" 维度的每个向量中找 top-k
        # 对于 per-channel Key (quant_dim=2): 每个 [B,H,:,d] 向量找 outlier → 太复杂
        # 简化: 沿最后一个维度 (D) 找每个 token 的 outlier
        abs_tensor = tensor.abs()
        # topk 沿 D 维度
        _, outlier_idx = abs_tensor.topk(num_outliers, dim=-1)  # [B, H, T, k]

        # 提取 outlier 值
        outlier_vals = torch.gather(tensor, dim=-1, index=outlier_idx)  # [B, H, T, k]

        # 创建 mask 并置零 outlier
        dense_tensor = tensor.clone()
        dense_tensor.scatter_(dim=-1, index=outlier_idx, value=0.0)
    else:
        dense_tensor = tensor
        outlier_vals = None
        outlier_idx = None

    # 均匀量化 dense 部分
    t_min = dense_tensor.amin(dim=quant_dim, keepdim=True)
    t_max = dense_tensor.amax(dim=quant_dim, keepdim=True)
    t_range = (t_max - t_min).clamp(min=1e-8)

    scale = t_range / qmax
    zero_point = t_min

    q = ((dense_tensor - zero_point) / scale).round().clamp(qmin, qmax).to(torch.uint8)

    return {
        "q": q,
        "scale": scale,
        "zero_point": zero_point,
        "outlier_values": outlier_vals,
        "outlier_indices": outlier_idx,
    }


def _dequantize_with_outliers(qdata: dict, dtype: torch.dtype) -> torch.Tensor:
    """
    反量化: dense 部分 + sparse outlier 加回。
    """
    q = qdata["q"]
    scale = qdata["scale"]
    zero_point = qdata["zero_point"]

    # dense 反量化
    result = q.to(dtype) * scale + zero_point

    # 加回 outliers
    if qdata["outlier_values"] is not None:
        result.scatter_(dim=-1, index=qdata["outlier_indices"], src=qdata["outlier_values"])

    return result


class KVQuantCache:
    """
    KVQuant KV Cache 管理器。

    Key: per-channel 量化 + outlier 隔离
    Value: per-token 量化 + outlier 隔离
    最近 residual_length 个 token 保持 FP16。
    """

    def __init__(self, key_bits: int = 2, value_bits: int = 2,
                 num_outliers: int = 1, residual_length: int = 128):
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.num_outliers = num_outliers
        self.residual_length = residual_length

        self.quantized_key: dict | None = None
        self.quantized_value: dict | None = None
        self.residual_key: torch.Tensor | None = None
        self.residual_value: torch.Tensor | None = None
        self.total_tokens_quantized = 0

    def update(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        接收完整 KV，量化历史部分，保留最近 residual_length 个 token 为 FP16。
        """
        seq_len = key.size(2)
        dtype = key.dtype

        self.quantized_key = None
        self.quantized_value = None
        self.total_tokens_quantized = 0

        if seq_len <= self.residual_length:
            self.residual_key = key
            self.residual_value = value
            return key, value

        split_point = seq_len - self.residual_length
        hist_key = key[:, :, :split_point, :]
        hist_value = value[:, :, :split_point, :]
        self.residual_key = key[:, :, split_point:, :].contiguous()
        self.residual_value = value[:, :, split_point:, :].contiguous()

        # Key: per-channel (quant_dim=2, 沿 seq 维度计算统计量 → 每个 channel 独立)
        self.quantized_key = _quantize_with_outliers(
            hist_key, self.key_bits, quant_dim=2, num_outliers=self.num_outliers
        )

        # Value: per-token (quant_dim=3, 沿 head_dim 维度计算统计量 → 每个 token 独立)
        self.quantized_value = _quantize_with_outliers(
            hist_value, self.value_bits, quant_dim=3, num_outliers=self.num_outliers
        )

        self.total_tokens_quantized = split_point

        # 重构
        full_key = self._reconstruct(is_key=True, dtype=dtype)
        full_value = self._reconstruct(is_key=False, dtype=dtype)

        return full_key, full_value

    def _reconstruct(self, is_key: bool, dtype: torch.dtype) -> torch.Tensor:
        qdata = self.quantized_key if is_key else self.quantized_value
        residual = self.residual_key if is_key else self.residual_value

        parts = []
        if qdata is not None:
            parts.append(_dequantize_with_outliers(qdata, dtype))
        if residual is not None:
            parts.append(residual)

        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=2)


def _patch_attention_layers_kvquant(model: nn.Module, kvq_config: dict) -> list[KVQuantCache]:
    """
    Monkey-patch Attention 层，植入 KVQuant Cache。
    """
    key_bits = kvq_config.get("key_bits", 2)
    value_bits = kvq_config.get("value_bits", 2)
    num_outliers = kvq_config.get("num_outliers", 1)
    residual_length = kvq_config.get("residual_length", 128)

    caches = []
    patched_count = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Attention" not in module_type:
            continue
        if not (hasattr(module, "k_proj") and hasattr(module, "v_proj")):
            continue

        cache = KVQuantCache(
            key_bits=key_bits,
            value_bits=value_bits,
            num_outliers=num_outliers,
            residual_length=residual_length,
        )
        caches.append(cache)

        original_forward = module.forward

        def make_patched_forward(orig_fwd, kvq_cache):
            def patched_forward(*args, **kwargs):
                outputs = orig_fwd(*args, **kwargs)

                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    attn_output = outputs[0]
                    attn_weights = outputs[1]
                    past_kv = outputs[2]

                    if isinstance(past_kv, tuple) and len(past_kv) == 2:
                        key_states, value_states = past_kv
                        compressed_key, compressed_value = kvq_cache.update(
                            key_states, value_states
                        )
                        outputs = (attn_output, attn_weights, (compressed_key, compressed_value))

                return outputs
            return patched_forward

        module.forward = make_patched_forward(original_forward, cache)
        patched_count += 1
        print(f"  ⚡ KVQuant Patched: {name} ({module_type})")

    print(f"📋 KVQuant: 共 patch 了 {patched_count} 个 Attention 层")
    return caches


@register("kvquant")
class KVQuantMethod(BaseQuantMethod):
    """
    KVQuant — Sensitivity-Weighted KV Cache Quantization with Outlier Isolation.
    """

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

        caches = _patch_attention_layers_kvquant(model, kvq_config)

        if not caches:
            print("⚠️  未找到可 patch 的 Attention 层")
        else:
            print(f"✅ KVQuant: {len(caches)} 个 Attention 层已启用 INT{key_bits}/INT{value_bits} + outlier 隔离")

        model._kvquant_caches = caches
        return model
