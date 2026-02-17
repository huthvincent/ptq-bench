# -*- coding: utf-8 -*-
"""
KIVI — Tuning-free Asymmetric 2-bit KV Cache Quantization

核心思路:
- Key Cache: 按 **channel 维度** 分组量化 (per-channel)
  → 因为 Key 的异常值集中在固定 channel
- Value Cache: 按 **token 维度** 分组量化 (per-token)
  → Value 无明显 channel 异常值模式
- 未凑满 group 的残余 token 保持 FP16
- 免校准，运行时在线量化

参考: Zirui Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization
for KV Cache", ICML 2024.
"""

import torch
import torch.nn as nn
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


def _asymmetric_quantize(tensor: torch.Tensor, bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    沿指定维度做非对称均匀量化。

    公式: q = round((x - zero_point) / scale), scale = (max - min) / (2^bits - 1)

    参数:
        tensor: 待量化 tensor
        bits: 量化位宽 (2 or 4)
        dim: 量化维度 (沿哪个维度计算 min/max)

    返回:
        tuple: (quantized_int, scale, zero_point)
    """
    qmin = 0
    qmax = (1 << bits) - 1  # 2-bit: 0~3, 4-bit: 0~15

    # 沿指定维度计算 min/max
    t_min = tensor.amin(dim=dim, keepdim=True)
    t_max = tensor.amax(dim=dim, keepdim=True)

    # 防止 min == max (常量 tensor)
    t_range = (t_max - t_min).clamp(min=1e-8)

    scale = t_range / qmax
    zero_point = t_min

    # 量化
    q = ((tensor - zero_point) / scale).round().clamp(qmin, qmax).to(torch.uint8)

    return q, scale, zero_point


def _asymmetric_dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                           dtype: torch.dtype) -> torch.Tensor:
    """
    反量化: x_hat = q * scale + zero_point
    """
    return q.to(dtype) * scale + zero_point


class KiviKVCache:
    """
    KIVI KV Cache 管理器。

    Key: per-channel 量化 (沿 head_dim 维度)
    Value: per-token 量化 (沿 seq_len 维度)
    最近 residual_length 个 token 保持 FP16。
    """

    def __init__(self, key_bits: int = 2, value_bits: int = 2,
                 group_size: int = 32, residual_length: int = 128):
        """
        参数:
            key_bits: Key 量化位宽
            value_bits: Value 量化位宽
            group_size: 量化分组大小
            residual_length: 保持 FP16 的最近 token 数
        """
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.residual_length = residual_length

        # 已量化的 Key chunks: list of (q, scale, zero_point)
        self.quantized_key_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.quantized_value_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # 残余 (FP16)
        self.residual_key: torch.Tensor | None = None
        self.residual_value: torch.Tensor | None = None

        # 统计
        self.total_tokens_quantized = 0

    def update(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        接收完整 KV，量化历史部分，保留最近 residual_length 个 token 为 FP16。

        参数:
            key: [B, H, S, D]
            value: [B, H, S, D]

        返回:
            tuple: (full_key, full_value) 重构后的完整 KV
        """
        seq_len = key.size(2)
        dtype = key.dtype

        # 清空旧状态 (每次从完整 KV 重新量化)
        self.quantized_key_chunks.clear()
        self.quantized_value_chunks.clear()
        self.total_tokens_quantized = 0

        if seq_len <= self.residual_length:
            # 序列太短，全部保持 FP16
            self.residual_key = key
            self.residual_value = value
            return key, value

        # 分离: 历史部分量化，最近部分保持 FP16
        split_point = seq_len - self.residual_length
        hist_key = key[:, :, :split_point, :]
        hist_value = value[:, :, :split_point, :]
        self.residual_key = key[:, :, split_point:, :].contiguous()
        self.residual_value = value[:, :, split_point:, :].contiguous()

        # 量化历史 Key: per-channel (沿 D 维度，即 dim=-1 的分组)
        # KIVI 的 per-channel 意味着每个 channel 有独立的 scale/zp
        # 对于 [B, H, T, D]，per-channel = 对每个 d 分别量化 T 维
        # 实际操作: 沿 seq 维度 (dim=2) 计算统计量，这样每个 channel 有独立参数
        q_key, s_key, z_key = _asymmetric_quantize(hist_key, self.key_bits, dim=2)
        self.quantized_key_chunks.append((q_key, s_key, z_key))

        # 量化历史 Value: per-token (沿 T 维度)
        # per-token 意味着每个 token 有独立的 scale/zp
        # 对于 [B, H, T, D]，per-token = 沿 head_dim 维度 (dim=3) 计算统计量
        q_val, s_val, z_val = _asymmetric_quantize(hist_value, self.value_bits, dim=3)
        self.quantized_value_chunks.append((q_val, s_val, z_val))

        self.total_tokens_quantized = split_point

        # 重构
        full_key = self._reconstruct_all(is_key=True, dtype=dtype)
        full_value = self._reconstruct_all(is_key=False, dtype=dtype)

        return full_key, full_value

    def _reconstruct_all(self, is_key: bool, dtype: torch.dtype) -> torch.Tensor:
        """重构完整 KV (量化部分反量化 + 残余 FP16)。"""
        chunks = self.quantized_key_chunks if is_key else self.quantized_value_chunks
        residual = self.residual_key if is_key else self.residual_value

        parts = []
        for q, s, z in chunks:
            parts.append(_asymmetric_dequantize(q, s, z, dtype))
        if residual is not None:
            parts.append(residual)

        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=2)


def _patch_attention_layers_kivi(model: nn.Module, kivi_config: dict) -> list[KiviKVCache]:
    """
    Monkey-patch Attention 层，植入 KIVI KV Cache。

    兼容 LlamaAttention / Qwen2Attention / MistralAttention。
    """
    key_bits = kivi_config.get("key_bits", 2)
    value_bits = kivi_config.get("value_bits", 2)
    group_size = kivi_config.get("group_size", 32)
    residual_length = kivi_config.get("residual_length", 128)

    caches = []
    patched_count = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Attention" not in module_type:
            continue
        if not (hasattr(module, "k_proj") and hasattr(module, "v_proj")):
            continue

        cache = KiviKVCache(
            key_bits=key_bits,
            value_bits=value_bits,
            group_size=group_size,
            residual_length=residual_length,
        )
        caches.append(cache)

        original_forward = module.forward

        def make_patched_forward(orig_fwd, kivi_cache):
            def patched_forward(*args, **kwargs):
                outputs = orig_fwd(*args, **kwargs)

                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    attn_output = outputs[0]
                    attn_weights = outputs[1]
                    past_kv = outputs[2]

                    if isinstance(past_kv, tuple) and len(past_kv) == 2:
                        key_states, value_states = past_kv
                        compressed_key, compressed_value = kivi_cache.update(
                            key_states, value_states
                        )
                        outputs = (attn_output, attn_weights, (compressed_key, compressed_value))

                return outputs
            return patched_forward

        module.forward = make_patched_forward(original_forward, cache)
        patched_count += 1
        print(f"  ⚡ KIVI Patched: {name} ({module_type})")

    print(f"📋 KIVI: 共 patch 了 {patched_count} 个 Attention 层")
    return caches


@register("kivi")
class KiviMethod(BaseQuantMethod):
    """
    KIVI — Tuning-free Asymmetric 2-bit KV Cache Quantization.

    Key per-channel + Value per-token 非对称量化。
    """

    supported_tracks = ["C"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        kv_config = self.config.get("kv", {})
        key_bits = kv_config.get("key_bits", 2)
        value_bits = kv_config.get("value_bits", 2)
        group_size = kv_config.get("group_size", 32)
        residual_length = kv_config.get("residual_length", 128)

        print(f"📋 KIVI: key_bits={key_bits}, value_bits={value_bits}")
        print(f"📋 KIVI: group_size={group_size}, residual_length={residual_length}")
        print(f"📋 KIVI: 免校准，非对称 per-channel Key + per-token Value 量化")

        kivi_config = {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "group_size": group_size,
            "residual_length": residual_length,
        }

        caches = _patch_attention_layers_kivi(model, kivi_config)

        if not caches:
            print("⚠️  未找到可 patch 的 Attention 层")
        else:
            print(f"✅ KIVI: {len(caches)} 个 Attention 层已启用 INT{key_bits}/INT{value_bits} KV 量化")

        model._kivi_caches = caches
        return model
