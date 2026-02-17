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

修复说明 (2026-02-17):
  Transformers v5.x 的 Attention 不再通过 outputs[2] 返回 past_key_values，
  而是通过 DynamicCache.layers[i].update() 就地管理 KV Cache。
  因此将量化逻辑从 monkey-patch attention forward 迁移到自定义 CacheLayer。

参考: Zirui Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization
for KV Cache", ICML 2024.
"""

import torch
import torch.nn as nn
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any
from transformers.cache_utils import DynamicLayer, DynamicCache


# ──────────────────────────────────────────────
# 量化/反量化工具函数
# ──────────────────────────────────────────────

def _asymmetric_quantize(tensor: torch.Tensor, bits: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    沿指定维度做非对称均匀量化。

    公式: q = round((x - zero_point) / scale), scale = (max - min) / (2^bits - 1)
    """
    qmin = 0
    qmax = (1 << bits) - 1

    t_min = tensor.amin(dim=dim, keepdim=True)
    t_max = tensor.amax(dim=dim, keepdim=True)
    t_range = (t_max - t_min).clamp(min=1e-8)

    scale = t_range / qmax
    zero_point = t_min

    q = ((tensor - zero_point) / scale).round().clamp(qmin, qmax).to(torch.uint8)
    return q, scale, zero_point


def _asymmetric_dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                           dtype: torch.dtype) -> torch.Tensor:
    """反量化: x_hat = q * scale + zero_point"""
    return q.to(dtype) * scale + zero_point


# ──────────────────────────────────────────────
# KiviCacheLayer — 子类化 DynamicLayer
# ──────────────────────────────────────────────

class KiviCacheLayer(DynamicLayer):
    """
    KIVI 量化的 Cache Layer。

    继承 DynamicLayer，覆写 update() 来拦截 KV states 并量化。
    Key: per-channel 量化 (沿 head_dim 维度分组)
    Value: per-token 量化 (沿 seq_len 维度分组)
    最近 residual_length 个 token 保持 FP16。
    """

    def __init__(self, key_bits: int = 2, value_bits: int = 2,
                 residual_length: int = 128, layer_idx: int = 0):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_length = residual_length
        self.layer_idx = layer_idx
        self.cumulative_length = 0

        # 量化存储
        self._quantized_key: tuple | None = None  # (q, scale, zero_point)
        self._quantized_value: tuple | None = None
        self._residual_key: torch.Tensor | None = None
        self._residual_value: torch.Tensor | None = None

        # debug 统计
        self._quantize_count = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        拦截 KV 更新，量化历史部分，保留最近 residual_length 为 FP16。

        流程:
        1. 用 torch.cat 累积新 KV 到完整序列
        2. 如果总长度 > residual_length → 分离为历史+残差
        3. 历史部分量化 (Key per-channel, Value per-token)
        4. 返回 dequant(历史) + 残差
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        dtype = key_states.dtype

        # Step 1: 累积完整 KV
        if self._quantized_key is not None:
            # 已有量化历史 → 拼接: dequant(历史) + 残差 + 新token
            dequant_key = _asymmetric_dequantize(*self._quantized_key, dtype)
            dequant_value = _asymmetric_dequantize(*self._quantized_value, dtype)
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
            # 首次或仅有残差
            if self.keys.numel() > 0:
                full_key = torch.cat([self.keys, key_states], dim=-2)
                full_value = torch.cat([self.values, value_states], dim=-2)
            else:
                full_key = key_states
                full_value = value_states

        seq_len = full_key.shape[-2]

        # Step 2: 决定是否量化
        if seq_len <= self.residual_length:
            # 序列太短，全部保持 FP16
            self.keys = full_key
            self.values = full_value
            self._quantized_key = None
            self._quantized_value = None
            self._residual_key = None
            self._residual_value = None
            return full_key, full_value

        # Step 3: 分离历史 + 残差
        split_point = seq_len - self.residual_length
        hist_key = full_key[:, :, :split_point, :].contiguous()
        hist_value = full_value[:, :, :split_point, :].contiguous()
        self._residual_key = full_key[:, :, split_point:, :].contiguous()
        self._residual_value = full_value[:, :, split_point:, :].contiguous()

        # Step 4: 量化历史部分
        # Key: per-channel (沿 seq 维度 dim=2 计算统计量，每个 channel 独立)
        self._quantized_key = _asymmetric_quantize(hist_key, self.key_bits, dim=2)
        # Value: per-token (沿 head_dim 维度 dim=3 计算统计量，每个 token 独立)
        self._quantized_value = _asymmetric_quantize(hist_value, self.value_bits, dim=3)

        self._quantize_count += 1

        # 清空 self.keys/values (历史已量化)
        self.keys = torch.tensor([], dtype=dtype, device=key_states.device)
        self.values = torch.tensor([], dtype=dtype, device=key_states.device)

        # Step 5: 重构返回
        dequant_key = _asymmetric_dequantize(*self._quantized_key, dtype)
        dequant_value = _asymmetric_dequantize(*self._quantized_value, dtype)
        return_key = torch.cat([dequant_key, self._residual_key], dim=-2)
        return_value = torch.cat([dequant_value, self._residual_value], dim=-2)

        return return_key, return_value

    def get_seq_length(self) -> int:
        return self.cumulative_length


class KiviQuantizedCache(DynamicCache):
    """
    用 KiviCacheLayer 替代 DynamicLayer 的 DynamicCache。
    """

    def __init__(self, key_bits: int = 2, value_bits: int = 2,
                 residual_length: int = 128, num_layers: int = 32,
                 **kwargs):
        # 构造 KiviCacheLayer 列表
        layers = [
            KiviCacheLayer(
                key_bits=key_bits,
                value_bits=value_bits,
                residual_length=residual_length,
                layer_idx=i,
            )
            for i in range(num_layers)
        ]
        # 用 Cache 基类初始化 (跳过 DynamicCache.__init__ 的 config 逻辑)
        from transformers.cache_utils import Cache
        Cache.__init__(self, layers=layers)

    def get_quantize_stats(self) -> dict:
        """返回量化统计信息。"""
        stats = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, KiviCacheLayer) and layer._quantize_count > 0:
                stats[i] = {
                    "quantize_count": layer._quantize_count,
                    "cumulative_length": layer.cumulative_length,
                }
        return stats


# ──────────────────────────────────────────────
# 注入机制: hook model.generate 传入自定义 cache
# ──────────────────────────────────────────────

def _inject_kivi_cache(model: nn.Module, kivi_config: dict):
    """
    Monkey-patch model.generate() 使其使用 KiviQuantizedCache。

    在 generate 调用时，如果没有显式传入 past_key_values，
    则自动创建 KiviQuantizedCache 注入。
    """
    key_bits = kivi_config["key_bits"]
    value_bits = kivi_config["value_bits"]
    residual_length = kivi_config["residual_length"]

    # 获取层数
    num_layers = model.config.num_hidden_layers

    original_generate = model.generate

    def patched_generate(*args, **kwargs):
        # 如果没有显式传入 cache，注入 KiviQuantizedCache
        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            cache = KiviQuantizedCache(
                key_bits=key_bits,
                value_bits=value_bits,
                residual_length=residual_length,
                num_layers=num_layers,
            )
            kwargs["past_key_values"] = cache

        result = original_generate(*args, **kwargs)

        # 打印量化统计 (首次调用时)
        if hasattr(model, "_kivi_stats_printed"):
            return result
        model._kivi_stats_printed = True

        if "past_key_values" in kwargs and isinstance(kwargs["past_key_values"], KiviQuantizedCache):
            stats = kwargs["past_key_values"].get_quantize_stats()
            if stats:
                sample_layer = next(iter(stats.values()))
                print(f"  📊 KIVI 量化确认: layer 0 量化了 {sample_layer['quantize_count']} 次, "
                      f"累计 {sample_layer['cumulative_length']} tokens")
            else:
                print("  ⚠️ KIVI: 量化未触发 (序列可能太短)")

        return result

    model.generate = patched_generate

    # 同时 hook model.forward / model.__call__ 以支持 PPL 评测 (非 generate 场景)
    # PPL 评测直接调用 model(input_ids)，不经过 generate
    # 但 PPL 评测每个窗口独立 forward，KV Cache 不跨窗口，所以量化也不需要
    # 这里我们只需要确保 generate 场景下量化生效

    print(f"  ✅ KIVI Cache 注入完成: {num_layers} 层, "
          f"INT{key_bits}/INT{value_bits}, residual={residual_length}")


# ──────────────────────────────────────────────
# KiviMethod (注册到 registry)
# ──────────────────────────────────────────────

@register("kivi")
class KiviMethod(BaseQuantMethod):
    """
    KIVI — Tuning-free Asymmetric 2-bit KV Cache Quantization.

    Key per-channel + Value per-token 非对称量化。
    通过自定义 DynamicCache 实现，兼容 Transformers v5.x。
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

        _inject_kivi_cache(model, kivi_config)

        return model
