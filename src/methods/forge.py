# -*- coding: utf-8 -*-
"""
FORGE — 动态秩免训练潜空间注意力 KV Cache 压缩

Fast On-chip Reconstruction of Generative Embeddings (FORGE)

核心思路:
- 将 KV Cache 按 chunk_size 分块，对每块做 SVD 分解
- 根据奇异值能量谱动态决定每块保留的秩 (rank)
- 推理时用 U @ diag(S) @ V^T 重构 KV，用完即丢
- 用闲置算力换显存带宽，纯后训练 (Post-Training) 方案

修复说明 (2026-02-17):
  从 monkey-patch attention forward 迁移到自定义 CacheLayer。
"""

import torch
import torch.nn as nn
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any
from transformers.cache_utils import DynamicLayer, DynamicCache


# ──────────────────────────────────────────────
# SVD 压缩工具函数
# ──────────────────────────────────────────────

def _compute_dynamic_rank(singular_values: torch.Tensor, energy_threshold: float,
                          min_rank: int, max_rank: int) -> torch.Tensor:
    """
    根据奇异值能量谱动态计算最优秩。

    通过累积能量比判断保留多少主成分:
    retained_energy = cumsum(sigma^2) / sum(sigma^2)
    """
    energy = singular_values ** 2
    cumulative_energy = torch.cumsum(energy, dim=-1)
    total_energy = cumulative_energy[..., -1:]
    energy_ratio = cumulative_energy / total_energy.clamp(min=1e-10)

    rank = (energy_ratio >= energy_threshold).long().argmax(dim=-1) + 1
    rank = rank.clamp(min=min_rank, max=max_rank)
    return rank


def _svd_compress(tensor: torch.Tensor, energy_threshold: float,
                  min_rank: int, max_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    对一个 chunk 做 truncated SVD 压缩。

    返回: (U_trunc, S_trunc, Vh_trunc, rank)
    """
    U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
    rank = _compute_dynamic_rank(S, energy_threshold, min_rank, max_rank)
    r = rank.max().item()

    U_trunc = U[:, :, :, :r].to(tensor.dtype)
    S_trunc = S[:, :, :r].to(tensor.dtype)
    Vh_trunc = Vh[:, :, :r, :].to(tensor.dtype)
    return U_trunc, S_trunc, Vh_trunc, r


def _svd_reconstruct(compressed: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """从 (U, S, Vh) 三元组重构一个 chunk。"""
    U, S, Vh = compressed
    return torch.matmul(U * S.unsqueeze(-2), Vh)


# ──────────────────────────────────────────────
# ForgeCacheLayer — 子类化 DynamicLayer
# ──────────────────────────────────────────────

class ForgeCacheLayer(DynamicLayer):
    """
    FORGE 压缩的 Cache Layer。

    继承 DynamicLayer，覆写 update() 来拦截 KV states 并用 SVD 压缩。
    已凑满 chunk_size 的部分用 (U, S, Vh) 存储，残余保持原始精度。
    """

    def __init__(self, chunk_size: int = 64, energy_threshold: float = 0.95,
                 min_rank: int = 2, max_rank: int = 32, layer_idx: int = 0):
        super().__init__()
        self.chunk_size = chunk_size
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.layer_idx = layer_idx
        self.cumulative_length = 0

        # 压缩存储
        self._compressed_key_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._compressed_value_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._residual_key: torch.Tensor | None = None
        self._residual_value: torch.Tensor | None = None

        # 统计
        self._compress_count = 0
        self._ranks_used: list[int] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        拦截 KV 更新，用 SVD 按 chunk 压缩。

        流程:
        1. 拼接残余 + 新 token
        2. 凑满 chunk_size 的部分做 SVD 压缩
        3. 剩余继续保留为残余
        4. 返回完整重构的 KV (压缩 chunks 重构 + 残余)
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]

        # Step 1: 拼接残余
        if self._residual_key is not None and self._residual_key.numel() > 0:
            key_combined = torch.cat([self._residual_key, key_states], dim=-2)
            value_combined = torch.cat([self._residual_value, value_states], dim=-2)
        else:
            key_combined = key_states
            value_combined = value_states

        seq_len = key_combined.shape[-2]

        # Step 2: 分块压缩
        n_full_chunks = seq_len // self.chunk_size
        compressed_len = n_full_chunks * self.chunk_size

        if n_full_chunks > 0:
            for i in range(n_full_chunks):
                start = i * self.chunk_size
                end = start + self.chunk_size

                k_chunk = key_combined[:, :, start:end, :]
                v_chunk = value_combined[:, :, start:end, :]

                ku, ks, kvh, kr = _svd_compress(k_chunk, self.energy_threshold,
                                                 self.min_rank, self.max_rank)
                vu, vs, vvh, vr = _svd_compress(v_chunk, self.energy_threshold,
                                                 self.min_rank, self.max_rank)

                self._compressed_key_chunks.append((ku, ks, kvh))
                self._compressed_value_chunks.append((vu, vs, vvh))
                self._ranks_used.extend([kr, vr])
                self._compress_count += 1

        # Step 3: 保存残余
        if compressed_len < seq_len:
            self._residual_key = key_combined[:, :, compressed_len:, :].contiguous()
            self._residual_value = value_combined[:, :, compressed_len:, :].contiguous()
        else:
            self._residual_key = None
            self._residual_value = None

        # 清空父类的 self.keys/values
        self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
        self.values = torch.tensor([], dtype=key_states.dtype, device=key_states.device)

        # Step 4: 重构完整 KV
        key_parts = [_svd_reconstruct(c) for c in self._compressed_key_chunks]
        value_parts = [_svd_reconstruct(c) for c in self._compressed_value_chunks]

        if self._residual_key is not None and self._residual_key.numel() > 0:
            key_parts.append(self._residual_key)
            value_parts.append(self._residual_value)

        if not key_parts:
            empty = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            return empty, empty

        full_key = torch.cat(key_parts, dim=-2)
        full_value = torch.cat(value_parts, dim=-2)

        return full_key, full_value

    def get_seq_length(self) -> int:
        return self.cumulative_length


class ForgeQuantizedCache(DynamicCache):
    """用 ForgeCacheLayer 替代 DynamicLayer 的 DynamicCache。"""

    def __init__(self, chunk_size: int = 64, energy_threshold: float = 0.95,
                 min_rank: int = 2, max_rank: int = 32,
                 num_layers: int = 32, **kwargs):
        layers = [
            ForgeCacheLayer(
                chunk_size=chunk_size,
                energy_threshold=energy_threshold,
                min_rank=min_rank,
                max_rank=max_rank,
                layer_idx=i,
            )
            for i in range(num_layers)
        ]
        from transformers.cache_utils import Cache
        Cache.__init__(self, layers=layers)

    def get_compress_stats(self) -> dict:
        """返回压缩统计信息。"""
        all_ranks = []
        for layer in self.layers:
            if isinstance(layer, ForgeCacheLayer):
                all_ranks.extend(layer._ranks_used)
        if not all_ranks:
            return {"avg_rank": 0, "num_chunks": 0}
        return {
            "avg_rank": round(sum(all_ranks) / len(all_ranks), 1),
            "min_rank": min(all_ranks),
            "max_rank": max(all_ranks),
            "num_chunks": len(all_ranks) // 2,
        }


# ──────────────────────────────────────────────
# 注入机制
# ──────────────────────────────────────────────

def _inject_forge_cache(model: nn.Module, forge_config: dict):
    """Monkey-patch model.generate() 使其使用 ForgeQuantizedCache。"""
    chunk_size = forge_config["chunk_size"]
    energy_threshold = forge_config["energy_threshold"]
    min_rank = forge_config["min_rank"]
    max_rank = forge_config["max_rank"]
    num_layers = model.config.num_hidden_layers

    original_generate = model.generate

    def patched_generate(*args, **kwargs):
        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            cache = ForgeQuantizedCache(
                chunk_size=chunk_size,
                energy_threshold=energy_threshold,
                min_rank=min_rank,
                max_rank=max_rank,
                num_layers=num_layers,
            )
            kwargs["past_key_values"] = cache

        result = original_generate(*args, **kwargs)

        if not hasattr(model, "_forge_stats_printed"):
            model._forge_stats_printed = True
            if isinstance(kwargs.get("past_key_values"), ForgeQuantizedCache):
                stats = kwargs["past_key_values"].get_compress_stats()
                if stats["num_chunks"] > 0:
                    print(f"  📊 FORGE 压缩确认: {stats['num_chunks']} chunks, "
                          f"avg_rank={stats['avg_rank']}, range=[{stats['min_rank']},{stats['max_rank']}]")
                else:
                    print("  ⚠️ FORGE: 压缩未触发 (序列可能太短)")

        return result

    model.generate = patched_generate
    print(f"  ✅ FORGE Cache 注入完成: {num_layers} 层, "
          f"chunk={chunk_size}, energy={energy_threshold}")


@register("forge")
class ForgeMethod(BaseQuantMethod):
    """FORGE — 动态秩免训练 KV Cache 压缩。通过自定义 DynamicCache 实现。"""

    supported_tracks = ["C"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        kv_config = self.config.get("kv", {})
        chunk_size = kv_config.get("chunk_size", 64)
        energy_threshold = kv_config.get("energy_threshold", 0.95)
        min_rank = kv_config.get("min_rank", 2)
        max_rank = kv_config.get("max_rank", 32)

        print(f"📋 FORGE: chunk_size={chunk_size}, energy_threshold={energy_threshold}")
        print(f"📋 FORGE: rank_range=[{min_rank}, {max_rank}]")
        print(f"📋 FORGE: 免校准，动态秩 SVD 压缩")

        forge_config = {
            "chunk_size": chunk_size,
            "energy_threshold": energy_threshold,
            "min_rank": min_rank,
            "max_rank": max_rank,
        }

        _inject_forge_cache(model, forge_config)
        return model
