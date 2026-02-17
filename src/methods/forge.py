# -*- coding: utf-8 -*-
"""
FORGE — 动态秩免训练潜空间注意力 KV Cache 压缩

Fast On-chip Reconstruction of Generative Embeddings (FORGE)

核心思路:
- 将 KV Cache 按 chunk_size 分块，对每块做 SVD 分解
- 根据奇异值能量谱动态决定每块保留的秩 (rank)
- 推理时用 U @ diag(S) @ V^T 重构 KV，用完即丢
- 用闲置算力换显存带宽，纯后训练 (Post-Training) 方案
"""

import torch
import torch.nn as nn
from src.registry import register
from src.methods.base import BaseQuantMethod
from typing import Any


class ForgeKVCache:
    """
    FORGE KV Cache 管理器。

    将 KV 按 chunk 分块并用 truncated SVD 压缩存储。
    每个 chunk 只保存 (U, S, V) 三元组，秩由信息丰富度动态决定。
    """

    def __init__(self, chunk_size: int = 64, energy_threshold: float = 0.95,
                 min_rank: int = 2, max_rank: int = 32):
        """
        初始化 FORGE KV Cache。

        参数:
            chunk_size: 每个分块的 token 数
            energy_threshold: SVD 能量保留阈值 (0~1)
            min_rank: 每个 chunk 最少保留的秩
            max_rank: 每个 chunk 最多保留的秩
        """
        self.chunk_size = chunk_size
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank
        self.max_rank = max_rank

        # 已压缩的 chunks: list of (U, S, V) for key 和 value
        self.compressed_key_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.compressed_value_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # 尚未凑满一个 chunk 的残余 KV (原始格式)
        self.residual_key: torch.Tensor | None = None
        self.residual_value: torch.Tensor | None = None

        # 统计信息
        self.total_tokens_compressed = 0
        self.total_ranks_used: list[int] = []

    def update(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        接收新的 KV 并更新缓存。

        将新 KV 与残余拼接，凑满 chunk 的部分压缩存储，
        剩余的继续保留为残余。返回重构后的完整 KV 供 Attention 使用。

        参数:
            key: 新的 Key tensor, 形状 [B, H, S_new, D]
            value: 新的 Value tensor, 形状 [B, H, S_new, D]

        返回:
            tuple: (full_key, full_value) 重构后的完整 KV
        """
        # 拼接残余
        if self.residual_key is not None:
            key = torch.cat([self.residual_key, key], dim=2)
            value = torch.cat([self.residual_value, value], dim=2)

        seq_len = key.size(2)

        # 将凑满 chunk 的部分压缩
        n_full_chunks = seq_len // self.chunk_size
        compressed_len = n_full_chunks * self.chunk_size

        if n_full_chunks > 0:
            for i in range(n_full_chunks):
                start = i * self.chunk_size
                end = start + self.chunk_size

                k_chunk = key[:, :, start:end, :]   # [B, H, chunk_size, D]
                v_chunk = value[:, :, start:end, :]

                k_compressed = self._svd_compress(k_chunk)
                v_compressed = self._svd_compress(v_chunk)

                self.compressed_key_chunks.append(k_compressed)
                self.compressed_value_chunks.append(v_compressed)
                self.total_tokens_compressed += self.chunk_size

        # 保存残余
        if compressed_len < seq_len:
            self.residual_key = key[:, :, compressed_len:, :].contiguous()
            self.residual_value = value[:, :, compressed_len:, :].contiguous()
        else:
            self.residual_key = None
            self.residual_value = None

        # 重构完整 KV 供本次 Attention 使用
        full_key = self._reconstruct_all_keys()
        full_value = self._reconstruct_all_values()

        return full_key, full_value

    def _svd_compress(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对一个 chunk 做 truncated SVD 压缩。

        参数:
            tensor: 形状 [B, H, chunk_size, D]

        返回:
            tuple: (U, S, V) 截断后的三元组
                U: [B, H, chunk_size, r]
                S: [B, H, r]
                V: [B, H, r, D]
        """
        B, H, T, D = tensor.shape

        # SVD: tensor = U @ diag(S) @ V^T
        # torch.linalg.svd 返回 U: [B,H,T,K], S: [B,H,K], Vh: [B,H,K,D], K=min(T,D)
        U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)

        # 动态计算秩
        rank = _compute_dynamic_rank(S, self.energy_threshold, self.min_rank, self.max_rank)

        # 取所有 head 的最大秩 (简化实现，保证形状一致)
        r = rank.max().item()
        self.total_ranks_used.append(r)

        # 截断
        U_trunc = U[:, :, :, :r].to(tensor.dtype)    # [B, H, T, r]
        S_trunc = S[:, :, :r].to(tensor.dtype)        # [B, H, r]
        Vh_trunc = Vh[:, :, :r, :].to(tensor.dtype)   # [B, H, r, D]

        return (U_trunc, S_trunc, Vh_trunc)

    def _reconstruct_chunk(self, compressed: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        从 (U, S, V) 三元组重构一个 chunk。

        参数:
            compressed: (U, S, Vh) 三元组

        返回:
            torch.Tensor: 重构的 tensor, 形状 [B, H, chunk_size, D]
        """
        U, S, Vh = compressed
        # reconstructed = U @ diag(S) @ Vh
        # U: [B, H, T, r], S: [B, H, r], Vh: [B, H, r, D]
        return torch.matmul(U * S.unsqueeze(-2), Vh)

    def _reconstruct_all_keys(self) -> torch.Tensor:
        """重构所有 Key (压缩 chunks + 残余)。"""
        parts = []
        for compressed in self.compressed_key_chunks:
            parts.append(self._reconstruct_chunk(compressed))
        if self.residual_key is not None:
            parts.append(self.residual_key)
        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=2)

    def _reconstruct_all_values(self) -> torch.Tensor:
        """重构所有 Value (压缩 chunks + 残余)。"""
        parts = []
        for compressed in self.compressed_value_chunks:
            parts.append(self._reconstruct_chunk(compressed))
        if self.residual_value is not None:
            parts.append(self.residual_value)
        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=2)

    def get_seq_len(self) -> int:
        """返回当前缓存的总 token 数。"""
        n = len(self.compressed_key_chunks) * self.chunk_size
        if self.residual_key is not None:
            n += self.residual_key.size(2)
        return n

    def get_memory_stats(self) -> dict:
        """
        返回压缩统计信息。

        返回:
            dict: 包含平均秩、压缩比等信息
        """
        if not self.total_ranks_used:
            return {"avg_rank": 0, "compression_ratio": 1.0, "num_chunks": 0}

        avg_rank = sum(self.total_ranks_used) / len(self.total_ranks_used)
        # 原始存储: chunk_size * D per chunk
        # 压缩存储: chunk_size * r + r + r * D ≈ r * (chunk_size + D)
        # 假设 D ≈ head_dim (通常 128)
        # 压缩比 ≈ (chunk_size * D) / (r * (chunk_size + D))
        D_est = 128  # 典型 head_dim
        original = self.chunk_size * D_est
        compressed = avg_rank * (self.chunk_size + D_est)
        ratio = original / compressed if compressed > 0 else 1.0

        return {
            "avg_rank": round(avg_rank, 1),
            "min_rank_used": min(self.total_ranks_used),
            "max_rank_used": max(self.total_ranks_used),
            "num_chunks": len(self.total_ranks_used) // 2,  # key 和 value 各一半
            "compression_ratio": round(ratio, 2),
            "total_tokens_compressed": self.total_tokens_compressed,
        }


def _compute_dynamic_rank(singular_values: torch.Tensor, energy_threshold: float,
                          min_rank: int, max_rank: int) -> torch.Tensor:
    """
    根据奇异值能量谱动态计算最优秩。

    通过累积能量比判断保留多少主成分:
    retained_energy = cumsum(sigma^2) / sum(sigma^2)
    找到第一个 >= energy_threshold 的位置即为秩。

    参数:
        singular_values: 奇异值, 形状 [B, H, K]
        energy_threshold: 能量保留阈值 (0~1)
        min_rank: 最小秩
        max_rank: 最大秩

    返回:
        torch.Tensor: 每个 (batch, head) 的最优秩, 形状 [B, H]
    """
    # 计算能量: sigma^2
    energy = singular_values ** 2

    # 累积能量占比
    cumulative_energy = torch.cumsum(energy, dim=-1)
    total_energy = cumulative_energy[..., -1:]  # [B, H, 1]
    energy_ratio = cumulative_energy / total_energy.clamp(min=1e-10)

    # 找到第一个 >= threshold 的位置
    # argmax 在 bool tensor 上返回第一个 True 的位置
    rank = (energy_ratio >= energy_threshold).long().argmax(dim=-1) + 1  # [B, H]

    # 边界保护
    rank = rank.clamp(min=min_rank, max=max_rank)

    return rank


def _patch_attention_layers(model: nn.Module, forge_config: dict) -> list[ForgeKVCache]:
    """
    Monkey-patch 模型的 Attention 层，植入 FORGE KV Cache。

    遍历模型找到所有 Attention 层，替换其 forward 方法，
    使其使用 ForgeKVCache 管理 KV 而非原始的 past_key_values。

    兼容 LlamaAttention / Qwen2Attention 等标准 HuggingFace 架构。

    参数:
        model: HuggingFace 模型
        forge_config: FORGE 配置字典 (chunk_size, energy_threshold, etc.)

    返回:
        list[ForgeKVCache]: 所有 Attention 层的 FORGE cache 实例列表
    """
    chunk_size = forge_config.get("chunk_size", 64)
    energy_threshold = forge_config.get("energy_threshold", 0.95)
    min_rank = forge_config.get("min_rank", 2)
    max_rank = forge_config.get("max_rank", 32)

    caches = []
    patched_count = 0

    for name, module in model.named_modules():
        # 匹配常见 Attention 层名称
        module_type = type(module).__name__
        if not any(kw in module_type for kw in ("Attention",)):
            continue

        # 检查是否有 k_proj / v_proj (标准 HF Attention 特征)
        has_kv_proj = hasattr(module, "k_proj") and hasattr(module, "v_proj")
        if not has_kv_proj:
            continue

        # 创建该层的 FORGE cache
        cache = ForgeKVCache(
            chunk_size=chunk_size,
            energy_threshold=energy_threshold,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        caches.append(cache)

        # 保存原始 forward
        original_forward = module.forward

        # 创建 patched forward — 通过闭包捕获 cache 和 original_forward
        def make_patched_forward(orig_fwd, forge_cache, attn_module):
            """
            构建 patched forward 函数。

            采用"先正常跑 → 再压缩 KV"的策略:
            1. 调用原始 forward 获得 attention 输出和 KV
            2. 将 KV 送入 ForgeKVCache 压缩
            3. 用重构的 KV 替换 past_key_values
            """
            def patched_forward(*args, **kwargs):
                # 调用原始 forward
                outputs = orig_fwd(*args, **kwargs)

                # outputs 通常是 (attn_output, attn_weights, past_key_values)
                # 或者 (attn_output, None, past_key_values)
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    attn_output = outputs[0]
                    attn_weights = outputs[1]
                    past_kv = outputs[2]

                    # past_kv 通常是 (key, value) tuple
                    if isinstance(past_kv, tuple) and len(past_kv) == 2:
                        key_states, value_states = past_kv

                        # 清空旧 cache (因为原始 forward 已经累积了)
                        # 我们每次从头压缩完整 KV
                        forge_cache.compressed_key_chunks.clear()
                        forge_cache.compressed_value_chunks.clear()
                        forge_cache.residual_key = None
                        forge_cache.residual_value = None
                        forge_cache.total_tokens_compressed = 0

                        # 对完整 KV 做 FORGE 压缩
                        compressed_key, compressed_value = forge_cache.update(
                            key_states, value_states
                        )

                        # 用压缩-重构后的 KV 替换
                        outputs = (attn_output, attn_weights, (compressed_key, compressed_value))

                return outputs

            return patched_forward

        module.forward = make_patched_forward(original_forward, cache, module)
        patched_count += 1
        print(f"  ⚡ Patched: {name} ({module_type})")

    print(f"📋 FORGE: 共 patch 了 {patched_count} 个 Attention 层")
    return caches


@register("forge")
class ForgeMethod(BaseQuantMethod):
    """
    FORGE 量化方法 — 动态秩免训练 KV Cache 压缩。

    通过 monkey-patch Attention 层，植入基于 SVD 的动态秩
    KV Cache 压缩机制。不需要校准数据，纯后训练方案。
    """

    supported_tracks = ["C"]

    def quantize(self, model: Any, tokenizer: Any, calib_data: Any | None = None) -> Any:
        """
        执行 FORGE "量化"（实际是安装 KV Cache 压缩器）。

        步骤:
        1. 从 config 读取 FORGE 参数
        2. Monkey-patch 所有 Attention 层的 forward
        3. 返回 patched 后的模型

        参数:
            model: 原始 FP16/BF16 模型
            tokenizer: tokenizer（未使用）
            calib_data: 校准数据（FORGE 不使用）

        返回:
            Any: 安装了 FORGE KV Cache 的模型
        """
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

        # Patch Attention 层
        caches = _patch_attention_layers(model, forge_config)

        if not caches:
            print("⚠️  未找到可 patch 的 Attention 层，FORGE 未生效")
        else:
            print(f"✅ FORGE 安装完成: {len(caches)} 个 Attention 层已启用动态秩 KV 压缩")

        # 将 caches 挂在模型上，方便后续获取统计信息
        model._forge_caches = caches

        return model
