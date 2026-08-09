# Copyright © 2026 Apple Inc. / mlx-community
#
# DeepSeek-V4 (Pro / Flash) for mlx-lm.
# Architecture: Multi-head Latent Attention (num_kv_heads=1) + grouped low-rank output,
# sliding-window + compressed KV + indexer topk (sparse attention), hash-routed MoE
# with sqrtsoftplus scoring, Manifold-constrained Hyper-Connections (mHC) replacing
# residuals. Weights are native FP8 (e4m3) with 128x128 block scaling (ue8m0).
#
# Reference: deepseek-ai/DeepSeek-V4 (Apr 2026). mHC: arXiv:2512.24880.

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .hyper_connection import HyperConnection, HyperHead
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1

    # Attention (MLA-style with single shared KV head)
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    attention_bias: bool = False
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)

    # Compressor / Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_rope_theta: float = 160000.0

    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Hyper-Connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # Conventional MTP. The official 0731 checkpoint instead uses DSpark.
    num_nextn_predict_layers: int = 1

    # DSpark configuration in the official DeepSeek-V4-Flash-0731 snapshot.
    # This target-only port records these fields so it can reject the incompatible
    # conventional MTP path without discarding the target model.
    dspark_block_size: int = 0
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_noise_token_id: int = 0
    dspark_markov_rank: int = 256
    n_mtp_layers: int = 0

    # RoPE / YaRN
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rms_norm_eps: float = 1e-6

    # Quantization (FP8 block)
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        # Auto-fill compress_ratios with V4 defaults if not specified, and
        # validate length / values. Adapted from @eauchs c6a7828 (#1192).
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        total_layers = self.num_hidden_layers + self.num_nextn_predict_layers
        self.compress_ratios = list(self.compress_ratios[:total_layers])
        # MTP layers default to compress_ratio=0 (no compression)
        while len(self.compress_ratios) < total_layers:
            self.compress_ratios.append(0)
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


# --------------------------------------------------------------------------- #
# Fused partial-RoPE Metal kernel                                             #
# --------------------------------------------------------------------------- #
#
# Decode dispatch reduction: the scalar-Python rotation (slice -> reshape ->
# index x0/x1 -> 4 muls + add/sub -> stack -> reshape) issues ~5 graph ops per
# rope call, and DeepseekV4 invokes rope ~3x per attention layer (q_pe, k_pe,
# inverse on attention output) -> 129 calls/token at L=1. Collapsing the chain
# into a single Metal kernel removes ~600 dispatches/token on the decode path.
#
# Adapted from @0xClandestine's optimization PR
# (https://github.com/Blaizzy/mlx-lm/pull/13) targeting Blaizzy's V4 branch.
# We use the rope-only signature (V4Attention splits nope/rope outside the
# rope call), so the nope passthrough loop is dropped from the source.
#
# One SIMD-group per (b, h, l) work item; lane t handles the interleaved
# pair (x[2t], x[2t+1]).

def _make_partial_rope_kernel():
    # Env-var escape hatch so benchmarks can A/B kernel ON vs OFF without
    # monkey-patching: MLX_LM_DISABLE_PARTIAL_ROPE_KERNEL=1 -> falls back
    # to the pure-MLX path used pre-2026-04-25.
    import os
    if os.environ.get("MLX_LM_DISABLE_PARTIAL_ROPE_KERNEL", "0") == "1":
        return None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        constexpr int DRH = D_ROPE / 2;
        int L_v = dims[0];
        int H_v = dims[1];
        uint l   = gid % (uint)L_v;
        uint tmp = gid / (uint)L_v;
        uint h   = tmp % (uint)H_v;
        uint b   = tmp / (uint)H_v;

        const auto xp = x  + ((uint64_t)b * H_v * L_v + h * L_v + l) * D_ROPE;
        auto       yp = y  + ((uint64_t)b * H_v * L_v + h * L_v + l) * D_ROPE;
        const auto cp = cos_s + l * DRH;
        const auto sp = sin_s + l * DRH;

        // Lane t handles one interleaved pair (x[2t], x[2t+1]).
        if ((int)tid < DRH) {
            float x0 = float(xp[2 * tid]);
            float x1 = float(xp[2 * tid + 1]);
            float c  = float(cp[tid]);
            float s  = float(sp[tid]);
            if (INVERSE) {
                store_elem(yp[2 * tid],     fma( x1, s, x0 * c));   //  x0*c + x1*s
                store_elem(yp[2 * tid + 1], fma(-x0, s, x1 * c));   // -x0*s + x1*c
            } else {
                store_elem(yp[2 * tid],     fma(-x1, s, x0 * c));   //  x0*c - x1*s
                store_elem(yp[2 * tid + 1], fma( x0, s, x1 * c));   //  x0*s + x1*c
            }
        }
    """
    return mx.fast.metal_kernel(
        name="ds4_partial_rope",
        input_names=["x", "cos_s", "sin_s", "dims"],
        output_names=["y"],
        header="template<typename T> inline void store_elem(device T& dst, float v) { dst = T(v); }",
        source=source,
    )


_partial_rope_kernel = _make_partial_rope_kernel()


class DeepseekV4RoPE(nn.Module):
    """DeepSeek-V4 rotary embedding.

    The reference implementation applies RoPE to the KV tensor before attention
    and applies the conjugate rotation to the attention output. The generic MLX
    RoPE layers do not expose an inverse path, so keep the small DeepSeek-specific
    implementation here.
    """

    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.dims = dims

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings
                        / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = math.floor(correction_dim(beta_fast))
            high = math.ceil(correction_dim(beta_slow))
            low = max(low, 0)
            high = min(high, dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default", "linear"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type {rope_type}")

        # This is derived from config, not a checkpoint parameter.
        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(self, x: mx.array, offset: int = 0, inverse: bool = False):
        dtype = x.dtype
        T = x.shape[-2]
        if isinstance(offset, mx.array):
            if offset.size == 1:
                offset = offset.item()
            else:
                B = offset.shape[0]
                pos = offset[:, None] + mx.arange(T, dtype=mx.float32)[None, :]
                theta = pos[..., None] * self.inv_freq[None, None, :]
                if inverse:
                    theta = -theta
                # theta: [B, T, dims//2]. Reshape for x dims: [B,H,T,D] or [B,1,T,D]
                target_shape = (B,) + (1,) * (x.ndim - 3) + (T, self.dims // 2)
                cos = mx.cos(theta).reshape(target_shape).astype(dtype)
                sin = mx.sin(theta).reshape(target_shape).astype(dtype)
                rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
                x0 = rot[..., 0]
                x1 = rot[..., 1]
                r0 = x0 * cos - x1 * sin
                r1 = x0 * sin + x1 * cos
                rotated = mx.stack([r0, r1], axis=-1).reshape(*x.shape[:-1], self.dims)
                if self.dims < x.shape[-1]:
                    return mx.concatenate([rotated, x[..., self.dims:]], axis=-1)
                return rotated
        # Fast path: fused Metal kernel for the rope-only 4D case used by
        # V4Attention. Falls through to the pure-MLX path on CPU, on Mode-B
        # (x has a nope tail), or on non-4D inputs (e.g. Indexer rope).
        # The kernel itself handles inverse via formula sign-flip; theta is
        # always forward-direction (do NOT negate it here).
        if (
            _partial_rope_kernel is not None
            and x.shape[-1] == self.dims
            and x.ndim == 4
        ):
            B, H, L, _ = x.shape
            pos = mx.arange(offset, offset + T, dtype=mx.float32)
            theta = pos[:, None] * self.inv_freq[None, :]
            cos = mx.cos(theta).astype(mx.float32)
            sin = mx.sin(theta).astype(mx.float32)
            dims_arr = mx.array([L, H], dtype=mx.int32)
            return _partial_rope_kernel(
                inputs=[x, cos, sin, dims_arr],
                template=[("D_ROPE", self.dims), ("INVERSE", 1 if inverse else 0)],
                grid=(B * H * L * 32, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]

        pos = mx.arange(offset, offset + T, dtype=mx.float32)
        theta = pos[:, None] * self.inv_freq[None, :]
        if inverse:
            theta = -theta

        broadcast_shape = (1,) * (x.ndim - 2) + theta.shape
        cos = mx.cos(theta).reshape(broadcast_shape).astype(dtype)
        sin = mx.sin(theta).reshape(broadcast_shape).astype(dtype)

        rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
        x0 = rot[..., 0]
        x1 = rot[..., 1]
        y = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
        y = y.reshape(*x.shape[:-1], self.dims)
        if x.shape[-1] == self.dims:
            return y
        return mx.concatenate([y, x[..., self.dims :]], axis=-1)

    def at_positions(
        self, x: mx.array, positions: mx.array, inverse: bool = False
    ) -> mx.array:
        """Apply RoPE at explicit, possibly non-consecutive positions.

        Compressed rows are born at chunk starts (0, ratio, 2 * ratio, ...),
        so the consecutive-offset interface above cannot position them.
        """
        if positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
            raise ValueError(
                "positions must be one-dimensional and match x's sequence length"
            )

        dtype = x.dtype
        theta = positions.astype(mx.float32)[:, None] * self.inv_freq[None, :]
        if inverse:
            theta = -theta
        broadcast_shape = (1,) * (x.ndim - 2) + theta.shape
        cos = mx.cos(theta).reshape(broadcast_shape).astype(dtype)
        sin = mx.sin(theta).reshape(broadcast_shape).astype(dtype)

        rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
        x0 = rot[..., 0]
        x1 = rot[..., 1]
        y = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
        y = y.reshape(*x.shape[:-1], self.dims)
        if x.shape[-1] == self.dims:
            return y
        return mx.concatenate([y, x[..., self.dims :]], axis=-1)


# --------------------------------------------------------------------------- #
# Gate (hash + score-based)                                                   #
# --------------------------------------------------------------------------- #

# Pre-allocated scalar zero for sqrtsoftplus: avoids mx.zeros_like() allocation per call.
_SCORE_ZERO = mx.array(0.0)


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    # sqrtsoftplus: sqrt(softplus(x))  — used by V4
    # Scalar broadcast avoids allocating a zeros tensor every call.
    return mx.sqrt(mx.logaddexp(scores, _SCORE_ZERO))


class MoEGate(nn.Module):
    """Routing gate. First `num_hash_layers` layers use a deterministic hash
    (token-id -> expert-id table) instead of learned score-based topk. Remaining
    layers run sqrtsoftplus scoring + e_score_correction_bias + topk, with
    post-softmax renormalization if score_func != 'softmax'."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_routed = args.n_routed_experts
        self.top_k = args.num_experts_per_tok
        self.hash = layer_idx < args.num_hash_layers
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob

        self.weight = mx.zeros((self.n_routed, args.hidden_size))
        # Cache transposed weight to avoid recomputing .T every forward call.
        self._weight_t = None
        if self.hash:
            # tid2eid: [vocab, top_k] int32 — predetermined expert routing per token id
            self.tid2eid = mx.zeros((args.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros((self.n_routed,), dtype=mx.float32)

    @property
    def weight_t(self):
        if self._weight_t is None:
            self._weight_t = self.weight.T
        return self._weight_t

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        # x: [B, S, D] or [N, D]
        if self.hash:
            # x shape -> [B*S, D]; input_ids -> [B, S] flattened to [B*S]
            flat = x.reshape(-1, x.shape[-1])
            scores = flat.astype(mx.float32) @ self.weight_t.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            ids = input_ids.reshape(-1)
            inds = self.tid2eid[ids].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            # Reshape inds/weights back to match x's leading dims so SwitchGLU
            # can broadcast against x: [B, S, top_k] (mirrors non-hash branch).
            inds = inds.reshape(*x.shape[:-1], self.top_k)
            weights = weights.reshape(*x.shape[:-1], self.top_k)
        else:
            scores = x.astype(mx.float32) @ self.weight_t.astype(mx.float32)
            scores = _score_func(scores, self.score_func)
            orig = scores
            biased = scores + self.e_score_correction_bias
            inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
            weights = mx.take_along_axis(orig, inds, axis=-1)

        if self.score_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.route_scale
        return inds, weights


# --------------------------------------------------------------------------- #
# MoE                                                                          #
# --------------------------------------------------------------------------- #

def _swiglu_limited(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        up = mx.clip(up, -limit, limit)
        gate = mx.minimum(gate, limit)
    return nn.silu(gate) * up


class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_swiglu_limited(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )
        self.gate = MoEGate(args, layer_idx)
        if args.n_shared_experts:
            self.shared_experts = DeepseekV4MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.n_shared_experts,
                swiglu_limit=0.0,
            )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, weights = self.gate(x, input_ids)
        # Compute shared_experts before switch_mlp so MLX can overlap both
        # on the GPU — shared_experts doesn't depend on routing results.
        shared_y = self.shared_experts(x) if hasattr(self, "shared_experts") else None
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        if shared_y is not None:
            y = y + shared_y
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


# --------------------------------------------------------------------------- #
# Attention: MLA (num_kv_heads=1) + sliding window + optional compressed KV   #
# --------------------------------------------------------------------------- #

class CompressedKVCache(KVCache):
    """Cache for compressed-attention layers: sliding-window local cache + compressed KV pool.

    During prefill, the compressor produces all compressed rows at once.
    During decode, projected tokens accumulate in the compressor's reference
    state; every `ratio` tokens a compressed row is appended to the pool.

    Inherits from KVCache so external engines (vllm-mlx) recognize it via
    isinstance checks. All state is proxied through self.local (RotatingKVCache).
    """

    # The compressor state has no batched representation. BatchGenerator reads
    # this capability before it mutates a live scheduler batch.
    supports_batching = False

    def __init__(self, max_size: int = 128):
        # Skip KVCache.__init__ — we proxy everything through self.local
        self.local = RotatingKVCache(max_size=max_size, keep=0)
        self._pool = None
        self._state_kv = None
        self._state_score = None
        self._abs_pos = 0
        self._compress_ratio = None
        self._compress_overlap = None

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def values(self):
        return self.local.values

    @values.setter
    def values(self, value):
        self.local.values = value

    @property
    def pool(self):
        return self._pool

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    @property
    def state(self):
        return self.local.state

    @state.setter
    def state(self, value):
        self.local.state = value

    @property
    def nbytes(self):
        n = self.local.nbytes
        if self._pool is not None:
            n += self._pool.nbytes
        if self._state_kv is not None:
            n += self._state_kv.nbytes
            n += self._state_score.nbytes
        return n

    @property
    def meta_state(self):
        raise NotImplementedError(
            "DeepSeek-V4 CompressedKVCache prompt-cache serialization is unsupported"
        )

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value

    def is_trimmable(self):
        # Rolling the local window back without also reconstructing the
        # compressor pool and overlap state would silently corrupt the next
        # decode step.  Report the capability as unavailable from cache
        # creation onward so speculative decoding rejects this cache before
        # its prefill-time capability check can become stale.
        return False

    def trim(self, n):
        if n and self._abs_pos:
            raise RuntimeError(
                "CompressedKVCache cannot be trimmed after compression state "
                "has been initialized"
            )
        return self.local.trim(n)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError(
            "DeepSeek-V4 CompressedKVCache KV-cache quantization is unsupported"
        )

    def _configure(self, compressor: 'Compressor'):
        signature = (compressor.ratio, compressor.overlap)
        current = (self._compress_ratio, self._compress_overlap)
        if self._compress_ratio is None:
            self._compress_ratio, self._compress_overlap = signature
        elif current != signature:
            raise ValueError(
                "CompressedKVCache cannot be reused with a different compressor"
            )

    @classmethod
    def merge(cls, caches):
        """Clone the unbatched cache used by a singleton scheduler batch."""
        if not caches:
            raise ValueError("CompressedKVCache.merge requires at least one cache")
        if len(caches) != 1:
            raise ValueError(
                "DeepSeek-V4 CompressedKVCache does not support batching multiple streams"
            )

        source = caches[0]
        merged = cls.__new__(cls)
        merged.local = RotatingKVCache(source.local.max_size, source.local.keep)
        merged.local.keys = (
            source.local.keys + 0 if source.local.keys is not None else None
        )
        merged.local.values = (
            source.local.values + 0 if source.local.values is not None else None
        )
        merged.local.offset = source.local.offset
        merged.local._idx = source.local._idx
        for name in ("_pool", "_state_kv", "_state_score"):
            value = getattr(source, name)
            setattr(merged, name, value + 0 if value is not None else None)
        merged._abs_pos = source._abs_pos
        merged._compress_ratio = source._compress_ratio
        merged._compress_overlap = source._compress_overlap

        return merged

    def filter(self, batch_indices):
        if isinstance(self.local, RotatingKVCache):
            indices = (
                batch_indices.tolist()
                if hasattr(batch_indices, "tolist")
                else list(batch_indices)
            )
            if indices != [0]:
                raise ValueError(
                    "CompressedKVCache only supports singleton filtering"
                )
            return

        self.local.filter(batch_indices)
        if self._pool is not None:
            self._pool = self._pool[batch_indices]
        if self._state_kv is not None:
            self._state_kv = self._state_kv[batch_indices]
            self._state_score = self._state_score[batch_indices]

    def extend(self, other):
        raise ValueError(
            "DeepSeek-V4 CompressedKVCache does not support extending another stream"
        )

    def finalize(self):
        if hasattr(self.local, 'finalize'):
            self.local.finalize()

    def extract(self, idx):
        extracted = CompressedKVCache.__new__(CompressedKVCache)
        if isinstance(self.local, RotatingKVCache):
            if idx != 0:
                raise ValueError("CompressedKVCache only has singleton index 0")
            extracted.local = RotatingKVCache(self.local.max_size, self.local.keep)
            extracted.local.keys = (
                self.local.keys + 0 if self.local.keys is not None else None
            )
            extracted.local.values = (
                self.local.values + 0 if self.local.values is not None else None
            )
            extracted.local.offset = self.local.offset
            extracted.local._idx = self.local._idx
            extracted._pool = self._pool + 0 if self._pool is not None else None
            extracted._state_kv = (
                self._state_kv + 0 if self._state_kv is not None else None
            )
            extracted._state_score = (
                self._state_score + 0 if self._state_score is not None else None
            )
        else:
            extracted.local = self.local.extract(idx)
            extracted._pool = (
                self._pool[idx : idx + 1] if self._pool is not None else None
            )
            extracted._state_kv = (
                self._state_kv[idx : idx + 1] if self._state_kv is not None else None
            )
            extracted._state_score = (
                self._state_score[idx : idx + 1]
                if self._state_score is not None
                else None
            )
        extracted._abs_pos = self._abs_pos
        extracted._compress_ratio = self._compress_ratio
        extracted._compress_overlap = self._compress_overlap
        return extracted

    @property
    def batch_size(self):
        if hasattr(self.local, 'batch_size'):
            return self.local.batch_size
        return 1

    def accumulate(self, x: mx.array, compressor: 'Compressor') -> Optional[mx.array]:
        """Advance the official 0731 compressor state and return its raw pool.

        Args:
            x: [B, S, D] hidden states for current step(s)
            compressor: the Compressor module to apply

        Returns:
            The full compressed pool [B, N_compressed, head_dim], or None if empty.
        """
        B, S, _ = x.shape
        r = compressor.ratio
        self._configure(compressor)

        if S > 1:
            if self._abs_pos != 0:
                raise RuntimeError(
                    "CompressedKVCache does not support chunked prefill after "
                    "compressor state has been initialized"
                )
            kv, score = compressor.project(x)
            ckv = compressor.pool_projected(kv, score, x.dtype)
            if ckv.shape[1] > 0:
                self._pool = ckv if self._pool is None else mx.concatenate([self._pool, ckv], axis=1)

            remainder = S % r
            cutoff = S - remainder
            offset = r if compressor.overlap else 0
            width = kv.shape[-1]
            rows = 2 * r if compressor.overlap else r
            self._state_kv = mx.zeros(
                (B, rows, width), dtype=mx.float32
            )
            self._state_score = mx.full(
                (B, rows, width), float("-inf"), dtype=mx.float32
            )

            if compressor.overlap and cutoff >= r:
                self._state_kv[:, :r] = kv[:, cutoff-r:cutoff]
                self._state_score[:, :r] = (
                    score[:, cutoff-r:cutoff] + compressor.ape
                )
            if remainder > 0:
                self._state_kv[:, offset:offset+remainder] = kv[:, cutoff:]
                self._state_score[:, offset:offset+remainder] = (
                    score[:, cutoff:] + compressor.ape[:remainder]
                )
            self._abs_pos = S
            return self._pool

        pos = self._abs_pos
        pos_mod = pos % r
        kv_cur, sc_cur = compressor.project(x)
        sc_cur = sc_cur + compressor.ape[pos_mod:pos_mod+1]

        if self._state_kv is None:
            rows = 2 * r if compressor.overlap else r
            self._state_kv = mx.zeros(
                (B, rows, kv_cur.shape[-1]), dtype=mx.float32
            )
            self._state_score = mx.full(
                (B, rows, kv_cur.shape[-1]),
                float("-inf"),
                dtype=mx.float32,
            )

        row = (r + pos_mod) if compressor.overlap else pos_mod
        self._state_kv[:, row:row+1, :] = kv_cur
        self._state_score[:, row:row+1, :] = sc_cur

        self._abs_pos = pos + 1

        if (pos + 1) % r == 0:
            if compressor.overlap:
                pooled_kv = mx.concatenate(
                    [
                        self._state_kv[:, :r, :compressor.head_dim],
                        self._state_kv[:, r:, compressor.head_dim:],
                    ],
                    axis=1,
                )
                pooled_score = mx.concatenate(
                    [
                        self._state_score[:, :r, :compressor.head_dim],
                        self._state_score[:, r:, compressor.head_dim:],
                    ],
                    axis=1,
                )
            else:
                pooled_kv = self._state_kv
                pooled_score = self._state_score
            weights = mx.softmax(pooled_score, axis=1, precise=True)
            pooled = (pooled_kv * weights).sum(axis=1, keepdims=True)
            ckv = compressor.norm(pooled.astype(x.dtype))
            self._pool = ckv if self._pool is None else mx.concatenate([self._pool, ckv], axis=1)
            if compressor.overlap:
                self._state_kv[:, :r] = self._state_kv[:, r:]
                self._state_score[:, :r] = self._state_score[:, r:]

        return self._pool


class Compressor(nn.Module):
    """Learned gated pooling over `ratio` consecutive tokens for KV compression.

    At prefill, produces ~ seq/ratio compressed KV rows. At decode, accumulates
    tokens in a state buffer and emits a compressed row every `ratio` steps.
    Pure-MLX; a fused Metal kernel may replace this in a follow-up.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.ratio = compress_ratio
        self.overlap = compress_ratio == 4
        out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(self.dim, out_dim, bias=False)
        self.wgate = nn.Linear(self.dim, out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, out_dim), dtype=mx.float32)
        self.norm  = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

    def _overlap_transform(self, tensor: mx.array, value: float) -> mx.array:
        B, S, R, _ = tensor.shape
        D = self.head_dim
        out = mx.full((B, S, 2 * R, D), value, dtype=tensor.dtype)
        out[:, :, R:] = tensor[:, :, :, D:]
        out[:, 1:, :R] = tensor[:, :-1, :, :D]
        return out

    def project(self, x: mx.array):
        """Project hidden states in fp32, matching the official reference."""
        xf = x.astype(mx.float32)
        return self.wkv(xf), self.wgate(xf)

    def pool_projected(
        self, kv: mx.array, score: mx.array, dtype
    ) -> mx.array:
        """Pool complete projected chunks without changing rolling state."""
        B, S, _ = kv.shape
        r = self.ratio
        keep = (S // r) * r
        if keep == 0:
            return mx.zeros((B, 0, self.head_dim), dtype=dtype)
        kv = kv[:, :keep].reshape(B, keep // r, r, -1)
        score = score[:, :keep].reshape(B, keep // r, r, -1) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))
        weights = mx.softmax(score, axis=2, precise=True)
        kv = (kv * weights).sum(axis=2)
        return self.norm(kv.astype(dtype))

    def __call__(self, x: mx.array) -> mx.array:
        kv, score = self.project(x)
        return self.pool_projected(kv, score, x.dtype)


class V4Attention(nn.Module):
    """V4 attention block.

    Checkpoint shapes (Flash):
        n_heads=64, head_dim=512, rope_head_dim=64 (nope=448)
        q_lora_rank=1024,  wq_a: [dim, 1024], wq_b: [1024, n_heads*head_dim]
        wkv: [dim, head_dim]            (single shared K=V head, MQA-style)
        attn_sink: [n_heads] fp32
        wo_a: [n_heads*head_dim/n_groups, n_groups*o_lora_rank]
        wo_b: [n_groups*o_lora_rank, dim]
        For compress_ratio != 0: compressor.wkv/wgate/ape/norm; and if ratio==4, indexer.*

    Forward path (MVP):
        - Project Q (64 heads), K=V (1 head); apply RoPE to last `rope_head_dim` dims.
        - For ratio=0 layers: sliding window mask of size `sliding_window`.
        - For ratio!=0 layers: append compressed KV rows to attend to (no topk filtering
          yet — full compressed cache). Use attn_sink via SDPA `sinks=` argument.
        - Grouped low-rank output projection: wo_a per group -> concat -> wo_b.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.window = args.sliding_window
        self.eps = args.rms_norm_eps

        ratios = args.compress_ratios or []
        self.compress_ratio = ratios[layer_idx] if layer_idx < len(ratios) else 0

        self.scale = self.head_dim ** -0.5

        # q path
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # kv path (single shared head)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        # attention sink (per-head learnable bias added in softmax denominator)
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # grouped low-rank output projection
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(group_feat, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=args.attention_bias)

        # The official reference chooses the RoPE regime per layer. Compressed
        # layers use the YaRN-scaled compressed base; dense sliding-window
        # layers use unscaled ordinary RoPE.
        if self.compress_ratio:
            self.rope = DeepseekV4RoPE(
                self.rope_head_dim,
                args.compress_rope_theta,
                args.rope_scaling,
            )
        else:
            self.rope = DeepseekV4RoPE(
                self.rope_head_dim,
                args.rope_theta,
                scaling_config=None,
            )

        # Compressor / Indexer — present only when compress_ratio > 0
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        B, S = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        out = out.reshape(B, S, self.n_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            # Batched grouped quantized matmul: collapse the per-group Python
            # loop (8 dispatches) into a single mx.quantized_matmul call by
            # treating the group dim as a broadcast batch dim. Adapted from
            # @Blaizzy's pc/add-deepseekv4flash-model branch.
            #
            # Shapes:
            #   out (after transpose): [G, B, S, group_feat]
            #   weight (reshaped):     [G, 1, o_lora_rank, group_feat / pack_factor]
            #   scales:                [G, 1, o_lora_rank, group_feat / group_size]
            # Single dispatch returns [G, B, S, o_lora_rank], then transpose
            # back to [B, S, G, o_lora_rank] -> [B, S, G * o_lora_rank].
            out_g = out.transpose(2, 0, 1, 3)
            weight = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            scales = self.wo_a.scales.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            )
            out_g = mx.quantized_matmul(
                out_g,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=self.wo_a.group_size,
                bits=self.wo_a.bits,
                mode=self.wo_a.mode,
            )
            out = out_g.transpose(1, 2, 0, 3).reshape(B, S, self.n_groups * self.o_lora_rank)
            if "bias" in self.wo_a:
                out = out + self.wo_a.bias
            return out

        wa = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, group_feat)
        out = mx.einsum("bsgd,grd->bsgr", out, wa)
        out = out.reshape(B, S, self.n_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out

    def _rotate_compressed_pool(self, pool: mx.array) -> mx.array:
        """Return a positioned view of the raw cached compressed pool.

        `CompressedKVCache` intentionally owns unrotated compressor outputs.
        Positioning here keeps a decode step from rotating old cached rows again.
        It also happens before any indexer gather, preserving every row's true
        chunk-start position.
        """
        nope, rope = mx.split(pool, [self.nope_head_dim], axis=-1)
        positions = (
            mx.arange(pool.shape[1], dtype=mx.int32) * self.compress_ratio
        )
        rope = self.rope.at_positions(rope, positions)
        return mx.concatenate([nope, rope], axis=-1)

    def _compressed_pool_mask(
        self, mask: mx.array, pool_indices: mx.array, offset: int
    ) -> mx.array:
        """Build the causal allow mask for selected compressed-pool rows."""
        query_delta = mx.arange(mask.shape[-2], dtype=mx.int32)
        if isinstance(offset, mx.array) and offset.size > 1:
            if offset.ndim != 1:
                raise ValueError("batched cache offsets must be one-dimensional")
            query_positions = offset[:, None] + query_delta[None, :]
        else:
            if isinstance(offset, mx.array):
                offset = offset.reshape(()).item()
            query_positions = offset + query_delta

        chunk_ends = pool_indices * self.compress_ratio + (self.compress_ratio - 1)
        if query_positions.ndim == 1 and pool_indices.ndim == 1:
            allowed = query_positions[:, None] >= chunk_ends[None, :]
        elif query_positions.ndim == 1:
            allowed = query_positions[None, :, None] >= chunk_ends[:, None, :]
        elif pool_indices.ndim == 1:
            allowed = query_positions[:, :, None] >= chunk_ends[None, None, :]
        else:
            if query_positions.shape[0] != pool_indices.shape[0]:
                raise ValueError(
                    "batched cache offsets and compressed-pool indices must "
                    "have the same batch size"
                )
            allowed = query_positions[:, :, None] >= chunk_ends[:, None, :]

        if mask.dtype == mx.bool_:
            return allowed
        return mx.where(
            allowed,
            mx.array(0.0, dtype=mask.dtype),
            mx.array(float("-inf"), dtype=mask.dtype),
        )

    def _prepend_compressed_pool_mask(
        self, mask: mx.array, pool_indices: mx.array, offset: int
    ) -> mx.array:
        """Prepend compressed columns while preserving batch/head dimensions."""
        comp_mask = self._compressed_pool_mask(mask, pool_indices, offset)

        # Per-batch indexer selections are [B, Q, Kc]. Attention masks use an
        # explicit head axis, so normalize both operands to [B, H, Q, K].
        if comp_mask.ndim == 3:
            comp_mask = mx.expand_dims(comp_mask, axis=1)
            if mask.ndim == 3 and mask.shape[0] == comp_mask.shape[0]:
                mask = mx.expand_dims(mask, axis=1)

        while mask.ndim < comp_mask.ndim:
            mask = mx.expand_dims(mask, axis=0)
        while comp_mask.ndim < mask.ndim:
            comp_mask = mx.expand_dims(comp_mask, axis=0)

        prefix = []
        for mask_dim, comp_dim in zip(mask.shape[:-2], comp_mask.shape[:-2]):
            if mask_dim != comp_dim and mask_dim != 1 and comp_dim != 1:
                raise ValueError(
                    "compressed and local attention masks have incompatible "
                    "batch/head dimensions"
                )
            prefix.append(max(mask_dim, comp_dim))

        mask = mx.broadcast_to(mask, (*prefix, *mask.shape[-2:]))
        comp_mask = mx.broadcast_to(
            comp_mask, (*prefix, *comp_mask.shape[-2:])
        )
        return mx.concatenate([comp_mask, mask], axis=-1)

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, S, _ = x.shape

        # --- Q (shared intermediate reused by indexer) ---
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rms_norm(q, weight=None, eps=self.eps)

        # --- K = V (shared single-head) ---
        kv = self.kv_norm(self.wkv(x))
        kv = kv.reshape(B, 1, S, self.head_dim)

        offset = cache.offset if cache is not None else 0

        # Apply RoPE only to the last rope_head_dim dims
        q_nope, q_pe = mx.split(q,  [self.nope_head_dim], axis=-1)
        k_nope, k_pe = mx.split(kv, [self.nope_head_dim], axis=-1)
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)
        q = mx.concatenate([q_nope, q_pe], axis=-1)
        k = v = mx.concatenate([k_nope, k_pe], axis=-1)

        # --- Compressed sparse attention ---
        compressed_k = compressed_v = None
        if self.compress_ratio:
            comp_cache = cache if isinstance(cache, CompressedKVCache) else None
            if comp_cache is not None:
                pool = comp_cache.accumulate(x, self.compressor)
            elif S > 1:
                pool = self.compressor(x)
                pool = pool if pool.shape[1] > 0 else None
            else:
                pool = None

            if pool is not None:
                # Position the complete raw pool before selection. Positioning a
                # selected subset as 0..K-1 would relabel its source chunks.
                ckv = self._rotate_compressed_pool(pool)
                pool_indices = mx.arange(pool.shape[1], dtype=mx.int32)
                if hasattr(self, "indexer") and ckv.shape[1] > self.args.index_topk:
                    topk_idx = self.indexer(x, qr)
                    if topk_idx is not None:
                        idx = mx.broadcast_to(
                            topk_idx[:, :, None],
                            (B, topk_idx.shape[1], self.head_dim),
                        )
                        ckv = mx.take_along_axis(ckv, idx, axis=1)
                        pool_indices = topk_idx
                compressed_k = ckv[:, None, :, :]
                compressed_v = compressed_k

        # Update KV cache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Prepend compressed KV to cached KV for sparse attention
        if compressed_k is not None:
            k = mx.concatenate([compressed_k, k], axis=2)
            v = mx.concatenate([compressed_v, v], axis=2)
            if mask is not None:
                mask = self._prepend_compressed_pool_mask(
                    mask, pool_indices, offset
                )

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )

        out_nope, out_pe = mx.split(out, [self.nope_head_dim], axis=-1)
        out_pe = self.rope(out_pe, offset=offset, inverse=True)
        out = mx.concatenate([out_nope, out_pe], axis=-1)

        # Grouped low-rank projection: [B, n_heads, S, head_dim] -> [B, S, n_heads*head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)


class Indexer(nn.Module):
    """Top-k selector over compressed KV rows for ratio-4 sparse attention.

    Two-pass design: this module uses a lightweight compressor (index_head_dim,
    typically 128) to score all compressed rows cheaply, then returns topk
    indices used to gather from the main attention compressor's output
    (head_dim, typically 512). This reduces per-layer attention from O(S/4)
    to O(topk) compressed rows — 500x at 1M context with topk=512.

    This is not yet the official 0731 indexer: the reference scores a
    persistent compressed cache independently for each query position. This
    implementation selects one aggregate set from the current input. Keep that
    broader fidelity gap explicit; do not reinterpret selected rows' positions
    to hide it.

    Checkpoint params:
        wq_b: [q_lora_rank, n_heads * index_head_dim]
        weights_proj: [hidden_size, n_heads]
        compressor.{wkv, wgate, ape, norm}
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.scale = args.index_head_dim ** -0.5
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.compressor = Compressor(args, compress_ratio, self.head_dim)

    def __call__(
        self,
        x: mx.array,
        q_intermediate: mx.array,
    ) -> Optional[mx.array]:
        """Score compressed rows and return topk indices.

        Args:
            x: [B, S, D] hidden state (fed to the lightweight compressor).
            q_intermediate: [B, S, q_lora_rank] post wq_a+q_norm (shared with main attn).

        Returns:
            topk_indices [B, topk] or None when there are too few compressed rows.
            Indices are shared across heads (head-weighted scores are aggregated).
        """
        B, S, _ = x.shape

        ck = self.compressor(x)
        n_compressed = ck.shape[1]
        if n_compressed == 0:
            return None

        q = self.wq_b(q_intermediate)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)

        scores = (q @ ck[:, None].transpose(0, 1, 3, 2)) * self.scale

        hw = mx.sigmoid(self.weights_proj(x))
        hw = hw.transpose(0, 2, 1)[..., None]
        scores = scores * hw

        agg = scores.sum(axis=2).mean(axis=1)

        topk = min(self.index_topk, n_compressed)
        return mx.argpartition(-agg, kth=topk - 1, axis=-1)[:, :topk]


# --------------------------------------------------------------------------- #
# Block                                                                       #
# --------------------------------------------------------------------------- #

class DeepseekV4Block(nn.Module):
    """V4 block: mHC-wrapped (attention-norm -> attention), mHC-wrapped (moe-norm -> moe).

    The block maintains `hc_mult` parallel hidden-state copies. Each sub-layer
    reduces them to 1 via hc_pre, applies its block, then expands back via hc_post.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = V4Attention(args, layer_idx)
        self.hc_attn = HyperConnection(
            args.hidden_size, args.hc_mult,
            args.rms_norm_eps, args.hc_sinkhorn_iters, args.hc_eps,
        )

        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = DeepseekV4MoE(args, layer_idx)
        self.hc_ffn = HyperConnection(
            args.hidden_size, args.hc_mult,
            args.rms_norm_eps, args.hc_sinkhorn_iters, args.hc_eps,
        )

    def __call__(self, h: mx.array, mask, cache, input_ids: mx.array) -> mx.array:
        # h: [B, S, hc, D]
        # Attention half
        residual = h
        y, post, comb = self.hc_attn.hc_pre(h)
        y = self.attn_norm(y)
        y = self.attn(y, mask=mask, cache=cache)
        h = self.hc_attn.hc_post(y, residual, post, comb)

        # FFN half
        residual = h
        y, post, comb = self.hc_ffn.hc_pre(h)
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        h = self.hc_ffn.hc_post(y, residual, post, comb)
        return h


# --------------------------------------------------------------------------- #
# MTP Block (next-N-token prediction head, from Blaizzy/mlx-lm PR #15)        #
# --------------------------------------------------------------------------- #

class MTPBlock(nn.Module):
    """Next-N-token prediction head. Each MTP block predicts one extra future
    token by re-mixing the previous hidden state with the embedded "next" token,
    then running it through a copy of the V4 transformer block + hc_head.

    Adapted from Blaizzy/mlx-lm PR #15. HyperHead signature matches our fork
    (hidden_size, hc_mult, rms_norm_eps, hc_eps) instead of PR's HyperHead(config).
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.block = DeepseekV4Block(args, layer_idx)
        self.e_proj = nn.Linear(dim, dim, bias=False)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.enorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.hc_head = HyperHead(
            args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
        )

    def __call__(
        self,
        h: mx.array,
        embed_tokens: nn.Embedding,
        input_ids: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> mx.array:
        e = embed_tokens(input_ids)
        e = self.enorm(e)
        h_norm = self.hnorm(h)
        x = self.e_proj(e)[:, :, None, :] + self.h_proj(h_norm)
        x = mx.contiguous(x)
        x = self.block(x, mask, cache, input_ids)
        return x


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

class DeepseekV4Model(nn.Module, PipelineMixin):
    def __init__(self, args: ModelArgs):
        super().__init__()
        PipelineMixin.__init__(self)
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DeepseekV4Block(args, i) for i in range(args.num_hidden_layers)]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Final HC head (reduces hc copies -> 1 before lm_head)
        self.hc_head = HyperHead(
            args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
        )

    def __call__(self, inputs: mx.array, cache=None, return_raw_hidden: bool = False):
        h = self.embed_tokens(inputs)                        # [B, S, D]
        # Expand to hc_mult parallel copies
        h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]))
        # Make it contiguous — broadcast_to gives a view
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * self.num_layers

        first_cache = cache[0]
        if isinstance(first_cache, CompressedKVCache):
            first_cache = first_cache.local
        elif isinstance(first_cache, (list, tuple)):
            first_cache = first_cache[0]
        mask = create_attention_mask(
            h[:, :, 0, :],
            first_cache if first_cache is not None else None,
            return_array=True,
        )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i], inputs)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            last_cache = cache[-1]
            if last_cache is not None:
                lc = last_cache.local if isinstance(last_cache, CompressedKVCache) else last_cache
                if hasattr(lc, 'keys') and lc.keys is not None:
                    lc.keys = mx.depends(lc.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # Reduce [B,S,hc,D] -> [B,S,D] then RMSNorm
        out = self.norm(self.hc_head(h))
        if return_raw_hidden:
            return out, h
        return out


class Model(nn.Module):
    supports_batching = False

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # 0731's `mtp.*` namespace contains a three-stage DSpark drafter, not
        # conventional MTP blocks. Target-only loading must not instantiate a
        # structurally incompatible speculative module merely because the root
        # config retains `num_nextn_predict_layers = 1`.
        if (
            not args.dspark_block_size
            and getattr(args, "num_nextn_predict_layers", 0) > 0
        ):
            n = args.num_hidden_layers
            self.mtp = [
                MTPBlock(args, n + i)
                for i in range(args.num_nextn_predict_layers)
            ]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
    ):
        if return_hidden:
            h, h_raw = self.model(inputs, cache, return_raw_hidden=True)
            return self.lm_head(h), h_raw
        h = self.model(inputs, cache)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def pred(k: str):
            # Keep mHC params and gate biases in fp32
            if "hc_" in k or "e_score_correction_bias" in k or "attn_sink" in k:
                return False
            if k.endswith(".fn") or k.endswith(".base") or k.endswith(".scale"):
                return False
            return True
        return pred

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attn.compress_ratio:
                caches.append(CompressedKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def make_mtp_cache(self):
        if not hasattr(self, "mtp"):
            return None
        caches = []
        for mtp_block in self.mtp:
            attn = mtp_block.block.attn
            if attn.compress_ratio:
                caches.append(CompressedKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches

    def mtp_forward(
        self,
        h: mx.array,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if not hasattr(self, "mtp"):
            raise RuntimeError(
                "DeepSeek-V4 DSpark speculation is not implemented; "
                "this model supports target generation only."
            )
        if cache is None:
            cache = [None] * len(self.mtp)

        first_cache = cache[0]
        mask_cache = (
            first_cache.local
            if isinstance(first_cache, CompressedKVCache)
            else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :] if h.ndim == 4 else h,
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        for mtp_block, layer_cache in zip(self.mtp, cache):
            h = mtp_block(
                h, self.model.embed_tokens, input_ids, mask, layer_cache
            )

        out = mtp_block.hc_head(h)
        out = mtp_block.norm(out)
        return self.lm_head(out)

    # ------------------------------------------------------------------- #
    # Weight loading                                                      #
    # ------------------------------------------------------------------- #

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Handle DeepSeek-V4 checkpoint conversion.

        Supports both raw HF checkpoints (FP8/FP4 block scales) and
        pre-quantized MLX checkpoints (e.g. mlx-community 8-bit).

        Checkpoint naming (from HF):
            layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
            layers.N.attn.{q_norm,kv_norm,attn_sink}
            layers.N.attn.compressor.{wkv,wgate,ape,norm}
            layers.N.attn.indexer.{wq_b,weights_proj,compressor.*}
            layers.N.ffn.gate.{weight,bias,tid2eid}
            layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}
            layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}
            layers.N.{attn_norm,ffn_norm}.weight
            layers.N.hc_{attn,ffn}_{fn,base,scale}
            embed.weight, head.weight, hc_head_{fn,base,scale}
            mtp.0.* (dropped)

        MLX-quantized naming (community 8-bit):
            embed.{weight,biases,scales}, head.{weight,biases,scales}
            layers.N.attn.wo_a.G.{weight,biases,scales} (per-group)
            layers.N.ffn.experts.w{1,2,3}.{weight,biases,scales} (pre-stacked)
        """
        n_layers = self.args.num_hidden_layers

        # 1) Keep conventional MTP weights only when that module exists. The
        # official 0731 `mtp.0/1/2.*` tensors are DSpark and must be dropped
        # until a dedicated DSpark backend is implemented.
        has_mtp = hasattr(self, "mtp")
        has_mtp_weights = any(k.startswith("mtp.") for k in weights)
        if self.args.dspark_block_size and has_mtp_weights:
            dropped = sum(k.startswith("mtp.") for k in weights)
            logger.warning(
                "DeepSeek-V4 DSpark is not implemented; dropping %d mtp.* "
                "weights and loading the target model only.",
                dropped,
            )
        # Disable MTP module if weights are absent (e.g. quantized checkpoints)
        if has_mtp and not has_mtp_weights:
            del self.mtp
            has_mtp = False
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                if not has_mtp:
                    continue
                new_weights[k] = v
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    idx = int(parts[1])
                except ValueError:
                    new_weights[k] = v
                    continue
                if idx >= n_layers:
                    continue
            new_weights[k] = v
        weights = new_weights

        def _scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        # 2) FP8/FP4 block dequant:
        #    `X.weight` + `X.scale` -> dequantized bf16 `X.weight`
        #    Routed experts in Flash are FP4-packed int8; other scaled matrices
        #    are FP8 e4m3 with 128x128 block scales.
        def _dequant_fp8_block(weight: mx.array, scale: mx.array, bs: int = 128) -> mx.array:
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            scale = _scale_to_float(scale)
            m, n = weight.shape
            pad_b = (-m) % bs
            pad_s = (-n) % bs
            weight = mx.pad(weight, ((0, pad_b), (0, pad_s)))
            weight = weight.reshape(((m + pad_b) // bs, bs, (n + pad_s) // bs, bs))
            weight = (weight * scale[:, None, :, None]).reshape(m + pad_b, n + pad_s)
            return weight[:m, :n].astype(mx.bfloat16)

        def _dequant_fp4_block(weight: mx.array, scale: mx.array, bs: int = 32) -> mx.array:
            table = mx.array(
                [
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
                ],
                dtype=mx.float32,
            )
            packed = weight.astype(mx.uint8)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            unpacked = mx.stack([mx.take(table, low), mx.take(table, high)], axis=-1)
            unpacked = unpacked.reshape(weight.shape[0], weight.shape[1] * 2)
            scale = mx.repeat(_scale_to_float(scale), bs, axis=-1)
            return (unpacked * scale).astype(mx.bfloat16)

        new = {}
        for k, v in weights.items():
            if k.endswith(".scale"):
                wk = k[:-len(".scale")] + ".weight"
                weight = weights.get(wk)
                if (
                    weight is not None
                    and ".ffn.experts." in wk
                    and "shared_experts" not in wk
                    and weight.dtype in (mx.int8, mx.uint8)
                    and v.shape[-1] * 16 == weight.shape[-1]
                ):
                    new[wk] = _dequant_fp4_block(weight, v)
                elif weight is not None and weight.dtype in (mx.uint8,):
                    new[wk] = _dequant_fp8_block(weights[wk], v)
                else:
                    new[k] = v
            elif k not in new:
                new[k] = v
        weights = new

        # 3) Remap top-level names to our module structure
        #    Prefix-based remap handles both raw (.weight) and quantized
        #    (.weight, .biases, .scales) checkpoints.
        top_prefix_remap = {
            "embed.":      "model.embed_tokens.",
            "head.":       "lm_head.",
        }
        top_exact_remap = {
            "norm.weight":     "model.norm.weight",
            "hc_head_fn":      "model.hc_head.fn",
            "hc_head_base":    "model.hc_head.base",
            "hc_head_scale":   "model.hc_head.scale",
        }
        new = {}
        for k, v in weights.items():
            nk = k
            for old_pfx, new_pfx in top_prefix_remap.items():
                if nk.startswith(old_pfx):
                    nk = new_pfx + nk[len(old_pfx):]
                    break
            if nk in top_exact_remap:
                nk = top_exact_remap[nk]
            new[nk] = v
        weights = new

        # 4) Remap layer-level names: layers.N.X -> model.layers.N.X
        #    Also remap gate.bias -> gate.e_score_correction_bias,
        #    hc_{attn,ffn}_{fn,base,scale} -> hc_{attn,ffn}.{fn,base,scale},
        #    shared_experts.w{1,2,3} -> shared_experts.{gate,down,up}_proj
        new = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        mtp_block_subs = (
            "attn.", "ffn.", "attn_norm.", "ffn_norm.",
            "hc_attn_", "hc_ffn_",
        )
        for k, v in weights.items():
            nk = k
            # Add model. prefix for main-model layers
            if nk.startswith("layers."):
                nk = "model." + nk

            # MTP block: nest block-internal weights under .block.
            if nk.startswith("mtp."):
                parts = nk.split(".", 2)  # ["mtp", "0", "rest"]
                if len(parts) == 3:
                    rest = parts[2]
                    if any(rest.startswith(s) for s in mtp_block_subs):
                        nk = f"mtp.{parts[1]}.block.{rest}"
                    # HC head weights for MTP block
                    for param in ("fn", "base", "scale"):
                        if rest == f"hc_head_{param}":
                            nk = f"mtp.{parts[1]}.hc_head.{param}"

            # gate.bias -> gate.e_score_correction_bias
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")

            # hc_attn_fn -> hc_attn.fn (etc.) — raw HF checkpoint underscores
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".hc_{sub}.{param}")

            # attn_hc.X -> hc_attn.X (mlx-community/DeepSeek-V4-Flash-8bit
            # naming order: per-layer hyper-connections stored as <sub>_hc.<param>
            # rather than hc_<sub>.<param>). Apply after the underscore rename so
            # both naming orders converge to the model's hc_<sub>.<param> layout.
            for sub in ("attn", "ffn"):
                nk = nk.replace(f".{sub}_hc.", f".hc_{sub}.")

            # shared_experts.w1 -> shared_experts.gate_proj (etc.)
            for w_old, w_new in w_remap.items():
                nk = nk.replace(f".shared_experts.{w_old}.", f".shared_experts.{w_new}.")

            new[nk] = v
        weights = new

        # 5) Stack expert weights: experts.E.w{1,2,3}.weight -> switch_mlp.{gate,down,up}_proj.weight
        #    Also handle pre-stacked experts (community quants): experts.w{1,2,3}.X -> switch_mlp.{proj}.X
        expert_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for l in range(n_layers):
            prefix = f"model.layers.{l}.ffn.experts"
            for src, dst in expert_remap.items():
                # Case A: per-expert weights need stacking (raw HF checkpoint)
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stack = [weights.pop(f"{prefix}.{e}.{src}.weight")
                             for e in range(self.args.n_routed_experts)]
                    weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.weight"] = mx.stack(stack)
                # Case B: already-stacked (community quant) — rename experts.w1.X -> switch_mlp.gate_proj.X
                for suffix in ("weight", "biases", "scales"):
                    old = f"{prefix}.{src}.{suffix}"
                    if old in weights:
                        weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.{suffix}"] = weights.pop(old)

        # 6) Fuse split wo_a: community quants store wo_a.G.{weight,biases,scales}
        #    per-group; our model uses a single QuantizedLinear with grouped dequant.
        n_groups = self.args.o_groups
        for l in range(n_layers):
            prefix = f"model.layers.{l}.attn.wo_a"
            if f"{prefix}.0.weight" in weights:
                for suffix in ("weight", "biases", "scales"):
                    parts = []
                    for g in range(n_groups):
                        key = f"{prefix}.{g}.{suffix}"
                        if key in weights:
                            parts.append(weights.pop(key))
                    if parts:
                        weights[f"{prefix}.{suffix}"] = mx.concatenate(parts, axis=0)

        # Stack routed expert weights for MTP layers
        if has_mtp:
            for mtp_idx in range(self.args.num_nextn_predict_layers):
                prefix = f"mtp.{mtp_idx}.block.ffn.experts"
                for src, dst in (
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    key0 = f"{prefix}.0.{src}.weight"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.weight")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"mtp.{mtp_idx}.block.ffn.switch_mlp.{dst}.weight"
                        ] = mx.stack(stacked)

        return weights

    # ------------------------------------------------------------------- #
    # Distributed sharding                                                 #
    # ------------------------------------------------------------------- #

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()
        for layer in self.model.layers:
            a = layer.attn
            a.wq_b = shard_linear(a.wq_b, "all-to-sharded", group=group)
            a.wo_b = shard_linear(a.wo_b, "sharded-to-all", group=group)
            a.n_heads //= N
            # Slice attn_sink to local heads (mirrors gpt_oss.py:308-312).
            # Order matters: must run AFTER `a.n_heads //= N` so the stride is
            # the post-division (local) head count.
            a.attn_sink = a.attn_sink[a.n_heads * R : a.n_heads * (R + 1)]

            # wo_a: shape (n_groups * o_lora_rank, group_feat).
            # group_feat = n_heads * v_head_dim / n_groups. After sharding,
            # n_heads //= N and n_groups //= N cancel in the ratio, so group_feat
            # stays constant. Only the OUTPUT dim (n_groups axis) gets sharded —
            # each rank owns n_groups//N consecutive groups.
            # wo_b is "sharded-to-all" so its input = n_groups_local * o_lora_rank.
            old_n_groups = a.n_groups
            new_n_groups = old_n_groups // N
            gs = new_n_groups * R
            ge = new_n_groups * (R + 1)
            if isinstance(a.wo_a, nn.QuantizedLinear):
                gf = a.wo_a.weight.shape[-1]
                w = a.wo_a.weight.reshape(old_n_groups, a.o_lora_rank, gf)
                a.wo_a.weight = w[gs:ge].reshape(new_n_groups * a.o_lora_rank, gf)
                sc_gf = a.wo_a.scales.shape[-1]
                s = a.wo_a.scales.reshape(old_n_groups, a.o_lora_rank, sc_gf)
                a.wo_a.scales = s[gs:ge].reshape(new_n_groups * a.o_lora_rank, sc_gf)
                if getattr(a.wo_a, "biases", None) is not None:
                    b_gf = a.wo_a.biases.shape[-1]
                    b = a.wo_a.biases.reshape(old_n_groups, a.o_lora_rank, b_gf)
                    a.wo_a.biases = b[gs:ge].reshape(new_n_groups * a.o_lora_rank, b_gf)
            else:
                gf = a.wo_a.weight.shape[-1]
                w = a.wo_a.weight.reshape(old_n_groups, a.o_lora_rank, gf)
                a.wo_a.weight = w[gs:ge].reshape(new_n_groups * a.o_lora_rank, gf)
            a.n_groups = new_n_groups

            if isinstance(layer.ffn, DeepseekV4MoE):
                layer.ffn.sharding_group = group
                if hasattr(layer.ffn, "shared_experts"):
                    shard_inplace(layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group)
                    shard_inplace(layer.ffn.shared_experts.up_proj,   "all-to-sharded", group=group)
                shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
                shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
                shard_inplace(layer.ffn.switch_mlp.up_proj,   "all-to-sharded", group=group)
