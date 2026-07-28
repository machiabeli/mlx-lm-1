# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .mla import MultiLinear
from .switch_layers import SwiGLU, SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    linear_attn_config: Dict[str, Any]
    num_experts: int
    moe_intermediate_size: int
    kv_lora_rank: int
    # Kimi-K3 ships none of these three in config.json, so they cannot stay
    # required -- from_dict would raise TypeError on the real checkpoint.
    # head_dim is derivable from the MLA split; K3 is NoPE so rope_theta is
    # unused; model_max_length only bounds the default context.
    head_dim: int = 0
    rope_theta: float = 10000.0
    model_max_length: int = 32768
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = False
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1

    # ── Kimi-K3 additions ────────────────────────────────────────────────
    # K3 reuses model_type "kimi_linear", so this module must handle both.
    # Every field below is absent in a Kimi-Linear config and the default
    # reproduces the old behaviour exactly.
    hidden_act: str = "silu"
    activation_situ_beta: float = 1.0
    activation_situ_linear_beta: Optional[float] = None
    #  Attention residuals: layers summarise the residual stream every
    #  attn_res_block_size layers and later layers attend over the summaries.
    attn_res_block_size: Optional[int] = None
    #  Latent MoE: experts operate in a routed_expert_hidden_size space that
    #  differs from hidden_size (3584 vs 7168 on K3), with down/up projections
    #  either side. NOTE this is NOT moe_intermediate_size (3072).
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    mla_use_output_gate: bool = False
    q_lora_rank: Optional[int] = None

    def __post_init__(self):
        if not self.head_dim:
            nope = self.qk_nope_head_dim or 0
            self.head_dim = nope or (self.hidden_size // self.num_attention_heads)


def situ(gate: mx.array, up: mx.array, beta: float,
         linear_beta: Optional[float]) -> mx.array:
    """Kimi-K3's activation: beta*tanh(g/beta)*sigmoid(g) * up.

    Reference: SituAndMul in modeling_kimi_linear.py. The reference upcasts to
    float32 before the tanh/sigmoid and casts back at the end -- tanh(g/4)
    saturates and sigmoid underflows in bf16, so keep the upcast.
    """
    dtype = gate.dtype
    g = gate.astype(mx.float32)
    u = up.astype(mx.float32)
    a = beta * mx.tanh(g / beta) * mx.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * mx.tanh(u / linear_beta)
    return (a * u).astype(dtype)


class KimiMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.use_situ = args.hidden_act == "situ"
        self.situ_beta = args.activation_situ_beta
        self.situ_linear_beta = args.activation_situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        g, u = self.gate_proj(x), self.up_proj(x)
        if self.use_situ:
            return self.down_proj(situ(g, u, self.situ_beta, self.situ_linear_beta))
        return self.down_proj(swiglu(g, u))


@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates)
    elif score_function == "softmax":
        scores = mx.softmax(gates, axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class KimiSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        if experts is None:
            raise ValueError("num_experts must be specified for MoE layers")

        self.gate = nn.Linear(hidden, experts, bias=False)

        # Latent MoE (K3): experts live in routed_expert_hidden_size space, not
        # hidden_size, with down/up projections either side. On K3 that is
        # 3584 vs 7168 -- and it is NOT moe_intermediate_size (3072), which
        # remains the per-expert FFN width.
        latent_dim = args.routed_expert_hidden_size
        self.use_latent = latent_dim is not None
        expert_in: int = latent_dim if latent_dim is not None else hidden

        activation = SwiGLU()
        if args.hidden_act == "situ":
            beta = args.activation_situ_beta
            lin = args.activation_situ_linear_beta
            activation = lambda g, u: situ(g, u, beta, lin)  # noqa: E731

        self.switch_mlp = SwitchGLU(
            expert_in, args.moe_intermediate_size, experts, activation=activation
        )
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if self.use_latent:
            self.routed_expert_down_proj = nn.Linear(hidden, expert_in, bias=False)
            self.routed_expert_up_proj = nn.Linear(expert_in, hidden, bias=False)
            self.routed_expert_norm = (
                nn.RMSNorm(expert_in, eps=args.rms_norm_eps)
                if args.latent_moe_use_norm
                else None
            )

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        # The router always scores the FULL hidden state, before any latent
        # down-projection (reference: gate() is called on the input).
        scores = self.gate(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        h = self.routed_expert_down_proj(x) if self.use_latent else x
        out = self.switch_mlp(h, inds)
        out = (out * weights[..., None]).sum(axis=-2)
        if self.use_latent:
            if self.routed_expert_norm is not None:
                out = self.routed_expert_norm(out)
            out = self.routed_expert_up_proj(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class KimiMLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        # DeepSeek-style low-rank query path. Kimi-Linear projects q densely;
        # K3 sets q_lora_rank (1536) and factors it through a normed
        # bottleneck, so the checkpoint carries q_a_proj/q_a_layernorm/q_b_proj
        # and NO q_proj.
        self.q_lora_rank = args.q_lora_rank
        if self.q_lora_rank:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

        # K3 gates the attention output on a sigmoid of the layer input before
        # o_proj (reference: KimiMLAAttention.forward, `use_output_gate`).
        self.use_output_gate = args.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                hidden, self.num_heads * self.v_head_dim, bias=False
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_output_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])

        return out, new_state


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        # KDA's output gate comes in two forms. Kimi-Linear factors it through
        # a head_dim bottleneck (g_a_proj -> g_b_proj); K3 sets
        # linear_attn_config.use_full_rank_gate and ships one dense g_proj
        # instead. Supporting only the low-rank pair leaves g_proj unconsumed
        # and both gate weights missing on a real K3 checkpoint.
        self.use_full_rank_gate = bool(cfg.get("use_full_rank_gate", False))
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.expand_dims(
            mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))),
            (0, 1, 3),
        )
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = None
            k_state = None
            v_state = None
            ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state = s
            k_state = s
            v_state = s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        inv_scale = self.scale
        # The reference uses l2norm: x / sqrt(sum(x^2) + 1e-6). mx.fast.rms_norm
        # adds eps to MEAN(x^2), so the equivalent epsilon is 1e-6 / head_dim --
        # a plain 1e-6 here is head_dim (128) times too large, and the error
        # compounds through the delta-rule recurrence across all 69 KDA layers.
        # Matches upstream mlx-lm#1624.
        eps = 1e-6 / self.head_dim
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
        k = inv_scale * mx.fast.rms_norm(k, None, eps)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        gate = (
            self.g_proj(x)
            if self.use_full_rank_gate
            else self.g_b_proj(self.g_a_proj(x))
        ).reshape(B, T, self.num_heads, self.head_dim)
        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


def _apply_attn_res(
    prefix_sum: mx.array,
    block_residual: mx.array,
    proj: nn.Linear,
    norm: nn.RMSNorm,
    eps: float,
) -> mx.array:
    """Attend over the stack of per-block residual summaries.

    prefix_sum:     (B, L, H)      the running residual
    block_residual: (B, L, N, H)   N summaries captured every attn_res_block_size

    Reference `_apply_attn_res` in modeling_kimi_linear.py. Two details that
    matter: the whole reduction runs in float32 (the softmax is over N+1
    candidates and bf16 loses the ordering), and the RMS norm weight is folded
    into the projection weight rather than applied as a separate norm --
    `score_weight = norm.weight * proj.weight`, so this is one fused scoring
    pass, not norm-then-project.
    """
    v = mx.concatenate([block_residual, prefix_sum[..., None, :]], axis=-2)
    v32 = v.astype(mx.float32)
    variance = mx.mean(mx.square(v32), axis=-1, keepdims=True)
    k = v32 * mx.rsqrt(variance + eps)
    score_weight = norm.weight.astype(mx.float32) * proj.weight.reshape(-1).astype(
        mx.float32
    )
    scores = mx.sum(k * score_weight, axis=-1)
    probs = mx.softmax(scores, axis=-1, precise=True)[..., None, :]
    out = (probs @ v32).squeeze(-2)
    return out.astype(v.dtype)


#: MXFP4 packs two 4-bit codes per byte and shares one E8M0 scale byte across
#: 32 elements. Both are fixed by the format, not by the checkpoint.
_MXFP4_GROUP_SIZE = 32
_MXFP4_BITS = 4


def _fuse_mxfp4(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Turn compressed-tensors `weight_packed`/`weight_scale` pairs into the
    `weight`/`scales` pair MLX's quantized layers expect.

    K3 ships MXFP4 in the compressed-tensors layout: `w.weight_packed` is uint8
    with two nibbles per byte, `w.weight_scale` is one uint8 E8M0 exponent per
    32-element group. MLX's mxfp4 kernels want the codes as uint32, so the only
    transform needed is a little-endian repack of 4 bytes into 1 word -- the
    nibble order already matches (verified bit-exact against an independent
    E2M1/E8M0 decode: 0 mismatches over 11M values).

    Repacking rather than dequantizing is what keeps K3 at its native 1561 GB;
    materializing bf16 would be roughly 4x that and would not fit the mesh.
    """
    packed_keys = [k for k in weights if k.endswith(".weight_packed")]
    if not packed_keys:
        return weights

    for pk in packed_keys:
        base = pk[: -len(".weight_packed")]
        sk = f"{base}.weight_scale"
        scale = weights.pop(sk, None)
        p = weights.pop(pk)
        if scale is None:
            # A packed tensor with no scale cannot be decoded; failing here is
            # far cheaper to diagnose than garbage logits later.
            raise ValueError(f"{pk} has no matching {sk}")
        if p.dtype == mx.uint8:
            u = p.reshape(*p.shape[:-1], -1, 4).astype(mx.uint32)
            p = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)
        weights[f"{base}.weight"] = p
        weights[f"{base}.scales"] = scale
    return weights


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiMLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiSparseMoE(args)
        else:
            self.mlp = KimiMLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.layer_idx = layer_idx
        self.eps = args.rms_norm_eps
        self.attn_res_block_size = args.attn_res_block_size or 0
        self.use_attn_residuals = args.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.self_attention_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        block_residual: Optional[mx.array] = None,
    ) -> Any:
        attn_cache = None if cache is None else cache

        if not self.use_attn_residuals:
            y = self.self_attn(self.input_layernorm(x), mask, attn_cache)
            h = x + y
            z = self.mlp(self.post_attention_layernorm(h))
            return h + z

        # K3 attention-residual path. `prefix_sum` is the running residual;
        # every attn_res_block_size layers it is PUSHED onto block_residual and
        # set aside (None), so the block's own contribution restarts from the
        # attention output instead of accumulating across the boundary.
        prefix_sum: Optional[mx.array] = x
        h = x
        if block_residual is not None and block_residual.shape[-2] > 0:
            h = _apply_attn_res(
                x,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.eps,
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            summary = x[..., None, :]
            block_residual = (
                summary
                if block_residual is None
                else mx.concatenate([block_residual, summary], axis=-2)
            )
            prefix_sum = None

        y = self.self_attn(self.input_layernorm(h), mask, attn_cache)
        prefix_sum = y if prefix_sum is None else prefix_sum + y

        # block_residual is non-None here: layer 0 satisfies
        # `layer_idx % block_size == 0` and pushes the first summary, so every
        # layer reaches this point with at least one entry on the stack.
        assert block_residual is not None
        h = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.eps,
        )
        z = self.mlp(self.post_attention_layernorm(h))
        return prefix_sum + z, block_residual


class KimiLinearModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [KimiDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        kda_layers = args.linear_attn_config["kda_layers"]
        self.ssm_idx = kda_layers[0] - 1
        for i in range(len(self.layers)):
            if (i + 1) not in kda_layers:
                self.attn_idx = i
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        # K3 threads a growing stack of per-block residual summaries through
        # every layer; Kimi-Linear does not and the loop stays scalar.
        block_residual: Optional[mx.array] = None
        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            out = layer(
                h, mask=mask, cache=layer_cache, block_residual=block_residual
            )
            if isinstance(out, tuple):
                h, block_residual = out
            else:
                h = out

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiLinearModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # K3 checkpoints nest the text model under `language_model.` (the
        # vision tower is a sibling). Strip it so the rest of this function --
        # and the module tree -- sees the Kimi-Linear layout.
        if any(k.startswith("language_model.") for k in weights):
            weights = {
                k[len("language_model.") :]: v
                for k, v in weights.items()
                if k.startswith("language_model.")
            }

        weights = _fuse_mxfp4(weights)
        weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, KimiSparseMoE):
                src_prefix = f"{prefix}.block_sparse_moe"
                dst_prefix = f"{prefix}.mlp"
                for src, dst in [
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ]:
                    # Stack every quantization component present, not just
                    # `weight`: K3 is MXFP4 so each expert also carries
                    # `scales`, and stacking the codes while dropping the
                    # scales yields silent garbage rather than an error.
                    for part in ("weight", "scales", "biases"):
                        key = f"{src_prefix}.experts.0.{src}.{part}"
                        if key not in weights:
                            continue
                        stacked = [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.{part}")
                            for i in range(self.args.num_experts)
                        ]
                        weights[f"{dst_prefix}.switch_mlp.{dst}.{part}"] = mx.stack(
                            stacked
                        )

                for name in ("gate_proj", "up_proj", "down_proj"):
                    src_key = f"{src_prefix}.shared_experts.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.shared_experts.{name}.weight"] = (
                            weights.pop(src_key)
                        )

                # Latent-MoE projections keep their names, but move from the
                # checkpoint's `block_sparse_moe` prefix to our `mlp`.
                for name in (
                    "routed_expert_down_proj",
                    "routed_expert_up_proj",
                    "routed_expert_norm",
                ):
                    for part in ("weight", "scales", "biases"):
                        src_key = f"{src_prefix}.{name}.{part}"
                        if src_key in weights:
                            weights[f"{dst_prefix}.{name}.{part}"] = weights.pop(
                                src_key
                            )

                gate_key = f"{src_prefix}.gate.weight"
                if gate_key in weights:
                    weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

                bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                if bias_key in weights:
                    weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(
                        bias_key
                    )

            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, KimiDeltaAttention):
                attn_prefix = f"{prefix}.self_attn"
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{attn_prefix}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.moveaxis(2, 1)
                        weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
                dt_key = f"{attn_prefix}.dt_bias"
                if dt_key in weights:
                    if weights[dt_key].ndim > 1:
                        weights[dt_key] = mx.reshape(weights[dt_key], (-1,))

            attn_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = self.args.qk_nope_head_dim or self.args.head_dim
                v_head = self.args.v_head_dim or self.args.head_dim
                head_dim = qk_nope + v_head
                num_heads = self.args.num_attention_heads

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
