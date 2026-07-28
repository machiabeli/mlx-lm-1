"""Verify tensor-parallel output matches single-node, bit-for-bit-ish.

Run under `mlx.launch`. Rank 0 builds the model, computes the single-node
reference, then every rank shards a copy of the SAME weights and recomputes.
If the shard map is wrong the logits diverge grossly -- a partially-sharded
model still runs and still emits plausible numbers, which is exactly the
failure mode a shape check cannot catch.
"""
import json
import sys

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, "/Users/ma/Projects/OpenSource/mlx-lm")
from mlx_lm.models.kimi_linear import Model, ModelArgs  # noqa: E402

CFG = dict(
    model_type="kimi_linear",
    vocab_size=512,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=8,
    intermediate_size=512,
    rms_norm_eps=1e-5,
    num_experts=8,
    moe_intermediate_size=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    mla_use_nope=True,
    num_experts_per_token=2,
    num_shared_experts=1,
    first_k_dense_replace=1,
    hidden_act="situ",
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    attn_res_block_size=3,
    routed_expert_hidden_size=128,
    latent_moe_use_norm=True,
    mla_use_output_gate=True,
    q_lora_rank=64,
    linear_attn_config=dict(
        num_heads=8,
        head_dim=32,
        short_conv_kernel_size=4,
        kda_layers=[1, 2, 4, 5],
        full_attn_layers=[3, 6],
        use_full_rank_gate=True,
    ),
)

group = mx.distributed.init()
rank, size = group.rank(), group.size()

# Identical weights on every rank: same seed, same construction order.
mx.random.seed(1234)
ref_model = Model(ModelArgs.from_dict(CFG))
ref_model.eval()
weights = dict(tree_flatten(ref_model.parameters()))

x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

ref = ref_model(x)
mx.eval(ref)

# Fresh model, same weights, then shard.
mx.random.seed(1234)
tp_model = Model(ModelArgs.from_dict(CFG))
tp_model.update(tree_unflatten(list(weights.items())))
tp_model.eval()
tp_model.shard(group)
mx.eval(tp_model.parameters())

got = tp_model(x)
mx.eval(got)

denom = float(mx.max(mx.abs(ref)))
rel = float(mx.max(mx.abs(ref - got))) / denom

if rank == 0:
    print(json.dumps({
        "world_size": size,
        "rel_max_diff": rel,
        "verdict": "MATCH" if rel < 1e-3 else "DIVERGE",
        "ref_absmax": denom,
    }, indent=2))
