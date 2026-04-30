"""Decode-cache logit-parity regression test for DeepSeek-V4.

Bug: @anerjy reported on PR #1189 that with-cache autoregressive decode
produces different logits at S=1 than re-prefilling the same token sequence
without cache. Symptom: greedy temp=0 generation corrupts after the first
generated token (e.g. '{"verdict":"yes"}' becomes '{"ver":"yes"}').

This test asserts byte-parity (within tight float tolerance) of the last-position
logits between two paths driving the same token sequence, on a tiny synthetic
DeepSeek-V4 model with random weights:

    Path A (no-cache ground truth):
        for i in range(N):  out_i = model(prompt[:i+1])

    Path B (with-cache):
        cache = make_prompt_cache(model)
        out_0 = model(prompt, cache=cache)            # full prefill
        for k in range(N-prompt_len):
            out_k = model([next_tok], cache=cache)    # single-step decode

    Assert: out_A[i] is close to out_B[i] for every position i in the decode range.

The synthetic config matches the problem-shape: compress_ratios with both 0
(pure window) and 4 (light compression) layers, sliding_window=8 to force
rotating cache wrap-around inside the test. We parameterize the prompt length
so that all 3 branches of RotatingKVCache._update_in_place are exercised:
  - prompt_len=4 < sliding_window: pre-grow phase only
  - prompt_len=8 == sliding_window: trim+grow boundary
  - prompt_len=12 > sliding_window: wrap rotation engaged before any decode

Test FAILS on unpatched mlx-lm PR #1189 HEAD 7d20c1d (per @anerjy bisection).
PASSES once the RotatingKVCache write/read or DeepseekV4 Compressor cache
path is patched.

Run:  pytest tests/test_deepseek_v4_decode_cache.py -v
"""

import mlx.core as mx
import pytest

from mlx_lm.models import deepseek_v4
from mlx_lm.models.cache import make_prompt_cache


def _build_tiny_model(seed: int = 42):
    """Tiny V4 model — 4 layers, mixed compress_ratios to exercise both
    rotating-only (compress_ratio=0) and compressed-KV (compress_ratio=4) layers.
    We deliberately place compress_ratio=4 NOT at the first layer so the bug
    surface (a divergence in any layer) propagates through subsequent layers.

    DETERMINISM CONTRACT: every call seeds the RNG, builds the Model, then
    forces lazy parameter init off that seeded RNG via mx.eval. Two back-to-back
    calls with the same seed produce byte-identical weights, regardless of any
    MLX RNG calls between them — because re-seeding resets RNG state. Test
    bodies that need byte-identical Path A and Path B models should call
    `_build_tiny_model(seed=42)` twice, even if other RNG operations occurred
    in between."""
    mx.random.seed(seed)
    args = deepseek_v4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=1024,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        head_dim=32,
        qk_rope_head_dim=8,
        sliding_window=8,
        compress_ratios=[0, 4, 0, 4],   # mix: layers 1 and 3 are compressed
        index_n_heads=4,
        index_head_dim=16,
        index_topk=8,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        max_position_embeddings=64,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 2,
            "original_max_position_embeddings": 32,
            "type": "yarn",
        },
    )
    model = deepseek_v4.Model(args)
    model.eval()
    # mx.eval triggers lazy init of all parameters with the seeded RNG so
    # both Path A and Path B see byte-identical weights.
    mx.eval(model.parameters())
    return model


def _last_logits(out):
    """Extract last-position logits from a forward pass output (cast to f32 for
    a tolerance-stable comparison; the model's internal precision is f16 in
    several quantization paths)."""
    return out[0, -1].astype(mx.float32)


# Parameterize the prompt length so all 3 _update_in_place branches see traffic:
#   4  -> pre-grow (offset < keys.shape[2] && shape < max_size)
#   8  -> trim/boundary (offset == max_size, _idx == max_size triggers wrap)
#   12 -> post-wrap (offset > max_size, ring buffer in steady state)
@pytest.mark.parametrize("prompt_len", [4, 8, 12])
def test_deepseek_v4_decode_cache_logit_parity(prompt_len):
    """Path A (no-cache, ground truth) vs Path B (with-cache) must produce
    equivalent logits at every decode position.

    On the unpatched buggy branch, divergence appears at step 2 (per @anerjy)
    and propagates from there. We assert through 8 decode steps to catch
    cumulative drift.

    atol=1e-5 is the right tolerance for this comparison: f32 last-layer logits,
    identical f32 weights, identical inputs, no stochastic elements (no dropout,
    no random sampling). Any divergence larger than 1e-5 is a correctness bug,
    not numerical noise. (Larger atol would silently mask small but real drift.)"""
    model_a = _build_tiny_model(seed=42)
    model_b = _build_tiny_model(seed=42)
    # Determinism sanity-check: confirm Path A and Path B builds produced
    # byte-identical weights. If this fails, the rest of the test is invalid.
    from mlx.utils import tree_flatten
    pa = tree_flatten(model_a.parameters())
    pb = tree_flatten(model_b.parameters())
    assert len(pa) == len(pb), "param tree shape mismatch"
    for (na, va), (nb, vb) in zip(pa, pb):
        assert na == nb, f"param name mismatch {na} vs {nb}"
        assert mx.array_equal(va, vb).item(), (
            f"weight {na} differs between two _build_tiny_model(seed=42) calls; "
            f"determinism contract broken — check that no MLX RNG calls happened "
            f"after construction but before mx.eval(model.parameters())"
        )

    # Use a separate seed for the prompt so the prompt is independent of weights.
    mx.random.seed(0)
    prompt = mx.random.randint(low=0, high=1024, shape=(prompt_len,)).tolist()

    # ---- Path A: no-cache ground truth ----
    prefix = list(prompt)
    nc_logits = []
    for _ in range(8):
        out = model_a(mx.array([prefix]))
        last = _last_logits(out)
        nc_logits.append(last)
        prefix.append(int(mx.argmax(last).item()))

    # ---- Path B: with-cache autoregressive ----
    # model_b was already built above (with determinism check). Use it directly.
    cache = make_prompt_cache(model_b)
    out = model_b(mx.array([list(prompt)]), cache=cache)
    last = _last_logits(out)
    wc_logits = [last]
    nxt = int(mx.argmax(last).item())
    for _ in range(7):
        out = model_b(mx.array([[nxt]]), cache=cache)
        last = _last_logits(out)
        wc_logits.append(last)
        nxt = int(mx.argmax(last).item())

    # ---- Assert byte-parity ----
    diff_steps = []
    for k in range(8):
        if not bool(mx.allclose(nc_logits[k], wc_logits[k], atol=1e-5, rtol=1e-5).item()):
            max_abs = float(mx.max(mx.abs(nc_logits[k] - wc_logits[k])).item())
            diff_steps.append((k, max_abs))

    assert not diff_steps, (
        f"decode-cache divergence at prompt_len={prompt_len} step(s) {diff_steps}. "
        f"Path A (no-cache) and Path B (with-cache) must produce equivalent "
        f"last-position logits. Bug per @anerjy on PR #1189."
    )


@pytest.mark.parametrize("prompt_len", [4, 8, 12])
def test_deepseek_v4_decode_cache_argmax_parity(prompt_len):
    """Strict downstream assertion: even if logits diverge by epsilon, the
    argmax-derived token sequence must match between paths. This is the
    exact user-visible symptom @anerjy reported."""
    model_a = _build_tiny_model(seed=42)
    model_b = _build_tiny_model(seed=42)

    mx.random.seed(0)
    prompt = mx.random.randint(low=0, high=1024, shape=(prompt_len,)).tolist()

    # Path A
    prefix = list(prompt)
    nc_tokens = []
    for _ in range(8):
        out = model_a(mx.array([prefix]))
        last = _last_logits(out)
        nxt = int(mx.argmax(last).item())
        nc_tokens.append(nxt)
        prefix.append(nxt)

    # Path B
    cache = make_prompt_cache(model_b)
    out = model_b(mx.array([list(prompt)]), cache=cache)
    last = _last_logits(out)
    wc_tokens = [int(mx.argmax(last).item())]
    for _ in range(7):
        out = model_b(mx.array([[wc_tokens[-1]]]), cache=cache)
        last = _last_logits(out)
        wc_tokens.append(int(mx.argmax(last).item()))

    assert nc_tokens == wc_tokens, (
        f"argmax-token divergence at prompt_len={prompt_len}: "
        f"no-cache picks {nc_tokens}, with-cache picks {wc_tokens}. "
        f"First divergence at index "
        f"{next((i for i, (a, b) in enumerate(zip(nc_tokens, wc_tokens)) if a != b), -1)}."
    )


def test_rotating_kv_cache_meta_state_round_trip_post_wrap():
    """Sentinel for cache.py changes: meta_state must round-trip losslessly
    AND behaviorally — driving the cache past the wrap boundary, saving its
    state via the public save/load API, then continuing the decode on a fresh
    cache with the loaded state, must produce byte-equal subsequent outputs.

    This test exercises the bug zone (post-wrap _update_in_place) AND catches
    silent field-addition bugs (a new instance attribute that's NOT serialized
    in meta_state will diverge after save/load when the cache is driven
    further). The behavioral assertion catches more than a structural
    `meta_state == meta_state` round-trip would.
    """
    from mlx_lm.models.cache import RotatingKVCache, save_prompt_cache, load_prompt_cache
    import tempfile, os

    cache = RotatingKVCache(max_size=8, keep=0)

    # Phase 1: prefill 4 tokens (S=4 → _update_concat path, NOT _update_in_place).
    k_pre = mx.random.uniform(shape=(1, 1, 4, 8))
    v_pre = mx.random.uniform(shape=(1, 1, 4, 8))
    cache.update_and_fetch(k_pre, v_pre)

    # Phase 2: decode 6 single-token steps (S=1 → _update_in_place path),
    # which forces the cache past max_size=8 and triggers wrap-rotation.
    decode_inputs = [
        (mx.random.uniform(shape=(1, 1, 1, 8)), mx.random.uniform(shape=(1, 1, 1, 8)))
        for _ in range(6)
    ]
    for k_step, v_step in decode_inputs:
        cache.update_and_fetch(k_step, v_step)
    mx.eval(cache.keys, cache.values)

    # Save and reload via the public API (mirrors what mlx_lm.server does).
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "round_trip.safetensors")
        save_prompt_cache(path, [cache])
        loaded_caches = load_prompt_cache(path)
        cache2 = loaded_caches[0]

    # Structural parity: meta_state and array contents.
    assert cache2.meta_state == cache.meta_state, (
        f"meta_state mismatch after round-trip. saved={cache.meta_state}, "
        f"loaded={cache2.meta_state}."
    )
    assert mx.array_equal(cache.keys, cache2.keys).item(), "keys drift after round-trip"
    assert mx.array_equal(cache.values, cache2.values).item(), "values drift after round-trip"

    # Behavioral parity: drive both caches forward with the same next step
    # and assert the returned slices are byte-equal. If a NEW field was added
    # to the class but not serialized, this is where the divergence appears.
    k_next = mx.random.uniform(shape=(1, 1, 1, 8))
    v_next = mx.random.uniform(shape=(1, 1, 1, 8))
    out1_k, out1_v = cache.update_and_fetch(k_next, v_next)
    out2_k, out2_v = cache2.update_and_fetch(k_next, v_next)
    mx.eval(out1_k, out1_v, out2_k, out2_v)
    assert mx.array_equal(out1_k, out2_k).item(), (
        "post-load decode step diverges from pre-save: a RotatingKVCache field "
        "was likely added without updating meta_state. Audit the class __init__ "
        "and ensure every persistent field is in the meta_state tuple."
    )
    assert mx.array_equal(out1_v, out2_v).item(), "values diverge — same root cause"

    # Sanity check on nbytes (server LRU eviction depends on this).
    assert cache.nbytes == cache2.nbytes, (
        f"nbytes drift after round-trip: pre={cache.nbytes}, post={cache2.nbytes}. "
        f"Server LRU eviction would mis-account."
    )
