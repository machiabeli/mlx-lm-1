# Copyright © 2025 Apple Inc.

"""Unit tests for the DeepSeek-V4 S=1 SDPA-with-sinks workaround.

``mx.fast.scaled_dot_product_attention`` with ``sinks=`` produces
~1-ULP different logits for S=1 (decode) vs S>=2 (prefill); the drift
compounds through DeepSeek-V4's 43 layers and flips the greedy argmax
on long-prompt decode (mlx#3452, surfaced in PR #1189 review). The
workaround in ``V4Attention.__call__`` pads the S=1 query to S=2 by
duplicating the row, runs the correct S>=2 kernel path, and slices the
first output row back.

The first test pins the *invariant* the workaround relies on (two
identical query rows produce two identical output rows). The second
test smoke-tests ``V4Attention`` with a true S=1 input to confirm the
padded-and-sliced path runs and yields a finite, correctly-shaped
output. Numerical equivalence to a known-good reference (antirez/ds4)
is the oracle test, deferred to the validation harness once the ds4
GGUF download completes.
"""

import unittest

import mlx.core as mx

from mlx_lm.models import deepseek_v4


class TestSdpaSinksS1Workaround(unittest.TestCase):
    def test_padded_q_rows_are_identical_in_output(self):
        """When q has two identical rows along the sequence axis, SDPA
        with sinks must produce two identical output rows. The S=1
        workaround relies on this invariant: it duplicates q to S=2,
        invokes the (correct) S>=2 kernel, and slices the first row."""
        B, H, D, T = 1, 4, 8, 10
        mx.random.seed(0)
        q_row = mx.random.normal((B, H, 1, D))
        q = mx.concatenate([q_row, q_row], axis=2)  # [B, H, 2, D]
        k = mx.random.normal((B, H, T, D))
        v = mx.random.normal((B, H, T, D))
        sinks = mx.random.normal((H,)).astype(mx.float32)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=0.125, mask=None, sinks=sinks
        )
        # Two identical inputs → two identical outputs (small tolerance
        # for any floating-point non-determinism within the kernel).
        row0 = out[:, :, 0, :]
        row1 = out[:, :, 1, :]
        max_abs_diff = float(mx.max(mx.abs(row0 - row1)).item())
        self.assertLess(
            max_abs_diff, 1e-6,
            f"padded-q rows should be identical; max abs diff {max_abs_diff}",
        )

    def test_v4_attention_s1_decode_runs_and_produces_finite_output(self):
        """End-to-end smoke: V4Attention with an S=1 input goes through
        the workaround branch, returns the expected shape, and yields
        finite values (no NaN/Inf)."""
        args = deepseek_v4.ModelArgs(
            model_type="deepseek_v4",
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=2,
            head_dim=16,
            qk_rope_head_dim=4,
            sliding_window=8,
            compress_ratios=[0, 0],  # no compression on this layer
            index_n_heads=4,
            index_head_dim=4,
            index_topk=4,
            moe_intermediate_size=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=1,
            hc_mult=2,
            hc_sinkhorn_iters=2,
        )
        attn = deepseek_v4.V4Attention(args, layer_idx=0)
        x = mx.random.normal((1, 1, args.hidden_size))  # S=1
        out = attn(x, mask=None, cache=None)
        self.assertEqual(out.shape, (1, 1, args.hidden_size))
        self.assertTrue(mx.all(mx.isfinite(out)).item())


if __name__ == "__main__":
    unittest.main()
