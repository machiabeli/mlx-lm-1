# Copyright © 2025 Apple Inc.
"""Kimi-K3 support in kimi_linear.py.

K3 reuses `model_type: "kimi_linear"`, so mlx-lm dispatches both models to the
same module. That makes two things worth testing explicitly: the K3-only
features actually engage, and a plain Kimi-Linear config still behaves exactly
as it did before (a silent regression there would be invisible — the model
would load and emit plausible garbage rather than raise).
"""

import unittest

import mlx.core as mx

from mlx_lm.models.kimi_linear import (
    ModelArgs,
    Model,
    _fuse_mxfp4,
    situ,
)


def _base_config(**overrides):
    cfg = dict(
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
        linear_attn_config=dict(
            num_heads=8,
            head_dim=32,
            short_conv_kernel_size=4,
            kda_layers=[1, 2, 4, 5],
            full_attn_layers=[3, 6],
        ),
    )
    # A None override means "this key is absent from config.json", which is
    # exactly how a real Kimi-Linear checkpoint disables a K3 feature.
    for k, v in overrides.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    return cfg


def _k3_config(**overrides):
    """A K3-shaped config: every K3-only feature switched on."""
    k3 = dict(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=3,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=True,
        mla_use_output_gate=True,
        q_lora_rank=64,
    )
    k3.update(overrides)
    return _base_config(**k3)


class TestKimiK3(unittest.TestCase):
    def test_config_without_rope_theta_or_model_max_length(self):
        """K3's config.json ships neither, and head_dim is absent too.

        These were required fields with no default, so `from_dict` raised
        TypeError on the real checkpoint before it ever built a layer.
        """
        cfg = _k3_config()
        for absent in ("rope_theta", "model_max_length", "head_dim"):
            self.assertNotIn(absent, cfg)
        args = ModelArgs.from_dict(cfg)
        # head_dim is derived from the MLA nope split, not guessed.
        self.assertEqual(args.head_dim, cfg["qk_nope_head_dim"])

    def test_k3_features_are_off_by_default(self):
        """A plain Kimi-Linear config must not pick up any K3 behaviour."""
        args = ModelArgs.from_dict(_base_config())
        self.assertIsNone(args.attn_res_block_size)
        self.assertIsNone(args.routed_expert_hidden_size)
        self.assertEqual(args.hidden_act, "silu")
        self.assertFalse(args.mla_use_output_gate)
        self.assertFalse(args.latent_moe_use_norm)

    def test_kimi_linear_still_works(self):
        args = ModelArgs.from_dict(_base_config(head_dim=32, rope_theta=10000.0))
        model = Model(args)
        model.eval()
        out = model(mx.array([[1, 2, 3, 4, 5, 6]]))
        mx.eval(out)
        self.assertEqual(out.shape, (1, 6, args.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(out)).item())

    def test_k3_forward(self):
        args = ModelArgs.from_dict(_k3_config())
        model = Model(args)
        model.eval()
        out = model(mx.array([[1, 2, 3, 4, 5]]))
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, args.vocab_size))
        self.assertTrue(mx.all(mx.isfinite(out)).item())

    def _decode_parity(self, args, tokens=(1, 2, 3, 4, 5, 6, 7, 8)):
        model = Model(args)
        model.eval()
        full = model(mx.array([list(tokens)]))
        mx.eval(full)
        cache = model.make_cache()
        step = None
        for t in tokens:
            step = model(mx.array([[t]]), cache=cache)
            mx.eval(step)
        denom = mx.max(mx.abs(full[:, -1, :])).item()
        return mx.max(mx.abs(full[:, -1, :] - step[:, -1, :])).item() / denom

    def test_k3_prefill_matches_incremental_decode(self):
        """The attention-residual stack is stateful across layers AND tokens.

        A mistake in "push a summary every attn_res_block_size layers, then
        restart prefix_sum" shows up here as gross divergence rather than a
        rounding difference — this is the test a port written before the
        weights shipped could not have passed.
        """
        rel = self._decode_parity(ModelArgs.from_dict(_k3_config()))
        self.assertLess(rel, 1e-4, f"prefill/decode diverged: rel={rel:.3e}")

    def test_kimi_linear_prefill_matches_incremental_decode(self):
        args = ModelArgs.from_dict(_base_config(head_dim=32, rope_theta=10000.0))
        rel = self._decode_parity(args)
        self.assertLess(rel, 1e-4, f"prefill/decode diverged: rel={rel:.3e}")

    def test_attn_residual_path_is_actually_taken(self):
        """Guard against the residual path being wired but never executed.

        Rather than compare two differently-shaped models, perturb the
        residual projections: if the mechanism runs, the logits must move.
        A no-op path would leave them bit-identical.
        """
        model = Model(ModelArgs.from_dict(_k3_config()))
        model.eval()
        x = mx.array([[1, 2, 3, 4, 5]])
        before = model(x)
        mx.eval(before)

        for layer in model.layers:
            layer.mlp_res_proj.weight = layer.mlp_res_proj.weight + 5.0
            layer.self_attention_res_proj.weight = (
                layer.self_attention_res_proj.weight + 5.0
            )
        after = model(x)
        mx.eval(after)

        self.assertFalse(mx.allclose(before, after, atol=1e-5).item())

    def test_layers_expose_residual_modules_only_for_k3(self):
        k3 = Model(ModelArgs.from_dict(_k3_config()))
        self.assertTrue(all(l.use_attn_residuals for l in k3.layers))
        self.assertTrue(hasattr(k3.layers[0], "mlp_res_proj"))

        plain = Model(
            ModelArgs.from_dict(_base_config(head_dim=32, rope_theta=10000.0))
        )
        self.assertFalse(any(l.use_attn_residuals for l in plain.layers))
        self.assertFalse(hasattr(plain.layers[0], "mlp_res_proj"))

    def test_situ_activation_matches_reference(self):
        """beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(up/linear_beta)."""
        g = mx.array([[-3.0, -0.5, 0.0, 0.5, 3.0]])
        u = mx.array([[1.0, 2.0, -1.0, 0.25, -4.0]])
        beta, lin = 4.0, 25.0

        got = situ(g, u, beta, lin)
        want = (
            beta
            * mx.tanh(g / beta)
            * mx.sigmoid(g)
            * (lin * mx.tanh(u / lin))
        )
        mx.eval(got, want)
        self.assertTrue(mx.allclose(got, want, atol=1e-6).item())

    def test_situ_without_linear_beta_leaves_up_untouched(self):
        g = mx.array([[0.5, -0.5]])
        u = mx.array([[2.0, 3.0]])
        got = situ(g, u, 1.0, None)
        want = mx.tanh(g) * mx.sigmoid(g) * u
        mx.eval(got, want)
        self.assertTrue(mx.allclose(got, want, atol=1e-6).item())

    def test_fuse_mxfp4_repacks_uint8_to_uint32(self):
        """compressed-tensors ships uint8 codes; MLX's mxfp4 wants uint32."""
        rows, groups = 4, 2
        packed = mx.random.randint(
            0, 256, (rows, groups * 16), dtype=mx.uint32
        ).astype(mx.uint8)
        scales = mx.full((rows, groups), 127, dtype=mx.uint8)
        weights = {
            "layers.0.w1.weight_packed": packed,
            "layers.0.w1.weight_scale": scales,
        }
        out = _fuse_mxfp4(dict(weights))

        self.assertEqual(set(out), {"layers.0.w1.weight", "layers.0.w1.scales"})
        w = out["layers.0.w1.weight"]
        self.assertEqual(w.dtype, mx.uint32)
        self.assertEqual(w.shape, (rows, groups * 4))

        # Little-endian, low byte first — the order MLX's kernels assume.
        p = packed.astype(mx.uint32)
        expect0 = (
            p[0, 0] | (p[0, 1] << 8) | (p[0, 2] << 16) | (p[0, 3] << 24)
        ).item()
        self.assertEqual(w[0, 0].item(), expect0)

        deq = mx.dequantize(
            w, scales=out["layers.0.w1.scales"], group_size=32, bits=4, mode="mxfp4"
        )
        mx.eval(deq)
        self.assertEqual(deq.shape, (rows, groups * 32))
        self.assertTrue(mx.all(mx.isfinite(deq)).item())

    def test_fuse_mxfp4_is_a_noop_without_packed_weights(self):
        weights = {"model.norm.weight": mx.ones((4,))}
        self.assertEqual(set(_fuse_mxfp4(dict(weights))), set(weights))

    def test_fuse_mxfp4_fails_closed_on_missing_scale(self):
        """Packed codes with no scale cannot be decoded — raise, don't guess."""
        weights = {"layers.0.w1.weight_packed": mx.zeros((2, 16), dtype=mx.uint8)}
        with self.assertRaises(ValueError):
            _fuse_mxfp4(weights)

    def test_sanitize_strips_language_model_prefix(self):
        """K3 nests the text model under `language_model.`; the vision tower
        is a sibling and must not reach the text module tree."""
        args = ModelArgs.from_dict(_k3_config())
        model = Model(args)
        weights = {
            "language_model.model.norm.weight": mx.ones((args.hidden_size,)),
            "vision_tower.encoder.layers.0.weight": mx.ones((4, 4)),
        }
        out = model.sanitize(weights)
        self.assertIn("model.norm.weight", out)
        self.assertNotIn("vision_tower.encoder.layers.0.weight", out)

    def test_latent_moe_projections_are_built(self):
        args = ModelArgs.from_dict(_k3_config())
        moe = None
        for layer in Model(args).layers:
            if hasattr(layer.mlp, "switch_mlp"):
                moe = layer.mlp
                break
        self.assertIsNotNone(moe)
        self.assertTrue(moe.use_latent)
        self.assertIsNotNone(moe.routed_expert_norm)
        # Experts live in the latent space (128), not hidden_size (256), and
        # the FFN width is moe_intermediate_size (64) — three distinct dims.
        self.assertEqual(moe.routed_expert_down_proj.weight.shape, (128, 256))
        self.assertEqual(moe.routed_expert_up_proj.weight.shape, (256, 128))
        self.assertEqual(moe.switch_mlp.gate_proj.weight.shape[1:], (64, 128))


if __name__ == "__main__":
    unittest.main()
