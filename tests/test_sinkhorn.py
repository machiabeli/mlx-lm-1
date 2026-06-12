# Copyright © 2025 Apple Inc.

"""Unit tests for ``mlx_lm.models.sinkhorn``.

The fused Metal Sinkhorn kernel currently produces incorrect results
(hidden-state std explodes across DeepSeek-V4's 43 layers; PR #1189
review). It is therefore disabled by default behind the
``MLX_LM_HC_SINKHORN_KERNEL=1`` opt-in. These tests pin the default
(pure-MLX) path: shapes are correct, and the produced ``comb`` matrix
satisfies the doubly-stochastic property (rows and columns each sum to
~1). If the kernel default is silently flipped on before the kernel is
re-validated, ``test_comb_is_doubly_stochastic`` will fail.
"""

import os
import unittest

import mlx.core as mx

from mlx_lm.models.sinkhorn import hc_split_sinkhorn


def _make_inputs(n: int = 8, hc_mult: int = 4, seed: int = 0):
    """Build a (mixes, scale, base) triple matching ``hc_split_sinkhorn``'s
    expected layout: mixes is ``[N, (2 + hc_mult) * hc_mult]`` fp32.
    """
    mx.random.seed(seed)
    feat_dim = (2 + hc_mult) * hc_mult
    mixes = mx.random.normal(shape=(n, feat_dim)).astype(mx.float32)
    scale = mx.array([1.0, 1.0, 1.0], dtype=mx.float32)
    base = mx.zeros((feat_dim,), dtype=mx.float32)
    return mixes, scale, base


class TestHcSplitSinkhorn(unittest.TestCase):
    def setUp(self):
        # Pin the default (kernel-off) path for the duration of each test.
        # The kernel is opt-in via MLX_LM_HC_SINKHORN_KERNEL=1; we make sure
        # the env var is unset so we exercise the verified pure-MLX path.
        os.environ.pop("MLX_LM_HC_SINKHORN_KERNEL", None)

    def test_shapes(self):
        n, hc = 8, 4
        mixes, scale, base = _make_inputs(n=n, hc_mult=hc)
        pre, post, comb = hc_split_sinkhorn(
            mixes, scale, base, hc_mult=hc, sinkhorn_iters=20
        )
        self.assertEqual(pre.shape, (n, hc))
        self.assertEqual(post.shape, (n, hc))
        self.assertEqual(comb.shape, (n, hc, hc))

    def test_comb_is_doubly_stochastic(self):
        """Rows AND columns of ``comb`` should each sum to ~1 after Sinkhorn.

        This catches the kernel std-explosion regression: if the kernel path
        is re-enabled without the fix, row/col sums drift far from 1 and
        this test fails.
        """
        n, hc, iters = 16, 4, 20
        mixes, scale, base = _make_inputs(n=n, hc_mult=hc, seed=1)
        _, _, comb = hc_split_sinkhorn(
            mixes, scale, base, hc_mult=hc, sinkhorn_iters=iters
        )
        row_sums = comb.sum(axis=2)  # [N, hc]
        col_sums = comb.sum(axis=1)  # [N, hc]
        mx.eval(row_sums, col_sums)
        # Eps-padding (1e-6) per row/col accumulates a bounded error; 1e-3
        # is generous and still catches the kernel's divergence (which
        # drifts by orders of magnitude).
        tol = 1e-3
        max_row_err = float(mx.max(mx.abs(row_sums - 1.0)).item())
        max_col_err = float(mx.max(mx.abs(col_sums - 1.0)).item())
        self.assertLess(
            max_row_err, tol,
            f"row sums diverge from 1 by {max_row_err} (tol {tol})",
        )
        self.assertLess(
            max_col_err, tol,
            f"col sums diverge from 1 by {max_col_err} (tol {tol})",
        )


if __name__ == "__main__":
    unittest.main()
