# Copyright © 2025 Apple Inc.

"""Unit tests for ``CompressedKVCache`` batched merge/extend validation.

PR #1189 review reported "request #2 reads stale pool rows" in batched
serving: the previous merge/extend implementations zero-padded mismatched
``_pool``/``_buf`` lengths before concatenating, which silently
interleaved padding with subsequently-appended decode tokens. The fix
requires synchronized state across input caches and fails loudly on
mismatch rather than corrupting decode output. These tests pin that
contract: every mismatch path raises ``NotImplementedError`` with a
diagnostic that points at the offending field, and synchronized inputs
go through unchanged.
"""

import unittest
from unittest.mock import MagicMock

import mlx.core as mx

from mlx_lm.models.deepseek_v4 import CompressedKVCache


def _fresh_cache() -> CompressedKVCache:
    """Return a ``CompressedKVCache`` whose ``.local`` is a ``MagicMock``.

    The mock lets us exercise the batched-validation logic without
    standing up a real ``RotatingKVCache`` — ``merge``/``extend``/etc. on
    ``.local`` resolve to inert mock calls, so anything they do is
    transparent to the assertions below.
    """
    c = CompressedKVCache()
    c.local = MagicMock()
    return c


class TestCompressedKVCacheBatchedValidation(unittest.TestCase):
    # --- merge --------------------------------------------------------------

    def test_merge_two_fresh_caches_succeeds(self):
        """Synchronized empty caches merge without raising."""
        c1, c2 = _fresh_cache(), _fresh_cache()
        merged = CompressedKVCache.merge([c1, c2])
        self.assertIsNone(merged._pool)
        self.assertIsNone(merged._buf)
        self.assertEqual(merged._buf_count, 0)

    def test_merge_raises_on_mismatched_pool_lens(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._pool = mx.zeros((1, 4, 8))
        c2._pool = mx.zeros((1, 7, 8))
        with self.assertRaises(NotImplementedError) as ctx:
            CompressedKVCache.merge([c1, c2])
        self.assertIn("pool_lens", str(ctx.exception))

    def test_merge_raises_on_pool_none_vs_set(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._pool = mx.zeros((1, 4, 8))
        # c2._pool stays None
        with self.assertRaises(NotImplementedError):
            CompressedKVCache.merge([c1, c2])

    def test_merge_raises_on_mismatched_buf_lens(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._buf = mx.zeros((1, 3, 8))
        c2._buf = mx.zeros((1, 5, 8))
        c1._buf_count = c2._buf_count = 0
        with self.assertRaises(NotImplementedError) as ctx:
            CompressedKVCache.merge([c1, c2])
        self.assertIn("buf_lens", str(ctx.exception))

    def test_merge_raises_on_mismatched_buf_counts(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._buf = mx.zeros((1, 3, 8))
        c2._buf = mx.zeros((1, 3, 8))
        c1._buf_count = 2
        c2._buf_count = 3
        with self.assertRaises(NotImplementedError) as ctx:
            CompressedKVCache.merge([c1, c2])
        self.assertIn("buf_counts", str(ctx.exception))

    # --- extend -------------------------------------------------------------

    def test_extend_two_fresh_caches_succeeds(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1.extend(c2)
        self.assertIsNone(c1._pool)
        self.assertIsNone(c1._buf)

    def test_extend_raises_on_mismatched_pool_lens(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._pool = mx.zeros((1, 4, 8))
        c2._pool = mx.zeros((1, 7, 8))
        with self.assertRaises(NotImplementedError) as ctx:
            c1.extend(c2)
        self.assertIn("pool_len", str(ctx.exception))

    def test_extend_raises_on_pool_none_vs_set(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._pool = mx.zeros((1, 4, 8))
        with self.assertRaises(NotImplementedError):
            c1.extend(c2)

    def test_extend_raises_on_mismatched_buf_lens(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._buf = mx.zeros((1, 3, 8))
        c2._buf = mx.zeros((1, 5, 8))
        c1._buf_count = c2._buf_count = 0
        with self.assertRaises(NotImplementedError) as ctx:
            c1.extend(c2)
        self.assertIn("buf_len", str(ctx.exception))

    def test_extend_raises_on_mismatched_buf_counts(self):
        c1, c2 = _fresh_cache(), _fresh_cache()
        c1._buf = mx.zeros((1, 3, 8))
        c2._buf = mx.zeros((1, 3, 8))
        c1._buf_count = 2
        c2._buf_count = 3
        with self.assertRaises(NotImplementedError) as ctx:
            c1.extend(c2)
        self.assertIn("buf_count", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
