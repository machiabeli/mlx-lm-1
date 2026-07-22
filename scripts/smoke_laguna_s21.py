#!/usr/bin/env python3
"""Smoke-load Poolside Laguna S 2.1 MLX quant with vendored mlx_lm laguna.py.

Usage:
  PYTHONPATH=/path/to/mlx-lm python3 scripts/smoke_laguna_s21.py \
      --model ~/Models/Laguna-S-2.1-MLX-4bit \
      [--prompt "Write a Python hello world"] \
      [--max-tokens 64]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local path to MLX quant dir")
    ap.add_argument("--prompt", default="Write a one-line Python hello world.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        return 2
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: no config.json in {model_path}", file=sys.stderr)
        return 2
    cfg = json.loads(cfg_path.read_text())
    print(f"model_type={cfg.get('model_type')} layers={cfg.get('num_hidden_layers')} "
          f"experts={cfg.get('num_experts')} topk={cfg.get('num_experts_per_tok')} "
          f"quant={cfg.get('quantization', {}).get('bits') if isinstance(cfg.get('quantization'), dict) else cfg.get('quantization')}")

    # Prefer the PYTHONPATH-vendored mlx_lm (with laguna.py) over site-packages.
    from mlx_lm import load, generate
    from mlx_lm.models import laguna as _laguna  # noqa: F401 — force import fail-fast

    print(f"laguna module: {_laguna.__file__}")
    t0 = time.perf_counter()
    print(f"loading {model_path} ...")
    model, tokenizer = load(str(model_path))
    t_load = time.perf_counter() - t0
    print(f"loaded in {t_load:.1f}s")

    # One short generation
    t1 = time.perf_counter()
    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=True,
        temp=args.temp,
    )
    t_gen = time.perf_counter() - t1
    print("---")
    print(text)
    print(f"--- gen wall {t_gen:.2f}s for up to {args.max_tokens} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
