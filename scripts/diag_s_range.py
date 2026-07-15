#!/usr/bin/env python3
"""Diagnostic: does S = Q*K^T overflow Q8.8 range for the golden data?"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

N = D = 64
GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")


def load_hex(name):
    vals = []
    with open(os.path.join(GOLD, name)) as f:
        for line in f:
            v = int(line.strip(), 16)
            if v & 0x8000:
                v -= 0x10000
            vals.append(v)
    return np.array(vals, dtype=np.int64).reshape(N, D)


Q = load_hex("q.hex").astype(np.float64) / 256.0
K = load_hex("k.hex").astype(np.float64) / 256.0

S = Q @ K.T  # true float scores

Q8_8_MAX = 127.99609375
print(f"S stats: min={S.min():.2f} max={S.max():.2f} std={S.std():.2f}")
print(f"Q8.8 representable range: [-128, {Q8_8_MAX:.2f}]")

over = np.abs(S) > 128
print(f"\nElements with |S| > 128 (would wrap): {over.sum()} / {N*D} "
      f"({100*over.mean():.1f}%)")

rows_with_wrap = over.any(axis=1).sum()
print(f"Rows containing at least one wrapped element: {rows_with_wrap} / {N} "
      f"({100*rows_with_wrap/N:.1f}%)")
print(f"  -> a wrapped element can become a false row-max and corrupt that")
print(f"     entire row's softmax (all {D} of its output elements).")

# What input range WOULD keep S safely in Q8.8?
print(f"\nFor reference, |S| percentiles: "
      f"99%={np.percentile(np.abs(S),99):.1f} "
      f"99.9%={np.percentile(np.abs(S),99.9):.1f} "
      f"max={np.abs(S).max():.1f}")
