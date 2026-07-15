#!/usr/bin/env python3
"""Attribute the end-to-end error: S-overflow vs A-weight-quantization.

Simulates the IDEAL Q8.8 quantization floor (perfect exp/divide, only the
format's rounding), so we separate 'the Q8.8 format itself is too coarse'
from 'the RTL implementation has extra error'."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

N = D = 64
GOLD = os.path.join(os.path.dirname(__file__), "..", "..", "golden")


def load_hex(name):
    vals = []
    with open(os.path.join(GOLD, name)) as f:
        for line in f:
            v = int(line.strip(), 16)
            if v & 0x8000:
                v -= 0x10000
            vals.append(v)
    return np.array(vals, dtype=np.int64).reshape(N, D)


def q8_8(x):
    """Round-to-nearest quantize to Q8.8, WITH wrap on overflow (like RTL)."""
    raw = np.round(x * 256.0).astype(np.int64)
    raw = ((raw + 32768) & 0xFFFF) - 32768   # wrap into signed 16-bit
    return raw.astype(np.float64) / 256.0


def q1_15(x):
    """Quantize to Q1.15 (15 frac bits) -- for attention weights in [0,1)."""
    raw = np.round(x * 32768.0).astype(np.int64)
    raw = np.clip(raw, 0, 65535)
    return raw.astype(np.float64) / 32768.0


def q8_8_nowrap(x):
    """Quantize to Q8.8 but CLAMP instead of wrap (hypothetical wider-int)."""
    raw = np.round(x * 256.0).astype(np.int64)
    raw = np.clip(raw, -32768, 32767)
    return raw.astype(np.float64) / 256.0


def softmax(S):
    Ss = S / np.sqrt(D)
    Ss = Ss - Ss.max(axis=1, keepdims=True)
    E = np.exp(Ss)
    return E / E.sum(axis=1, keepdims=True)


# Sweep input range: for each, generate fresh Q/K/V at that range, quantize
# to Q8.8, run the IDEAL Q8.8 pipeline, and report the error floor. This
# finds the range (if any) where Q8.8 meets the 0.01 target.
print("Ideal Q8.8 error floor vs input range (fresh random Q/K/V each):\n")
print(f"{'range':>6} | {'max|S|':>7} {'wrap%':>6} | "
      f"{'O max err':>9} {'O mean err':>10} | verdict")
print("-" * 66)

for rng in [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5]:
    g = np.random.default_rng(0)
    Qf = q8_8_nowrap(g.uniform(-rng, rng, (N, D)))
    Kf = q8_8_nowrap(g.uniform(-rng, rng, (N, D)))
    Vf = q8_8_nowrap(g.uniform(-rng, rng, (N, D)))

    S_true = Qf @ Kf.T
    A_true = softmax(S_true)
    O_true = A_true @ Vf

    # ideal Q8.8 pipeline: S quantized (wrap, like RTL), A quantized Q8.8
    A_q = q8_8(softmax(q8_8(S_true)))
    O_q = A_q @ Vf
    err = np.abs(O_q - O_true)

    wrap_pct = 100 * (np.abs(S_true) > 128).mean()
    verdict = "PASS" if err.max() < 0.01 else "fail"
    print(f"{rng:>6.2f} | {np.abs(S_true).max():>7.1f} {wrap_pct:>5.1f}% | "
          f"{err.max():>9.4f} {err.mean():>10.4f} | {verdict}")

print(f"\nDoc target: max < 0.01")
print("Note: even with no S overflow, A-weight quantization (probabilities")
print("~0.01-0.5 stored at 1/256 resolution) sets a floor independent of range.")

# --- What widening attention weights to Q1.15 buys (S stays Q8.8) ---
print("\n\nSame sweep, but attention weights A kept at Q1.15 (15 frac bits):\n")
print(f"{'range':>6} | {'O max err':>9} {'O mean err':>10} | verdict")
print("-" * 40)
for rng in [4.0, 2.0, 1.0, 0.5]:
    g = np.random.default_rng(0)
    Qf = q8_8_nowrap(g.uniform(-rng, rng, (N, D)))
    Kf = q8_8_nowrap(g.uniform(-rng, rng, (N, D)))
    Vf = q8_8_nowrap(g.uniform(-rng, rng, (N, D)))
    S_true = Qf @ Kf.T
    O_true = softmax(S_true) @ Vf
    A_q = q1_15(softmax(q8_8(S_true)))   # A widened, S still Q8.8
    err = np.abs(A_q @ Vf - O_true)
    verdict = "PASS" if err.max() < 0.01 else "fail"
    print(f"{rng:>6.2f} | {err.max():>9.4f} {err.mean():>10.4f} | {verdict}")
