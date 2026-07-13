#!/usr/bin/env python3
"""
gen_golden.py -- golden reference generator for the attention accelerator.

Produces two independent references from the same random Q/K/V:

  1. S_fixed  -- the QK^T score matrix computed with the EXACT arithmetic
                 qk_systolic.sv uses: a 32-bit accumulator that wraps on
                 overflow (matches mac_pe's `acc <= acc + product`), then
                 truncated by taking bits [23:8] (matches
                 `pe_out[r][c][FRAC_BITS +: DATA_WIDTH]`).
                 Use this to check qk_systolic RTL output today.

  2. O_float  -- the "true" attention output computed entirely in float32:
                 softmax(Q K^T / sqrt(d_k)) @ V. This is the project's
                 actual pass/fail target once softmax_unit/av_multiply/
                 top_fsm exist: max(abs(rtl_out - O_float)) < 0.01.

Unlike tb_qk_systolic.sv's internal golden_s() (which mirrors the RTL's
arithmetic and therefore can never disagree with a correctly-wired RTL,
and which is restricted to non-negative inputs), this script uses signed
Q8.8 values -- real attention scores are signed -- and is a fully
independent implementation, so it can catch bugs golden_s() structurally
cannot.

Usage:
    python scripts/gen_golden.py
    python scripts/gen_golden.py --n 8 --d 8 --seed 1 --out golden_8x8
"""
import argparse
import os

# Must be set before numpy is imported: this conda/MKL build crashes (no
# traceback, just a hard process exit) on matrix multiply due to a
# conflict between MKL's Intel OpenMP runtime and llvm-openmp both being
# loaded. This is a known Windows conda+MKL issue, not specific to this
# script -- forcing single-threaded MKL avoids the conflict.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

FRAC_BITS = 8
DATA_WIDTH = 16
Q_SCALE = 1 << FRAC_BITS                # 256
Q_MIN = -(1 << (DATA_WIDTH - 1))        # -32768
Q_MAX = (1 << (DATA_WIDTH - 1)) - 1     # 32767
MASK32 = (1 << 32) - 1
MASK16 = (1 << 16) - 1


def to_q8_8(x):
    """Quantize a float array to Q8.8: round to nearest, clip to 16-bit signed range."""
    q = np.round(x * Q_SCALE).astype(np.int64)
    return np.clip(q, Q_MIN, Q_MAX).astype(np.int32)


def from_q8_8(q):
    return q.astype(np.float64) / Q_SCALE


def fixed_qkt(Q_fx, K_fx):
    """
    Bit-exact mirror of qk_systolic.sv's datapath:
      acc = sum_k Q[i][k] * K[j][k], accumulated in a 32-bit register
            that silently wraps on overflow (no saturation)
      S[i][j] = acc[FRAC_BITS +: DATA_WIDTH]   (raw bit-select, i.e.
                (acc >> FRAC_BITS) & 0xFFFF, reinterpreted as signed --
                NOT an arithmetic/sign-extending shift)
    """
    n, d = Q_fx.shape
    S = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            acc = 0
            for k in range(d):
                acc = (acc + int(Q_fx[i, k]) * int(K_fx[j, k])) & MASK32
            bits16 = (acc >> FRAC_BITS) & MASK16
            S[i, j] = bits16 - 0x10000 if (bits16 & 0x8000) else bits16
    return S


def float_attention(Q, K, V):
    """Reference attention in float64 (plenty of headroom vs. the float32 doc target)."""
    d_k = Q.shape[1]
    S = Q @ K.T
    S_scaled = S / np.sqrt(d_k)
    S_shift = S_scaled - S_scaled.max(axis=1, keepdims=True)
    E = np.exp(S_shift)
    A = E / E.sum(axis=1, keepdims=True)
    O = A @ V
    return S, A, O


def write_hex(path, arr):
    """One 16-bit hex value per line, row-major, two's-complement -- $readmemh-ready."""
    with open(path, "w") as f:
        for v in arr.flatten():
            f.write(f"{int(v) & 0xFFFF:04x}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64, help="sequence length N")
    ap.add_argument("--d", type=int, default=64, help="head dimension d_k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="golden", help="output directory")
    ap.add_argument("--range", type=float, default=4.0,
                     help="Q/K/V float values drawn uniformly from [-range, range)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    Qf = rng.uniform(-args.range, args.range, size=(args.n, args.d)).astype(np.float32)
    Kf = rng.uniform(-args.range, args.range, size=(args.n, args.d)).astype(np.float32)
    Vf = rng.uniform(-args.range, args.range, size=(args.n, args.d)).astype(np.float32)

    Q_fx = to_q8_8(Qf)
    K_fx = to_q8_8(Kf)
    V_fx = to_q8_8(Vf)

    print(f"Generating golden vectors: N={args.n} D={args.d} seed={args.seed} "
          f"range=[-{args.range}, {args.range})")

    print("Computing fixed-point QK^T (bit-exact RTL model)...")
    S_fixed = fixed_qkt(Q_fx, K_fx)

    print("Computing float32 golden attention output...")
    S_float, A_float, O_float = float_attention(
        from_q8_8(Q_fx), from_q8_8(K_fx), from_q8_8(V_fx)
    )

    os.makedirs(args.out, exist_ok=True)
    write_hex(os.path.join(args.out, "q.hex"), Q_fx)
    write_hex(os.path.join(args.out, "k.hex"), K_fx)
    write_hex(os.path.join(args.out, "v.hex"), V_fx)
    write_hex(os.path.join(args.out, "s_fixed_expected.hex"), S_fixed)
    np.savetxt(os.path.join(args.out, "o_float_expected.txt"), O_float, fmt="%.6f")
    np.save(os.path.join(args.out, "o_float_expected.npy"), O_float)

    print(f"\nWrote to ./{args.out}/:")
    print("  q.hex, k.hex, v.hex          -- Q8.8 signed inputs, $readmemh-ready")
    print("  s_fixed_expected.hex         -- bit-exact QK^T qk_systolic should produce today")
    print("  o_float_expected.{txt,npy}   -- float32 attention target for softmax/av_multiply")
    print(f"\nQ8.8 quantization error introduced by --range={args.range}: "
          f"max |Qf - dequant(Q_fx)| = "
          f"{float(np.max(np.abs(Qf - from_q8_8(Q_fx)))):.6f} "
          f"(should be <= {1 / Q_SCALE / 2})")


if __name__ == "__main__":
    main()
