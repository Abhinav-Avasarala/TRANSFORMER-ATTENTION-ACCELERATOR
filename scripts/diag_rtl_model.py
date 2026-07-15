#!/usr/bin/env python3
"""Faithful fixed-point mirror of the RTL pipeline, used to test which
precision interventions actually reach max error < 0.01 BEFORE doing the
RTL surgery. Models the real error sources: Q8.8 S, LUT exp, Q8.8-coarse
reciprocal, per-stage truncation. Reciprocal is modeled as true 1/x
quantized to its storage format (the NR unit converges to ~1-2 LSB of
that, per tb_reciprocal_nr)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

N = D = 64
GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")
EXP_FRAC = 16
LUT_DEPTH = 2048


def load_hex(name, signed=True):
    vals = []
    with open(os.path.join(GOLD, name)) as f:
        for line in f:
            v = int(line.strip(), 16)
            if signed and (v & 0x8000):
                v -= 0x10000
            vals.append(v)
    return np.array(vals, dtype=np.int64)


def trunc_q(x_scaled_int, drop_bits):
    """Arithmetic right shift (truncation toward -inf, like Verilog >>)."""
    return x_scaled_int >> drop_bits


# exp LUT as integers (EXP_FRAC frac bits), exactly like exp_lut.hex
lut = np.array([min(round(np.exp(-i / 256.0) * (1 << EXP_FRAC)), (1 << EXP_FRAC))
                for i in range(LUT_DEPTH)], dtype=np.int64)

Qf = (load_hex("q.hex").reshape(N, D)).astype(np.float64) / 256.0
Kf = (load_hex("k.hex").reshape(N, D)).astype(np.float64) / 256.0
Vf = (load_hex("v.hex").reshape(N, D)).astype(np.float64) / 256.0
Q = load_hex("q.hex").reshape(N, D)
K = load_hex("k.hex").reshape(N, D)
V = load_hex("v.hex").reshape(N, D)

# --- true float reference ---
S_true = Qf @ Kf.T
As = S_true / np.sqrt(D)
As = As - As.max(axis=1, keepdims=True)
E = np.exp(As)
A_true = E / E.sum(axis=1, keepdims=True)
O_true = A_true @ Vf

SCALE_CONST = round(256 / np.sqrt(D))   # =32 for D=64


def pipeline(a_frac_bits, recip_frac_bits, o_frac_bits=8):
    """Run the fixed-point pipeline with configurable precisions.
    a_frac_bits    : fractional bits for attention weights A (RTL: 8)
    recip_frac_bits: fractional bits for the reciprocal (RTL: 8)
    Returns O as float."""
    # QK: 32-bit accumulate, truncate to Q8.8 (wrap)
    S = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            acc = int((Q[i].astype(np.int64) * K[j].astype(np.int64)).sum()) & 0xFFFFFFFF
            s16 = trunc_q(acc, 8) & 0xFFFF
            S[i, j] = s16 - 0x10000 if (s16 & 0x8000) else s16

    O = np.zeros((N, D), dtype=np.float64)
    for i in range(N):
        row_max = S[i].max()
        delta = row_max - S[i]                      # >=0, Q8.8
        dscaled = trunc_q(delta * SCALE_CONST, 8)   # *1/sqrt(D), back to Q8.8
        idx = np.clip(dscaled, 0, LUT_DEPTH - 1)
        exp_v = lut[idx]                            # EXP_FRAC frac bits
        row_sum = int(exp_v.sum())                  # EXP_FRAC frac bits

        # reciprocal of row_sum, at recip_frac_bits precision.
        row_sum_real = row_sum / (1 << EXP_FRAC)    # true value (~1..N)
        recip_int = round((1.0 / row_sum_real) * (1 << recip_frac_bits))

        # A = exp_v * recip, result frac = EXP_FRAC + recip_frac_bits,
        # truncate to a_frac_bits. Clamp to signed-16-bit positive max
        # (0x7FFF) so av_multiply's signed mac_pe reads A as positive --
        # this is the real hardware constraint, modeled here to confirm
        # it doesn't break the pass.
        A_int = trunc_q(exp_v.astype(np.int64) * recip_int,
                        EXP_FRAC + recip_frac_bits - a_frac_bits)
        A_int = np.clip(A_int, 0, 0x7FFF)
        A = A_int.astype(np.float64) / (1 << a_frac_bits)

        # AV for this row's output: sum A * V, truncate to o_frac_bits
        for c in range(D):
            acc = int((A_int * V[:, c]).sum())      # frac = a_frac_bits + 8
            o_int = trunc_q(acc, a_frac_bits + 8 - o_frac_bits)
            O[i, c] = o_int / (1 << o_frac_bits)
    return O


def report(name, O):
    err = np.abs(O - O_true)
    verdict = "PASS" if err.max() < 0.01 else "fail"
    print(f"{name:48s} max={err.max():.4f} mean={err.mean():.4f} "
          f">0.01:{100*(err>0.01).mean():3.0f}%  {verdict}")


def a_error(a_frac_bits, recip_frac_bits):
    """Max |A_rtl - A_true| over all weights -- sets tb_softmax_unit tolerance.
    Here S is the direct input (no QK quantization error), matching the
    softmax-unit test scenario."""
    max_e = 0.0
    for i in range(N):
        row_max = S_true[i].max() if False else None  # use fixed-point S below
    # Reuse the pipeline's per-row A by recomputing against true softmax of
    # the SAME quantized S the RTL sees.
    Sq = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            acc = int((Q[i].astype(np.int64) * K[j].astype(np.int64)).sum()) & 0xFFFFFFFF
            s16 = (acc >> 8) & 0xFFFF
            Sq[i, j] = s16 - 0x10000 if (s16 & 0x8000) else s16
    for i in range(N):
        rm = Sq[i].max()
        d = rm - Sq[i]
        ds = (d * SCALE_CONST) >> 8
        idx = np.clip(ds, 0, LUT_DEPTH - 1)
        ev = lut[idx]
        rs = int(ev.sum())
        recip = round((1 << recip_frac_bits) / (rs / (1 << EXP_FRAC)))
        A_int = np.clip((ev.astype(np.int64) * recip) >> (EXP_FRAC + recip_frac_bits - a_frac_bits),
                        0, 0x7FFF)
        A_rtl = A_int / (1 << a_frac_bits)
        # true softmax of the SAME quantized S
        ss = (Sq[i] / 256.0) / np.sqrt(D)
        ss = ss - ss.max()
        e = np.exp(ss); A_t = e / e.sum()
        max_e = max(max_e, np.abs(A_rtl - A_t).max())
    return max_e


print(f"Max attention-weight error (A=Q1.15, recip=16): {a_error(15, 16):.6f}")
print("  -> sets tb_softmax_unit real-valued tolerance\n")

print("Faithful RTL-mirror pipeline, testing precision interventions:\n")
report("current RTL: A=Q8.8, recip=Q8.8", pipeline(8, 8))
report("A=Q1.15, recip=Q8.8", pipeline(15, 8))
report("A=Q8.8,  recip=Q1.15", pipeline(8, 15))
report("A=Q1.15, recip=Q1.15", pipeline(15, 15))
report("A=Q1.15, recip=Q8.16 (16 frac)", pipeline(15, 16))
