#!/usr/bin/env python3
"""Mirror reciprocal_nr.sv's EXACT fixed-point NR iteration to confirm it
converges at OUT_FRAC=16 with N iterations, across the row_sum input range
softmax actually uses (d real in [1, 64], i.e. d_raw Q8.8 in [256, 16384])."""
FRAC_BITS = 8
OUT_FRAC = 16


def seed_lut(out_frac):
    t = []
    for frac in range(8):
        center = 1.0 + (2.0 * frac + 1.0) / 16.0
        t.append(round((1 << out_frac) / center))
    return t


def recip_nr(d_raw, out_frac, iterations):
    SEED = seed_lut(out_frac)
    msb = d_raw.bit_length() - 1
    frac_bits = (d_raw >> (msb - 3)) & 0x7
    r = SEED[frac_bits] >> (msb - FRAC_BITS)
    two = 2 << out_frac
    for _ in range(iterations):
        t1 = (d_raw * r) >> FRAC_BITS       # -> out_frac frac bits
        t2 = two - t1
        r = (r * t2) >> out_frac
    return r


for iters in [3, 4]:
    max_rel = 0.0
    worst_d = 0
    for d_raw in range(256, 16385):
        d_real = d_raw / (1 << FRAC_BITS)
        r = recip_nr(d_raw, OUT_FRAC, iters)
        r_real = r / (1 << OUT_FRAC)
        true = 1.0 / d_real
        rel = abs(r_real - true) / true
        if rel > max_rel:
            max_rel, worst_d = rel, d_raw
    lsb_rel = (1 / (1 << OUT_FRAC)) / (1.0 / (16384 / 256))  # ~1 LSB rel at largest d
    print(f"OUT_FRAC={OUT_FRAC}, iterations={iters}: "
          f"max relative error = {max_rel:.2e} at d={worst_d/256:.3f} "
          f"({'OK' if max_rel < 2e-4 else 'MARGINAL'})")
