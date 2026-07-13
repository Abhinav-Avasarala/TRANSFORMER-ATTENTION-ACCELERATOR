#!/usr/bin/env python3
"""
gen_exp_lut.py -- generates the exp() lookup table used by softmax_unit.sv.

softmax_unit computes, per row: delta = row_max - score (always >= 0, since
score <= row_max), then looks up exp(-delta) in this table. Domain is
clamped to delta in [0, 8.0) in steps of 1/256 (Q8.8): beyond delta=8.0,
exp(-delta) < 1/512, which already rounds to 0 in Q8.8 (smallest positive
representable value is 1/256), so clamping there loses no precision.

Table has 2048 entries (11-bit address), each a Q8.8 unsigned value:
    LUT[idx] = round(256 * exp(-idx / 256))
    LUT[0]      = 256  (exp(0)    = 1.0)
    LUT[2047]   = 0     (exp(-8)  ~ 0.000335, rounds to 0)

Usage:
    python scripts/gen_exp_lut.py --out exp_lut.hex
"""
import argparse
import math


LUT_DEPTH = 2048
FRAC_BITS = 8
Q_SCALE = 1 << FRAC_BITS  # 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp_lut.hex")
    args = ap.parse_args()

    with open(args.out, "w") as f:
        for idx in range(LUT_DEPTH):
            delta_real = idx / Q_SCALE
            val = math.exp(-delta_real)
            raw = round(val * Q_SCALE)
            raw = max(0, min(raw, 0xFFFF))
            f.write(f"{raw:04x}\n")

    print(f"Wrote {LUT_DEPTH} entries to {args.out}")
    print(f"  LUT[0]    = {round(1.0 * Q_SCALE)} (exp(0) = 1.0)")
    print(f"  LUT[256]  = {round(math.exp(-1.0) * Q_SCALE)} (exp(-1) ~= 0.368)")
    print(f"  LUT[2047] = {round(math.exp(-2047/Q_SCALE) * Q_SCALE)} (exp(-8) ~= 0, clamp point)")


if __name__ == "__main__":
    main()
