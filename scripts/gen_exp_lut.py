#!/usr/bin/env python3
"""
gen_exp_lut.py -- generates the exp() lookup table used by softmax_unit.sv.

softmax_unit computes, per row: delta = row_max - score (always >= 0, since
score <= row_max), then looks up exp(-delta) in this table. The INDEX
domain is always delta in [0, 8.0) in steps of 1/256 (Q8.8, 2048 entries,
11-bit address) -- this never changes, because row_max/score are Q8.8.

The STORED VALUE at each index uses --exp-frac-bits fractional bits
(default 16, i.e. twice Q8.8's precision), not the index domain's 8.
This matters because exp(-delta) can be much smaller than Q8.8's smallest
representable step (1/256): e.g. exp(-6) ~= 0.00248, which Q8.8 can only
round to 0 or 1/256 -- a >50% relative error on a value that gets summed
many times per row, producing a real, measurable bias in row_sum (this
was caught empirically: a "one dominant token" test case came out with
softmax weight 0.803 instead of the true 0.865). Storing exp() with more
fractional bits keeps that per-entry error small enough that it doesn't
compound into a visible bias after summing.

Table has 2048 entries, each an unsigned Q(1).exp_frac_bits value:
    LUT[idx] = round(2^exp_frac_bits * exp(-idx / 256))
    LUT[0]    = 2^exp_frac_bits         (exp(0)   = 1.0)
    LUT[2047] ~= round(2^exp_frac_bits * 0.000335)  (exp(-8), no longer
                 rounds to exactly 0 at higher precision -- that's fine,
                 it's now a more accurate small nonzero value instead of
                 a bigger rounding error).

Usage:
    python scripts/gen_exp_lut.py --out exp_lut.hex
    python scripts/gen_exp_lut.py --exp-frac-bits 16 --out exp_lut.hex
"""
import argparse
import math


LUT_DEPTH  = 2048
FRAC_BITS  = 8            # index domain precision -- fixed, matches Q8.8 delta
Q_SCALE    = 1 << FRAC_BITS  # 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp_lut.hex")
    ap.add_argument("--exp-frac-bits", type=int, default=16,
                     help="fractional bits for STORED values (default 16, "
                          "double Q8.8's 8 -- the index domain always stays Q8.8)")
    args = ap.parse_args()

    out_scale = 1 << args.exp_frac_bits
    hex_digits = (args.exp_frac_bits + 1 + 3) // 4  # +1 for the "1.0 exactly" case

    with open(args.out, "w") as f:
        for idx in range(LUT_DEPTH):
            delta_real = idx / Q_SCALE
            val = math.exp(-delta_real)
            raw = round(val * out_scale)
            raw = max(0, min(raw, out_scale))
            f.write(f"{raw:0{hex_digits}x}\n")

    print(f"Wrote {LUT_DEPTH} entries to {args.out} (exp_frac_bits={args.exp_frac_bits})")
    print(f"  LUT[0]    = {round(1.0 * out_scale)} (exp(0) = 1.0)")
    print(f"  LUT[256]  = {round(math.exp(-1.0) * out_scale)} (exp(-1) ~= 0.368)")
    print(f"  LUT[1536] = {round(math.exp(-6.0) * out_scale)} (exp(-6) ~= 0.00248)")
    print(f"  LUT[2047] = {round(math.exp(-2047/Q_SCALE) * out_scale)} (exp(-8) ~= 0.000335, clamp point)")


if __name__ == "__main__":
    main()
