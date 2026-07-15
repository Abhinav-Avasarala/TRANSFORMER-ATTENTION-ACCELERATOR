# Transformer Attention Accelerator

A synthesizable Transformer attention engine written from scratch in SystemVerilog, verified entirely in simulation against an independent NumPy golden reference. Computes `O = softmax(Q·Kᵀ / √d_k) · V` — the mathematical core of every Transformer — in custom fixed-point RTL hardware, at N=64 sequence length, d_k=64 head dimension.

**Final result:** max absolute error **0.00527** against a float32 reference (target: < 0.01), mean absolute error **0.00198** (target: < 0.005). Both pass with real margin. See [§8 Results](#8-final-results) for the full picture, and `docs/project_report_and_interview_prep.pdf` for a formatted version of most of this document plus a dedicated interview Q&A packet.

---

## Table of Contents

1. [The Attention Algorithm, Briefly](#1-the-attention-algorithm-briefly)
2. [Project Scope and Key Decisions](#2-project-scope-and-key-decisions)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Module-by-Module Deep Dive](#4-module-by-module-deep-dive)
5. [Fixed-Point Format Decisions](#5-fixed-point-format-decisions)
6. [The Debugging Journey — Four Real Bugs](#6-the-debugging-journey--four-real-bugs)
7. [Verification Strategy](#7-verification-strategy)
8. [Final Results](#8-final-results)
9. [Known Limitations / What's Left](#9-known-limitations--whats-left)
10. [File Structure](#10-file-structure)
11. [How to Run](#11-how-to-run)

---

## 1. The Attention Algorithm, Briefly

Every token in a sequence needs to know how relevant every other token is to it. Each token gets three vectors:

| Vector | Meaning | Analogy |
|---|---|---|
| **Q** (Query) | What is this token looking for? | A search query |
| **K** (Key) | What does this token advertise? | A webpage's title/tags |
| **V** (Value) | What content does this token hold? | The webpage's actual content |

Four steps, given matrices `Q`, `K`, `V` (each `N × d_k`, one row per token):

1. **Score**: `S = Q·Kᵀ` — every token scores every other token. `S[i][j]` = how much token `i` should attend to token `j`.
2. **Scale**: `S / √d_k` — keeps the softmax input from getting too large/peaked as `d_k` grows.
3. **Normalize**: row-wise `softmax(S)` — converts each row of raw scores into probabilities summing to 1. Uses the numerically-stable form (subtract the row max before exponentiating) since `exp(large number)` overflows: `softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))`, which is mathematically identical to plain softmax but keeps every input to `exp()` ≤ 0.
4. **Blend**: `O = A·V` — each token's new representation is a weighted average of every token's value vector, weighted by how much it was attended to.

**Row semantics matter for the RTL**: in every one of Q, K, V, S, A, O, a *row* corresponds to one token. A *column* of V, for instance, isn't a meaningful unit on its own — it's "the c-th feature value, across all tokens" — it only matters combined with A's weights.

---

## 2. Project Scope and Key Decisions

- **Target config**: `N = 64` (sequence length), `d_k = 64` (head dimension), **Q8.8 fixed-point** (16-bit: 8 integer bits + 8 fractional bits) for all external module I/O. No floating-point anywhere in the RTL — FPGAs don't have cheap floating-point hardware, and fixed-point with careful bit-width choices gets acceptable accuracy at much lower resource cost.
- **Simulation-only scope, decided partway through.** No Vivado synthesis or implementation was run. This was deliberate: correctness bugs are far cheaper to find in behavioral simulation (direct signal visibility, instant golden-model comparison) than post-synthesis, and getting the math verifiably correct first was prioritized over resource/timing optimization. See [§9](#9-known-limitations--whats-left) for exactly what this costs — no real LUT/DSP/BRAM/FF numbers, no clock frequency, no WNS.
- **No AXI4-Lite.** The original spec allowed either AXI4-Lite or a direct-port BRAM controller for the memory interface. Since there's no real host processor in this simulation-only project, AXI4-Lite would be unused complexity — `bram_if.sv` uses the simpler direct-port option.
- **No ping-pong buffering** in `bram_if` — strictly sequential load → compute → drain, no overlap across passes. The doc's fuller spec mentions double-buffering V for pipelined back-to-back passes; not needed for a single-pass design.

---

## 3. Pipeline Architecture

```
Q, K, V (external)
      │
      ▼
┌─────────────┐   raw S = Q·Kᵀ        ┌──────────────┐   A (Q1.15)         ┌─────────────┐
│ qk_systolic │ ─────────────────────▶│ softmax_unit │────────────────────▶│ av_multiply │──▶ O
│  (systolic) │   (unscaled, Q8.8)    │ (FSM, /√d_k, │                     │  (systolic) │  (Q8.8)
└─────────────┘                       │  softmax)    │                     └─────────────┘
                                       └──────────────┘
                                              ▲
                                       reciprocal_nr
                                       (1/row_sum, NR)

All three compute stages sequenced by top_fsm, which also generates the
skewed feeds qk_systolic/av_multiply need (see §4).

bram_if sits in front of top_fsm as the real-world I/O boundary (built,
verified standalone, not yet wired in — see §9).
```

| Stage | Module | Job |
|---|---|---|
| ① Score | `qk_systolic.sv` | Systolic PE array, computes raw `Q·Kᵀ` |
| ② Scale + Normalize | `softmax_unit.sv` | Row-sequential FSM: `/√d_k`, row max, `exp()` via LUT, `1/row_sum`, normalize |
| ③ Blend | `av_multiply.sv` | Systolic PE array, computes `A·V` |
| — | `reciprocal_nr.sv` | Standalone `1/D` approximation used by softmax (no hardware divider) |
| Control | `top_fsm.sv` | Master sequencer + skew-feed generator |
| Memory front-end | `bram_if.sv` | Serial ⟷ full-array translation (not yet integrated) |
| Building block | `mac_pe.sv` | 3-stage pipelined signed multiply-accumulate, used inside both systolic arrays |

**Latency (measured, `tb_top_fsm`)**: ~13,200 cycles per full attention pass. QK ≈194 cycles, softmax ≈12,736 cycles (**~96.5% of total** — the dominant cost), AV ≈194 cycles. Softmax is a deliberately non-pipelined-across-rows design; overlapping rows would cut this significantly but wasn't built.

---

## 4. Module-by-Module Deep Dive

### 4.1 `mac_pe.sv` — the shared building block

A 3-stage pipelined signed multiply-accumulate: latch inputs → multiply → accumulate. Both systolic arrays (`qk_systolic`, `av_multiply`) are grids of these. Ports are declared `signed` explicitly (unlike most of this project's plain `logic` conventions) — this is what makes the signed multiply correct even though the connecting wires elsewhere are plain unsigned `logic` vectors; signedness is about *interpretation* at the point of arithmetic, not about how the bits are declared upstream.

### 4.2 `qk_systolic.sv` — computing `S = Q·Kᵀ`

An **output-stationary systolic array**: an `N × N` grid of `mac_pe`s (4,096 PEs at N=64), where PE`[i][j]` accumulates `dot(Q[i], K[j])` — i.e., `S[i][j]`.

**The "free transpose" trick**: `S = Q·Kᵀ` means both Q and K contract over their own trailing dimension (`k`, 0..D-1). Q is fed from the **left**, row-wise. K is fed from the **top**, but *also* row-wise (`K[j]`'s row streamed into column `j`) — this works with zero actual transpose hardware because a row of K **is** a column of Kᵀ, by definition of transpose.

**The skew derivation** (why the feed formulas look the way they do): each PE`[i][j]` needs `Q[i][k]` and `K[j][k]` for the *same* local index `k`, arriving at the *same* global cycle. The array's physical wiring only gives each operand delay in one dimension for free — `a_wire` (Q) only delays by column position `j` (rightward shifts), `b_wire` (K) only delays by row position `i` (downward shifts). So each edge feed has to inject the *other* dimension's delay itself:

```
q_in[i] = Q[i][t-i]     (pre-skewed by ROW index i, compensating for a_wire's missing row-delay)
k_in[j] = K[j][t-j]     (pre-skewed by COLUMN index j, compensating for b_wire's missing column-delay)
```

Substituting through the shift registers, PE`[i][j]` ends up seeing local index `k = t - i - j` for *both* operands simultaneously — exactly what's needed. Valid range: `t ∈ [i+j, i+j+D-1]`.

**Latency formula**: `LAT = 2N + D` (derived as `(N-1) + (N-1) + D + 2`: worst-case PE is `[N-1][N-1]`, needs `2(N-1)` cycles of pure skew delay plus `D` cycles to fold in all MACs plus `2` cycles for `mac_pe`'s own pipeline depth). At N=D=64: **194 cycles**.

**Truncation**: `pe_out[FRAC_BITS +: DATA_WIDTH]` — a raw bit-select, no rounding, no saturation. The 32-bit accumulator can in principle wrap on overflow with no warning (flagged, never hit in practice at the tested input range).

**Known unresolved architectural issue**: a literal 64×64 spatial array needs 4,096 DSPs; the KV260 target board has only 1,728. This design would not fit as synthesized today — would need a tiled/reused smaller array (e.g. 16×16, reused across multiple passes). Never resolved, since synthesis was out of scope for this phase.

### 4.3 `av_multiply.sv` — computing `O = A·V`

Same systolic architecture, same `mac_pe` building block — but **not a copy-paste of `qk_systolic`**, because the math is genuinely different:

| | `qk_systolic` (`S=Q·Kᵀ`) | `av_multiply` (`O=A·V`) |
|---|---|---|
| Contraction dimension | `D` (head dim) | `N` (sequence length) |
| Transpose trick available? | Yes (K row-fed) | **No** — `O=A·V` is a plain matmul |
| A/Q feed | Row-wise, `A[i][t-i]` | Same — row-wise, `A[i][t-i]` |
| K/V feed | Row-wise (the trick) | **Column-wise**, `V[t-c][c]` |

**Why V is fed differently from K**: for a plain matmul `C=A·B`, whichever operand feeds a PE row/column has its *contracted* index vary over time and its *other* index fixed per PE. A contracts over its column index → row fixed, column varies → row-fed (same as Q). V contracts over its own *row* index → column fixed, row varies → **must be fed by column**. K only got to cheat and be fed row-wise because K's row happened to equal what Kᵀ's column needed; V has no such shortcut.

PE grid is `N × D` (not necessarily square, though it is at N=D=64). Same latency formula shape, `LAT = 2N + D`, but now `N` plays a dual role (grid rows *and* contraction length) while `D` is just grid columns — conceptually distinct from `qk_systolic`'s roles even though numerically identical here.

`A_FRAC_BITS` is a parameter (default 8, matching Q8.8) — `top_fsm` instantiates it with `A_FRAC_BITS=15`, since `softmax_unit` emits attention weights at Q1.15 precision, not Q8.8 (see [§5](#5-fixed-point-format-decisions)). The output truncation drops `A_FRAC_BITS` fractional bits, not a hardcoded `FRAC_BITS`, to correctly handle A and V being in *different* fixed-point formats.

### 4.4 `reciprocal_nr.sv` — `1/D` without a hardware divider

FPGAs don't have cheap division hardware. This computes `1/D` via:

1. **Seed**: a priority encoder finds `D`'s most-significant set bit (which power-of-two "bucket" `D` is in), plus 3 more mantissa bits (which of 8 sub-buckets within that). An 8-entry lookup table, **computed at elaboration time** (not hardcoded), maps this to a starting estimate within ~6% of the true reciprocal.
2. **Newton-Raphson refinement**: `r ← r·(2 − D·r)`, 3 iterations, each roughly doubling the number of correct bits.

Input format is fixed at Q8.8 (`DATA_WIDTH`/`FRAC_BITS`), but **output format is independently parameterized** (`OUT_WIDTH`/`OUT_FRAC`) — this is what let `softmax_unit` get a 16-fractional-bit reciprocal while still feeding it an ordinary Q8.8 `row_sum`.

**Precondition**: `D ≥ 2^FRAC_BITS` (i.e. `D ≥ 1.0`). Guaranteed by `softmax_unit`'s usage — a row's own max-scoring element always contributes `exp(0)=1.0` to `row_sum`. No divide-by-zero guard exists, deliberately, since the precondition is structurally guaranteed — see [Bug 4](#6-the-debugging-journey--four-real-bugs) for what happens when an *upstream* bug violates it anyway.

### 4.5 `softmax_unit.sv` — the row-sequential FSM

**Not a systolic array** — softmax genuinely can't be: you need a row's max *and* total sum before you can normalize any single element in that row. Instead, one FSM processes rows one at a time, three passes each:

```
MAX_SCAN  : find row_max = max(S[row][:])                              (N cycles)
EXP_SCAN  : delta = row_max - S[row][col]  (always >= 0)
            scale delta by 1/√d_k, look up exp(-delta) in a LUT,
            accumulate into row_sum                                    (N cycles)
(reciprocal_nr computes 1/row_sum here — ~5 cycles)
NORM_SCAN : A[row][col] = exp_val * (1/row_sum), Q1.15                 (N cycles)
```

**Scaling by `1/√d_k`** is applied to `delta` (not to `row_max` and `S` separately) — since scaling is multiplication by a positive constant, `SCALE·row_max − SCALE·S = SCALE·(row_max−S) = SCALE·delta`, so one multiply per element is mathematically equivalent to two, and cheaper. `SCALE_CONST = round(256/√d_k)`, computed once at elaboration via `$sqrt()`.

**The `exp()` LUT** (`scripts/gen_exp_lut.py`): 2048 entries, indexed by the (post-scaling) Q8.8 delta value, domain clamped to `[0, 8.0)` (beyond that, `exp()` is negligible). The *index* domain is always Q8.8 — what changed during debugging was the *stored value*'s precision (see Bug 2 below).

### 4.6 `top_fsm.sv` — the master sequencer

Takes `Q`, `K`, `V` as full arrays (not the doc's literal serial streaming interface — that's `bram_if`'s job) and produces a full `O` array. Sequences the three compute stages:

```
IDLE → QK_START → QK_FEED → QK_DRAIN → SM_START → SM_WAIT → AV_START → AV_FEED → AV_DRAIN → ALL_DONE
```

The `_FEED` states are the real content: they're **synthesizable hardware versions of exactly what the individual testbenches (`tb_qk_systolic`, `tb_av_multiply`) faked in simulation** — generating the skewed, one-element-per-cycle streams each systolic stage needs, using the identical formulas derived in §4.2/§4.3. Before `top_fsm` existed, only testbench code could produce this skew; now it's real RTL. `softmax_unit` needs no feed generation at all — its `s_in` wires directly to `qk_systolic`'s `s_out`, since both are full-array interfaces.

### 4.7 `bram_if.sv` — the memory front-end (built, not yet wired in)

The translation layer between a real serial external interface (matching the doc's `q_data`/`k_data`/`v_data`/`data_valid` spec, one word per cycle) and `top_fsm`'s full-array convenience interface. No real chip can expose 4,096 parallel wires — data has to arrive serially, and something has to turn that serial stream into the full arrays `top_fsm` expects.

Two phases: **LOADING** (serial in, written directly into `Q_out`/`K_out`/`V_out` output registers) and **DRAINING** (full `O_in` array captured in one cycle on `capture_o`, then streamed back out serially). Two separate completion signals, `load_done` and `out_done` — a deliberate deviation from every other module's single `done`, since this module genuinely has two separate jobs at different points in the pipeline.

**Honesty caveat, explicitly documented in the file**: exposing `Q_out`/`K_out`/`V_out` as full simultaneous arrays isn't literally how BRAM works (real BRAM has 1-2 read ports, not 4,096). This will synthesize as distributed RAM/registers despite the module's name — same category of simplification as `exp_lut`'s combinational read in `softmax_unit`. The *load* (serial writes) and *drain* (serial reads) sides are both realistic single-port-style patterns; it's specifically the "hand `top_fsm` everything at once" step that's a simulation-era convenience.

**Not yet connected to `top_fsm`** — verified standalone (12/12 tests passing: sequential + random load/drain, back-to-back round trips), but no top-level wrapper exists yet linking the two.

---

## 5. Fixed-Point Format Decisions

**Baseline: Q8.8 everywhere** (16-bit, 8 integer + 8 fractional bits) for all module I/O — chosen for the target dynamic range of scores and values.

**Two places needed to be wider**, both discovered empirically, not assumed up front:

- **The `exp()` LUT and running sum** inside `softmax_unit`: `exp()` values are often much smaller than Q8.8's 1/256 resolution can represent accurately. Widened to **16 internal fractional bits**, truncated back to Q8.8 only at the boundary.
- **Attention weights and the reciprocal**: probabilities are always in `[0,1)`, so Q8.8 wastes its 8 integer bits and leaves only 1/256 resolution for values often near 0.01. Attention weights are emitted as **Q1.15** (not Q8.8), and the reciprocal is computed at **16 fractional bits**. Both had to change *together* — a Python model showed widening only the weights still failed (see Bug 3 below).

A Q1.15 weight is unsigned and always `< 1.0`, clamped to `0x7FFF` (0.99997) so it reads as positive through `av_multiply`'s *signed* `mac_pe` multiplier.

---

## 6. The Debugging Journey — Four Real Bugs

This is the most substantive engineering content of the project. Each was root-caused to a specific, understood mechanism and verified fixed with before/after measurements — not just patched until tests passed.

### Bug 1 — Newton-Raphson seed collapse (`reciprocal_nr.sv`)

The original seed used only the coarse power-of-two bucket (MSB position), giving up to **2× worst-case error**. When the seed landed close to exactly 2× off, the NR correction term `(2 − D·r)` collapsed to ≈0 — multiplying `r` by that destroyed the estimate instead of refining it, and Q8.8's resolution meant it could never recover in the remaining iterations.

- **Found via**: hand-verifying a claim from an external code review, rather than trusting it blind.
- **Concrete evidence**: `D=511` (row_sum≈2.0) — seed computed to `256` (true value: `128`, seed is 2× too big) — collapsed to `r=1` and stayed there. **~50% error**, not rounding noise.
- **Fix**: 8-entry seed table keyed on 3 extra mantissa bits, narrowing worst-case seed error from 2× to ~6%.
- **Verified**: exhaustive sweep of all 65,280 valid inputs (`D = 256..65535`), 0 failures.

### Bug 2 — `exp()` LUT precision too coarse (`softmax_unit.sv` / `gen_exp_lut.py`)

LUT stored values at Q8.8 (1/256 resolution). For a token scoring well below the row max, the true `exp()` value (e.g. ~0.0025) was *smaller than the table's own resolution*, forcing a round-up of ~57% on that single entry. With up to 63 such entries summed per row, the overshoots compounded into a systematic bias — worst in low-entropy ("peaked") rows.

- **Found via**: a directed "one dominant token" testbench case, not random testing (random data rarely produces this exact distribution shape, even though real attention often does).
- **Concrete evidence**: computed weight `0.803` vs. true `0.865` for the dominant token — a ~7% error on the single largest weight in the row.
- **Fix**: widened the LUT and running-sum accumulator to 16 internal fractional bits, truncating to Q8.8 once at the very end instead of once per entry before summing.

### Bug 3 — attention-weight / reciprocal precision floor (`softmax_unit.sv`, `reciprocal_nr.sv`, `av_multiply.sv`)

Even with Bug 2 fixed, Q8.8-precision weights and reciprocal capped end-to-end accuracy at ~0.07 max error, independent of input range. A **faithful Python model of the exact fixed-point pipeline** (`scripts/diagnostics/diag_rtl_model.py`), built and swept *before* touching more RTL, showed widening only the weights (to Q1.15) still failed at max error `0.0124` — both the weights **and** the reciprocal needed widening together to reach the predicted `0.0058`.

- **Fix**: weights → Q1.15 (clamped to `0x7FFF`); reciprocal output → 16 fractional bits (via `reciprocal_nr`'s new independent `OUT_WIDTH`/`OUT_FRAC` params).
- This is the fix that took the design from failing to genuinely close to target — but one more bug (#4) was still hiding underneath it.

### Bug 4 — double-rounding via `int'()` vs `$rtoi()` (`softmax_unit.sv`)

The `1/√64` scaling constant was computed as `int'(0.125 × 256 + 0.5)`. `0.125 × 256 = 32.0` exactly — no rounding needed. But SystemVerilog's `int'()` **cast** already rounds to nearest on its own, unlike `$rtoi()`, which truncates (and genuinely needs the `+0.5` idiom used correctly elsewhere in this project). Adding `0.5` to an already-exact value created an artificial tie at `32.5`, which rounded **up** to `33` — a silent, systematic **3.125% scaling error applied to every delta value computed in softmax, in every row**.

- **Found via** an isolate-then-trace methodology, after the pipeline still failed post-Bug-3-fix and didn't match the Python model's prediction:
  1. Built an **exhaustive isolated test of `reciprocal_nr` in its actual wide-precision config** (`tb/tb_reciprocal_wide.sv`) — proved the reciprocal module itself was mathematically exact, ruling it out.
  2. Built a **targeted diagnostic testbench** (`tb/debug/tb_softmax_debug.sv`) that printed `softmax_unit`'s internal signals (`row_sum`, `recip_d`, `recip_result`, `norm_prod`) via hierarchical reference, for one case with hand-derived expected values.
  3. The very first internal signal, `row_sum`, was already wrong (`74041` vs. hand-derived `75742`) — pointing upstream of the reciprocal entirely, to the scaling step.
- **Fix**: removed the redundant `+ 0.5`. Also fixed the same latent pattern in `reciprocal_nr`'s seed table (harmless there in practice, since NR self-corrects a slightly-off seed, but fixed for consistency).
- **Result**: max error `0.0617` → **`0.00527`** — the final passing number.

### Related methodology finding (not an RTL bug)

Early end-to-end runs showed a catastrophic max error of **6.55** — traced to golden test vectors generated with too wide an input range (`Q/K ~ ±4.0`), causing raw `S = Q·Kᵀ` scores to reach `±175`, overflowing Q8.8's `±128` representable range and wrapping. Regenerating golden data at a realistic range (`±2.0` — real attention keeps `Q/K ~ O(1)` specifically so `S/√d_k` stays `O(1)`) dropped max error to `0.091` immediately, with **zero RTL changes** — confirming the RTL was already correct and the problem was test methodology.

---

## 7. Verification Strategy

Different modules needed genuinely different testing philosophies — not one template copy-pasted everywhere:

| Approach | Used for | Why |
|---|---|---|
| **Internal bit-exact golden model** | `qk_systolic`, `av_multiply` | Mirrors the RTL's own truncation/rounding exactly — fast, no external deps, catches wiring bugs immediately. Structurally *cannot* catch a bug present in "the arithmetic as coded" since it mirrors that same arithmetic. |
| **Independent golden cross-check** | `tb_qk_systolic_golden.sv` | A from-scratch Python reimplementation (`gen_golden.py`), not derived from reading the RTL's control flow — catches the class of bug an internal model structurally can't. |
| **Exhaustive sweep** | `reciprocal_nr` (both configs) | For small-enough input spaces, sampling has real odds of missing a bug that only shows at specific values (exactly what happened with Bug 1) — exhaustive coverage removes that risk entirely. |
| **Tolerance-based, vs. true math** | `softmax_unit`, `tb_top_fsm` | For modules with inherent approximation (LUT, NR), comparing against a *re-derived approximate* model would just duplicate whatever bugs exist in both places. Compare against true floating-point math instead. |
| **Targeted internal-signal tracing** | `tb_softmax_debug.sv` | When a pass/fail testbench isn't enough to localize a bug, hook into internal registers directly via hierarchical reference and compare against hand-derived expected values at each step. |

**The golden-reference pipeline** (`scripts/gen_golden.py`): generates random Q/K/V, computes the true attention output in float64/NumPy (including the `/√d_k` scaling, correct from day one — it was the RTL that was initially missing that step, not the golden model), and quantizes inputs to Q8.8 hex files the SystemVerilog testbenches load via `$readmemh`.

---

## 8. Final Results

From `tb/tb_top_fsm.sv`, comparing the real RTL pipeline against `gen_golden.py`'s independent float32 reference:

| Metric | Measured | Target | Result | Margin |
|---|---|---|---|---|
| Max Absolute Error | 0.005270 | < 0.01 | **PASS** | ~1.9× |
| Mean Absolute Error | 0.001982 | < 0.005 | **PASS** | ~2.5× |
| Elements exceeding tolerance | 0 / 4096 | 0 | **PASS** | — |
| Latency (one attention pass) | ~13,200 cycles | — | — | measured, not synthesized timing |

**What this certifies**: the real SystemVerilog RTL, in Vivado xsim, processing the same Q/K/V (quantized to Q8.8) fed to a from-scratch NumPy implementation of the true attention formula, matches within 0.53% max absolute error — a direct correctness check, not a self-consistency check against the RTL's own arithmetic.

**Caveat**: this is one fixed golden dataset (one random seed, 4096 output elements) — solid coverage, but not multiple independent trials.

---

## 9. Known Limitations / What's Left

- **No synthesis was run.** No real LUT/DSP/BRAM/FF utilization, no achievable clock frequency, no timing closure (WNS) data exists. Every number above is behavioral simulation. Deliberate scope decision, not an oversight.
- **DSP budget mismatch, unresolved.** `qk_systolic`'s one-shot 64×64 spatial array needs 4,096 PEs/DSPs; the KV260 target has 1,728. Would need a tiled/reused array before real synthesis could succeed.
- **No hardware deployment.** Verified entirely via simulation + Python golden-model cross-checks.
- **`bram_if` not integrated.** Built and verified standalone; no top-level wrapper connects it to `top_fsm` yet.
- **No ping-pong buffering** in `bram_if` — no overlap between loading the next input set and computing the current one.
- **Single golden dataset** for the final end-to-end PASS.

---

## 10. File Structure

```
src/                    7 design modules: mac_pe, qk_systolic, av_multiply,
                        reciprocal_nr, softmax_unit, top_fsm, bram_if
tb/                     8 regression testbenches (one per src/ module, plus
                        tb_qk_systolic_golden for the independent cross-check)
tb/debug/               tb_softmax_debug.sv — diagnostic-only, not part of
                        the regression suite (see Bug 4's debugging story)
scripts/                gen_golden.py, gen_exp_lut.py — the reusable generators
scripts/diagnostics/    4 one-off analysis scripts from the precision
                        debugging investigation (diag_s_range, diag_error_budget,
                        diag_rtl_model, diag_recip_nr)
golden/                 generated test vectors + expected outputs (q.hex,
                        k.hex, v.hex, s_fixed_expected.hex, o_float_expected.*)
docs/                   project_report_and_interview_prep.pdf (formatted
                        version of most of this README + interview packet),
                        attention_accelerator_project_doc.pdf (original spec)
exp_lut.hex             generated exp() LUT data — left at repo root
                        deliberately; referenced via hardcoded absolute paths
                        in several already-verified files, not worth the risk
                        of moving for a cosmetic gain
```

---

## 11. How to Run

**Simulation is done in Vivado** (xsim), with source files copied into a separate Vivado project folder (not this repo directly).

**Two gotchas that will bite you** if you forget them:

1. **Working directory.** xsim runs from its own simulation folder, not this repo. Any file referenced by a relative path (`$readmemh`, `$fopen`) — `exp_lut.hex`, the `golden/` directory — will silently fail to load and default to zeros unless the referencing `.sv` file uses an **absolute path**. Every testbench that needs these already hardcodes the absolute path; if you add a new one, do the same.
2. **Default simulation runtime is too short.** Vivado's default (`run 1000ns`) will cut off mid-test for anything nontrivial. Use `run -all` in the Tcl console — every testbench here calls `$finish` itself once done, so `run -all` runs to completion rather than an arbitrary fixed time.

**To regenerate golden data** (from this repo, in a Python env with `numpy`):
```
python scripts/gen_golden.py --range 2.0 --out golden
python scripts/gen_exp_lut.py --out exp_lut.hex
```
Note: `--range 2.0` is intentional, not a default — see the [overflow finding in §6](#6-the-debugging-journey--four-real-bugs). A wider range will overflow Q8.8's representable score range.
