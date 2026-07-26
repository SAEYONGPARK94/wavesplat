# RESULT (GPU) — the learned arm on all 18 collapse windows; P2's bar resolved

**2026-07-26. Protocol frozen in `PREREG_GPU_ADDENDUM.md` (sha256 `39af85b8d05e2a87…`)
before the learned arm was run on any held-out collapse window. Nothing committed. No file
under `scripts/`, `wavesplat/` or `data/` modified.**

Artifacts: `PREREG_GPU_ADDENDUM.md`; `run_learned_arm.py` → `learned_arm.json`
(18 windows × 5 diffusion seeds × 4 posterior samples, 345 s, RTX 4080 Super);
`analyze_learned.py` → `results_learned.json`, `analysis_learned.txt`.
This file fills the PENDING-GPU slot of `RESULT.md` §5. It adds no endpoint and no window.

---

## 0. Headline

**Both of P2's bars FAIL, and by a wide margin.**

| criterion | `C_best` (best of six GT-free escapes) | learned must reach (1.5×) | **learned actual** | verdict |
|---|---|---|---|---|
| **(a)** base − arm ≥ 0.5 | 6/18 (E2-double) | ≥ 9/18 | **6/18** | **FAIL** |
| **(b)** arm < 1.0 | 11/18 (E2-double) | ≥ 17/18 | **6/18** | **FAIL** |

Pre-committed consequence, now triggered on **both** criteria (`PREREG.md` §3, verbatim):
**Ch.7's interlock is a paragraph about initialization robustness.**

The learned arm does not merely fail to beat the cheap escape by 1.5×; **it does not beat it
at all**. `kp_hint × 2` — one fixed, GT-free, ~1 s reconfiguration of the analytic arm — is
numerically better than the learned arm on **17 of 18** windows, and matches it on count (a)
while nearly doubling it on count (b).

**New and not previously visible: the learned arm has own-failures out of sample.** On 7 of
the 15 held-out collapse windows it makes the error *worse* than the analytic baseline it
was supposed to repair (up to 1.041 → 1.961). All 7 are BS-site windows; none is a design
window. The circulating statement "learned reduces error on 3/3 collapse windows with 0 own
collapses" was measured on the 3 design windows only and **does not survive the held-out
extension** — out of sample the learned arm repairs 8/15 and damages 7/15.

## 1. Validity control — PASS at exactly zero

The three design collapse windows were **recomputed live** (not read from disk) and
reproduce `tmp/wp_real_2026-07-25/wp_real.json` exactly:

| window | live | stored | \|Δ\| median | \|Δ\| per-seed vector |
|---|---|---|---|---|
| AA14_w0 | 1.070895584754853 | 1.070895584754853 | 0.000e+00 | 0.000e+00 |
| AA15a_w3 | 0.791312057969709 | 0.791312057969709 | 0.000e+00 | 0.000e+00 |
| LJ18g_w3 | 0.961485926604453 | 0.961485926604453 | 0.000e+00 | 0.000e+00 |

**max |Δ| = 0.000e+00 ⟹ PASS** (tolerance 1e-9). The held-out readings below were produced
by the same arm whose numbers are in circulation. This also licenses the cache reuse
declared in `tmp/blind_auto_real_2026-07-26/PREREG.md` §1.2.

## 2. Per-window — baseline hand | learned (median-of-5) | E2-double

`d_learn`, `d_E2d` = baseline − arm (positive = repair). `npx` = scored finite pixels.

| window | arena | base | **learned** | sd | **E2-double** | d_learn | d_E2d | npx |
|---|---|---|---|---|---|---|---|---|
| BS11b_w4 | heldout | 1.2855 | 1.2202 | 0.149 | 1.1895 | +0.065 | +0.096 | 16 |
| BS11c_w3 | heldout | 1.1753 | **1.7344** | 0.309 | 1.2815 | **−0.559** | −0.106 | 16 |
| BS11c_w4 | heldout | 1.0496 | **1.7676** | 0.456 | 0.8885 | **−0.718** | +0.161 | 16 |
| BS13a_w0 | heldout | 1.0687 | 0.7389 | 0.092 | 0.7308 | +0.330 | +0.338 | 25 |
| BS13a_w1 | heldout | 1.1481 | **1.4958** | 0.168 | 1.2291 | **−0.348** | −0.081 | 25 |
| BS13a_w2 | heldout | 1.0415 | **1.9609** | 0.148 | 1.0857 | **−0.919** | −0.044 | 25 |
| BS13a_w4 | heldout | 1.1722 | **1.7072** | 0.246 | 0.8956 | **−0.535** | +0.277 | 25 |
| BS13a_w5 | heldout | 1.1551 | **1.3245** | 0.274 | 0.8275 | **−0.169** | +0.328 | 25 |
| BS13c_w2 | heldout | 1.0288 | **1.5261** | 0.218 | 1.0442 | **−0.497** | −0.015 | 25 |
| BS13c_w3 | heldout | 1.5034 | 1.4873 | 0.489 | 1.0154 | +0.016 | +0.488 | 25 |
| LJ18n_w0 | heldout | 7.6111 | 1.1989 | 0.437 | 0.9540 | +6.412 | +6.657 | 121 |
| LJ18n_w1 | heldout | 1.6102 | **0.6306** | 0.086 | 0.6786 | +0.980 | +0.932 | 121 |
| LJ18n_w3 | heldout | 1.3685 | 0.6366 | 0.135 | 0.5854 | +0.732 | +0.783 | 121 |
| LJ18n_w4 | heldout | 4.2517 | 0.7140 | 0.181 | 0.5478 | +3.538 | +3.704 | 121 |
| LJ18n_w5 | heldout | 1.9137 | 1.6055 | 0.161 | 1.4773 | +0.308 | +0.436 | 121 |
| AA14_w0 | design | 2.3091 | 1.0709 | 0.149 | 0.8253 | +1.238 | +1.484 | 162 |
| AA15a_w3 | design | 1.1200 | 0.7913 | 0.041 | 0.6960 | +0.329 | +0.424 | 230 |
| LJ18g_w3 | design | 2.9254 | 0.9615 | 0.034 | 0.7256 | +1.964 | +2.200 | 3182 |

Declared in the PREREG before running: the BS held-out windows are scored on 16–25 finite
GT pixels. That is a low-count statistic — but it is the **identical mask** for both arms
(paired), so it cannot manufacture the learned-vs-E2-double ordering. It is a caveat on the
per-window *precision*, not on the comparison.

## 3. The two frozen counts — never merged

| arm | (a) Δ ≥ 0.5 | (b) < 1.0 |
|---|---|---|
| **learned** | **6/18** | **6/18** |
| E2-double (`kp_hint × 2`) | 6/18 | 11/18 |

By arena:

| arm | held-out 15 (a)/(b) | design 3 (a)/(b) |
|---|---|---|
| learned | 4/15 · 4/15 | 2/3 · 2/3 |
| E2-double | 4/15 · 8/15 | 2/3 · 3/3 |

## 4. P2's bar — FAIL on both criteria

`C_best(criterion)` = max over the six GT-free escapes {E1-median, E2-peak2, E2-half,
E2-double, E3-half, E3-double}; measured `C_best(a) = 6`, `C_best(b) = 11` (both E2-double).

- **(a): need ≥ 9/18, learned 6/18 → FAIL.**
- **(b): need ≥ 17/18, learned 6/18 → FAIL.**

**The comparator is deliberately generous to the control** — the control may be chosen with
hindsight over six configurations while the learned arm may not. That asymmetry is stated
here and must be restated wherever this comparison appears. It does not rescue the learned
arm: even against E2-double *alone, pre-committed*, learned ties on (a) and loses 6 vs 11
on (b).

**Pre-committed consequence, verbatim from `PREREG.md` §3, now in force on both criteria:**

> Ch.7's interlock is **a paragraph about initialization robustness**.

No oracle-flavored reading was used in either direction.

## 5. Operational subset — the 5 windows the frozen τ = 1.5 detector fires on

*(post-hoc, defined by deployment rather than by scores, exactly as in `RESULT.md` §4 —
these are the only windows on which a fallback would ever run.)*

| window | base | **learned** | **E2-double** | better |
|---|---|---|---|---|
| AA14_w0 | 2.3091 | 1.0709 | **0.8253** | E2-double |
| LJ18g_w3 | 2.9254 | 0.9615 | **0.7256** | E2-double |
| LJ18n_w0 | 7.6111 | 1.1989 | **0.9540** | E2-double |
| LJ18n_w1 | 1.6102 | **0.6306** | 0.6786 | learned |
| LJ18n_w4 | 4.2517 | 0.7140 | **0.5478** | E2-double |

- learned **(a) 5/5, (b) 3/5**; E2-double **(a) 5/5, (b) 5/5**. Head-to-head **4/5 to
  E2-double**.
- Both arms repair every fired window by ≥ 0.5. The separation is on the *below-line* count:
  the learned arm leaves AA14_w0 (1.071) and LJ18n_w0 (1.199) above the collapse line;
  E2-double puts all five below it.
- The two previously-known head-to-heads reproduce exactly (0.825 < 1.071, 0.726 < 0.961)
  and the three new ones extend the pattern 2/3 further in the same direction.

## 6. Diagnostics (declared diagnostics, not bars)

- Median baseline reduction over 18: **learned +0.187**, **E2-double +0.381**.
- **E2-double is numerically better than the learned arm on 17/18 windows.**
- **The learned arm is worse than the baseline it was called to repair on 7/18 windows**
  (BS11c_w3, BS11c_w4, BS13a_w1, BS13a_w2, BS13a_w4, BS13a_w5, BS13c_w2) — all held-out,
  all BS-site. E2-double is worse than baseline on 4/18 and never by more than 0.11.
- Windows still ≥ 1.0 after the arm runs: **learned 12/18**, E2-double 7/18.
- Per record, learned repairs: LJ18n 5/5, LJ18g 1/1, AA14 1/1, AA15a 1/1, BS11b 1/1,
  BS13c 1/2, BS13a 1/5, BS11c 0/2. **The learned repair is site-structured**: it works where
  the analytic collapse is catastrophic (LJ18n, base 1.4–7.6) and hurts where the collapse is
  marginal (BS, base 1.03–1.50).
- Learned per-seed sd on held-out windows is up to 0.49 (median 0.18), versus 0.03–0.15 on
  design windows — the arm is also markedly less stable out of sample.

## 7. Honest summary sentence

> On all 18 harvested real-record collapse windows, the learned arm repairs 6/18 by ≥ 0.5
> and lands 6/18 below nRMSE 1.0, against 6/18 and 11/18 for a single frozen GT-free
> reconfiguration of the analytic arm (`kp_hint × 2`); it therefore misses P2's
> pre-registered 1.5× bar (≥ 9/18 and ≥ 17/18) on both criteria, is numerically worse than
> that escape on 17/18 windows and on 4 of the 5 windows the deployed τ = 1.5 detector
> actually fires on, and — newly visible out of sample — makes the error *worse* than the
> analytic baseline on 7/15 held-out collapse windows, so the previously circulated
> "3/3 repairs, 0 own collapses" is a design-set property that does not generalize; under
> the pre-committed rule, Ch.7's interlock is a paragraph about initialization robustness.

## 8. Consequences now triggered (pre-committed; no new interpretation added)

1. **The interlock's fallback need not be learned, and on this evidence should not be.**
   The honest statement is now stronger than the stub in `RESULT.md` §8: it is not merely
   that "the learned arm has not been shown to beat" the analytic reconfiguration — it is
   **measured not to**, on the frozen counts, on the fired subset, and window-by-window.
2. **`docs/HANDOFF_OPUS_2026-07-26.md` §0 and §3 must be corrected.** "reduces error on all
   3 analytic collapse windows … with 0 own collapses" is true of the design set and false
   of the held-out set (7/15 damaged). Any sentence attaching "collapse insurance" to the
   learned arm without the held-out counts is now a known-wrong claim.
3. **The real-data contribution of the learned arm is now empty on both axes measured so
   far**: accuracy (pre-registered LOSS, `wp_real.json`) and collapse repair (this file).
   What survives is the *architecture* — a GT-free detector triggering a fallback — with the
   fallback's identity an open, and currently analytic, question.
4. **The follow-up flagged in `RESULT.md` §8 is now the priority question, not an aside:**
   if `kp_hint` misspecification drives these collapses, a GT-free `kp_hint` self-check or a
   two-config val-selected `kp` would be an entirely analytic fix. It must be
   pre-registered separately; nothing here claims it works.
