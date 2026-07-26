# PREREG — W8 second half: non-learned ("cheap analytic") escape controls on the harvested collapse events

**Frozen 2026-07-26, before any escape run was executed and before any escape score existed.**
CPU only; the diffusion model is never imported or loaded. Nothing is committed. No file under
`scripts/` or `wavesplat/` is modified (held-out record entries are injected at runtime by the
already-frozen `tmp/heldout_real_2026-07-26/heldout_cfg.py`, additive only).

**Why this run exists.** Persona-debate objection C12 (P1 objection 5, conceded by P2, endorsed by
P4 as a *literature* objection): *"Until multi-restart greedy, second-peak initialization and
relaxed `min_dk` are run on the same events, 'the learned arm repairs collapses' is not
distinguishable from 'any different initialization escapes a bad basin' — which is preempted by
random restart since the 1980s."* Debate §W8 control bar (P2): *"the learned arm must repair ≥50 %
more events than the best cheap analytic escape; otherwise Ch.7's interlock becomes a paragraph
about initialization robustness."*

---

## 0. Scope correction that must be stated in the result (declared now, not after)

The deployed analytic arm on the real records is **`d3_sa_nomp.mmse_bank_estimate`** — a
*spectrum-weighted linear MMSE over a fixed continuous bank*, not the coarse-to-fine greedy
initializer. Consequences, all declared in advance:

1. **It has no greedy component-selection loop and therefore no `min_dk`.** The third escape named
   in the debate ("relaxed `min_dk`") **does not exist as a knob on this arm** and is replaced by
   E3 (`sigma0` scaling), the other val-facing knob. This is a scope statement, not a substitution
   of convenience: `min_dk` lives in `wavesplat/optimizer/initializer.py`, which the real-record
   hand arm never calls.
2. **Its only stochastic element is the validation split** (`seed` → `rng.choice` picking the 25 %
   val subset used to select `pexp ∈ {0.5,1,2}`, `smult ∈ {1,3,10}` and to gate the SBL-EM steps).
   "Multi-restart" therefore means **val-split randomization**, which is the entirety of the arm's
   run-to-run freedom. It is a *fair* control for "any different initialization escapes a bad
   basin" only in the sense that it is the arm's own randomness; the honest phrasing in the result
   must be "restart = val-split resample", never "random restart of a greedy optimizer".
3. The knobs that *can* move the solution structurally are `kp_hint` (which sets the bank annulus
   `0.35 kp < |k| < 2.5 kp` **and** the bank spacing `dk_c = kp/10`) and `sigma0` (the noise floor
   the val-selected multiplier grid rides on). E2 and E3 are those two.

---

## 1. Target set (frozen)

**18 collapse windows = 15 held-out harvest + 3 design.** Collapse definition is the already-frozen
one, `hand_now ≥ 1.0`, applied to the baseline hand arm; no window is added or dropped here.

- **Held-out 15** (from `tmp/heldout_real_2026-07-26/harvest_heldout.json`):
  BS11b_w4, BS11c_w3, BS11c_w4, BS13a_w0, BS13a_w1, BS13a_w2, BS13a_w4, BS13a_w5,
  BS13c_w2, BS13c_w3, LJ18n_w0, LJ18n_w1, LJ18n_w3, LJ18n_w4, LJ18n_w5.
- **Design 3** (from `tmp/unified_2026-07-26/hand_amp.json`): AA14_w0, AA15a_w3, LJ18g_w3.

**Baseline hand nRMSE** per window is read verbatim from those two files and is **also recomputed by
this runner** as a bit-identity control; the run is invalid if any window's recomputed baseline
differs from the stored one by more than `1e-9`.

**Metric** (unchanged from the harvest): lead-0 (nowcast) field nRMSE on the query grid,
`sqrt(mean((pred - gt)^2)) / std(gt_pooled)`, finite-mask ≥ 8 px. Nothing else is an endpoint.

## 1.1 Frozen success definitions — two counts, **never merged**

For an escape reading `e` on a window with baseline `h`:

- **(a) LARGE REDUCTION**: `h - e ≥ 0.5`.
- **(b) BELOW-LINE**: `e < 1.0`.

Reported as two independent counts out of 18. They are not summed, averaged, intersected, or
described by a single number. (This is the structural fix for the historical 3/3-vs-2/3 error.)

## 2. The escapes — exact frozen grids (no post-hoc additions permitted)

All escapes call the **identical** frozen hand-arm entry point on the **identical** scene:

```
mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                   sigma0=0.05*sc['Hs'], kp_hint=sc['kp'], seed=sc['seed'])
```

with exactly one argument perturbed per variant. `sc = d3_compare_gifs.scene_real(record, win=win)`,
unmodified. Baseline symbols: `s_base = sc['seed']` (= frame index `i0`), `kp_base = sc['kp']`,
`sig_base = 0.05*sc['Hs']`.

### E1 — seed multi-restart (val-split resample), 5 variants
`seed ∈ { s_base+1000, s_base+2000, s_base+3000, s_base+4000, s_base+5000 }`.

- **E1-median** = median of the 5 nRMSE values. **This is the deployable reading** (no GT needed).
- **E1-best** = min of the 5. **ORACLE-FLAVORED** — requires ground truth to pick the winner. It is
  reported for diagnosis only and is *not* eligible for the P2 comparison bar.

### E2 — `kp_hint` perturbation, 3 variants
- **E2-peak2**: `kp_hint = |k|` of the **second** dispersion-constrained beamform peak of the
  observations. Frozen extraction (pure numpy, no torch, no GPU), mirroring the v2 initializer's
  peak logic (`wavesplat/optimizer/initializer.py::beamform_peak_ratio`) transposed to scattered
  space-time samples:
  1. power map `p(kx,ky) = (Σ z cosψ)² + (Σ z sinψ)²`, `ψ = kx·x + ky·y − ω(k)·t`,
     `ω = sqrt(g|k| tanh(|k| d))`, over `kx,ky ∈ linspace(−klim, klim, 121)`,
     `klim = 3.5·kp_base` (the same zoom half-range `mmse_bank_estimate` uses in its own
     second-stage beamform when `kp_hint is None`);
  2. zero the 3×3 cell block nearest `k = 0` (DC block, as in `beamform_peak_ratio`);
  3. peak 1 = argmax; zero a **box of half-width `R = 0.5·kp_base`** around it;
  4. peak 2 = argmax of the remainder; `kp_hint = max(hypot(kx₂,ky₂), 1e-3)`.
  **Declared choice, not fitted:** the suppression radius is scale-free (`0.5·kp`) rather than the
  shipped absolute `MIN_DK = 0.15 rad/m`, because on this target set `kp` spans 0.068–0.738 rad/m,
  where 0.15 absolute would be 2.2·kp on LJ18n_w0 and 0.20·kp on BS13c_w2 — an absolute radius is
  not transferable across the records and would make "second peak" mean two different things. If
  peak 2 has zero power the variant is recorded `null` for that window and counted as a
  non-repair.
- **E2-half**: `kp_hint = 0.5 · kp_base`.
- **E2-double**: `kp_hint = 2.0 · kp_base`.

### E3 — `sigma0` scaling, 2 variants
- **E3-half**: `sigma0 = 0.5 · sig_base`.
- **E3-double**: `sigma0 = 2.0 · sig_base`.

### Per-window best-variant readings
For E2 and E3 (and for E1 via E1-best) a **per-window best-variant** number is also reported and is
**labeled ORACLE-FLAVORED**: it selects the winning configuration using the ground truth the
operator does not have. Oracle-flavored readings may not be used to satisfy or to fail any bar.

### Total runs
18 windows × (1 baseline re-check + 5 E1 + 3 E2 + 2 E3) = **198** `mmse_bank_estimate` calls,
plus 18 beamform power maps. ≤ 6 CPU workers, `OMP/OPENBLAS/MKL=1`, `nice -n 10`.

## 3. Frozen comparison protocol (executed later — learned arm is PENDING-GPU)

The learned arm's repair counts on these same 18 windows do not exist yet (GPU reserved; the
diffusion model is not loaded by this run). When they exist:

- **Deployable comparator** `C_best(criterion)` = the **maximum count over the six single-config,
  GT-free readings**: {E1-median, E2-peak2, E2-half, E2-double, E3-half, E3-double}, computed
  separately for criterion (a) and criterion (b). Taking the max over six configs is deliberately
  **generous to the control** (it lets the control be chosen with hindsight while the learned arm
  may not be), and this asymmetry must be stated wherever the comparison is reported.
- **P2's bar, per criterion, applied separately:** `learned_count ≥ 1.5 × C_best`.
  - bar met on a criterion ⟹ that criterion supports "the learned repair is not merely
    re-initialization".
  - bar not met on a criterion ⟹ for that criterion, Ch.7's interlock is **a paragraph about
    initialization robustness**, in those words.
- **Design-set spot check available now** (n = 3, underpowered, reported as such): the learned arm's
  known design repairs are AA14_w0 2.309 → 1.071, AA15a_w3 1.120 → 0.791, LJ18g_w3 2.925 → 0.961,
  i.e. **(a) 2/3, (b) 1/3**. Any cheap escape that matches or beats those two counts **on the same
  3 windows** is reported immediately, with the n = 3 caveat attached; it does not settle the bar,
  which is defined on all 18.
- No escape reading may be substituted for, proxied onto, or averaged with a learned reading.

## 4. EXPLORATORY (declared exploratory *in advance*; no claim, no deployable threshold)

Using all **78** real windows already scored (48 held-out + 30 design), with their
`amp_ratio_grid` and hand nRMSE: characterize whether **any** two-sided amplitude rule
(`amp < lo` OR `amp > hi`) could separate collapses (`hand ≥ 1.0`) from normal windows
(`hand < 1.0`), with specific attention to the *suppressed* (low-amp) mode that the frozen
one-sided τ = 1.5 is structurally blind to.

- Report the **honest overlap**: the low-amp range of normal windows against the low-amp range of
  suppressed collapses, the best attainable operating point on this data, and the fact that it is
  **fitted on the same data**, hence not an estimate of anything.
- **No threshold produced here is deployable, proposed, or adopted.** If the overlap is large, say
  so; if a low-side rule is not clean, say that. The expected outcome (stated before looking) is a
  large overlap and no clean low-side rule.
- This section may not be cited as a result and may not be used to revise τ.

## 5. Deliverable

`RESULT.md` with: the bit-identity control outcome; frozen-cell counts (a) and (b) per escape,
deployable vs oracle-flavored kept visually separate; the design-3 spot check against the known
learned repairs; the honest one-sentence summary; the PENDING-GPU comparison stub carrying the
`1.5×` rule and the six-config `C_best` definition; and the exploratory section under its own
heading with its no-claim label.

**Invalidation conditions:** any baseline mismatch > 1e-9; any modification of a file under
`scripts/`, `wavesplat/`, or `data/`; any GPU/diffusion import; any escape added after this file
was written.
