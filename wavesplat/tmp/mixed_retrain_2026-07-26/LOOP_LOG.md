# LOOP LOG — mixed-curriculum retrain, autonomous overnight rounds

Bounds (user-authorised 2026-07-26, relayed by coordinator):
* max **3 training rounds total** (round 1 + at most 2 improvement rounds)
* each improvement round preceded by a **prereg addendum**: diagnosis / single change / expected gate
* checkpoints never overwritten: `_mixed_v1.pt`, `_mixed_v2.pt`, `_mixed_v3.pt`
* **gates stay frozen** (PREREG.md Sec 5 bars), one shot per round
* architecture changes remain **off-limits**
* if round 3 still fails → stop, write honest failure analysis; no round 4
* prefer fewer, better-diagnosed rounds; a single round should stay under ~4 h

Frozen bars (PREREG Sec 5): **G1 ≤ 0.17878** · **G2a max per-episode median n_selected ≤ 6** ·
**G2b ≤ 0.01855** · G3, G4 report-only.
Old-net references: G1 0.16878 · G2a 30 · G2b 0.03567 (v2 0.01355) · G3 0.2330 · G4 0.805 (27 normal).

---

## Round 1 — `beamform_cond_diffusion_mixed_v1.pt`

**Launched** 2026-07-26 23:13:47 KST. PREREG.md sha256 `b151c710…`, frozen 23:55 (clock skew: the
prereg was written and hashed before the first training step; `FROZEN.sha256` is the record).

**Change vs the shipped checkpoint (training data only):**
| axis | shipped | round 1 |
|---|---|---|
| cardinality | always 32-line JONSWAP | stratified `nc ∈ {1,2,5,15,30}` + continuum strata, exact 32-slot energy-preserving target |
| (Hs,Tp) box | `U[0.8,3.2] × U[5,11]` (**misses BF2 = 0.3/4.0**) | 50 % jittered BF2–BF5 anchors, 50 % `U[0.25,3.2] × U[3.8,11.5]` |
| continuum | none | `gen_cells.expand()` recipe, 465 sub-components, k-means 32-atom target (25 % of the mixture) |
| obs geometry | pooled footprint × **independent** `t` | `GeometryBank` replays real LiDAR sweep windows (~130 distinct frame times, 3.1 pts/time) |
| steps | 15 000 | 40 000 (cosine anneal), from scratch |

Architecture / schedule / loss / optimiser: **byte-for-byte unchanged**.

**Validity controls run before any gate number was read**
* generator selftest: energy preservation |ΔE| < 2e-7 (replicate) and < 3e-8 (k-means);
  distinct-atom counts sparse 1 / moderate 5 / crowded 15; bank footprint r-percentiles
  10.7/15.6/21.2/31.5/76.7/144.5 vs gate 10.9/15.5/21.4/32.7/82.4/157.5.
* collapse scorer reproduces the recorded old-net counts exactly: held-out repairs **8**, damages
  **7**; below-1.0 **6/18**; analytic escape **11/18**.
* harness-integrity control: the copied `run_arms_new.py` fed the **old** atoms cache reproduces
  the recorded old-net per-episode numbers on all 28 gate episodes,
  **max |Δ| = 0.000e+00 → the copy is the same arm.**
* mechanism diagnostic re-measured on the **old** checkpoint (`mech_diag_OLD.json`): BF2_1comp
  median nearest-atom **0.2355 kp** vs beamform dominant peak **0.0550 kp**, BF2 median
  n_selected **24.5**; all-episode median nearest-atom 0.0012 kp. The D1 signature reproduces —
  the conditioning input locates the swell an order of magnitude better than the proposal does.
  (The recorded 0.742 kp is the median over the five *in-sample slope* BF2 episodes; 0.2355 is
  the median over all four fresh BF2 episodes, which include the two the old net got right.)

**Training curve** (held-out synthetic eps-MSE, reporting only — selects nothing). Monotone,
no divergence, no plateau-then-rise:
| step | 0 | 1k | 3k | 5k | 10k | 15k | 20k | 25k | 30k | 40k |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 1.6407 | 0.11488 | 0.10819 | 0.10029 | 0.09662 | 0.09311 | 0.09205 | 0.08825 | 0.08709 | **0.08619** |

**Completed** 00:32, wall **4 712 s = 78.5 min** (budget said ~1.6 h — on target).
Realised strata vs declared: crowded 0.2998/0.30, sparse 0.2000/0.20, moderate 0.1003/0.10,
continuum_mild 0.0998/0.10, continuum_strong 0.1505/0.15, intermediate 0.1497/0.15.

**Gate results (each gate run exactly once, `gate_results_v1.json`):**

| gate | bar | round 1 | old net | verdict |
|---|---|---|---|---|
| **G1** crowded, n=16 | ≤ 0.17878 | **0.16052** | 0.16878 (v2 0.21845) | **PASS** (margin +0.01826; medianΔ −0.00876, 12/16 wins, p=0.058 — *better* than the old net) |
| **G2a** max per-ep median n_selected, n=12 | ≤ 6 | **12.0** | 30 | **FAIL** |
| **G2b** learned mean nRMSE, n=12 | ≤ 0.01855 | **0.01128** | 0.03567 (v2 0.01355) | **PASS** — and it *beats pure v2* |
| G3 continuum cell_c, n=10 | report only | 0.2291 | 0.2330 | medianΔ −0.0048, 7/10, p=0.064 |
| G4 real accuracy, 27 normal | report only | 0.797 | 0.805 (hand 0.744) | medianΔ −0.0237, 17/27, p=0.194 |
| G4 collapse, 15 held-out | report only | 7 repair / 8 damage | 8 / 7 | below-1.0 6/18 both; new better than old on 12/18 |

**Frozen verdict (PREREG Sec 6): FAIL branch**, because G2a failed. Recorded as such.

**What actually moved.** The targeted defect is fixed: `BF2_1comp` n_selected 30/19/30/16 → 3/6/3/2,
and the catastrophic `BF2_1comp_seed888_slope` episode went 0.3286 → **0.0174** (v2 0.0301). The
whole sparse slice now beats the analytic initializer. G1 improved rather than regressed. What did
NOT move to bar is the count on the *long-wave* 1-comp episodes (BF4, k=0.252) where the old net was
already fine.

**Mechanism diagnostic** (`mech_diag_v1.json` vs `mech_diag_OLD.json`), distance from the true
swell to the nearest SELECTED atom, in kp units, median over 5 diffusion seeds:

| episodes | old net | **round 1** | beamform dominant peak (the net's own conditioning input) |
|---|---|---|---|
| BF2_1comp (k=0.770) | 0.2355 | **0.0018** | 0.055 |
| BF3_1comp (k=0.407) | ~0.002 | **0.0031** | 0.022–0.040 |
| BF4_1comp (k=0.252) | ~0.002 | **0.0060** | **0.197–0.219** |
| all 12, median | 0.0012 | **0.0039** | — |

The relationship **inverted**: the old net was 4× *worse* than the beamform peak on BF2; round 1 is
**10–50× better** than the beamform peak on every 1-comp episode, including BF4 where the analytic
beamformer is itself poor (0.20 kp). The proposal is genuinely super-resolving its conditioner —
which is why G2b passes and beats v2 despite G2a's count failing.

---
**Post-freeze arm-purity audit (00:23, coordinator warning re `init_method='auto'` default since
`aa8e4c3`)** — full table in PREREG.md Appendix A. Result: `run_arms_new.py` sets `init_method`
explicitly for every arm and pins `TRACK_PHASE_PREDICT=False` per worker; the G3/G4 runners never
touch `run_estimation` at all; the quoted `pure_v2` baselines come from `v2_arm.py`, which sets
`'v2_konly'` explicitly. **No contamination.** Changes no bar.

Incident: the first two gate-chain waiters were killed by the harness before firing; the chain was
re-armed at 00:23 while training was at step 34 250/40 000. No gate had run at that point, so no
gate was run twice.

---

## Round 2 — `beamform_cond_diffusion_mixed_v2.pt`

**Addendum:** PREREG.md "ADDENDUM 1", frozen 01:20 before the first round-2 step.
**Launched** 2026-07-27 01:22 KST. Wall estimate ~69 min (104 ms/step measured).

**Diagnosis** (from `atoms_cache_mixed_v1.jsonl`, no new compute): placement is already exact
(nearest-atom 0.0003–0.019 kp); the residual is a **count** problem that is monotone in wavenumber
(BF2 k=0.770 → 3 atoms; BF3 k=0.407 → 3.5; BF4 k=0.252 → 9) because the proposal cloud has roughly
**constant absolute width (~0.09–0.14 rad/m)** while `pool_and_select` dedups on a **constant
absolute radius** (0.03). Round 1's exactly-energy-preserving target gave all 32 slots equal energy,
so every atom cleared the 3 %-of-max energy filter and dedup was the only pruning mechanism.

**The single change:** energy-graded 32-slot target for discrete strata — first `n_true` slots at
full amplitude, the rest co-located at `0.10 × A_max` (1 % of peak energy, below the 3 % filter).
Continuum targets unchanged. Verified: slots surviving the 3 % filter are now
sparse **2**, moderate **5**, crowded **19**, continuum **32** (round 1: 32/32/19/32).

**Expected to move:** G2a. **Risk:** G1, G2b.
**Completed** 02:27, wall **4 162 s = 69 min**, final val **0.08535** (round 1: 0.08619).
Realised strata identical to round 1 to 4 decimals (same master seed → same scenes).

**Gate results (`gate_results_v2.json`):**

| gate | bar | round 2 | round 1 | old net | verdict |
|---|---|---|---|---|---|
| **G1** crowded | ≤ 0.17878 | **0.16379** | 0.16052 | 0.16878 | **PASS** (medianΔ −0.00653, 11/16, p=0.083) |
| **G2a** max n_selected | ≤ 6 | **8.0** | 12.0 | 30 | **FAIL** (but 12 → 8) |
| **G2b** learned mean | ≤ 0.01855 | **0.01092** | 0.01128 | 0.03567 | **PASS** — best of the three |
| G3 continuum | report | 0.2304 | 0.2291 | 0.2330 | medianΔ −0.0013, 8/10, p=0.084 |
| G4 real accuracy | report | 0.800 | 0.797 | 0.805 (hand 0.744) | medianΔ −0.0104, 14/27, p=0.455 |
| G4 collapse (held-out 15) | report | 8 repair / 7 damage | 7 / 8 | 8 / 7 | below-1.0 6/18; better than old on 14/18 |

**The lever worked but was under-turned.** Raw atoms clearing the 3 %-of-max energy filter on the
three worst 1-comp episodes: round 1 **24 / 31 / 13** → round 2 **18 / 19 / 15**; G2a 12 → 8.
Nothing else regressed materially (G1 −0.003 vs round 1, still well inside the bar; G2b improved).

**Secondary observation, recorded, not acted on:** the residual sits on *slope*-raydrop episodes at
long wavelength (BF3/BF4 slope 4/7/8/6 vs uniform twins 2/2/3/2). The `slope` model's specular test
reduces to near-normal incidence, which never fires at this grazing geometry, so it degenerates to
uniform Bernoulli with a longer-range footprint (mean range of valid returns 37.5 m vs 23.4 m) —
and that footprint is already in the geometry bank (6 of 14 donors are slope episodes). With one
round left, the demonstrated lever was preferred over this unconfirmed one.

---

## Round 3 — `beamform_cond_diffusion_mixed_v3.pt`

**Addendum:** PREREG.md "ADDENDUM 2", frozen 03:00 before the first round-3 step.
**Launched** 2026-07-27 03:01 KST.

**Diagnosis:** with `A_fill = 0.10 × A_max` the filler energy is 1 % of peak — only **3×** below the
3 % selection threshold, so the net's own noisy energy predictions still push many fillers over it.

**The single change:** `A_fill = min(0.03 × A_max, 0.5 × A_min_real)`, floored at the representation
floor → filler energy **0.09 % of peak, a 33× margin**; the `0.5 × A_min` cap guarantees a filler is
never stronger than the weakest real component, protecting crowded ordering. Verified: filler/peak
energy ratio 1.0e-2 → **9.0e-4** (sparse/moderate), crowded 2.0e-5 → 4.9e-6, continuum unchanged;
no normalisation clamping.

**Expected to move:** G2a (8.0 → ≤ 6). **Risk:** G1 ordering, G2b.
**Completed** 04:08, wall **4 169 s = 69 min**.

**Gate results (`gate_results_v3.json`) — FULL PASS on all three hard gates:**

| gate | bar | round 3 | round 2 | round 1 | old net | verdict |
|---|---|---|---|---|---|---|
| **G1** crowded | ≤ 0.17878 | **0.17491** | 0.16379 | 0.16052 | 0.16878 | **PASS** (margin +0.00387; medianΔ **+0.00736** vs old, 4/16 wins, p=0.117) |
| **G2a** max n_selected | ≤ 6 | **6.0** | 8.0 | 12.0 | 30 | **PASS** (at the bar) |
| **G2b** learned mean | ≤ 0.01855 | **0.01047** | 0.01092 | 0.01128 | 0.03567 | **PASS** — best of all rounds, beats v2-analytic 0.01355 |
| G3 continuum | report | **0.2287** | 0.2304 | 0.2291 | 0.2330 | medianΔ −0.0043, 8/10, **p=0.0098** |
| G4 real accuracy | report | **0.783** | 0.800 | 0.797 | 0.805 (hand 0.744) | medianΔ −0.0079, 15/27, p=0.178 |
| G4 collapse (held-out 15) | report | 8 rep / 7 dam | 8 / 7 | 7 / 8 | 8 / 7 | **below-1.0 8/18** (old 6/18, analytic escape 11/18) |

The prediction held: filler energy contrast ×11 took G2a 8.0 → 6.0. G2b and both report-only gates
also reached their best values of the three rounds.

**The cost is visible and must be reported:** G1 degrades monotonically with the energy-grading
strength — 0.16052 (r1) → 0.16379 (r2) → 0.17491 (r3), i.e. from **−0.0088 better than the old net**
to **+0.0074 worse**, with the wins-vs-old count falling 12/16 → 11/16 → 4/16. Round 3 still clears
the +0.010 non-inferiority bar (margin +0.00387) and still beats the analytic initializer by a wide
margin (0.17491 vs 0.21845), but the trade is real: **sparse-sea cardinality is bought with crowded-sea
accuracy.**

**No round 4** (bound (1) of the directive: 3 rounds total).
