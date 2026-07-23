# Hunting the cause of the bimodal variance — three hypotheses, three refutations, one real by-product — 2026-07-23

Follows `tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md` (bimodal run-to-run variance, CV up to
53 % on a fixed episode) and its independent confirmation
`tmp/VARIANCE_VERIFY_fable_2026-07-23.md` (same seed reproduces bit-identically ⟹ genuinely a
seed effect; freezing the initial phase draw collapses CV to 0.0 %).

Every experiment below states its decision rule **before** the numbers, uses the same
8 seeds, and runs the CV = 53 % case (`BF3 2comp`) plus a crowded control (`BF4 15comp`).

## H1 — the energy prune threshold → **NOT THE LEVER**
`tmp/prune_threshold_fix.py`. Sweeping `prune_threshold` 0.10 → 0.02 → 0.005 (a 20× cut in
the discarded energy tail):

| episode | thr 0.10 | thr 0.02 | thr 0.005 |
|---|---|---|---|
| BF3 2comp | CV 53.0 %, min 0.0524 max 0.1682 | CV 57.4 %, **min 0.0524 max 0.1682** | CV 57.2 %, **min 0.0524 max 0.1682** |
| BF4 15comp | CV 2.8 % | CV 4.0 % | CV 4.3 % |

Min and max are **identical to the last digit** across the sweep. ⟹ the energy cut has no
influence on which basin a run lands in. **This refutes half of the mechanism fable
proposed** (the energy criterion collapsing N=2→N=1): the prune is a *symptom*, downstream
of the fork, not its cause. (fable's phase-freeze result is untouched and still holds.)

## H2 — active random padding components → **REFUTED (branch never executes)**
`initialize_from_frames_v2` pads with random `kx, ky` (`rng.randn`, std 1.0), random phase,
and amplitude **0.01** when fewer peaks are found than `n_gaussians`. Plausible source of
spurious energy. `tmp/pad_w_fix.py` set that amplitude to exactly 0:

Results at `PAD_W = 0.01` vs `0.0` are **byte-identical on both episodes** (mean, sd, CV,
min, max all unchanged). Not merely similar — identical ⟹ the padding branch never runs for
these episodes. Hypothesis dead, and the code path is irrelevant here.

## H3 — the `v2_konly` random phase re-draw → **REFUTED as the variance cause; real accuracy by-product**
Reading `initialize_from_frames` (`initializer.py`, `v2_konly` branch) shows v2 fits `(w, φ)`
for every detected component by least-squares linear projection and the `v2_konly` path —
the project's **documented default** — discards them:

```python
kx, ky, w, phi = initialize_from_frames_v2(...)
# Keep beamforming kx/ky but randomize w/phi
# so optimizer explores the amplitude/phase landscape freely
init_w   = torch.ones(n_gaussians) * 0.01
init_phi = torch.rand(n_gaussians) * 2 * math.pi
```

`tmp/konly_phase_fix.py` compares that against keeping the fitted values
(`KONLY_RANDOM_WPHI = False`; the flag defaults to `True`, so **shipped behaviour is
unchanged**).

| episode | random w/φ (current) | fitted w/φ | paired Δ | SE | t | improved |
|---|---|---|---|---|---|---|
| BF3 2comp | mean 0.0930, CV 53.0 % | mean 0.0773, **CV 47.8 %** | −0.0156 | 0.0249 | −0.63 | 4/8 |
| BF4 15comp | mean 0.2895, CV 2.8 % | mean 0.2797, **CV 2.3 %** | **−0.0098** | 0.0031 | **−3.19** | **7/8** |

1. ⛔ **Not the variance fix.** CV falls only 53.0 → 47.8 % and the bad basin persists — it
   simply moves to a different seed (0.1666 now appears at seed 4). The pre-registered
   requirement was a *collapse*; it did not happen.
2. ✅ **But a real accuracy gain on the crowded episode**: −0.0098 nRMSE, **t = −3.19,
   7/8 seeds improved** (MEASURED). On the 2-comp episode the change is **not** significant
   (t = −0.63). ⟹ the "so optimizer explores the amplitude/phase landscape freely" rationale
   is not paying for itself where the problem is crowded; the least-squares values v2 already
   computed are better there — and they are **free**, since they are currently thrown away.
3. **Scope:** n = 2 episodes. Generalization is **ARGUED**, not measured. Do **not** change
   the shipped default on this evidence — it needs a sweep across the episode set first.

## Where the remaining stochasticity lives (next target, not yet tested)
With the initializer's phase draw removed, runs still diverge ⟹ another random consumer
remains. The leading suspect is the **adaptive-N growth path**:
`optimizer/coarse_to_fine.py:152` draws `init_phi = torch.rand(n_new) * 2π` for every newly
grown component. Same defect, different site. `initialize_from_frames_v2`'s internal RNG is
the secondary suspect.

## What this changes about the programme — **nothing yet**
The practical conclusion from `ESTIMATOR_VARIANCE_RESULT` is unaffected: per-episode numbers
still need **median-of-k with spread**, not single-seed point estimates, and GT-free
best-of-k selection still fails (`tmp/RESTART_SELECTION_RESULT_2026-07-23.md`). The mechanism
hunt is a bonus track; three of three hypotheses died, which narrows the search but has not
yet produced a fix.

## Code changes made (both behaviour-preserving)
- `initializer.py`: `PAD_W` and `KONLY_RANDOM_WPHI` module constants, defaults reproducing
  historical behaviour exactly (verified: control arms reproduce the original numbers to the
  last digit).
- `run.py` (earlier): `obs_loss` / `avg_obs_loss` exposed — pure addition.
