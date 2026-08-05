# fable red-team + independent eval — Better-net (comp-diverse γ) 2026-07-22

> Prereg: `tmp/BETTERNET_prereg_2026-07-22.md`. Job: (1) validity red-team of the "15-comp =
> sparse/low-rank swell regime" framing, (2) design red-team of gap_ladder/wavenumber_id_crb/
> challenge_refine/betternet_eval, (3) independent eval of `prior_net_compdiverse.pt` on
> held-out data with an independently-written harness. All three done. Headline: **eval =
> dual-verified FAIL (candidate net regresses on BOTH regimes)**; **validity = weak (ARGUED),
> the "15-comp" regime is not a sound proxy for real narrow-band swell — it's a discretization
> artifact, and the whole eval chain commits a textbook inverse crime.**

## 1. ★VALIDITY RED-TEAM — verdict: WEAK (ARGUED, code-confirmed)

**Claim under test**: does the 15-comp "sparse/low-rank" regime represent something
deployment-relevant (narrow-band real swell, or sparse-observation deployment), such that
closing its gap matters for the thesis?

### 1.1 `n_components` is a discretization/quadrature-density knob, NOT a spectral-width knob
Read `wavesim/wave_generator.py:jonswap_components` directly (MEASURED, code-level):
- `omega_min, omega_max = 0.5*omega_p, 3.0*omega_p` is **fixed regardless of `n_components`**.
  `n_components` only sets how many `linspace` bins tile this **same fixed band**; each bin's
  amplitude is `sqrt(2*S(omega_center)*d_omega)`, i.e. straight quadrature of the **same**
  JONSWAP `S(ω)` (same `gamma`, `Tp`, `Hs`). Direction is an i.i.d. `Normal(mdir, spread)` draw
  **per component**, also independent of `n_components`.
- Confirmed empirically on the actual eval data: `data/episode_BF4_{1,5,15,30}comp_seed0_uniform.h5`
  all have **identical** `Hs=1.2, Tp=7.0, gamma=3.3, sea_state=BF4`; only `n_components` differs.
  So "15-comp" vs "200-comp" (bwtask/compdiverse broadband) are the **same nominal spectral
  shape**, sampled with 15 vs 200 Monte-Carlo/quadrature draws.
- `scripts/d3_prior_net_data.py`'s `--nc_diverse` path confirms the same: `NC_SET=[12,20,...,200]`
  is drawn **independently of** `gamma_J~U[1.5,6]` and `spread~U[15,45]deg` — nc-diversity never
  touches the parameters that control real physical spectral width.
- Real narrow-band swell in the literature is characterized by **frequency spread σf and
  directional spread σφ** (continuous quantities) — e.g. "a low spectral bandwidth value
  indicates a narrow spectrum...associated with long-period swell" (Nortek coastal-monitoring
  wave-parameter note; Qp peakedness parameter). Nowhere is swell narrowness operationalized as
  "few discrete plane-wave atoms." ⟹ **the prereg's own flagged instinct is correct: component
  count ≠ spectral width, and this eval chain uses the wrong knob for the "swell realism" claim.**

### 1.2 The whole gap_ladder/wavenumber_id_crb/challenge_refine/betternet_eval chain is an inverse crime
"Inverse crime" (Colton & Kress terminology, standard in inverse-problems lit): using the **same**
finite-parameter model class to both generate synthetic "truth" and to perform the inversion
artificially inflates achievable reconstruction accuracy (WebSearch-confirmed: "results become
overoptimistic... misleadingly confident reconstructions").
- `oracle_ls_pred` in `gap_ladder.py`/`challenge_refine.py` fits **exactly** the true finite atom
  set (`theta[:,1], theta[:,2]`, N=15 components) with plain LS — and gets corr≈0.999. That's not
  surprising: the "truth" field literally **is** 15 sinusoids, so a 15-atom LS fit on 250 obs
  (over-determined ~4:1) recovers it almost exactly. This is definitionally what an inverse crime
  produces.
- No episode's `z` observations pass through anything but `predict(theta)+Gaussian(σ=0.02)` — the
  same closed-form linear model class used at inversion time. No HOS, no real LiDAR raydrop /
  incidence-angle physics, no `frames/lidar_points`-measured elevation is ever used, despite the
  episode files storing them.
- **⟹ this directly triggers the project's own standing directive** (`feedback_validation_
  nonlinearity.md`, memory): *"검증 데이터셋엔 항상 HOS 비선형 포함, 헤드라인은 HOS서 재현"*. None
  of gap_ladder.py / wavenumber_id_crb.py / challenge_refine.py / betternet_eval.py touch HOS —
  **this is a standing-directive gap, not just a caveat**, for every headline number built on this
  chain today (+0.45 headroom, "REALIZABLE", the comp-diverse PASS bar itself).
- Practical implication: the oracle-LS≈CRB≈1.0 ceiling, and therefore the "+0.45 headroom" the
  whole better-net project chases, is **inflated by construction**. On any non-inverse-crime eval
  (HOS z, or a continuous/high-nc "true" field fit by a necessarily-finite estimator) the ceiling
  would sit below 1.0 by an amount nobody has measured yet — this was already flagged as a caveat
  in `WAVENUMBER_ID_CRB_RESULT`/`GAP_LADDER_RESULT` ("synthetic 낙관...천장<1") but I'm elevating
  it: it isn't a vague scope caveat, it's a **named, well-understood methodological pitfall** that
  the current design falls squarely into, repo-wide (not new to today — `data/episode_BF*_*comp_*`
  has always been generated this way — but today's whole under-det arc rests its headline number
  on it without HOS reproduction, which is the repo's own bar for headline claims).

### 1.3 Steelman for relevance (both sides argued honestly)
- **For**: sparse ship-LiDAR deployment genuinely is under-determined (250 pts is fixed and real).
  A net that is more robust to "energy concentrated in a few dominant modes" (which does happen,
  loosely, whenever `gamma` is large / spread is narrow, even at nc=200) is not obviously useless.
  The observation-sparsity axis (nobs=250, fixed both regimes) is deployment-real and untouched by
  my critique.
- **Against (dominant)**: the specific axis varied here (nc, holding gamma/spread/Tp/Hs fixed) is
  not what produces physical narrow-bandedness, and the "achievability" ceiling used to justify
  chasing this gap is inflated by inverse crime. The honest deployment-relevant experiment would
  hold `n_components` fixed at a large, representationally-adequate value (200) and vary `gamma`
  (toward 6+) / `spread` (toward 15° or below) as the swell-narrowness axis, then check whether
  net-MMSE underperforms oracle there — that test has **not been run**, so "does the net need help
  with narrow real swell" remains untested even after today's work.
- **Verdict**: **ARGUED, not deployment-established.** The "better net" motivation (bank-ceiling:
  net already beats oracle-γ on broadband but not on sparse/15-comp) is a real, reproducible
  *synthetic-regime* finding, but its bridge to "real narrow-band swell" or "sparse deployment"
  is currently asserted, not shown. This should be labeled a synthetic-regime capability gap, not
  a swell-realism gap, until the gamma/spread-varying experiment is run.

## 2. Design red-team — FATAL / FIX / OK

1. **FIX (real, but doesn't overturn today's verdict)** — **pseudo-replication in the n=16 sparse
   episode set.** Verified by direct h5 inspection: for a given seed, `_fixedT`, `_slope`,
   `_uniform` share **byte-identical** `wave_components` (only LiDAR sampling geometry differs);
   `_fixedT_dir180` is a deterministic 180°-mirror of the same draw. So Opus's default sparse
   eval (`sorted(glob(15comp))[:16]` = seed0-3 × 4 geometry variants) has **4 independent sea
   draws, not 16**. Bootstrap CIs computed over the pooled 16 understate true uncertainty. Same
   bug applies to `gap_ladder.py`, `wavenumber_id_crb.py`, `challenge_refine.py`. — I re-aggregated
   per independent seed in my own eval (§3) and the verdict is unchanged (still a clear FAIL), so
   this doesn't rescue today's result, but it should be fixed before any future PASS is trusted.
2. **FIX (minor, citation-accuracy)** — the **30-comp code path is dead** in `gap_ladder.py` /
   `challenge_refine.py` / `wavenumber_id_crb.py`: `sorted(glob(15comp)+glob(30comp))[:16]` — since
   `'15comp' < '30comp'` lexically and 25 15-comp files exist, the slice never reaches any 30-comp
   file at default n. `GAP_LADDER_RESULT`/`CHALLENGE_RESULT`/`WAVENUMBER_ID_CRB_RESULT` all say
   "under-det 바다(15/30comp)" — this should read **"15-comp only"**; all headline numbers in
   those files are from a single (Hs=1.2,Tp=7.0,γ=3.3,BF4,nc=15) point, not a 15/30 mix.
3. **FIX (real, likely explains most of the observed FAIL)** — **severe train-data-volume
   confound.** `prior_net_compdiverse.pt` trained on `tmp/prior_train/compdiverse_*` = **8 shards
   × 500 = 4,000 samples**; `prior_net_v3.pt` (dated 2026-07-05, predates `bwtask` by 3 days) was
   almost certainly trained on `lin1` = **80 shards × 500 = 40,000 samples** — a **10× data
   disadvantage** for the candidate, confounded with the nc-diversity manipulation. Training log
   (`tmp/train_compdiverse.log`) shows val loss bottoming at **step 5000/20000** (best-val
   checkpoint) then **monotonically rising** through step 20000 — classic small-N overfitting (≈89
   epochs over 3,600 train examples at bs=64). ⟹ **A FAIL here cannot cleanly falsify "comp-diverse
   training helps"** — it falsifies *this particular underpowered execution*. A fair test needs
   matched (or larger) total data volume across the nc ladder before attributing outcome to
   nc-diversity per se.
4. **OK** — the sparse-vs-broadband regime split and the general PASS-bar logic
   (Δ≥+0.05 CI-excl-0 sparse AND Δ≥−0.02 broadband) are reasonable in isolation; `bwtask` is
   genuinely temporally prior to and disjoint from `prior_net_v3`'s training (mtimes: bwtask
   2026-07-08 vs v3 2026-07-05), so broadband held-out-ness for v3 checks out (MEASURED, verified
   by file mtimes) — no leakage found there.
5. **Answer to "is component-count a fair axis, or is spectral width the more honest knob"**:
   **confirmed spectral width (gamma/spread) is the more honest knob** (§1.1) — this should be
   the axis for any follow-up claiming swell relevance.

## 3. Independent eval — **dual-verified FAIL**

Wrote `tmp/BETTERNET_fable_indep_eval.py` (independent harness, not a copy of `tmp/betternet_
eval.py`) using **held-out data Opus's default script never touches**:
- Sparse: `data/episode_BF4_15comp_seed{4..9}_*.h5` (9 files, indices 16-24 of the sorted 25-file
  glob — Opus's `[:16]` slice only reaches indices 0-15). 6 independent sea draws (seed4-9).
- Broadband: `tmp/prior_train/bwtask_s000.npz` (Opus's `eval_broadband` takes `[-3:]` = s001-003;
  s000 is untouched by either net's training or Opus's eval).
- Different RNG stream (obs-subsample/phase seed=1, not 0).
- Sanity check: net_v3 vs itself on 3 episodes gives **Δ=0.000 exactly** on all three (harness has
  no bug; net-swap is doing real work).

**Result (MEASURED, `results/direction3/betternet_fable_indep_eval.json`)**:

| regime | v3 median | cand median | Δ(cand−v3) | CI | win |
|---|---|---|---|---|---|
| SPARSE, pooled n=9 | 0.476 | 0.343 | **−0.144** | [−0.164,−0.120] | 0/9 |
| SPARSE, per-independent-seed n=6 | — | — | **−0.153** | [−0.172,−0.135] | 0/6 |
| BROADBAND, held-out shard, n=80 | 0.430 | 0.313 | **−0.090** | [−0.124,−0.057] | 21/80 |

**Cross-check**: ran Opus's own `tmp/betternet_eval.py` on the same checkpoint (its own,
non-held-out, non-overlapping-with-mine episode/shard slice):
`sparse Δ=−0.143 CI[−0.166,−0.118]`, `broadband Δ=−0.081 CI[−0.111,−0.053]`
(`results/direction3/betternet_eval.json`) — **within ~0.01 of my independently-derived numbers**
on completely disjoint held-out data, from independently written code. This is strong
double-verification that the effect is real (a genuine property of the checkpoint), not a bug or
an unlucky held-out sample in either script.

**VERDICT on `prior_net_compdiverse.pt` (current checkpoint) = clear FAIL, both regimes regress**,
in the **opposite direction** from the prereg's prediction (predicted: sparse gain, broadband
neutral; observed: both worse, sparse regression larger). Most likely cause (ARGUED, given §2.3):
the 10× data-volume disadvantage + early-stopped/overfit small-N training, not evidence that
comp-diverse training is a bad idea per se — but as executed, **this is not a PASS and should not
be reported as one.**

## 4. Bottom line for Opus

- **Do not report PASS.** Both my independent harness and your own script agree: current
  checkpoint fails on sparse (−0.14 to −0.19) and fails the no-regression bar on broadband
  (−0.08 to −0.09), consistently across two harnesses and disjoint data.
- **Before retrying**: match training-data volume to `v3`'s (~40k, not 4k) — the current gap is
  large enough that undertraining alone plausibly explains it, so a bigger-data rerun is the
  natural next diagnostic, not a redesign.
- **Validity ceiling on the whole arc, independent of whether a rerun passes**: the "+0.45
  headroom" / "REALIZABLE" framing this whole chain rests on is an inverse-crime artifact (§1.2)
  and has never been checked against HOS, which the project's own standing directive requires for
  headline claims. Even a clean comp-diverse PASS would only be a synthetic-regime capability
  demo, not a validated swell/deployment result, until (a) HOS reproduction and (b) a gamma/spread-
  varying (not nc-varying) version of this test are run.
- Fix the pseudo-replication (§2.1) and dead 30-comp path (§2.2) before citing n=16 CIs in
  writing.

**Tags**: §1.1/§1.2/§2.2/§2.3/§3 = MEASURED (code-read + file inspection + reproduced numbers).
§1.3 relevance verdict = ARGUED (explicit both-sides reasoning, no new experiment run). §2.1
pseudo-replication mechanism = MEASURED (h5 byte-comparison); its quantitative effect on
previously-reported CIs = not re-derived (would need rerunning gap_ladder/wavenumber_id_crb with
per-seed aggregation — flagged, not fixed, here).
