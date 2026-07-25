# Pre-registration — WavePrior STAGE-2: e2e-style field-loss fine-tune of the cond-diffusion proposal net

Written and **frozen before any eval run** (2026-07-25). Companion result:
`tmp/wp_e2e_2026-07-25/WP_E2E_2026-07-25.md`. Freezes the fine-tune route, training distribution,
the (max 2) hyperparameter configs + selection rule, the eval endpoints, seed protocol and
decision rule so the read-out cannot be chosen after seeing the numbers.

## 0. Question (confirmatory + falsifiable)

Stage-1 (`tmp/wp_integration_2026-07-25/`) established, PASS, that the learned set-proposal wired
as `init_method='learned'` beats `v2_konly` on the 40 crowded finemesh episodes (tracked median
Δ −0.062, p=5e-11), and that **the entire advantage is the learned kx/ky component placement**
(`learned_konly ≈ learned`, Δ +0.005, p=0.94). Stage-1's recommended Stage-2 is a **proposal-level
field-loss fine-tune with w/phi LS-fixed, concentrating gradient on placement — NOT backprop
through the tracker's Adam loops.**

STAGE-2 asks: **does fine-tuning the proposal net against a downstream FIELD loss beat the current
checkpoint** at the proposal-level nowcast? Known risk: retrain has FAILED to beat hand/frozen on
two prior axes (γ / bank-MMSE). A null/negative here is a legitimate, reportable outcome that would
bound "e2e helps" and leave Stage-1's modular result as the product.

## 1. Route through the DDPM-sampler blocker (frozen) — x0-hat field-consistency fine-tune

The eval PRIMARY runs the **identical frozen ancestral DDPM sampler + 80-iter polish +
pool_and_select** on the fine-tuned weights (same `cd.sample`/`cd.polish_atoms`/`cd.atoms_to_field`
as the current ckpt), so fine-tuning must improve the eps-predictor's implicit **x0-hat decode**,
which every one of the 500 ancestral steps computes. We therefore **supervise x0-hat directly**
(blocker #1's "supervise the x0-prediction directly" option, `WP_INTEGRATION §10`).

**Why this ONE route (minimal risk):**
- No change to the eval sampler → apples-to-apples eval (same inference, different weights).
- No distillation / new set-prediction head → no architecture risk, no second model; the saved
  checkpoint is an **identical `CondSetDenoiser` state_dict** that `learned_init.py` / precompute /
  `run_cond_crowded` load unchanged.
- No backprop through 500 stochastic no_grad steps (avoids the memory/instability of an unrolled
  DDPM, and the train/eval mismatch of a few-step DDIM proxy vs the 500-step ancestral eval).

**Objective (per step, batch B=32):** synthetic scene → GT atoms x0, beamform map, obs (P,z), true
field on a held-out 20×20 grid at nowcast time t=Tobs/2. Per-sample timestep: anchor half over
[0,T), field half over [0, t_lo). x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·eps; net → eps_pred.
- **anchor** = MSE(eps_pred, eps) over ALL samples, all t — preserves the generative model and the
  **energy channel** that `pool_and_select` uses at eval.
- **field** (field-half only) = decode x0-hat=(x_t−√(1−ᾱ_t)·eps_pred)/√ᾱ_t, clamp(−1,1) (the
  samplers' bound decode) → `denorm_atoms` → **differentiable torch ridge-LS** (λ=0.5, matches
  `fit_amplitudes_phases`) solves w/φ on the obs → render on the held-out grid with the SAME (kx,ky)
  → mean (RMSE/Hs)² vs the true field.
- **total = anchor + λ_field · field.** Because w/φ are an LS function of kx/ky and amplitude is
  re-solved, the field gradient lands almost entirely on **kx/ky placement** — the Stage-1
  mechanism quantity. grad-clip 5.0.

## 2. Training distribution (frozen — UNCHANGED from the current ckpt)

Training scenes use **`cd.make_batch`'s exact distribution** (mirrored in `make_scene_batch`, which
additionally returns the obs and true comps needed for the field target): 32-component JONSWAP,
Hs~U[0.8,3.2], Tp~U[5,11], dir~U[−50,50], spread~U[12,40], γ~U[1.5,4.0], depth=50, 400 obs over the
`pool_xy` LiDAR footprint, times U[0,2.2·Tp], noise 0.02, **linear** forward (make_batch default
hos_frac=0). Scenes are freshly sampled synthetic every step → **held-out from the fixed finemesh
eval episodes** (which are separate HDF5 files). The distribution is NOT changed beyond returning
the extra tensors the field target needs.

## 3. Hyperparameter configs (frozen — MAX 2) + selection rule

| config | λ_field | t_lo | lr | steps | bsz |
|---|---|---|---|---|---|
| **A** (light) | 1.0 | T/4=125 | 5e-5 | 4000 | 32 |
| **B** (strong) | 4.0 | T/4=125 | 5e-5 | 4000 | 32 |

Both fine-tune from `results/direction3/beamform_cond_diffusion.pt` (read-only), AdamW, cosine LR,
wd 1e-4. **NO further hyperparameter iteration beyond these two.**

**Selection rule (FROZEN, no test peeking):** after training both, evaluate each config AND the
current ckpt on a **held-out synthetic validation set** = 16 fixed synthetic scenes (seed 12345,
n_components ∈ {15,30} to match the test regime, otherwise the training distribution) run through
the **full frozen eval pipeline** (ancestral `cd.sample` n=4 → `cd.polish_atoms` 80 → `atoms_to_field`
BMA field nRMSE). **Chosen config = lower MEDIAN validation field nRMSE.** The chosen config is the
sole one carried to the PRIMARY eval. (If both configs are WORSE than the current ckpt on val, the
better of the two is still carried to PRIMARY and the likely FAIL is reported faithfully.)

## 4. Primary endpoint (frozen)

- **Arms:** `ft` (chosen fine-tuned ckpt) vs `current` (`beamform_cond_diffusion.pt`).
- **Protocol:** the Stage-1 'cond' protocol, byte-identical via reuse of
  `tmp/wp_crowded_2026-07-25/run_cond_crowded.eval_episode`: per (episode, seed) the 4-sample BMA +
  80-iter polish nowcast at the 3 frames `sel=linspace(nb+3,len−2,3)`, `/Hs`; **median-of-5
  diffusion sampling seeds (0,1,2,3,4)** per episode per arm. Data `data_finemesh_2026-07-23/`.
- **Episode set (PRIMARY):** the exact 40 crowded finemesh episodes
  (`episode_BF{4,5}_{15,30}comp_seed{0..4}_{uniform,slope}.h5`).
- **Comparison unit:** per-episode paired median-of-5 `cond_polish_nrmse_Hs`.
- **Baseline source:** the `current` cond numbers are **reused from the head-to-head
  `cond_runs.jsonl`** (same ckpt, same protocol, same seeds/frames/data). Reproducibility is
  **verified before the readout** by re-running the current ckpt through *this* harness on a
  3-episode × 5-seed spot check; median-of-5 must match the recorded values to |Δ|<0.01 on all 3,
  else the full current arm is re-run.
- **Test:** one-sided Wilcoxon signed-rank on the 40 paired deltas `Δ = ft − current`, H1: ft <
  current (Δ<0), **α=0.05**. Effect size = matched-pairs rank-biserial; report median Δ, wins, p.

### Decision rule (FROZEN)
Primary **PASS** iff BOTH:
- **P1:** one-sided Wilcoxon p < 0.05, AND
- **P2:** median paired Δ < 0.

Otherwise **FAIL** (reported faithfully: full 40-row table, median Δ, wins, p, effect size). A FAIL
bounds "e2e fine-tuning helps at the proposal level" and leaves Stage-1's modular result as the
product; it is an anticipated outcome given the gamma/bank-MMSE retrain-failure precedent.

## 5. Secondary endpoints (reported, NOT decision-primary)

- **(a) Integrated-arm spot check — do gains propagate through the tracker?** 10 crowded episodes
  (frozen §7), 5 paired lanes (est_seed,diff_seed)=(42,0),(1,1),(2,2),(3,3),(4,4). GPU-precompute
  the **ft-ckpt** atoms, run `init_method='learned'` under the phase-fixed tracker
  (`TRACK_PHASE_PREDICT=False`), tracked `avg_nRMSE` median-of-5. Paired vs the **Stage-1 old-ckpt
  learned** results (`arm_runs.jsonl`, same episodes/lanes) — same-ckpt-family paired delta;
  reported with sign, wins, one-sided Wilcoxon (context only, n=10).
- **(b) Sparse 12 control incl. OOD.** The head-to-head's fixed sparse-12 (incl. the catastrophic
  **BF2_1comp_slope**, current-ckpt cond 0.320 ≈ zero-predictor floor), proposal-level, ft vs
  current. Question: **does fine-tuning WORSEN OOD?** Report per-episode Δ; flag BF2_1comp_slope
  explicitly. A worsening here is a cost to weigh against any crowded gain.
- **(c) Training-curve sanity.** anchor + field loss vs step for both configs.

## 6. Compute (frozen)
- GPU is ours; one other agent runs CPU real-data work. `torch.set_num_threads(4)`; CPU tracker
  workers `nice -10`, threads=1, 4 workers. Cap ~1.5h train + ~40min eval.
- Order: freeze prereg → train A,B → val-select → PRIMARY (ft on 40+12; verify current on 3) →
  integrated spot check (precompute 10×5 + tracker) → analyze. If eval projects over budget: drop
  the integrated spot check first, then the sparse control; NEVER cut the crowded primary.

## 7. Frozen episode lists
- **PRIMARY 40 crowded:** `episode_BF{4,5}_{15,30}comp_seed{0..4}_{uniform,slope}.h5`.
- **Integrated spot check — 10 crowded (frozen):** the seed-0 and seed-1 slice of the 2×2 grid =
  `BF4_15comp_seed{0,1}_{uniform,slope}`, `BF4_30comp_seed{0,1}_{uniform,slope}`,
  `BF5_15comp_seed0_uniform`, `BF5_30comp_seed0_slope`.
- **Sparse 12 control:** identical to the head-to-head sparse list (`run_cond_crowded.SPARSE`),
  including `BF2_1comp_seed0_slope` (OOD).

## 8. What a PASS / FAIL licenses (frozen scope)
Same scope caveats as Stage-1 §7 / head-to-head §7: an **in-distribution synthetic** result only.
Does NOT license real-data claims (synthetic-clean transfer caveat stands), does NOT reopen the
refuted γ/bank-MMSE axis, does NOT authorize any default flip, paper edit, checkpoint overwrite, or
`data/` write. New ckpts saved as `results/direction3/prior_net_e2eft_{A,B}.pt` (NEVER overwrite
existing .pt). No commits.
