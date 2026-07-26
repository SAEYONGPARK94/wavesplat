# HANDOFF — WavePrior program closure, 2026-07-26 (for the next session / Opus)

> 목적: 다음 세션(Opus 포함)이 **이 문서 하나로** 이틀간(07-25/26)의 결과를 오해 없이
> 이어받도록 함. 모든 수치는 아래 원본 파일에서 재도출 가능. 근거 없는 문장 없음.
> Branch: `learn-boundwave-retrain`. All paths relative to `wavesplat/` package root.

## 0. TL;DR (the PhD narrative as measured)

**"물리·학습의 레짐 경계를 사전등록으로 긋고, GT-free 자동 스위치 단일 추정기로 봉인."**

One estimator, one config (`init_method='auto'`, guard on) now measures
**0.0875 mean nRMSE on the 100-episode finemesh benchmark**, beating both pure arms
(analytic 0.1145, learned 0.1011), with oracle switching at 0.0818. On real stereo data
the learned arm's role is **failure-mode insurance, not accuracy**: it reduces error on
all 3 analytic collapse windows (2.31→1.07, 1.12→0.79, 2.93→0.96) with 0 own collapses —
but under the FROZEN criterion (fixed = nRMSE<1.0) that is **2/3**, and the deployed
detector∘fallback chain brings only **1/3** below the line (it misses the mild window
that had the most useful repair). Deployed fallback: 30-window MEAN 0.881→0.775 —
driven entirely by 2 windows; the MEDIAN is unchanged (0.809→0.809), so do not quote
the mean delta as a general win (own do-not-cite rule §5). False alarms 0/27 (in-sample
τ). Three retraining axes failed honestly and are sealed (fan-geometry, e2e fine-tune,
learned-only sparse parity). This is the boundary-narrative skeleton for the dissertation.
> 🛑 CORRECTION 2026-07-26 (same day, found by P3 in the persona debate): the first
> version of this document said "repairs 3/3" and quoted 0.881→0.775 without the
> median caveat — exactly the criterion-relaxation drift it warns about in §5.

## 1. The estimator ladder (all: 100 ep × 5 seeds, median-of-5 per episode, then mean; `data_finemesh_2026-07-23/`; tracker flag `TRACK_PHASE_PREDICT=False` at run time)

| step | mean nRMSE | source of number |
|---|---|---|
| shipped default (tracker double phase-advance bug ON) | 0.1990 | `tmp/rerun_2026-07-25/batchB_ws.jsonl` |
| tracker fix, pure analytic `v2_konly` | 0.1145 | `tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl` |
| pure learned init (cond-diffusion set-proposal) | 0.1011 | `tmp/wp_integration_2026-07-25/arm_runs.jsonl` + `tmp/unified_2026-07-26/arm_runs_extra.jsonl` |
| `init_method='auto'` (GT-free router, n_selected≥6) | 0.0979 | `tmp/unified_2026-07-26/switch_results.json` (`by_threshold["6"]`) |
| auto + beamform guard (ratio 0.05) — **deployed config** | 0.0875 | `tmp/unified_guard_2026-07-26/task1_results.json` |
| oracle per-episode switch (upper bound) | 0.0818 | same `switch_results.json` |

Per-regime (pure v2 / pure learned / auto+guard): sparse(1–2comp, 40ep)
0.020 / 0.053 / **0.025**; moderate(5comp, 20ep) 0.027 / 0.037 / **0.028**;
crowded(15–30comp, 40ep) 0.252 / 0.181 / **0.180**.
Pre-guard misroutes were concentrated in `BF2_1comp*_slope` episodes (proposal OOD,
§4); the guard takes the aggregate 0.0979→0.0875 with **zero crowded flips**.
> 🛑 E3 (found in the 07-26 persona debate, judge-confirmed): do NOT say "closes X% of
> the oracle gap" — the oracle (per-episode min-of-medians) and the deployed number
> (per-lane-then-median) are aggregation-inconsistent (crowded slice 0.17982 sits BELOW
> its "oracle" 0.18053). Unquotable until one script computes both the same way.
> What IS measured: the residual gap is concentrated — 46% in one episode
> (BF2_1comp_seed3_slope, bf_peak_ratio 0.19 → guard cannot see it), 91% in eight.
> 🛑 E4: the shipped router default is `auto_switch_thresh = 8`
> (`sequential_update.py`); the 0.0979 rung above was assembled at threshold 6
> (flat grid: 8 gives 0.09786, Δ4e-5). Freeze **8** in writing before any held-out run.

## 2. Code state (what is default vs what is opt-in) — READ THIS BEFORE CITING

- **UPDATE 2026-07-26 (commit `aa8e4c3`, user decision via Discord): both defaults are
  now FLIPPED** — `TRACK_PHASE_PREDICT = False` and `DEFAULT_CONFIG['init_method'] =
  'auto'` (with graceful fallback to v2_konly if the diffusion checkpoint is missing).
  158 tests pass; live sanity: sparse BF2_1comp → guard veto → v2 (0.026), crowded
  BF4_15comp → learned (0.1945 single diff-seed draw). **The OE paper still carries the
  bug-era numbers (0.199/0.223 and the tracking-limitation narrative L2183–2213) —
  paper propagation is the remaining open item, and citing the paper's numbers is a
  live integrity exposure until resolved.**
- `auto` path (`sequential_update.py` ~L536-554): router = learned proposal
  `n_selected >= auto_switch_thresh` (shipped default **8**; see 🛑E4 in §1) → learned,
  else `v2_konly`; then **guard** (`auto_guard=True`,
  `auto_guard_ratio=0.05` by default *inside the auto path*): if routed learned but
  beamform dominant-peak scalar (`initializer.py:292`) `< 0.05` → veto to v2.
  Live single-entry-point path validated bit-identical to the pre-registered lanes:
  `tmp/unified_2026-07-26/demo_auto.json` (max reproduction diff 0.0).
- Learned proposal: `wavesplat/optimizer/learned_init.py` (cond-diffusion + polish,
  needs GPU for live proposal; cached atoms at
  `tmp/wp_integration_2026-07-25/atoms_cache.jsonl` + `tmp/unified_2026-07-26/atoms_cache_extra.jsonl`, key `diff_seed`).
- Real-data deployment arm (separate from synthetic router): hand estimator everywhere
  + **collapse fallback**: fire iff `amp_ratio_grid > 1.5` (frozen prereg) → use
  learned for that window. `tmp/unified_2026-07-26/analyze_fallback.py`.

## 3. Evidence map (claim → prereg → result → commit)

| claim | prereg | result file | commit |
|---|---|---|---|
| tracker double phase-advance bug, −42.5%, 100/100, p=3.9e-18 | `tmp/TRACKER_AMPDECAY_PREREG_2026-07-25.md` | `tmp/rerun_2026-07-25/PHASEFIX_100EP_2026-07-25.md` | `2975a0d` (found in `a1a1d4e`) |
| learned set-proposal beats analytic init on crowded, init-level −45% (39/40, p=1.8e-12) | `tmp/wp_crowded_2026-07-25/PREREG.md` | `.../results.json` | `da1a6cb` |
| integrated: crowded tracked 0.252→0.181 (36/40, p=5e-11); mechanism = learned kx/ky placement (ablation) | `tmp/wp_integration_2026-07-25/PREREG.md` | `.../results.json` | `d4ec99c` |
| unified auto: 0.0979 beats both pures; threshold-insensitive | `tmp/unified_2026-07-26/PREREG.md` | `.../switch_results.json`, narrative `.../UNIFIED_SYSTEM_2026-07-26.md` | `40b635d` |
| guard PASS: 0.0979→0.0875, sparse restored to 0.0249 (within 0.005 of pure v2), 0 crowded flips | `tmp/unified_guard_2026-07-26/PREREG.md` | `.../task1_results.json` | `049696c` |
| real data: accuracy LOSS (6/27, p=0.005, expected cell); learned reduces error on 3/3 analytic collapse windows but **frozen fixed-criterion (<1.0) = 2/3** (scorer: "AA14 w0 [not fixed]"), own collapses 0 | `tmp/wp_real_2026-07-25/PREREG.md` | `.../wp_real.json`, `.../analysis.txt`, `.../WP_REAL_2026-07-25.md` | `a611be0` |
| deployed real fallback: detector catches the 2 severe collapses (misses the mild 1.12 one — which had the most useful repair), **end-to-end only 1/3 ends below the collapse line**; 30-window mean 0.881→0.775 but median unchanged 0.809 (2 windows drive the mean); false alarms 0/27, in-sample τ | `tmp/unified_2026-07-26/PREREG.md` (Part-C frozen) | `.../fallback_results.json` | `40b635d` |

## 4. Sealed honest negatives (do not reopen without new evidence)

1. **e2e fine-tune = null** (medΔ −0.0007, p=0.30) — third retraining failure ⟹
   **modular = product**. Leak localized: synthetic +9% evaporates at the
   observation-distribution gap (random-footprint vs LiDAR-sweep). If ever revisited,
   observation-distribution matching is the prerequisite. `tmp/wp_e2e_2026-07-25/`, commit `6dd65be`.
2. **Learned-only sparse parity = sealed FAIL** (two pre-registered rounds). Root cause
   measured: **proposal-generation OOD** on `BF2_1comp` — the net does not emit atoms
   near the true swell at all, so no post-processing can fix it; would need retraining.
   Peak-anchor post-processing is **crowded-harmful, do not deploy**.
   `tmp/learned_only_2026-07-26/`, commit `7c6e68d`.
3. **Fan-geometry retrain = axis closed** (PRIMARY FAIL; architecture[stretch] card
   spent). `tmp/G1_waveprior_power/`, commit `23abddc`.
4. **Doppler real-data attribution**: pooled REFUTE (9/27, p=0.73); only YS17
   conditionally confirmed (6/6, +16%, U converges 6°). Within-record U-stability is
   the GT-free discriminator. Commit `5d58f22`.

## 5. ⛔ Do-NOT-cite list (carried from memory, still binding)

- 실측 **정확도** 학습 우위 — 없음. 실측 기여는 **붕괴 보험뿐**.
- "복잡할수록 학습이 이긴다" — **합성 한정** (실측 전이 안 됨, within-record ρ만 +0.035 n.s.).
- mean 기반 실측 승리 주장 (붕괴창 아티팩트).
- "학습가치 = prior 미스매치" 기제, "비선형성이 클수록 학습 승리" — 반박됨.
- 에피소드-단위 델타를 다중시드 없이 인용 금지 (CV 53%; median-of-k 규약:
  `docs/REPORTING_CONVENTION_median_of_k_2026-07-23.md`).
- 벤치 GT는 pure-linear; HOS 팔 미실행 상태의 수치임을 병기할 것.

## 6. Figures & GIFs made 2026-07-26 (all in `results/gifs/`)

- `program_journey.png` — the 0.199→0.0875 ladder (fig 1 above).
- `program_phasefix.png` — tracker fix by complexity (sparse near-exact, wall survives).
- `program_regime_cross.png` — per-regime pure v2 / pure learned / auto+guard.
- `program_real_fallback.png` — real 30-window aggregate + 3 collapse windows.
- `unified_auto_crowded_BF4_15comp.gif` — GT | pure v2 | auto(→learned), live pipeline.
- `unified_auto_sparse_BF2_1comp.gif` — GT | pure v2 | auto, sparse routing behavior.
- Generator scripts: `tmp/figs_program_2026-07-26/make_figs.py`, `make_gif.py`
  (figures read only session JSONs; GIFs re-run the live pipeline, cached atoms).

## 7. Open decisions (user, not agent)

1. ~~Default flip~~ **DONE 2026-07-26** (`aa8e4c3`): `TRACK_PHASE_PREDICT=False` +
   `init_method='auto'` are now the shipped defaults.
2. **OE-paper propagation** (0.199→0.115, vs ETKF 40%→65%, rewrite of the
   tracking-limitation narrative) — still open, now the top integrity item.
3. 집필 착수(경계 서사 골격) + 지도교수 difference-first 자료.
4. Persona-debate follow-ups (see `docs/PERSONA_DEBATE_novelty_phd_value_2026-07-26.md`
   once written): blind auto-on-real run, held-out episodes/records, FNO baseline.

## 8. Reproduction quick-start

```bash
cd wavesplat/
# unified estimator, live (GPU for proposal), single entry point:
python3 -m wavesplat.run data_finemesh_2026-07-23/episode_BF4_15comp_seed0_uniform.h5 \
    --init_method auto --n_seeds 5
# validation that live == prereg lanes: tmp/unified_2026-07-26/demo_auto_live.py
# regenerate summary figures:      python3 tmp/figs_program_2026-07-26/make_figs.py
# regenerate comparison GIFs:      python3 tmp/figs_program_2026-07-26/make_gif.py
```

Discipline: 7규율(`memory/feedback_direction_flip_discipline.md`),
이중검증 기본(`memory/feedback_dual_verification_default.md`), 사전등록 없는 방향 전환 금지.
