# HANDOFF — WavePrior program closure, 2026-07-26 (for the next session / Opus)

> 목적: 다음 세션(Opus 포함)이 **이 문서 하나로** 이틀간(07-25/26)의 결과를 오해 없이
> 이어받도록 함. 모든 수치는 아래 원본 파일에서 재도출 가능. 근거 없는 문장 없음.
> Branch: `learn-boundwave-retrain`. All paths relative to `wavesplat/` package root.

## 0. TL;DR (the PhD narrative as measured)

**"물리·학습의 레짐 경계를 사전등록으로 긋고, GT-free 자동 스위치 단일 추정기로 봉인."**

One estimator, one config (`init_method='auto'`, guard on) now measures
**0.0875 mean nRMSE on the 100-episode finemesh benchmark**, beating both pure arms
(analytic 0.1145, learned 0.1011), with oracle switching at 0.0818. On real stereo data
the learned arm's role is **collapse insurance, not accuracy** (repairs analytic collapse
windows; deployed amp-guard fallback 0.881→0.775, false alarms 0/27). Three retraining
axes failed honestly and are sealed (fan-geometry, e2e fine-tune, learned-only sparse
parity). This is the boundary-narrative skeleton for the dissertation.

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
§4); the guard closes ~2/3 of the oracle gap with **zero crowded flips**.

## 2. Code state (what is default vs what is opt-in) — READ THIS BEFORE CITING

- `wavesplat/optimizer/sequential_update.py:65` — **`TRACK_PHASE_PREDICT = True` is
  STILL the shipped default** (the bug). Every 07-25/26 arm sets it to `False`
  explicitly. **Flipping the default + propagating 0.199→0.1145 into the OE paper is an
  OPEN USER DECISION** (rewrites tracking-limitation narrative, VR/SS attributions).
- `wavesplat/run.py` `DEFAULT_CONFIG['init_method'] = 'v2_konly'` — **`auto` is NOT the
  default**; it is selected via config/CLI (`--init_method auto`).
- `auto` path (`sequential_update.py` ~L536-554): router = learned proposal
  `n_selected >= 6` → learned, else `v2_konly`; then **guard** (`auto_guard=True`,
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
| real data: accuracy LOSS (6/27, p=0.005, expected cell) but learned repairs analytic collapse windows 3/3, own collapses 0 | `tmp/wp_real_2026-07-25/PREREG.md` | `.../wp_real.json`, `.../WP_REAL_2026-07-25.md` | `a611be0` |
| deployed real fallback: 30-window mean 0.881→0.775, detector catches 2/3 collapses (misses only the mild 1.12 one), false alarms 0/27 | `tmp/unified_2026-07-26/PREREG.md` (Part-C frozen) | `.../fallback_results.json` | `40b635d` |

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

1. **Default flip** `TRACK_PHASE_PREDICT` → False + OE-paper propagation (0.199→0.115,
   vs ETKF 40%→65%). Currently NOT flipped.
2. Whether `init_method='auto'` becomes the shipped default (currently `v2_konly`).
3. 집필 착수(경계 서사 골격) + 지도교수 difference-first 자료.
4. [보류] OE 논문에 트래커 수정 반영 여부/방식.

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
