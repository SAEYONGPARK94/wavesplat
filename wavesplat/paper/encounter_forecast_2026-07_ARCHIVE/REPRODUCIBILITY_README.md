# Reproducibility package — encounter forecasting paper
(2026-07-06. 투고 보충자료용 초안. 저장소 공개 시 이 README가 루트.)

## One-line verification
```bash
python3 scripts/d3_final_verification.py   # 31 headline numbers vs JSONs -> 31/31 PASS
```

## Structure
- `results/direction3/*.json` — all result files (every number in the
  paper lives here; mapping table in `docs/REPRODUCIBILITY_2026-07-06.md`)
- `results/figs/` — paper figures (PNG+PDF) + generators
  `scripts/d3_paper_figs.py`, `d3_paper_figs2.py`
- `results/gifs/` — supplementary animations (list below)
- `scripts/` — one experiment = one script; deterministic seeds
  (window seeds rng(700+wi); i0 grids by linspace; all caps/protocol
  constants in-file)
- `wavesplat/tests/test_stream_seqmmse.py` — unit tests for the
  streaming core (RLS≡batch, rebuild, symmetry, NIS); run
  `python3 -m pytest wavesplat/tests/ wavesim/tests/` (141 tests)

## Key experiment entry points
| claim | script |
|---|---|
| design point & horizon | `d3_falcon_hos_probe.py --tag _design22kt_n12 ...` |
| robustness sweeps | same, `--noise_range/--shadow_h/--heading_deg/--Hs/--Tp` |
| real 5 sites | `d3_falcon_real_operator.py --record {YS17,AA15a,AA15b,AA14,LJ18g}` |
| streaming + latency | `d3_stream_design_pipeline.py {hos,ys17} --m_frame 48` |
| limit ladder | `d3_slope_oracle_gap.py` (+ `--n_comp 2000` — see testbed note) |
| gate + pre-registrations | `d3_val_gate_probe.py`, `d3_gate_confirmatory{,2}.py` |
| E2E control | `d3_e2e_stream_ff.py`, `d3_e2e_ctrl_tune.py` |
| bootstrap CIs | `d3_bootstrap_ci.py` |

## Data
Third-party stereo sea-surface records: Guimarães et al. (2020),
Sci. Data — CC BY 4.0, IFREMER SEXTANT,
doi:10.12770/af599f42-2770-4d6d-8209-13f40e2c292f. Place under
`data/real_bergamasco/` (site folders BS_2011/BS_2013/AA_2014/AA_2015/
YS_2017/LJ_2018).

## Testbed caution (documented audit)
Linear synthetic seas MUST use `n_components >= 2000`; 200-component
seas are artificially hard (discrete-line artifact) and invert oracle
comparisons. See `docs/PHD_NOVELTY_adversarial_review_2026-07-06.md`
v4 section for the audit that found this, and the pre-registration
documents `docs/GATE_CONFIRMATORY*_prereg_2026-07-06.md` for the gate
trail (protocol commits precede execution).

## Supplementary animations (results/gifs/)
1. `falcon_stream_demo_22kt.gif` — real-time pipeline at the design
   point; verified 2 s-lead trace (corr 0.94)
2. `falcon_stream_demo_{BF2,BF5,BF4s3}.gif` — FF-band forecast across
   sea states (0.84/0.97/0.96)
3. `falcon_real_YS17_forecast.gif` — real sea, best window (0.88)
4. `falcon_real_YS17_{goodwin,badwin}_forecast.gif` — the window-skill
   heterogeneity that motivates §5 (+0.61 vs −0.11)
5. `falcon_real_LJ18g_swell_forecast.gif` — storm swell λ≫fan (0.93)
6. `falcon_real_AA15b_site4_forecast.gif` — site #4 (0.74)
7. `falcon_sim_{linear_U3,hos_U3,hos_design22kt}.gif` — simulation legs
