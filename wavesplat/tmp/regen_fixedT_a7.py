"""
regen_fixedT_a7.py — regenerate the fixedT episodes A-7 v2 needs, at the adequate mesh.

Why this exists: A-7 v2 (`scripts/d3_a7v2_refine.py`) is gated on **BF2 1comp fixedT ×5** and
calibrates its kp-invariant detector on **BF3/BF4 1comp-vs-5comp fixedT**. Those files live
only in `data/`; the main regeneration deliberately excluded the fixedT family
(`tmp/REGEN_SPEC_fable_2026-07-23.md`). Retrying A-7 on old fixedT files would defeat the
purpose, and calibrating the detector on old BF3/BF4 while judging on new BF2 would be worse
— so the calibration sets are regenerated too.

`wavesim.pipeline`'s CLI exposes no mesh override, hence this driver (same approach as
`tmp/regen_run_finemesh.py`): set `cfg.mesh` on a `WaveSimConfig` and call `run_episode`
directly. Mesh 200×167 (dx = 1.206 m) matches the main regeneration; same seeds ⇒ same
wave components as the shipped files, so the comparison is paired.

Nothing in `data/` is touched; output goes to `data_finemesh_fixedT_2026-07-23/`. Resumable.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesim.config import WaveSimConfig
from wavesim.pipeline import run_episode

OUT = 'data_finemesh_fixedT_2026-07-23'
os.makedirs(OUT, exist_ok=True)
NX, NY = 200, 167
COMBOS = [('BF2', 1), ('BF3', 1), ('BF3', 5), ('BF4', 1), ('BF4', 5)]  # gate + calibration
SEEDS = [0, 1, 2, 3, 4]

for ss, nc in COMBOS:
    for seed in SEEDS:
        target = os.path.join(OUT, f'episode_{ss}_{nc}comp_seed{seed}_fixedT.h5')
        if os.path.exists(target):
            print(f'skip {os.path.basename(target)}', flush=True)
            continue
        cfg = WaveSimConfig()
        cfg.mesh.nx, cfg.mesh.ny = NX, NY
        t0 = time.time()
        try:
            p = run_episode(sea_state=ss, n_components=nc, n_pts=8000, seed=seed,
                            raydrop_model='fixedT', duration=30.0, dt=0.1, depth=50.0,
                            output_dir=OUT, verbose=False, cfg=cfg, use_gpu_raycast=True)
            if os.path.abspath(p) != os.path.abspath(target):
                os.rename(p, target)
            print(f'done {os.path.basename(target)}  [{time.time()-t0:.0f}s]', flush=True)
        except Exception as e:
            print(f'FAIL {os.path.basename(target)}: {e}', flush=True)

print(f"TOTAL: {len([f for f in os.listdir(OUT) if f.endswith('.h5')])} files", flush=True)
