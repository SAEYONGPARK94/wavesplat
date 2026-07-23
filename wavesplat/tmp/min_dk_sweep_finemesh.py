"""
min_dk_sweep.py — is the crowded-regime resolution floor INFORMATIONAL or a HARDCODED CONSTANT?

fable's scale audit (tmp/SCALE_AUDIT_fable_2026-07-23.md) found `min_dk = 0.15` rad/m --
the suppression radius of the greedy beamform peak search -- hardcoded identically at three
sites (initializer.py, coarse_to_fine.py, sequential_update.py). Census over all 195 shipped
episodes: 126/195 (64.6%) contain a true component pair closer than 0.15, and that is
**100% of every 15/20/30-component episode**. End-to-end on BF4_15comp_seed0, four true
components spanning kx 0.028-0.094 collapse into a single recovered Gaussian.

Why this matters beyond numerics: the project's empirical case for an information wall
("no method recovers close / short-lambda components") was measured with THIS estimator. If
0.15 is what blocks them rather than the information content, that empirical leg is
confounded. The CRB theory is estimator-independent and is not touched either way.

Pre-registered decision rule (before the numbers), on median-of-5-seeds to respect the
run-to-run variance documented in tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md:
  - Lowering min_dk materially LOWERS nRMSE on crowded episodes -> the floor is ALGORITHMIC.
    The empirical information-wall evidence is confounded and must be re-derived.
  - No material change -> the floor is NOT this constant; the empirical evidence survives
    this particular challenge (it does not prove the wall, it just isn't this artifact).
  - Lowering min_dk makes it WORSE -> the suppression radius is doing useful regularisation
    (duplicate-peak protection); 0.15 is defensible, but should be justified/normalised
    rather than hardcoded at three sites.
  Control: BF3 2comp has NO true pair closer than 0.15, so it must be ~unchanged. If it
  moves as much as the crowded episodes, the sweep is measuring something else and the
  reading is void.
"""
# RE-RUN ON THE REGENERATED (Nyquist-adequate) DATASET.
# The original sweep (tmp/MIN_DK_RESULT_2026-07-23.md) ran on shipped episodes where the
# census (tmp/REVALIDATION_CENSUS_2026-07-23.md) measured the affected components as
# transmitted at only 7-42% amplitude. The DIRECTION of that result is expected to survive;
# the QUANTITATIVE bound ("~5% of crowded-regime error is algorithmic") may not, and it was
# already borderline (t = -2.06). Same pre-registered rule as the original.
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import wavesplat.optimizer.initializer as INIT
from wavesplat.run import run_estimation

RESULT = 'tmp/min_dk_sweep_finemesh_results.json'
SEEDS = [42, 1, 2, 3, 4]
MIN_DKS = [0.15, 0.05, 0.02]
EPISODES = [
    ('data_finemesh_2026-07-23/episode_BF4_15comp_seed0_uniform.h5', 'crowded: 100% of 15comp have pairs < 0.15'),
    ('data_finemesh_2026-07-23/episode_BF3_5comp_seed0_uniform.h5', 'medium'),
    ('data_finemesh_2026-07-23/episode_BF3_2comp_seed0_uniform.h5', 'CONTROL: no true pair < 0.15, must not move'),
]

res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

for ep, note in EPISODES:
    key = os.path.basename(ep)
    res.setdefault(key, {})
    for md in MIN_DKS:
        mk = str(md)
        res[key].setdefault(mk, {})
        INIT.MIN_DK = md
        for s in SEEDS:
            if str(s) in res[key][mk]:
                continue
            t0 = time.time()
            out = run_estimation(ep, config={
                'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
                'adaptive_n': True, 'tracking_window': 3, 'seed': s})
            res[key][mk][str(s)] = float(out['avg_nRMSE'])
            json.dump(res, open(RESULT, 'w'), indent=2)
            print(f'  {key:<38} min_dk={md:<5} seed={s:<3} nRMSE={out["avg_nRMSE"]:.4f}'
                  f'  [{time.time()-t0:.0f}s]', flush=True)
        a = np.array([res[key][mk][str(s)] for s in SEEDS])
        print(f'{key:<38} min_dk={md:<5} median {np.median(a):.4f}  mean {a.mean():.4f}  '
              f'sd {a.std(ddof=1):.4f}  min {a.min():.4f}  max {a.max():.4f}', flush=True)
    print(f'  ({note})\n', flush=True)
INIT.MIN_DK = 0.15
