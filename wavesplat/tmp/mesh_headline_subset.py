"""
mesh_headline_subset.py — does the paper's headline margin survive mesh refinement?
Scaled-up replication of Reviewer C's n=1 probe (2026-07-23).

Background: `wavesim.mesh_adequacy.check()` fails 0/102 shipped BF2-BF5 episodes
(50 INADEQUATE / 52 DEGRADED; BF2_1comp sits at 1.18 pts/lambda at the PEAK, below the
Nyquist floor of 2.0). Reviewer C regenerated ONE episode (BF2_1comp_seed0, the paper's
quoted worst case) at a fine mesh and found WaveSplat 0.315 -> 0.225 (-28.5%) and
ETKF 0.483 -> 0.359 (-25.7%): absolute numbers inflated, ordinal ranking intact.

n=1 cannot tell us whether the 100-episode headline (WS 0.223 vs ETKF 0.319, "30.1%")
moves by a similar amount. This script runs the SAME paired coarse/fine test over the
paper's own 12-episode main comparison set (scripts/run_p1_comparison.py:EPISODES).

Pre-registered reads (stated before the numbers):
  A. Both arms improve by a similar factor and the WS-ETKF margin holds within a few
     points -> the paper's CONCLUSION stands; only the absolute numbers must be restated
     after a full dataset regeneration.
  B. The margin shrinks materially (say, below ~20% from 30.1%) -> the reported advantage
     is partly a mesh-artifact-handling advantage; that is a substantive correction, not
     just a restatement.
  C. The margin grows -> as C's n=1 suggested (34.8% -> 37.3%); conclusion stands and the
     coarse mesh was costing WaveSplat more than ETKF.
No result here licenses new headline numbers: those require regenerating all 100 episodes.

Nothing is overwritten: episodes are regenerated into tmp/mesh_subset_ep/.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from wavesim.config import WaveSimConfig
from wavesim.pipeline import run_episode
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics

OUT_DIR = 'tmp/mesh_subset_ep'
os.makedirs(OUT_DIR, exist_ok=True)
RESULT = 'tmp/mesh_headline_subset_results.json'

# the paper's own main comparison set (scripts/run_p1_comparison.py:EPISODES)
TARGETS = [
    ('BF2', 1, 0, 'uniform'), ('BF3', 1, 0, 'uniform'),
    ('BF3', 2, 0, 'uniform'), ('BF3', 2, 0, 'slope'),
    ('BF3', 5, 0, 'uniform'), ('BF3', 5, 0, 'slope'),
    ('BF4', 1, 0, 'uniform'), ('BF4', 5, 0, 'uniform'),
    ('BF4', 15, 0, 'uniform'), ('BF4', 30, 0, 'uniform'),
    ('BF5', 15, 0, 'uniform'), ('BF5', 30, 0, 'uniform'),
]
FINE = (180, 150)     # same refinement Reviewer C / the E-3 audit used


def gen(sea_state, ncomp, seed, raydrop, tag, nx, ny):
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = nx, ny
    p = run_episode(sea_state=sea_state, n_components=ncomp, n_pts=50, seed=seed,
                    raydrop_model=raydrop, duration=30.0, dt=0.1, depth=50.0,
                    output_dir=OUT_DIR, verbose=False, cfg=cfg, use_gpu_raycast=True)
    q = p.replace('.h5', f'_{tag}.h5')
    os.rename(p, q)
    return q


def run_ws(h5):
    return float(run_estimation(h5, config={
        'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
        'adaptive_n': True, 'tracking_window': 3, 'seed': 42})['avg_nRMSE'])


def run_etkf(h5):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40); egy = np.linspace(gy[0], gy[-1], 30)
    etkf = ETKFGrid(egx, egy, depth=loader.depth, n_ensemble=50)
    out = []
    Xg, Yg = np.meshgrid(gx, gy)
    for fr in loader:
        x = fr.x_obs.cpu().numpy(); y = fr.y_obs.cpu().numpy(); z = fr.z_obs.cpu().numpy()
        etkf.update(x, y, z)
        interp = RegularGridInterpolator((egy, egx), etkf.predict_grid(), method='linear',
                                         bounds_error=False, fill_value=0.0)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(len(gy), len(gx))
        out.append(compute_all_metrics(pred, fr.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                       obs_x=x if len(x) else None, obs_y=y if len(y) else None))
    return float(np.mean([r['nRMSE'] for r in out])) if out else float('nan')


results = {}
if os.path.exists(RESULT):
    results = json.load(open(RESULT))

for sea_state, ncomp, seed, raydrop in TARGETS:
    tag = f'{sea_state}_{ncomp}c_s{seed}_{raydrop}'
    if tag in results:
        print(f'skip {tag} (done)', flush=True)
        continue
    t0 = time.time()
    pc = gen(sea_state, ncomp, seed, raydrop, tag + '_coarse', 40, 30)
    pf = gen(sea_state, ncomp, seed, raydrop, tag + '_fine', *FINE)
    wc, wf = run_ws(pc), run_ws(pf)
    ec, ef = run_etkf(pc), run_etkf(pf)
    r = dict(ws_coarse=wc, ws_fine=wf, etkf_coarse=ec, etkf_fine=ef,
             margin_coarse=100 * (ec - wc) / ec, margin_fine=100 * (ef - wf) / ef)
    results[tag] = r
    json.dump(results, open(RESULT, 'w'), indent=2)
    print(f'{tag:<26} WS {wc:.4f}->{wf:.4f}   ETKF {ec:.4f}->{ef:.4f}   '
          f'margin {r["margin_coarse"]:.1f}%->{r["margin_fine"]:.1f}%   [{time.time()-t0:.0f}s]',
          flush=True)

if results:
    wc = np.array([r['ws_coarse'] for r in results.values()])
    wf = np.array([r['ws_fine'] for r in results.values()])
    ec = np.array([r['etkf_coarse'] for r in results.values()])
    ef = np.array([r['etkf_fine'] for r in results.values()])
    print(f'\n=== SUBSET MEAN over n={len(wc)} ===')
    print(f'  WaveSplat  coarse {wc.mean():.4f}  fine {wf.mean():.4f}  ({100*(wf.mean()-wc.mean())/wc.mean():+.1f}%)')
    print(f'  ETKF       coarse {ec.mean():.4f}  fine {ef.mean():.4f}  ({100*(ef.mean()-ec.mean())/ec.mean():+.1f}%)')
    print(f'  margin     coarse {100*(ec.mean()-wc.mean())/ec.mean():.1f}%  fine {100*(ef.mean()-wf.mean())/ef.mean():.1f}%')
    d = (ef - wf) / ef - (ec - wc) / ec
    print(f'  per-episode margin change: mean {100*d.mean():+.1f} pts, SE {100*d.std(ddof=1)/np.sqrt(len(d)):.1f}, '
          f'{int((d>0).sum())}/{len(d)} episodes improved')
