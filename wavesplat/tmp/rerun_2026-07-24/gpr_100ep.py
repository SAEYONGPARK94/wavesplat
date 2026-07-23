"""GPR on all 100 fine-mesh episodes, collecting the metric panel (nRMSE/VR/rho/SS).

GPR is deterministic given the data, so one pass suffices. GT grid is 80x60 in both datasets;
only the observation z-values differ between meshes. Self-contained metric aggregation so the
five-metric and by-BF tables can be refreshed. RMSE_unobs is intentionally not recomputed here
(it needs an obs-mask pass); the panel refresh is scoped to the four primary metrics.
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
warnings.filterwarnings('ignore')

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.evaluation.metrics import compute_all_metrics

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'gpr_100ep_runs.jsonl')
KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score']


def run_one(path):
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    gpr = GPRBaseline(kernel_type='matern', length_scale=20.0,
                      noise_sigma=0.02, window_frames=1)
    fr = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        gpr.reset()
        ms = gpr.update(x, y, z)
        pred = gpr.predict_grid(gx, gy)
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gx, gy, loader.Hs_true)
        m['elapsed_ms'] = ms
        fr.append(m)
    out = {}
    for k in KEYS:
        v = [f[k] for f in fr if k in f and np.isfinite(f.get(k, np.nan))]
        out[k] = float(np.mean(v)) if v else float('nan')
    out['elapsed_ms'] = float(np.mean([f['elapsed_ms'] for f in fr])) if fr else float('nan')
    return out


def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['episode'])
                except (json.JSONDecodeError, KeyError):
                    pass
    eps = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
    with open(OUT, 'a') as fh:
        for i, ep in enumerate(eps):
            if ep in done:
                continue
            try:
                rec = {'episode': ep, **run_one(os.path.join(DATA, ep))}
            except Exception as e:
                rec = {'episode': ep, 'error': repr(e)}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            if (i + 1) % 10 == 0:
                print(f'{i+1}/{len(eps)}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
