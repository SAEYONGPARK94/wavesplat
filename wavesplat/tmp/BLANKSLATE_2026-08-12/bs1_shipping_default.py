#!/usr/bin/env python3
"""BS1: run the SHIPPING DEFAULT arm and measure its real wall-clock cost.

Two things results/ cannot answer:
  (a) results/100ep_ws_letkf.json used init_method='v2_konly', hardcoded in
      scripts/run_100ep_letkf_and_ws_metrics.py:39 -- NOT the shipping default
      'auto'.  So no file in results/ measures the deployed configuration on
      the 100-episode set.
  (b) results/compute_cost_comparison.json's WaveSplat row is not the shipping
      estimator: init_s comes from scripts/gp_disp_pilot.estimate_components_
      wavesplat_opt and inference_ms is the literal constant 13.4
      (scripts/eval_compute_cost.py:90).

This script calls wavesplat.run.run_estimation with (i) the shipping DEFAULT
config and (ii) the 100ep 'v2_konly' config, on the same episodes, and records
wall-clock init time and per-frame tracking time from the same process.
"""
import os, sys, time, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavesplat.run import run_estimation

OUT = os.path.join(HERE, 'shipping_default.json')

# two per (BF, ncomp) cell -> both raydrop masks of seed 0, plus seed 1 uniform
CELLS = ['BF2_1comp', 'BF3_1comp', 'BF3_2comp', 'BF3_5comp', 'BF4_1comp',
         'BF4_5comp', 'BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']
SUFF = ['seed0_uniform', 'seed0_slope', 'seed1_uniform']

ARMS = {
    'shipping_default': {},                       # everything from DEFAULT_CONFIG
    'v2_konly_100ep': {'init_method': 'v2_konly',  # what 100ep_ws_letkf.json ran
                       'use_dispersion': True, 'use_confidence': True,
                       'adaptive_n': True, 'tracking_window': 3, 'seed': 42},
}


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for cell in CELLS:
        for suf in SUFF:
            key = f'{cell}_{suf}'
            p = os.path.join('data', f'episode_{key}.h5')
            if not os.path.exists(p):
                continue
            for arm, cfg in ARMS.items():
                kk = f'{key}|{arm}'
                if kk in out:
                    continue
                t0 = time.time()
                try:
                    r = run_estimation(p, config=dict(cfg) if cfg else None,
                                       verbose=False)
                except Exception as e:
                    out[kk] = {'error': f'{type(e).__name__}: {e}'}
                    print(kk, 'ERROR', e, flush=True)
                    json.dump(out, open(OUT, 'w'))
                    continue
                wall = time.time() - t0
                fr = r.get('frame_results', [])
                ms = [f['elapsed_ms'] for f in fr if 'elapsed_ms' in f]
                out[kk] = dict(
                    avg_nRMSE=float(r.get('avg_nRMSE', np.nan)),
                    init_nrmse=float(r.get('init_nrmse', np.nan)),
                    n_gaussians=r.get('n_gaussians', -1),
                    n_tracking_frames=len(fr),
                    wall_total_s=wall,
                    track_ms_median=float(np.median(ms)) if ms else float('nan'),
                    track_total_s=float(np.sum(ms) / 1000.0) if ms else float('nan'),
                    init_wall_s=wall - (float(np.sum(ms) / 1000.0) if ms else 0.0),
                )
                print(f'{kk:<48s} nRMSE={out[kk]["avg_nRMSE"]:.4f} '
                      f'init={out[kk]["init_wall_s"]:.1f}s '
                      f'track={out[kk]["track_ms_median"]:.1f}ms/fr '
                      f'wall={wall:.1f}s', flush=True)
                json.dump(out, open(OUT, 'w'))
    json.dump(out, open(OUT, 'w'))


if __name__ == '__main__':
    main()
