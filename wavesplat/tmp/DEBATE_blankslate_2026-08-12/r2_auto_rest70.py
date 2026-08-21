#!/usr/bin/env python3
"""R2/P3: run the SHIPPING DEFAULT arm on the 70 episodes that BS-1 did NOT run.

BS-1 ran only 3 fixed suffixes per cell (seed0_uniform, seed0_slope,
seed1_uniform) = 30/100.  The remaining 70 have a LETKF column already
(tmp/BLANKSLATE_2026-08-12/letkf_framematch.json per_episode['letkf_match'],
frame-matched to t>2Tp) and a zero-field column.  So the extrapolation
question "would the other 70 give the same answer" is answerable for the
cost of the WaveSplat arm alone.
"""
import os, sys, time, json, glob, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavesplat.run import run_estimation
from wavesplat.optimizer.sequential_update import SequentialUpdater

OUT = os.path.join(HERE, 'r2_auto_rest70.json')
FM = json.load(open(os.path.join(ROOT, 'tmp', 'BLANKSLATE_2026-08-12',
                                 'letkf_framematch.json')))['per_episode']
DONE30 = set()
for cell in ['BF2_1comp', 'BF3_1comp', 'BF3_2comp', 'BF3_5comp', 'BF4_1comp',
             'BF4_5comp', 'BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']:
    for suf in ['seed0_uniform', 'seed0_slope', 'seed1_uniform']:
        DONE30.add(f'{cell}_{suf}')

HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()
LAM = SequentialUpdater({}).lambda_l1


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    keys = [k for k in sorted(FM) if k not in DONE30]
    print(f'{len(keys)} episodes to run; HEAD={HEAD} lambda_l1={LAM}', flush=True)
    for i, key in enumerate(keys):
        if key in out:
            continue
        p = os.path.join('data', f'episode_{key}.h5')
        if not os.path.exists(p):
            print(f'MISSING {p}', flush=True); continue
        t0 = time.time()
        try:
            r = run_estimation(p, config=None, verbose=False)
            out[key] = {
                'avg_nrmse': float(r['avg_nRMSE']),
                'n_tracking_frames': len(r.get('frame_results', [])),
                'n_gaussians': int(r.get('n_gaussians', -1)),
                'wall_s': time.time() - t0,
                'git_head': HEAD, 'lambda_l1': LAM,
            }
        except Exception as e:
            out[key] = {'error': f'{type(e).__name__}: {e}'}
        print(f'[{i+1}/{len(keys)}] {key} {out[key].get("avg_nrmse")} '
              f'({time.time()-t0:.1f}s)', flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('done', len(out), flush=True)


if __name__ == '__main__':
    main()
