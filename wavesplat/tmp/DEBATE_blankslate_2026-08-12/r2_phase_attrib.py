#!/usr/bin/env python3
"""R2/P3 task C: WHICH code change moved WaveSplat between 2026-06-24 and today?

git log --since=2026-06-24 -- wavesplat/wavesplat/ gives 19 commits.  Only one
of them changes the TRACKING math for every init_method:

  a1a1d4e (07-25) adds the flag TRACK_PHASE_PREDICT (default True = shipped
          behaviour: phi += omega*dt every frame, which double-counts time
          against the forward model's own -omega*t term)
  aa8e4c3 (07-26) flips the DEFAULT to False and sets init_method='auto'

results/100ep_ws_letkf.json (06-24) therefore ran with the double advance.
This script re-runs today's HEAD with the flag forced back to True, so the
only difference from today's rerun is that one boolean.

Arms (all on the same 30 episodes BS-1 used):
  v2k_phaseTrue   init_method=v2_konly, TRACK_PHASE_PREDICT=True
  auto_phaseTrue  shipping default,     TRACK_PHASE_PREDICT=True
Compared against the already-measured v2k / auto at HEAD
(tmp/BLANKSLATE_2026-08-12/shipping_default.json) and the published column.
"""
import os, sys, time, json, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import wavesplat.optimizer.sequential_update as SU
SU.TRACK_PHASE_PREDICT = True          # <-- the 2026-06-24 behaviour
from wavesplat.run import run_estimation

OUT = os.path.join(HERE, 'r2_phase_attrib.json')
CELLS = ['BF2_1comp', 'BF3_1comp', 'BF3_2comp', 'BF3_5comp', 'BF4_1comp',
         'BF4_5comp', 'BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']
SUFF = ['seed0_uniform', 'seed0_slope', 'seed1_uniform']
ARMS = {
    'auto_phaseTrue': {},
    'v2k_phaseTrue': {'init_method': 'v2_konly', 'use_dispersion': True,
                      'use_confidence': True, 'adaptive_n': True,
                      'tracking_window': 3, 'seed': 42},
}
HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()


def main():
    assert SU.TRACK_PHASE_PREDICT is True
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
                    out[kk] = {'avg_nRMSE': float(r['avg_nRMSE']),
                               'n_frames': len(r.get('frame_results', [])),
                               'wall_s': time.time() - t0, 'head': HEAD,
                               'TRACK_PHASE_PREDICT': True}
                except Exception as e:
                    out[kk] = {'error': f'{type(e).__name__}: {e}'}
                print(f'{kk:<48s} {out[kk].get("avg_nRMSE")} '
                      f'({time.time()-t0:.0f}s)', flush=True)
                json.dump(out, open(OUT, 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()
