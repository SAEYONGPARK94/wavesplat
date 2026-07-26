"""Feasibility probe: can each held-out record be built by the UNMODIFIED scene_real?
NO scoring, NO estimator. Reports only scene shape/inputs."""
import os, sys, json, time, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/heldout_real_2026-07-26')
import heldout_cfg
names = heldout_cfg.install()
import d3_compare_gifs as cg

rows = []
for rec in names:
    for win in (0, 5):
        t0 = time.time()
        try:
            sc = cg.scene_real(rec, win=win)
            gt = sc['maps_gt']
            nfin = int(np.isfinite(gt[0]).sum())
            rows.append(dict(record=rec, win=win, ok=True, n_obs=int(len(sc['z'])),
                             npx=int(sc['npx']), nqx=int(len(sc['gxq'])),
                             nqy=int(len(sc['gyq'])), n_finite_gt0=nfin,
                             Hs=float(sc['Hs']), kp=float(sc['kp']),
                             lam_p=float(2*np.pi/sc['kp']), sec=time.time()-t0))
            print(f"  {rec} w{win}: OK n_obs={len(sc['z'])} grid={len(sc['gyq'])}x"
                  f"{len(sc['gxq'])} finite_gt0={nfin} Hs={sc['Hs']:.3f} "
                  f"lam_p={2*np.pi/sc['kp']:.1f}m ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            rows.append(dict(record=rec, win=win, ok=False, err=repr(e)))
            print(f"  {rec} w{win}: FAIL {e!r}", flush=True)
            traceback.print_exc()
json.dump(rows, open('tmp/heldout_real_2026-07-26/probe_load.json', 'w'), indent=1)
