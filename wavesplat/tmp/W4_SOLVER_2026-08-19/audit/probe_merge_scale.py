"""AUDIT_PRO: what does `learned_merge_frac` actually DO at each gridpoint?
ZERO estimator runs -- replays the real deployment selection path
(learned_init.pool_and_select -> learned_init.merge_atoms) on cached raw proposal atoms.

  cd /home/vexit/wavesplat/wavesplat && python3 tmp/W4_SOLVER_2026-08-19/audit/probe_merge_scale.py
"""
import json, re, sys, os
import numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat')
from wavesplat.optimizer.learned_init import pool_and_select, merge_atoms

CACHES = ('tmp/wp_integration_2026-07-25/atoms_cache.jsonl',
          'tmp/unified_2026-07-26/atoms_cache_extra.jsonl')
C = {}
for src in CACHES:
    for l in open(src):
        d = json.loads(l)
        if d.get('kx') is not None and d.get('err') is None:
            C[(d['episode'], d['diff_seed'])] = d
crowded = sorted(k for k in C if int(re.search(r'_(\d+)comp', k[0]).group(1)) >= 15)
out = {}
for frac in (0.0, 0.15, 0.25, 0.375, 0.5):
    nin, nout, kps = [], [], []
    for key in crowded:
        d = C[key]
        kx, ky, en = pool_and_select(np.array(d['kx']), np.array(d['ky']),
                                     np.array(d['energy']),
                                     energy_thresh=0.03, dedup_dk=0.03, max_n=30)
        kps.append(float(np.hypot(kx[int(np.argmax(en))], ky[int(np.argmax(en))])))
        nin.append(len(kx))
        nout.append(len(merge_atoms(kx, ky, en, frac)[0]) if frac > 0 else len(kx))
    nin, nout = np.array(nin), np.array(nout); rem = nin - nout
    out[frac] = dict(cells=len(nin), kp_med=float(np.median(kps)),
                     r_med=float(frac * np.median(kps)),
                     n_in_med=float(np.median(nin)), n_out_med=float(np.median(nout)),
                     removed_med=float(np.median(rem)),
                     cells_with_removal=int((rem > 0).sum()), max_removed=int(rem.max()))
    print(frac, out[frac])
json.dump(out, open('tmp/W4_SOLVER_2026-08-19/audit/probe_merge_scale.json', 'w'), indent=1)
