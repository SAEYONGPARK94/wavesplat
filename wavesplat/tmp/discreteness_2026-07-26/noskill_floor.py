"""Zero-predictor (no-skill) nRMSE at the EXACT 3 nowcast frames both arms are scored on.

Descriptive diagnostic added after the frozen verdict was computed. It changes no threshold and
no decision variable; it answers "did the learned advantage collapse because the learned arm lost
its edge, or because BOTH arms fell to no-skill?" -- which the paired ratio R cannot distinguish.

floor_i = mean over the 3 nowcast frames of sqrt(mean(gt_frame^2)) / Hs_true
skill   = 1 - nRMSE / floor
"""
import os, sys, json
import warnings; warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, 'scripts')
import numpy as np
from wavesplat.data_loader import EpisodeDataLoader
from d23_main_results import K_TP

D = 'tmp/discreteness_2026-07-26'
CELLS = ['a', 'b', 'c']
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (10, 11, 12, 13, 14) for r in ('uniform', 'slope')]

out = {}
for c in CELLS:
    for ep in EPS:
        L = EpisodeDataLoader(f'{D}/data/cell_{c}/{ep}')
        nb = int(round(K_TP * L.Tp_true / 0.1))
        sel = np.linspace(nb + 3, len(L) - 2, 3).astype(int)
        v = [float(np.sqrt(np.mean(L[int(i)].gt_field.numpy() ** 2)) / max(L.Hs_true, 1e-6))
             for i in sel]
        out[f'{c}/{ep}'] = float(np.mean(v))

res = json.load(open(f'{D}/results.json'))
print(f"{'cell':5s}{'floor(zero-pred)':>18s}{'cond':>9s}{'ws_best':>9s}"
      f"{'skill_cond':>12s}{'skill_ws':>10s}{'skill gap':>11s}")
summ = {}
for c in CELLS:
    rows = [r for r in res['rows'][c]]
    fl = np.array([out[f"{c}/{r['ep']}"] for r in rows])
    cd = np.array([r['cond'] for r in rows]); wb = np.array([r['ws_best'] for r in rows])
    sc, sw = 1 - cd / fl, 1 - wb / fl
    summ[c] = dict(floor=float(fl.mean()), cond=float(cd.mean()), ws=float(wb.mean()),
                   skill_cond=float(sc.mean()), skill_ws=float(sw.mean()),
                   skill_cond_med=float(np.median(sc)), skill_ws_med=float(np.median(sw)),
                   n_cond_above_floor=int((cd >= fl).sum()),
                   n_ws_above_floor=int((wb >= fl).sum()))
    print(f"{c:5s}{fl.mean():18.4f}{cd.mean():9.4f}{wb.mean():9.4f}"
          f"{sc.mean():12.3f}{sw.mean():10.3f}{sc.mean()-sw.mean():11.3f}")
print("\nepisodes at-or-worse-than the zero predictor (of 10):")
for c in CELLS:
    print(f"  cell {c}: cond {summ[c]['n_cond_above_floor']}, ws_best {summ[c]['n_ws_above_floor']}")
json.dump(dict(per_episode_floor=out, summary=summ),
          open(f'{D}/noskill_floor.json', 'w'), indent=1)
