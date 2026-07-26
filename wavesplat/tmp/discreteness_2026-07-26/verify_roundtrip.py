"""G-roundtrip (PREREG §2): open one generated episode per cell with the estimator-side loader
and confirm it is schema-compatible with data_finemesh_2026-07-23 (no library change, no cap on
component count reaching the estimator)."""
import os, sys, json
import warnings; warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, h5py
from wavesplat.data_loader import EpisodeDataLoader

REF = 'data_finemesh_2026-07-23/episode_BF4_15comp_seed0_uniform.h5'
D = 'tmp/discreteness_2026-07-26/data'
EP = 'episode_BF4_15comp_seed10_uniform.h5'


def probe(path):
    L = EpisodeDataLoader(path)
    f0 = L[0]
    with h5py.File(path, 'r') as h:
        at = dict(h['config'].attrs)
        gt = h['frames/gt_wave_field'][:]
        natom = h['config/wave_components'].shape[0]
    return dict(path=path, n_frames=len(L), Hs_true=L.Hs_true, Tp_true=L.Tp_true,
                depth=L.depth, sea_state=L.sea_state,
                gt_shape=list(gt.shape), n_atoms=int(natom),
                Hs_realized=float(4 * np.std(gt)),
                init_window_frames=len(L.get_init_window(2.0)),
                n_obs_frame0=int(len(f0.x_obs)),
                mesh_verdict=str(at.get('mesh_verdict')),
                var_under=float(at.get('var_under_resolved', np.nan)),
                pts_per_lambda=float(at.get('pts_per_lambda', np.nan)),
                wave_components_loaded=None if L.wave_components is None
                else list(np.asarray(L.wave_components).shape))


rows = [probe(REF)] + [probe(f'{D}/cell_{c}/{EP}') for c in ('a', 'b', 'c')
                       if os.path.exists(f'{D}/cell_{c}/{EP}')]
for r in rows:
    print(json.dumps(r))
ref = rows[0]
ok = True
for r in rows[1:]:
    for k in ('n_frames', 'Hs_true', 'Tp_true', 'depth', 'sea_state', 'gt_shape',
              'init_window_frames'):
        if r[k] != ref[k]:
            print(f"  MISMATCH {r['path']} {k}: {r[k]} vs ref {ref[k]}"); ok = False
    if abs(r['Hs_realized'] - 1.2) / 1.2 > 0.10:
        print(f"  Hs_realized OUT OF BAND {r['path']}: {r['Hs_realized']:.3f}"); ok = False
    if r['mesh_verdict'] != 'ADEQUATE':
        print(f"  MESH NOT ADEQUATE {r['path']}: {r['mesh_verdict']}"); ok = False
print("G-roundtrip:", "PASS" if ok else "FAIL")
json.dump(rows, open('tmp/discreteness_2026-07-26/roundtrip.json', 'w'), indent=1)
