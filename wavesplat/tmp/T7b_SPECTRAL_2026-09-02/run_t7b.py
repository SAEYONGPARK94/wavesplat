#!/usr/bin/env python3
"""`T7b-SPECTRAL` 러너 — `T7` 과 **다른 코퍼스**(`data_s8_crowded_2026-08-10`).

⛔`T7` 자료로 재판정하지 않는다(사후 완화 방지). 시드 대역과 참값 생성기가 다르다.
"""
import os
import sys

for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '1'

import json                                                    # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import h5py                                                    # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
torch.set_num_threads(1)

from wavesplat.run import DEFAULT_CONFIG                       # noqa: E402
from wavesplat.data_loader import EpisodeDataLoader            # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater  # noqa: E402
from wavesplat.evaluation.metrics import hs_error, directional_r2  # noqa: E402

SEED = 42
STRIDE = 40                       # PREREG §1: 40 프레임 간격 표집
DATA = 'data_s8_crowded_2026-08-10'


def episodes():
    """`S8` 혼잡 코퍼스 32 에피소드(BF4/BF5 x 15/30성분 x 시드 32--39)."""
    return sorted(f for f in os.listdir(os.path.join(ROOT, DATA))
                  if f.endswith('.h5') and ('_15comp_' in f or '_30comp_' in f))


def main():
    rows = []
    for i, ep in enumerate(episodes()):
        path = os.path.join(ROOT, DATA, ep)
        with h5py.File(path, 'r') as f:
            hs_true = float(f['config'].attrs['Hs'])
        loader = EpisodeDataLoader(path)
        init_w = loader.get_init_window(tp_factor=2.0)
        track = loader.get_tracking_frames(tp_factor=2.0)
        idx = list(range(0, len(track), STRIDE))
        gx = loader.gt_grid_x.cpu().numpy()
        gy = loader.gt_grid_y.cpu().numpy()
        XX, YY = np.meshgrid(gx, gy)
        for arm, init_method in (('LEARN', 'auto'), ('ANA', 'v2_konly')):
            cfg = dict(DEFAULT_CONFIG)
            cfg['depth'] = float(loader.depth)
            cfg['seed'] = SEED
            cfg['init_method'] = init_method
            torch.manual_seed(SEED)
            upd = make_updater(cfg)
            upd.initialize(init_w)
            prim = getattr(upd, '_primary', upd)
            for fr in track:
                upd.update(fr)
            a_vals, b_vals, z_vals = [], [], []
            for k in idx:
                fr = track[k]
                gt = fr.gt_field.cpu().numpy()
                with torch.no_grad():
                    est = prim.model(
                        torch.tensor(XX.ravel(), dtype=torch.float32),
                        torch.tensor(YY.ravel(), dtype=torch.float32),
                        torch.tensor(float(fr.t))).cpu().numpy().reshape(gt.shape)
                a_vals.append(float(directional_r2(est, gt)))
                b_vals.append(float(hs_error(est, hs_true)))
                z_vals.append(float(directional_r2(np.zeros_like(gt), gt)))
            eff = {k: getattr(prim, k) for k in
                   ('lambda_l1', 'tracking_steps', 'tracking_window', 'tracking_mode')}
            eff.update({k: prim.config[k] for k in
                        ('varpro_steps', 'learned_init_iters')})
            rows.append(dict(
                episode=ep, arm=arm, seed=SEED, hs_true=hs_true,
                route_effective=(getattr(prim, '_auto_selected', None)
                                 or prim.init_method),
                scored_idx=idx, n_scored=len(idx),
                dir_r2=a_vals, hs_err=b_vals, dir_r2_zero=z_vals,
                effective=eff, final_n=int(prim.model.n_gaussians)))
            print(f'[{i + 1:2d}/32] {arm:5s} {ep[:46]:46s} '
                  f'dirR2 {np.median(a_vals):6.3f}  hsErr {np.median(b_vals):6.3f}',
                  flush=True)
    out = os.path.join(HERE, 'results_t7b.jsonl')
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
