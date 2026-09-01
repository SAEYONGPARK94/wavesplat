#!/usr/bin/env python3
"""`T7-SPECTRAL` 러너 — 스펙트럼 영역 지표 재채점. 한 프로세스, 1 스레드."""
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
DATA = 'tmp/HOSGATE_2026-08-13/data_hos'


def episodes():
    """`B-12` 코퍼스 = HOS 비선형 참값의 15·30 성분 전부."""
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
            print(f'[{i + 1:2d}/48] {arm:5s} {ep[:46]:46s} '
                  f'dirR2 {np.median(a_vals):6.3f}  hsErr {np.median(b_vals):6.3f}',
                  flush=True)
    out = os.path.join(HERE, 'results_t7.jsonl')
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
