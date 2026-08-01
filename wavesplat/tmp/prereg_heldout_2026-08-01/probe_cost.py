"""Cost/feasibility probe for the 2026-08-01 held-out prereg run.

Measures, on ONE window per record: grid size, number of taus, wall-time of a
median-of-3 bank-MMSE fit, of a median-of-3 v2_konly init (arm C's unit cost),
and of one grid render for each.  No prereg quantity is computed here.
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'bankmmse_port_2026-07-29'))
import run_rolling_refit as rr   # sets ROOT/scripts paths and chdirs to ROOT

import torch
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.optimizer.bank_mmse import fit_bank_mmse
from wavesplat.data_loader import FrameData
from wavesplat.run import DEFAULT_CONFIG, set_seed

W_MAX = rr.W_MAX


def frames_from(P, z, t_shift):
    out = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        out.append(FrameData(t=float(tt - t_shift),
                             x_obs=torch.tensor(P[m, 0], dtype=torch.float32),
                             y_obs=torch.tensor(P[m, 1], dtype=torch.float32),
                             z_obs=torch.tensor(z[m], dtype=torch.float32),
                             confidence=torch.ones(n, dtype=torch.float32),
                             incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return out


def main():
    for record, win in [('AA14', 0), ('LJ18g', 0), ('YS17', 0)]:
        t = time.time()
        st = rr.build_stream(record, win)
        tb = time.time() - t
        t = time.time()
        rr.assert_replication(st, record, win)
        ta = time.time() - t
        taus = np.arange(0.0, st['lead_max'] + 1e-9, 1.0)
        ny, nx = len(st['gyq']), len(st['gxq'])

        m = (st['P_all'][:, 2] >= st['t0'] - W_MAX) & (st['P_all'][:, 2] <= st['t0'] + 1e-9)
        P, z = st['P_all'][m], st['z_all'][m]
        if len(z) > 600:
            ii = np.random.default_rng(0).choice(len(z), 600, replace=False)
            P, z = P[ii], z[ii]
        s0 = 0.2 * float(np.std(z))

        t = time.time()
        fits = [fit_bank_mmse(P, z, st['depth'], s0, kp_hint=None, seed=s)
                for s in (42, 1042, 2042)]
        t_bank = time.time() - t
        Xq, Yq = np.meshgrid(st['gxq'], st['gyq'])
        Q = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, st['t0'])])
        t = time.time()
        np.median(np.stack([f.render(Q) for f in fits], 0), 0)
        t_rbank = time.time() - t

        # arm C unit cost: v2_konly init, one seed (x3 in the real run)
        t_shift = float(P[:, 2].min())
        fr = frames_from(P, z, t_shift)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(depth=float(st['depth']), seed=42, device='cpu',
                   init_method='v2_konly')
        set_seed(42)
        upd = SequentialUpdater(cfg)
        t = time.time()
        upd.initialize(fr, x_range=(float(st['gxq'].min()), float(st['gxq'].max())),
                       y_range=(float(st['gyq'].min()), float(st['gyq'].max())),
                       verbose=False)
        t_v2 = time.time() - t
        t = time.time()
        upd.predict_field(torch.tensor(st['gxq'], dtype=torch.float32),
                          torch.tensor(st['gyq'], dtype=torch.float32),
                          float(st['t0'] - t_shift)).cpu().numpy()
        t_rv2 = time.time() - t

        print(f'{record} w{win}: grid {ny}x{nx}={ny*nx} px, n_tau={len(taus)}, '
              f'n_obs={len(z)}, lead_max={st["lead_max"]}, dt={st["dt"]}\n'
              f'   stream {tb:.1f}s  assert {ta:.1f}s | bank m3 fit {t_bank:.2f}s '
              f'render {t_rbank*1000:.0f}ms | v2_konly 1-seed init {t_v2:.1f}s '
              f'render {t_rv2*1000:.0f}ms', flush=True)


if __name__ == '__main__':
    main()
