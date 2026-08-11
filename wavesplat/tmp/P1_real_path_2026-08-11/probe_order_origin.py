"""★순서가 어디서 갈리는가 — 600-cap 이 no-op 이 되는 순간을 특정한다.

scene_real: 시간순 P_all(≈4800점) -> rng(0).choice(n,600) -> **셔플된** 600점.
배포 래퍼(make_gif_bank_deployed.frames_from_scene): 그 셔플된 600점을 받아
np.unique(t) 로 다시 **시간순 정렬** -> _bank_fit_window 의 cap 은 len==600 이라
**발동하지 않음** -> fit_bank_mmse 의 위치기반 val 분할이 다른 물리점을 고른다.

가설: 배포 경로에 **원본 스트림(미캡)** 을 주면 cap 이 발동해 scene_real 과 같은
셔플을 재현하고 두 경로가 일치한다.  측정한다.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'))
os.chdir(ROOT)

import d3_compare_gifs as cg
import run_rolling_refit as RR
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.optimizer.bank_mmse import fit_bank_mmse
from wavesplat.data_loader import FrameData

REC, WIN = 'AA15a', 5
sc = cg.scene_real(REC, win=WIN)
st = RR.build_stream(REC, WIN)
P0, z0 = np.asarray(sc['P'], float), np.asarray(sc['z'], float)

# --- 원본(미캡) 40 s 스트림: scene_real 의 P_all 과 동일해야 한다
m = (st['P_all'][:, 2] >= st['t0'] - 40.0) & (st['P_all'][:, 2] <= st['t0'] + 1e-9)
P_full, z_full = st['P_all'][m], st['z_all'][m]
print(f'미캡 스트림: {len(z_full)}점 (시간순)')
ii = np.random.default_rng(0).choice(len(z_full), 600, replace=False)
print(f'  rng(0) cap 재현 == sc[P]: '
      f'{np.array_equal(P_full[ii], P0) and np.array_equal(z_full[ii], z0)}')

print()
print('=== A) FLOAT32 만의 영향 (비트 수준) ===')
Pf = P0.copy()
Pf[:, 0] = np.asarray(P0[:, 0], np.float32).astype(float)
Pf[:, 1] = np.asarray(P0[:, 1], np.float32).astype(float)
zf = np.asarray(z0, np.float32).astype(float)
s0 = 0.05 * 4.0 * float(np.std(z0))
s0f = 0.05 * 4.0 * float(np.std(zf))
gxq, gyq = np.asarray(st['gxq']), np.asarray(st['gyq'])
Xq, Yq = np.meshgrid(gxq, gyq)
Q = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, st['t0'] + 14.0)])
a = np.median(np.stack([fit_bank_mmse(P0, z0, sc['depth'], s0, seed=s).render(Q)
                        for s in RR.SEEDS], 0), 0)
b = np.median(np.stack([fit_bank_mmse(Pf, zf, sc['depth'], s0f, seed=s).render(Q)
                        for s in RR.SEEDS], 0), 0)
print(f'  f64 vs f32 관측: 최대|Δ| = {np.abs(a-b).max():.3e}  (array_equal={np.array_equal(a,b)})')

print()
print('=== B) 미캡 스트림을 배포 경로(SequentialUpdater)에 먹인다 ===')
def frames_from(P, z, t_shift=0.0, f32=True):
    dt_ = torch.float32 if f32 else torch.float64
    out = []
    for tt in np.unique(P[:, 2]):
        mm = P[:, 2] == tt
        n = int(mm.sum())
        out.append(FrameData(t=float(tt - t_shift),
                             x_obs=torch.tensor(P[mm, 0], dtype=dt_),
                             y_obs=torch.tensor(P[mm, 1], dtype=dt_),
                             z_obs=torch.tensor(z[mm], dtype=dt_),
                             confidence=torch.ones(n, dtype=torch.float32),
                             incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return out

for f32 in (True, False):
    cfg = {'init_method': 'auto', 'depth': float(sc['depth']), 'seed': 42,
           'device': 'cpu', 'real_bank_ensemble_k': 3}
    upd = SequentialUpdater(cfg)
    upd.initialize(frames_from(P_full, z_full, f32=f32), verbose=False)
    bank = upd._bank
    print(f'  [FrameData f32={f32}] n_obs={[f.n_obs for f in bank]} '
          f'pexp={[f.pexp for f in bank]} smult={[f.smult for f in bank]} '
          f'vr={[round(f.vr,6) for f in bank]}')
    d = np.median(np.stack([f.render(Q) for f in bank], 0), 0)
    print(f'    vs 연구(f64) 최대|Δ| = {np.abs(a-d).max():.3e}   '
          f'vs 연구(f32관측) 최대|Δ| = {np.abs(b-d).max():.3e}')
