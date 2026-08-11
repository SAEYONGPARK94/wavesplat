"""★잔존 위험 측정 — 출하 경로에서 관측이 cap(600) 이하이면 fit 이 caller 의
배열 ORDER 에 의존한다.  _bank_fit_window 의 `if len(z) > cap` 때문에 결정적
rng(0) 재배열이 발동하지 않기 때문이다.  같은 관측 '집합'을 순서만 바꿔 넣고
출하 SequentialUpdater 의 예측이 달라지는지 측정한다 (합성 바다, 데이터 불필요).
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavesplat.data_loader import FrameData
from wavesplat.optimizer.sequential_update import SequentialUpdater

G, DEPTH = 9.81, 30.0


def om(k):
    return np.sqrt(G * k * np.tanh(k * DEPTH))


def make(n_frames, n_per, seed=11):
    rng = np.random.default_rng(seed)
    P, z = [], []
    for j in range(n_frames):
        t = j * 2.0
        x, y = rng.uniform(0, 80, n_per), rng.uniform(0, 80, n_per)
        Q = np.column_stack([x, y, np.full(n_per, t)])
        zz = (0.8 * np.cos(0.12 * np.cos(.3) * x + 0.12 * np.sin(.3) * y
                           - om(0.12) * t + 0.5)
              + 0.4 * np.cos(0.2 * np.cos(-.8) * x + 0.2 * np.sin(-.8) * y
                             - om(0.2) * t + 1.7)
              + rng.normal(0, 0.03, n_per))
        P.append(Q); z.append(zz)
    return np.vstack(P), np.concatenate(z)


def frames_of(P, z):
    """같은 집합, 프레임 안에서만 순서를 섞는다 (프레임 경계는 그대로)."""
    out = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        out.append(FrameData(t=float(tt),
                             x_obs=torch.tensor(P[m, 0], dtype=torch.float32),
                             y_obs=torch.tensor(P[m, 1], dtype=torch.float32),
                             z_obs=torch.tensor(z[m], dtype=torch.float32),
                             confidence=torch.ones(n, dtype=torch.float32),
                             incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return out


gxn, gyn = np.linspace(0, 80, 21), np.linspace(0, 80, 17)
Xg, Yg = np.meshgrid(gxn, gyn)
gt = (0.8 * np.cos(0.12 * np.cos(.3) * Xg + 0.12 * np.sin(.3) * Yg
                   - om(0.12) * 34.0 + 0.5)
      + 0.4 * np.cos(0.2 * np.cos(-.8) * Xg + 0.2 * np.sin(-.8) * Yg
                     - om(0.2) * 34.0 + 1.7))

for label, nf, npf in (('cap 미발동 (n=560 <= 600)', 14, 40),
                       ('cap 발동   (n=800 >  600)', 20, 40)):
    P, z = make(nf, npf)
    fields = []
    for perm_seed in (None, 5, 9):
        if perm_seed is None:
            Pp, zp = P, z
        else:
            pr = np.random.default_rng(perm_seed).permutation(len(z))
            Pp, zp = P[pr], z[pr]
        upd = SequentialUpdater({'init_method': 'auto', 'depth': DEPTH,
                                 'seed': 7, 'real_bank_ensemble_k': 3,
                                 'device': 'cpu'})
        upd.initialize(frames_of(Pp, zp), verbose=False)
        f = upd.predict_field(torch.tensor(gxn), torch.tensor(gyn),
                              34.0).cpu().numpy()
        fields.append((f, [x.smult for x in upd._bank],
                       [x.pexp for x in upd._bank], upd._bank[0].n_obs))
    base = fields[0][0]
    print(f'\n{label}   n_obs_used={fields[0][3]}')
    for i, (f, sm, pe, _) in enumerate(fields):
        nr = float(np.sqrt(np.mean((f - gt) ** 2)) / np.std(gt))
        print(f'  perm{i}: smult={sm} pexp={pe} nRMSE={nr:.4f} '
              f'max|Δ vs perm0|={np.abs(f-base).max():.3e}')
