"""`R8-HELDOUT` 장면 생성 — `scene_station` 의 **한 줄 변형**.

⛔`tmp/B2_REAL_2026-08-23/scene_station.py` 를 **수정하지 않는다**(임포트만).
⛔바뀌는 것은 **프레임별 관측 선택 마스크에 AND 되는 영역 마스크** 하나뿐이고,
`region='ALL'` 이면 원본과 **비트 동일**해야 한다(`G-BIT` 가 강제).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for _p in (B2DIR, ROOT, os.path.join(ROOT, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import scene_station as SS                                           # noqa: E402

M_SCAN = 12
M_CAP = 600
NOISE = 0.02
W_WIN = 40.0
LEAD_MAX = 14.0


def _raw_window(rec, i0, n):
    """`R1b` 와 같은 규약 — demean 하지 않는다."""
    from d3_sa_multirecord import _unpack
    s = rec.stride
    Z = (_unpack(rec.zd, np.s_[i0:i0 + n, ::s, ::s]) if s > 1
         else _unpack(rec.zd, np.s_[i0:i0 + n]))
    Z[:, ~rec.valid] = np.nan
    return Z


def scene_heldout(record, win=3, region='ALL'):
    """`scene_station` 과 같되 관측 선택에 **영역 마스크**를 AND 한다.

    `region`:
      · `'ALL'`  — 마스크 전부 True ⟹ ★`scene_station` 과 **비트 동일**
      · `'LEFT'` — `x <= x_split` 인 격자 열에서만 관측을 뽑는다
    """
    if region not in ('ALL', 'LEFT'):
        raise ValueError(f'region={region!r} — ALL 또는 LEFT')
    from realdata_analysis import directional_spread
    from d3_sa_multirecord import MRecord

    cfg = SS.RECORDS_12[record]
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    nfw = int(round(W_WIN / rec.dt))
    lm = LEAD_MAX
    n_lead = int(round(lm / rec.dt))
    if rec.T_all < nfw + n_lead + 4:
        n_lead = max(2, (rec.T_all - nfw - 4))
        lm = n_lead * rec.dt
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    if hi <= lo:
        raise RuntimeError(f'{record}: 창을 놓을 수 없다 (T={rec.T_all})')
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    x_split = float(np.median(rec.gx))
    scene_seed = 700 + win
    rng = np.random.default_rng(scene_seed)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    _, _, _, kp = directional_spread(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
    lam = float(2.0 * np.pi / kp) if kp and kp > 0 else float('nan')

    # ★영역 마스크 — 격자 열(x) 기준. `'ALL'` 이면 전부 True ⟹ idx 가 원본과 동일.
    col_ok = (rec.gx <= x_split) if region == 'LEFT' else np.ones(len(rec.gx), bool)
    reg = np.broadcast_to(col_ok[None, :], Zmem[0].shape)

    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        m = np.isfinite(Zmem[j]) & reg
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        pick = rng.choice(idx, min(M_SCAN, idx.size), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, NOISE, len(pick)))
    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    m = P_all[:, 2] >= t0 - W_WIN
    P, z = P_all[m], z_all[m]
    if len(z) > M_CAP:
        ii = np.random.default_rng(0).choice(len(z), M_CAP, replace=False)
        P, z = P[ii], z[ii]

    leads = np.arange(0.0, lm + 1e-9, 1.0)
    sq = max(1, int(round(4.0 / rec.dx)))
    gxq, gyq = rec.gx[::sq], rec.gy[::sq]
    Xq, Yq = np.meshgrid(gxq, gyq)
    npx = Xq.size
    Pq = np.vstack([np.column_stack([Xq.ravel(), Yq.ravel(),
                                     np.full(npx, t0 + l_)]) for l_ in leads])
    maps_gt = np.array([Zfut[int(round(l_ / rec.dt))][::sq, ::sq] for l_ in leads])
    sg = float(np.nanstd(Zmem))

    # ★`R1b` 수리 규약의 보정장
    mu_mem = np.nanmean(_raw_window(rec, i0 - nfw + 1, nfw), axis=0)
    mu_fut = np.nanmean(_raw_window(rec, i0, n_lead + 1), axis=0)
    delta_q = (mu_fut - mu_mem)[::sq, ::sq]

    # 채점 영역 마스크 (질의 격자 기준)
    in_mask = np.broadcast_to((gxq <= x_split)[None, :], Xq.shape)
    out_mask = ~in_mask

    # 질의 → 최근접 관측 간격 (x-y 평면)
    d = np.sqrt((Xq.ravel()[:, None] - P[None, :, 0]) ** 2
                + (Yq.ravel()[:, None] - P[None, :, 1]) ** 2).min(axis=1)
    d = d.reshape(Xq.shape)

    return dict(P=P, z=z, Pq=Pq, gxq=gxq, gyq=gyq, leads=leads, maps_gt=maps_gt,
                sg=sg, delta_q=delta_q, x_split=x_split, lam=lam, kp=float(kp),
                in_mask=in_mask, out_mask=out_mask, gap=d, depth=cfg['depth'],
                scene_seed=scene_seed, i0=i0, sq=sq, dt=float(rec.dt),
                dx=float(rec.dx), region=region)
