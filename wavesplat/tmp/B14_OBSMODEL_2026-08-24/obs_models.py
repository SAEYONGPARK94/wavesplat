"""B-14 장면 빌더 — `scene_station` 과 **같은 창·같은 채점**, 관측 모형만 갈아 끼운다.

⛔`tmp/B2_REAL_2026-08-23/scene_station.py` 는 **동결 파일이라 수정하지 않는다.**
여기서 그 함수의 구조를 옮기되 *표본 추출 + 잡음* 단계만 교체한다.
★그 이식이 충실한지는 **게이트 G-1** 이 잡는다: `IID · σ=0.02` 팔은 `B-2b` 의
`noise=0.02` 행과 **비트 동일**해야 한다(같은 난수 호출 순서를 유지했기 때문).

관측 모형 4종 (전부 **스테레오** 계측에서 물리적 동기가 있다):
  IID   : 유효 격자 균등 추출 + 등방 i.i.d. 가우스              (= 배포)
  CORR  : 균등 추출 + 공간 상관 잡음(상관거리 CORR_LEN, 주변 σ 동일)
          — 스테레오 재구성 오차는 정합 창 크기 규모로 상관된다
  CLUST : 군집 추출(중심 N_CENTER개, 반경 CLUST_R) + i.i.d.
          — 스테레오 정합은 질감이 있는 곳에 몰린다
  SLOPE : 균등 추출 + 국소 경사 비례 이분산(평균 분산 동일)
          — 경사면에서 정합 오차가 커진다
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (B2, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from scene_station import RECORDS_12          # noqa: E402  (동결 파일에서 가져온다)

# ⚠**2026-08-24 수정 이력**: `noise=0` 조기 반환이 난수 소비를 건너뛰어
# 잡음 수준마다 **표본이 달라졌다**(짝 개입이 아니게 됨). `B-14b` 가 수정 후 다시 잰다.
CORR_LEN = 8.0      # m — 결과 전 고정
CLUST_R = 5.0       # m
N_CENTER = 3
_MODELS = ('IID', 'CORR', 'CLUST', 'SLOPE')


def _sample_frame(Z, gx, gy, rng, model, noise, m_scan):
    """한 프레임에서 (ix, iy, z_obs) 를 낸다. 반환 dict 에 개입 계수기를 함께 싣는다."""
    m = np.isfinite(Z)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    n_take = min(m_scan, idx.size)

    if model == 'CLUST':
        # 중심 N_CENTER 개 → 반경 CLUST_R 안에서 추출 (부족하면 균등으로 채운다)
        c = rng.choice(idx, min(N_CENTER, idx.size), replace=False)
        ciy, cix = np.unravel_index(c, Z.shape)
        iy_all, ix_all = np.unravel_index(idx, Z.shape)
        near = np.zeros(idx.size, bool)
        for yy, xx in zip(gy[ciy], gx[cix]):
            near |= (np.hypot(gx[ix_all] - xx, gy[iy_all] - yy) <= CLUST_R)
        pool = idx[near] if near.sum() >= n_take else idx
        pick = rng.choice(pool, min(n_take, pool.size), replace=False)
    else:
        pick = rng.choice(idx, n_take, replace=False)

    iy, ix = np.unravel_index(pick, Z.shape)
    z = Z[iy, ix].astype(float)

    # ⛔**난수 소비 동수 유지가 필수다.**  `scene_station` 은 `noise=0` 에서도
    # `rng.normal(0, 0.0, n)` 을 호출해 **난수 상태를 전진**시킨다. 여기서 조기 반환하면
    # 이후 프레임의 `rng.choice` 가 달라져 **잡음만 바꾼 개입이 아니라 표본까지 바뀐다**.
    # (2026-08-24 `G-1` 양성대조가 이 결함을 잡았다 — B-14 축 A 는 그래서 무효다.)
    if noise <= 0.0:
        eps = rng.normal(0, 0.0, len(pick))      # ★상태 전진을 위해 반드시 호출한다
    elif model == 'CORR':
        w = rng.normal(0.0, 1.0, Z.shape)
        dx = float(np.median(np.diff(gx))) if len(gx) > 1 else 1.0
        f = gaussian_filter(w, CORR_LEN / max(dx, 1e-6), mode='nearest')
        sd = float(np.std(f))
        eps = (f[iy, ix] / sd * noise) if sd > 0 else np.zeros(len(pick))
    elif model == 'SLOPE':
        gyz, gxz = np.gradient(np.nan_to_num(Z))
        s = np.hypot(gxz, gyz)
        rms = float(np.sqrt(np.mean(s[m] ** 2)))
        sj = (s[iy, ix] / rms * noise) if rms > 0 else np.full(len(pick), noise)
        eps = rng.normal(0.0, 1.0, len(pick)) * sj
    else:                       # IID (배포) — 난수 호출 순서를 원본과 동일하게 유지
        eps = rng.normal(0, noise, len(pick))
    return ix, iy, z + eps


def scene_obs(record, win=3, model='IID', noise=0.02, m_scan=12, m_cap=600,
              W=40.0, lead_max=14.0):
    """`scene_station` 과 같은 dict. 관측 모형만 `model` 로 갈아 끼운다."""
    assert model in _MODELS, model
    from realdata_analysis import directional_spread
    from d3_sa_multirecord import MRecord

    cfg = RECORDS_12[record]
    depth = cfg['depth']
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    lm = lead_max
    n_lead = int(round(lm / rec.dt))
    if rec.T_all < nfw + n_lead + 4:
        n_lead = max(2, (rec.T_all - nfw - 4))
        lm = n_lead * rec.dt
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    if hi <= lo:
        raise RuntimeError(f'{record}: 창을 놓을 수 없다 (T={rec.T_all})')
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)      # ★원본과 같은 시드 규약
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    _, _, _, kp = directional_spread(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)

    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        got = _sample_frame(Zmem[j], rec.gx, rec.gy, rng, model, noise, m_scan)
        if got is None:
            continue
        ix, iy, zz = got
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy], np.full(len(ix), t_j)]))
        z_all.append(zz)
    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    m = P_all[:, 2] >= t0 - W
    P, z = P_all[m], z_all[m]
    if len(z) > m_cap:
        ii = np.random.default_rng(0).choice(len(z), m_cap, replace=False)
        P, z = P[ii], z[ii]

    leads = np.arange(0.0, lm + 1e-9, 1.0)
    sq = max(1, int(round(4.0 / rec.dx)))
    gxq, gyq = rec.gx[::sq], rec.gy[::sq]
    Xq, Yq = np.meshgrid(gxq, gyq)
    npx = Xq.size
    Pq = np.vstack([np.column_stack([Xq.ravel(), Yq.ravel(), np.full(npx, t0 + l_)])
                    for l_ in leads])
    maps_gt = np.array([Zfut[int(round(l_ / rec.dt))][::sq, ::sq] for l_ in leads])
    return dict(P=P, z=z, Pq=Pq, npx=npx, leads=leads, Xq=Xq, Yq=Yq,
                gxq=gxq, gyq=gyq, depth=depth, Hs=Hs, kp=kp, seed=i0,
                maps_gt=maps_gt, dt=rec.dt, t0=t0, sg=float(np.nanstd(Zmem)),
                obs_model=model, noise=noise, m_scan=m_scan, n_obs_used=int(len(z)),
                label=f'B-14 {record} win{win} obs={model} σ={noise} m={m_scan}')
