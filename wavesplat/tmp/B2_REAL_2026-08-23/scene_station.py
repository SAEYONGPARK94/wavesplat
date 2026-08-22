"""팬 없는 **고정 관측소** 장면 빌더 — 실측 레코드를 5개 → 12개로 연다.

## 왜
`d3_falcon_real_operator.RECORDS` 는 주석에 축자로 적혀 있다:
    `# only records whose patch can host a (possibly scaled) forward fan`
⟹ ★**5개 상한의 이유는 자료가 아니라 "배 앞쪽 부채꼴" 관측 모형이다.**
그런데 **보유 실측 13 레코드는 전부 고정 관측소**이고 경쟁 문헌도 정지 플랫폼이다.
⟹ 팬을 빼면 **독립 레코드 12개**가 되고, 실측에서 **처음으로 α=0.05 가 가능**해진다
(부호검정 바닥: n=5 는 p=0.0625 로 불가, n=12 는 10/12 에서 p=0.0386).

## 무엇을 바꾸고 무엇을 그대로 두는가
- **바꾼 것**: 관측 선택 마스크만. 부채꼴(`r0..r1`, `cos≥0.5`) → **유효 격자점 균등 추출**.
  선속·배 위치·조우 기하가 사라진다(고정 관측소이므로 애초에 없던 것이다).
- **그대로**: 창 선택·시각·잡음·`m_cap`·질의 격자·채점(σ 정규화)·난수 시드 규약.
⟹ ★**"같은 WASS 표면 · 같은 합성 희소화 · 같은 채점"** 이 최대 주장이다.
⛔**"경쟁자와 정면 비교" 는 못 쓴다** — 디스크에 스테레오 원영상이 없어 그들의 입력을
   재현할 수 없다(검수 D1).
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

G = 9.81

# ── 12 독립 레코드 (LJ Grid1/Grid2 는 동시각 ⟹ Grid1 만) ──────────────────
# ⚠BS 7개는 박스가 13–19 m 로 개구/λ≈1 이라 **훨씬 어렵다**. 사후배제 금지.
# ⚠LJ Grid2 는 클램프 인공물(Hs 26 m)로 제외 — 알려진 결함(2026-08-13 감사).
RECORDS_12 = {
    'AA14':   dict(path='data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc',
                   depth=17.0, stride=1),
    'AA15a':  dict(path='data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc',
                   depth=17.0, stride=1),
    'AA15b':  dict(path='data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc',
                   depth=17.0, stride=1),
    'BS11a':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc',
                   depth=17.0, stride=1),
    'BS11b':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc',
                   depth=17.0, stride=1),
    'BS11c':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_130700_short.nc',
                   depth=17.0, stride=1),
    'BS11d':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_153000_short.nc',
                   depth=17.0, stride=1),
    'BS13a':  dict(path='data/real_bergamasco/BS_2013/Surfaces_20130922_130001_short.nc',
                   depth=17.0, stride=1),
    'BS13b':  dict(path='data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc',
                   depth=17.0, stride=1),
    'BS13c':  dict(path='data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc',
                   depth=17.0, stride=1),
    'LJ18a':  dict(path='data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc',
                   depth=45.0, stride=1),
    'YS17':   dict(path='data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc',
                   depth=50.0, stride=2),
}


def scene_station(record, win=3, W=40.0, m_scan=12, m_cap=600, noise=0.02,
                  lead_max=None):
    """`scene_real` 과 같은 dict 을 반환하되 **부채꼴 없이** 균등 추출한다."""
    from realdata_analysis import directional_spread
    from d3_sa_multirecord import MRecord

    cfg = RECORDS_12[record]
    depth = cfg['depth']
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    W_max = 40.0
    nfw = int(round(W_max / rec.dt))
    # ⚠짧은 레코드가 있으므로 lead 를 자료 길이에 맞춘다(사후 조정 금지: 규칙으로 고정).
    lm = lead_max if lead_max is not None else 14.0
    n_lead = int(round(lm / rec.dt))
    if rec.T_all < nfw + n_lead + 4:
        n_lead = max(2, (rec.T_all - nfw - 4))
        lm = n_lead * rec.dt
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    if hi <= lo:
        raise RuntimeError(f'{record}: 창을 놓을 수 없다 (T={rec.T_all})')
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    _, _, _, kp = directional_spread(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)

    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        m = np.isfinite(Zmem[j])                      # ★부채꼴 없음: 유효 전체
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        pick = rng.choice(idx, min(m_scan, idx.size), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, noise, len(pick)))
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
    Pq = np.vstack([np.column_stack([Xq.ravel(), Yq.ravel(),
                                     np.full(npx, t0 + l_)]) for l_ in leads])
    maps_gt = np.array([Zfut[int(round(l_ / rec.dt))][::sq, ::sq] for l_ in leads])
    return dict(P=P, z=z, Pq=Pq, npx=npx, leads=leads, Xq=Xq, Yq=Yq,
                gxq=gxq, gyq=gyq, depth=depth, Hs=Hs, kp=kp, seed=i0,
                maps_gt=maps_gt, dt=rec.dt, t0=t0, sg=float(np.nanstd(Zmem)),
                label=f'Real WASS surface, fixed station ({record}, win {win})')
