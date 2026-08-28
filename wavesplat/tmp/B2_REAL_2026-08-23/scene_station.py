"""팬 없는 **고정 관측소** 장면 빌더 — 실측 레코드를 5개 → 12개로 연다.

## 왜
`d3_falcon_real_operator.RECORDS` 는 주석에 축자로 적혀 있다:
    `# only records whose patch can host a (possibly scaled) forward fan`
⟹ ★**5개 상한의 이유는 자료가 아니라 "배 앞쪽 부채꼴" 관측 모형이다.**
그런데 **보유 실측 13 레코드는 전부 고정 관측소**이고 경쟁 문헌도 정지 플랫폼이다.
⟹ 팬을 빼면 **독립 레코드 12개**가 되고, 실측에서 **처음으로 α=0.05 가 가능**해진다
⛔**(2026-08-23 정정) 여기 적었던 *"n=5 는 p=0.0625 로 불가"* 는 양측값 오용이다** —
`sign_test_one_sided(5,5)=0.03125` 로 **전승이면 α=0.05 를 달성한다**. 정확한 사유는
*"n=5 는 전승만 유의 = 검정력이 없다"* 이고, n=12 는 10/12·11/12·12/12 세 단계가 유의하다
(단측 0.0193 · 0.0032 · 0.0002).

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
# ⛔2026-08-23 수정: BS 7개의 depth 를 17.0(=AA 값을 복사한 오류) → **30.0**(Black Sea ~30 m,
# reference_bergamasco_stereo_dataset.md:17) 으로 고쳤다. ⚠**B-2…B-2f 는 17.0 으로 돌았다.**
# 잰 효과는 첨두에서 Δω = 0.0000–0.0012% (kd≫1 심해) ⟹ 수치적으로 무해하다. AMENDMENT_ZEROREF.md §5.
RECORDS_12 = {
    'AA14':   dict(path='data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc',
                   depth=17.0, stride=1),
    'AA15a':  dict(path='data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc',
                   depth=17.0, stride=1),
    'AA15b':  dict(path='data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc',
                   depth=17.0, stride=1),
    'BS11a':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc',
                   depth=30.0, stride=1),
    'BS11b':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc',
                   depth=30.0, stride=1),
    'BS11c':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_130700_short.nc',
                   depth=30.0, stride=1),
    'BS11d':  dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_153000_short.nc',
                   depth=30.0, stride=1),
    'BS13a':  dict(path='data/real_bergamasco/BS_2013/Surfaces_20130922_130001_short.nc',
                   depth=30.0, stride=1),
    'BS13b':  dict(path='data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc',
                   depth=30.0, stride=1),
    'BS13c':  dict(path='data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc',
                   depth=30.0, stride=1),
    'LJ18a':  dict(path='data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc',
                   depth=45.0, stride=1),
    'YS17':   dict(path='data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc',
                   depth=50.0, stride=2),
}


def scene_station(record, win=3, W=40.0, m_scan=12, m_cap=600, noise=0.02,
                  lead_max=None, with_future_obs=False, cap_mode='uniform'):
    """`scene_real` 과 같은 dict 을 반환하되 **부채꼴 없이** 균등 추출한다.

    `with_future_obs=True` 면 **미래 관측 프레임**도 함께 낸다(`P_fut`/`z_fut`, 1 s 간격,
    과거 프레임과 **완전히 같은 추출·잡음 규약**). 순차 추적(`update()`) 을 재려면 필요하다.
    ⚠**기본 False 이고 그때는 반환 dict 이 비트 동일**하다 — B-2…B-2f 재현을 깨지 않는다.
    ⛔미래 관측은 **채점에 쓰지 마라**. 채점은 `maps_gt` 로만 한다.

    ★`cap_mode` (2026-08-28 신설, `M5` 진단 `AMENDMENT_1.md §2` 의 처방) —
    `m_cap` 추첨 규약. **기본 `'uniform'` 은 2026-08-28 이전 동작과 비트 동일**하다
    (선례 = 위 `with_future_obs`). 회귀 시험 =
    `tmp/M5b_WINDOW_2026-08-28/regress_scene_station.py`.

      · `'uniform'`      — 잘라낸 뒤 후보 **전체**에서 `default_rng(0)` 균등추첨(현행).
      · `'end_anchored'` — ***가장 최신의 비어있지 않은 스캔은 언제나 통째로 남기고***,
                           남은 예산 `m_cap − k` 만 나머지에서 같은 난수로 균등추첨한다.

    ⛔**왜 필요한가** — `'uniform'` 에서 `W` 는 `m = P_all[:,2] >= t0 - W` **한 줄**로만
    들어가는 것처럼 보이지만, 잘라낸 뒤의 후보 수 `len(z) = N(W)` 가 **곧 추첨의 모집단
    크기**다. `default_rng(0).choice(N, m_cap, replace=False)` 는 `N` 이 바뀌면 인덱스
    집합이 통째로 바뀌므로, 배열 꼬리(=가장 최신 프레임)를 통째로 놓치면
    **관측 끝 시각이 `W` 에 따라 뒤로 밀린다**(실측: `dt=1/15` 레코드에서 마지막 3프레임
    탈락, Δ=0.2 s). ⟹ ★***`W` 에 미공시 두 번째 경로가 있었다.*** `'end_anchored'` 는
    끝 시각을 **구조적 불변식**으로 만들고, `n_obs` 는 `k + (m_cap − k) = m_cap` 으로
    여전히 정확히 `m_cap` 이다(개수 교락 없음).
    ⚠`'end_anchored'` 의 장면은 `B-2`…`B-10` 동결 장면과 **비트동일하지 않다**
    (588-of-(N−k) 추첨은 600-of-N 과 다른 집합이다) — *"`B-2` 와 같은 장면"* 이라 쓰지 마라."""
    if cap_mode not in ('uniform', 'end_anchored'):
        # ⛔조용한 no-op 금지: 오타난 규약이 기본 동작으로 흘러 들어가면 그 배치는
        #    "처치를 안 걸고 걸었다고 적는" 상태가 된다(오늘 `M4 G-13` 과 같은 부류).
        raise ValueError(f"cap_mode={cap_mode!r} — 'uniform' 또는 'end_anchored'")
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
        if cap_mode == 'end_anchored':
            # 최신 스캔을 통째로 확보하고 **남은 예산만** 균등추첨한다.
            last = np.flatnonzero(P[:, 2] >= P[:, 2].max() - 1e-9)
            if len(last) >= m_cap:                     # 구조적으로 불가(m_scan ≪ m_cap)
                raise RuntimeError(f'{record}: 최신 스캔 {len(last)} ≥ m_cap {m_cap}')
            rest = np.setdiff1d(np.arange(len(z)), last)
            ii = np.concatenate([last,
                                 np.random.default_rng(0).choice(
                                     rest, m_cap - len(last), replace=False)])
        else:                                          # 'uniform' = 2026-08-28 이전 현행
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
    fut = {}
    if with_future_obs:
        # ★과거와 같은 규약: 프레임당 m_scan 점, 유효 격자에서 균등, 같은 잡음.
        # 시드는 과거와 겹치지 않도록 분리한다(`900+win`) — 결과 전 고정.
        rng_f = np.random.default_rng(900 + win)
        Pf_all, zf_all = [], []
        for l_ in leads[1:]:                       # lead 0 은 관측 없음(초기화 시점)
            j = int(round(l_ / rec.dt))
            if j >= Zfut.shape[0]:
                continue
            t_j = float(rec.t[i0 + j])
            mfin = np.isfinite(Zfut[j])
            idx = np.flatnonzero(mfin)
            if idx.size == 0:
                continue
            pick = rng_f.choice(idx, min(m_scan, idx.size), replace=False)
            iy, ix = np.unravel_index(pick, Zfut[j].shape)
            Pf_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                           np.full(len(pick), t_j)]))
            zf_all.append(Zfut[j][iy, ix] + rng_f.normal(0, noise, len(pick)))
        fut = dict(P_fut=np.vstack(Pf_all) if Pf_all else np.zeros((0, 3)),
                   z_fut=np.concatenate(zf_all) if zf_all else np.zeros(0))
    return dict(**fut, P=P, z=z, Pq=Pq, npx=npx, leads=leads, Xq=Xq, Yq=Yq,
                gxq=gxq, gyq=gyq, depth=depth, Hs=Hs, kp=kp, seed=i0,
                maps_gt=maps_gt, dt=rec.dt, t0=t0, sg=float(np.nanstd(Zmem)),
                label=f'Real WASS surface, fixed station ({record}, win {win})')
