"""B-17 장면 빌더 — `scene_station` 과 **같은 창·같은 표본·같은 채점**, **잡음 모형만** 바꾼다.

⛔`tmp/B2_REAL_2026-08-23/scene_station.py` 는 동결 파일이라 **수정하지 않는다.**
여기서 그 함수의 구조를 옮기되 *잡음 산출* 한 단계만 교체한다.
★이식 충실도는 **게이트 `G-1`** 이 잡는다: `CONST`(σ≡0.02) 팔은
`B2_REAL_2026-08-23/results.jsonl` 의 `PROP` 행과 **비트 동일**해야 한다.

## ★난수 소비 동수를 어떻게 지키는가 (B-14 축 A 를 죽인 결함)
원본(`scene_station.py:110,114`)은 프레임마다
    `pick = rng.choice(idx, n, replace=False)` → `eps = rng.normal(0, noise, n)`
을 부른다. 여기서는 **`u = rng.normal(0, 1.0, n)`(표준정규)** 를 부르고 뒤에서 `σ_j` 를 곱한다.
`Generator.normal` 은 scale 과 무관하게 출력 원소당 정확히 한 값을 소비하므로
**난수 상태 전진이 동일**하고, 스칼라 σ 에 대해서는 결과까지 **비트 동일**하다
(`np.random.default_rng(s).normal(0,σ,n)` ≡ `σ*np.random.default_rng(s).normal(0,1,n)`,
2026-08-25 실측 확인 — n∈{1,7,12}·σ∈{0.005,0.01,0.02} 전건 True).
⟹ ★**모든 팔이 정확히 같은 관측점 `P` 를 받는다**(게이트 `G-3` 이 해시로 검사).

## 잡음 모형 (논문에서 인용, 손으로 고른 것 없음)
⛔**줄번호로 인용하지 않는다**(2026-08-26 수리 m-5) — 논문 편집으로 어긋난다.
상수는 `gates_b17` 이 **코드 단일출처에서 파싱**하고(σ_r ← `wavesim/noise_model.py`
`add_noise(sigma_range=)`, d_ref ← `wavesplat/model/confidence.py`
`observation_weights(d_ref=)`), σ_0 만 선언한 뒤 `gates_b17.thesis_anchors()` 가
**정규식 앵커**로 논문 본문과 런타임 대조한다(게이트 `G-10`).
`thesis_main.tex` 관측 잡음 분산식 (§관측 모형) 축자:
    Var(ε_j) = σ_0² + σ_r² (r_j / d_ref)²,   σ_0 = 0.01 m, σ_r = 0.02 m, d_ref = 125 m
같은 절의 기하 신뢰 가중치 축자:
    c_j = cos θ_j / (r_j / d_ref),  θ_j = 빔 방향과 해면 법선 사이 입사각
가중최소자승에서 가중치는 분산의 역수이므로, 위 두 식이 함께 선언하는 θ 의존은 **1/cos θ_j** 다.
⟹ `range_angle` 은 두 식의 곱이다:  Var(ε_j) = [σ_0² + σ_r²(r_j/d_ref)²] / cos θ_j
⚠**곱으로 합성한다는 것만이 이 파일의 선택**이며 PREREG §4 에 결과 전에 적었다.
(θ=0 이면 `range_angle` ≡ `range` ⟹ 귀속이 깨끗하다.)

## 국소 경사 → 입사각
`B14_OBSMODEL_2026-08-24/obs_models.py:77-78` 의 국소 경사 계산을 재사용하되
**단위를 격자 인덱스에서 미터로 고친다**(B-14 `SLOPE` 는 rms 정규화라 단위가 소거됐다).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B14 = os.path.abspath(os.path.join(HERE, '..', 'B14_OBSMODEL_2026-08-24'))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, B2, B14, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from scene_station import RECORDS_12          # noqa: E402  (동결 파일에서 가져온다)


def local_slope(Z, dx, dy):
    """국소 경사 (dZ/dx, dZ/dy) [무차원].

    `obs_models.py:77` 과 **같은 호출**(`np.gradient(np.nan_to_num(Z))`) 을 쓴다.
    `Z` 는 `[iy, ix]` 이므로 `np.gradient` 의 axis0 = y, axis1 = x 다.
    ⚠**단위 보정**: `np.gradient` 는 격자 인덱스당 차분을 낸다 ⟹ dx, dy 로 나눈다.
    """
    gyz, gxz = np.gradient(np.nan_to_num(Z))
    return gxz / dx, gyz / dy


def sigma_field(x, y, z, Zx, Zy, spec, sigma0, sigma_r, d_ref, cos_min):
    """점마다의 관측 잡음 표준편차 σ_j 와 진단량을 낸다.

    `spec` = `gates_b17.ARMS[...]` 의 dict.
    """
    model = spec['model']
    n = len(x)
    if model == 'const':
        s = np.full(n, float(spec['sigma']))
        return s, dict(r_med=None, cos_med=None, frac_cos_clamped=0.0)
    h = float(spec['h'])
    xs, ys = float(spec['x_s']), float(spec['y_s'])
    # ── 3-D 시선 벡터 (점 → 센서) ────────────────────────────────────────────
    dxv, dyv, dzv = xs - x, ys - y, h - z
    r = np.sqrt(dxv ** 2 + dyv ** 2 + dzv ** 2)
    var = sigma0 ** 2 + sigma_r ** 2 * (r / d_ref) ** 2
    diag = dict(r_med=float(np.median(r)), r_min=float(r.min()), r_max=float(r.max()))
    if model == 'range':
        diag.update(cos_med=None, frac_cos_clamped=0.0)
        return np.sqrt(var), diag
    # ── 입사각: 해면 법선 n̂ 과 시선 d̂ 사이 ───────────────────────────────────
    nn = np.sqrt(1.0 + Zx ** 2 + Zy ** 2)
    nx, ny, nz = -Zx / nn, -Zy / nn, 1.0 / nn
    ct = (nx * dxv + ny * dyv + nz * dzv) / r
    clamped = float(np.mean(ct < cos_min))
    ct = np.clip(ct, cos_min, 1.0)
    diag.update(cos_med=float(np.median(ct)), frac_cos_clamped=clamped)
    return np.sqrt(var / ct), diag


def scene_noise(record, spec, win=3, W=40.0, m_scan=12, m_cap=600, lead_max=14.0,
                sigma0=0.01, sigma_r=0.02, d_ref=125.0, cos_min=0.05,
                mm_target=0.02):
    """`scene_station` 과 같은 dict. **잡음 모형만** `spec` 으로 갈아 끼운다."""
    from realdata_analysis import directional_spread
    from d3_sa_multirecord import MRecord

    cfg = RECORDS_12[record]
    depth = cfg['depth']
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))                 # scene_station.py:83 과 동일
    lm = lead_max
    n_lead = int(round(lm / rec.dt))
    if rec.T_all < nfw + n_lead + 4:                # scene_station.py:87-89
        n_lead = max(2, (rec.T_all - nfw - 4))
        lm = n_lead * rec.dt
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    if hi <= lo:
        raise RuntimeError(f'{record}: 창을 놓을 수 없다 (T={rec.T_all})')
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)          # ★원본과 같은 시드 규약
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    _, _, _, kp = directional_spread(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)

    # ── 표본 + 표준정규 (팔과 무관하게 동일한 소비) ──────────────────────────
    P_all, z_true, u_all, Zx_all, Zy_all = [], [], [], [], []
    need_slope = spec['model'] == 'range_angle'
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        m = np.isfinite(Zmem[j])                    # ★부채꼴 없음: 유효 전체
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        pick = rng.choice(idx, min(m_scan, idx.size), replace=False)
        u = rng.normal(0, 1.0, len(pick))           # ★소비 동수 (원본의 rng.normal 자리)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_true.append(Zmem[j][iy, ix].astype(float))
        u_all.append(u)
        if need_slope:
            Zx, Zy = local_slope(Zmem[j], rec.dx, rec.dy)
            Zx_all.append(Zx[iy, ix])
            Zy_all.append(Zy[iy, ix])
        else:
            Zx_all.append(np.zeros(len(pick)))
            Zy_all.append(np.zeros(len(pick)))
    P_all = np.vstack(P_all)
    z_true = np.concatenate(z_true)
    u_all = np.concatenate(u_all)
    Zx_all, Zy_all = np.concatenate(Zx_all), np.concatenate(Zy_all)

    m = P_all[:, 2] >= t0 - W                       # scene_station.py:116
    P, z_true, u_all = P_all[m], z_true[m], u_all[m]
    Zx_all, Zy_all = Zx_all[m], Zy_all[m]
    if len(z_true) > m_cap:                         # scene_station.py:118-120
        ii = np.random.default_rng(0).choice(len(z_true), m_cap, replace=False)
        P, z_true, u_all = P[ii], z_true[ii], u_all[ii]
        Zx_all, Zy_all = Zx_all[ii], Zy_all[ii]

    # ── ★σ_j 는 **추정기에 실제로 들어가는 점**에서 계산한다 ─────────────────
    sig, diag = sigma_field(P[:, 0], P[:, 1], z_true, Zx_all, Zy_all, spec,
                            sigma0, sigma_r, d_ref, cos_min)
    mm_scale = 1.0
    if spec.get('mm'):
        # 평균 분산을 배포값(mm_target²)에 맞춘다 — **형태만 남기고 수준을 통제**한다.
        mv = float(np.mean(sig ** 2))
        mm_scale = float(np.sqrt(mm_target ** 2 / mv)) if mv > 0 else 1.0
        sig = sig * mm_scale
    z = z_true + sig * u_all

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
                # ── 개입 계수기 / 채점 창 (행에 그대로 실린다) ──────────────
                sigma_stats=dict(min=float(sig.min()), med=float(np.median(sig)),
                                 max=float(sig.max()),
                                 rms=float(np.sqrt(np.mean(sig ** 2))),
                                 mm_scale=mm_scale, **diag),
                window=dict(i0=i0, nfw=nfw, n_lead=n_lead, T_all=int(rec.T_all),
                            dt=float(rec.dt), lm=float(lm), t0=t0, sq=int(sq),
                            score_frames=[i0, i0 + n_lead]),
                n_obs_used=int(len(z)), noise_arm=spec.get('name'),
                label=f'B-17 {record} win{win} noise={spec.get("name")}')
