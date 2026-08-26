"""B-17 — **센서 원점을 자료에서 도출한다**. 추정기 런 0회 (순수 기하).

## 왜 이 파일이 따로 있는가
`B-17` 은 잡음을 `sigma(r, theta)` 로 바꾼다. 그러면 **`r` 을 어디서부터 재는지**가
결과를 지배한다. ⛔**임의로 고르면 그것이 곧 손복사 상수다.**
저장소 전례: `ZERO_REF=1.0` 하나가 여섯 문서의 계수를 동시에 틀리게 했다.
⟹ ★**센서 위치는 선언하지 말고 자료에서 도출한다.** 이 파일이 그 유일한 경로다.

## 무엇을 도출할 수 있고 무엇을 못 하는가 (결과 전에 적는다)
자료(`Surfaces_*.nc`)에는 **카메라 위치 변수가 없다** — 전역 속성은
`Cameras distance [m]`(기선 2.03 m) · `Acquisition rate` · `Grid discretization` 뿐이다
(`h5py` 로 전수 확인). 따라서:

- ✅**수평 위치 (x_s, y_s)** 는 도출된다. WASS 재구성 격자의 `mask_Z` 유효영역이
  **카메라 시야가 해면에 투영된 쐐기(wedge)** 이고, 그 **꼭짓점**이 카메라의 연직 아래점이다.
  AA/YS 처럼 유효영역이 쐐기인 레코드에서 `halfspan(x) = a·|x - x0|` 를 최소자승으로 맞춰
  `x0` 를 낸다. BS/LJ 는 유효율 1.0 (직사각 크롭) 이라 쐐기가 보이지 않으므로
  **격자 중심**을 쓰고 그 사실을 함께 낸다.
- ⛔**높이 h_s 는 도출되지 않는다.** 자료에도 저장소에도 없다.
  ⟹ `gates_b17.SENSOR_H` 에 **결과 전에 선언**하고, 민감도 팔(`RANGEANG-H30`)로 노출한다.

## 산출
`sensor_origin.json` — 레코드별 `{x_s, y_s, method, halfangle_deg, fit_r2, n_cols}`.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, B2, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(HERE, 'sensor_origin.json')

# 쐐기 적합을 신뢰하는 최소 조건 (결과 전 고정)
MIN_VALID_FRAC_FOR_WEDGE = 0.05   # 유효율이 이보다 낮으면 자료가 없다
MAX_VALID_FRAC_FOR_WEDGE = 0.98   # 유효율이 이보다 높으면 직사각 크롭 = 쐐기 안 보임
MIN_R2 = 0.80                     # 선형 적합이 이보다 나쁘면 쐐기로 인정하지 않는다


def _wedge_apex(gx, gy, valid):
    """`halfspan(x) = a·|x - x0|` 를 유효영역에서 맞춰 꼭짓점 x0 을 낸다.

    `valid` 는 `(len(gy), len(gx))`. 열마다 유효한 y 의 반폭과 중심을 구하고,
    **꼭짓점에 가까운 단조 증가 구간만** 쓴다(먼 쪽은 스테레오 사거리 한계로 다시 줄어든다).
    """
    w, yc, xs = [], [], []
    for ix in range(len(gx)):
        col = valid[:, ix]
        if col.sum() < 3:
            continue
        yy = gy[col]
        w.append(0.5 * float(yy.max() - yy.min()))
        yc.append(0.5 * float(yy.max() + yy.min()))
        xs.append(float(gx[ix]))
    if len(xs) < 8:
        return None
    xs, w, yc = np.array(xs), np.array(w), np.array(yc)
    # 최대 반폭 지점까지가 "꼭짓점에서 멀어지는" 단조 구간이다.
    imax = int(np.argmax(w))
    # 꼭짓점은 |x| 가 작은 쪽이므로, imax 를 기준으로 |x| 가 더 작은 쪽 절반을 쓴다.
    if abs(xs[imax]) > abs(xs[0]):
        sel = slice(0, imax + 1) if xs[imax] > xs[0] else slice(imax, len(xs))
    else:
        sel = slice(imax, len(xs)) if xs[-1] > xs[imax] else slice(0, imax + 1)
    X, W, YC = xs[sel], w[sel], yc[sel]
    if len(X) < 8:
        return None
    A = np.column_stack([X, np.ones_like(X)])
    (p, q), *_ = np.linalg.lstsq(A, W, rcond=None)
    if abs(p) < 1e-9:
        return None
    pred = A @ np.array([p, q])
    ss_res = float(np.sum((W - pred) ** 2))
    ss_tot = float(np.sum((W - W.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    x0 = float(-q / p)
    (m, c), *_ = np.linalg.lstsq(np.column_stack([X, np.ones_like(X)]), YC, rcond=None)
    y0 = float(m * x0 + c)
    return dict(x_s=x0, y_s=y0, halfangle_deg=float(np.degrees(np.arctan(abs(p)))),
                fit_r2=float(r2), n_cols=int(len(X)))


def derive(record):
    from scene_station import RECORDS_12
    from d3_sa_multirecord import MRecord
    cfg = RECORDS_12[record]
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    vf = float(rec.valid.mean())
    gxc = 0.5 * float(rec.gx.min() + rec.gx.max())
    gyc = 0.5 * float(rec.gy.min() + rec.gy.max())
    out = dict(record=record, valid_frac=vf,
               grid_x=[float(rec.gx.min()), float(rec.gx.max())],
               grid_y=[float(rec.gy.min()), float(rec.gy.max())],
               grid_center=[gxc, gyc], dx=float(rec.dx), dy=float(rec.dy),
               patch_halfdiag=float(np.hypot(rec.gx.max() - rec.gx.min(),
                                             rec.gy.max() - rec.gy.min()) / 2))
    # ★두 증거를 **항상 함께** 낸다 (하나가 실패해도 조용히 다른 값으로 갈아타지 않는다).
    fit = None
    if MIN_VALID_FRAC_FOR_WEDGE < vf < MAX_VALID_FRAC_FOR_WEDGE:
        fit = _wedge_apex(rec.gx, rec.gy, rec.valid)
    out['wedge'] = fit
    if fit is not None and fit['fit_r2'] >= MIN_R2:
        out['method'] = 'wedge_apex'
        out['x_s'], out['y_s'] = fit['x_s'], fit['y_s']
    elif vf >= MAX_VALID_FRAC_FOR_WEDGE:
        # 유효율 ~1 = 직사각 크롭. 쐐기가 보이지 않으므로 격자 중심이 유일한 증거다.
        out['method'] = 'patch_center'
        out['x_s'], out['y_s'] = gxc, gyc
    else:
        # ⛔둘 다 약하다 — **아무 값도 도출됐다고 주장하지 않는다.**
        out['method'] = 'inconclusive'
        out['x_s'], out['y_s'] = None, None
    return out


def main():
    from scene_station import RECORDS_12
    res = {}
    for r in RECORDS_12:
        d = derive(r)
        res[r] = d
        wg = d['wedge']
        xs = '  (none)' if d['x_s'] is None else f"({d['x_s']:7.2f},{d['y_s']:6.2f})"
        print(f"{r:7s} valid={d['valid_frac']:.3f} method={d['method']:13s} origin={xs} "
              f"|origin|={'  n/a' if d['x_s'] is None else format(np.hypot(d['x_s'], d['y_s']), '6.2f')} "
              f"patch_halfdiag={d['patch_halfdiag']:7.2f} "
              f"wedge_r2={'n/a' if not wg else round(wg['fit_r2'], 3)} "
              f"halfang={'n/a' if not wg else round(wg['halfangle_deg'], 1)}", flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    off = {k: (None if d['x_s'] is None else float(np.hypot(d['x_s'], d['y_s'])))
           for k, d in res.items()}
    got = [v for v in off.values() if v is not None]
    print(f"\n★ WASS 세계좌표 원점 (0,0) 로부터의 도출 원점 거리: "
          f"n={len(got)}/12 · 중앙 {np.median(got):.2f} m · 최대 {max(got):.2f} m "
          f"(패치 반대각선 중앙 {np.median([d['patch_halfdiag'] for d in res.values()]):.1f} m)")
    print(f'★ sensor_origin.json 기록 ({len(res)} 레코드)')


if __name__ == '__main__':
    main()
