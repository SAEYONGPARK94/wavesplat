#!/usr/bin/env python3
"""B-3 부록 — **어느 개구부터 레짐 질문이 답 가능한가** (합성만, 실측 미사용).

PRESPEC §2 가 산출물로 지정한 "계단": *"계단 자체가 산출물이다 — 어느 개구부터 레짐
질문이 답 가능한가."*  T1/T2/T3 세 점만으로는 곡선이 안 그려져서 개구를 훑는다.

⛔이것은 **탐색적**이다.  실측 배치 규칙(PRESPEC §4)은 T1/T2/T3 에 동결돼 있고
여기서 나온 개구로 실측을 다시 배치하지 **않는다**(사후 문턱 이동 금지).
합성만 쓰므로 실측을 엿보지 않는다.
"""
import os
import sys
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b3_regime import synth_episode, regime, auc, TIERS, REPO   # noqa: E402

APERTURES = [45.0, 60.0, 90.0, 120.0, 150.0, 180.0, 195.0]      # dx=3.0 고정
DX = 3.0

if __name__ == '__main__':
    eps = sorted(glob.glob(os.path.join(REPO, 'data_s9_sparse_2026-08-11', '*.h5'))
                 + glob.glob(os.path.join(REPO, 'data_s9_crowded_2026-08-11', '*.h5')))
    out_path = os.path.join(HERE, 'aperture_curve.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    print(f'# 합성 {len(eps)} 에피소드 x {len(APERTURES)} 개구 (실측 미사용)')
    for L in APERTURES:
        key = f'{L:.0f}'
        if key not in res:
            TIERS['_SWEEP'] = (L, L, DX)
            rows = []
            for p in eps:
                a = synth_episode(p, '_SWEEP')
                if a:
                    a['regime'] = regime(p)
                    rows.append(a)
            res[key] = rows
            json.dump(res, open(out_path, 'w'), indent=1)
        rows = res[key]
        sp = [r['PR'] for r in rows if r['regime'] == 'sparse']
        cr = [r['PR'] for r in rows if r['regime'] == 'crowded']
        A = auc(sp, cr)
        v = '판정가능' if A >= 0.90 else ('약함' if A >= 0.70 else '⛔판정불가')
        print(f'  개구 {L:5.0f} m  격자 {rows[0]["grid"]:>7s}  대역셀 {rows[0]["n_band"]:4d}  '
              f'희소 med {np.median(sp):6.2f}  혼잡 med {np.median(cr):6.2f}  '
              f'**AUC={A:.3f}**  {v}  (n={len(sp)}/{len(cr)})', flush=True)
    print('\n→ aperture_curve.json')
