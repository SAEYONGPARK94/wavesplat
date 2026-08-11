"""★결정 실험: 오늘 배포한 `lambda_l1` 변경이 **실측 경로에 도달하는가?**

실측(GT-free) 입력은 `real_domain_gate` 가 잡아 **bank_mmse** 로 라우팅한다
(`sequential_update.py:455-475`), 그 경로는 가우시안 성분을 만들지 않는다(`run.py:434`).
⟹ `lambda_l1`(가우시안 목적함수의 L1 항)이 실측에서 **아무 일도 안 할 가능성**이 있다.
⛔코드를 읽고 단정하지 않는다 — **같은 실측 창을 두 λ 로 돌려 예측면을 비교한다.**
(S1 전례: "출하 팔은 beamform 을 0회 호출했다" = 개입 지점 부재)
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'))
os.chdir(ROOT)

import wavesplat.run as WR
import make_gif_bank_deployed as M
import d3_compare_gifs as cg

REC, WIN, SEED = 'AA15a', 5, 42
_BASE = dict(WR.DEFAULT_CONFIG)


def field_for(sc, lam):
    WR.DEFAULT_CONFIG.clear(); WR.DEFAULT_CONFIG.update(_BASE)
    WR.DEFAULT_CONFIG['lambda_l1'] = lam
    upd, t0, el, cfg = M.estimate(sc, seed=SEED)
    gx, gy = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    GX, GY = np.meshgrid(gx, gy)
    import torch
    Z = upd.predict_field(torch.tensor(GX, dtype=torch.float32),
                          torch.tensor(GY, dtype=torch.float32), t0).cpu().numpy()
    route = getattr(upd, '_auto_selected', None)
    gate = getattr(upd, '_auto_real_gate', None)
    return Z, route, gate, cfg['lambda_l1'], el


if __name__ == '__main__':
    print(f'# 실측 창 로드: {REC} win={WIN}', flush=True)
    sc = cg.scene_real(REC, win=WIN)
    out = {}
    for lam in (1e-3, 1e-4):
        Z, route, gate, used, el = field_for(sc, lam)
        out[lam] = Z
        print(f'  lambda_l1={lam:.0e}  라우트={route}  실측게이트={gate}  '
              f'cfg 반영={used:.0e}  init {el:.1f}s  '
              f'field std={np.nanstd(Z):.5f}', flush=True)
    A, B = out[1e-3], out[1e-4]
    same = np.allclose(A, B, rtol=0, atol=0, equal_nan=True)
    print()
    print(f'★두 예측면이 **비트 동일**한가: {"예 — lambda_l1 은 실측에 도달하지 않는다" if same else "아니오 — 영향 있음"}')
    if not same:
        d = np.abs(A - B)
        print(f'  최대 절대차 {np.nanmax(d):.3e} · 평균 {np.nanmean(d):.3e} · '
              f'상대 {np.nanmean(d)/np.nanstd(A)*100:.4f}% of field std')
