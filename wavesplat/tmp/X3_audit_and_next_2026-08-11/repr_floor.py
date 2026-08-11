"""표현 바닥 — **런 0회**. CON B-1 을 증명하거나 반증한다.

## 질문
한방풀이(bank)가 성분수를 15배 줄여도 1.7배밖에 안 좋아진다. 이것이
(a) 추정이 어려워서인가, 아니면 (b) **그 격자 표현으로는 정답을 애초에 담을 수 없어서**인가.

## 방법
정답 해면을 bank 의 **기저(고정 파수 격자)에 최소제곱 투영**한다. 관측도 잡음도 없는
**완벽한 정보** 조건이다. 그래도 남는 오차 = **표현 바닥**.
바닥이 관측된 nRMSE 를 설명하면 (b) 가 참이고, 이 축은 종결된다.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from wavesplat.data_loader import EpisodeDataLoader
import wavesplat.optimizer.bank_mmse as BM

EPS = {
    2:  'data_s9_sparse_2026-08-11/episode_BF4_2comp_seed48_uniform_ra_rf_ds.h5',
    5:  'data_s9_sparse_2026-08-11/episode_BF4_5comp_seed48_uniform_ra_rf_ds.h5',
    15: 'data_s9_crowded_2026-08-11/episode_BF4_15comp_seed48_uniform_ra_rf_ds.h5',
    30: 'data_s9_crowded_2026-08-11/episode_BF4_30comp_seed48_uniform_ra_rf_ds.h5',
}
BAND = (0.35, 2.5)          # bank_mmse 기본 대역
bank = json.load(open('tmp/X1_bank_on_synth_2026-08-11/result.json'))

print('★표현 바닥 — 완벽한 정보(관측·잡음 없음)에서도 bank 격자가 못 담는 몫')
print(f'{"성분":>4} {"표현바닥":>9} {"X1 실측":>9} {"바닥/실측":>9}')
for nc, p in EPS.items():
    L = EpisodeDataLoader(p)
    fr = L.get_tracking_frames()[len(L.get_tracking_frames()) // 2]
    gt = fr.gt_field.numpy()
    gx, gy = np.asarray(L.gt_grid_x), np.asarray(L.gt_grid_y)
    GX, GY = (np.meshgrid(gx, gy) if gx.ndim == 1 else (gx, gy))
    Q = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, fr.t)])

    # ★수정: kp 를 내가 고르지 않고 **bank 가 실제로 쓴 격자를 그대로 가져온다**.
    # (첫 시도는 GT 중앙 kp 를 썼고, 2성분에서 바닥 > 실측 이라는 **불가능한 값**이 나왔다
    #  = 내 격자가 bank 의 격자가 아니었다는 증거. 그래서 실제 적합에서 뽑는다.)
    init = L.get_init_window()
    Pobs = np.concatenate([np.column_stack([f.x_obs.numpy(), f.y_obs.numpy(),
                                            np.full(len(f.x_obs), f.t)]) for f in init])
    zobs = np.concatenate([f.z_obs.numpy() for f in init])
    if len(zobs) > 600:
        ii = np.random.default_rng(0).choice(len(zobs), 600, replace=False)
        Pobs, zobs = Pobs[ii], zobs[ii]
    fit = BM.fit_bank_mmse(Pobs, zobs, float(L.depth),
                           0.05 * 4 * float(np.std(zobs)), kp_hint=None, seed=42)
    kxb, kyb, om = fit.kxb, fit.kyb, fit.om
    ph = Q[:, 0:1]*kxb[None] + Q[:, 1:2]*kyb[None] - Q[:, 2:3]*om[None]
    Phi = np.concatenate([np.cos(ph), np.sin(ph)], 1)       # bank 기저 (2B)
    w, *_ = np.linalg.lstsq(Phi, gt.ravel(), rcond=None)    # **완벽 정보** 최소제곱
    resid = gt.ravel() - Phi @ w
    floor = float(np.sqrt(np.mean(resid**2)) / L.Hs_true)
    obs = bank.get(p, {}).get('bank_nRMSE', float('nan'))
    print(f'{nc:4d} {floor:9.4f} {obs:9.4f} {floor/obs*100 if obs==obs else float("nan"):8.1f}%  (B={len(kxb)})')
