"""X1 (탐색적) — **운용 추정기(bank-MMSE)를 시뮬레이터 데이터에서 재본다.**

## 왜
실제 운용은 항상 bank-MMSE 로 간다(게이트가 'GT 없음'에 반응). 그런데 그 추정기는
**정답이 있는 데이터에서 대규모로 채점된 적이 없다** — 실측 5창이 전부다.
시뮬레이터엔 정답이 있으므로 두 풀이법을 **같은 조건에서** 비교할 수 있다.

⚠**탐색적**: 사전등록 없음. 판정 문구를 만들지 않는다. 방향과 크기만 본다.
★표본 오염 없음 — 이 96 에피소드는 **bank-MMSE 로 채점된 적이 한 번도 없다**(새 대조).
정규화는 합성 규약 그대로 **RMSE/Hs** (`metrics.nrmse`) — S9 수치와 직접 비교 가능.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.bank_mmse import fit_bank_mmse
from wavesplat.evaluation.metrics import nrmse

# ⚠렌더가 비싸다: 에피소드당 160 프레임 × 4800 격자 × 3 멤버 = 230만 질의점.
# 탐색적 목적이므로 프레임을 **균일 서브샘플**한다. 프레임 평균의 불편추정이고,
# S9 와 같은 "에피소드당 프레임 평균" 규약을 유지한다. (내 최초 10분 추정은 틀렸다)
FRAME_STRIDE = 8                  # 160 -> 20 프레임
SEEDS = [42, 1042, 2042]          # 배포 규약: median-of-3 필드 앙상블
OUT = 'tmp/X1_bank_on_synth_2026-08-11/result.json'


def episodes():
    for rg, d in (('sparse', 'data_s9_sparse_2026-08-11'),
                  ('crowded', 'data_s9_crowded_2026-08-11')):
        for f in sorted(os.listdir(d)):
            if f.endswith('.h5'):
                yield rg, os.path.join(d, f)


def run_one(path):
    L = EpisodeDataLoader(path)
    init = L.get_init_window()
    P = np.concatenate([np.column_stack([f.x_obs.numpy(), f.y_obs.numpy(),
                                         np.full(len(f.x_obs), f.t)]) for f in init])
    z = np.concatenate([f.z_obs.numpy() for f in init])
    if len(z) > 600:                                   # 배포 규약과 동일한 캡
        ii = np.random.default_rng(0).choice(len(z), 600, replace=False)
        P, z = P[ii], z[ii]
    sigma0 = 0.05 * (4.0 * float(np.std(z)))           # 관측-전용 파라미터화
    fits = [fit_bank_mmse(P, z, float(L.depth), sigma0, kp_hint=None, seed=s) for s in SEEDS]

    gx, gy = np.asarray(L.gt_grid_x), np.asarray(L.gt_grid_y)
    GX, GY = (np.meshgrid(gx, gy) if gx.ndim == 1 else (gx, gy))
    vals = []
    for fr in L.get_tracking_frames()[::FRAME_STRIDE]:
        gt = fr.gt_field.numpy()
        Q = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, fr.t)])
        pred = np.median(np.stack([f.render(Q) for f in fits], 0), 0).reshape(gt.shape)
        vals.append(nrmse(pred, gt, L.Hs_true))
    return float(np.mean(vals)), len(vals)


if __name__ == '__main__':
    res, t0 = {}, time.time()
    eps = list(episodes())
    for i, (rg, p) in enumerate(eps, 1):
        try:
            v, nf = run_one(p)
            res[p] = dict(regime=rg, bank_nRMSE=v, n_frames=nf)
            if i % 8 == 0 or i <= 2:
                print(f'[{i}/{len(eps)}] {rg:8s} bank nRMSE={v:.4f}', flush=True)
        except Exception as e:
            res[p] = dict(regime=rg, error=f'{type(e).__name__}: {e}')
            print(f'[{i}/{len(eps)}] ⛔ {os.path.basename(p)[:44]}: {type(e).__name__}: {e}', flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    ok = [v for v in res.values() if 'error' not in v]
    print(f'\nDONE {(time.time()-t0)/60:.1f} min · 성공 {len(ok)}/{len(eps)} -> {OUT}')
