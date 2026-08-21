"""Rx-5 §3 — `RX5_LR_W_MATCHED` 를 확정하기 위한 측정 (결과 채점 **전**).

Rx-4 의 채널 대조가 교락된 이유: Adam 은 그래디언트 크기와 거의 무관하게 **≈lr 크기의
절대 스텝**을 밟는데, 진폭 `w` [m] 와 위상 `phi` [rad] 의 자연 스케일이 100배 다르다.

정합 규칙 (사전등록 §3, 결과 보기 전 동결):
    위상 상대 스텝 = 0.005 / (2*pi) = 7.9577e-4
    LR_W_MATCHED   = 7.9577e-4 * median|w| at the START of tracking

`median|w|` 는 **모델 파라미터**를 잰다(GT 진폭이 아니다 — GT 를 lr 에 넣으면 누수다).
출하 작동점(`init_method='auto'`, `analytic_ensemble_k=3`)으로 Phase 1 만 돌리고,
추적이 시작되기 직전의 `|w|` 를 읽는다.  추적은 돌리지 않는다.

한 번 재서 **상수로 동결**한다. 런마다 바꾸면 lr 이 데이터 의존이 되어 셀이 묶음이 된다.
"""
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

DATA = os.path.join(REPO, 'data_indep2_2026-08-07')
EPISODES = [f'episode_BF{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in (4, 5) for nc in (15, 30) for s in range(4)]
SEEDS = [1527, 1528, 1529]
PHASE_REL_STEP = 0.005 / (2 * np.pi)


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import (SequentialUpdater,
                                                       MedianEnsembleUpdater)

    rows = []
    for ep in EPISODES:
        for sd in SEEDS:
            loader = EpisodeDataLoader(os.path.join(DATA, ep), device='cuda')
            frames = loader.get_init_window(tp_factor=2.0)
            # shipped operating point == wavesplat/run.py DEFAULT_CONFIG
            cfg = dict(init_method='auto', analytic_ensemble_k=3,
                       use_dispersion=True, use_confidence=True,
                       adaptive_n=True, tracking_window=3,
                       prune_threshold=0.10, max_n=30,
                       n_gaussians=[3, 8, 15],
                       loss_thresholds=[0.1, 0.05, None],
                       max_iters_per_stage=[200, 200, 500],
                       device='cuda', seed=sd, Hs_hint=float(loader.Hs_true))
            upd = MedianEnsembleUpdater(cfg)
            upd.initialize(frames)
            w = np.abs(upd.model.w.detach().cpu().numpy()).ravel()
            rows.append(dict(episode=ep, seed=sd, n=int(w.size),
                             median_abs_w=float(np.median(w)),
                             mean_abs_w=float(w.mean()),
                             min_abs_w=float(w.min()), max_abs_w=float(w.max())))
            print(f"  {ep[8:-14]:26s} s{sd} N={w.size:3d} "
                  f"median|w|={rows[-1]['median_abs_w']:.5f} "
                  f"[{w.min():.5f}, {w.max():.5f}]", flush=True)

    med = st.median(r['median_abs_w'] for r in rows)
    lr = PHASE_REL_STEP * med
    print('\n' + '=' * 66)
    print(f'  블록 중앙 median|w| (추적 시작 시점) = {med:.6f} m')
    print(f'  위상 상대 스텝  = 0.005 / 2pi        = {PHASE_REL_STEP:.6e}')
    print(f'  ★RX5_LR_W_MATCHED = {PHASE_REL_STEP:.6e} * {med:.6f} = {lr:.6e}')
    print(f'  (참고) 출하 lr 0.005 는 이 진폭의 {100*0.005/med:.1f} %')
    json.dump(dict(rows=rows, median_abs_w=med, phase_rel_step=PHASE_REL_STEP,
                   RX5_LR_W_MATCHED=lr),
              open(os.path.join(HERE, 'wscale.json'), 'w'), indent=1)
    print(f'\n  -> wscale.json 기록 완료')


if __name__ == '__main__':
    main()
