#!/usr/bin/env python3
"""★배포 체크포인트의 **학습 지지집합**을 재도출한다 — `PREREG_CHECKLIST.md B14` 의 계측기.

⛔**판정 도구가 아니다.** 어떤 배치의 verdict 도 바꾸지 않고, 동결 파일도 읽기만 한다.
⛔`run_all.sh` 에 등록하지 않는다(에피소드 14개 적재 + 장면 표집으로 수 분 걸린다).

무엇을 재는가
-------------
배포 사전은 `results/direction3/beamform_cond_diffusion_mixed_v1.pt`
(`wavesplat/optimizer/learned_init.py:61-63`)이고, 그것을 만든
`tmp/mixed_retrain_2026-07-26/train_mixed.py` 는 헤더에 축자로 적는다 —
*"The ONLY change is where the training batch comes from:
  `cd.make_batch` -> `gen_mixed.sample_scene_ok` + `gen_mixed.batch_from_scenes`"*.
따라서 **배포 사전이 실제로 본 관측 기하는 `gen_mixed.GeometryBank`(donor 14개)** 이지
`scripts/d40_learned_acquisition.pool_xy()`(교체된 `cd.make_batch` 의 풀)가 아니다.

⛔`scale_rules_t14.py:12` 의 `L_TRAIN = 220.55` 는 **후자**에서 나왔고,
`analyze_t14.py:123-130` 의 `G-LTRAIN` 이 **그것을** 재도출해 통과시킨다.
게이트는 정직하게 통과했다 — 핀한 소스가 배포 네트의 것이 아닐 뿐이다.
⟹ **상수는 그대로 둔다. 이 스크립트는 그 상수의 「범위」를 문서화한다.**

정직한 한계
-----------
- 학습 스트림을 **비트 재현하지 않는다**(30,000 step x batch 32 = 약 96만 장면).
  같은 생성 과정에서 다시 표집할 뿐이므로 결과는 **분포의 추정**이다.
- `--nscene` 이 크면 오래 걸린다. 개구(`L_obs`)만 필요하면 `--no-ruler` 가 훨씬 싸다.
- 빔포머 자(`--ruler`)는 `bank_mmse._beamform_top` 의 **numpy 이식**이다(torch 금지).
  실측 72셀에서 동결 `kp2_hat`(`T17`)과 상대편차 **최대 3.95e-06** 으로 일치했다
  (관측은 `obs_hash` 72/72 비트 동일). 차이는 `linspace` 의 float32 ulp 다.

사용
----
    cd /home/vexit/wavesplat/wavesplat
    python3 tmp/_audit/derive_train_support.py --nscene 1500
    python3 tmp/_audit/derive_train_support.py --nscene 400 --no-ruler
"""
import argparse
import os
import sys
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
G_ACC = 9.81                 # bank_mmse.py:37
N1, ZOOM, FLOOR = 120, 3.5, 1e-3      # bank_mmse.py:158-165 / scale_rules_t16.py:19-21
DEPTH = 50.0                 # gen_mixed.py:57

# ── 동결 상수(대조용, 손복사 아님: 임포트한다) ────────────────────────────────
def frozen():
    sys.path.insert(0, os.path.join(ROOT, 'tmp/T14_AUTOSCALE_2026-09-04'))
    import scale_rules_t14 as SR
    return float(SR.L_TRAIN), float(SR.K_TRAIN)


# ── `bank_mmse._beamform_top` 의 numpy 이식 (torch 금지) ─────────────────────
def _beamform_top_np(P, r, depth, klim, n=N1):
    x = np.asarray(P[:, 0], np.float32); y = np.asarray(P[:, 1], np.float32)
    t = np.asarray(P[:, 2], np.float32); rt = np.asarray(r, np.float32)
    ax = np.linspace(-klim, klim, n, dtype=np.float32)
    KX, KY = np.meshgrid(ax, ax, indexing='xy')
    kx, ky = KX.reshape(-1), KY.reshape(-1)
    k = np.sqrt(kx ** 2 + ky ** 2 + np.float32(1e-9))
    om = np.sqrt(np.float32(G_ACC) * k * np.tanh(k * np.float32(depth)) + np.float32(1e-9))
    pw = np.empty(kx.shape[0], np.float32)
    for i in range(0, kx.shape[0], 2048):
        ps = (kx[i:i + 2048, None] * x[None] + ky[i:i + 2048, None] * y[None]
              - om[i:i + 2048, None] * t[None])
        pw[i:i + 2048] = ((rt[None] * np.cos(ps)).sum(1) ** 2
                          + (rt[None] * np.sin(ps)).sum(1) ** 2)
    j = int(np.argmax(pw))
    return float(kx[j]), float(ky[j])


def kp_hat_abs(x, y, t, z, depth):
    """`scale_rules_t14.kp_hat` 의 numpy 쌍둥이 — **격자가 고정**(klim=2.0)이라
    개구가 다른 두 코퍼스에 같은 자를 댈 수 있다."""
    P = np.column_stack([x, y, t]).astype(np.float32)
    r = np.asarray(z, np.float32)
    bx, by = _beamform_top_np(P, r, depth, klim=2.0)
    kp = max(float(np.hypot(bx, by)), FLOOR)
    bx, by = _beamform_top_np(P, r, depth, klim=ZOOM * kp)
    return max(float(np.hypot(bx, by)), FLOOR)


def kp_from_Tp(Tp, depth=DEPTH):
    """`analyze_t14.py:133-139`(`G-KTRAIN`)과 같은 절차 — 분산관계 brentq."""
    from scipy.optimize import brentq
    w = 2.0 * np.pi / float(Tp)
    return brentq(lambda k: G_ACC * k * np.tanh(k * depth) - w * w, 1e-6, 10.0)


def pct(a):
    return np.percentile(np.asarray(a, float), [0, 5, 25, 50, 75, 95, 100])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nscene', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=20260726)   # train_mixed.py --seed 기본값
    ap.add_argument('--no-ruler', action='store_true', help='빔포머 자를 생략(개구만)')
    a = ap.parse_args()

    os.chdir(ROOT)
    for p in (ROOT, os.path.join(ROOT, 'scripts'),
              os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26')):
        sys.path.insert(0, p)
    warnings.filterwarnings('ignore')
    import gen_mixed as gm

    L_TRAIN, K_TRAIN = frozen()
    assert 'pool_xy' not in open(
        os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26/gen_mixed.py'), encoding='utf-8').read(), \
        '★전제 파손: gen_mixed 가 pool_xy 를 쓰게 됐다 — 이 스크립트의 서술을 재검토하라'

    bank = gm.GeometryBank()
    print(f'donor {len(bank.donors)} 개 (gen_mixed.DONOR_FILES)')
    for d in bank.donors:
        print(f"   {d['name']:44s} L_full={max(float(np.ptp(d['x'])), float(np.ptp(d['y']))):8.2f} m"
              f"  N={len(d['x']):7d}  t_max={d['t_max']:.1f}s")

    rng = np.random.default_rng(a.seed)
    L, kn, kb = [], [], []
    for i in range(a.nscene):
        s = gm.sample_scene_ok(rng, bank)
        x, y, t = s['P'][:, 0], s['P'][:, 1], s['P'][:, 2]
        L.append(max(float(np.ptp(x)), float(np.ptp(y))))
        kn.append(kp_from_Tp(s['Tp']))
        if not a.no_ruler:
            kb.append(kp_hat_abs(x, y, t, s['z'], DEPTH))
        if (i + 1) % 250 == 0:
            print(f'   {i+1}/{a.nscene}', flush=True)
    L, kn = np.array(L), np.array(kn)

    hdr = ' '.join(f'{h:>9s}' for h in ('min', 'p5', 'p25', 'med', 'p75', 'p95', 'max'))
    print(f'\n== 배포 체크포인트의 학습 지지집합 (n={a.nscene}) ==\n{"":30s}{hdr}')
    print(f'{"L_obs = max(ptp x, ptp y) [m]":30s}' + ' '.join(f'{v:9.4g}' for v in pct(L)))
    print(f'{"lambda_p  nominal(Tp) [m]":30s}' + ' '.join(f'{v:9.4g}' for v in pct(2 * np.pi / kn)))
    print(f'{"L/lambda  nominal":30s}' + ' '.join(f'{v:9.4g}' for v in pct(L * kn / (2 * np.pi))))
    if kb:
        kb = np.array(kb)
        print(f'{"lambda_p  beamformer [m]":30s}' + ' '.join(f'{v:9.4g}' for v in pct(2 * np.pi / kb)))
        print(f'{"L/lambda  beamformer":30s}' + ' '.join(f'{v:9.4g}' for v in pct(L * kb / (2 * np.pi))))

    print(f'\n동결 L_TRAIN = {L_TRAIN} m  ⟹ 이 분포의 p{100*float((L < L_TRAIN).mean()):.0f}'
          f' · 학습 장면의 {100*float((L > L_TRAIN).mean()):.1f}% 가 그보다 크다'
          f' · 중앙 대비 {L_TRAIN/np.median(L):.3f}배')
    print(f'동결 K_TRAIN = {K_TRAIN} rad/m  vs 배포 draw 의 기하평균 k_p(nominal) '
          f'= {np.exp(np.mean(np.log(kn))):.5f}'
          + ('' if not len(kb) else f' · beamformer = {np.exp(np.mean(np.log(kb))):.5f}'))
    print('\n⛔이 출력은 어떤 배치의 판정도 바꾸지 않는다 — 상수의 **범위**를 적을 뿐이다'
          ' (`PREREG_CHECKLIST.md B14`).')


if __name__ == '__main__':
    main()
