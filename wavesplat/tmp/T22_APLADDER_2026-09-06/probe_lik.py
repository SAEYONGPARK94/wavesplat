r"""`T22` **동결 전 프로브 8** — 다듬기가 **우도를 실제로 올리는가**, 아니면 평탄한 방향으로 표류하는가.

⛔**판정이 아니다.** 사전등록 미작성 · 게이트·분석기·러너 0개 · **추정기 적합 0회**.

## 왜 — 내 사전등록의 §0.3 **N-2** 를 친다

`PREREG.md §0.3 N-2` 에 내가 직접 적었다:
> *"다듬기는 「고장」이 아니라 「설계」일 수 있다. `polish_atoms` 는 관측 우도를 올린다.
>  파수가 커지는 것이 **관측에 더 잘 맞기 때문**일 수 있고, 그렇다면 끄는 것은 개악이다.
>  프로브 3–6 은 `k_true` 대비만 쟀지 **우도값을 비교하지 않았다.**"*

이 프로브가 그 구멍을 닫는다. 세 원자 집합의 **같은 관측 위 로그우도**를 잰다:

  ① 확산 원출력 (`polish_iters=0`)
  ② 다듬은 것 (`polish_iters=80` = 배포)
  ★③ **참 원자** (`gen_mixed` 장면의 `tkx`·`tky`·`tA`) — **기준선**

★③ 이 있어서 *"얼마가 많은 것인가"* 를 물을 수 있다.
- ②가 ③보다 **높으면** → 우도가 부풀린 집합을 **진짜로 선호**한다(400 관측에 과적합).
  ⟹ 다듬기는 설계대로 작동하는 것이고 ⛔*"고장" 서사는 무너진다.*
- ②가 ③보다 **낮은데** k 만 크게 움직였으면 → **평탄한 방향으로의 표류**.

⛔우도는 `d3_setdiff_posterior.loglik_atoms` 를 **임포트**한다(재구현 0).
그것이 `polish_atoms` 가 실제로 최대화하는 바로 그 목적함수다.

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T22_APLADDER_2026-09-06/probe_lik.py --nscene 15
"""
import argparse
import json
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp/T11_EXTACC_2026-09-03'),
          os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nscene', type=int, default=15)
    ap.add_argument('--seed', type=int, default=20260906)
    ap.add_argument('--dseed', type=int, default=0)
    ap.add_argument('--holdout', action='store_true',
                    help='★적합에 안 쓴 관측에서 우도를 잰다 (in-sample 과적합 판별)')
    a = ap.parse_args()

    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import gates_t11 as G
    import run_t11 as R11
    import gen_mixed as gm
    from d3_setdiff_posterior import loglik_atoms, SIGMA
    from wavesplat.optimizer import learned_init as LI

    cap = int(LI.M_OBS_CAP)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'우도 = d3_setdiff_posterior.loglik_atoms (SIGMA={SIGMA}) · '
          f'M_OBS_CAP={cap} · device={dev}')
    print('★③ 참 원자가 기준선이다 — ②가 ③보다 높으면 우도가 부풀린 집합을 진짜로 선호한다\n')

    bank = gm.GeometryBank()
    rng = np.random.default_rng(a.seed)
    print(f"{'i':>3s}{'k_true':>9s}{'k_raw':>9s}{'k_pol':>9s}"
          f"{'LL_true':>12s}{'LL_raw':>12s}{'LL_pol':>12s}"
          f"{'ΔLL(pol-raw)':>14s}{'pol>true?':>10s}")
    rows = []
    for i in range(a.nscene):
        s = gm.sample_scene_ok(rng, bank)
        P = np.asarray(s['P'], float); z = np.asarray(s['z'], float)
        tkx = np.asarray(s['tkx'], float); tky = np.asarray(s['tky'], float)
        tA = np.asarray(s['tA'], float)
        j = int(np.argmax(tA)); k_true = float(np.hypot(tkx[j], tky[j]))

        frames = R11.build_frames(P[:, 0], P[:, 1], P[:, 2] - P[:, 2].min(), z, G.GT_CONST)
        # ★우도를 재는 관측 = 제안이 실제로 본 400점(같은 규약)
        Pf = np.column_stack([P[:, 0], P[:, 1], P[:, 2]]).astype(np.float32)
        zf = z.astype(np.float32)
        if len(zf) > cap:
            idx = np.random.default_rng(0).choice(len(zf), cap, replace=False)
            Pf, zf = Pf[idx], zf[idx]
        Ps = Pf.copy(); Ps[:, 2] -= Ps[:, 2].min()
        if a.holdout:
            # ★적합은 400점(위 idx)에서, 채점은 **쓰이지 않은** 관측에서.
            used = set(map(tuple, np.round(Pf[:, :3], 9).tolist()))
            allP = np.column_stack([P[:, 0], P[:, 1], P[:, 2]]).astype(np.float32)
            keep = np.array([tuple(np.round(r, 9)) not in used for r in allP])
            hP, hz = allP[keep], z.astype(np.float32)[keep]
            if len(hz) > 400:
                hi = np.random.default_rng(7).choice(len(hz), 400, replace=False)
                hP, hz = hP[hi], hz[hi]
            hPs = hP.copy(); hPs[:, 2] -= Pf[:, 2].min()   # ★같은 시간 원점
            Pt = torch.tensor(hPs, device=dev, dtype=torch.float32)
            zt = torch.tensor(hz, device=dev, dtype=torch.float32)
            n_score = int(len(hz))
        else:
            Pt = torch.tensor(Ps, device=dev, dtype=torch.float32)
            zt = torch.tensor(zf, device=dev, dtype=torch.float32)
            n_score = int(len(zf))

        def LL(kx, ky, en):
            return float(loglik_atoms(
                torch.tensor(np.asarray(kx, np.float32), device=dev),
                torch.tensor(np.asarray(ky, np.float32), device=dev),
                torch.tensor(np.asarray(en, np.float32), device=dev),
                Pt, zt, 50.0).item())

        rkx, rky, ren = LI.propose_atoms(frames, 50.0, nsamp=4,
                                         diffusion_seed=a.dseed, polish_iters=0)
        pkx, pky, pen = LI.propose_atoms(frames, 50.0, nsamp=4,
                                         diffusion_seed=a.dseed, polish_iters=80)
        ll_t = LL(tkx, tky, tA ** 2)          # ⚠a = energy = A²
        ll_r = LL(rkx, rky, ren)
        ll_p = LL(pkx, pky, pen)
        k_r = float(np.median(np.hypot(rkx, rky)))
        k_p = float(np.median(np.hypot(pkx, pky)))
        better = 'YES' if ll_p > ll_t else 'no'
        print(f'{i:3d}{k_true:9.4f}{k_r:9.4f}{k_p:9.4f}'
              f'{ll_t:12.1f}{ll_r:12.1f}{ll_p:12.1f}{ll_p-ll_r:14.1f}{better:>10s}',
              flush=True)
        rows.append(dict(i=i, k_true=k_true, k_raw=k_r, k_pol=k_p,
                         ll_true=ll_t, ll_raw=ll_r, ll_pol=ll_p,
                         holdout=bool(a.holdout), n_score=n_score,
                         d_ll=ll_p - ll_r, n_true=int(len(tA))))

    out = os.path.join(HERE, 'probe_lik.jsonl')
    with open(out, 'a', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    from math import comb
    n = len(rows)
    up = sum(1 for r in rows if r['ll_pol'] > r['ll_true'])
    gain = sum(1 for r in rows if r['d_ll'] > 0)
    pv = lambda k: sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    dll = np.array([r['d_ll'] for r in rows])
    print(f'\n{"="*80}')
    print(f'n={n}')
    print(f'★다듬기가 우도를 **올린** 장면            = {gain}/{n}  단측 p={pv(gain):.3e}'
          f'  ΔLL 중앙 {np.median(dll):+.1f}')
    print(f'★다듬은 집합이 **참 원자보다** 우도가 높은 장면 = {up}/{n}  단측 p={pv(up):.3e}')
    rr = np.array([r['ll_raw'] - r['ll_true'] for r in rows])
    print(f'  (원출력이 참보다 높은 장면 = {int((rr>0).sum())}/{n})')
    print(f'\n⛔이 프로브는 정확도를 재지 않았다. 우도는 **400 관측 위의 값**이다 —'
          f' 참 장 전체에 대한 적합도가 아니다.')


if __name__ == '__main__':
    main()
