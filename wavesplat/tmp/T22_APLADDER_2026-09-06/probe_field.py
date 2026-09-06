r"""`T22` **동결 전 프로브 11** — ★다듬기가 **렌더링한 장**을 나쁘게 만드는가.

⛔**판정이 아니다.** 사전등록 미동결. ⚠**적합기는 안 돈다** — 패키지의 진단 함수
`learned_init.raw_field_nrmse`(*"nRMSE of the learned proposal rendered field
(LS w/phi, NO joint fit)"*)를 그대로 쓴다. ⟹ **제안 단계의 장**만 채점한다.

## 왜 — 프로브 10 이 남긴 정확히 한 칸

프로브 10: 다듬기가 **홀드아웃 우도도** 올린다(4/4). ⟹ 과적합이 아니다.
⛔**그러나 그 홀드아웃은 같은 격자·같은 프레임에서 왔다** — 표집 기하를 공유한다.
***연속 장에 대한 적합도는 그것으로 판정되지 않는다.***

이 프로브가 그 칸을 채운다: 같은 원자 집합을 **참 격자 전체**에 렌더링해 참 장과 비교한다.

**갈림 (⛔결과 보기 전에 적는다)**
- 다듬기가 장도 좋게 하면 ⟹ 파수 수축은 **무해**하고 `T22` 는 `P0` 열세를 등록해야 한다.
- ★다듬기가 **우도는 올리는데 장은 나쁘게** 하면 ⟹ ***목적함수가 목표량과 어긋난다*** 는
  **직접 증거**이고, 그것이 이 연구의 가장 강한 산출 형태다.

사용
----
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T22_APLADDER_2026-09-06/probe_field.py --stride 3
"""
import argparse
import glob
import json
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='data/episode_*_uniform.h5')
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--dseed', type=int, default=0)
    ap.add_argument('--iters', type=int, nargs='+', default=[0, 80])
    a = ap.parse_args()

    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import h5py
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer import learned_init as LI
    from d23_main_results import K_TP

    files = sorted(glob.glob(a.pattern))[::a.stride]
    print(f'홈 에피소드 {len(files)} · K_TP={K_TP} · iters={a.iters} · '
          f'채점 = learned_init.raw_field_nrmse (LS w/phi, 결합적합 없음)')
    print(f"{'episode':34s}{'k_true':>8s}" +
          ''.join(f"{'nRMSE@'+str(i):>12s}" for i in a.iters) + f"{'Δ(80-0)':>10s}")
    rows = []
    for fp in files:
        with h5py.File(fp, 'r') as f:
            wc = np.asarray(f['config/wave_components'])
        if wc.size == 0:
            continue
        j = int(np.argmax(wc[:, 0]))
        k_true = float(np.hypot(wc[j, 2], wc[j, 3]))
        L = EpisodeDataLoader(fp)
        dt = float(L[1].t - L[0].t)
        Tp = float(getattr(L, 'Tp_true', 2 * math.pi / max(wc[j, 1], 1e-9)))
        nb = max(1, int(round(K_TP * Tp / dt)))
        i0 = min(len(L) - 2, nb + 3)
        frames = [L[i] for i in range(max(0, i0 - nb), i0 + 1)]
        fstar = L[i0]
        gx = np.asarray(L.gt_grid_x); gy = np.asarray(L.gt_grid_y)
        gt = np.asarray(fstar.gt_field)
        Hs = float(L.Hs_true)
        vals = []
        for it in a.iters:
            cfg = {'learned_diffusion_seed': a.dseed, 'learned_polish_iters': it}
            v = LI.raw_field_nrmse(cfg, frames, float(L.depth), gt, gx, gy,
                                   float(fstar.t), Hs, seed=0)
            vals.append(float(v))
        nm = os.path.basename(fp).replace('episode_', '').replace('.h5', '')
        d = vals[-1] - vals[0]
        print(f'{nm:34s}{k_true:8.4f}' + ''.join(f'{v:12.4f}' for v in vals)
              + f'{d:+10.4f}', flush=True)
        rows.append(dict(episode=nm, k_true=k_true, Tp=Tp, depth=float(L.depth),
                         iters=list(a.iters), nrmse=vals, d_80_0=d))

    with open(os.path.join(HERE, 'probe_field.jsonl'), 'a', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    from math import comb
    d = np.array([r['d_80_0'] for r in rows]); n = len(d)
    worse = int((d > 0).sum())
    p = sum(comb(n, i) for i in range(worse, n + 1)) / 2 ** n
    print(f'\n{"="*80}\nn={n}')
    print(f'★다듬기가 **장을 나쁘게** 한 에피소드 = {worse}/{n}  단측 p={p:.4f}')
    print(f'  Δ nRMSE 중앙 = {np.median(d):+.4f} · 평균 = {d.mean():+.4f} '
          f'· 범위 [{d.min():+.4f}, {d.max():+.4f}]')
    kt = np.array([r['k_true'] for r in rows])
    if n >= 6:
        print(f'  corr(log k_true, Δ) = {np.corrcoef(np.log(kt), d)[0,1]:+.3f} '
              f'(음수면 **긴 파도에서 더 해롭다**)')
    print('\n⛔적합기는 안 돌았다 — 제안 단계의 장만 잰 것이고 배치의 PRIMARY 가 아니다.')


if __name__ == '__main__':
    main()
