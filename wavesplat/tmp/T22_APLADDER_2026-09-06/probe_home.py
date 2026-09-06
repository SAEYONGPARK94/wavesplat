r"""`T22` **동결 전 프로브 7** — 수축 법칙이 **홈 코퍼스**에서도 성립하는가.

⛔**판정이 아니다.** 사전등록 미작성 · 게이트·분석기·러너 0개 · **추정기 적합 0회**
(`propose_atoms` 까지만).

## 왜

프로브 3–6 은 전부 **`gen_mixed` 학습 장면**과 **제3자 코퍼스** 위에서 쟀다.
그런데 배포 팔이 **실제로 이기는 곳은 홈 코퍼스**(`data/episode_*.h5`, `wavesim` 생성기)다.
법칙이 거기서도 성립하면 **생성기 무관**이고, 아니면 **학습 생성기 특유**다.
⟹ 둘 중 어느 쪽이든 사전등록의 코퍼스 선택을 정한다.

★홈 에피소드는 `config/wave_components`(`[A, omega, kx, ky, phi, Q]`)를 갖고 있으므로
`k_true` = **최대 진폭 성분의 |k|** 를 추정 없이 안다(제3자 코퍼스는 FFT 추정이었다).

⛔**재구현 0** — 프레임은 `wavesplat.data_loader.EpisodeDataLoader` 가 만든 `FrameData` 를
그대로 쓰고, 창 길이 규약은 `d23_main_results.K_TP` 를 임포트한다.

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T22_APLADDER_2026-09-06/probe_home.py --polish-iters 80
"""
import argparse
import glob
import json
import math
import os
import re
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
    ap.add_argument('--polish-iters', type=int, default=80)
    ap.add_argument('--dseed', type=int, default=0)
    ap.add_argument('--pattern', default='data/episode_*_uniform.h5')
    ap.add_argument('--stride', type=int, default=3, help='에피소드 목록에서 n개마다 하나')
    ap.add_argument('--tag', default='home80')
    a = ap.parse_args()

    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import h5py
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer import learned_init as LI
    from d23_main_results import K_TP

    files = sorted(glob.glob(a.pattern))[::a.stride]
    print(f'홈 에피소드 {len(files)} 개 (패턴 {a.pattern}, stride {a.stride}) · '
          f'K_TP={K_TP} · polish_iters={a.polish_iters}')
    print(f"{'episode':38s}{'ncomp':>6s}{'k_true':>9s}{'Tp':>7s}{'nb':>5s}"
          f"{'k_atom':>9s}{'ratio':>8s}{'n_sel':>7s}{'k_sel':>9s}")
    rows = []
    for fp in files:
        with h5py.File(fp, 'r') as f:
            wc = np.asarray(f['config/wave_components'])
        if wc.size == 0:
            continue
        j = int(np.argmax(wc[:, 0]))
        k_true = float(np.hypot(wc[j, 2], wc[j, 3]))
        om = float(wc[j, 1])
        Tp = 2 * math.pi / om if om > 0 else float('nan')

        L = EpisodeDataLoader(fp)
        dt = float(L[1].t - L[0].t)
        nb = max(1, int(round(K_TP * (L.Tp_true if hasattr(L, 'Tp_true') else Tp) / dt)))
        i0 = min(len(L) - 2, nb + 3)
        frames = [L[i] for i in range(max(0, i0 - nb), i0 + 1)]

        akx, aky, aen = LI.propose_atoms(frames, float(L.depth), nsamp=4,
                                         diffusion_seed=a.dseed,
                                         polish_iters=a.polish_iters)
        am = np.hypot(np.asarray(akx, float), np.asarray(aky, float))
        k_at = float(np.median(am)) if am.size else float('nan')
        skx, sky, _ = LI.pool_and_select(akx, aky, aen, energy_thresh=0.03,
                                         dedup_dk=0.03, max_n=int(LI.POOL_CAP_DEFAULT))
        sm = np.hypot(np.asarray(skx, float), np.asarray(sky, float))
        k_sel = float(np.median(sm)) if sm.size else float('nan')

        nm = os.path.basename(fp).replace('episode_', '').replace('.h5', '')
        print(f'{nm:38s}{len(wc):6d}{k_true:9.4f}{Tp:7.2f}{nb:5d}'
              f'{k_at:9.4f}{k_at/k_true:8.2f}{sm.size:7d}{k_sel:9.4f}', flush=True)
        rows.append(dict(domain='home', tag=a.tag, episode=nm, ncomp=int(len(wc)),
                         k_true=k_true, Tp=Tp, nb=nb, depth=float(L.depth),
                         k_atom=k_at, ratio=k_at / k_true, n_sel=int(sm.size),
                         k_sel=k_sel, ratio_sel=k_sel / k_true,
                         pol=a.polish_iters, dseed=a.dseed))

    out = os.path.join(HERE, 'probe_home.jsonl')
    with open(out, 'a', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    kt = np.array([r['k_true'] for r in rows]); ka = np.array([r['k_atom'] for r in rows])
    ks = np.array([r['k_sel'] for r in rows])
    ok = np.isfinite(kt) & np.isfinite(ka) & (kt > 0) & (ka > 0)
    print(f'\nn={int(ok.sum())} · k_true {kt[ok].min():.4f}–{kt[ok].max():.4f} '
          f'({kt[ok].max()/kt[ok].min():.1f}배)')
    if ok.sum() >= 5:
        b, c = np.polyfit(np.log(kt[ok]), np.log(ka[ok]), 1)
        r = np.corrcoef(np.log(kt[ok]), np.log(ka[ok]))[0, 1]
        res = np.log(ka[ok]) - (b * np.log(kt[ok]) + c)
        sx = ((np.log(kt[ok]) - np.log(kt[ok]).mean()) ** 2).sum()
        se = math.sqrt((res @ res) / (ok.sum() - 2) / sx)
        print(f'★홈 적합: k_atom = {math.exp(c):.4f} · k_true^{b:.4f}  '
              f'95% CI [{b-1.96*se:.3f}, {b+1.96*se:.3f}] · pearson(log)={r:.3f}')
        ok2 = ok & np.isfinite(ks) & (ks > 0)
        if ok2.sum() >= 5:
            b2 = np.polyfit(np.log(kt[ok2]), np.log(ks[ok2]), 1)[0]
            print(f'  선택 후 기울기 = {b2:.4f}')
    print('\n★대조 (학습 분포, 프로브 3): 기울기 0.4595 · 95% CI [0.419, 0.500] (polish 80)')
    print('⛔동결 전 프로브다 — 정확도를 재지 않았다.')


if __name__ == '__main__':
    main()
