#!/usr/bin/env python3
r"""`T22` **동결 전 프로브 3** — 제안망은 장면의 파수를 따라가는가, 자기 학습 밴드에 붙어 있는가.

⛔**판정이 아니다.** 사전등록 미작성 · 게이트·분석기·러너 0개 · **추정기 적합 0회**
(`propose_atoms` 까지만). 노출 관리: 제3자는 `gates_t11.SMOKE_FILES` 2개만.

## 질문 (프로브 2 가 낳았다)

프로브 2 는 개구를 4배 좁혀 조건화 지도를 정확하게(1.21배 = 양자화) 만들어도
제안 원자가 여전히 **참값의 약 4배**임을 봤다. 그 원자의 절대 파수는 **0.15–0.17 rad/m** 였고,
배포 체크포인트의 **학습 기하평균 `k_p` 는 0.0918(nominal) / 0.1218(빔포머)** 이다
(`memory/reference_train_support_deployed_ckpt_2026-09-05.md` §2.2).
⟹ ★***"4배 편향" 이 아니라 "학습 밴드에 붙어 있는 것" 일 수 있다.***

두 가설은 **다른 것을 예측한다**:

| | `k_true` 를 바꾸면 원자 \|k\| 는 |
|---|---|
| **H-비례** (곱셈 편향) | `k_true` 를 따라간다 (비 ≈ 4 유지) |
| **H-앵커** (학습 밴드 고정) | **거의 안 움직인다** (절대값 ≈ 0.12–0.17 유지) |

## 설계

`k_true` 를 넓게 훑기 위해 **학습 분포 자신**에서 장면을 뽑는다 —
`gen_mixed.sample_scene_ok` 는 **참 원자**(`tkx`·`tky`·`tA`)를 함께 주므로
`k_true` = 최대 진폭 원자의 `|k|` 를 **정확히** 안다(FFT 추정이 아니다).
그 위에 제3자 코퍼스 2블록(개구 1/4 창 = 지도가 정확한 칸)을 얹는다.

⚠**분포 안 장면에서 비가 1 에 가깝다면** 그것은 사전이 조건화를 쓴다는 뜻이고,
분포 밖에서만 앵커로 붙는다면 **그것이 작동 경계의 기제 후보**다.
⛔단 이 프로브는 **정확도를 재지 않는다.**

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T22_APLADDER_2026-09-06/probe_scaleanchor.py --nscene 40
"""
import argparse
import json
import os
import sys
import warnings
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp/T11_EXTACC_2026-09-03'),
          os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

TRAIN_GM_NOMINAL, TRAIN_GM_BEAM = 0.0918, 0.1218   # ⚠인용(재도출 아님)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nscene', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20260906)
    ap.add_argument('--dseed', type=int, default=0, help='learned_diffusion_seed')
    ap.add_argument('--tag', default='r1', help='복제 태그 (행에 기록)')
    ap.add_argument('--polish-iters', type=int, default=80,
                    help='0 이면 확산 원출력(polish 없음) — 수축이 어디서 생기는지 가른다')
    a = ap.parse_args()

    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import gates_t11 as G
    import run_t11 as R11
    import probe_geom as PG
    import gen_mixed as gm
    from wavesplat.optimizer import learned_init as LI

    print(f'배포 학습 기하평균 k_p = {TRAIN_GM_NOMINAL} (nominal) / '
          f'{TRAIN_GM_BEAM} (빔포머) rad/m  ⚠인용값\n')
    rows = []

    # ── ① 학습 분포 자신 (참 원자를 안다) ──────────────────────────────────
    bank = gm.GeometryBank()
    rng = np.random.default_rng(a.seed)
    print(f'== 학습 분포 (gen_mixed.sample_scene_ok, n={a.nscene})')
    for i in range(a.nscene):
        s = gm.sample_scene_ok(rng, bank)
        tkx, tky, tA = (np.asarray(s['tkx'], float), np.asarray(s['tky'], float),
                        np.asarray(s['tA'], float))
        j = int(np.argmax(tA))
        k_true = float(np.hypot(tkx[j], tky[j]))
        P = np.asarray(s['P'], float); z = np.asarray(s['z'], float)
        frames = R11.build_frames(P[:, 0], P[:, 1], P[:, 2] - P[:, 2].min(), z, G.GT_CONST)
        akx, aky, _en = LI.propose_atoms(frames, 50.0, nsamp=4, diffusion_seed=a.dseed,
                                         polish_iters=a.polish_iters)
        am = np.hypot(np.asarray(akx, float), np.asarray(aky, float))
        k_at = float(np.median(am))
        # ★배포 선택기를 그대로 통과시킨다 (learned_init_params 기본 노브)
        skx, sky, sen = LI.pool_and_select(akx, aky, _en, energy_thresh=0.03,
                                           dedup_dk=0.03,
                                           max_n=int(LI.POOL_CAP_DEFAULT))
        sm = np.hypot(np.asarray(skx, float), np.asarray(sky, float))
        k_sel = float(np.median(sm)) if sm.size else float('nan')
        L = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])))
        rows.append(dict(domain='train', tag=a.tag, seed=a.seed, dseed=a.dseed, pol=a.polish_iters, idx=i, k_true=k_true, k_atom=k_at,
                         ratio=k_at / max(k_true, 1e-9), L_obs=L,
                         k_sel=k_sel, n_sel=int(sm.size),
                         ratio_sel=k_sel / max(k_true, 1e-9),
                         n_obs=int(len(z)), Tp=float(s['Tp'])))
        if (i + 1) % 10 == 0:
            print(f'   {i+1}/{a.nscene}', flush=True)

    # ── ② 제3자 코퍼스 (1/4 창 = 지도가 정확한 칸) ─────────────────────────
    print('\n== 제3자 코퍼스 (frac=0.25, 지도가 정확한 칸)')
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, name in enumerate(G.SMOKE_FILES, 1):
            E, gx, gy, tarr = R11.read_mat(zf, name)
            Ew, tw = E[G.J0:G.J0 + int(G.WIN_S)], tarr[G.J0:G.J0 + int(G.WIN_S)]
            k_true = PG.true_peak_k(Ew[-1], gx, gy)
            r = np.random.default_rng(G.SEED + fi)
            x, y, tt, zz = PG.sample_sub(Ew, gx, gy, tw, G.DENSITY, r, 0.25)
            frames = R11.build_frames(x, y, tt, zz, G.GT_CONST)
            akx, aky, _en3 = LI.propose_atoms(frames, G.DEPTH, nsamp=4, diffusion_seed=a.dseed,
                                         polish_iters=a.polish_iters)
            am = np.hypot(np.asarray(akx, float), np.asarray(aky, float))
            k_at = float(np.median(am))
            # ★제3자에도 선택기 열을 켠다 (2026-09-06 심사 지적: 낙관값이 외삽이었다)
            tskx, tsky, _ts = LI.pool_and_select(akx, aky, _en3, energy_thresh=0.03,
                                                 dedup_dk=0.03,
                                                 max_n=int(LI.POOL_CAP_DEFAULT))
            tsm = np.hypot(np.asarray(tskx, float), np.asarray(tsky, float))
            k_sel3 = float(np.median(tsm)) if tsm.size else float('nan')
            L = max(float(np.ptp(x)), float(np.ptp(y)))
            rows.append(dict(k_sel=k_sel3, n_sel=int(tsm.size),
                             ratio_sel=k_sel3 / k_true, domain='third', tag=a.tag, seed=a.seed, dseed=a.dseed, pol=a.polish_iters, idx=name, k_true=k_true, k_atom=k_at,
                             ratio=k_at / k_true, L_obs=L, n_obs=int(len(zz)), Tp=None))
            print(f'   {name}: k_true={k_true:.4f} k_atom={k_at:.4f} '
                  f'ratio={k_at/k_true:.2f}')

    # ── 요약 ────────────────────────────────────────────────────────────────
    tr = [r for r in rows if r['domain'] == 'train']
    kt = np.array([r['k_true'] for r in tr]); ka = np.array([r['k_atom'] for r in tr])
    print(f'\n{"="*74}')
    print(f'학습 분포 n={len(tr)}')
    print(f'  k_true : min {kt.min():.4f} med {np.median(kt):.4f} max {kt.max():.4f}'
          f'   (변동 {kt.max()/kt.min():.1f}배)')
    print(f'  k_atom : min {ka.min():.4f} med {np.median(ka):.4f} max {ka.max():.4f}'
          f'   (변동 {ka.max()/ka.min():.1f}배)')
    print(f'  ratio  : min {(ka/kt).min():.2f} med {np.median(ka/kt):.2f} '
          f'max {(ka/kt).max():.2f}')
    lk, la = np.log(kt), np.log(ka)
    slope = float(np.polyfit(lk, la, 1)[0])
    rho = float(np.corrcoef(lk, la)[0, 1])
    print(f'  ★log-log 기울기 = {slope:.3f} · pearson(log) = {rho:.3f}')
    print(f'     ⟹ H-비례 는 기울기 ≈ 1 을, H-앵커 는 기울기 ≈ 0 을 예측한다')
    print(f'  k_atom 의 변동계수 = {ka.std()/ka.mean():.3f} vs '
          f'k_true 의 = {kt.std()/kt.mean():.3f}')

    out = os.path.join(HERE, 'probe_scaleanchor.jsonl')
    with open(out, 'a', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n{len(rows)} 행 → {out}')
    print('⛔동결 전 프로브다 — 정확도를 재지 않았다.')


if __name__ == '__main__':
    main()
