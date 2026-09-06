#!/usr/bin/env python3
"""`T22-APLADDER` **동결 전 프로브 2** — 제안망 **자신의** 지도와 **자신의** 원자를 본다.

⛔**판정이 아니다.** 사전등록 미작성 · 게이트·분석기·러너 0개 · **추정기 적합 0회**
(제안 단계까지만 — `propose_atoms` 는 부르지만 `initialize()`/`update()` 는 부르지 않는다).
노출 관리: `gates_t11.SMOKE_FILES`(이미 노출된 감사 표본)만 쓴다.

## 왜 프로브 2 인가

프로브 1(`PROBE_1.md`)은 `ρ_R` 이 학습 범위 밖일 때 첨두 파수 추정이 **58.9–160.3배** 틀리고
범위 안에서 3/4 칸이 **0.98–1.13배**로 맞는 것을 봤다. ⛔**그러나 그것은 진단용 자**
(`scale_rules_t14.kp_hat`)였고 **제안망 자신의 입력이 아니다.**

이 프로브는 그 간극을 닫는다 — 배포 경로가 실제로 쓰는 것 둘을 직접 읽는다:
  ① `d3_beamform_cond_diffusion._bmap_from_obs` 가 만든 **조건화 지도의 argmax**
  ② `learned_init.propose_atoms` 가 낸 **원자의 |k| 중앙값** (= `T12` 의 `kmed_raw`)
그리고 둘을 조밀 참 장의 FFT 첨두 `k_true` 로 나눈다.

★`T12` 는 배포 팔이 **참값의 10.378배** 파수에 제안한다고 쟀다(`verdict_t12.json.mediation`).
이 프로브는 ***그 배율이 `ρ_R` 사다리를 따라 움직이는가*** 를 묻는다.

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T22_APLADDER_2026-09-06/probe_prop.py
"""
import json
import os
import sys
import time
import warnings
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp/T11_EXTACC_2026-09-03'),
          os.path.join(ROOT, 'tmp/T14_AUTOSCALE_2026-09-04'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

FRACS = (1.0, 0.5, 1.0 / 3.0, 0.25)


def main():
    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import gates_t11 as G
    import run_t11 as R11                      # ★`read_mat`·`build_frames` 를 임포트(재구현 0)
    import probe_geom as PG                    # ★`sample_sub`·`true_peak_k` 를 임포트
    import d3_beamform_net as BN
    import d3_beamform_cond_diffusion as CD
    from wavesplat.optimizer import learned_init as LI

    KLIM, NK = float(BN.KLIM), int(BN.NK)
    dk_map = 2.0 * KLIM / (NK - 1)
    cap = int(LI.M_OBS_CAP)
    kax = np.linspace(-KLIM, KLIM, NK)
    print(f'지도 {NK}×{NK} · KLIM={KLIM} · dk_map={dk_map:.6f} rad/m · M_OBS_CAP={cap}')
    print(f'⚠지도 argmax 의 분해능 하한 = {dk_map:.4f} rad/m '
          f'⟹ 참 첨두 0.0402 는 DC 에서 약 2칸 거리다\n')

    rows = []
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, name in enumerate(G.SMOKE_FILES, 1):
            E, gx, gy, tarr = R11.read_mat(zf, name)
            nf = int(G.WIN_S)
            Ew, tw = E[G.J0:G.J0 + nf], tarr[G.J0:G.J0 + nf]
            k_true = PG.true_peak_k(Ew[-1], gx, gy)
            print(f'== {name}   k_true = {k_true:.5f} rad/m (λ={2*np.pi/k_true:.1f} m)')
            print(f"   {'frac':>6s}{'L400':>8s}{'rho_R':>8s}{'rhoIN?':>7s}"
                  f"{'map|k|':>9s}{'map/true':>10s}{'atom|k|med':>11s}"
                  f"{'atom/true':>10s}{'n_atom':>7s}{'wall':>7s}")
            for f_ in FRACS:
                rng = np.random.default_rng(G.SEED + fi)
                x, y, tt, zz = PG.sample_sub(Ew, gx, gy, tw, G.DENSITY, rng, f_)
                frames = R11.build_frames(x, y, tt, zz, G.GT_CONST)

                # ── ① 제안망 조건화 지도 — `propose_atoms` 와 **같은 규약**으로 400점 ──
                P = np.column_stack([x, y, tt]).astype(np.float32)
                zf32 = zz.astype(np.float32)
                if len(zf32) > cap:
                    idx = np.random.default_rng(0).choice(len(zf32), cap, replace=False)
                    P, zf32 = P[idx], zf32[idx]
                L4 = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])))
                Ps = P.copy(); Ps[:, 2] -= Ps[:, 2].min()
                B = CD._bmap_from_obs(Ps, zf32)
                Bn = np.asarray(B.detach().cpu().numpy()).squeeze()
                iy, ix = np.unravel_index(int(np.argmax(Bn)), Bn.shape[-2:])
                k_map = float(np.hypot(kax[ix], kax[iy]))

                # ── ② 제안망이 실제로 낸 원자 ──
                t0 = time.time()
                akx, aky, aen = LI.propose_atoms(frames, G.DEPTH, nsamp=4,
                                                 diffusion_seed=0)
                wall = time.time() - t0
                amag = np.hypot(np.asarray(akx, float), np.asarray(aky, float))
                k_atom = float(np.median(amag)) if amag.size else float('nan')

                rho = dk_map * L4 / (2 * np.pi)
                rin = 'IN' if 0.219 <= rho <= 0.724 else 'OUT'
                print(f'   {f_:6.3f}{L4:8.1f}{rho:8.3f}{rin:>7s}'
                      f'{k_map:9.4f}{k_map/k_true:10.2f}{k_atom:11.4f}'
                      f'{k_atom/k_true:10.2f}{amag.size:7d}{wall:7.1f}', flush=True)
                rows.append(dict(block=name, frac=float(f_), L400=L4, rho_R=rho,
                                 rho_in=rin, k_true=k_true, k_map=k_map,
                                 map_over_true=k_map / k_true, k_atom_med=k_atom,
                                 atom_over_true=k_atom / k_true,
                                 n_atoms_raw=int(amag.size), wall_s=round(wall, 2),
                                 dk_map=dk_map, KLIM=KLIM, NK=NK))
            print()

    out = os.path.join(HERE, 'probe_prop.jsonl')
    with open(out, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{len(rows)} 행 → {out}')
    print('★대조: `T12 verdict.mediation` 의 배포 팔 배율 = 10.378 '
          '(제3자 코퍼스 전체 개구, 다른 블록 표본)')
    print('⛔동결 전 프로브다 — 어떤 주장도 지탱하지 않는다.')


if __name__ == '__main__':
    main()
