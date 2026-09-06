#!/usr/bin/env python3
"""`T22-APLADDER` **동결 전 프로브 1** — 개구 사다리가 실제로 개구를 움직이는가.

⛔**판정이 아니다.** 사전등록 미작성 · 게이트·분석기·러너 0개 · 추정기 적합 **0회**.
노출 관리: `gates_t11.SMOKE_FILES`(**이미 노출된** 감사 표본) 만 쓴다 —
`sample_t11.json` 의 프리즌 표본은 **건드리지 않는다**.

## 왜 이 배치인가

2026-09-06 조사가 내 아침 서술을 깼다:
- 제3자 코퍼스 개구 `L_obs` = **619.645 m 상수**(`T14` 212행 전건, 분산 0)
- 학습 지지집합 = **[72.34, 238.75] m** ⟹ 0/36 겹침
- ⛔**그러나 이긴 노브가 그것을 안으로 옮기지 않는다**: `A-MAPSCALE` 은
  `KLIM 0.6→0.3 · K_LIM 1.1→0.55` = **모델 상수만**(`consts_t12.py:57`), 관측 무변경.
  `RESCALE(s=2)` 은 619.6 → **309.8** = **여전히 밖**.
- 그리고 제3자 파수는 학습 밴드 **안**이다(36/36).
⟹ 최강 경쟁설명 ***"지지집합과 무관하게 2–4배 상수면 아무거나 듣는다"*** 를 **죽일 수 없다**.
제3자 개구가 **상수**라 개구 축이 구조적으로 검정 불가이기 때문이다.

## ★설계 논거 — `M_OBS_CAP=400` 이 이 사다리를 단일요인으로 만든다

`T11 AMENDMENT_1 §1`(BLOCKER-1)이 **밀도** 사다리를 죽였다: `learned_init.py:65 M_OBS_CAP=400`
때문에 제안 단계가 **네 칸 전부에서 정확히 400점**을 본다 ⟹ 밀도가 제안에 도달하지 않는다.

★그 캡이 **개구 사다리에서는 반대로 작동한다** — 점 예산이 400 으로 **고정**되므로
칸 사이에서 변하는 것은 그 400점이 덮는 **공간·시간 범위**뿐이다.
그리고 그것이 학습 분포와 이 코퍼스를 가르는 바로 그 양이다
(학습 장면도 제안에 400점을 주고, 그 범위가 **72–239 m** 다).

⚠**정직한 한계**: 점 예산 고정에서 개구와 (제안이 보는) 밀도는 **분리 불가능**하다 —
400점을 좁은 창에 놓으면 밀도가 오른다. ⟹ 축의 이름은 *"개구"* 가 아니라
***"고정 400점 예산이 덮는 공간 범위"*** 여야 하고, 그 병기는 의무다.

## 이 프로브가 재는 것 (적합 0회)

부분창 비율 `f ∈ {1.0, 0.5, 1/3, 0.25}` 에 대해, 격자 중앙의 `f × f` 창에서만
`sample_obs` 와 **같은 규약**으로 표집하고:
  · 실제 `L_obs = max(ptp x, ptp y)` 와 학습 지지집합 [72.34, 238.75] 대비 위치
  · 제안이 실제로 보는 점 수(`M_OBS_CAP` 적용 후)와 그 400점의 `L_obs`
  · 그 400점의 `k̂_p`(고정격자 빔포머, `scale_rules_t14.kp_hat`) — ★**밴드가 따라 움직이면
    사다리가 단일요인이 아니다**
  · 격자 자체의 제원(dims · dx · 물리 범위 · 프레임 수 · dt)

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T22_APLADDER_2026-09-06/probe_geom.py
"""
import io as _io
import json
import os
import sys
import warnings
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'tmp/T11_EXTACC_2026-09-03'),
          os.path.join(ROOT, 'tmp/T14_AUTOSCALE_2026-09-04')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

FRACS = (1.0, 0.5, 1.0 / 3.0, 0.25)
TRAIN_LO, TRAIN_HI = 72.34, 238.75      # `derive_train_support.py` 재도출값(재현 확인)


def read_mat(z, name):
    """★`run_t11.read_mat` 과 **축자 동일**(재중심화 포함)."""
    import h5py
    with h5py.File(_io.BytesIO(z.read(name)), 'r') as f:
        E = np.asarray(f['eta']).squeeze()
        x = np.asarray(f['x']).squeeze(); y = np.asarray(f['y']).squeeze()
        t = np.asarray(f['t']).squeeze()
    return E, x - x.mean(), y - y.mean(), t


def sample_sub(E, gx, gy, t, npts, rng, frac):
    """`run_t11.sample_obs` 와 같은 규약이되 **격자 중앙의 frac×frac 창**에서만 뽑는다."""
    ny, nx = E.shape[1], E.shape[2]
    hy, hx = max(1, int(round(ny * frac))), max(1, int(round(nx * frac)))
    y0, x0 = (ny - hy) // 2, (nx - hx) // 2
    ys_i = np.arange(y0, y0 + hy)
    xs_i = np.arange(x0, x0 + hx)
    X, Y = np.meshgrid(gx[xs_i], gy[ys_i], indexing='xy')
    sub = E[:, y0:y0 + hy, x0:x0 + hx]
    xs, ys, ts, zs = [], [], [], []
    t0 = float(t[0])
    for j in range(sub.shape[0]):
        n = sub[j].size
        k = min(npts, n)
        idx = rng.choice(n, k, replace=False)
        iy, ix = np.unravel_index(idx, sub[j].shape)
        xs.append(X[iy, ix]); ys.append(Y[iy, ix])
        ts.append(np.full(k, float(t[j]) - t0)); zs.append(sub[j][iy, ix])
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(ts), np.concatenate(zs))


def true_peak_k(field, gx, gy):
    """조밀 참 장의 2-D FFT 첨두 파수 크기 [rad/m]. ⛔진단용 — 어떤 팔도 이것을 쓰지 않는다."""
    F = np.fft.fft2(np.asarray(field, float))
    P = np.abs(F) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(field.shape[1], gx[1] - gx[0])
    ky = 2 * np.pi * np.fft.fftfreq(field.shape[0], gy[1] - gy[0])
    P[0, 0] = 0.0
    i, j = np.unravel_index(int(np.argmax(P)), P.shape)
    return float(np.hypot(kx[j], ky[i]))


def main():
    warnings.filterwarnings('ignore')
    import gates_t11 as G
    import scale_rules_t14 as SR
    from wavesplat.optimizer import learned_init as LI

    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import d3_beamform_net as BN      # ★손복사 0 — 배포 지도 상수를 임포트한다
    KLIM, NK = float(BN.KLIM), int(BN.NK)
    dk_map = 2.0 * KLIM / (NK - 1)
    cap = int(LI.M_OBS_CAP)
    print(f'배포 지도: KLIM={KLIM} · NK={NK} ⟹ dk_map={dk_map:.6f} rad/m'
          f'  (scripts/d3_beamform_net.py)')
    print('rho_R = dk_map · L_obs / 2pi · 학습 범위 [0.219, 0.724]'
          '  ⚠그 범위는 재도출하지 않고 인용한다')
    print(f'M_OBS_CAP = {cap}  (learned_init.py) · DENSITY = {G.DENSITY} 점/프레임 · '
          f'WIN_S = {G.WIN_S} · J0 = {G.J0}')
    print(f'학습 지지집합 개구 = [{TRAIN_LO}, {TRAIN_HI}] m\n')

    rows = []
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, name in enumerate(G.SMOKE_FILES, 1):
            E, gx, gy, tarr = read_mat(zf, name)
            nf = int(G.WIN_S)
            Ew, tw = E[G.J0:G.J0 + nf], tarr[G.J0:G.J0 + nf]
            print(f'== {name}')
            print(f'   grid {Ew.shape[1]}×{Ew.shape[2]} · dx={gx[1]-gx[0]:.3f} m · '
                  f'dy={gy[1]-gy[0]:.3f} m · x∈[{gx.min():.1f},{gx.max():.1f}] · '
                  f'프레임 {Ew.shape[0]} · dt={tarr[1]-tarr[0]:.3f} s')
            k_true = true_peak_k(Ew[-1], gx, gy)
            print(f'   참 첨두 |k| (조밀 FFT) = {k_true:.5f} rad/m  '
                  f'(lambda = {2*np.pi/max(k_true,1e-9):.1f} m)')
            print(f'   {"frac":>6s}{"L_obs":>9s}{"vs train":>10s}{"n_prov":>8s}'
                  f"{'n_prop':>8s}{'L400':>9s}{'dens400':>10s}{'kp400':>9s}"
                  f"{'kp/ktrue':>10s}{'rho_R':>8s}{'rhoIN?':>8s}")
            for f_ in FRACS:
                rng = np.random.default_rng(G.SEED + fi)
                x, y, tt, zz = sample_sub(Ew, gx, gy, tw, G.DENSITY, rng, f_)
                L = max(float(np.ptp(x)), float(np.ptp(y)))
                # 제안이 실제로 보는 400점 (learned_init.py:193-196 과 같은 규약)
                if len(zz) > cap:
                    sel = np.random.default_rng(0).choice(len(zz), cap, replace=False)
                else:
                    sel = np.arange(len(zz))
                x4, y4, t4, z4 = x[sel], y[sel], tt[sel], zz[sel]
                L4 = max(float(np.ptp(x4)), float(np.ptp(y4)))
                kp4 = float(SR.kp_hat(x4, y4, t4, z4, G.DEPTH))
                lam = 2 * np.pi / max(kp4, 1e-9)
                dens = len(z4) / (L4 * L4) * 1e4      # 점 / (100 m)²
                tag = 'IN' if TRAIN_LO <= L4 <= TRAIN_HI else (
                    'OUT>hi' if L4 > TRAIN_HI else 'OUT<lo')
                lam_t = 2 * np.pi / max(k_true, 1e-9)
                print(f'   {f_:6.3f}{L:9.1f}{tag:>10s}{len(zz):8d}{len(z4):8d}'
                      f'{L4:9.1f}{dens:10.2f}{kp4:9.4f}{kp4/k_true:10.2f}'
                      f'{dk_map*L4/(2*np.pi):8.3f}'
                      f'{("IN" if 0.219 <= dk_map*L4/(2*np.pi) <= 0.724 else "OUT"):>8s}')
                rows.append(dict(block=name, frac=float(f_), L_obs=L,
                                 n_provided=int(len(zz)), n_proposal=int(len(z4)),
                                 L400=L4, dens400_per_100m2=dens, kp400=kp4,
                                 k_true=k_true, kp_over_ktrue=kp4 / k_true,
                                 L_over_lam_ruler=L4 / lam,
                                 L_over_lam_true=L4 / lam_t, in_train=tag,
                                 rho_R=dk_map * L4 / (2 * np.pi), dk_map=dk_map,
                                 KLIM=KLIM, NK=NK))
            print()

    out = os.path.join(HERE, 'probe_geom.jsonl')
    with open(out, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{len(rows)} 행 → {out}')
    print('⛔동결 전 프로브다 — 어떤 주장도 지탱하지 않는다.')


if __name__ == '__main__':
    main()
