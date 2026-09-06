r"""`T22` **동결 전 프로브 10** — 다듬기가 **적합에 안 쓴 관측**에서도 우도를 올리는가.

⛔**판정이 아니다.** 사전등록 미작성 · 적합 0회(제안 단계까지만).

## 왜

프로브 8 은 다듬기가 우도를 **13/15** 에서 올리고(p=0.0037) **9/15** 에서 참 원자보다도
높음을 쟀다. ⟹ *"과적합인가, 진짜 더 잘 맞는 것인가"* 가 갈리지 않는다 —
그 우도는 **적합에 쓴 바로 그 400점**에서 잰 in-sample 값이기 때문이다.

⛔**프로브 8 의 홀드아웃 시도는 무효였다**: 학습 장면(`gen_mixed`)의 관측 수가 정확히
`M_OBS_CAP = 400` 이라 **홀드아웃 집합이 비었다**(전 행 0 출력). 인용하지 않는다.

★이 프로브는 **제3자 코퍼스**를 쓴다 — 창당 `200점 × 40프레임 = 8,000점` 이라
제안이 400점을 쓰고도 **7,600점이 남는다**.

**갈림**
- 다듬기가 **홀드아웃 우도도 올린다** ⟹ 진짜로 더 잘 맞는다 ⟹ ⛔`P0` 이 정확도를 **해칠** 것을 예측.
- 다듬기가 **in-sample 만 올리고 홀드아웃은 내린다** ⟹ **과적합** ⟹ `P0` 이 정확도를 **도울** 것을 예측.

⟹ 이것이 `T22` 의 **등록 방향**을 정한다. ⛔결과를 보기 전에 이 문장을 적어 둔다.

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T22_APLADDER_2026-09-06/probe_lik3.py
"""
import json
import os
import sys
import warnings
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp/T11_EXTACC_2026-09-03'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)


def main():
    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import gates_t11 as G
    import run_t11 as R11
    import probe_geom as PG
    from d3_setdiff_posterior import loglik_atoms, SIGMA
    from wavesplat.optimizer import learned_init as LI

    cap = int(LI.M_OBS_CAP)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'우도 = d3_setdiff_posterior.loglik_atoms (SIGMA={SIGMA}) · '
          f'M_OBS_CAP={cap} · device={dev}')
    print('★적합에 쓴 400점(IN) 과 **쓰지 않은** 점에서 뽑은 400점(OUT) 을 나란히 잰다\n')

    rows = []
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, name in enumerate(G.SMOKE_FILES, 1):
            E, gx, gy, tarr = R11.read_mat(zf, name)
            Ew, tw = E[G.J0:G.J0 + int(G.WIN_S)], tarr[G.J0:G.J0 + int(G.WIN_S)]
            k_true = PG.true_peak_k(Ew[-1], gx, gy)
            for frac in (1.0, 0.25):
                rng = np.random.default_rng(G.SEED + fi)
                x, y, tt, zz = PG.sample_sub(Ew, gx, gy, tw, G.DENSITY, rng, frac)
                frames = R11.build_frames(x, y, tt, zz, G.GT_CONST)
                allP = np.column_stack([x, y, tt]).astype(np.float32)
                allz = zz.astype(np.float32)
                assert len(allz) > cap, f'관측이 {len(allz)} 뿐 — 홀드아웃 불가'
                idx = np.random.default_rng(0).choice(len(allz), cap, replace=False)
                mask = np.zeros(len(allz), bool); mask[idx] = True
                Pin, zin = allP[mask], allz[mask]
                rest = np.flatnonzero(~mask)
                ho = np.random.default_rng(7).choice(rest, min(cap, len(rest)),
                                                     replace=False)
                Pout, zout = allP[ho], allz[ho]
                t0 = float(Pin[:, 2].min())          # ★두 집합에 **같은** 시간 원점
                Pin_s = Pin.copy(); Pin_s[:, 2] -= t0
                Pout_s = Pout.copy(); Pout_s[:, 2] -= t0

                def LL(kx, ky, en, P_, z_):
                    return float(loglik_atoms(
                        torch.tensor(np.asarray(kx, np.float32), device=dev),
                        torch.tensor(np.asarray(ky, np.float32), device=dev),
                        torch.tensor(np.asarray(en, np.float32), device=dev),
                        torch.tensor(P_, device=dev, dtype=torch.float32),
                        torch.tensor(z_, device=dev, dtype=torch.float32),
                        G.DEPTH).item())

                r0 = LI.propose_atoms(frames, G.DEPTH, nsamp=4, diffusion_seed=0,
                                      polish_iters=0)
                r8 = LI.propose_atoms(frames, G.DEPTH, nsamp=4, diffusion_seed=0,
                                      polish_iters=80)
                rec = dict(block=name, frac=float(frac), k_true=k_true,
                           n_obs=int(len(allz)), n_hold=int(len(zout)),
                           k0=float(np.median(np.hypot(r0[0], r0[1]))),
                           k80=float(np.median(np.hypot(r8[0], r8[1]))))
                for tag, r in (('p0', r0), ('p80', r8)):
                    rec[f'll_in_{tag}'] = LL(r[0], r[1], r[2], Pin_s, zin)
                    rec[f'll_out_{tag}'] = LL(r[0], r[1], r[2], Pout_s, zout)
                rec['d_in'] = rec['ll_in_p80'] - rec['ll_in_p0']
                rec['d_out'] = rec['ll_out_p80'] - rec['ll_out_p0']
                rows.append(rec)
                print(f"{os.path.basename(name):22s} frac={frac:.2f} n_obs={len(allz)} "
                      f"n_hold={len(zout)}")
                print(f"   IN : p0 {rec['ll_in_p0']:10.1f}  p80 {rec['ll_in_p80']:10.1f}  "
                      f"Δ={rec['d_in']:+10.1f}  {'↑' if rec['d_in']>0 else '↓'}")
                print(f"   OUT: p0 {rec['ll_out_p0']:10.1f}  p80 {rec['ll_out_p80']:10.1f}  "
                      f"Δ={rec['d_out']:+10.1f}  {'↑' if rec['d_out']>0 else '↓'}",
                      flush=True)

    with open(os.path.join(HERE, 'probe_lik3.jsonl'), 'a', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    up_in = sum(1 for r in rows if r['d_in'] > 0)
    up_out = sum(1 for r in rows if r['d_out'] > 0)
    print(f'\n{"="*72}\nn={len(rows)}')
    print(f'★다듬기가 **IN-sample** 우도를 올린 셀  = {up_in}/{len(rows)}')
    print(f'★다듬기가 **HELD-OUT** 우도를 올린 셀   = {up_out}/{len(rows)}')
    print('\n⛔n 이 작다(고유 블록 2개 × 개구 2). 방향만 본다 — 검정이 아니다.')


if __name__ == '__main__':
    main()
