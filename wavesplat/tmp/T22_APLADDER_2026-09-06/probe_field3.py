r"""`T22` **프로브 12** — 프로브 11 을 **제3자 코퍼스**에서 (그 프로브가 스스로 남긴 칸).

⛔판정이 아니다. 적합기 미실행 — `learned_init.raw_field_nrmse`(LS w/phi, 결합적합 없음).
노출: `gates_t11.SMOKE_FILES` 2블록만.

## 왜
`PROBE_11.md §4-2` 가 스스로 적었다: *"제3자 코퍼스에서는 안 쟀다 — 거기서 배포 팔은
무기술이므로 그림이 다를 수 있다. ★그것이 남은 유일하게 흥미로운 칸이다."*
홈에서는 다듬기가 장을 17/20 에서 좋게 만들었다. **무기술 영역에서도 그런가?**
"""
import json, os, sys, warnings, zipfile
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
    import torch; torch.set_num_threads(1)
    import gates_t11 as G, run_t11 as R11, probe_geom as PG
    from wavesplat.optimizer import learned_init as LI

    ITERS = [0, 80]
    rows = []
    print(f"채점 = raw_field_nrmse (LS w/phi, 결합적합 없음) · iters={ITERS}")
    print(f"{'block':22s}{'frac':>6s}{'k_true':>9s}" +
          ''.join(f"{'nRMSE@'+str(i):>12s}" for i in ITERS) + f"{'Δ(80-0)':>10s}")
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, name in enumerate(G.SMOKE_FILES, 1):
            E, gx, gy, tarr = R11.read_mat(zf, name)
            Ew, tw = E[G.J0:G.J0 + int(G.WIN_S)], tarr[G.J0:G.J0 + int(G.WIN_S)]
            k_true = PG.true_peak_k(Ew[-1], gx, gy)
            gt = np.asarray(Ew[-1], float)
            Hs = 4.0 * float(np.std(Ew))
            for frac in (1.0, 0.5, 0.25):
                rng = np.random.default_rng(G.SEED + fi)
                x, y, tt, zz = PG.sample_sub(Ew, gx, gy, tw, G.DENSITY, rng, frac)
                frames = R11.build_frames(x, y, tt, zz, G.GT_CONST)
                tq = float(tt.max())
                vals = []
                for it in ITERS:
                    cfg = {'learned_diffusion_seed': 0, 'learned_polish_iters': it}
                    vals.append(float(LI.raw_field_nrmse(
                        cfg, frames, G.DEPTH, gt, gx, gy, tq, Hs, seed=0)))
                d = vals[-1] - vals[0]
                print(f'{os.path.basename(name):22s}{frac:6.2f}{k_true:9.4f}'
                      + ''.join(f'{v:12.4f}' for v in vals) + f'{d:+10.4f}', flush=True)
                rows.append(dict(block=name, frac=float(frac), k_true=k_true,
                                 Hs=Hs, iters=ITERS, nrmse=vals, d_80_0=d))
    with open(os.path.join(HERE, 'probe_field3.jsonl'), 'a', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    d = np.array([r['d_80_0'] for r in rows])
    print(f'\nn={len(d)} · 다듬기가 장을 나쁘게 한 셀 = {int((d>0).sum())}/{len(d)}'
          f' · Δ 중앙 {np.median(d):+.4f}')
    print('★대조 (홈, 프로브 11): 나쁘게 한 에피소드 3/20 · Δ 중앙 −0.0691')
    print('⛔n=6(고유 블록 2) — 방향만 본다.')


if __name__ == '__main__':
    main()
