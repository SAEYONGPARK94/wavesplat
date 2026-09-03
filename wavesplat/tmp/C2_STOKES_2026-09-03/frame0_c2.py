"""`C2` 개정 1 의 측정 — **t=0 프레임이 선험적으로 지는 칸인가**. ⛔추정기 런 0회.

적대적 재검의 지적: HOS 참값이 **선형 초기면**으로 시작하므로(`hos_simulator.py:402-429`)
t=0 에는 구속 고조파가 **없다**. 그런데 `run_c2.py:40` 이 `j=0` 부터 센다 ⟹ 그 프레임은
「2차항을 더하면 반드시 나빠지는」 칸이고, 답을 결과 전에 알 수 있었다.
"""
import glob, json, os, sys
import numpy as np, h5py, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import gates_c2 as G                                                 # noqa: E402
from wavesplat.model.stokes2 import Stokes2Transfer                  # noqa: E402
from wavesim.config import EpisodeCfg                                # noqa: E402

DEPTH = float(EpisodeCfg().depth)


def measure(path):
    with h5py.File(path, 'r') as f:
        C = np.asarray(f['config/wave_components'])
        gx = np.asarray(f['frames/gt_grid_x']); gy = np.asarray(f['frames/gt_grid_y'])
        Z = np.asarray(f['frames/gt_wave_field']); t = np.asarray(f['frames/timestamps'])
    X, Y = np.meshgrid(gx, gy)
    A, w, kx, ky, ph = C[:, 0], C[:, 1], C[:, 2], C[:, 3], C[:, 4]
    xt = torch.tensor(X.ravel()); yt = torch.tensor(Y.ravel())
    At = torch.tensor(A); kxt = torch.tensor(kx); kyt = torch.tensor(ky)
    pht = torch.tensor(ph); wt = torch.tensor(w)
    tr = Stokes2Transfer(kx, ky, DEPTH)      # ⛔numpy — run_c2.py 와 같은 호출
    sg = float(np.std(Z)); out = []
    for j in range(0, Z.shape[0], G.STRIDE):
        z = Z[j].ravel()
        lin = np.zeros_like(z)
        for i in range(len(A)):
            lin += A[i] * np.cos(kx[i] * X.ravel() + ky[i] * Y.ravel() - w[i] * t[j] + ph[i])
        r = lin - z
        c = tr.correction(xt, yt, float(t[j]), At, kxt, kyt, pht, wt).cpu().numpy()
        out.append(dict(j=int(j), t=float(t[j]),
                        r=float(np.sqrt((r ** 2).mean()) / sg),
                        c=float(np.sqrt((c ** 2).mean()) / sg),
                        rc=float(np.sqrt(((r + c) ** 2).mean()) / sg)))
    return out


def main():
    files = sorted(glob.glob(os.path.join(G.HOS, '*.h5')))
    rows = []
    for p in files:
        nm = os.path.basename(p)
        if not G.is_crowded(nm):
            continue
        for d in measure(p):
            d['episode'] = nm; rows.append(d)
    json.dump(rows, open(os.path.join(HERE, 'frame0_c2.json'), 'w'), indent=1)

    def tally(skip0):
        eps = sorted({r['episode'] for r in rows}); w = 0
        for e in eps:
            s = [r for r in rows if r['episode'] == e and not (skip0 and r['j'] == 0)]
            a = np.sqrt(np.mean([r['r'] ** 2 for r in s]))
            b = np.sqrt(np.mean([r['rc'] ** 2 for r in s]))
            w += (b < a)
        return w, len(eps)
    f0 = [r for r in rows if r['j'] == 0]
    print(f"t=0 프레임: |잔차| 중앙 {np.median([r['r'] for r in f0]):.4f} · "
          f"|보정| 중앙 {np.median([r['c'] for r in f0]):.4f}  ⟹ 선험적 손해")
    print('전 프레임 포함 :', tally(False), '(원 판정)')
    print('t=0 제외      :', tally(True))


if __name__ == '__main__':
    main()
