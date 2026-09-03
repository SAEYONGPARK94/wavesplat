"""`C2-STOKES` 러너 — ⛔**추정기 런 0회 · 적합 0회**. 이론 2차 항을 **계산**해서 더한다.

⛔`Stokes2Transfer` 는 `wavesplat.model.stokes2` 에서 **임포트**한다(재구현 금지).
"""
import glob, json, os, sys
import numpy as np, h5py, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_c2 as G                                                 # noqa: E402
from wavesplat.model.stokes2 import Stokes2Transfer                  # noqa: E402
OUT = os.path.join(HERE, 'results_c2.jsonl'); META = os.path.join(HERE, 'meta_c2.json')

# 에피소드 파일명에는 수심이 없다 — 생성기 기본값을 단일출처에서 읽는다.
from wavesim.config import EpisodeCfg                                # noqa: E402
DEPTH = float(EpisodeCfg().depth)   # ⛔단일출처(wavesim/config.py) — 손으로 적지 않는다


def measure(path, group):
    with h5py.File(path, 'r') as f:
        C = np.asarray(f['config/wave_components'])
        gx = np.asarray(f['frames/gt_grid_x']); gy = np.asarray(f['frames/gt_grid_y'])
        Z = np.asarray(f['frames/gt_wave_field']); t = np.asarray(f['frames/timestamps'])
        mesh = None
        try:
            mesh = f['config'].attrs.get('mesh_verdict')
        except Exception:                                            # noqa: BLE001
            pass
    X, Y = np.meshgrid(gx, gy)
    A, w, kx, ky, ph = C[:, 0], C[:, 1], C[:, 2], C[:, 3], C[:, 4]
    tr = Stokes2Transfer(kx, ky, DEPTH)                 # ⛔계산, 적합 아님
    xt = torch.tensor(X.ravel(), dtype=torch.float32)
    yt = torch.tensor(Y.ravel(), dtype=torch.float32)
    At = torch.tensor(A, dtype=torch.float32); kxt = torch.tensor(kx, dtype=torch.float32)
    kyt = torch.tensor(ky, dtype=torch.float32); pht = torch.tensor(ph, dtype=torch.float32)
    wt = torch.tensor(w, dtype=torch.float32)
    se_l = se_s = 0.0; n = 0
    for j in range(0, Z.shape[0], G.STRIDE):
        z = Z[j].ravel()
        lin = np.zeros_like(z)
        for i in range(len(A)):
            lin += A[i] * np.cos(kx[i] * X.ravel() + ky[i] * Y.ravel() - w[i] * t[j] + ph[i])
        with torch.no_grad():
            z2 = tr.correction(xt, yt, float(t[j]), At, kxt, kyt, pht, wt).cpu().numpy()
        d_l = lin - z; d_s = (lin + z2) - z
        se_l += float(d_l @ d_l); se_s += float(d_s @ d_s); n += z.size
    sg = float(np.std(Z))
    return dict(episode=os.path.basename(path), group=group,
                crowded=bool(G.is_crowded(os.path.basename(path))),
                n_comp=int(C.shape[0]), sigma=sg, depth=DEPTH,
                rms_lin=float(np.sqrt(se_l / n) / sg),
                rms_st2=float(np.sqrt(se_s / n) / sg),
                mesh=(mesh.decode() if isinstance(mesh, bytes) else
                      (str(mesh) if mesh is not None else None)))


def main():
    files = [(p, 'hos') for p in sorted(glob.glob(os.path.join(G.HOS, '*.h5')))]
    files += [(p, 'linear') for p in
              sorted(glob.glob(os.path.join(G.LIN, 'episode_BF*comp_seed*.h5')))[:12]]
    with open(OUT, 'w', encoding='utf-8') as fh:
        for i, (p, grp) in enumerate(files, 1):
            try:
                r = measure(p, grp)
            except Exception as e:                                   # noqa: BLE001
                r = dict(episode=os.path.basename(p), group=grp,
                         error=f'{type(e).__name__}: {e}')
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            if i % 16 == 0 or 'error' in r:
                msg = ('⛔' + r['error'][:60] if 'error' in r
                       else 'lin %.4f -> st2 %.4f' % (r['rms_lin'], r['rms_st2']))
                print('  [%d/%d] %-42s %s' % (i, len(files), r['episode'][:40], msg), flush=True)
    json.dump(dict(depth=DEPTH, n_files=len(files)),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'✅ 완료 {len(files)} · depth {DEPTH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
