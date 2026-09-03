"""`C1-MODELGAP` 러너 — ⛔**추정기 런 0회**. 성분표로 참값을 재구성해 잔차만 잰다."""
import glob
import json
import os
import re
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_c1 as G                                                 # noqa: E402

OUT = os.path.join(HERE, 'results_c1.jsonl')
META = os.path.join(HERE, 'meta_c1.json')


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
    mx = 0.0
    for j in range(0, Z.shape[0], G.STRIDE):
        Zr = np.zeros_like(Z[j])
        for i in range(len(A)):
            Zr += A[i] * np.cos(kx[i] * X + ky[i] * Y - w[i] * t[j] + ph[i])
        mx = max(mx, float(np.max(np.abs(Zr - Z[j]))))
    sg = float(np.std(Z))
    name = os.path.basename(path)
    return dict(episode=name, group=group, crowded=bool(G.is_crowded(name)),
                n_comp=int(C.shape[0]), max_abs_delta=mx, sigma=sg,
                rel_gap=(mx / sg if sg > 0 else float('nan')),
                mesh=(mesh.decode() if isinstance(mesh, bytes) else
                      (str(mesh) if mesh is not None else None)),
                n_frames=int(Z.shape[0]), stride=G.STRIDE)


def norun_check():
    pat = re.compile(r'^\s*(from|import)\s+(wavesplat)\b', re.M)
    hits = [fn for fn in ('run_c1.py', 'analyze_c1.py', 'gates_c1.py')
            if pat.search(open(os.path.join(HERE, fn), encoding='utf-8').read())]
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


def main():
    files = [(p, 'hos') for p in sorted(glob.glob(os.path.join(G.HOS, '*.h5')))]
    lin = sorted(glob.glob(os.path.join(G.LIN, 'episode_BF*comp_seed*.h5')))
    files += [(p, 'linear') for p in lin[:12]]
    with open(OUT, 'w', encoding='utf-8') as fh:
        for i, (p, grp) in enumerate(files, 1):
            try:
                r = measure(p, grp)
            except Exception as e:                                   # noqa: BLE001
                r = dict(episode=os.path.basename(p), group=grp,
                         error=f'{type(e).__name__}: {e}')
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            if i % 16 == 0 or 'error' in r:
                print(f"  [{i}/{len(files)}] {r['episode'][:44]:46s} "
                      f"{'⛔' + r['error'][:50] if 'error' in r else f'rel {r['rel_gap']:.4f}'}",
                      flush=True)
    nok, nmsg = norun_check()
    json.dump(dict(cols=list(G.cols_from_source() or ()), norun=nok, norun_msg=nmsg,
                   n_files=len(files)),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'✅ 완료 {len(files)} · 열규약 {G.cols_from_source()} · {nmsg}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
