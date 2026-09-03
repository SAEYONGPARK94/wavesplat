"""`C1b-BESTFIT` 러너 — ⛔추정기 런 0회. 파수 고정, 진폭·위상만 최소제곱 재적합."""
import glob, json, os, re, sys
import numpy as np, h5py
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_c1b as G                                                # noqa: E402
OUT = os.path.join(HERE, 'results_c1b.jsonl'); META = os.path.join(HERE, 'meta_c1b.json')
C1V = os.path.join(G.C1, 'verdict_c1.json')


def measure(path, group, c1rel):
    with h5py.File(path, 'r') as f:
        C = np.asarray(f['config/wave_components'])
        gx = np.asarray(f['frames/gt_grid_x']); gy = np.asarray(f['frames/gt_grid_y'])
        Z = np.asarray(f['frames/gt_wave_field'])
    X, Y = np.meshgrid(gx, gy)
    kx, ky = C[:, 2], C[:, 3]
    th = kx[None, :] * X.ravel()[:, None] + ky[None, :] * Y.ravel()[:, None]
    B = np.hstack([np.cos(th), np.sin(th)])                 # (npix, 2N)
    mx = 0.0
    for j in range(0, Z.shape[0], G.STRIDE):
        z = Z[j].ravel()
        c, *_ = np.linalg.lstsq(B, z, rcond=None)
        mx = max(mx, float(np.max(np.abs(B @ c - z))))
    sg = float(np.std(Z))
    name = os.path.basename(path)
    return dict(episode=name, group=group, crowded=bool(G.is_crowded(name)),
                n_comp=int(C.shape[0]), max_abs_delta_ls=mx, sigma=sg,
                rel_ls=(mx / sg if sg > 0 else float('nan')), rel_gap_c1=c1rel)


def norun_check():
    pat = re.compile(r'^\s*(from|import)\s+wavesplat\b', re.M)
    hits = [fn for fn in ('run_c1b.py', 'analyze_c1b.py', 'gates_c1b.py')
            if pat.search(open(os.path.join(HERE, fn), encoding='utf-8').read())]
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


def main():
    c1 = json.load(open(C1V, encoding='utf-8'))['per_episode']
    files = [(p, 'hos') for p in sorted(glob.glob(os.path.join(G.HOS, '*.h5')))]
    files += [(p, 'linear') for p in
              sorted(glob.glob(os.path.join(G.LIN, 'episode_BF*comp_seed*.h5')))[:12]]
    miss = []
    with open(OUT, 'w', encoding='utf-8') as fh:
        for i, (p, grp) in enumerate(files, 1):
            nm = os.path.basename(p)
            if nm not in c1:
                miss.append(nm); continue
            try:
                r = measure(p, grp, float(c1[nm]['rel_gap']))
            except Exception as e:                                   # noqa: BLE001
                r = dict(episode=nm, group=grp, error=f'{type(e).__name__}: {e}')
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            if i % 16 == 0:
                print(f"  [{i}/{len(files)}] {nm[:44]:46s} rel_ls {r.get('rel_ls', float('nan')):.4f}"
                      f" (C1 {r.get('rel_gap_c1', float('nan')):.4f})", flush=True)
    nok, nmsg = norun_check()
    json.dump(dict(paired=not miss, pair_msg=(f'미매칭 {len(miss)}' if miss else '전건 맺힘'),
                   norun=nok, norun_msg=nmsg), open(META, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'✅ 완료 · 맺힘 {"전건" if not miss else miss[:3]} · {nmsg}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
