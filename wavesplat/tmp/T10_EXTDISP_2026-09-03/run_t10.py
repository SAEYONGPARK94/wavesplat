"""`T10-EXTDISP` 러너 — ⛔추정기 런 0회. **입장 검사**: 심해로 다뤄도 되는가.

⛔사전등록(`PREREG.md`)이 동결된 뒤 처음으로 제3자 `eta` 에 3D 스펙트럼 자를 건다.
`G-POWER` 는 **그날** 같은 자로 참 수심 합성을 재실행해 이빨을 확인한다.
"""
import hashlib, io as _io, json, os, re, sys, time, zipfile
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t10 as G                                                # noqa: E402
from ridge2_t10 import ridge2, fit                                   # noqa: E402
import control_t10 as CT                                             # noqa: E402

OUT = os.path.join(HERE, 'results_t10.jsonl'); META = os.path.join(HERE, 'meta_t10.json')


def mode_counts(shape, dx, dy):
    Ny, Nx = shape
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy); kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.sqrt(KX ** 2 + KY ** 2)
    e = np.linspace(G.KMIN, G.KMAX, G.NBIN + 1)
    return [int(((K >= a) & (K < b)).sum()) for a, b in zip(e[:-1], e[1:])]


def measure(E, dx, dy, dt, tag):
    t0 = time.time()
    k, om, wt = ridge2(E, dx, dy, dt, kmin=G.KMIN, kmax=G.KMAX, nbin=G.NBIN,
                       min_modes=G.MIN_MODES)
    d, rel, rel_deep = fit(k, om, wt)
    mc = mode_counts(E.shape[1:], dx, dy)
    return dict(arm=tag, n_ridge=int(len(k)), min_modes_used=int(min(mc)),
                ratio=float(rel / rel_deep) if rel_deep > 0 else float('nan'),
                rel=float(rel), rel_deep=float(rel_deep), fit_depth=float(d),
                dx=float(dx), dy=float(dy), dt=float(dt),
                kmin=G.KMIN, kmax=G.KMAX, wall_s=round(time.time() - t0, 1))


def norun_check():
    pat = re.compile(r'^\s*(from|import)\s+wavesplat\b', re.M)
    hits = [f for f in ('run_t10.py', 'analyze_t10.py', 'gates_t10.py', 'ridge2_t10.py')
            if os.path.exists(os.path.join(HERE, f))
            and pat.search(open(os.path.join(HERE, f), encoding='utf-8').read())]
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


def main(limit=None):
    rows = []
    # ── G-POWER: 같은 자로 참 수심 합성을 **그날** 재실행 ──────────────────
    for d in list(G.POWER_SHALLOW) + [G.POWER_DEEP]:
        E = CT.synth(d, n_comp=200, seed=int(d))
        r = measure(E, CT.DX, CT.DY, CT.DT, 'power')
        r.update(true_depth=float(d),
                 detected=bool(r['ratio'] < G.THRESH))
        rows.append(r)
        print(f"  [power] d={d:6.1f}  ratio {r['ratio']:.4f}  "
              f"{'검출' if r['detected'] else '  -'} ({r['wall_s']}s)", flush=True)

    # ── 처치 ──────────────────────────────────────────────────────────────
    size = os.path.getsize(G.ZIP)
    with zipfile.ZipFile(G.ZIP) as z:
        names = sorted(x for x in z.namelist() if x.lower().endswith('.mat'))
        picked = names[G.SKIP:G.SKIP + (limit or G.N_FILE)]
        for n in picked:
            with h5py.File(_io.BytesIO(z.read(n)), 'r') as f:
                E = np.asarray(f['eta']).squeeze()
                x = np.asarray(f['x']).squeeze(); y = np.asarray(f['y']).squeeze()
                t = np.asarray(f['t']).squeeze()
            r = measure(E, float(x[1] - x[0]), float(y[1] - y[0]), float(t[1] - t[0]), 'tp')
            r.update(block=n, shape=list(E.shape), sigma=float(E.std()),
                     shallow=bool(r['ratio'] < G.THRESH))
            rows.append(r)
            print(f"  [tp] {os.path.basename(n):20s} ratio {r['ratio']:.4f} "
                  f"{'⟸ 얕다' if r['shallow'] else ''} (능선 {r['n_ridge']}, "
                  f"{r['wall_s']}s)", flush=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    t1 = json.load(open(os.path.join(HERE, '..', 'T1_THIRDPARTY_2026-09-03',
                                     'meta_t1.json'), encoding='utf-8'))
    t1b = json.load(open(os.path.join(HERE, '..', 'T1B_THIRDPARTY_2026-09-03',
                                      'meta_t1b.json'), encoding='utf-8'))
    prev = set(t1['picked']) | set(t1b['fresh'])
    ok, msg = norun_check()
    json.dump(dict(zip_size=size, size_ok=size == G.WANT_BYTES, picked=picked,
                   disjoint=sorted(set(picked) & prev), threshold=G.THRESH,
                   null_md5=hashlib.md5(
                       open(os.path.join(HERE, 'null_t10.json'), 'rb').read()).hexdigest(),
                   norun=ok, norun_msg=msg),
              open(META, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
