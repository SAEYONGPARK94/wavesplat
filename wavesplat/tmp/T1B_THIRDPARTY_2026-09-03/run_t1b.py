"""`T1b` 러너 — ⛔추정기 런 0회. `T1` 의 자를 **502 프레임**으로 신선 블록에 건다.

⛔`T1` 의 `block_test`·`our_blocks` 를 **import 한다**(사본 재구현 금지 — 이 저장소가
`R2B`→`R2C` 에서 대가를 치른 부류). 프레임 수만 `gates_t1b.N_FRAME` 로 바뀐다.
"""
import io, json, os, re, sys, time, zipfile
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t1b as G                                                # noqa: E402
sys.path.insert(0, G.SHARED)
sys.path.insert(0, os.path.join(G.ROOT, 'tmp', 'C1_MODELGAP_2026-09-03'))
import gates_c1 as C                                                 # noqa: E402
import run_t1 as R1                                                  # noqa: E402

OUT = os.path.join(HERE, 'results_t1b.jsonl'); META = os.path.join(HERE, 'meta_t1b.json')


def our_blocks(paths, limit):
    """대조는 `T1` 과 **같은 프레임 수**로 둔다 — `G-POWER` 문턱이 그 위에서 정해졌다."""
    return R1.our_blocks(paths, limit)


def tp_blocks(z, names, nframe, tag):
    rows = []
    for n in names:
        t0 = time.time()
        with h5py.File(io.BytesIO(z.read(n)), 'r') as f:
            E = np.asarray(f['eta']).squeeze()
        idx = np.linspace(0, E.shape[0] - 1, min(nframe, E.shape[0])).astype(int)
        r = R1.block_test([E[j] for j in idx], np.random.default_rng(G.SEED))
        r.update(block=n, group=tag, n_frames=len(idx), shape=list(E.shape),
                 sigma=float(E.std()), wall_s=round(time.time() - t0, 1))
        rows.append(r)
        print(f"  [{tag}] {os.path.basename(n):20s} skew {r['skew']:+.4f} "
              f"q95 {r['skew_q95']:+.4f} {'초과' if r['skew_exceed'] else '  -'} "
              f"({r['wall_s']}s)", flush=True)
    return rows


def norun_check():
    pat = re.compile(r'^\s*(from|import)\s+wavesplat\b', re.M)
    hits = [fn for fn in ('run_t1b.py', 'analyze_t1b.py', 'gates_t1b.py')
            if pat.search(open(os.path.join(HERE, fn), encoding='utf-8').read())]
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


def main(limit=None):
    rows = []
    for r in our_blocks(sorted(__import__('glob').glob(os.path.join(C.HOS, '*.h5'))), 8):
        r['group'] = 'ctrl_hos'; rows.append(r)
    for r in our_blocks(sorted(__import__('glob').glob(os.path.join(
            C.LIN, 'episode_BF*comp_seed*.h5'))), 8):
        r['group'] = 'ctrl_linear'; rows.append(r)
    print(f'  대조 완료 ({len(rows)} 행)', flush=True)

    n = limit or G.N_BLOCK
    with zipfile.ZipFile(G.ZIP) as z:
        names = sorted(x for x in z.namelist() if x.lower().endswith('.mat'))
        fresh = names[G.SKIP:G.SKIP + n]
        same = names[:n]
        rows += tp_blocks(z, fresh, G.N_FRAME, 'fresh')
        rows += tp_blocks(z, same, G.N_FRAME, 'same')
    with open(OUT, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    t1meta = json.load(open(os.path.join(
        HERE, '..', 'T1_THIRDPARTY_2026-09-03', 'meta_t1.json'), encoding='utf-8'))
    ok, msg = norun_check()
    import hashlib
    json.dump({'zip_size': os.path.getsize(G.ZIP),
               'size_ok': os.path.getsize(G.ZIP) == G.WANT_BYTES,
               'fresh': fresh, 'same': same, 't1_picked': t1meta['picked'],
               'disjoint': sorted(set(fresh) & set(t1meta['picked'])),
               'solver_named': t1meta['solver_named'],
               'n_frame': G.N_FRAME, 'n_sur': G.N_SUR, 'seed': G.SEED,
               'norun': ok, 'norun_msg': msg,
               'shared_md5': hashlib.md5(
                   open(os.path.join(G.SHARED, 'surrogate.py'), 'rb').read()).hexdigest()},
              open(META, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
