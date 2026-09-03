"""`T1-THIRDPARTY` v2 러너 — ⛔추정기 런 0회. 제3자 참값에 **위상 결합**이 있는가.

⛔v1 은 자료 도착 전에 폐기됐다(`AMENDMENT_1_PREDATA.md`) — PRIMARY 통계량이
`|FFT|` 만의 함수여서 위상무작위화 귀무에 대해 검정력이 **정확히 0** 이었다.
v2 의 자는 **프레임 모음 왜도**이고, 이 파일이 그것을 우리 코퍼스에서 먼저 검증한다
(`G-POWER`) — 제3자 자료와 무관하게 이빨을 보인 뒤에야 처치를 잰다.
"""
import glob, hashlib, io, json, os, re, sys, zipfile
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t1 as G                                                 # noqa: E402
sys.path.insert(0, G.SHARED)
import surrogate as S                                                # noqa: E402
sys.path.insert(0, os.path.join(G.ROOT, 'tmp', 'C1_MODELGAP_2026-09-03'))
import gates_c1 as C                                                 # noqa: E402

OUT = os.path.join(HERE, 'results_t1.jsonl'); META = os.path.join(HERE, 'meta_t1.json')


def q95(a):
    """nearest-rank 백분위 — 결과 전 고정한 규약."""
    return float(np.sort(np.asarray(a))[int(np.ceil(G.Q * len(a))) - 1])


def block_test(frames, rng):
    """한 블록(프레임 목록)의 왜도·마루골비를 대체표본 귀무에 대해 검정."""
    pool = np.concatenate([f.ravel() for f in frames])
    out = {}
    for nm, fn in (('skew', S.skewness), ('ct', S.crest_trough)):
        v = float(fn(pool))
        null = [float(fn(np.concatenate(
            [S.ft_surrogate2d(f, rng).ravel() for f in frames]))) for _ in range(G.N_SUR)]
        out[nm] = v
        out[nm + '_q95'] = q95(null)
        out[nm + '_null_mean'] = float(np.mean(null))
        out[nm + '_exceed'] = bool(v > q95(null))
    # ⚠v1 의 자를 **보고 전용**으로 함께 싣는다(검정력 0 임을 같은 행에 적는다).
    z0 = frames[0]
    F = np.fft.fft2(z0)
    sur0 = S.ft_surrogate2d(z0, np.random.default_rng(G.SEED))
    out['null_fft_relerr'] = float(
        np.max(np.abs(np.abs(np.fft.fft2(sur0)) - np.abs(F))) / max(np.abs(F).max(), 1e-30))
    return out


def our_blocks(paths, limit):
    """우리 동결 코퍼스에서 같은 절차로 블록을 만든다(`G-POWER` 용)."""
    rows = []
    for p in paths[:limit]:
        with h5py.File(p, 'r') as f:
            Z = np.asarray(f['frames/gt_wave_field'])
        idx = np.linspace(0, Z.shape[0] - 1, min(G.N_FRAME, Z.shape[0])).astype(int)
        r = block_test([Z[j] for j in idx], np.random.default_rng(G.SEED))
        r.update(block=os.path.basename(p), n_frames=len(idx))
        rows.append(r)
    return rows


def norun_check():
    pat = re.compile(r'^\s*(from|import)\s+wavesplat\b', re.M)
    hits = [fn for fn in ('run_t1.py', 'analyze_t1.py', 'gates_t1.py')
            if pat.search(open(os.path.join(HERE, fn), encoding='utf-8').read())]
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


def main(limit=None):
    rows = []
    # ── 대조 (우리 자료) ───────────────────────────────────────────────────
    for r in our_blocks(sorted(glob.glob(os.path.join(C.HOS, '*.h5'))), 8):
        r['group'] = 'ctrl_hos'; rows.append(r)
    for r in our_blocks(sorted(glob.glob(os.path.join(
            C.LIN, 'episode_BF*comp_seed*.h5'))), 8):
        r['group'] = 'ctrl_linear'; rows.append(r)

    # ── 처치 (제3자) ──────────────────────────────────────────────────────
    size = os.path.getsize(G.ZIP)
    solver_named, doc_hits = False, []
    with zipfile.ZipFile(G.ZIP) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.mat'))
        for d in [n for n in z.namelist() if not n.endswith('/')
                  and os.path.splitext(n)[1].lower() in ('.txt', '.md', '.pdf', '.json')]:
            txt = z.read(d)[:200000].decode('utf-8', 'replace').lower()
            for kw in ('hos', 'higher-order spectral', 'boundary element',
                       'potential flow', 'stokes'):
                if kw in txt:
                    solver_named = True; doc_hits.append({'file': d, 'kw': kw})
        picked = names[:(limit or G.N_BLOCK)]
        for n in picked:
            with h5py.File(io.BytesIO(z.read(n)), 'r') as f:
                E = np.asarray(f['eta']).squeeze()
                x = np.asarray(f['x']).squeeze(); t = np.asarray(f['t']).squeeze()
            idx = np.linspace(0, E.shape[0] - 1, G.N_FRAME).astype(int)
            r = block_test([E[j] for j in idx], np.random.default_rng(G.SEED))
            r.update(block=n, group='tp', n_frames=len(idx),
                     shape=list(E.shape), dx=float(x[1] - x[0]), dt=float(t[1] - t[0]),
                     sigma=float(E.std()))
            rows.append(r)
            print(f"  {os.path.basename(n):22s} skew {r['skew']:+.4f} "
                  f"q95 {r['skew_q95']:+.4f} {'초과' if r['skew_exceed'] else '  -'}")
    with open(OUT, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    ok, msg = norun_check()
    json.dump({'zip_size': size, 'size_ok': size == G.WANT_BYTES,
               'solver_named': solver_named, 'solver_hits': doc_hits,
               'n_mat_total': len(names), 'picked': picked,
               'norun': ok, 'norun_msg': msg,
               'shared_md5': hashlib.md5(
                   open(os.path.join(G.SHARED, 'surrogate.py'), 'rb').read()).hexdigest(),
               'seed': G.SEED, 'n_sur': G.N_SUR, 'n_frame': G.N_FRAME},
              open(META, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
