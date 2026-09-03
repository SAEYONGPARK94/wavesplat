"""`C1c-BESTFIT` 러너 — ⛔추정기 런 0회. 파수 고정, 진폭·위상만 LS 재적합.

★자는 **RMS** 다(선행 `C1b` 가 `max` 로 `G-LE` 를 적었다가 차단됐다 — LS 는 $L_2$ 를
최소화하지 $\\max$ 를 최소화하지 않는다). `max` 는 보고 전용으로 함께 싣는다.
"""
import glob, json, os, re, sys
import numpy as np, h5py
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_c1c as G                                                # noqa: E402
OUT = os.path.join(HERE, 'results_c1c.jsonl'); META = os.path.join(HERE, 'meta_c1c.json')


def measure(path, group):
    with h5py.File(path, 'r') as f:
        C = np.asarray(f['config/wave_components'])
        gx = np.asarray(f['frames/gt_grid_x']); gy = np.asarray(f['frames/gt_grid_y'])
        Z = np.asarray(f['frames/gt_wave_field']); t = np.asarray(f['frames/timestamps'])
    X, Y = np.meshgrid(gx, gy)
    A0, w0, kx, ky, ph0 = C[:, 0], C[:, 1], C[:, 2], C[:, 3], C[:, 4]
    th = kx[None, :] * X.ravel()[:, None] + ky[None, :] * Y.ravel()[:, None]
    B = np.hstack([np.cos(th), np.sin(th)])
    se_ls = se_st = 0.0; n = 0; mx_ls = mx_st = 0.0
    for j in range(0, Z.shape[0], G.STRIDE):
        z = Z[j].ravel()
        # 저장 계수 그대로 (= `C1` 의 재구성)
        zs = np.zeros_like(z)
        for i in range(len(A0)):
            zs += A0[i] * np.cos(kx[i] * X.ravel() + ky[i] * Y.ravel() - w0[i] * t[j] + ph0[i])
        d_st = zs - z
        c, *_ = np.linalg.lstsq(B, z, rcond=None)
        d_ls = B @ c - z
        se_st += float(d_st @ d_st); se_ls += float(d_ls @ d_ls); n += z.size
        mx_st = max(mx_st, float(np.max(np.abs(d_st))))
        mx_ls = max(mx_ls, float(np.max(np.abs(d_ls))))
    sg = float(np.std(Z))
    name = os.path.basename(path)
    return dict(episode=name, group=group, crowded=bool(G.is_crowded(name)),
                n_comp=int(C.shape[0]), sigma=sg,
                rms_stored=float(np.sqrt(se_st / n) / sg),
                rms_ls=float(np.sqrt(se_ls / n) / sg),
                max_stored=float(mx_st / sg), max_ls=float(mx_ls / sg))


def norun_check():
    pat = re.compile(r'^\s*(from|import)\s+wavesplat\b', re.M)
    hits = [fn for fn in ('run_c1c.py', 'analyze_c1c.py', 'gates_c1c.py')
            if pat.search(open(os.path.join(HERE, fn), encoding='utf-8').read())]
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


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
            if i % 16 == 0:
                print(f"  [{i}/{len(files)}] {r['episode'][:42]:44s} "
                      f"rms {r.get('rms_stored', float('nan')):.4f} -> {r.get('rms_ls', float('nan')):.4f}",
                      flush=True)
    nok, nmsg = norun_check()
    json.dump(dict(norun=nok, norun_msg=nmsg, paired=True, pair_msg='C1 불필요(자체 측정)'),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'✅ 완료 · {nmsg}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
