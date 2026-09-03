"""`H3-FRF` 러너 — ⛔**추정기 런 0회**. 실물 LiDAR 반환율 대 배포 모형.

⛔셀 구성은 `tmp/frf_lidar_2026-08-01/measure_dropout.py` 의 규약을 그대로 따른다
(ring = depression 분위 32, az bin = 0.30°, 시간이 반복을 공급한다).
⛔모형은 `wavesim.raydrop_uniform` 에서 **임포트**한다 --- 재구현은 다른 것을 재는 일이다.
"""
import glob, hashlib, json, os, sys, time
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_h3 as G                                                # noqa: E402
from wavesim.raydrop_uniform import fresnel_reflectance              # noqa: E402

OUT = os.path.join(HERE, 'results_h3.jsonl')
CELLS = os.path.join(HERE, 'cells_h3.npz')


def load(path):
    with h5py.File(path, 'r') as f:
        X = np.asarray(f['xFRF']); Y = np.asarray(f['yFRF']); Z = np.asarray(f['elevation'])
        lx = float(np.asarray(f['lidarX'])); ly = float(np.asarray(f['lidarY']))
        lz = float(np.asarray(f['lidarZ']))
    # ⚠자료가 `(1, frames, points)` 로 온다 --- 선행 singleton 축을 누른다.
    X, Y, Z = np.squeeze(X), np.squeeze(Y), np.squeeze(Z)
    V = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    return X, Y, Z, V, lx, ly, lz


def analyse(path):
    X, Y, Z, V, lx, ly, lz = load(path)
    if X.ndim != 2 or X.shape[0] < G.MIN_FRAMES:
        return None
    nfr = X.shape[0]
    dx, dy, dz = X - lx, Y - ly, lz - Z
    rho = np.hypot(dx, dy)
    az = np.degrees(np.arctan2(dy, dx)) % 360.0
    dep = np.degrees(np.arctan2(dz, rho))
    rng = np.sqrt(rho ** 2 + dz ** 2)
    dv = dep[V]
    if dv.size < 5000:
        return None
    edges = np.quantile(dv, np.linspace(0, 1, G.N_RING + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    ring = np.digitize(dep, edges) - 1
    azb = (az / G.AZ_STEP).astype(int)
    key = ring.astype(np.int64) * 100000 + azb
    uk, inv = np.unique(key[V], return_inverse=True)
    fr = np.repeat(np.arange(nfr), X.shape[1]).reshape(X.shape)[V]
    hit = np.zeros((uk.size, nfr), bool); hit[inv, fr] = True
    n_hit = hit.sum(axis=1); rate = n_hit / nfr

    def cellmean(a):
        s = np.zeros(uk.size); c = np.zeros(uk.size)
        np.add.at(s, inv, a[V]); np.add.at(c, inv, 1)
        return s / np.maximum(c, 1)

    d_cell, dep_cell = cellmean(rng), cellmean(dep)
    theta = np.radians(90.0 - dep_cell)
    # ★모형 --- 임포트한 `fresnel_reflectance` 를 쓴다(재구현 아님)
    P = fresnel_reflectance(theta) * np.cos(theta) / (d_cell / G.D_REF) ** 2
    keep = (n_hit >= 1) & np.isfinite(P) & (d_cell > 1.0) & (rate > 0)
    return dict(rate=rate[keep], d=d_cell[keep], theta=np.degrees(theta[keep]),
                P=P[keep], nfr=nfr, ncell=int(keep.sum()), valid_frac=float(V.mean()))


def main():
    files = sorted(glob.glob(os.path.join(G.WS, '*.nc')))
    acc = {k: [] for k in ('rate', 'd', 'theta', 'P')}
    with open(OUT, 'w') as fh:
        for i, p in enumerate(files, 1):
            t0 = time.time()
            try:
                r = analyse(p)
            except Exception as e:                              # noqa: BLE001
                r = None; err = f'{type(e).__name__}: {e}'
            if r is None:
                fh.write(json.dumps(dict(file=os.path.basename(p),
                                         error='프레임/셀 부족 또는 예외'),
                                    ensure_ascii=False) + '\n'); fh.flush()
                print(f'  [{i}/{len(files)}] {os.path.basename(p)[:44]} ⛔건너뜀', flush=True)
                continue
            for k in acc:
                acc[k].append(r[k])
            row = dict(file=os.path.basename(p), nfr=r['nfr'], ncell=r['ncell'],
                       valid_frac=r['valid_frac'], rate_med=float(np.median(r['rate'])),
                       model_imported=True, model_md5=G.model_md5(),
                       wall_s=round(time.time() - t0, 1))
            fh.write(json.dumps(row, ensure_ascii=False) + '\n'); fh.flush()
            print(f"  [{i}/{len(files)}] {os.path.basename(p)[:44]} 셀{r['ncell']:6d} "
                  f"율{row['rate_med']:.3f} 유효{r['valid_frac']:.2f} {row['wall_s']:.1f}s",
                  flush=True)
    np.savez_compressed(CELLS, **{k: np.concatenate(v) for k, v in acc.items() if v})
    print(f'✅ 완료 — 셀 {sum(len(v) for v in acc["rate"]) if acc["rate"] else 0}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
