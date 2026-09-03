"""`T10` 자 검증 — **3D 스펙트럼 능선에서 유효 수심을 복원할 수 있는가**.

⛔제3자 자료를 열기 **전에** 우리 코퍼스(수심 **50 m** 로 생성됨)에서 자를 검증한다.
★`T1b` 의 교훈: 양성 대조가 통과하기 전에는 처치를 재지 않는다.

방법: `eta(t,y,x)` 의 3D FFT → `(ω, kx, ky)` 에너지. `|k|` 띠마다 에너지 최대인 `ω` 를
능선으로 잡고 `ω = √(g|k| tanh(|k|d))` 를 `d` 하나로 최소제곱 적합한다.
"""
import glob, json, os, sys
import numpy as np, h5py

G_ACC = 9.80665
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'C1_MODELGAP_2026-09-03'))
import gates_c1 as C                                                 # noqa: E402


def ridge(E, dx, dy, dt, kmin=0.02, kmax=0.40, min_bins=8):
    """`|k|` 띠별 에너지 최대 `ω`. 반환 (k, omega, weight)."""
    T, Ny, Nx = E.shape
    F = np.fft.fftn(E - E.mean())
    P = np.abs(F) ** 2
    w = 2 * np.pi * np.fft.fftfreq(T, dt)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.sqrt(KX ** 2 + KY ** 2)
    pos = w > 0                                    # ω>0 반평면만(실수장 대칭)
    P, w = P[pos], w[pos]
    edges = np.linspace(kmin, kmax, 25)
    ks, ws, wt = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (K >= a) & (K < b)
        if m.sum() < min_bins:
            continue
        prof = P[:, m].sum(axis=1)
        if prof.sum() <= 0:
            continue
        j = int(np.argmax(prof))
        ks.append(0.5 * (a + b)); ws.append(float(w[j])); wt.append(float(prof[j]))
    return np.array(ks), np.array(ws), np.array(wt)


def fit_depth(k, om, wt):
    """`ω = √(g k tanh(k d))` 를 `d` 하나로 적합(가중 최소제곱, 격자 탐색+이분)."""
    def sse(d):
        pred = np.sqrt(G_ACC * k * np.tanh(k * d))
        return float(np.sum(wt * (pred - om) ** 2) / np.sum(wt))
    ds = np.geomspace(0.5, 2000.0, 400)
    s = [sse(d) for d in ds]
    i = int(np.argmin(s))
    lo, hi = ds[max(0, i - 1)], ds[min(len(ds) - 1, i + 1)]
    for _ in range(80):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if sse(m1) < sse(m2):
            hi = m2
        else:
            lo = m1
    d = 0.5 * (lo + hi)
    pred = np.sqrt(G_ACC * k * np.tanh(k * d))
    rel = float(np.sqrt(np.sum(wt * (pred - om) ** 2) / np.sum(wt)) / np.mean(om))
    deep = np.sqrt(G_ACC * k)
    rel_deep = float(np.sqrt(np.sum(wt * (deep - om) ** 2) / np.sum(wt)) / np.mean(om))
    return d, rel, rel_deep


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(C.HOS, '*.h5')))[:6]:
        with h5py.File(p, 'r') as f:
            Z = np.asarray(f['frames/gt_wave_field'])
            gx = np.asarray(f['frames/gt_grid_x']); gy = np.asarray(f['frames/gt_grid_y'])
            t = np.asarray(f['frames/timestamps'])
            depth = float(f['config'].attrs.get('depth', np.nan))
        dx = float(gx[1] - gx[0]); dy = float(gy[1] - gy[0]); dt = float(t[1] - t[0])
        k, om, wt = ridge(Z, dx, dy, dt)
        if len(k) < 5:
            print(f'  {os.path.basename(p):40s} 능선 {len(k)}점 — 건너뜀'); continue
        d, rel, rel_deep = fit_depth(k, om, wt)
        rows.append(dict(ep=os.path.basename(p), true_depth=depth, fit_depth=d,
                         rel=rel, rel_deep=rel_deep, n_ridge=len(k),
                         dx=dx, dy=dy, dt=dt, T=int(Z.shape[0])))
        print(f'  {os.path.basename(p)[:38]:38s} 참 {depth:5.1f} m → 적합 '
              f'{d:8.1f} m  잔차 {rel:.4f} (심해가정 {rel_deep:.4f})  능선 {len(k)}점')
    json.dump(rows, open(os.path.join(HERE, 'ruler_t10.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
