"""능선 추정 2판 — **포물선 보간 + 에너지 문턱**. ⛔여전히 제3자 자료 미접촉.

1판(`ruler_t10.py:ridge`)은 `|k|` 띠마다 `argmax` 를 한 번 취했다 ⟹ ω 가 **격자에 갇힌다**
(잔차 바닥 = Δω). 여기서는 첨두 주변 3점에 포물선을 맞춰 **서브빈**으로 올린다.
"""
import numpy as np

G_ACC = 9.80665


def ridge2(E, dx, dy, dt, kmin=0.012, kmax=0.30, nbin=24,
           min_modes=8, energy_frac=0.0):
    T, Ny, Nx = E.shape
    P = np.abs(np.fft.fftn(E - E.mean())) ** 2
    w = 2 * np.pi * np.fft.fftfreq(T, dt)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.sqrt(KX ** 2 + KY ** 2)
    pos = w > 0
    P, w = P[pos], w[pos]
    edges = np.linspace(kmin, kmax, nbin + 1)
    ks, ws, wt = [], [], []
    tot = P.sum()
    for a, b in zip(edges[:-1], edges[1:]):
        m = (K >= a) & (K < b)
        if m.sum() < min_modes:
            continue
        prof = P[:, m].sum(axis=1)
        if prof.sum() <= 0 or prof.sum() / tot < energy_frac:
            continue
        j = int(np.argmax(prof))
        if 0 < j < len(prof) - 1:
            # ★포물선 보간 — 세 점 로그영역(스펙트럼 첨두는 지수적으로 떨어진다)
            y0, y1, y2 = prof[j - 1], prof[j], prof[j + 1]
            den = (y0 - 2 * y1 + y2)
            shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            shift = float(np.clip(shift, -0.5, 0.5))
        else:
            shift = 0.0
        dw = w[1] - w[0]
        ks.append(0.5 * (a + b)); ws.append(float(w[j] + shift * dw))
        wt.append(float(prof[j]))
    return np.array(ks), np.array(ws), np.array(wt)


def fit(k, om, wt):
    def sse(d):
        return float(np.sum(wt * (np.sqrt(G_ACC * k * np.tanh(k * d)) - om) ** 2) / wt.sum())
    ds = np.geomspace(0.5, 3000.0, 600)
    i = int(np.argmin([sse(d) for d in ds]))
    lo, hi = ds[max(0, i - 1)], ds[min(len(ds) - 1, i + 1)]
    for _ in range(80):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (lo, m2) if sse(m1) < sse(m2) else (m1, hi)
    d = 0.5 * (lo + hi)
    rel = np.sqrt(sse(d)) / om.mean()
    rel_deep = float(np.sqrt(np.sum(wt * (np.sqrt(G_ACC * k) - om) ** 2) / wt.sum()) / om.mean())
    return d, float(rel), rel_deep
