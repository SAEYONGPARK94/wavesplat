"""
d3_stream_longrun.py — 10-minute continuous streaming stability test
(2026-07-06, roadmap B4). Linear sea (cheap exact GT) at the design
speed, 10 FPS x 600 s = 6000 filter steps: does the RLS/Kalman filter
drift numerically (P asymmetry, negative variances, NIS runaway) or in
skill (FF-band corr per minute)?

Usage: python3 scripts/d3_stream_longrun.py [--minutes 10]
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_stream_seqmmse import make_bank, gamma_beamform, StreamMMSE

G = 9.81
OUT = 'results/direction3/stream_longrun.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=10.0)
    ap.add_argument('--U', type=float, default=11.3)
    ap.add_argument('--tau', type=float, default=20.0)
    a = ap.parse_args()
    Hs, Tp, depth = 1.2, 7.0, 50.0
    kp = (2 * np.pi / Tp) ** 2 / G
    comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=depth,
                               direction_spread_deg=25.0, seed=42)
    kx = np.array([c.kx for c in comps])
    ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps])
    ph = np.array([c.phi for c in comps])
    om = omega_of(kx, ky, depth)

    def eta(x, y, t):
        ps = kx[None] * np.atleast_1d(x)[:, None] \
            + ky[None] * np.atleast_1d(y)[:, None] \
            - om[None] * np.atleast_1d(t)[:, None] + ph[None]
        return (A[None] * np.cos(ps)).sum(1)

    rng = np.random.default_rng(0)
    dt, m = 0.1, 12
    lead = 2.0

    def ship_pos(t):
        return np.array([-a.U * t, 0.0])
    h = np.array([-1.0, 0.0])

    # prior from first 10 s
    Pw, zw = [], []
    for t in np.arange(0.0, 10.0, dt):
        sp = ship_pos(t)
        rr = rng.uniform(20.0, 100.0, m)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, m)
        xs, ys = sp[0] - rr * np.cos(aa), sp[1] + rr * np.sin(aa)
        Pw.append(np.column_stack([xs, ys, np.full(m, t)]))
        zw.append(eta(xs, ys, t) + rng.normal(0, 0.02, m))
    Pw, zw = np.vstack(Pw), np.concatenate(zw)
    ii = np.random.default_rng(1).choice(len(zw), 600, False)
    kxb, kyb, omb = make_bank(kp, depth)
    gam, s2 = gamma_beamform(kxb, kyb, omb, Pw[ii], zw[ii], 0.05 * Hs,
                             depth)
    st = StreamMMSE(kxb, kyb, omb, gam, s2, tau=a.tau, dt=dt, prune=400)

    n_steps = int(a.minutes * 60 / dt)
    per_min, buf = [], []
    nis_run, lat = [], []
    t = 10.0
    for k in range(n_steps):
        sp = ship_pos(t)
        rr = rng.uniform(20.0, 100.0, m)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, m)
        xs, ys = sp[0] - rr * np.cos(aa), sp[1] + rr * np.sin(aa)
        z = eta(xs, ys, t) + rng.normal(0, 0.02, m)
        t1 = time.perf_counter()
        nis = st.step(np.column_stack([xs, ys, np.full(m, t)]), z)
        tq = t + lead
        xq, yq = ship_pos(tq)
        f, _ = st.predict(np.array([[xq, yq, tq]]))
        lat.append(time.perf_counter() - t1)
        nis_run.append(nis)
        buf.append((float(f[0]), float(eta([xq], [yq], tq)[0])))
        t += dt
        if (k + 1) % 600 == 0:                       # each minute
            pp = np.array([x for x, _ in buf])
            gg = np.array([g for _, g in buf])
            asym = 0.0                               # kept symmetric
            dmin = float(np.diag(st.P).min())
            per_min.append(dict(
                minute=len(per_min) + 1,
                corr=float(np.corrcoef(pp, gg)[0, 1]),
                nis=float(np.mean(nis_run[-600:])),
                Pdiag_min=dmin))
            print(f"min {per_min[-1]['minute']:2d}: corr "
                  f"{per_min[-1]['corr']:+.3f} NIS "
                  f"{per_min[-1]['nis']:.2f} Pdiag_min {dmin:.2e}",
                  flush=True)
            buf = []
    lat = np.array(lat) * 1e3
    print(f"\nlatency median {np.median(lat):.1f} ms p95 "
          f"{np.percentile(lat, 95):.1f} ms over {n_steps} steps")
    json.dump(dict(minutes=a.minutes, U=a.U, tau=a.tau, per_min=per_min,
                   lat_ms_median=float(np.median(lat)),
                   lat_ms_p95=float(np.percentile(lat, 95))),
              open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
