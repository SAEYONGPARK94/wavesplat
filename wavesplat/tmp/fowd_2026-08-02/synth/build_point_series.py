"""
Step A-1: turn every synthetic episode into a 30-min point elevation record.

Rationale
---------
FOWD / CDIP sea-state parameters are defined on ~30-min buoy displacement
records at 1.28 Hz.  Our episodes are 30 s long, so they cannot be fed in
directly.  The generator, however, *defines* the sea state for all time:
the episode stores its exact wave components (A, omega, kx, ky, phi, Q), and
the surface is an analytic superposition.  We therefore evaluate the same
analytic surface at a single point for 30 min.  No new assumption is added --
this is the generator's own sea, observed the way a buoy observes.

Two observation conventions are produced:
  * lagrangian : Z at a fixed Gerstner *label* (what a Datawell buoy measures)
  * eulerian   : Z at a fixed horizontal *position* (what our LiDAR measures)

Three groups are produced:
  * synth_used      : the 110 episodes as the paper uses them (N = 1..5)
  * synth_continuum : same Hs/Tp/gamma/depth but N = 500 components
  * (real CDIP is handled separately)
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
sys.path.insert(0, os.path.dirname(__file__))

from wavesim.wave_generator import jonswap_components, WaveComponent  # noqa: E402
from seastate import sea_state_params, analytic_excess_kurtosis  # noqa: E402

DATA = '/home/vexit/wavesplat/wavesplat/data'
OUT = '/home/vexit/wavesplat/wavesplat/tmp/fowd_2026-08-02/synth'
FS = 1.28              # CDIP / FOWD native sampling rate
DUR = 1800.0           # 30 min, the FOWD sea-state window
X0, Y0 = 130.0, 0.0    # centre of the episode grid (x in [10, 250], y in [-100, 100])


def load_components(path):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        attrs = dict(f['config'].attrs)
    comps = []
    for A, omega, kx, ky, phi, Q in comp:
        k = float(np.hypot(kx, ky))
        comps.append(dict(A=float(A), omega=float(omega), kx=float(kx),
                          ky=float(ky), phi=float(phi), Q=float(Q), k=k))
    return comps, attrs


def eta_lagrangian(comps, t, x=X0, y=Y0):
    """Z at fixed Gerstner label -- vertical displacement of a water particle."""
    z = np.zeros_like(t)
    for c in comps:
        z += c['A'] * np.cos(c['kx'] * x + c['ky'] * y - c['omega'] * t + c['phi'])
    return z


def eta_eulerian(comps, t, x=X0, y=Y0, n_iter=25):
    """
    Z at a fixed horizontal position.  Gerstner maps label (a,b) to
    position (a + Xd, b + Yd); we invert by fixed-point iteration.
    """
    a = np.full_like(t, x)
    b = np.full_like(t, y)
    for _ in range(n_iter):
        Xd = np.zeros_like(t)
        Yd = np.zeros_like(t)
        for c in comps:
            ph = c['kx'] * a + c['ky'] * b - c['omega'] * t + c['phi']
            s = np.sin(ph)
            Xd += -c['Q'] * c['A'] * (c['kx'] / c['k']) * s
            Yd += -c['Q'] * c['A'] * (c['ky'] / c['k']) * s
        a_new, b_new = x - Xd, y - Yd
        if np.max(np.abs(a_new - a)) < 1e-10 and np.max(np.abs(b_new - b)) < 1e-10:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    z = np.zeros_like(t)
    for c in comps:
        z += c['A'] * np.cos(c['kx'] * a + c['ky'] * b - c['omega'] * t + c['phi'])
    return z


def recurrence_period(comps, tol=1e-9):
    """
    Exact repeat period of the superposition, if the omegas are commensurate.
    Returns (period_s, is_exact).
    """
    om = np.array([c['omega'] for c in comps])
    if len(om) == 1:
        return float(2 * np.pi / om[0]), True
    # find the largest g such that every omega is an integer multiple of g,
    # searching over the natural candidate: gcd of the omega grid
    d = np.diff(np.sort(om))
    g = d[0] if len(d) else om[0]
    for _ in range(200):
        r = om / g
        frac = np.abs(r - np.round(r))
        if np.all(frac < 1e-6):
            return float(2 * np.pi / g), True
        g = g / 2.0
        if g < 1e-4:
            break
    return float('nan'), False


def main():
    t = np.arange(0.0, DUR, 1.0 / FS)
    rows = []

    files = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
    print(f'{len(files)} episodes')
    for i, fn in enumerate(files):
        comps, attrs = load_components(os.path.join(DATA, fn))
        z_lag = eta_lagrangian(comps, t)
        z_eul = eta_eulerian(comps, t)
        Trec, exact = recurrence_period(comps)
        amps = [c['A'] for c in comps]
        row = dict(
            name=fn, group='synth_used',
            n_components=int(attrs['n_components']),
            sea_state=str(attrs['sea_state']),
            Hs_cfg=float(attrs['Hs']), Tp_cfg=float(attrs['Tp']),
            depth=float(attrs['depth']),
            raydrop='slope' if 'slope' in fn else 'uniform',
            T_recurrence=Trec, recurrence_exact=bool(exact),
            kurt_analytic=float(analytic_excess_kurtosis(amps)),
        )
        for tag, z in (('lag', z_lag), ('eul', z_eul)):
            p = sea_state_params(z, FS, depth=float(attrs['depth']))
            for k, v in p.items():
                row[f'{tag}_{k}'] = v
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(files)}')

    # ---- continuum reference: same sea states, N = 500 ----
    seen = {}
    for r in rows:
        seen.setdefault((r['sea_state'], r['Hs_cfg'], r['Tp_cfg'], r['depth']), 0)
    print(f'{len(seen)} distinct sea states -> continuum reference')
    for (ss, Hs, Tp, depth) in seen:
        for seed in range(5):
            wc = jonswap_components(Hs=Hs, Tp=Tp, n_components=500,
                                    depth=depth, seed=1000 + seed)
            comps = [dict(A=c.A, omega=c.omega, kx=c.kx, ky=c.ky, phi=c.phi,
                          Q=c.Q, k=c.k) for c in wc]
            z_lag = eta_lagrangian(comps, t)
            z_eul = eta_eulerian(comps, t)
            Trec, exact = recurrence_period(comps)
            row = dict(
                name=f'continuum_{ss}_N500_seed{seed}', group='synth_continuum',
                n_components=500, sea_state=ss, Hs_cfg=Hs, Tp_cfg=Tp,
                depth=depth, raydrop='n/a',
                T_recurrence=Trec, recurrence_exact=bool(exact),
                kurt_analytic=float(analytic_excess_kurtosis([c['A'] for c in comps])),
            )
            for tag, z in (('lag', z_lag), ('eul', z_eul)):
                p = sea_state_params(z, FS, depth=depth)
                for k, v in p.items():
                    row[f'{tag}_{k}'] = v
            rows.append(row)

    with open(os.path.join(OUT, 'synth_seastate.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    print('wrote synth_seastate.json:', len(rows), 'records')


if __name__ == '__main__':
    main()
