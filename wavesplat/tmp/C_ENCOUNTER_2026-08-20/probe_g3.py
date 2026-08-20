#!/usr/bin/env python3
"""G-3 probe: does a moving sensor actually see the encounter frequency?

Cheapest decisive test of the load-bearing kinematic claim.  Raycasting samples the
surface; it does not change what frequencies are present in the time series a moving
observer sees.  So evaluate the analytic field along the sensor path and take its FFT.

Predicted, from omega_e = omega - k . U:
    following (waves travel +x, ship travels +x)  ->  omega_e = omega - k*U
    head      (waves travel -x, ship travels +x)  ->  omega_e = omega + k*U
At U = 11.3 m/s and BF4 Tp = 7.0 s the following case sits within 3% of the degenerate
point omega_e = 0, i.e. an almost frozen pattern.  A stationary sensor must show omega
itself in BOTH cases -- that is the negative control that makes this test non-vacuous.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wavesim.config import WaveSimConfig            # noqa: E402
from wavesim.wave_generator import jonswap_components  # noqa: E402

G = 9.81
U_DESIGN = 11.3          # reference_agreed_operating_point.md
DUR, DT = 180.0, 0.05    # long enough to resolve ~0.03 rad/s; 20 Hz


def series(components, U, duration=DUR, dt=DT):
    """Surface elevation at the sensor, which rides at x = U t."""
    t = np.arange(0.0, duration, dt)
    z = np.zeros_like(t)
    for c in components:
        kx, ky, w, a, ph = c.kx, c.ky, c.omega, c.A, c.phi
        z += a * np.cos(kx * (U * t) + ky * 0.0 - w * t + ph)
    return t, z


def peak_omega(t, z):
    z = z - z.mean()
    Z = np.abs(np.fft.rfft(z * np.hanning(len(z))))
    f = np.fft.rfftfreq(len(z), t[1] - t[0])
    return 2 * np.pi * f[int(np.argmax(Z))]


def main():
    cfg = WaveSimConfig()
    print('=' * 84)
    print('G-3 PROBE -- encounter frequency at a moving sensor   (run-0 kinematics test)')
    print('=' * 84)

    for sea, Tp in (('BF4', 7.0), ('BF5', 9.0)):
        w_p = 2 * np.pi / Tp
        k_p = w_p ** 2 / G
        print('\n%s  Tp=%.1f s  omega_p=%.4f  k_p=%.5f  c_p=%.2f m/s  U/c_p=%.3f'
              % (sea, Tp, w_p, k_p, w_p / k_p, U_DESIGN / (w_p / k_p)))

        for label, direction in (('following (dir 0)', 0.0), ('head (dir 180)', 180.0)):
            Hs = {'BF4': 1.2, 'BF5': 2.0}[sea]        # config.py:130-131
            comps = jonswap_components(
                Hs=Hs, Tp=Tp, n_components=1, depth=50.0,
                direction_deg=direction, direction_spread_deg=0.0,
                seed=20, peak_locked=True)
            c = comps[0]
            k_signed = c.kx                       # ky = 0 at spread 0
            for U in (0.0, U_DESIGN):
                t, z = series(comps, U)
                measured = peak_omega(t, z)
                predicted = abs(c.omega - k_signed * U)
                tag = ('  <- STATIONARY control: must equal omega'
                       if U == 0.0 else '')
                print('    %-18s U=%5.1f  omega=%.4f  k=%+.5f  '
                      'predicted|omega_e|=%.4f  measured=%.4f%s'
                      % (label, U, c.omega, k_signed, predicted, measured, tag))
                if abs(predicted - measured) > max(0.02, 0.05 * predicted):
                    print('        !! MISMATCH -- kinematics do not behave as derived')

    print('\n' + '-' * 84)
    print('Verdict rule: the stationary rows must reproduce omega in both directions')
    print('(non-vacuous control), and the U=11.3 following row must collapse toward 0.')
    print('-' * 84)


if __name__ == '__main__':
    main()
