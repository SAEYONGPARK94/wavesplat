"""Independent re-derivation: can the slope raydrop model's specular branch ever fire
under Falcon grazing geometry? (2026-07-24 prose sweep, task 6)

Two independent routes:
  A. Closed form. alignment = dot(reflected, -d) = -1 + 2*(d.n)^2, so the specular test
     alignment >= cos(alpha) is exactly |d.n| >= sqrt((1+cos alpha)/2) = cos(alpha/2).
     With alpha = 5 deg that is an incidence angle within 2.5 deg of NORMAL incidence.
  B. Numerical, using the shipped function on Gerstner-like surface normals over the full
     Falcon elevation fan, swept over sea states.
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from wavesim.raydrop_slope import specular_alignment

ALPHA = 5.0

# ---- Route A -----------------------------------------------------------------
need = np.rad2deg(np.arccos(np.cos(np.deg2rad(ALPHA / 2))))
print(f'[A] closed form: specular requires incidence within {need:.2f} deg of normal')

# ---- Route B -----------------------------------------------------------------
# Falcon elevation fan: rays leave the sensor at 78-90 deg from vertical (grazing).
# Surface normal tilts away from vertical by the local wave slope angle.
rng = np.random.default_rng(0)
n_rays = 200_000

print(f'\n[B] numerical sweep ({n_rays:,} rays per case, shipped specular_alignment)')
print(f'{"case":<28}{"max slope":>11}{"specular hits":>16}{"max alignment":>15}')
for label, kA_max in [('BF2  (eps~0.02)', 0.02), ('BF4  (eps~0.06)', 0.06),
                      ('BF7  (eps~0.12)', 0.12), ('unphysically steep', 0.40)]:
    graze = np.deg2rad(rng.uniform(78.0, 90.0, n_rays))       # from vertical
    az = rng.uniform(0, 2 * np.pi, n_rays)
    # ray direction: sensor -> surface, pointing downward
    d = np.stack([np.sin(graze) * np.cos(az), np.sin(graze) * np.sin(az),
                  -np.cos(graze)], axis=1)

    slope_ang = np.arctan(kA_max * rng.uniform(-1, 1, n_rays))
    n_az = rng.uniform(0, 2 * np.pi, n_rays)
    n = np.stack([np.sin(slope_ang) * np.cos(n_az), np.sin(slope_ang) * np.sin(n_az),
                  np.cos(slope_ang)], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    mask, align = specular_alignment(d, n, ALPHA)
    print(f'{label:<28}{np.rad2deg(np.arctan(kA_max)):>10.2f}d'
          f'{int(mask.sum()):>16,}{align.max():>15.4f}')

print(f'\n(threshold for a hit: alignment >= {np.cos(np.deg2rad(ALPHA)):.5f})')
