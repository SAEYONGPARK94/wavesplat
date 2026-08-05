"""Generator-side counterpart: what OUR generator actually produces, and what
knob setting would be needed to match the real sigma1_total.

No wave runs -- jonswap_components() only (component construction, no fields).
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/depth3_ceiling_2026-08-05')
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components  # noqa: E402

FMIN, FMAX = 0.04, 0.50
NS = (1, 2, 5, 15, 30, 60, 120, 400)
KNOBS = (30.0, 37.2, 45.0, 60.0)
NDRAW = 300

REAL = json.load(open(os.path.join(HERE, 'ceiling.json')))


def s1deg(r1):
    return float(np.degrees(np.sqrt(max(2.0 * (1.0 - min(max(r1, 0.0), 1.0)), 0.0))))


def gen_sigma(n, spread_deg, seed, Tp=7.0, Hs=1.0, band=True):
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=n, depth=50.0,
                           direction_spread_deg=spread_deg, seed=seed)
    A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
    th = np.arctan2([x.ky for x in c], [x.kx for x in c])
    fr = om / (2 * np.pi); E = 0.5 * A ** 2
    m = ((fr >= FMIN) & (fr <= FMAX)) if band else np.ones_like(fr, bool)
    if not m.any():
        return np.nan
    w = E[m] / E[m].sum()
    return s1deg(np.hypot((w * np.cos(th[m])).sum(), (w * np.sin(th[m])).sum()))


out = {'note': 'component construction only; no wave field was simulated',
       'population_sigma1_analytic': {}, 'ladder': {}}

for s in KNOBS:
    # population sigma1 of a wrapped normal with std s:  r1 = exp(-s^2/2)
    r1 = float(np.exp(-np.deg2rad(s) ** 2 / 2))
    out['population_sigma1_analytic'][str(s)] = s1deg(r1)

for s in KNOBS:
    lad = {}
    for n in NS:
        v = np.array([gen_sigma(n, s, 900_000 + 7919 * k) for k in range(NDRAW)])
        v = v[np.isfinite(v)]
        lad[str(n)] = dict(median=float(np.median(v)), p05=float(np.percentile(v, 5)),
                           p95=float(np.percentile(v, 95)), n_draws=int(v.size))
    out['ladder'][str(s)] = lad
    print(f'knob={s:5.1f}deg  pop_sigma1={out["population_sigma1_analytic"][str(s)]:.2f}  '
          + '  '.join(f'N{n}={lad[str(n)]["median"]:.2f}' for n in NS), flush=True)

# ---- what knob is needed to match the real sigma1_total at N=30 ?
real_tot_all6 = REAL['all6']['sigma1_total_deg']
real_tot_028 = REAL['028p1']['sigma1_total_deg']
real_across_all6 = REAL['all6']['sigma1_across_deg__ARCHITECTURE_CEILING']

grid = np.arange(20.0, 90.1, 0.5)
curve = {}
for n in (30, 400):
    med = np.array([np.median([gen_sigma(n, float(s), 800_000 + 7919 * k)
                               for k in range(60)]) for s in grid])
    curve[str(n)] = med.tolist()
    for tag, target in (('all6', real_tot_all6), ('028p1', real_tot_028)):
        j = int(np.argmin(np.abs(med - target)))
        out.setdefault('knob_needed_to_match_real_total', {})[f'N{n}_{tag}'] = {
            'target_sigma1_total_deg': target, 'knob_deg': float(grid[j]),
            'achieved_deg': float(med[j])}
out['knob_curve_grid_deg'] = grid.tolist()
out['knob_curve_median_sigma1'] = curve

# ---- the substitution the knob performs
k30 = out['knob_needed_to_match_real_total']['N30_all6']
out['substitution'] = {
    'real_sigma1_total_deg': real_tot_all6,
    'real_sigma1_within_avg_deg': REAL['all6']['sigma1_within_avg_deg'],
    'real_sigma1_ACROSS_deg': real_across_all6,
    'synth_sigma1_within_deg': 0.0,
    'synth_sigma1_ACROSS_deg_when_total_matched': k30['achieved_deg'],
    'across_overshoot_deg': k30['achieved_deg'] - real_across_all6,
    'across_overshoot_pct': (k30['achieved_deg'] / real_across_all6 - 1) * 100,
    'knob_deg_required': k30['knob_deg'],
}

json.dump(out, open(os.path.join(HERE, 'generator_side.json'), 'w'), indent=2)
print(json.dumps({k: out[k] for k in ('knob_needed_to_match_real_total', 'substitution')}, indent=2))
