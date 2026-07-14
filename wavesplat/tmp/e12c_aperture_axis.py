"""E-12c — recompute the aperture-law axes with the corrected wavelengths.

This is a CORRECTION, not a gated test: for 5 of 10 records the pipeline's kp is
pinned at the patch fundamental, so every quantity indexed by lam_p / c_p / T_p is
wrong. The experiments themselves are fine — they moved the sensor at real
physical speeds U [m/s] over real physical apertures D_eff [m]. What is wrong is
the LABEL on the axis:

    U/c_p      : the runs set U = ratio * cp_pipe. The physical speed is real;
                 the true ratio is U / cp_true = ratio * (cp_pipe / cp_true).
    D_eff/lam_p: D_eff is in metres (real); only lam_p was wrong.

So the axes can be relabelled from the stored JSONs without re-running anything.

The question this answers: did the sweeps ever actually reach the high-speed /
large-aperture regime they claim to have probed?
"""
import sys
import os
import json
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_multirecord import RECORDS            # noqa: E402

G = 9.81
TRUE = {r['name']: r for r in json.load(open('tmp/record_wavelength.json'))}


def cp_of(lam, d):
    k = 2 * np.pi / lam
    om = np.sqrt(G * k * np.tanh(k * d))
    return float(om / k), float(2 * np.pi / om)


src = sorted(glob.glob('results/direction3/sa_multirecord*.json'))
print('multirecord result files:', [os.path.basename(s) for s in src])

rows = []
for path in src:
    d = json.load(open(path))
    if not isinstance(d, dict):
        continue
    recs = d.get('records', d)
    if not isinstance(recs, dict):
        continue
    for key, out in recs.items():
        if not isinstance(out, dict) or 'windows' not in out:
            continue
        cfg = RECORDS.get(key)
        t = TRUE.get(key)
        if cfg is None or t is None:
            continue
        depth = cfg['depth']
        lam_true = t['lam_temporal']
        cp_true, Tp_true = cp_of(lam_true, depth)
        for w in out['windows']:
            lam_pipe = w.get('lam_p')
            cp_pipe = w.get('cp')
            if lam_pipe is None or cp_pipe is None:
                continue
            for cname, e in w.get('conditions', {}).items():
                if not cname.startswith('B_'):
                    continue
                ratio_nom = e.get('ratio')          # the label the sweep used
                U = e.get('U')                      # the PHYSICAL speed [m/s]
                D_eff_m = e.get('D_eff_m')          # the PHYSICAL aperture [m]
                if U is None or D_eff_m is None:
                    continue
                rows.append(dict(
                    file=os.path.basename(path), record=key, cond=cname,
                    U=U, D_eff_m=D_eff_m,
                    lam_pipe=lam_pipe, lam_true=lam_true,
                    cp_pipe=cp_pipe, cp_true=cp_true,
                    U_over_cp_pipe=e.get('U_over_cp', U / cp_pipe),
                    U_over_cp_true=U / cp_true,
                    Deff_lam_pipe=e.get('D_eff_lam', D_eff_m / lam_pipe),
                    Deff_lam_true=D_eff_m / lam_true))

if not rows:
    print('\nno multirecord JSON with per-condition metadata found — '
          'the axes must be recomputed at the next run of d3_sa_multirecord.')
    sys.exit(0)

json.dump(rows, open('tmp/e12c_aperture_axis.json', 'w'), indent=1)

print(f"\n{'rec':<8}{'nominal U/cp':>14}{'TRUE U/cp':>11}{'shift':>8}"
      f"{'nominal Deff/lam':>18}{'TRUE Deff/lam':>15}{'shift':>8}")
print('-' * 84)
for key in sorted({r['record'] for r in rows}):
    rr = [r for r in rows if r['record'] == key]
    for nom in sorted({round(r['U_over_cp_pipe'], 2) for r in rr}):
        sub = [r for r in rr if abs(r['U_over_cp_pipe'] - nom) < 0.02]
        if not sub:
            continue
        tu = np.mean([r['U_over_cp_true'] for r in sub])
        dp = np.mean([r['Deff_lam_pipe'] for r in sub])
        dt = np.mean([r['Deff_lam_true'] for r in sub])
        print(f"{key:<8}{nom:14.2f}{tu:11.2f}{tu/nom:7.2f}x"
              f"{dp:18.2f}{dt:15.2f}{dt/dp:7.2f}x")

print("\nthe claim to check: did the sweep reach the regimes it says it probed?")
for key in sorted({r['record'] for r in rows}):
    rr = [r for r in rows if r['record'] == key]
    mx_nom = max(r['U_over_cp_pipe'] for r in rr)
    mx_true = max(r['U_over_cp_true'] for r in rr)
    da_nom = max(r['Deff_lam_pipe'] for r in rr)
    da_true = max(r['Deff_lam_true'] for r in rr)
    print(f"  {key:<8} max U/cp: claimed {mx_nom:.2f} -> actually {mx_true:.2f}"
          f"   |  max D_eff/lam: claimed {da_nom:.2f} -> actually {da_true:.2f}")
