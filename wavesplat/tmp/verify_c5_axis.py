"""C5: is the aperture-axis RELABEL valid, or does the ratio-unit sweep design break it?

His claim: runs used REAL U [m/s] over REAL D_eff [m]; only the labels
(U/cp, D_eff/lam) were wrong -> relabel from stored JSON, no re-run needed.

The relabel of the two axes is arithmetically fine. The question is whether the
DESIGN survives. Two design quantities were ALSO derived from the bottomed lam:
   fp = 0.18 * lam_p(record)     (d3_sa_multirecord.py:63 comment)
   T  = ~10 * Tp(record)
These were meant to be HELD CONSTANT across records so that D_eff/lam is the
only thing varying. In TRUE units they are no longer constant -- and no
relabelling can fix that, because they are baked into the observations.
"""
import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from d3_sa_multirecord import RECORDS

G = 9.81
TRUE = {r['name']: r for r in json.load(open('tmp/record_wavelength.json'))}
J = json.load(open('results/direction3/sa_multirecord_mmse_net.json'))


def cp_Tp(k, d):
    om = np.sqrt(G * k * np.tanh(k * d))
    return om / k, 2 * np.pi / om


print("=== C5a: relabel the axes from the stored JSON (his claim) ===")
print(f"{'rec':7s}{'maxU/cp_lab':>13}{'maxU/cp_true':>14}"
      f"{'maxDeff/lam_lab':>17}{'maxDeff/lam_true':>18}")
rows = []
for name, rec in J.items():
    if not isinstance(rec, dict) or 'windows' not in rec:
        continue
    t = TRUE.get(name)
    if t is None:
        continue
    depth = RECORDS[name]['depth']
    lam_true = t['lam_temporal']
    k_true = 2 * np.pi / lam_true
    cp_true, Tp_true = cp_Tp(k_true, depth)
    ulab, utrue, dlab, dtrue = [], [], [], []
    for w in rec['windows']:
        for cname, c in w['conditions'].items():
            if not cname.startswith('B_'):
                continue
            U = c.get('U')
            D = c.get('D_eff_m')
            if U is None or D is None:
                continue
            ulab.append(c.get('U_over_cp'))
            dlab.append(c.get('D_eff_lam'))
            utrue.append(U / cp_true)
            dtrue.append(D / lam_true)
    if not ulab:
        continue
    rows.append(dict(rec=name, maxU_lab=max(ulab), maxU_true=max(utrue),
                     maxD_lab=max(dlab), maxD_true=max(dtrue),
                     lam_true=lam_true, cp_true=cp_true, Tp_true=Tp_true))
    print(f"{name:7s}{max(ulab):13.2f}{max(utrue):14.2f}"
          f"{max(dlab):17.2f}{max(dtrue):18.2f}")

print("\n=== C5b: THE CONFOUND he glossed -- fp/lam and T/Tp were DESIGN CONSTANTS ===")
print("design intent: fp = 0.18*lam  and  T = 10*Tp  (held constant across records)")
print(f"{'rec':7s}{'fp[m]':>7}{'T[s]':>7}{'fp/lam_pipe':>13}{'fp/lam_TRUE':>13}"
      f"{'T/Tp_pipe':>11}{'T/Tp_TRUE':>11}")
fp_true_all, T_true_all = [], []
for r in rows:
    name = r['rec']
    cfg = RECORDS[name]
    fp, T = cfg['fp'], cfg['T']
    lam_pipe = np.median([w['lam_p'] for w in J[name]['windows']])
    Tp_pipe = np.median([w['Tp'] for w in J[name]['windows']])
    a, b = fp / lam_pipe, fp / r['lam_true']
    c, d = T / Tp_pipe, T / r['Tp_true']
    fp_true_all.append(b)
    T_true_all.append(d)
    print(f"{name:7s}{fp:7.1f}{T:7.0f}{a:13.3f}{b:13.3f}{c:11.1f}{d:11.1f}")

print(f"\n  fp/lam  : design 0.18 constant | in TRUE units spans "
      f"{min(fp_true_all):.3f} - {max(fp_true_all):.3f}  "
      f"({max(fp_true_all)/min(fp_true_all):.1f}x spread)")
print(f"  T/Tp    : design ~10 constant   | in TRUE units spans "
      f"{min(T_true_all):.1f} - {max(T_true_all):.1f}  "
      f"({max(T_true_all)/min(T_true_all):.1f}x spread)")
print("\n  -> the sensor FOOTPRINT and the WINDOW LENGTH are baked into the")
print("     observations. Relabelling the x-axis does NOT restore the design:")
print("     at matched D_eff/lam_TRUE, records no longer share fp/lam or T/Tp.")
print("     That is a CONFOUND, and it requires a RE-RUN, not a relabel.")
