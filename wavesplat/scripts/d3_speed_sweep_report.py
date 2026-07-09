"""Report the speed sweep: does the learned prior start to pay as the ship
goes faster?

Reads results/direction3/required_horizon.json (keys <case>_U<speed>) and
emits a table plus results/direction3/speed_sweep.png.

The headline is the paired per-rep difference at the operational band,

    delta = nRMSE(WaveSplat) - nRMSE(WavePrior)      (positive = learned wins)

on alpha_wave where it exists. At U = 0 alpha_wave is undefined (the foil is
not foilborne and w/U diverges), so the stationary point is reported on slope
and drawn with an open marker — it is the left edge of the trend, not a
control requirement.

Also prints the physical driver: the fraction of an ENCOUNTER period that the
0.5 s preview spans, omega_e = omega_p + k_p U (head seas). If the learned
prior only pays once that fraction grows, the curve should track it.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

G = 9.81
SRC = 'results/direction3/required_horizon.json'
PNG = 'results/direction3/speed_sweep.png'
BAND = 'operational 0-0.5s'
PREVIEW = 0.5

CASES = [('linear', 'Linear sim', '#5B7EC9'),
         ('hos', 'HOS nonlinear sim', '#4C9A6B'),
         ('real_YS17', 'Real YS17 (wind sea)', '#D08770'),
         ('real_LJ18g', 'Real LJ18g (storm)', '#8E5572')]

# peak wavenumber per case, for the encounter-period diagnostic
KP = {'linear': (2 * np.pi / 8.0) ** 2 / G, 'hos': (2 * np.pi / 8.0) ** 2 / G,
      'real_YS17': 0.1342, 'real_LJ18g': 0.0312}


def enc_fraction(kp, U, depth=50.0):
    om = np.sqrt(G * kp * np.tanh(kp * depth))
    ome = om + kp * U                       # head seas
    return PREVIEW / (2 * np.pi / ome)


def main():
    d = json.load(open(SRC))
    speeds = sorted({float(k.split('_U')[1]) for k in d if '_U' in k})
    print(f"speeds: {speeds}\n")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=110)
    ax, ax2, ax3 = axes

    for key, label, col in CASES:
        xs, ys, es, qs = [], [], [], []
        for U in speeds:
            k = f'{key}_U{U:g}'
            if k not in d:
                continue
            pd_ = d[k]['paired_delta_0-0.5s']
            q = 'alpha_wave' if 'alpha_wave' in pd_ else 'slope'
            xs.append(U)
            ys.append(pd_[q]['mean'])
            es.append(pd_[q]['std'] / np.sqrt(pd_[q]['n']))   # s.e.m.
            qs.append(q)
        if not xs:
            continue
        xs, ys, es = np.array(xs), np.array(ys), np.array(es)
        ax.errorbar(xs, ys, yerr=es, color=col, marker='o', ms=5, lw=1.7,
                    capsize=3, label=label)
        # open marker where the headline fell back to slope (U = 0)
        for x, y, q in zip(xs, ys, qs):
            if q != 'alpha_wave':
                ax.plot([x], [y], 'o', ms=9, mfc='white', mec=col, mew=1.6)

        print(f"=== {label} ===")
        for U, y, e, q in zip(xs, ys, es, qs):
            k = f'{key}_U{U:g}'
            pd_ = d[k]['paired_delta_0-0.5s'][q]
            ws = d[k]['WaveSplat'][BAND][f'{q}_nrmse']['mean']
            wp = d[k]['WavePrior'][BAND][f'{q}_nrmse']['mean']
            sig = 'YES' if abs(y) > 2 * e else ' no'
            print(f"  U={U:5.1f}  {q:<10} WS {ws:.3f} / WP {wp:.3f}   "
                  f"delta {y:+.3f} +- {e:.3f} (s.e.m.)  "
                  f"WP wins {pd_['wp_wins']}/{pd_['n']}  2se-sig {sig}   "
                  f"| 0.5s = {100*enc_fraction(KP[key], U):.0f}% of Te")
        print()

        ax2.plot([100 * enc_fraction(KP[key], U) for U in xs], ys,
                 color=col, marker='o', ms=5, lw=1.7, label=label)

        # eta is the estimation-native quantity: defined at every speed and
        # not reweighted by U the way alpha_wave = -slope - w/U is.
        ey, ee = [], []
        for U in xs:
            p = d[f'{key}_U{U:g}']['paired_delta_0-0.5s']['eta']
            ey.append(p['mean'])
            ee.append(p['std'] / np.sqrt(p['n']))
        ax3.errorbar(xs, ey, yerr=ee, color=col, marker='s', ms=4.5, lw=1.7,
                     capsize=3, label=label)

    for a, xl in ((ax, 'ship speed U [m/s]'),
                  (ax2, '0.5 s preview as % of encounter period'),
                  (ax3, 'ship speed U [m/s]')):
        a.axhline(0, color='0.4', lw=1, ls='--')
        a.set_xlabel(xl, fontsize=9)
        a.set_ylabel(r'paired $\Delta$ nRMSE   (+ = WavePrior better)',
                     fontsize=9)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.25)
    ax.axvline(11.3, color='0.3', lw=1, ls=':')
    ax.text(11.3, ax.get_ylim()[1], ' 22 kt design pt', fontsize=7,
            va='top', color='0.3')
    ax.set_title('Learned-prior gain vs speed (open marker: U=0, slope only)',
                 fontsize=10)
    ax2.set_title('Same, against the physical driver', fontsize=10)
    ax3.set_title(r'$\Delta$ on $\eta$ (estimation-native quantity)',
                  fontsize=10)
    ax.legend(fontsize=7.5, loc='best')
    ax3.legend(fontsize=7.5, loc='best')
    fig.tight_layout()
    fig.savefig(PNG)
    print(f"-> {PNG}")


if __name__ == '__main__':
    sys.exit(main())
