"""Paired MMSE-vs-RELAX read-out on the real multirecord runs.
Compares results/direction3/sa_multirecord_mmse.json (today) against
sa_multirecord.json (RELAX baseline) window-by-window: absolute
common_unobs corr_bl per condition, motion gains, and paired deltas.
NOTE: window starts are identical (same protocol/starts/seeds per record),
so per-window pairing is valid."""
import json
import sys

import numpy as np

BASE = 'results/direction3/sa_multirecord.json'
MMSE = 'results/direction3/sa_multirecord_mmse.json'


def table(d, key):
    out = d[key]
    rows = {}
    A = []
    for w in out['windows']:
        e0 = w['conditions']['A']
        A.append(e0['mean']['common_unobs']['corr_bl'])
        for name, e in w['conditions'].items():
            if not name.startswith('B_'):
                continue
            rows.setdefault(name, []).append(
                (e['mean']['common_unobs']['corr_bl'],
                 e['mean']['common_unobs']['corr_bl']
                 - e0['mean']['common_unobs']['corr_bl'],
                 e['U_over_cp']))
    return np.array(A), {n: np.array(v) for n, v in rows.items()}


def main():
    b = json.load(open(BASE))
    m = json.load(open(MMSE))
    for key in m:
        if key not in b:
            continue
        Ab, rb = table(b, key)
        Am, rm = table(m, key)
        nb, nm = len(Ab), len(Am)
        print(f"\n===== {key} (RELAX n={nb} win, MMSE n={nm} win) =====")
        print(f"A(fixed) abs:  RELAX {Ab.mean():+.3f}  MMSE {Am.mean():+.3f}"
              f"  d={Am.mean()-Ab.mean():+.3f}")
        for name in sorted(rb, key=lambda n: rb[n][:, 2].mean()):
            if name not in rm:
                continue
            vb, vm = rb[name], rm[name]
            npair = min(len(vb), len(vm))
            dabs = vm[:npair, 0] - vb[:npair, 0]
            print(f"{name:12s} U/cp={vb[:,2].mean():.2f} | abs "
                  f"RELAX {vb[:,0].mean():+.3f} MMSE {vm[:,0].mean():+.3f} "
                  f"(d {dabs.mean():+.3f}, win+ {int((dabs>0).sum())}/{npair})"
                  f" | gain RELAX {vb[:,1].mean():+.3f} "
                  f"MMSE {vm[:,1].mean():+.3f}")


if __name__ == '__main__':
    main()
