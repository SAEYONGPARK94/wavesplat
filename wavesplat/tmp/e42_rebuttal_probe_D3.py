"""e42 rebuttal probe (D3): SNAPPED ORACLE — how much of oracle-cur=+0.0715
survives when the true (d1,a1,d2,a2,w) is snapped to the v2 blind lattice
(10 deg dirs x 0.1 a-grid x W_GRID)? This is the reachable ceiling of ANY
lattice-based blind inference; nobody measured it. Same seas si=100..163,
same solver, same family convention as e42_detector_v2 (Tp_eff = tp_of_k).
"""
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, mu_hat_of
from e41_regime_prize_ladder import lobe_grid, tp_of_k, TP2_RATIO_NODES, \
    D2_OFF_NODES, W_MARG
from e42_detector_v2 import DIRS, A_GRID, W_GRID, GAMMA, SPREAD, true_rings


def lobe(kp, a, mu):
    Tp_eff = float(tp_of_k(a * kp))
    return lobe_grid(kp, [(Tp_eff, GAMMA, 1.0)], [(mu, SPREAD, 1.0)])


def snap_dir(x):
    return DIRS[np.argmin(np.abs(np.angle(np.exp(1j * (DIRS - x)))))]


def snap(vals, x):
    v = np.asarray(vals)
    return float(v[np.argmin(np.abs(v - x))])


def main():
    t0 = time.time()
    rows = []
    for si in range(100, 164):
        S = build_sea_crosstp(si)
        kp = S['kp']
        a1t, a2t = true_rings(S)
        # continuous oracle, e41 x_marginals convention
        L1 = lobe_grid(kp, [(7.0, GAMMA, 1.0)], [(S['d1'], SPREAD, 1.0)])
        L2 = lobe_grid(kp, [(S['Tp2'], GAMMA, 1.0)], [(S['d2'], SPREAD, 1.0)])
        r2 = S['r2']
        orc = solve_with_grid(S, (1 - r2) * L1 + r2 * L2)
        # snapped to the v2 blind lattice
        d1s, d2s = snap_dir(S['d1']), snap_dir(S['d2'])
        a1s, a2s = snap(A_GRID, a1t), snap(A_GRID, a2t)
        ws = snap(W_GRID, r2)
        g = (1 - ws) * lobe(kp, a1s, d1s) + ws * lobe(kp, a2s, d2s)
        snapv = solve_with_grid(S, g)
        # half-snapped: lattice dirs+rings, TRUE w (isolates w-grid cost)
        g2 = (1 - r2) * lobe(kp, a1s, d1s) + r2 * lobe(kp, a2s, d2s)
        snap_wtrue = solve_with_grid(S, g2)
        rows.append(dict(si=si, oracle=orc, snap=snapv, snap_wtrue=snap_wtrue,
                         r2=r2))
        print(f"X{si} orc {orc:+.3f} snap {snapv:+.3f} "
              f"snapWt {snap_wtrue:+.3f} [{time.time()-t0:.0f}s]", flush=True)

    J = json.load(open('results/direction3/e42_blind_recovery.json'))
    cur = {r['si']: r['cur_grid'] for r in J['rows']}
    m2 = {r['si']: r['m2'] for r in J['rows']}
    bmaF = {r['si']: r['fit_bmaF'] for r in J['rows']}

    def paired(xs, ys):
        d = np.array([x - y for x, y in zip(xs, ys)
                      if np.isfinite(x) and np.isfinite(y)])
        se = d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), se, (d > 0).sum(), len(d)

    O = [r['oracle'] for r in rows]
    Sn = [r['snap'] for r in rows]
    Swt = [r['snap_wtrue'] for r in rows]
    C = [cur[r['si']] for r in rows]
    M = [m2[r['si']] for r in rows]
    B = [bmaF[r['si']] for r in rows]
    for nm, a, b in [('snap - oracle', Sn, O), ('snapWt - oracle', Swt, O),
                     ('oracle - cur', O, C), ('snap - cur', Sn, C),
                     ('snap - m2', Sn, M), ('snap - bmaF', Sn, B)]:
        m, se, w, n = paired(a, b)
        print(f"{nm:16s}: {m:+.4f} +- {se:.4f}  win {w}/{n}")
    json.dump(rows, open('results/direction3/e42_rebuttal_snap_D3.json', 'w'),
              indent=1, default=float)


if __name__ == '__main__':
    main()
