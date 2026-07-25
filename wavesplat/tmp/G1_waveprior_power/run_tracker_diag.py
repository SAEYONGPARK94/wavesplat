"""Tracker diagnostic — the A1 step2 finding: on sparse seas the paper's beamform
init is near-perfect (nRMSE/Hs ~0.011) but the w_phi tracker degrades it to ~0.20.
Is that degradation fixable by a SIMPLER tracker (fewer free params = less over-fit
to per-frame noise)? Compare tracking_mode ∈ {w_phi (shipped), phi_only, all} on the
same finemesh episodes, median-of-3 seeds. init_nrmse is pre-tracking (mode-independent).
"""
import json
import numpy as np
from wavesplat.run import run_estimation

FINE = 'data_finemesh_2026-07-23/'
SEEDS = (42, 1, 2)
MODES = ('w_phi', 'phi_only', 'all')
cond = json.load(open('tmp/G1_waveprior_power/a1_cond_finemesh.json'))
EPISODES = list(cond.keys())


def main():
    out = {}
    print(f"{'episode':38s} {'nc':>3s} {'init':>6s} " + " ".join(f"{m:>9s}" for m in MODES))
    for ep in EPISODES:
        row = {'ncomp': cond[ep]['ncomp']}
        init_vals = []
        for mode in MODES:
            tr = []
            for s in SEEDS:
                r = run_estimation(FINE + ep, config={'adaptive_n': True, 'seed': s,
                                                      'device': 'cpu', 'tracking_mode': mode})
                tr.append(r['avg_nRMSE'])
                init_vals.append(r['init_nrmse'])
            row[mode] = float(np.median(tr))
        row['init'] = float(np.median(init_vals))
        out[ep] = row
        print(f"{ep:38s} {row['ncomp']:3d} {row['init']:6.3f} "
              + " ".join(f"{row[m]:9.3f}" for m in MODES), flush=True)

    with open('tmp/G1_waveprior_power/tracker_diag.json', 'w') as f:
        json.dump(out, f, indent=1)

    print("\n=== TRACKER DIAG (nRMSE/Hs; init vs tracked-by-mode) ===")
    for grp, label in ((lambda c: c <= 5, 'sparse(5c)'), (lambda c: c >= 15, 'crowded(15/30c)')):
        eps = [e for e in EPISODES if grp(out[e]['ncomp'])]
        init = np.mean([out[e]['init'] for e in eps])
        print(f"  {label:16s} init {init:.3f} | " +
              " ".join(f"{m} {np.mean([out[e][m] for e in eps]):.3f}" for m in MODES))
    # verdict: does any mode preserve sparse init markedly better than w_phi?
    sp = [e for e in EPISODES if out[e]['ncomp'] <= 5]
    wp = np.mean([out[e]['w_phi'] for e in sp]); po = np.mean([out[e]['phi_only'] for e in sp])
    print(f"\n  sparse: w_phi {wp:.3f} vs phi_only {po:.3f} vs init {np.mean([out[e]['init'] for e in sp]):.3f}")
    print("  ⟹ " + ("phi_only recovers much of the lost init quality — SIMPLE tracker fix"
                     if po < wp - 0.03 else
                     "no simple mode recovers it — tracker degradation is not just over-fit"))


if __name__ == '__main__':
    main()
