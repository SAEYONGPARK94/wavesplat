"""Is the strong oracle's FORECAST blow-up a fit-conditioning artifact?
The true-support LS oracle fits 60 components' A,φ from ~500 obs (120 unknowns) and
propagates forward. Under-regularization overfits → forecast phase drift. Sweep
lambda_reg and see whether a well-conditioned oracle becomes a stable, low forecast
ceiling (⟹ real forecast headroom the learned methods leave) or stays high (⟹ the
forecast ceiling is genuinely limited, not just my oracle).
Reference: cond forecast ≈0.813, oracle_g (mmse true-γ) forecast ≈0.691 (from expB2).
"""
import argparse
import json
import time
import numpy as np

from run_mechanism import hos_scene_known
from run_expB2 import per_lead_oracle_ls, band

LAMS = (0.5, 5.0, 20.0, 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=6)
    a = ap.parse_args()
    now = {lam: [] for lam in LAMS}
    far = {lam: [] for lam in LAMS}
    for r in range(a.reps):
        t0 = time.time()
        sc = hos_scene_known(2.0, 8.0, r, nc=60)
        leads = sc['leads']
        msg = []
        for lam in LAMS:
            pl = per_lead_oracle_ls(sc, lam=lam)
            now[lam].append(band(pl, leads, 0, 0))
            far[lam].append(band(pl, leads, 10, 20))
            msg.append(f"λ{lam:g} now {now[lam][-1]:.2f} far {far[lam][-1]:.2f}")
        print(f"[{r}/{a.reps}] " + " | ".join(msg) + f" | {time.time()-t0:.0f}s", flush=True)

    out = {str(lam): {'now': float(np.mean(now[lam])), 'now_max': float(np.max(now[lam])),
                      'far': float(np.mean(far[lam])), 'far_max': float(np.max(far[lam]))}
           for lam in LAMS}
    json.dump(out, open('tmp/G1_waveprior_power/oracle_cond.json', 'w'), indent=1)
    print("\n=== ORACLE CONDITIONING (oracle_ls nRMSE/gt_std) ===")
    print(f"{'lambda':>8s} {'now_mean':>9s} {'now_max':>8s} {'far_mean':>9s} {'far_max':>8s}")
    for lam in LAMS:
        o = out[str(lam)]
        print(f"{lam:8g} {o['now']:9.3f} {o['now_max']:8.3f} {o['far']:9.3f} {o['far_max']:8.3f}")
    best = min(LAMS, key=lambda L: out[str(L)]['far'])
    bf = out[str(best)]['far']
    print(f"\n  best-conditioned forecast oracle: λ={best:g} → far {bf:.3f}  (cond far ≈0.813)")
    print("  ⟹ " + ("well-conditioned oracle IS a low, stable forecast ceiling ⟹ real forecast headroom"
                     if bf < 0.813 - 0.05 else
                     "even well-conditioned, forecast ceiling stays ≈ cond ⟹ no clean forecast headroom"))


if __name__ == '__main__':
    main()
