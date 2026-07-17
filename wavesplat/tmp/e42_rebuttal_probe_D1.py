"""D1 rebuttal probe — NOISY-ORACLE DOSE-RESPONSE (diagnostic, not a verdict).

Question: can a POINT-committed 2-lobe prior at the v2 detector's *achieved*
error levels profit at all, absent the pipeline's fallback and 48-grid mixing?
  gain(s) = score(oracle with params perturbed by s x empirical v2 errors) - cur
  s in {0.25, 0.5, 1.0}; s=0 is the stored oracle (+0.0715).
Errors resampled from the fresh-run (si=100..163) empirical pools:
  d1_err, d2_err (random sign), a2_rel (signed), w_err (signed).
Channel split at s=1: ANG-only (d1,d2 perturbed; a2,w true) vs AMP-only.
n=24 seas (si=100..123), 2 reps, fixed rng. Reuses stored cur/oracle rows.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'tmp')
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp
from e41_regime_prize_ladder import lobe_grid, tp_of_k
from d3_sa_achievability_gap import dispersion_lam

DEPTH = 50.0
G = 9.81

d = json.load(open('results/direction3/e42_blind_recovery.json'))
R = {r['si']: r for r in d['rows']}
pool_d1 = np.array([r['e_d1_err'] for r in d['rows']])
pool_d2 = np.array([r['e_d2_err'] for r in d['rows']])
pool_a2 = np.array([r['e_a2_rel'] for r in d['rows']])
pool_w = np.array([r['e_w_err'] for r in d['rows']])
print(f"error pools (fresh n=64): d1 med {np.median(pool_d1):.1f} "
      f"d2 med {np.median(pool_d2):.1f} a2 med|.| "
      f"{np.median(np.abs(pool_a2)):.3f} w med|.| "
      f"{np.median(np.abs(pool_w)):.3f}")

rng = np.random.default_rng(20260717)
S_LEVELS = [0.25, 0.5, 1.0]
REPS = 2
N = 24


def pert_grid(S, s, mode='all'):
    """Oracle 2-lobe grid with parameters perturbed at scale s."""
    dd1 = np.deg2rad(rng.choice(pool_d1) * rng.choice([-1, 1]))
    dd2 = np.deg2rad(rng.choice(pool_d2) * rng.choice([-1, 1]))
    da2 = rng.choice(pool_a2)
    dw = rng.choice(pool_w)
    if mode == 'ang':
        da2, dw = 0.0, 0.0
    if mode == 'amp':
        dd1, dd2 = 0.0, 0.0
    d1 = S['d1'] + s * dd1
    d2 = S['d2'] + s * dd2
    k2 = 2 * np.pi / dispersion_lam(S['Tp2'], DEPTH)
    Tp2 = float(tp_of_k(k2 * max(1 + s * da2, 0.2)))
    w = float(np.clip(S['r2'] + s * dw, 0.06, 0.6))
    L1 = lobe_grid(S['kp'], [(7.0, 3.3, 1.0)], [(d1, 30.0, 1.0)])
    L2 = lobe_grid(S['kp'], [(Tp2, 3.3, 1.0)], [(d2, 30.0, 1.0)])
    return (1 - w) * L1 + w * L2


rows = []
for si in range(100, 100 + N):
    S = build_sea_crosstp(si)
    ref = R[si]
    row = dict(si=si, cur=ref['cur_grid'], oracle=ref['oracle'])
    for s in S_LEVELS:
        row[f's{s}'] = float(np.mean(
            [solve_with_grid(S, pert_grid(S, s)) for _ in range(REPS)]))
    row['ang1'] = float(np.mean(
        [solve_with_grid(S, pert_grid(S, 1.0, 'ang')) for _ in range(REPS)]))
    row['amp1'] = float(np.mean(
        [solve_with_grid(S, pert_grid(S, 1.0, 'amp')) for _ in range(REPS)]))
    rows.append(row)
    print(f"si{si} cur {row['cur']:+.3f} orc {row['oracle']:+.3f} | " +
          " ".join(f"s{s} {row[f's{s}']:+.3f}" for s in S_LEVELS) +
          f" | ang1 {row['ang1']:+.3f} amp1 {row['amp1']:+.3f}", flush=True)

print(f"\n=== dose-response, gain vs cur (n={N}, reps={REPS}) ===")
cur = np.array([r['cur'] for r in rows])
orc = np.array([r['oracle'] for r in rows])
m = np.isfinite(orc)
print(f"  s=0.00 (stored oracle): {np.nanmean(orc - cur):+.4f}")
for k in ['s0.25', 's0.5', 's1.0', 'ang1', 'amp1']:
    v = np.array([r[k] for r in rows]) - cur
    se = v.std(ddof=1) / np.sqrt(len(v))
    print(f"  {k:6s}: {v.mean():+.4f} +- {se:.4f}  "
          f"(win {(v > 0).sum()}/{len(v)})")
json.dump(rows, open('results/direction3/e42_rebuttal_probe_D1.json', 'w'),
          indent=1, default=float)
print("saved -> results/direction3/e42_rebuttal_probe_D1.json")
