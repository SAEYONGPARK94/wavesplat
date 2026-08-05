"""e33 — IS THE WELL-SPECIFIED ENSEMBLE MMSE ALREADY BAYES-OPTIMAL?

THE ARGUMENT I now believe closes this (and it is NOT the one I was asked to
defeat):

For a LINEAR GAUSSIAN sea with a KNOWN spectrum, the posterior mean E[field|z]
is EXACTLY LINEAR in z, and it is EXACTLY the MMSE solve with the ensemble
prior. It is Bayes-optimal. NOTHING that sees only z can beat it -- not a
learned prior, not a sharper prior, not a deconvolution.

Under that reading the gate-0 ladder's rung4 is NOT an attainable ceiling.
gamma_true is ORACLE SIDE INFORMATION about the realization -- information
that is not in z. Its +0.216 is the value of a cheat, not a headroom. That is
why e32 found EVERY data-derived prior loses to the constant, and why the
trained net (task loss, free to represent any of them) collapsed to a
constant: the constant IS the optimum.

TEST: build the WELL-SPECIFIED ensemble prior (true mean direction, true
spread=30, true signal variance, TRUE noise s0 rather than the inflated
smult=3) and compare to the learned incumbent.
  If well-spec >= cur, the hand Bayes prior BEATS the learned net and the
  gamma class is closed BY THEOREM -- the only headroom left is NUISANCE
  estimation (mu, spread, noise), not "a better prior net".

Also decompose WHERE the residual nuisance headroom is: the ladder's
rung3-rung2 = +0.053 is dominated by ONE sea with a 180deg beamform flip.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, bank_atoms, design_t, corr, DEVICE,
                          DEPTH, _MASK, REGIMES, NGRID)
from e30_conditionality_probe import radial_of, ens_grid_norm
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/e33_wellspec.json'


def solve_cfg(S, g_free_grid, smult=3.0, sig_var=None):
    """ladder solver, but with configurable noise multiplier / signal variance."""
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    s0 = S['sigma0']
    gb = np.zeros_like(g_free_grid)
    g = np.concatenate([g_free_grid[_MASK], gb[_MASK]])
    if g.sum() <= 0:
        return float('nan')
    g = g / g.sum()
    kxb, kyb, om = bank_atoms(S['kp'])
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(P))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(Pq))
    sv = float(np.var(z)) if sig_var is None else sig_var
    gam = t(np.concatenate([g, g]) * sv)
    K = (Phi * gam[None]) @ Phi.T + (smult * s0) ** 2 * torch.eye(len(z), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(z))
    return float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), gt))


def main(n_seas=16):
    kn, en = radial_of(7.0, 3.3)
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    lad = json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']
    R = REGIMES['spread']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        mu_hat = np.arctan2(by, bx)
        s0 = S['sigma0']
        sv_true = max(float(np.var(S['z'])) - s0 ** 2, 1e-9)

        g_hat = ens_grid_norm(kn, en, mu_hat, S['kp'], 30.0)
        g_true = ens_grid_norm(kn, en, mu_true, S['kp'], 30.0)

        r = dict(si=si, cur=ref[si]['cur'],
                 dir_err=float(np.rad2deg(np.abs(np.angle(
                     np.exp(1j * (mu_hat - mu_true)))))))
        # ladder configs (smult=3, var(z))
        r['ens_hat_s3'] = solve_cfg(S, g_hat, 3.0)
        r['ens_true_s3'] = solve_cfg(S, g_true, 3.0)
        # WELL-SPECIFIED: true noise, true signal variance
        r['ens_hat_ws'] = solve_cfg(S, g_hat, 1.0, sv_true)
        r['ens_true_ws'] = solve_cfg(S, g_true, 1.0, sv_true)
        rows.append(r)
        print(f'sea{si:2d} cur {r["cur"]:+.3f} | ens_hat_s3 {r["ens_hat_s3"]:+.3f} '
              f'| ens_true_s3 {r["ens_true_s3"]:+.3f} | ens_hat_WS {r["ens_hat_ws"]:+.3f} '
              f'| ens_true_WS {r["ens_true_ws"]:+.3f} | derr {r["dir_err"]:5.1f}',
              flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')
    print(f'\n=== n={n_seas}. learned incumbent cur = {cur.mean():+.4f} ===')
    for k, lab in [('ens_hat_s3', 'ens(mu_hat) smult3        [=ladder rung2]'),
                   ('ens_true_s3', 'ens(mu_TRUE) smult3       [=ladder rung3]'),
                   ('ens_hat_ws', 'ens(mu_hat)  WELL-SPEC    '),
                   ('ens_true_ws', 'ens(mu_TRUE) WELL-SPEC    ')]:
        v = a(k); d = v - cur
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  {lab:42s} {v.mean():+.4f}  vs cur {d.mean():+.4f} '
              f'+- {se:.4f} (t={d.mean()/se:+.2f}, pos {int((d>0).sum())}/{len(d)})')

    # where does the direction headroom live?
    derr = a('dir_err')
    gain = a('ens_true_s3') - a('ens_hat_s3')
    flip = derr > 90
    print(f'\n=== direction headroom (rung3-rung2 = {gain.mean():+.4f}) ===')
    print(f'  seas with a 180-flip (dir_err>90): {int(flip.sum())}/{n_seas} '
          f'-> their gain {gain[flip].mean() if flip.sum() else 0:+.4f} '
          f'(contributes {gain[flip].sum()/n_seas:+.4f} of the mean)')
    print(f'  the other {int((~flip).sum())} seas (mean derr '
          f'{derr[~flip].mean():.1f}deg) -> gain {gain[~flip].mean():+.4f} '
          f'(contributes {gain[~flip].sum()/n_seas:+.4f})')
    print(f'\n  => fixing ONLY the 180 flip buys {gain[flip].sum()/n_seas:+.4f} '
          f'over rung2 with zero learning')
    json.dump(dict(rows=rows), open(OUT, 'w'), indent=1)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
