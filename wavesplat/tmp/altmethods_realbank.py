"""
altmethods_realbank.py — DEFINITIVE arm of the non-beamforming family gate.

Instead of my own re-implementation of a bank estimator (which risks being a strawman),
this calls the PROJECT'S OWN tuned estimator `mmse_bank_estimate`
(`scripts/d3_sa_nomp.py:499`) — the 2026-07-05 winner, with beamform-shaped prior,
val-selected (pexp, smult) and val-gated SBL-EM — on the same episodes, same frames,
same field-nRMSE metric as the beamforming (greedy+joint) arm.

Pre-registered read (before numbers):
  - BANK(real) ~ BEAM      -> the two families tie on FIELD reconstruction; the earlier
                              flat-bank 4x gap was purely a strawman artifact.
  - BANK(real) ~ ZERO-PRED -> the bank family genuinely cannot reconstruct the field off
                              the observed strip (it is tuned for prediction NEAR/ALONG
                              the track, which is the companion paper's metric), and the
                              gate is a clean negative for "swap the estimator family" as
                              a novelty lever.
  - BANK(real) < BEAM      -> would overturn the beamforming pipeline; would need
                              immediate replication before any claim.
Either way this says nothing about the companion paper, whose metric is encounter
correlation along the ship track, not wide-grid field nRMSE.
"""
import sys, numpy as np, time
sys.path.insert(0, 'tmp')
sys.path.insert(0, 'scripts')
from blasso_feasibility import (load_obs, greedy, joint_refine, field_nrmse,
                                build_grid, corr_at)
from d3_sa_nomp import mmse_bank_estimate

RNG = np.random.default_rng(0)
EPS = [('episode_BF3_5comp_seed0_uniform.h5', 5),
       ('episode_BF3_5comp_seed1_uniform.h5', 5),
       ('episode_BF4_15comp_seed0_uniform.h5', 15)]


def run(fn, n_true, nsub=2600):
    obs = load_obs('data/' + fn)
    frames = [obs['nfr'] // 4, obs['nfr'] // 2, 3 * obs['nfr'] // 4]
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(nsub, M), replace=False)
    xs, ys, ts, zs = obs['x'][sub], obs['y'][sub], obs['t'][sub], obs['z'][sub]
    P = np.column_stack([xs, ys, ts])
    sigma0 = 0.05 * obs['Hs']

    gx, gy = obs['gx'], obs['gy']
    X, Y = np.meshgrid(gx, gy, indexing='xy')

    # --- BANK (project's own tuned estimator), queried at the same grid frames
    t0 = time.time()
    bank_nr = []
    for f in frames:
        Pq = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, obs['ts'][f])])
        pred = mmse_bank_estimate(P, zs, obs['depth'], Pq, sigma0).reshape(X.shape)
        bank_nr.append(float(np.sqrt(np.mean((pred - obs['gtf'][f]) ** 2)) / obs['Hs']))
    t_bank = time.time() - t0

    # --- BEAM (greedy + joint), same recipe / same frames
    gridk = build_grid()
    lam_b = 0.03 * np.abs(corr_at(*gridk, xs, ys, ts, zs, obs['depth']).max())
    gj = joint_refine(greedy(obs, n_true + 1, gridk, sub), obs, sub, lam_b)
    beam = np.mean([field_nrmse(gj, obs, f) for f in frames])
    z0 = np.mean([float(np.sqrt(np.mean(obs['gtf'][f] ** 2)) / obs['Hs']) for f in frames])

    print(f"\n{fn}")
    print(f"   field nRMSE (avg 3 frames):  ZERO-PRED={z0:.4f}   BEAM(g+j)={beam:.4f}   "
          f"BANK-MMSE(project's own, tuned)={np.mean(bank_nr):.4f}  "
          f"{[f'{v:.3f}' for v in bank_nr]}   [{t_bank:.1f}s]")


if __name__ == "__main__":
    for fn, nt in EPS:
        try:
            run(fn, nt)
        except Exception:
            import traceback; traceback.print_exc()
