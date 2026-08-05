"""F-6 (coordinator-assigned, 2026-07-22) — C8 cross-estimator independent
re-verification, "fresh reimplementation" half of dual-verification.

DOES NOT import or read tmp/c8_cross_estimator.py (by instruction). Only reuses
scripts/d1_crb's fisher/jacobian/predict/collect_obs/omega_of_k (explicitly
permitted) -- the query loop, phase-averaging, and BOTH estimators (dense
ridge-LS and a from-scratch sparse greedy matching-pursuit, standing in for
"relax") are written independently here.

Question: does sigma_CRB (a Gaussian-noise field-reconstruction CRB, estimator-
agnostic by construction) predict the actual error of TWO DIFFERENT estimator
classes -- not just its own ridge-LS twin?

Method:
  - 2 episodes: BF4_15comp seed{0,1}_uniform (under-determined regime).
  - obs = predict(theta_r, .) + N(0, 0.02) at the REAL lidar sample geometry
    (collect_obs, t <= t_now = mid-episode frame), subsampled to <=1500 pts.
  - query = gt grid (strided to ~400 pts) x leads [0,2,4,6] (abs time = t_now+lead).
  - R=16 independent phase realizations (theta_r = theta0 with phi redrawn
    ~U(0,2pi) per component -- same convention as the project's other envelope-
    validation experiments, i.e. what "sigma_CRB / sigma_actual" mean in this
    codebase) -- CRB variance and both estimators' squared error are accumulated
    over the SAME R draws (paired), then RMS'd and normalized by Hs.
  - Estimator A: dense k-grid ridge-LS (ring 0.35-1.75 kp, NK=15 per axis,
    ridge_rel=1e-3) -- same "vehicle" family as the CRB's own model, i.e. NOT
    an independent estimator class.
  - Estimator B: from-scratch greedy Orthogonal-Matching-Pursuit (OMP) sparse
    selection over the SAME candidate k-grid -- greedily picks the single
    (kx,ky) atom-pair that best explains the current residual (periodogram-
    style power), refits jointly-selected atoms by least squares, repeats for
    K_ATOMS=15 steps (matches the known BF4_15comp component budget -- NOT the
    true kx/ky/phi, just a component-count budget, same kind of prior info
    RELAX-family estimators in this project normally use). This is a genuinely
    different estimator CLASS (sparse/greedy/nonlinear selection vs dense
    linear regression), matching the intent of "relax-like."

Usage: python3 tmp/f6_c8_own_verify.py
Out:   tmp/f6_c8_verify_raw.json (+ printed summary; report written to
       tmp/F6_c8_verify_2026-07-22.md by hand after inspecting output)
"""
import os
import sys
import json
import time

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from wavesim.data_writer import load_episode
from d1_crb import DEVICE, DT, fisher, jacobian, predict, collect_obs
from wavesplat.model.dispersion import G

SIGMA = 0.02
DEPTH = 50.0
LEADS = [0.0, 2.0, 4.0, 6.0]
R = 16
NK = 15
RING = (0.35, 1.75)
RIDGE_REL = 1e-3
K_ATOMS = 15
MAX_OBS = 1500
NQ_TARGET = 400
EPISODES = [
    'data/episode_BF4_15comp_seed0_uniform.h5',
    'data/episode_BF4_15comp_seed1_uniform.h5',
]
OUT_JSON = 'tmp/f6_c8_verify_raw.json'


def load_ep_own(path, depth=DEPTH):
    """Minimal independent loader (not d1_crb.load_ep) -- same data, own code."""
    d = load_episode(path)
    wc = np.array(d['wave_components'])            # [A, omega, kx, ky, phi, Q]
    theta = torch.tensor(np.stack([wc[:, 0], wc[:, 2], wc[:, 3], wc[:, 4]], 1),
                         dtype=DT, device=DEVICE)
    ts = np.array(d['timestamps'])
    lp = np.array(d['lidar_points']); vm = np.array(d['lidar_valid_mask']).astype(bool)
    gx = np.array(d['gt_grid_x']); gy = np.array(d['gt_grid_y'])
    Hs = float(d['Hs_true'])
    return dict(theta=theta, ts=ts, lp=lp, vm=vm, gx=gx, gy=gy, Hs=Hs, depth=depth)


def dom_kp(theta):
    w = theta[:, 0].abs()
    i = int(torch.argmax(w))
    return float(torch.hypot(theta[i, 1], theta[i, 2]))


def build_kgrid(kp, nk=NK, ring=RING):
    ax = np.linspace(-ring[1] * kp, ring[1] * kp, nk)
    KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel()
    km = np.hypot(kx, ky) / kp
    keep = (km >= ring[0]) & (km <= ring[1])
    return kx[keep], ky[keep]


def design_np(kxg, kyg, x, y, t, depth):
    """Own numpy design matrix (cos, sin cols), matches d1_crb's omega_of_k
    physics via the SAME imported G constant -- not copied from any other
    envelope-validation script."""
    k = np.hypot(kxg, kyg)
    om = np.sqrt(G * k * np.tanh(k * depth) + 1e-30)
    psi = kxg[None, :] * x[:, None] + kyg[None, :] * y[:, None] - om[None, :] * t[:, None]
    return np.cos(psi), np.sin(psi)


def omp_fit(Cc, Cs, z, k_atoms=K_ATOMS):
    """Own from-scratch greedy Orthogonal Matching Pursuit over (cos,sin) atom
    pairs. Selection score = periodogram-style projected power onto the
    residual; refit ALL active atoms jointly by least squares each step."""
    n_cand = Cc.shape[1]
    norm_c = (Cc ** 2).sum(0) + 1e-12
    norm_s = (Cs ** 2).sum(0) + 1e-12
    active = []
    r = z.copy()
    c_act = np.zeros(0)
    for _ in range(min(k_atoms, n_cand)):
        proj_c = Cc.T @ r
        proj_s = Cs.T @ r
        power = proj_c ** 2 / norm_c + proj_s ** 2 / norm_s
        power[active] = -np.inf
        j = int(np.argmax(power))
        active.append(j)
        cols = []
        for a in active:
            cols.append(Cc[:, a]); cols.append(Cs[:, a])
        Phi_act = np.stack(cols, axis=1)
        c_act, *_ = np.linalg.lstsq(Phi_act, z, rcond=None)
        r = z - Phi_act @ c_act
    return active, c_act


def omp_predict(Ccq, Csq, active, c_act):
    pred = np.zeros(Ccq.shape[0])
    for idx, a in enumerate(active):
        pred += c_act[2 * idx] * Ccq[:, a] + c_act[2 * idx + 1] * Csq[:, a]
    return pred


def run_episode(path, seed):
    t0 = time.time()
    ep = load_ep_own(path)
    theta0 = ep['theta']; N = theta0.shape[0]; Hs = ep['Hs']; depth = ep['depth']
    kp = dom_kp(theta0)
    t_now = float(ep['ts'][len(ep['ts']) // 2])

    x_t, y_t, t_t = collect_obs(ep, t_now)            # (M,) torch, d1_crb reused
    rng = np.random.default_rng(seed)
    if x_t.numel() > MAX_OBS:
        idx = rng.choice(x_t.numel(), MAX_OBS, replace=False)
        idx_t = torch.tensor(idx, device=DEVICE)
        x_t, y_t, t_t = x_t[idx_t], y_t[idx_t], t_t[idx_t]
    x_np, y_np, t_np = x_t.cpu().numpy(), y_t.cpu().numpy(), t_t.cpu().numpy()
    n_obs = x_t.numel()

    # --- query points: gt grid strided x leads (OWN loop) -----------------
    nx, ny = ep['gx'].size, ep['gy'].size
    stride = max(1, int(round(np.sqrt(nx * ny / NQ_TARGET))))
    GX, GY = np.meshgrid(ep['gx'][::stride], ep['gy'][::stride])
    qx0, qy0 = GX.ravel(), GY.ravel()
    qx_all, qy_all, qt_all = [], [], []
    for lead in LEADS:
        qx_all.append(qx0); qy_all.append(qy0)
        qt_all.append(np.full_like(qx0, t_now + lead))
    qx_np = np.concatenate(qx_all); qy_np = np.concatenate(qy_all); qt_np = np.concatenate(qt_all)
    qx_t = torch.tensor(qx_np, dtype=DT, device=DEVICE)
    qy_t = torch.tensor(qy_np, dtype=DT, device=DEVICE)
    qt_t = torch.tensor(qt_np, dtype=DT, device=DEVICE)
    n_q = qx_np.size

    # --- candidate k-grid (shared dictionary for BOTH estimators) ---------
    kxg, kyg = build_kgrid(kp)
    Cc_obs, Cs_obs = design_np(kxg, kyg, x_np, y_np, t_np, depth)     # (n_obs, ncand)
    Cc_q, Cs_q = design_np(kxg, kyg, qx_np, qy_np, qt_np, depth)      # (n_q, ncand)
    n_cand = kxg.size

    # ridge design + closed-form solve pieces (obs geometry fixed across R,
    # so the inverse is realization-independent -- precompute once)
    Phi_obs = np.concatenate([Cc_obs, Cs_obs], axis=1)     # (n_obs, 2*ncand)
    Phi_q = np.concatenate([Cc_q, Cs_q], axis=1)            # (n_q, 2*ncand)
    Amat = Phi_obs.T @ Phi_obs
    lam = RIDGE_REL * np.trace(Amat) / Amat.shape[0]
    Ainv = np.linalg.inv(Amat + lam * np.eye(Amat.shape[0]))

    crb_acc = np.zeros(n_q)
    se_ridge_acc = np.zeros(n_q)
    se_omp_acc = np.zeros(n_q)

    for r in range(R):
        phi_r = rng.uniform(0.0, 2 * np.pi, N)
        theta_r = theta0.clone()
        theta_r[:, 3] = torch.tensor(phi_r, dtype=DT, device=DEVICE)

        # --- own phase-averaged CRB accumulation ---------------------------
        Fr = fisher(theta_r, x_t, y_t, t_t, depth, SIGMA)
        nF = Fr.shape[0]
        Finv = torch.linalg.inv(Fr + 1e-10 * torch.trace(Fr) / nF
                                * torch.eye(nF, dtype=DT, device=DEVICE))
        Jq = jacobian(theta_r, qx_t, qy_t, qt_t, depth)
        var_q = torch.einsum('qa,ab,qb->q', Jq, Finv, Jq)
        crb_acc += var_q.cpu().numpy()

        # --- synth obs + truth at query, same theta_r (paired) -------------
        z_true_obs = predict(theta_r, x_t, y_t, t_t, depth).cpu().numpy()
        zobs = z_true_obs + SIGMA * rng.standard_normal(n_obs)
        truth_q = predict(theta_r, qx_t, qy_t, qt_t, depth).cpu().numpy()

        # --- estimator A: dense ridge-LS -----------------------------------
        c_ridge = Ainv @ (Phi_obs.T @ zobs)
        pred_ridge = Phi_q @ c_ridge
        se_ridge_acc += (pred_ridge - truth_q) ** 2

        # --- estimator B: own greedy OMP ------------------------------------
        active, c_act = omp_fit(Cc_obs, Cs_obs, zobs, K_ATOMS)
        pred_omp = omp_predict(Cc_q, Cs_q, active, c_act)
        se_omp_acc += (pred_omp - truth_q) ** 2

    sig_crb = np.sqrt(np.clip(crb_acc / R, 0, None)) / Hs
    sig_ridge = np.sqrt(se_ridge_acc / R) / Hs
    sig_omp = np.sqrt(se_omp_acc / R) / Hs

    r_crb_ridge = stats.spearmanr(sig_crb, sig_ridge).correlation
    r_crb_omp = stats.spearmanr(sig_crb, sig_omp).correlation
    r_ridge_omp = stats.spearmanr(sig_ridge, sig_omp).correlation

    # distance guard (own, cheap context -- not in the coordinator's ask but
    # useful to flag a geometric confound if one shows up)
    d2 = ((qx_np[:, None] - x_np[None, :]) ** 2
          + (qy_np[:, None] - y_np[None, :]) ** 2).min(axis=1)
    dist = np.sqrt(d2)
    r_dist_ridge = stats.spearmanr(dist, sig_ridge).correlation
    r_dist_omp = stats.spearmanr(dist, sig_omp).correlation

    out = dict(ep=os.path.basename(path), n_obs=int(n_obs), n_q=int(n_q),
              n_cand=int(n_cand), kp=kp, Hs=Hs,
              spear_crb_ridge=float(r_crb_ridge), spear_crb_omp=float(r_crb_omp),
              spear_ridge_omp=float(r_ridge_omp),
              spear_dist_ridge=float(r_dist_ridge), spear_dist_omp=float(r_dist_omp))
    print(f"  {os.path.basename(path):40s} n_obs={n_obs} n_cand={n_cand} n_q={n_q} "
          f"kp={kp:.3f}\n"
          f"    Spearman(CRB,ridge)={r_crb_ridge:+.3f}  Spearman(CRB,OMP)={r_crb_omp:+.3f}  "
          f"Spearman(ridge,OMP)={r_ridge_omp:+.3f}\n"
          f"    [dist guard] Spearman(dist,ridge)={r_dist_ridge:+.3f}  "
          f"Spearman(dist,OMP)={r_dist_omp:+.3f}  [{time.time()-t0:.0f}s]")
    return out


def main():
    results = []
    for i, p in enumerate(EPISODES):
        results.append(run_episode(p, seed=1000 + i))
    med = {}
    for key in ('spear_crb_ridge', 'spear_crb_omp', 'spear_ridge_omp'):
        med[key] = float(np.median([r[key] for r in results]))
    print(f"\n=== MEDIAN across {len(results)} episodes ===")
    for k, v in med.items():
        print(f"  {k}: {v:+.3f}")
    json.dump(dict(episodes=results, median=med,
                   config=dict(R=R, NK=NK, ring=RING, ridge_rel=RIDGE_REL,
                              k_atoms=K_ATOMS, max_obs=MAX_OBS, nq_target=NQ_TARGET,
                              leads=LEADS)),
              open(OUT_JSON, 'w'), indent=1, default=float)
    print(f"saved -> {OUT_JSON}")
    return results, med


if __name__ == '__main__':
    main()
