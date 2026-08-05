"""C8 — cross-estimator envelope test. Prereg = tmp/C8_prereg_2026-07-22.md.
Reuses e58 CRB/envelope machinery; swaps ONLY the sigma_actual estimator across
three model classes: dense-linear ridge (E_ridge, CRB's ML twin), sparse greedy
nonlinear RELAX (E_relax), nonparametric dispersion-GP (E_gp). Does sigma_CRB
predict the spatial error of estimators OTHER than its ML twin? = the only direct
test of the "estimator-INDEPENDENT" headline (E-59 flagged as untested C8).

Usage: python3 tmp/c8_cross_estimator.py [--smoke] [--n 12] [--R 8]
Out:   results/direction3/c8_cross_estimator.json
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts')

from scripts.d1_crb import (DEVICE, DT, load_ep, collect_obs, jacobian, fisher,
                            predict, omega_of_k)

OUT = 'results/direction3/c8_cross_estimator.json'
SIGMA = 0.02
LEADS = [0.0, 2.0, 4.0, 6.0]
RING = (0.35, 1.75)
NK = 15            # ridge dense grid (matches e58)
NK_RELAX = 25      # finer grid for greedy peak selection
K_BUDGET = 12      # non-oracle fixed sparse budget


def dom_kp(theta):
    w = theta[:, 0].abs()
    i = int(torch.argmax(w))
    return float(torch.hypot(theta[i, 1], theta[i, 2]))


def build_kgrid(kp, nk):
    ax = np.linspace(-RING[1] * kp, RING[1] * kp, nk)
    KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel()
    km = np.hypot(kx, ky) / kp
    keep = (km >= RING[0]) & (km <= RING[1])
    return (torch.tensor(kx[keep], dtype=DT, device=DEVICE),
            torch.tensor(ky[keep], dtype=DT, device=DEVICE))


def design(kxg, kyg, x, y, t, depth):
    k = torch.hypot(kxg, kyg)
    om = omega_of_k(k, depth)
    psi = (kxg[None, :] * x[:, None] + kyg[None, :] * y[:, None]
           - om[None, :] * t[:, None])
    return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)


# ---------- three estimators: fit on obs (x,y,t,z), predict at (qx,qy,qt) ----------
def fit_ridge(kxg, kyg, x, y, t, z, depth, ridge_rel=1e-3):
    Phi = design(kxg, kyg, x, y, t, depth)
    A = Phi.T @ Phi
    lam = ridge_rel * torch.trace(A) / A.shape[0]
    Ainv = torch.linalg.inv(A + lam * torch.eye(A.shape[0], dtype=DT, device=DEVICE))
    c = Ainv @ (Phi.T @ z)
    return c


def pred_ridge(c, kxg, kyg, qx, qy, qt, depth):
    return design(kxg, kyg, qx, qy, qt, depth) @ c


def fit_relax(kxg, kyg, x, y, t, z, depth, K=K_BUDGET):
    """Greedy sparse (RELAX/matching-pursuit style): iteratively pick the grid
    wavevector whose [cos,sin] atom best explains the residual, then joint-LS
    re-fit all selected atoms. Different model class from dense ridge."""
    Ncand = kxg.numel()
    cols = design(kxg, kyg, x, y, t, depth)            # (M, 2N) [cos|sin]
    cos_c, sin_c = cols[:, :Ncand], cols[:, Ncand:]
    r = z.clone()
    sel = []
    for _ in range(min(K, Ncand)):
        # matched power of each candidate on current residual
        pc = (cos_c.T @ r) ** 2 + (sin_c.T @ r) ** 2   # (N,)
        pc = pc.clone()
        if sel:
            pc[torch.tensor(sel, device=DEVICE)] = -1.0
        j = int(torch.argmax(pc))
        sel.append(j)
        idx = torch.tensor(sel, device=DEVICE)
        Phi_s = torch.cat([cos_c[:, idx], sin_c[:, idx]], dim=1)   # (M, 2|sel|)
        coef = torch.linalg.lstsq(Phi_s, z).solution
        r = z - Phi_s @ coef
    return torch.tensor(sel, device=DEVICE), coef


def pred_relax(sel, coef, kxg, kyg, qx, qy, qt, depth):
    ks_x, ks_y = kxg[sel], kyg[sel]
    Phi_q = design(ks_x, ks_y, qx, qy, qt, depth)      # (Q, 2|sel|)
    return Phi_q @ coef


def fit_gp(x, y, t, z, kp, depth, t_ref):
    """Nonparametric Matern GP. Propagate obs to t_ref by dominant-k phase shift,
    fit spatial GP. Returns callable predict(qx,qy,qt)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    xn, yn, tn, zn = (a.cpu().numpy() for a in (x, y, t, z))
    # dominant wavevector by coarse periodogram over obs (non-oracle direction)
    kxg, kyg = build_kgrid(kp, 13)
    cols = design(kxg, kyg, x, y, t, depth)
    Ncand = kxg.numel()
    pc = (cols[:, :Ncand].T @ z) ** 2 + (cols[:, Ncand:].T @ z) ** 2
    j = int(torch.argmax(pc))
    kdx, kdy = float(kxg[j]), float(kyg[j])
    om = float(omega_of_k(torch.hypot(kxg[j], kyg[j]), depth))
    kmag2 = kdx ** 2 + kdy ** 2 + 1e-12
    # phase speed vector to advect points to t_ref: x_ref = x + (c)*(t_ref-t)
    cx, cy = om * kdx / kmag2, om * kdy / kmag2
    Xx = xn + cx * (t_ref - tn)
    Yy = yn + cy * (t_ref - tn)
    lam_p = 2 * np.pi / np.sqrt(kmag2)
    ker = Matern(length_scale=lam_p / 3, nu=2.5,
                 length_scale_bounds=(lam_p / 20, lam_p * 3)) + \
        WhiteKernel(noise_level=SIGMA ** 2, noise_level_bounds='fixed')
    gp = GaussianProcessRegressor(kernel=ker, alpha=SIGMA ** 2,
                                  normalize_y=True, n_restarts_optimizer=0)
    gp.fit(np.column_stack([Xx, Yy]), zn)

    def _pred(qx, qy, qt):
        qxn, qyn, qtn = (a.cpu().numpy() for a in (qx, qy, qt))
        Qx = qxn + cx * (t_ref - qtn)
        Qy = qyn + cy * (t_ref - qtn)
        out = gp.predict(np.column_stack([Qx, Qy]))
        return torch.tensor(out, dtype=DT, device=DEVICE)
    return _pred


def run_episode(ep, t_now, kp, R=8):
    depth, Hs = ep['depth'], ep['Hs']
    theta0 = ep['theta']
    N = theta0.shape[0]
    x, y, t = collect_obs(ep, t_now)
    if x.numel() > 4000:
        idx = torch.randperm(x.numel(), device=DEVICE)[:4000]
        x, y, t = x[idx], y[idx], t[idx]
    kxg, kyg = build_kgrid(kp, NK)
    kxg_r, kyg_r = build_kgrid(kp, NK_RELAX)
    gx, gy = ep['gx'], ep['gy']
    stride = max(1, (gx.numel() * gy.numel()) // 400)
    Gx, Gy = torch.meshgrid(gx, gy, indexing='xy')
    qx0, qy0 = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
    ests = ['ridge', 'relax', 'gp']
    acc = {e: torch.zeros(qx0.numel() * len(LEADS), dtype=DT, device=DEVICE) for e in ests}
    crb_acc = torch.zeros(qx0.numel() * len(LEADS), dtype=DT, device=DEVICE)
    # emp-UQ (phase-independent) from ridge geometry
    Phi = design(kxg, kyg, x, y, t, depth)
    A = Phi.T @ Phi
    lam = 1e-3 * torch.trace(A) / A.shape[0]
    Ainv = torch.linalg.inv(A + lam * torch.eye(A.shape[0], dtype=DT, device=DEVICE))
    emp_parts, dist_parts = [], []
    for lead in LEADS:
        qt = torch.full_like(qx0, float(t_now + lead))
        Phi_q = design(kxg, kyg, qx0, qy0, qt, depth)
        ev = torch.einsum('qa,ab,qb->q', Phi_q, Ainv, Phi_q)
        emp_parts.append(torch.sqrt(torch.clamp(ev, min=0)) * SIGMA / Hs)
        d2 = ((qx0[:, None] - x[None, :]) ** 2 + (qy0[:, None] - y[None, :]) ** 2).min(1).values
        dist_parts.append(torch.sqrt(d2))
    sig_emp = torch.cat(emp_parts).cpu().numpy()
    dist = torch.cat(dist_parts).cpu().numpy()

    for _ in range(R):
        th = theta0.clone()
        th[:, 3] = torch.rand(N, dtype=DT, device=DEVICE) * (2 * np.pi)
        s = predict(th, x, y, t, depth)
        z = s + SIGMA * torch.randn_like(s)
        c_ridge = fit_ridge(kxg, kyg, x, y, t, z, depth)
        sel, coef = fit_relax(kxg_r, kyg_r, x, y, t, z, depth)
        gp_pred = fit_gp(x, y, t, z, kp, depth, float(t_now))
        crb_parts, r_parts, x_parts, g_parts = [], [], [], []
        for lead in LEADS:
            qt = torch.full_like(qx0, float(t_now + lead))
            truth = predict(th, qx0, qy0, qt, depth)
            # phase-averaged CRB
            Fr = fisher(th, x, y, t, depth, SIGMA)
            nF = Fr.shape[0]
            Finv = torch.linalg.inv(Fr + 1e-10 * torch.trace(Fr) / nF
                                    * torch.eye(nF, dtype=DT, device=DEVICE))
            Jq = jacobian(th, qx0, qy0, qt, depth)
            crb_parts.append(torch.einsum('qa,ab,qb->q', Jq, Finv, Jq))
            r_parts.append((pred_ridge(c_ridge, kxg, kyg, qx0, qy0, qt, depth) - truth) ** 2)
            x_parts.append((pred_relax(sel, coef, kxg_r, kyg_r, qx0, qy0, qt, depth) - truth) ** 2)
            g_parts.append((gp_pred(qx0, qy0, qt) - truth) ** 2)
        crb_acc += torch.cat(crb_parts)
        acc['ridge'] += torch.cat(r_parts)
        acc['relax'] += torch.cat(x_parts)
        acc['gp'] += torch.cat(g_parts)
    sig_crb = (torch.sqrt(torch.clamp(crb_acc / R, min=0)) / Hs).cpu().numpy()
    sig = {e: (torch.sqrt(acc[e] / R) / Hs).cpu().numpy() for e in ests}

    def spear(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(stats.spearmanr(a[m], b[m]).correlation)

    def partial(a, b, d):
        rab, rad, rbd = spear(a, b), spear(a, d), spear(b, d)
        den = np.sqrt(max(1e-9, (1 - rad ** 2) * (1 - rbd ** 2)))
        return (rab - rad * rbd) / den

    out = dict(n_q=int(len(sig_crb)), kp=kp, n_obs=int(x.numel()))
    for e in ests:
        out[f'crb_{e}'] = spear(sig_crb, sig[e])
        out[f'crb_{e}_pd'] = partial(sig_crb, sig[e], dist)
        out[f'emp_{e}'] = spear(sig_emp, sig[e])
        out[f'skill_{e}'] = float(np.nanmedian(sig[e]))   # competence proxy
    out['agree_ridge_relax'] = spear(sig['ridge'], sig['relax'])
    out['agree_ridge_gp'] = spear(sig['ridge'], sig['gp'])
    out['agree_relax_gp'] = spear(sig['relax'], sig['gp'])
    return out


def med_ci(vals, B=2000, seed=0):
    v = np.array([x for x in vals if np.isfinite(x)])
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(v, len(v))) for _ in range(B)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--R', type=int, default=8)
    ap.add_argument('--regime', choices=['all', 'well', 'under'], default='all')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    out_path = a.out
    n = 2 if a.smoke else a.n
    if a.regime == 'under':          # under-determined: 15/30-comp seas only
        g = sorted(glob.glob('data/episode_BF4_15comp_*.h5')
                   + glob.glob('data/episode_BF4_30comp_*.h5'))
    elif a.regime == 'well':         # well-determined: 5-comp seas
        g = sorted(glob.glob('data/episode_BF3_5comp_*.h5')
                   + glob.glob('data/episode_BF4_5comp_*.h5'))
    else:
        g = sorted(glob.glob('data/episode_BF3_5comp_*.h5')
                   + glob.glob('data/episode_BF4_5comp_*.h5')
                   + glob.glob('data/episode_BF4_15comp_*.h5')
                   + glob.glob('data/episode_BF4_30comp_*.h5'))
    paths = g[:n]
    if not paths:
        print('NO EPISODES'); sys.exit(1)
    t0 = time.time()
    res = []
    for p in paths:
        try:
            ep = load_ep(p)
            t_now = float(ep['ts'][len(ep['ts']) // 2])
            kp = dom_kp(ep['theta'])
            r = run_episode(ep, t_now, kp, R=a.R)
            r['ep'] = os.path.basename(p)
            res.append(r)
            print(f"  {r['ep']:32s} crb-ridge {r['crb_ridge']:+.3f} "
                  f"crb-relax {r['crb_relax']:+.3f} crb-gp {r['crb_gp']:+.3f} | "
                  f"agree r-x {r['agree_ridge_relax']:+.3f} r-g {r['agree_ridge_gp']:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback
            print(f"  {os.path.basename(p)} EXCLUDED ({type(ex).__name__}: {ex})")
            traceback.print_exc()
    # summary
    ests = ['ridge', 'relax', 'gp']
    summ = {}
    print(f"\n=== median per-episode Spearman(sigma_CRB, sigma_actual_EST) (n_ep={len(res)}) ===")
    skill_ridge = np.nanmedian([r['skill_ridge'] for r in res])
    for e in ests:
        m, lo, hi = med_ci([r[f'crb_{e}'] for r in res])
        pm, plo, phi = med_ci([r[f'crb_{e}_pd'] for r in res])
        sk = np.nanmedian([r[f'skill_{e}'] for r in res])
        comp = 'competent' if sk <= 2 * skill_ridge else 'EXPLORATORY(weak>2x)'
        summ[e] = dict(crb=[m, lo, hi], partial=[pm, plo, phi], skill=float(sk), competence=comp)
        print(f"  {e:6s}: crb-act {m:+.3f} CI[{lo:+.3f},{hi:+.3f}] | "
              f"partial|dist {pm:+.3f} CI[{plo:+.3f},{phi:+.3f}] | skill {sk:.3f} [{comp}]")
    for pair in ['agree_ridge_relax', 'agree_ridge_gp', 'agree_relax_gp']:
        m, lo, hi = med_ci([r[pair] for r in res])
        summ[pair] = dict(median=m, ci=[lo, hi])
        print(f"  {pair:20s}: {m:+.3f} CI[{lo:+.3f},{hi:+.3f}]")

    # pre-registered verdict
    verdict = 'SMOKE'
    if not a.smoke:
        def ok(e):
            s = summ[e]
            if 'EXPLORATORY' in s['competence']:
                return None
            supp = (s['crb'][0] >= 0.45 and s['crb'][1] > 0 and s['partial'][0] > 0.15
                    and abs(s['crb'][0] - summ['ridge']['crb'][0]) <= 0.15)
            refu = (s['crb'][0] < 0.30 or s['crb'][1] <= 0 or s['partial'][0] <= 0)
            return 'SUPPORT' if supp else ('REFUTE' if refu else 'PARTIAL')
        alt = {e: ok(e) for e in ('relax', 'gp')}
        agree_ok = summ['agree_ridge_relax']['median'] >= 0.4
        verdicts = [v for v in alt.values() if v is not None]
        if any(v == 'REFUTE' for v in verdicts) or not agree_ok:
            verdict = f"REFUTE/PARTIAL: envelope may predict mostly its ML twin (alt={alt}, agree_ok={agree_ok})"
        elif verdicts and all(v == 'SUPPORT' for v in verdicts):
            verdict = f"SUPPORT: envelope predicts other estimator classes (alt={alt}) -> estimator-independence SEALED (sandbox scope)"
        else:
            verdict = f"PARTIAL: mixed (alt={alt}, agree_ok={agree_ok}) -> report CIs, ARGUED only"
        print(f"\n=== PRE-REGISTERED VERDICT ===\n  -> {verdict}")
        print("  CAVEAT: synthetic-obs sandbox (e58 scope); production/ETKF + real operator = follow-up.")
    json.dump(dict(episodes=res, summary=summ, verdict=verdict,
                   config=dict(sigma=SIGMA, leads=LEADS, ring=RING, nk=NK,
                               nk_relax=NK_RELAX, k_budget=K_BUDGET, R=a.R)),
              open(out_path, 'w'), indent=1, default=float)
    print(f"saved -> {out_path}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
