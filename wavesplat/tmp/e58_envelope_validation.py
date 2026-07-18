"""e58 — §8 envelope-skill validation (CONDITIONAL->PASS keystone). Prereg =
tmp/E58_envelope_validation_prereg.md.

Q: does the CRB-anchored operability envelope predict WHERE real forecast skill
is high, vs an empirical-UQ baseline? For each episode: over query points
q=(xq,yq,t_now+lead), compare
  sigma_CRB(q)    = sqrt(g_q^T Finv g_q)/Hs   [reused d1_crb; estimator-INDEPENDENT bound]
  sigma_actual(q) = mean_m |forecast_m(q) - truth(q)|/Hs   [k-grid ridge-LS forecast,
                    NON-oracle wavenumbers, avg over M measurement-noise draws]
  sigma_emp(q)    = sqrt(Phi_q (Phi^T Phi+lam I)^-1 Phi_q^T) * sigma / Hs   [ridge posterior
                    var = empirical UQ, estimator-DEPENDENT; the Zhao-style baseline]
Primary: Spearman(sigma_CRB, sigma_actual). Guards: distance partial-correlation
(geometric confound), sigma_emp comparison. Obs signal = predict(theta,.)+N(0,sigma)
so the estimator input matches the FIM noise model (isolates envelope from lidar-operator
artifacts; caveat). k-grid is NON-oracle (prereg: avoid tautology).

Usage: python3 tmp/e58_envelope_validation.py [--smoke] [--n 12]
Out:   results/direction3/e58_envelope_validation.json
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
from wavesplat.model.dispersion import G

OUT = 'results/direction3/e58_envelope_validation.json'
SIGMA = 0.02                       # obs noise std (m); matches d1_crb + LiDAR 2cm spec
LEADS = [0.0, 2.0, 4.0, 6.0]       # forecast horizons (s)
M_NOISE = 24                       # noise draws for sigma_actual stabilization
RING = (0.35, 1.75)               # k-grid shell in units of kp
NK = 15                            # k-grid resolution


def dom_kp(theta):
    w = theta[:, 0].abs()
    i = int(torch.argmax(w))
    return float(torch.hypot(theta[i, 1], theta[i, 2]))


def build_kgrid(kp):
    ax = np.linspace(-RING[1] * kp, RING[1] * kp, NK)
    KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel()
    km = np.hypot(kx, ky) / kp
    keep = (km >= RING[0]) & (km <= RING[1])
    return (torch.tensor(kx[keep], dtype=DT, device=DEVICE),
            torch.tensor(ky[keep], dtype=DT, device=DEVICE))


def design(kxg, kyg, x, y, t, depth):
    """(M, 2Gatoms) real basis [cos, sin] with dispersion-set omega."""
    k = torch.hypot(kxg, kyg)
    om = omega_of_k(k, depth)
    psi = (kxg[None, :] * x[:, None] + kyg[None, :] * y[:, None]
           - om[None, :] * t[:, None])
    return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)


def run_episode(ep, t_now, kp, ridge_rel=1e-3, R=12):
    """sigma_CRB and sigma_actual are PHASE-AVERAGED local RMS (over R phase
    realizations of the same spectrum) — removes the local-amplitude confound
    of single-realization |error|. sigma_emp (posterior var) is phase-independent."""
    depth, Hs = ep['depth'], ep['Hs']
    theta0 = ep['theta']
    N = theta0.shape[0]
    x, y, t = collect_obs(ep, t_now)
    if x.numel() > 4000:
        idx = torch.randperm(x.numel(), device=DEVICE)[:4000]
        x, y, t = x[idx], y[idx], t[idx]
    # --- estimator geometry (phase-independent) ---
    kxg, kyg = build_kgrid(kp)
    Phi = design(kxg, kyg, x, y, t, depth)                # (Mobs, 2G)
    A = Phi.T @ Phi
    lam = ridge_rel * torch.trace(A) / A.shape[0]
    Ainv = torch.linalg.inv(A + lam * torch.eye(A.shape[0], dtype=DT, device=DEVICE))
    PhiT = Phi.T
    # --- query grid × leads ---
    gx, gy = ep['gx'], ep['gy']
    stride = max(1, (gx.numel() * gy.numel()) // 400)
    Gx, Gy = torch.meshgrid(gx, gy, indexing='xy')
    qx0, qy0 = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
    fx, fy = x, y
    rows = dict(crb=[], act=[], emp=[], dist=[], lead=[])
    for lead in LEADS:
        qt = torch.full_like(qx0, float(t_now + lead))
        Phi_q = design(kxg, kyg, qx0, qy0, qt, depth)      # (Q, 2G) phase-indep
        emp_var = torch.einsum('qa,ab,qb->q', Phi_q, Ainv, Phi_q)
        sig_emp = torch.sqrt(torch.clamp(emp_var, min=0)) * SIGMA / Hs
        crb_acc = torch.zeros(qx0.numel(), dtype=DT, device=DEVICE)
        act_acc = torch.zeros(qx0.numel(), dtype=DT, device=DEVICE)
        for _ in range(R):                                 # phase realizations
            th = theta0.clone()
            th[:, 3] = torch.rand(N, dtype=DT, device=DEVICE) * (2 * np.pi)
            # phase-averaged CRB
            Fr = fisher(th, x, y, t, depth, SIGMA)
            nF = Fr.shape[0]
            Finv = torch.linalg.inv(Fr + 1e-10 * torch.trace(Fr) / nF
                                    * torch.eye(nF, dtype=DT, device=DEVICE))
            Jq = jacobian(th, qx0, qy0, qt, depth)
            crb_acc += torch.einsum('qa,ab,qb->q', Jq, Finv, Jq)
            # actual estimator error for this realization
            s = predict(th, x, y, t, depth)
            z = s + SIGMA * torch.randn_like(s)
            c = Ainv @ (PhiT @ z)
            truth = predict(th, qx0, qy0, qt, depth)
            act_acc += (Phi_q @ c - truth) ** 2
        sig_crb = torch.sqrt(torch.clamp(crb_acc / R, min=0)) / Hs
        sig_act = torch.sqrt(act_acc / R) / Hs
        d2 = ((qx0[:, None] - fx[None, :]) ** 2
              + (qy0[:, None] - fy[None, :]) ** 2).min(dim=1).values
        dist = torch.sqrt(d2)
        rows['crb'].append(sig_crb); rows['act'].append(sig_act)
        rows['emp'].append(sig_emp); rows['dist'].append(dist)
        rows['lead'].append(torch.full_like(sig_crb, lead))
    cat = {k: torch.cat(v).cpu().numpy() for k, v in rows.items()}
    fin = np.isfinite(cat['crb']) & np.isfinite(cat['act']) & np.isfinite(cat['emp'])
    c, a, e, d = cat['crb'][fin], cat['act'][fin], cat['emp'][fin], cat['dist'][fin]
    r_ca = stats.spearmanr(c, a).correlation
    r_ea = stats.spearmanr(e, a).correlation
    r_da = stats.spearmanr(d, a).correlation
    r_cd = stats.spearmanr(c, d).correlation
    # partial Spearman(crb, act | dist)
    denom = np.sqrt(max(1e-9, (1 - r_cd ** 2) * (1 - r_da ** 2)))
    r_ca_d = (r_ca - r_cd * r_da) / denom
    return dict(n_q=int(fin.sum()), n_obs=int(x.numel()), kp=kp,
                spear_crb_act=float(r_ca), spear_emp_act=float(r_ea),
                spear_dist_act=float(r_da), spear_crb_dist=float(r_cd),
                partial_crb_act_dist=float(r_ca_d))


def summarize(res):
    def med_ci(key):
        v = np.array([r[key] for r in res])
        lo, hi = np.percentile(v, [25, 75])
        # wild-ish bootstrap CI of the median
        bs = [np.median(np.random.choice(v, len(v))) for _ in range(2000)]
        return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(lo), float(hi)
    out = {}
    print(f"\n=== per-episode Spearman (n_ep={len(res)}) — median [95% CI] ===")
    for key, lab in [('spear_crb_act', 'CRB↔actual (PRIMARY)'),
                     ('partial_crb_act_dist', 'CRB↔actual | dist (confound-guard)'),
                     ('spear_emp_act', 'emp-UQ↔actual (baseline)'),
                     ('spear_dist_act', 'dist↔actual (geometric)')]:
        m, lo, hi, q1, q3 = med_ci(key)
        out[key] = dict(median=m, ci_lo=lo, ci_hi=hi, iqr=[q1, q3])
        print(f"  {lab:38s}: {m:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  IQR[{q1:+.3f},{q3:+.3f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    n = 2 if a.smoke else a.n
    # multi-component seas only (1comp/2comp are degenerate: field fully
    # determined => sigma_CRB & sigma_actual both flat => no spatial signal).
    paths = sorted(glob.glob('data/episode_BF3_5comp_*.h5')
                   + glob.glob('data/episode_BF4_5comp_*.h5')
                   + glob.glob('data/episode_BF4_15comp_*.h5')
                   + glob.glob('data/episode_BF4_30comp_*.h5'))[:n]
    if not paths:
        print('NO EPISODES FOUND'); sys.exit(1)
    t0 = time.time()
    res = []
    for p in paths:
        try:
            ep = load_ep(p)
            t_now = float(ep['ts'][len(ep['ts']) // 2])
            kp = dom_kp(ep['theta'])
            r = run_episode(ep, t_now, kp)
            r['ep'] = os.path.basename(p)
            res.append(r)
            print(f"  {r['ep']:34s} crb-act {r['spear_crb_act']:+.3f} "
                  f"partial {r['partial_crb_act_dist']:+.3f} emp {r['spear_emp_act']:+.3f} "
                  f"dist {r['spear_dist_act']:+.3f} (n_q {r['n_q']}) [{time.time()-t0:.0f}s]",
                  flush=True)
        except Exception as ex:
            print(f"  {os.path.basename(p)} EXCLUDED ({type(ex).__name__}: {ex})", flush=True)
    summ = summarize(res)
    # pre-registered verdict
    verdict = 'SMOKE'
    if not a.smoke:
        P = summ['spear_crb_act']; PD = summ['partial_crb_act_dist']; E = summ['spear_emp_act']
        pass_ = (P['median'] >= 0.5 and P['ci_lo'] > 0 and PD['median'] > 0.2
                 and P['median'] >= E['median'] - 0.1)
        fail = (P['median'] < 0.3 or PD['median'] <= 0.0)
        verdict = ('PASS: CRB envelope predicts real skill (median>=0.5, partial survives dist, '
                   '>= empirical UQ) -> headline SEALED, CONDITIONAL->PASS cond1' if pass_ else
                   'FAIL: envelope does not predict skill / absorbed by distance -> headline retreats'
                   if fail else 'PARTIAL: report CIs, no forced label (discipline 6)')
        print(f"\n=== PRE-REGISTERED VERDICT ===\n  -> {verdict}")
        print("  CAVEAT: synthetic obs (predict+Gaussian noise), simulated operator, "
              "non-oracle k-grid, nowcast/short-lead. moving-sensor real anchor separate.")
    json.dump(dict(episodes=res, summary=summ, verdict=verdict,
                   config=dict(sigma=SIGMA, leads=LEADS, m_noise=M_NOISE, ring=RING, nk=NK)),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
