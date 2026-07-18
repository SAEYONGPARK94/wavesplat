"""E-58 §8 INDEPENDENT AUDIT (prereg in-file). Two questions:
 A. Does envelope predictive power scale with under-determination? (dose-response)
 B. Is the modest 5comp rho=+0.33 DEPRESSED by (b1) finite-R Monte-Carlo noise
    [attenuation] or (b2) ridge-LS estimator inefficiency [bias]? -> did estimator
    efficiency UNDERESTIMATE +0.33, or is it a genuine well-determined-regime ceiling?

Tautology guards (prereg): k-grid RING + NK held FIXED & non-oracle throughout;
only R (MC noise) and lambda (ridge bias) are varied. Raising R cannot move toward
oracle. Split-half reliability -> classic disattenuation rho_true = rho_obs/sqrt(rel_x*rel_y).
"""
import os, sys, glob, json, time
import numpy as np, torch
from scipy import stats
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from scripts.d1_crb import (DEVICE, DT, load_ep, collect_obs, jacobian, fisher,
                            predict, omega_of_k)
from tmp.e58_envelope_validation import build_kgrid, design, dom_kp, LEADS, SIGMA


def episode_raw(ep, t_now, kp, R=96, lam_rel=1e-3, seed=0):
    """Return per-realization crb_var (R,QL) and act_sq (R,QL), plus phase-indep
    emp (QL,) and dist (QL,), pooled over leads. NK/ring FIXED (non-oracle)."""
    torch.manual_seed(seed); np.random.seed(seed)
    depth, Hs = ep['depth'], ep['Hs']; theta0 = ep['theta']; N = theta0.shape[0]
    x, y, t = collect_obs(ep, t_now)
    if x.numel() > 4000:
        idx = torch.randperm(x.numel(), device=DEVICE)[:4000]; x, y, t = x[idx], y[idx], t[idx]
    kxg, kyg = build_kgrid(kp)
    Phi = design(kxg, kyg, x, y, t, depth); A = Phi.T @ Phi
    lam = lam_rel * torch.trace(A) / A.shape[0]
    Ainv = torch.linalg.inv(A + lam * torch.eye(A.shape[0], dtype=DT, device=DEVICE)); PhiT = Phi.T
    gx, gy = ep['gx'], ep['gy']; stride = max(1, (gx.numel()*gy.numel())//400)
    Gx, Gy = torch.meshgrid(gx, gy, indexing='xy')
    qx0, qy0 = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
    Q = qx0.numel()
    emp_l, dist_l, Phi_q_l, qt_l = [], [], [], []
    for lead in LEADS:
        qt = torch.full_like(qx0, float(t_now + lead))
        Phi_q = design(kxg, kyg, qx0, qy0, qt, depth); Phi_q_l.append(Phi_q); qt_l.append(qt)
        ev = torch.einsum('qa,ab,qb->q', Phi_q, Ainv, Phi_q)
        emp_l.append(torch.sqrt(torch.clamp(ev, min=0)) * SIGMA / Hs)
        d2 = ((qx0[:, None]-x[None, :])**2 + (qy0[:, None]-y[None, :])**2).min(dim=1).values
        dist_l.append(torch.sqrt(d2))
    emp = torch.cat(emp_l).cpu().numpy(); dist = torch.cat(dist_l).cpu().numpy()
    QL = Q * len(LEADS)
    crb_var = torch.zeros(R, QL, dtype=DT, device=DEVICE)
    act_sq = torch.zeros(R, QL, dtype=DT, device=DEVICE)
    for r in range(R):
        th = theta0.clone(); th[:, 3] = torch.rand(N, dtype=DT, device=DEVICE)*(2*np.pi)
        Fr = fisher(th, x, y, t, depth, SIGMA); nF = Fr.shape[0]
        Finv = torch.linalg.inv(Fr + 1e-10*torch.trace(Fr)/nF*torch.eye(nF, dtype=DT, device=DEVICE))
        s = predict(th, x, y, t, depth); z = s + SIGMA*torch.randn_like(s); c = Ainv @ (PhiT @ z)
        cv_l, as_l = [], []
        for Phi_q, qt in zip(Phi_q_l, qt_l):
            Jq = jacobian(th, qx0, qy0, qt, depth)
            cv_l.append(torch.einsum('qa,ab,qb->q', Jq, Finv, Jq))
            truth = predict(th, qx0, qy0, qt, depth)
            as_l.append((Phi_q @ c - truth)**2)
        crb_var[r] = torch.cat(cv_l) / (Hs**2)
        act_sq[r] = torch.cat(as_l) / (Hs**2)
    return crb_var.cpu().numpy(), act_sq.cpu().numpy(), emp, dist


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return stats.spearmanr(a[m], b[m]).correlation


def spearman_brown(r_half):
    r_half = max(min(r_half, 0.999), -0.999)
    return 2*r_half/(1+r_half)


def stats_from_raw(crb_var, act_sq, emp, dist, R):
    cv, ax = crb_var[:R], act_sq[:R]
    crb_full = np.sqrt(np.clip(cv.mean(0), 0, None))
    act_full = np.sqrt(np.clip(ax.mean(0), 0, None))
    rho = sp(crb_full, act_full)
    r_ea = sp(emp, act_full); r_da = sp(dist, act_full); r_cd = sp(crb_full, dist)
    denom = np.sqrt(max(1e-9, (1-r_cd**2)*(1-r_da**2))); partial = (rho - r_cd*r_da)/denom
    h = R//2; A_idx = np.arange(0, R, 2)[:h]; B_idx = np.arange(1, R, 2)[:h]
    actA = np.sqrt(np.clip(ax[A_idx].mean(0), 0, None)); actB = np.sqrt(np.clip(ax[B_idx].mean(0), 0, None))
    crbA = np.sqrt(np.clip(cv[A_idx].mean(0), 0, None)); crbB = np.sqrt(np.clip(cv[B_idx].mean(0), 0, None))
    rel_act = spearman_brown(sp(actA, actB)); rel_crb = spearman_brown(sp(crbA, crbB))
    disatt = rho/np.sqrt(max(1e-6, rel_act*rel_crb))
    return dict(rho=rho, partial=partial, emp=r_ea, dist=r_da,
                rel_act=rel_act, rel_crb=rel_crb, disatt=min(disatt, 1.0))


def pool(pat, k, ex=()):
    ps = sorted(glob.glob(f'data/episode_{pat}_*.h5'))
    ps = [p for p in ps if not any(e in p for e in ex)]
    return ps[:k]


if __name__ == '__main__':
    t0 = time.time()
    print("="*70); print("PART A - UNDER-DETERMINATION DOSE-RESPONSE (headline)"); print("="*70)
    doseA = {'2comp': pool('BF3_2comp', 6)+pool('BF4_2comp', 6),
             '5comp': pool('BF3_5comp', 10),
             '15comp': pool('BF4_15comp', 10),
             '30comp': pool('BF4_30comp', 10)}
    dose_rows = []
    for lvl, paths in doseA.items():
        for p in paths:
            try:
                ep = load_ep(p); tn = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
                cv, ax, emp, dist = episode_raw(ep, tn, kp, R=24, seed=0)
                s = stats_from_raw(cv, ax, emp, dist, R=24)
                dose_rows.append(dict(lvl=lvl, ncomp=int(lvl.replace('comp', '')), kp=kp,
                                      ep=os.path.basename(p), **s))
            except Exception as ex:
                print(f"    {os.path.basename(p)} EXCLUDED ({type(ex).__name__}: {ex})")
    for lvl in doseA:
        rr = [r for r in dose_rows if r['lvl'] == lvl]
        if not rr:
            print(f"  {lvl:7s} n=0 (none)"); continue
        rho = np.array([r['rho'] for r in rr]); par = np.array([r['partial'] for r in rr])
        emp = np.array([r['emp'] for r in rr]); dis = np.array([r['dist'] for r in rr])
        print(f"  {lvl:7s} n={len(rr):2d}  CRB-act med={np.median(rho):+.3f}  "
              f"partial={np.median(par):+.3f}  emp={np.median(emp):+.3f}  dist={np.median(dis):+.3f}  "
              f"kp={np.median([r['kp'] for r in rr]):.3f}")
    ncomp = np.array([r['ncomp'] for r in dose_rows]); rho = np.array([r['rho'] for r in dose_rows])
    kpv = np.array([r['kp'] for r in dose_rows])
    sc = stats.spearmanr(ncomp, rho); sk = stats.spearmanr(kpv, rho)
    print(f"\n  Spearman(ncomp, rho) = {sc.correlation:+.3f} p={sc.pvalue:.4g}")
    print(f"  Spearman(kp,    rho) = {sk.correlation:+.3f} p={sk.pvalue:.4g}")

    print("\n"+"="*70); print("PART B - DID ESTIMATOR EFFICIENCY UNDERESTIMATE +0.33?"); print("="*70)
    subB = {'5comp': pool('BF3_5comp', 8), '15comp': pool('BF4_15comp', 4)}
    print("\n  (b1) FINITE-R ATTENUATION: rho at R=12 vs R=96 + split-half disattenuated")
    for lvl, paths in subB.items():
        rows = []
        for p in paths:
            ep = load_ep(p); tn = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
            cv, ax, emp, dist = episode_raw(ep, tn, kp, R=96, lam_rel=1e-3, seed=0)
            s12 = stats_from_raw(cv, ax, emp, dist, R=12)
            s96 = stats_from_raw(cv, ax, emp, dist, R=96)
            rows.append((s12['rho'], s96['rho'], s96['rel_act'], s96['rel_crb'], s96['disatt']))
        rows = np.array(rows)
        print(f"    {lvl:7s} n={len(paths)}: rho@R12 med={np.median(rows[:, 0]):+.3f}  "
              f"rho@R96 med={np.median(rows[:, 1]):+.3f}  rel_act={np.median(rows[:, 2]):.3f}  "
              f"rel_crb={np.median(rows[:, 3]):.3f}  disatt(R96) med={np.median(rows[:, 4]):+.3f}")
    print("\n  (b2) RIDGE-BIAS EFFICIENCY: rho vs lambda (R=48, NK/ring fixed=non-oracle)")
    for lvl, paths in subB.items():
        line = f"    {lvl:7s}: "
        for lam in [1e-4, 1e-3, 1e-2]:
            rr = []
            for p in paths:
                ep = load_ep(p); tn = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
                cv, ax, emp, dist = episode_raw(ep, tn, kp, R=48, lam_rel=lam, seed=0)
                rr.append(stats_from_raw(cv, ax, emp, dist, R=48)['rho'])
            line += f"lam={lam:.0e} rho={np.median(rr):+.3f}   "
        print(line)
    json.dump(dict(dose=dose_rows), open('results/direction3/e58b_audit.json', 'w'), indent=1, default=float)
    print(f"\n[{time.time()-t0:.0f}s]  saved -> results/direction3/e58b_audit.json")
