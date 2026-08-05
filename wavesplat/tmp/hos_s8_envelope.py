"""HOS §8 envelope validation — the CONDITIONAL->PASS keystone, re-run on HOS (fixes
the synthetic inverse-crime version). Q: does sigma_CRB (the operability envelope,
computed at the FITTED model theta_hat, NOT oracle) predict WHERE the actual
reconstruction error is high/low on REAL nonlinear HOS seas? Compare to empirical-UQ
(ridge posterior var) and distance. Synthetic §8 got Spearman(CRB,actual)~0.33 (PARTIAL).
Usage: python3 tmp/hos_s8_envelope.py [--n 8]
Out: results/direction3/hos_s8_envelope.json
"""
import json, os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from e15_hos_multidraw import (SEAS, build_sea, T_MANUAL_50, N_OBS, LEADS, DEPTH,
                               NOISE, R_MIN, FOV_H, z_at)
from scripts.d1_crb import DT, DEVICE, jacobian, fisher, omega_of_k
from scipy import stats

RING = (0.35, 1.75)

def two_stage_kp(P, z):
    from d18_relax_estimator import beamform_top
    P32, z32 = P.astype(np.float32), z.astype(np.float32)
    bx, by = beamform_top(P32, z32, DEPTH, klim=2.0, n=120); kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P32, z32, DEPTH, klim=3.5*kp, n=120); return max(np.hypot(bx, by), 1e-3)

def kgrid(kp, nk):
    ax = np.linspace(-RING[1]*kp, RING[1]*kp, nk); KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel(); km = np.hypot(kx, ky)/kp; keep = (km >= RING[0]) & (km <= RING[1])
    return kx[keep], ky[keep]

def design_np(kx, ky, P):
    om = np.sqrt(9.81*np.hypot(kx, ky)*np.tanh(np.hypot(kx, ky)*DEPTH)+1e-12)
    psi = kx[None]*P[:, 0:1] + ky[None]*P[:, 1:2] - om[None]*P[:, 2:3]
    return np.concatenate([np.cos(psi), np.sin(psi)], 1), om

def fit_theta_and_pred(P, z, Q, kp, K=15, ridge=1e-3):
    """RELAX -> K components; LS coeffs -> theta_hat=(w,kx,ky,phi); ridge-LS field pred at Q."""
    kx, ky = kgrid(kp, 25); C, _ = design_np(kx, ky, P); N = len(kx)
    cos_c, sin_c = C[:, :N], C[:, N:]; r = z.copy(); sel = []
    for _ in range(min(K, N)):
        pw = (cos_c.T@r)**2 + (sin_c.T@r)**2
        if sel: pw[sel] = -1
        j = int(np.argmax(pw)); sel.append(j)
        Ps = np.concatenate([cos_c[:, sel], sin_c[:, sel]], 1)
        coef = np.linalg.lstsq(Ps, z, rcond=None)[0]; r = z - Ps@coef
    kxs, kys = kx[sel], ky[sel]; m = len(sel)
    a_cos, a_sin = coef[:m], coef[m:]
    w = np.sqrt(a_cos**2 + a_sin**2); phi = np.arctan2(-a_sin, a_cos)
    theta = np.stack([w, kxs, kys, phi], 1)                      # (m,4) theta_hat
    # ridge-LS field prediction at Q (the estimator whose error we measure)
    Cd, _ = design_np(kxs, kys, P); A = Cd.T@Cd; lam = ridge*np.trace(A)/A.shape[0]
    c = np.linalg.solve(A + lam*np.eye(A.shape[0]), Cd.T@z)
    Cq, _ = design_np(kxs, kys, Q); pred = Cq@c
    # empirical-UQ: ridge posterior variance at Q (Zhao proxy)
    Ainv = np.linalg.inv(A + lam*np.eye(A.shape[0]))
    emp = np.sqrt(np.clip(np.einsum('qa,ab,qb->q', Cq, Ainv, Cq), 0, None))*NOISE
    return theta, pred, emp

def draw(sea, rng):
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp']*rng.exponential(1.0, len(sc['amp'])); keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep], np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep]); rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
    edges = np.linspace(R_MIN, rr.max()+1e-6, 13); b = np.clip(np.digitize(rr, edges)-1, 0, 11)
    take = [rng.choice(np.where(b == j)[0], min((b == j).sum(), max(-(-N_OBS//12), 1)), False) for j in range(12) if (b == j).sum()]
    sel = np.concatenate(take); Ps, zs = P[sel], zv[sel]
    rf = np.sqrt(rng.uniform((15./200.)**2, 1.0, 700))*200.; af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 700)
    qx, qy = rf*np.cos(af), rf*np.sin(af)
    Q, gt = [], []
    for li, l_ in enumerate(LEADS):
        gt.append(z_at(sea['leads'][li], qx, qy)); Q.append(np.column_stack([qx, qy, np.full(700, l_)]))
    return Ps, zs, np.vstack(Q), np.concatenate(gt)

def crb_envelope(theta, Ps, Q, Hs):
    th = torch.tensor(theta, dtype=DT, device=DEVICE)
    x = torch.tensor(Ps[:, 0], dtype=DT, device=DEVICE); y = torch.tensor(Ps[:, 1], dtype=DT, device=DEVICE)
    t = torch.tensor(Ps[:, 2], dtype=DT, device=DEVICE)
    F = fisher(th, x, y, t, DEPTH, NOISE); nF = F.shape[0]
    Finv = torch.linalg.inv(F + 1e-8*torch.trace(F)/nF*torch.eye(nF, dtype=DT, device=DEVICE))
    qx = torch.tensor(Q[:, 0], dtype=DT, device=DEVICE); qy = torch.tensor(Q[:, 1], dtype=DT, device=DEVICE)
    qt = torch.tensor(Q[:, 2], dtype=DT, device=DEVICE)
    Jq = jacobian(th, qx, qy, qt, DEPTH)
    var = torch.einsum('qa,ab,qb->q', Jq, Finv, Jq)
    return (torch.sqrt(torch.clamp(var, min=0))/Hs).cpu().numpy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=8); a = ap.parse_args()
    rows = []; t0 = time.time()
    for si, (Tp, Hs) in enumerate(SEAS[:a.n]):
        try:
            sea = build_sea(Tp, Hs, 0); Ps, zs, Q, gt = draw(sea, np.random.default_rng(900000+31*si))
            Hs_e = 4*np.std(zs); kp = two_stage_kp(Ps, zs)
            theta, pred, emp = fit_theta_and_pred(Ps, zs, Q, kp)
            sig_crb = crb_envelope(theta, Ps, Q, Hs_e)
            sig_act = np.abs(pred - gt)/Hs_e
            d2 = ((Q[:, 0:1]-Ps[None, :, 0])**2 + (Q[:, 1:2]-Ps[None, :, 1])**2).min(1); dist = np.sqrt(d2)
            fin = np.isfinite(sig_crb)&np.isfinite(sig_act)&np.isfinite(emp)
            c, ac, e, dd = sig_crb[fin], sig_act[fin], emp[fin], dist[fin]
            r_ca = stats.spearmanr(c, ac).correlation; r_ea = stats.spearmanr(e, ac).correlation
            r_da = stats.spearmanr(dd, ac).correlation; r_cd = stats.spearmanr(c, dd).correlation
            pd = (r_ca - r_cd*r_da)/np.sqrt(max(1e-9, (1-r_cd**2)*(1-r_da**2)))
            rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', crb_act=r_ca, emp_act=r_ea, dist_act=r_da, partial=pd, n_q=int(fin.sum())))
            print(f"  {rows[-1]['sea']:12s} CRB-act {r_ca:+.3f} emp-act {r_ea:+.3f} dist-act {r_da:+.3f} partial|dist {pd:+.3f} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback; print(f"  sea {si} ERR {ex}"); traceback.print_exc()
    def med(k): return float(np.nanmedian([r[k] for r in rows]))
    print(f"\n=== HOS §8 envelope validation (n={len(rows)} HOS seas, CRB at FITTED theta, real HOS truth) ===")
    print(f"  Spearman(CRB, actual)   median {med('crb_act'):+.3f}   (synthetic §8 was +0.33; PASS bar >=0.5)")
    print(f"  partial | distance       median {med('partial'):+.3f}   (geometric confound guard)")
    print(f"  Spearman(emp-UQ, actual) median {med('emp_act'):+.3f}   (Zhao empirical-UQ baseline)")
    print(f"  Spearman(dist, actual)   median {med('dist_act'):+.3f}")
    v = ("PASS: CRB envelope predicts real HOS skill (>=0.5, partial survives)" if med('crb_act') >= 0.5 and med('partial') > 0.15
         else "PARTIAL: envelope predicts HOS skill but modest (report CIs)" if med('crb_act') >= 0.25
         else "FAIL: CRB envelope does NOT predict HOS skill")
    print(f"\n=== HOS VERDICT === {v}")
    json.dump(dict(rows=rows, crb_act=med('crb_act'), partial=med('partial'), emp_act=med('emp_act'),
                   dist_act=med('dist_act'), verdict=v), open('results/direction3/hos_s8_envelope.json', 'w'), indent=1, default=float)
    print(f"saved ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
