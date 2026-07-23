"""Agent G — RAW COLLECTION for the CRB-anchored calibrated-UQ gate.
Prereg: tmp/AGENT_G_PREREG_crb_calibrated_uq_2026-07-23.md (frozen before numbers).

Dumps, per (unit, realization, query point): signed error e (Hs units), sigma_CRB,
sigma_emp (ridge posterior std), dist (nearest obs), lead, plus unit metadata (kp, ncomp).
No calibration/verdict logic lives here -> the analysis (g_uq_analyze.py) and the
independent re-derivation both read the SAME npz, which is the dual-verification substrate.

Arm S (synthetic .h5 episodes, oracle theta, matched linear model class, regime axis 5/15/30comp)
Arm H (HOS nonlinear seas, FITTED theta_hat, real nonlinear truth)  <- headline arm

Usage:
  python3 tmp/g_uq_collect.py --arm S --n 24 --R 8
  python3 tmp/g_uq_collect.py --arm H --n 12 --D 6
"""
import os
import sys
import glob
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

OUT_DIR = 'results/direction3'


# =====================================================================
# ARM S — synthetic episodes (e58 sandbox), oracle theta
# =====================================================================
def arm_S(n, R, seed=0, oracle_basis=False):
    import torch
    from scripts.d1_crb import DEVICE, DT, load_ep, collect_obs, jacobian, fisher, predict
    from tmp.e58_envelope_validation import build_kgrid, design, dom_kp, LEADS, SIGMA

    # STRATIFIED across the under-determination axis (prereg G4 needs both regimes)
    per = max(1, n // 4)
    paths = []
    for pat in ('BF3_5comp', 'BF4_5comp', 'BF4_15comp', 'BF4_30comp'):
        paths += sorted(glob.glob(f'data/episode_{pat}_*.h5'))[:per]
    paths = paths[:n]
    recs = []
    t0 = time.time()
    for ui, p in enumerate(paths):
        try:
            ep = load_ep(p)
            torch.manual_seed(seed + 17 * ui)
            depth, Hs = ep['depth'], ep['Hs']
            theta0 = ep['theta']
            N = theta0.shape[0]
            t_now = float(ep['ts'][len(ep['ts']) // 2])
            kp = dom_kp(ep['theta'])
            x, y, t = collect_obs(ep, t_now)
            if x.numel() > 4000:
                idx = torch.randperm(x.numel(), device=DEVICE)[:4000]
                x, y, t = x[idx], y[idx], t[idx]
            if oracle_basis:            # matched model class: exact true wavenumbers
                kxg, kyg = theta0[:, 1].clone(), theta0[:, 2].clone()
            else:
                kxg, kyg = build_kgrid(kp)
            Phi = design(kxg, kyg, x, y, t, depth)
            A = Phi.T @ Phi
            lam = 1e-3 * torch.trace(A) / A.shape[0]
            Ainv = torch.linalg.inv(A + lam * torch.eye(A.shape[0], dtype=DT, device=DEVICE))
            PhiT = Phi.T
            gx, gy = ep['gx'], ep['gy']
            stride = max(1, (gx.numel() * gy.numel()) // 400)
            Gx, Gy = torch.meshgrid(gx, gy, indexing='xy')
            qx0, qy0 = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
            Q = qx0.numel()
            # phase-independent quantities per lead
            Phi_q_l, qt_l, emp_l, dist_l = [], [], [], []
            for lead in LEADS:
                qt = torch.full_like(qx0, float(t_now + lead))
                Phi_q = design(kxg, kyg, qx0, qy0, qt, depth)
                ev = torch.einsum('qa,ab,qb->q', Phi_q, Ainv, Phi_q)
                emp_l.append(torch.sqrt(torch.clamp(ev, min=0)) * SIGMA / Hs)
                d2 = ((qx0[:, None] - x[None, :]) ** 2
                      + (qy0[:, None] - y[None, :]) ** 2).min(dim=1).values
                dist_l.append(torch.sqrt(d2))
                Phi_q_l.append(Phi_q); qt_l.append(qt)
            emp = torch.cat(emp_l).cpu().numpy()
            dist = torch.cat(dist_l).cpu().numpy()
            lead_v = np.concatenate([np.full(Q, l) for l in LEADS])
            # runtime level proxy: hold out 20% of OBS, refit, RMS residual (no ground truth)
            nobs = x.numel()
            perm = torch.randperm(nobs, device=DEVICE)
            ho = perm[:max(8, nobs // 5)]; fi_ = perm[max(8, nobs // 5):]
            Phi_f = Phi[fi_]; A_f = Phi_f.T @ Phi_f
            lam_f = 1e-3 * torch.trace(A_f) / A_f.shape[0]
            Ainv_f = torch.linalg.inv(A_f + lam_f * torch.eye(A_f.shape[0], dtype=DT, device=DEVICE))
            for r in range(R):
                th = theta0.clone()
                th[:, 3] = torch.rand(N, dtype=DT, device=DEVICE) * (2 * np.pi)
                Fr = fisher(th, x, y, t, depth, SIGMA)
                nF = Fr.shape[0]
                Finv = torch.linalg.inv(Fr + 1e-10 * torch.trace(Fr) / nF
                                        * torch.eye(nF, dtype=DT, device=DEVICE))
                s = predict(th, x, y, t, depth)
                z = s + SIGMA * torch.randn_like(s)
                c = Ainv @ (PhiT @ z)
                c_f = Ainv_f @ (Phi_f.T @ z[fi_])
                resid = float(torch.sqrt(torch.mean((Phi[ho] @ c_f - z[ho]) ** 2))) / Hs
                crb_l, err_l = [], []
                for Phi_q, qt in zip(Phi_q_l, qt_l):
                    Jq = jacobian(th, qx0, qy0, qt, depth)
                    v = torch.einsum('qa,ab,qb->q', Jq, Finv, Jq)
                    crb_l.append(torch.sqrt(torch.clamp(v, min=0)) / Hs)
                    truth = predict(th, qx0, qy0, qt, depth)
                    err_l.append((Phi_q @ c - truth) / Hs)
                recs.append(dict(
                    unit=os.path.basename(p), draw=r, kp=kp, ncomp=N, Hs=Hs, resid=resid,
                    err=torch.cat(err_l).cpu().numpy().astype(np.float32),
                    crb=torch.cat(crb_l).cpu().numpy().astype(np.float32),
                    emp=emp.astype(np.float32), dist=dist.astype(np.float32),
                    lead=lead_v.astype(np.float32)))
            print(f"  [S] {os.path.basename(p):34s} N={N:2d} kp={kp:.4f} Q={Q} R={R} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback
            print(f"  [S] {os.path.basename(p)} EXCLUDED ({type(ex).__name__}: {ex})")
            traceback.print_exc()
    return recs


# =====================================================================
# ARM H — HOS nonlinear seas, FITTED theta_hat (headline arm)
# =====================================================================
def arm_H(n, D, seed=0):
    os.environ.setdefault('A1_N_OBS', '600')
    import numpy as _np
    from e15_hos_multidraw import SEAS, build_sea      # top-level (matches hos_s8_envelope)
    from hos_s8_envelope import two_stage_kp, fit_theta_and_pred, crb_envelope, draw
    recs = []
    t0 = time.time()
    for si, (Tp, Hs) in enumerate(SEAS[:n]):
        try:
            sea = build_sea(Tp, Hs, 0)
            for d in range(D):
                rng = _np.random.default_rng(900000 + 31 * si + 7919 * d + seed)
                Ps, zs, Q, gt = draw(sea, rng)
                Hs_e = 4 * _np.std(zs)
                kp = two_stage_kp(Ps, zs)
                theta, pred, emp = fit_theta_and_pred(Ps, zs, Q, kp)
                sig_crb = crb_envelope(theta, Ps, Q, Hs_e)
                # runtime level proxy: hold out 20% of OBS, refit, RMS residual (no GT)
                nobs = len(zs); pm = rng.permutation(nobs); nh = max(8, nobs // 5)
                ho, fi_ = pm[:nh], pm[nh:]
                _, pred_ho, _ = fit_theta_and_pred(Ps[fi_], zs[fi_], Ps[ho], kp)
                resid = float(_np.sqrt(_np.mean((pred_ho - zs[ho]) ** 2))) / Hs_e
                err = (pred - gt) / Hs_e
                d2 = ((Q[:, 0:1] - Ps[None, :, 0]) ** 2
                      + (Q[:, 1:2] - Ps[None, :, 1]) ** 2).min(1)
                dist = _np.sqrt(d2)
                recs.append(dict(
                    unit=f'Tp{Tp:g}_Hs{Hs:g}', draw=d, kp=kp, ncomp=int(theta.shape[0]),
                    Hs=float(Hs_e), resid=resid,
                    err=err.astype(np.float32), crb=sig_crb.astype(np.float32),
                    emp=(emp / Hs_e).astype(np.float32), dist=dist.astype(np.float32),
                    lead=Q[:, 2].astype(np.float32)))
            print(f"  [H] Tp{Tp:g}_Hs{Hs:g} kp={kp:.4f} D={D} nq={len(err)} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback
            print(f"  [H] sea {si} EXCLUDED ({type(ex).__name__}: {ex})")
            traceback.print_exc()
    return recs


def save(recs, path):
    if not recs:
        print('NO RECORDS'); return
    keys = ['err', 'crb', 'emp', 'dist', 'lead']
    cat = {k: np.concatenate([r[k] for r in recs]) for k in keys}
    lens = [len(r['err']) for r in recs]
    cat['unit'] = np.concatenate([np.full(L, r['unit']) for r, L in zip(recs, lens)])
    cat['draw'] = np.concatenate([np.full(L, r['draw']) for r, L in zip(recs, lens)]).astype(np.int16)
    cat['kp'] = np.concatenate([np.full(L, r['kp']) for r, L in zip(recs, lens)]).astype(np.float32)
    cat['resid'] = np.concatenate([np.full(L, r['resid']) for r, L in zip(recs, lens)]).astype(np.float32)
    cat['ncomp'] = np.concatenate([np.full(L, r['ncomp']) for r, L in zip(recs, lens)]).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **cat)
    print(f"saved -> {path}  rows={len(cat['err'])}  units={len(set(cat['unit']))}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['S', 'H'], required=True)
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--R', type=int, default=8)
    ap.add_argument('--D', type=int, default=6)
    ap.add_argument('--tag', default='')
    ap.add_argument('--oracle_basis', action='store_true')
    a = ap.parse_args()
    if a.arm == 'S':
        recs = arm_S(a.n, a.R, oracle_basis=a.oracle_basis)
        save(recs, f'{OUT_DIR}/g_uq_raw_S{a.tag}.npz')
    else:
        recs = arm_H(a.n, a.D)
        save(recs, f'{OUT_DIR}/g_uq_raw_H{a.tag}.npz')
