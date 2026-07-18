"""E-59 — §8 REAL-DATA anchor (condition 3, option A). Prereg = tmp/E59_real_anchor_prereg.md.
Q: on real stereo waves, does the CRB envelope rank-predict WHERE a sparse-obs forecast
matches the true surface, reproducing the synthetic §8 sign at matched aperture ratio?

 sigma_CRB(q)  = sqrt(J_q Finv J_q^T)/Hs   [nonlinear theta-hat CRB over [A,kx,ky,phi],
                 dispersion-set omega; estimator-INDEPENDENT; theta-hat fit from dense obs]
 sigma_actual  = |Phi_q c - Z_true(q)|/Hs   [k-grid ridge-LS forecast from SPARSE obs vs REAL
                 stereo surface; theta-FREE => not tautological with the theta-hat CRB]
 sigma_emp     = sqrt(Phi_q Ainv Phi_q^T) sigma/Hs  [linear-grid posterior var = empirical UQ]
 dist(q)       = nearest sparse-obs distance
NULL (decisive tautology-breaker): sigma_actual_null uses a PHASE-SCRAMBLED future surface
(same amplitude spectrum, coherent structure destroyed). If CRB<->actual_real >> CRB<->actual_null,
the envelope captures real coherent skill beyond geometry/spectrum.  [controls may be extended
after the fable red-team; run gate = red-team incorporated.]
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch
from scipy import stats
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from realdata_analysis import load_surface_nc, directional_spread
from d1_crb import DEVICE, DT, jacobian, omega_of_k

SIGMA = 0.02                       # Falcon range noise (m)
RING = (0.35, 1.75); NK = 15       # non-oracle k-grid (matches e58)
RECORDS = {
    'AA15a': dict(path='data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc', depth=17.0),
    'AA14':  dict(path='data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc', depth=17.0),
    'YS17':  dict(path='data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc', depth=50.0),
    'LJ18g': dict(path='data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc', depth=45.0),
}


def detrend(Z, GX, GY):
    """Remove per-frame planar trend a+bx+cy (NaN-aware)."""
    out = np.empty_like(Z)
    for i in range(len(Z)):
        F = Z[i]; m = np.isfinite(F)
        if m.sum() < 20:
            out[i] = F; continue
        A = np.c_[np.ones(m.sum()), GX[m], GY[m]]
        coef, *_ = np.linalg.lstsq(A, F[m], rcond=None)
        out[i] = F - (coef[0] + coef[1]*GX + coef[2]*GY)
    return out


def phase_scramble(frame, rng):
    """Randomize spatial phases, preserve amplitude spectrum (NaN->0)."""
    f = np.nan_to_num(frame)
    F = np.fft.fft2(f)
    ph = rng.uniform(0, 2*np.pi, F.shape)
    # keep hermitian-ish symmetry loosely (real output via .real)
    return np.fft.ifft2(np.abs(F) * np.exp(1j*ph)).real


def build_kgrid(kp):
    ax = np.linspace(-RING[1]*kp, RING[1]*kp, NK)
    KX, KY = np.meshgrid(ax, ax); kx, ky = KX.ravel(), KY.ravel()
    km = np.hypot(kx, ky)/kp; keep = (km >= RING[0]) & (km <= RING[1])
    return (torch.tensor(kx[keep], dtype=DT, device=DEVICE),
            torch.tensor(ky[keep], dtype=DT, device=DEVICE))


def design(kxg, kyg, x, y, t, depth):
    k = torch.hypot(kxg, kyg); om = omega_of_k(k, depth)
    psi = kxg[None, :]*x[:, None] + kyg[None, :]*y[:, None] - om[None, :]*t[:, None]
    return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)


def theta_hat(kxg, kyg, c, topk=24):
    """c = ridge-LS coeffs [cos(G); sin(G)] -> theta_hat (N,4)=[A,kx,ky,phi], keeping only
    the TOP-K dominant components (real seas ~ few components; keeps the nonlinear theta-CRB a
    SPARSE spectral-model bound, distinct from the full-grid linear posterior sigma_emp =>
    avoids the CRB collapsing onto emp). A_j cos(psi+phi)=c_cos cos+c_sin sin."""
    G = kxg.numel(); cc, cs = c[:G], c[G:]
    A = torch.hypot(cc, cs); phi = torch.atan2(-cs, cc)
    k = min(topk, G)
    idx = torch.topk(A, k).indices
    return torch.stack([A[idx], kxg[idx], kyg[idx], phi[idx]], dim=1)


def bilinear(Zf, gx, gy, xq, yq):
    """NaN-aware bilinear of one frame (ny,nx) at query arrays -> (Q,) with NaN outside/holes."""
    gx = gx.cpu().numpy(); gy = gy.cpu().numpy()
    xq = xq.cpu().numpy(); yq = yq.cpu().numpy()
    ix = np.interp(xq, gx, np.arange(len(gx))); iy = np.interp(yq, gy, np.arange(len(gy)))
    i0x = np.floor(ix).astype(int); i0y = np.floor(iy).astype(int)
    ok = (i0x >= 0) & (i0x < len(gx)-1) & (i0y >= 0) & (i0y < len(gy)-1)
    out = np.full(len(xq), np.nan); fx = ix-i0x; fy = iy-i0y
    for j in np.where(ok)[0]:
        q = Zf[i0y[j]:i0y[j]+2, i0x[j]:i0x[j]+2]
        if np.all(np.isfinite(q)):
            out[j] = (q[0,0]*(1-fx[j])*(1-fy[j]) + q[0,1]*fx[j]*(1-fy[j])
                      + q[1,0]*(1-fx[j])*fy[j] + q[1,1]*fx[j]*fy[j])
    return out


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return stats.spearmanr(a[m], b[m]).correlation if m.sum() > 10 else np.nan


def run_window(Zd, gx, gy, t, depth, Hs, kp, i_now, W_fr, lead_fr, n_obs=1500,
               ridge_rel=1e-3, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ny, nx = Zd.shape[1:]; GXg, GYg = np.meshgrid(gx.cpu().numpy(), gy.cpu().numpy())
    # --- sparse obs from window [i_now-W, i_now] ---
    ox, oy, ot, oz = [], [], [], []
    for fi in range(max(0, i_now-W_fr), i_now+1):
        F = Zd[fi]; m = np.isfinite(F)
        yy, xx = np.where(m)
        if len(xx) == 0: continue
        pick = rng.choice(len(xx), min(len(xx), n_obs//(W_fr+1)+1), replace=False)
        ox.append(GXg[yy[pick], xx[pick]]); oy.append(GYg[yy[pick], xx[pick]])
        ot.append(np.full(len(pick), t[fi])); oz.append(F[yy[pick], xx[pick]])
    if not ox: return None
    ox = np.concatenate(ox); oy = np.concatenate(oy); ot = np.concatenate(ot); oz = np.concatenate(oz)
    if len(ox) > n_obs:
        s = rng.choice(len(ox), n_obs, replace=False); ox, oy, ot, oz = ox[s], oy[s], ot[s], oz[s]
    X = torch.tensor(ox, dtype=DT, device=DEVICE); Y = torch.tensor(oy, dtype=DT, device=DEVICE)
    Tt = torch.tensor(ot, dtype=DT, device=DEVICE)
    Z = torch.tensor(oz + SIGMA*rng.standard_normal(len(oz)), dtype=DT, device=DEVICE)
    # --- estimator geometry ---
    kxg, kyg = build_kgrid(kp); Phi = design(kxg, kyg, X, Y, Tt, depth)
    Amat = Phi.T @ Phi; lam = ridge_rel*torch.trace(Amat)/Amat.shape[0]
    Ainv = torch.linalg.inv(Amat + lam*torch.eye(Amat.shape[0], dtype=DT, device=DEVICE))
    c = Ainv @ (Phi.T @ Z)
    th = theta_hat(kxg, kyg, c)
    if th.shape[0] < 2: return None
    # FIM from sparse-obs geometry with theta_hat
    Jo = jacobian(th, X, Y, Tt, depth); nF = Jo.shape[1]
    F = (Jo.T @ Jo)/(SIGMA**2)
    Finv = torch.linalg.inv(F + 1e-8*torch.trace(F)/nF*torch.eye(nF, dtype=DT, device=DEVICE))
    # --- query grid x leads ---
    stride = max(1, (nx*ny)//400)
    qi = np.arange(0, ny*nx, stride)
    qx0 = torch.tensor(GXg.ravel()[qi], dtype=DT, device=DEVICE)
    qy0 = torch.tensor(GYg.ravel()[qi], dtype=DT, device=DEVICE)
    rows = dict(crb=[], act=[], emp=[], dist=[], nul=[])
    for lf in lead_fr:
        fj = i_now + lf
        if fj >= len(Zd): continue
        qt = torch.full_like(qx0, float(t[fj]))
        Phi_q = design(kxg, kyg, qx0, qy0, qt, depth)
        Jq = jacobian(th, qx0, qy0, qt, depth)
        crb = torch.sqrt(torch.clamp(torch.einsum('qa,ab,qb->q', Jq, Finv, Jq), min=0))/Hs
        emp = torch.sqrt(torch.clamp(torch.einsum('qa,ab,qb->q', Phi_q, Ainv, Phi_q), min=0))*SIGMA/Hs
        fc = (Phi_q @ c).cpu().numpy()
        ztrue = bilinear(Zd[fj], gx, gy, qx0, qy0)
        znull = bilinear(phase_scramble(Zd[fj], rng), gx, gy, qx0, qy0)
        act = np.abs(fc - ztrue)/Hs; nul = np.abs(fc - znull)/Hs
        d2 = ((qx0[:, None]-X[None, :])**2 + (qy0[:, None]-Y[None, :])**2).min(dim=1).values
        rows['crb'].append(crb.cpu().numpy()); rows['act'].append(act)
        rows['emp'].append(emp.cpu().numpy()); rows['dist'].append(torch.sqrt(d2).cpu().numpy())
        rows['nul'].append(nul)
    if not rows['crb']: return None
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    r_ca = sp(cat['crb'], cat['act']); r_cn = sp(cat['crb'], cat['nul'])
    r_ea = sp(cat['emp'], cat['act']); r_da = sp(cat['dist'], cat['act'])
    r_cd = sp(cat['crb'], cat['dist'])
    den = np.sqrt(max(1e-9, (1-r_cd**2)*(1-r_da**2))); r_ca_d = (r_ca - r_cd*r_da)/den
    return dict(spear_crb_act=r_ca, spear_crb_null=r_cn, delta_real_minus_null=r_ca-r_cn,
                partial_crb_act_dist=r_ca_d, spear_emp_act=r_ea, spear_dist_act=r_da,
                n_obs=int(len(ox)), n_theta=int(th.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', nargs='+', default=['AA15a', 'AA14', 'YS17'])
    ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    t0 = time.time(); allres = []
    for rc in a.recs:
        cfg = RECORDS[rc]; depth = cfg['depth']
        Z, gxn, gyn, tvec, meta = load_surface_nc(cfg['path'], tmax_frames=a.frames)
        GX, GY = np.meshgrid(gxn, gyn)
        Zd = detrend(Z, GX, GY); Hs = 4*np.nanstd(Zd)
        _, lam, _, kp = directional_spread(np.nan_to_num(Zd[len(Zd)//2]), meta['dx'], meta['dy'])
        gx = torch.tensor(gxn, dtype=DT, device=DEVICE); gy = torch.tensor(gyn, dtype=DT, device=DEVICE)
        fps = meta['fps'] or 5.0
        T = len(Zd)
        lead_fr = [int(round(s*fps)) for s in ([0, 2] if a.smoke else [0, 2, 4, 6])]
        maxlead = max(lead_fr)
        # obs window <= half the usable span so start indices remain valid at high fps
        W_fr = min(int(round(8.0*fps)), max(4, (T - maxlead - 2)//2))
        lo, hi = W_fr, T - maxlead - 1
        if hi <= lo:
            print(f"  {rc}: record too short (T={T}, fps={fps:.1f}) — skipped"); continue
        starts = [lo] if a.smoke else list(range(lo, hi, max(1, (hi-lo)//5)))
        starts = starts[:2] if a.smoke else starts[:6]
        print(f"  [{rc}] T={T} fps={fps:.1f} W_fr={W_fr} leads(fr)={lead_fr} starts={starts}", flush=True)
        for wi, i_now in enumerate(starts):
            r = run_window(Zd, gx, gy, tvec, depth, Hs, kp, i_now, W_fr, lead_fr, seed=wi)
            if r is None: continue
            r.update(rec=rc, i_now=i_now, kp=kp, lam=lam, aper_lam=min(meta['patch'])/lam, Hs=Hs)
            allres.append(r)
            print(f"  {rc:6s} w{wi} inow{i_now:3d}: crb-act {r['spear_crb_act']:+.3f} "
                  f"null {r['spear_crb_null']:+.3f} Δ {r['delta_real_minus_null']:+.3f} "
                  f"partial {r['partial_crb_act_dist']:+.3f} emp {r['spear_emp_act']:+.3f} "
                  f"dist {r['spear_dist_act']:+.3f} (nθ{r['n_theta']}) [{time.time()-t0:.0f}s]", flush=True)
    # summarize per record + overall (primary = AA/YS, LJ separate)
    def med(rows, k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])]); return np.median(v) if len(v) else np.nan
    print("\n=== per-record medians ===")
    for rc in a.recs:
        rr = [r for r in allres if r['rec'] == rc]
        if not rr: continue
        print(f"  {rc:6s} n={len(rr)} aper/λ={med(rr,'aper_lam'):.2f}: crb-act {med(rr,'spear_crb_act'):+.3f} "
              f"null {med(rr,'spear_crb_null'):+.3f} Δ {med(rr,'delta_real_minus_null'):+.3f} "
              f"partial {med(rr,'partial_crb_act_dist'):+.3f} emp {med(rr,'spear_emp_act'):+.3f} "
              f"dist {med(rr,'spear_dist_act'):+.3f}")
    prim = [r for r in allres if r['rec'] in ('AA15a', 'AA14', 'YS17')]
    if prim:
        print(f"\n=== PRIMARY (AA/YS, n={len(prim)}) ===")
        for k in ['spear_crb_act', 'delta_real_minus_null', 'partial_crb_act_dist', 'spear_emp_act', 'spear_dist_act']:
            print(f"  {k:26s} median {med(prim,k):+.3f}")
    json.dump(dict(results=allres), open('results/direction3/e59_real_anchor.json', 'w'), indent=1, default=float)
    print(f"\n[{time.time()-t0:.0f}s] saved -> results/direction3/e59_real_anchor.json")


if __name__ == '__main__':
    main()
