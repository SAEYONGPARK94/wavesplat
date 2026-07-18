"""E-59 v2 — §8 REAL-DATA anchor, fable-red-teamed (C1-C7). Prereg = tmp/E59_real_anchor_prereg_v2.md.

Fixes vs v1: C1 phase-scramble null (Delta_rho = real - null = coherent-skill signal beyond
geometry/spectrum); C2 WITHIN-LEAD spatial Spearman (kills lead-pooling t^2 artifact); C3 matched
TWIN (synthetic R=1 same pipeline = honest bar under attenuation); C4 STRIP obs -> full-patch query
(creates information gradient => real data can reach the under-determined regime); C5 operator detrend
+ ring lower-edge clip to 2pi/patch (no super-patch components); spatial local-RMS sigma_actual
(mitigates single-realization attenuation, the R=1 analog of phase-averaging). Truth-source-agnostic:
the SAME pipeline runs real detrended stereo and the synthetic twin (real NaN mask applied).

Usage: python3 tmp/e59v2_real_anchor.py [--mode real|twin|both] [--recs ...] [--smoke]
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch
from scipy import stats, ndimage
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from realdata_analysis import load_surface_nc, directional_spread
from d1_crb import DEVICE, DT, jacobian, omega_of_k

SIGMA = 0.02
# frozen (C7)
W_SEC = 6.0; N_OBS = 1200; TOPK = 24; QSTRIDE_TARGET = 600
F_STRIP = 0.40; RING_HI = 1.75; NK = 19; RIDGE_REL = 1e-3
RAW_SINK = None   # set to a list to collect raw per-lead (crb, act, dist, emp) query arrays
RECORDS = {
    'AA15a': dict(path='data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc', depth=17.0),
    'AA14':  dict(path='data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc', depth=17.0),
    'YS17':  dict(path='data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc', depth=50.0),
    'LJ18g': dict(path='data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc', depth=45.0),
}


def detrend(Z, GX, GY):
    out = np.empty_like(Z)
    for i in range(len(Z)):
        F = Z[i]; m = np.isfinite(F)
        if m.sum() < 20:
            out[i] = F; continue
        A = np.c_[np.ones(m.sum()), GX[m], GY[m]]
        coef, *_ = np.linalg.lstsq(A, F[m], rcond=None)
        out[i] = F - (coef[0] + coef[1]*GX + coef[2]*GY)
    return out


def synth_truth(GX, GY, tvec, kp, lam_p, spread_deg, depth, mask3d, Hs_target, N=15, seed=0):
    """Matched synthetic sea (twin): N dispersive components ~ kp with directional spread.
    Evaluated on the real grid/time; real NaN mask applied so holes/geometry match."""
    rng = np.random.default_rng(seed)
    kmag = kp*(1+0.35*rng.standard_normal(N)); kmag = np.clip(kmag, 0.4*kp, 1.8*kp)
    mdir = rng.uniform(0, 2*np.pi)
    ang = mdir + np.radians(spread_deg)*rng.standard_normal(N)
    kx = kmag*np.cos(ang); ky = kmag*np.sin(ang)
    om = np.sqrt(9.81*kmag*np.tanh(kmag*depth))
    amp = np.abs(1.0 + 0.3*rng.standard_normal(N)); phi = rng.uniform(0, 2*np.pi, N)
    ny, nx = GX.shape; surf = np.empty((len(tvec), ny, nx))
    for it, tt in enumerate(tvec):
        psi = (kx[None, None, :]*GX[..., None] + ky[None, None, :]*GY[..., None]
               - om[None, None, :]*tt + phi[None, None, :])
        surf[it] = (amp[None, None, :]*np.cos(psi)).sum(-1)
    surf *= Hs_target/(4*np.std(surf))
    surf[~mask3d] = np.nan
    return surf


def build_kgrid(kp, patch_min):
    ring_lo = max(0.35, 2*np.pi/(0.9*patch_min)/kp)           # C5 clip: no super-patch lambda
    ax = np.linspace(-RING_HI*kp, RING_HI*kp, NK)
    KX, KY = np.meshgrid(ax, ax); kx, ky = KX.ravel(), KY.ravel()
    km = np.hypot(kx, ky)/kp; keep = (km >= ring_lo) & (km <= RING_HI)
    return (torch.tensor(kx[keep], dtype=DT, device=DEVICE),
            torch.tensor(ky[keep], dtype=DT, device=DEVICE), ring_lo)


def design(kxg, kyg, x, y, t, depth):
    k = torch.hypot(kxg, kyg); om = omega_of_k(k, depth)
    psi = kxg[None, :]*x[:, None] + kyg[None, :]*y[:, None] - om[None, :]*t[:, None]
    return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)


def theta_hat(kxg, kyg, c, topk=TOPK):
    G = kxg.numel(); cc, cs = c[:G], c[G:]
    A = torch.hypot(cc, cs); phi = torch.atan2(-cs, cc)
    idx = torch.topk(A, min(topk, G)).indices
    return torch.stack([A[idx], kxg[idx], kyg[idx], phi[idx]], dim=1)


def box_rms(err2, valid, r_pix):
    """NaN-aware local RMS of a full-grid error^2 field (box radius r_pix)."""
    size = 2*int(r_pix)+1
    num = ndimage.uniform_filter(np.where(valid, err2, 0.0), size=size, mode='nearest')
    den = ndimage.uniform_filter(valid.astype(float), size=size, mode='nearest')
    out = np.sqrt(np.clip(num/np.clip(den, 1e-6, None), 0, None))
    out[den < 0.15] = np.nan
    return out


def scramble(frame, rng):
    """Phase-scramble surrogate (same amplitude spectrum, coherent structure destroyed).
    Bug-B fix: zero-fill deflates variance by valid_frac (Parseval) -> rescale surrogate to
    match nanstd(truth) over the valid mask so the null is not re-weighted toward forecast geometry."""
    v = np.isfinite(frame); f = np.nan_to_num(frame); F = np.fft.fft2(f)
    sur = np.fft.ifft2(np.abs(F)*np.exp(1j*rng.uniform(0, 2*np.pi, F.shape))).real
    sv = sur[v].std()
    if v.sum() > 10 and sv > 1e-9:
        sur = sur * (float(np.nanstd(frame[v]))/sv)
    return sur


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return stats.spearmanr(a[m], b[m]).correlation if m.sum() > 12 else np.nan


def run_window(surf, gx, gy, tvec, depth, Hs, kp, lam_p, patch_min, i_now, W_fr, lead_fr, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ny, nx = surf.shape[1:]
    GXn = gx.cpu().numpy(); GYn = gy.cpu().numpy(); GX, GY = np.meshgrid(GXn, GYn)
    # C4 strip: obs only from first F_STRIP of x-extent
    xlo, xhi = GXn.min(), GXn.max(); x_strip = xlo + F_STRIP*(xhi-xlo)
    in_strip = GX <= x_strip
    ox, oy, ot, oz = [], [], [], []
    for fi in range(max(0, i_now-W_fr), i_now+1):
        F = surf[fi]; m = np.isfinite(F) & in_strip
        yy, xx = np.where(m)
        if len(xx) == 0: continue
        pick = rng.choice(len(xx), min(len(xx), N_OBS//(W_fr+1)+1), replace=False)
        ox.append(GX[yy[pick], xx[pick]]); oy.append(GY[yy[pick], xx[pick]])
        ot.append(np.full(len(pick), tvec[fi])); oz.append(F[yy[pick], xx[pick]])
    if not ox: return []
    ox = np.concatenate(ox); oy = np.concatenate(oy); ot = np.concatenate(ot); oz = np.concatenate(oz)
    if len(ox) > N_OBS:
        s = rng.choice(len(ox), N_OBS, replace=False); ox, oy, ot, oz = ox[s], oy[s], ot[s], oz[s]
    X = torch.tensor(ox, dtype=DT, device=DEVICE); Y = torch.tensor(oy, dtype=DT, device=DEVICE)
    Tt = torch.tensor(ot, dtype=DT, device=DEVICE)
    Z = torch.tensor(oz + SIGMA*rng.standard_normal(len(oz)), dtype=DT, device=DEVICE)
    kxg, kyg, ring_lo = build_kgrid(kp, patch_min)
    Phi = design(kxg, kyg, X, Y, Tt, depth); Amat = Phi.T @ Phi
    lam = RIDGE_REL*torch.trace(Amat)/Amat.shape[0]
    Ainv = torch.linalg.inv(Amat + lam*torch.eye(Amat.shape[0], dtype=DT, device=DEVICE))
    c = Ainv @ (Phi.T @ Z); th = theta_hat(kxg, kyg, c)
    if th.shape[0] < 2: return []
    Jo = jacobian(th, X, Y, Tt, depth); nF = Jo.shape[1]; Fim = (Jo.T @ Jo)/(SIGMA**2)
    Finv = torch.linalg.inv(Fim + 1e-8*torch.trace(Fim)/nF*torch.eye(nF, dtype=DT, device=DEVICE))
    # query grid (strided) — FULL patch
    r_pix = max(2, int(round((lam_p/4)/((GXn[-1]-GXn[0])/nx))))
    stride = max(1, int(round(np.sqrt(nx*ny/QSTRIDE_TARGET))))
    qi = (np.arange(0, ny, stride)[:, None]*nx + np.arange(0, nx, stride)[None, :]).ravel()
    qx0 = torch.tensor(GX.ravel()[qi], dtype=DT, device=DEVICE)
    qy0 = torch.tensor(GY.ravel()[qi], dtype=DT, device=DEVICE)
    # full-grid coords for forecast/spatial-RMS
    fx = torch.tensor(GX.ravel(), dtype=DT, device=DEVICE); fy = torch.tensor(GY.ravel(), dtype=DT, device=DEVICE)
    d2 = ((qx0[:, None]-X[None, :])**2 + (qy0[:, None]-Y[None, :])**2).min(dim=1).values
    dist = torch.sqrt(d2).cpu().numpy()
    rows = []
    for lf in lead_fr:
        fj = i_now + lf
        if fj >= len(surf): continue
        tf = float(tvec[fj])
        qt = torch.full_like(qx0, tf)
        Jq = jacobian(th, qx0, qy0, qt, depth)
        crb = (torch.sqrt(torch.clamp(torch.einsum('qa,ab,qb->q', Jq, Finv, Jq), min=0))/Hs).cpu().numpy()
        Phi_q = design(kxg, kyg, qx0, qy0, qt, depth)
        emp = (torch.sqrt(torch.clamp(torch.einsum('qa,ab,qb->q', Phi_q, Ainv, Phi_q), min=0))*SIGMA/Hs).cpu().numpy()
        # full-grid forecast + spatial-RMS error vs truth (and scrambled null)
        ftt = torch.full_like(fx, tf)
        fc = (design(kxg, kyg, fx, fy, ftt, depth) @ c).cpu().numpy().reshape(ny, nx)
        truth = surf[fj]; valid = np.isfinite(truth)
        err2 = (fc - np.nan_to_num(truth))**2
        sig_act_f = box_rms(err2, valid, r_pix)
        nul = scramble(truth, rng); errn2 = (fc - nul)**2
        sig_nul_f = box_rms(errn2, valid, r_pix)   # Bug-A fix: SAME valid mask as real (identical q-set)
        act = (sig_act_f.ravel()[qi])/Hs; nulq = (sig_nul_f.ravel()[qi])/Hs
        r_ca = sp(crb, act); r_cn = sp(crb, nulq); r_ea = sp(emp, act); r_da = sp(dist, act)
        if RAW_SINK is not None:
            RAW_SINK.append(dict(crb=crb, act=act, dist=dist, emp=emp))
        rows.append(dict(lead_fr=lf, spear_crb_act=r_ca, spear_crb_null=r_cn,
                         delta=r_ca-r_cn if (np.isfinite(r_ca) and np.isfinite(r_cn)) else np.nan,
                         spear_emp_act=r_ea, spear_dist_act=r_da, ring_lo=ring_lo, n_theta=int(th.shape[0])))
    return rows


def process_record(rc, mode, frames, smoke):
    cfg = RECORDS[rc]; depth = cfg['depth']
    Z, gxn, gyn, tvec, meta = load_surface_nc(cfg['path'], tmax_frames=frames)
    GX, GY = np.meshgrid(gxn, gyn); Zd = detrend(Z, GX, GY)
    _, lam, _, kp = directional_spread(np.nan_to_num(Zd[len(Zd)//2]), meta['dx'], meta['dy'])
    spread, _, _, _ = directional_spread(np.nan_to_num(Zd[len(Zd)//2]), meta['dx'], meta['dy'])
    Hs = 4*np.nanstd(Zd); patch_min = min(meta['patch']); mask3d = np.isfinite(Z)
    if mode == 'twin':
        surf = synth_truth(GX, GY, tvec, kp, lam, spread, depth, mask3d, Hs, seed=hash(rc) % 999)
        Hs = 4*np.nanstd(surf)
    else:
        surf = Zd
    gx = torch.tensor(gxn, dtype=DT, device=DEVICE); gy = torch.tensor(gyn, dtype=DT, device=DEVICE)
    fps = meta['fps'] or 5.0; T = len(surf)
    lead_fr = [int(round(s*fps)) for s in ([0, 2] if smoke else [0, 2, 4, 6])]
    maxlead = max(lead_fr); W_fr = min(int(round(W_SEC*fps)), max(4, (T-maxlead-2)//2))
    lo, hi = W_fr, T-maxlead-1
    if hi <= lo:
        print(f"  {rc}: too short (T={T})"); return []
    starts = ([lo, (lo+hi)//2] if smoke else list(range(lo, hi, max(1, (hi-lo)//5))))[:2 if smoke else 6]
    out = []
    for wi, i_now in enumerate(starts):
        for r in run_window(surf, gx, gy, tvec, depth, Hs, kp, lam, patch_min, i_now, W_fr, lead_fr, seed=wi):
            r.update(rec=rc, mode=mode, window=wi, kp=kp, lam=lam, spread=spread,
                     aper_lam=patch_min/lam, Hs=Hs)
            out.append(r)
    return out


def summarize(rows, tag):
    print(f"\n=== {tag}: within-lead spatial Spearman (per rec, median over window×lead) ===")
    recs = sorted(set(r['rec'] for r in rows))
    for rc in recs:
        rr = [r for r in rows if r['rec'] == rc]
        def m(k):
            v = np.array([x[k] for x in rr if np.isfinite(x[k])]); return np.median(v) if len(v) else np.nan
        print(f"  {rc:6s} n={len(rr):2d} aper/λ={m('aper_lam'):.2f} spread={m('spread'):.0f}° ring_lo={m('ring_lo'):.2f}: "
              f"crb-act {m('spear_crb_act'):+.3f} null {m('spear_crb_null'):+.3f} Δ {m('delta'):+.3f} "
              f"emp {m('spear_emp_act'):+.3f} dist {m('spear_dist_act'):+.3f}")
    prim = [r for r in rows if r['rec'] in ('AA15a', 'AA14', 'YS17')]
    if prim:
        def mm(k):
            v = np.array([x[k] for x in prim if np.isfinite(x[k])]); return v
        print(f"  --- PRIMARY {tag} (AA/YS): crb-act med {np.median(mm('spear_crb_act')):+.3f}  "
              f"Δ(real-null) med {np.median(mm('delta')):+.3f} "
              f"[blocked-boot CI {blocked_ci(prim,'delta')}]  emp {np.median(mm('spear_emp_act')):+.3f}")
    return


def blocked_ci(rows, key, nboot=2000):
    recs = sorted(set(r['rec'] for r in rows))
    by = {rc: np.array([r[key] for r in rows if r['rec'] == rc and np.isfinite(r[key])]) for rc in recs}
    by = {k: v for k, v in by.items() if len(v)}
    if len(by) < 2: return "n/a"
    rng = np.random.default_rng(0); bs = []
    for _ in range(nboot):
        chosen = rng.choice(list(by), len(by), replace=True)
        vals = np.concatenate([rng.choice(by[c], len(by[c]), replace=True) for c in chosen])
        bs.append(np.median(vals))
    return f"[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='both', choices=['real', 'twin', 'both'])
    ap.add_argument('--recs', nargs='+', default=['AA15a', 'AA14', 'YS17'])
    ap.add_argument('--frames', type=int, default=128)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); t0 = time.time(); allrows = []
    modes = ['twin', 'real'] if a.mode == 'both' else [a.mode]
    for mode in modes:
        rows = []
        for rc in a.recs:
            rows += process_record(rc, mode, a.frames, a.smoke)
        summarize(rows, mode.upper())
        allrows += rows
        print(f"  [{mode} done {time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(rows=allrows), open('results/direction3/e59v2_real_anchor.json', 'w'), indent=1, default=float)
    print(f"\n[{time.time()-t0:.0f}s] saved -> results/direction3/e59v2_real_anchor.json")


if __name__ == '__main__':
    main()
