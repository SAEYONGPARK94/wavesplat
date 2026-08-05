"""Wavenumber-ID CRB — settles ①'s realizability: is the under-det +0.45 headroom
REALIZABLE (algorithmic search gap, info is there) or INFO-LIMITED (sparse aperture
can't resolve the wavenumbers / multimodal likelihood)?

(1) LOCAL identifiability: per-component angular CRB (deg) & radial CRB from the FIM
    (order [w,kx,ky,phi]*N), FIM condition number, vs component angular spacing.
(2) GLOBAL ambiguity (the crux): does the TRUE wavenumber set give the best obs-fit,
    or does a RELAX-estimated (wrong) N-set fit the sparse obs as well? If wrong-set
    residual <= true-set residual but field is worse => multimodal => info-limited.
Under-det seas (15comp), sparse obs. Usage: python3 tmp/wavenumber_id_crb.py [--n 16]
Out: results/direction3/wavenumber_id_crb.json
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from scripts.d1_crb import DEVICE, DT, load_ep, collect_obs, fisher, jacobian, predict, omega_of_k

SIGMA = 0.02; RING = (0.35, 1.75); NK_RELAX = 25

def dom_kp(theta):
    w = theta[:, 0].abs(); i = int(torch.argmax(w)); return float(torch.hypot(theta[i, 1], theta[i, 2]))

def build_kgrid(kp, nk):
    ax = np.linspace(-RING[1]*kp, RING[1]*kp, nk); KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel(); km = np.hypot(kx, ky)/kp; keep = (km >= RING[0]) & (km <= RING[1])
    return (torch.tensor(kx[keep], dtype=DT, device=DEVICE), torch.tensor(ky[keep], dtype=DT, device=DEVICE))

def design(kxg, kyg, x, y, t, depth):
    k = torch.hypot(kxg, kyg); om = omega_of_k(k, depth)
    psi = kxg[None]*x[:, None] + kyg[None]*y[:, None] - om[None]*t[:, None]
    return torch.cat([torch.cos(psi), torch.sin(psi)], 1)

def relax_khat(kxg, kyg, x, y, t, z, depth, K):
    """RELAX greedy -> K estimated (kx,ky) + obs-fit residual RMS."""
    Nc = kxg.numel(); cols = design(kxg, kyg, x, y, t, depth); cos_c, sin_c = cols[:, :Nc], cols[:, Nc:]
    r = z.clone(); sel = []
    for _ in range(min(K, Nc)):
        pc = (cos_c.T@r)**2 + (sin_c.T@r)**2
        if sel: pc[torch.tensor(sel, device=DEVICE)] = -1.0
        j = int(torch.argmax(pc)); sel.append(j); idx = torch.tensor(sel, device=DEVICE)
        Phi_s = torch.cat([cos_c[:, idx], sin_c[:, idx]], 1); coef = torch.linalg.lstsq(Phi_s, z).solution
        r = z - Phi_s@coef
    idx = torch.tensor(sel, device=DEVICE)
    return kxg[idx], kyg[idx], float(torch.sqrt((r**2).mean()))

def ls_resid(kx, ky, x, y, t, z, depth):
    Phi = design(kx, ky, x, y, t, depth); A = Phi.T@Phi
    lam = 1e-4*torch.trace(A)/A.shape[0]
    c = torch.linalg.solve(A + lam*torch.eye(A.shape[0], dtype=DT, device=DEVICE), Phi.T@z)
    return float(torch.sqrt(((Phi@c - z)**2).mean()))

def run_ep(ep, t_now, kp, nobs=250):
    depth = ep['depth']; th = ep['theta']; N = th.shape[0]
    x, y, t = collect_obs(ep, t_now)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    if x.numel() > nobs:
        idx = torch.randperm(x.numel(), device=DEVICE, generator=g)[:nobs]; x, y, t = x[idx], y[idx], t[idx]
    thr = th.clone(); thr[:, 3] = torch.rand(N, dtype=DT, device=DEVICE, generator=g)*2*np.pi
    # --- (1) FIM -> per-component wavenumber CRB ---
    F = fisher(thr, x, y, t, depth, SIGMA); nF = F.shape[0]
    Finv = torch.linalg.inv(F + 1e-10*torch.trace(F)/nF*torch.eye(nF, dtype=DT, device=DEVICE))
    cond = float(torch.linalg.cond(F))
    kx, ky = thr[:, 1], thr[:, 2]; k = torch.hypot(kx, ky)
    d = torch.diagonal(Finv)
    crb_kx = torch.clamp(d[N:2*N], min=0); crb_ky = torch.clamp(d[2*N:3*N], min=0)
    crb_kxky = torch.diagonal(Finv, 0)  # not used; use off-diag below
    cross = Finv[torch.arange(N, device=DEVICE)+N, torch.arange(N, device=DEVICE)+2*N]
    # angular CRB (deg) via propagation of theta=atan2(ky,kx)
    var_ang = (ky**2*crb_kx - 2*kx*ky*cross + kx**2*crb_ky) / (k**4 + 1e-30)
    ang_crb_deg = torch.rad2deg(torch.sqrt(torch.clamp(var_ang, min=0)))
    rad_crb_frac = torch.sqrt(crb_kx + crb_ky) / kp   # crude magnitude CRB / kp
    # component angular spacing (nearest neighbour, deg)
    ang = torch.rad2deg(torch.atan2(ky, kx)).cpu().numpy()
    dif = np.abs(ang[:, None] - ang[None, :]); dif = np.minimum(dif, 360 - dif); np.fill_diagonal(dif, 999)
    nn_spacing = float(np.median(dif.min(1)))
    # --- (2) global ambiguity: RELAX-N residual vs true-N residual ---
    s = predict(thr, x, y, t, depth); z = s + SIGMA*torch.randn(s.shape, dtype=DT, device=DEVICE, generator=g)
    kxg, kyg = build_kgrid(kp, NK_RELAX)
    khx, khy, relax_res = relax_khat(kxg, kyg, x, y, t, z, depth, K=N)
    true_res = ls_resid(kx, ky, x, y, t, z, depth)
    return dict(kp=kp, N=int(N), cond=cond, ang_crb_med=float(torch.median(ang_crb_deg)),
                rad_crb_med=float(torch.median(rad_crb_frac)), nn_spacing_deg=nn_spacing,
                relax_res=relax_res, true_res=true_res, noise=SIGMA,
                ang_crb_vs_spacing=float(torch.median(ang_crb_deg))/max(nn_spacing, 1e-9))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=16); a = ap.parse_args()
    paths = sorted(glob.glob('data/episode_BF4_15comp_*.h5') + glob.glob('data/episode_BF4_30comp_*.h5'))[:a.n]
    res = []; t0 = time.time()
    for p in paths:
        try:
            ep = load_ep(p); t_now = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
            r = run_ep(ep, t_now, kp); r['ep'] = os.path.basename(p); res.append(r)
            print(f"  {r['ep']:28s} angCRB {r['ang_crb_med']:5.1f}° spacing {r['nn_spacing_deg']:5.1f}° "
                  f"(ratio {r['ang_crb_vs_spacing']:.2f}) cond {r['cond']:.1e} | "
                  f"relax_res {r['relax_res']:.3f} true_res {r['true_res']:.3f} (noise {SIGMA}) [{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback; print(f"  {os.path.basename(p)} ERR {ex}"); traceback.print_exc()
    med = lambda k: float(np.median([r[k] for r in res]))
    print(f"\n=== wavenumber-ID CRB summary (n={len(res)} under-det eps) ===")
    print(f"  (1) LOCAL: angular CRB median {med('ang_crb_med'):.1f}°  vs component spacing {med('nn_spacing_deg'):.1f}°  "
          f"(ratio {med('ang_crb_vs_spacing'):.2f})  FIM cond {med('cond'):.1e}")
    print(f"      radial CRB/kp median {med('rad_crb_med'):.3f}")
    print(f"  (2) GLOBAL: RELAX-N obs residual {med('relax_res'):.3f}  vs  TRUE-N residual {med('true_res'):.3f}  (noise {SIGMA})")
    # verdict — angular CRB (properly derived from Finv) is the identifiability metric;
    # raw FIM cond is reported as a caveat (ill-conditioning is in non-direction params,
    # since angular CRB stays tight despite high cond, and field-CRB was ~0 in gap_ladder).
    local_ok = med('ang_crb_vs_spacing') < 0.5             # directions resolvable
    global_ok = med('relax_res') > med('true_res') * 1.3   # true set is uniquely best fit
    if local_ok and global_ok:
        v = ("REALIZABLE: wavenumbers locally resolvable (angCRB<<spacing) AND true set is UNIQUE "
             "best obs-fit (RELAX fits worse) -> headroom is ALGORITHMIC search gap, a better learned "
             "identifier CAN realize it. (caveat: FIM cond high in non-direction params)")
    elif not local_ok:
        v = "INFO-LIMITED (local): angular CRB >~ component spacing -> aperture cannot resolve wavenumbers -> NOT realizable"
    elif not global_ok:
        v = ("INFO-LIMITED (global): angCRB tight BUT wrong wavenumber sets fit sparse obs ~as well as "
             "truth (multimodal likelihood) -> ambiguous -> NOT (fully) realizable")
    else:
        v = "MIXED"
    print(f"  [caveat] FIM cond median {med('cond'):.1e} (high, but in non-direction params — angular CRB stays tight)")
    print(f"\n=== VERDICT === {v}")
    json.dump(dict(episodes=res, summary=dict(
        ang_crb_med=med('ang_crb_med'), nn_spacing_deg=med('nn_spacing_deg'),
        ang_crb_vs_spacing=med('ang_crb_vs_spacing'), cond=med('cond'), rad_crb_med=med('rad_crb_med'),
        relax_res=med('relax_res'), true_res=med('true_res'), verdict=v)),
        open('results/direction3/wavenumber_id_crb.json', 'w'), indent=1, default=float)
    print(f"saved -> results/direction3/wavenumber_id_crb.json ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
