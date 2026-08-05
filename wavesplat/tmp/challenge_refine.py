"""CHALLENGE — demonstrate the under-det +0.45 headroom is REALIZABLE by a better
wavenumber identifier. CRB verdict: info is there, truth uniquely best-fit, current
methods fail the non-convex SEARCH. So: off-grid refinement (autograd variable-
projection) on the wavenumbers, from a grid-RELAX init AND from a net-MMSE-informed
(fine-beamform) init. Beat net-MMSE (0.55) toward oracle-LS (1.0) => realizable DEMONSTRATED.
Prereg: PASS = best refine beats net-MMSE by Delta>=0.05 (n>=16, CI excl 0).
Usage: python3 tmp/challenge_refine.py [--n 16] [--R 6]
Out: results/direction3/challenge_refine.json
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from scripts.d1_crb import DEVICE, DT, load_ep, collect_obs, predict, omega_of_k
from d3_prior_net_eval import get_net, net_gamma_mmse

SIGMA = 0.02; RING = (0.35, 1.75)
_net = None
def net():
    global _net
    if _net is None: _net = get_net()
    return _net

def dom_kp(theta):
    w = theta[:, 0].abs(); i = int(torch.argmax(w)); return float(torch.hypot(theta[i, 1], theta[i, 2]))

def build_kgrid(kp, nk):
    ax = np.linspace(-RING[1]*kp, RING[1]*kp, nk); KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel(); km = np.hypot(kx, ky)/kp; keep = (km >= RING[0]) & (km <= RING[1])
    return (torch.tensor(kx[keep], dtype=DT, device=DEVICE), torch.tensor(ky[keep], dtype=DT, device=DEVICE))

def design(kx, ky, x, y, t, depth):
    k = torch.hypot(kx, ky) + 1e-12; om = omega_of_k(k, depth)
    psi = kx[None]*x[:, None] + ky[None]*y[:, None] - om[None]*t[:, None]
    return torch.cat([torch.cos(psi), torch.sin(psi)], 1)

def ls_solve(Phi, z, rr=1e-3):
    A = Phi.T@Phi; lam = rr*torch.trace(A)/A.shape[0]
    return torch.linalg.solve(A + lam*torch.eye(A.shape[0], dtype=DT, device=DEVICE), Phi.T@z)

def greedy_relax(kxg, kyg, x, y, t, z, depth, K):
    Nc = kxg.numel(); cols = design(kxg, kyg, x, y, t, depth); cos_c, sin_c = cols[:, :Nc], cols[:, Nc:]
    r = z.clone(); sel = []
    for _ in range(min(K, Nc)):
        pc = (cos_c.T@r)**2 + (sin_c.T@r)**2
        if sel: pc[torch.tensor(sel, device=DEVICE)] = -1.0
        j = int(torch.argmax(pc)); sel.append(j); idx = torch.tensor(sel, device=DEVICE)
        Phi_s = torch.cat([cos_c[:, idx], sin_c[:, idx]], 1); coef = torch.linalg.lstsq(Phi_s, z).solution
        r = z - Phi_s@coef
    idx = torch.tensor(sel, device=DEVICE)
    return kxg[idx].clone(), kyg[idx].clone()

def refine_offgrid(kx0, ky0, x, y, t, z, depth, kp, iters=60):
    """Variable-projection off-grid refinement: minimize ||z - Phi(k)c||^2 over k
    (c solved by LS each step, differentiable)."""
    kx = kx0.clone().detach().requires_grad_(True); ky = ky0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([kx, ky], lr=0.02*kp)
    for _ in range(iters):
        opt.zero_grad()
        Phi = design(kx, ky, x, y, t, depth)
        c = ls_solve(Phi, z)
        loss = ((Phi@c - z)**2).mean()
        loss.backward(); opt.step()
    return kx.detach(), ky.detach()

def pred_field(kx, ky, x, y, t, z, qx, qy, qt, depth):
    Phi = design(kx, ky, x, y, t, depth); c = ls_solve(Phi, z, 1e-4)
    return design(kx, ky, qx, qy, qt, depth)@c

def corr(a, b):
    a, b = a.cpu().numpy(), b.cpu().numpy(); m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan

def run_ep(ep, t_now, kp, R=6, nobs=250):
    depth = ep['depth']; th0 = ep['theta']; N = th0.shape[0]
    x, y, t = collect_obs(ep, t_now)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    if x.numel() > nobs:
        idx = torch.randperm(x.numel(), device=DEVICE, generator=g)[:nobs]; x, y, t = x[idx], y[idx], t[idx]
    gx, gy = ep['gx'], ep['gy']; stride = max(1, (gx.numel()*gy.numel())//400)
    Gx, Gy = torch.meshgrid(gx, gy, indexing='xy'); qx, qy = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
    kxg, kyg = build_kgrid(kp, 25); kxf, kyf = build_kgrid(kp, 41)   # coarse init, fine init
    acc = {k: [] for k in ('relax', 'refine', 'refine_fine', 'netmmse', 'oracle')}
    for _ in range(R):
        thr = th0.clone(); thr[:, 3] = torch.rand(N, dtype=DT, device=DEVICE, generator=g)*2*np.pi
        qt = torch.full_like(qx, float(t_now)); truth = predict(thr, qx, qy, qt, depth)
        s = predict(thr, x, y, t, depth); z = s + SIGMA*torch.randn(s.shape, dtype=DT, device=DEVICE, generator=g)
        # RELAX grid greedy
        kx_r, ky_r = greedy_relax(kxg, kyg, x, y, t, z, depth, N)
        acc['relax'].append(corr(pred_field(kx_r, ky_r, x, y, t, z, qx, qy, qt, depth), truth))
        # off-grid refine (coarse init)
        kx_o, ky_o = refine_offgrid(kx_r, ky_r, x, y, t, z, depth, kp)
        acc['refine'].append(corr(pred_field(kx_o, ky_o, x, y, t, z, qx, qy, qt, depth), truth))
        # off-grid refine (fine-grid init = better search)
        kx_rf, ky_rf = greedy_relax(kxf, kyf, x, y, t, z, depth, N)
        kx_of, ky_of = refine_offgrid(kx_rf, ky_rf, x, y, t, z, depth, kp)
        acc['refine_fine'].append(corr(pred_field(kx_of, ky_of, x, y, t, z, qx, qy, qt, depth), truth))
        # net-MMSE
        P = torch.stack([x, y, t], 1).cpu().numpy(); Q = torch.stack([qx, qy, qt], 1).cpu().numpy()
        pn = np.asarray(net_gamma_mmse(net(), P, z.cpu().numpy(), depth, Q, SIGMA, seed=0,
                                       kp_hint=None, use_em=False, smult_fixed=3.25))
        acc['netmmse'].append(corr(torch.tensor(pn, dtype=DT, device=DEVICE), truth))
        # oracle
        acc['oracle'].append(corr(pred_field(thr[:, 1], thr[:, 2], x, y, t, z, qx, qy, qt, depth), truth))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=16); ap.add_argument('--R', type=int, default=6); a = ap.parse_args()
    paths = sorted(glob.glob('data/episode_BF4_15comp_*.h5') + glob.glob('data/episode_BF4_30comp_*.h5'))[:a.n]
    res = []; t0 = time.time()
    for p in paths:
        try:
            ep = load_ep(p); t_now = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
            r = run_ep(ep, t_now, kp, a.R); r['ep'] = os.path.basename(p); res.append(r)
            print(f"  {r['ep']:28s} relax {r['relax']:.3f} refine {r['refine']:.3f} "
                  f"refine_fine {r['refine_fine']:.3f} net {r['netmmse']:.3f} oracle {r['oracle']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback; print(f"  {os.path.basename(p)} ERR {ex}"); traceback.print_exc()
    def arr(k): return np.array([r[k] for r in res])
    print(f"\n=== CHALLENGE: achieved corr (n={len(res)}, higher=better) ===")
    for k in ['relax', 'netmmse', 'refine', 'refine_fine', 'oracle']:
        print(f"  {k:14s} {np.nanmedian(arr(k)):.3f}")
    # verdict: best refine vs net-MMSE (paired)
    best = 'refine_fine' if np.nanmedian(arr('refine_fine')) >= np.nanmedian(arr('refine')) else 'refine'
    d = arr(best) - arr('netmmse'); m = np.isfinite(d); d = d[m]
    rng = np.random.default_rng(0); bs = [np.mean(rng.choice(d, len(d))) for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    frac_closed = (np.nanmedian(arr(best)) - np.nanmedian(arr('netmmse'))) / max(1e-9, np.nanmedian(arr('oracle')) - np.nanmedian(arr('netmmse')))
    print(f"\n  best refine ({best}) − net-MMSE: Δ {d.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}] {int((d>0).sum())}/{len(d)}")
    print(f"  headroom closed vs oracle: {frac_closed*100:.0f}%")
    pass_ = d.mean() >= 0.05 and lo > 0
    v = ("PASS: better search (off-grid refine) BEATS net-MMSE -> realizability DEMONSTRATED (headroom is search-closable)"
         if pass_ else "FAIL: off-grid refine does not beat net-MMSE -> search alone insufficient, bigger redesign needed")
    print(f"\n=== VERDICT === {v}")
    json.dump(dict(episodes=res, summary=dict(
        relax=float(np.nanmedian(arr('relax'))), netmmse=float(np.nanmedian(arr('netmmse'))),
        refine=float(np.nanmedian(arr('refine'))), refine_fine=float(np.nanmedian(arr('refine_fine'))),
        oracle=float(np.nanmedian(arr('oracle'))), best=best, delta=float(d.mean()), ci=[float(lo), float(hi)],
        frac_closed=float(frac_closed), verdict=v)), open('results/direction3/challenge_refine.json', 'w'), indent=1, default=float)
    print(f"saved ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
