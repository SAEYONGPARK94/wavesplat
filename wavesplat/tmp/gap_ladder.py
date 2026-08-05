"""① under-det algorithmic-gap ladder. Question: on under-determined seas (fixed
geometry), is there REALIZABLE headroom for a learning estimator beyond net-MMSE
(the current best)? Measures achieved field-reconstruction corr(pred,truth) for a
ladder: RELAX(greedy) / ridge(dense LS) / net-MMSE(learned prior, beamform kp) /
oracle-LS(TRUE wavenumbers = perfect-ID ceiling), vs the field-CRB info ceiling.
Synthetic obs (predict(theta)+noise), 15/30-comp seas. Reuses c8/e58 machinery.
Usage: python3 tmp/gap_ladder.py [--n 16] [--R 6]
Out: results/direction3/gap_ladder.json
"""
import os, sys, glob, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from scripts.d1_crb import (DEVICE, DT, load_ep, collect_obs, jacobian, fisher,
                            predict, omega_of_k, field_design)
from d3_prior_net_eval import get_net, net_gamma_mmse

SIGMA = 0.02; RING = (0.35, 1.75); NK = 15; NK_RELAX = 25; K_BUDGET = 12
_net = None
def net():
    global _net
    if _net is None: _net = get_net()
    return _net

def dom_kp(theta):
    w = theta[:, 0].abs(); i = int(torch.argmax(w))
    return float(torch.hypot(theta[i, 1], theta[i, 2]))

def build_kgrid(kp, nk):
    ax = np.linspace(-RING[1]*kp, RING[1]*kp, nk); KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel(); km = np.hypot(kx, ky)/kp
    keep = (km >= RING[0]) & (km <= RING[1])
    return (torch.tensor(kx[keep], dtype=DT, device=DEVICE),
            torch.tensor(ky[keep], dtype=DT, device=DEVICE))

def design(kxg, kyg, x, y, t, depth):
    k = torch.hypot(kxg, kyg); om = omega_of_k(k, depth)
    psi = kxg[None]*x[:, None] + kyg[None]*y[:, None] - om[None]*t[:, None]
    return torch.cat([torch.cos(psi), torch.sin(psi)], 1)

def ls_fit_pred(kxg, kyg, x, y, t, z, qx, qy, qt, depth, ridge_rel=1e-3):
    Phi = design(kxg, kyg, x, y, t, depth); A = Phi.T@Phi
    lam = ridge_rel*torch.trace(A)/A.shape[0]
    c = torch.linalg.solve(A + lam*torch.eye(A.shape[0], dtype=DT, device=DEVICE), Phi.T@z)
    return design(kxg, kyg, qx, qy, qt, depth)@c

def relax_pred(kxg, kyg, x, y, t, z, qx, qy, qt, depth, K=K_BUDGET):
    N = kxg.numel(); cols = design(kxg, kyg, x, y, t, depth)
    cos_c, sin_c = cols[:, :N], cols[:, N:]; r = z.clone(); sel = []
    for _ in range(min(K, N)):
        pc = (cos_c.T@r)**2 + (sin_c.T@r)**2
        if sel: pc[torch.tensor(sel, device=DEVICE)] = -1.0
        j = int(torch.argmax(pc)); sel.append(j); idx = torch.tensor(sel, device=DEVICE)
        Phi_s = torch.cat([cos_c[:, idx], sin_c[:, idx]], 1)
        coef = torch.linalg.lstsq(Phi_s, z).solution; r = z - Phi_s@coef
    idx = torch.tensor(sel, device=DEVICE)
    Phi_q = design(kxg[idx], kyg[idx], qx, qy, qt, depth)
    return Phi_q@coef

def oracle_ls_pred(theta, x, y, t, z, qx, qy, qt, depth):
    """LS at the TRUE wavenumbers (perfect-ID ceiling)."""
    kx, ky = theta[:, 1], theta[:, 2]
    return ls_fit_pred(kx, ky, x, y, t, z, qx, qy, qt, depth, ridge_rel=1e-4)

def corr(a, b):
    a, b = a.cpu().numpy(), b.cpu().numpy()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])

def run_ep(ep, t_now, kp, R=6):
    depth, Hs = ep['depth'], ep['Hs']; th0 = ep['theta']; N = th0.shape[0]
    x, y, t = collect_obs(ep, t_now)
    nobs = int(os.environ.get('GAP_NOBS', '250'))   # SPARSE -> genuine under-det
    if x.numel() > nobs:
        idx = torch.randperm(x.numel(), device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(0))[:nobs]
        x, y, t = x[idx], y[idx], t[idx]
    kxg, kyg = build_kgrid(kp, NK); kxr, kyr = build_kgrid(kp, NK_RELAX)
    gx, gy = ep['gx'], ep['gy']; stride = max(1, (gx.numel()*gy.numel())//400)
    Gx, Gy = torch.meshgrid(gx, gy, indexing='xy')
    qx, qy = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
    Pnp = None
    accs = {k: [] for k in ('relax', 'ridge', 'oracle', 'netmmse')}
    crb_acc = []
    for _ in range(R):
        thr = th0.clone(); thr[:, 3] = torch.rand(N, dtype=DT, device=DEVICE)*2*np.pi
        qt = torch.full_like(qx, float(t_now))
        truth = predict(thr, qx, qy, qt, depth)
        s = predict(thr, x, y, t, depth); z = s + SIGMA*torch.randn_like(s)
        accs['relax'].append(corr(relax_pred(kxr, kyr, x, y, t, z, qx, qy, qt, depth), truth))
        accs['ridge'].append(corr(ls_fit_pred(kxg, kyg, x, y, t, z, qx, qy, qt, depth), truth))
        accs['oracle'].append(corr(oracle_ls_pred(thr, x, y, t, z, qx, qy, qt, depth), truth))
        # net-MMSE (beamform kp, non-oracle)
        P = torch.stack([x, y, t], 1).cpu().numpy(); Q = torch.stack([qx, qy, qt], 1).cpu().numpy()
        pn = np.asarray(net_gamma_mmse(net(), P, z.cpu().numpy(), depth, Q, SIGMA, seed=0,
                                       kp_hint=None, use_em=False, smult_fixed=3.25))
        accs['netmmse'].append(corr(torch.tensor(pn, dtype=DT, device=DEVICE), truth))
        # field CRB -> achievable nRMSE ceiling (corr proxy: corr>=sqrt(1-nRMSE^2))
        Fr = fisher(thr, x, y, t, depth, SIGMA); nF = Fr.shape[0]
        Finv = torch.linalg.inv(Fr + 1e-10*torch.trace(Fr)/nF*torch.eye(nF, dtype=DT, device=DEVICE))
        Gm = field_design(thr, gx, gy, float(t_now), depth)
        var = torch.trace(Finv @ Gm); nrmse_crb = float(torch.sqrt(torch.clamp(var, min=0)))/Hs
        crb_acc.append(nrmse_crb)
    out = {k: float(np.nanmean(v)) for k, v in accs.items()}
    out['nrmse_crb'] = float(np.mean(crb_acc)); out['kp'] = kp
    out['corr_ceil_crb'] = float(np.sqrt(max(0, 1 - min(1, np.mean(crb_acc))**2)))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=16); ap.add_argument('--R', type=int, default=6)
    a = ap.parse_args()
    paths = sorted(glob.glob('data/episode_BF4_15comp_*.h5') + glob.glob('data/episode_BF4_30comp_*.h5'))[:a.n]
    res = []; t0 = time.time()
    for p in paths:
        try:
            ep = load_ep(p); t_now = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
            r = run_ep(ep, t_now, kp, a.R); r['ep'] = os.path.basename(p); res.append(r)
            print(f"  {r['ep']:30s} relax {r['relax']:.3f} ridge {r['ridge']:.3f} "
                  f"net {r['netmmse']:.3f} oracle {r['oracle']:.3f} | crb-ceil {r['corr_ceil_crb']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback; print(f"  {os.path.basename(p)} ERR {ex}"); traceback.print_exc()
    # ladder summary
    def med(k): return float(np.nanmedian([r[k] for r in res]))
    print(f"\n=== achievability ladder (achieved corr, n={len(res)} under-det eps, higher=better) ===")
    for k in ['relax', 'ridge', 'netmmse', 'oracle', 'corr_ceil_crb']:
        print(f"  {k:14s} {med(k):.3f}")
    head_id = med('oracle') - med('netmmse')      # headroom to perfect-ID ceiling
    head_crb = med('corr_ceil_crb') - med('netmmse')  # headroom to CRB
    net_over_relax = med('netmmse') - med('relax')
    print(f"\n  net-MMSE − RELAX (learning gain, MEASURED before): {net_over_relax:+.3f}")
    print(f"  oracle-LS − net-MMSE (headroom via better ID): {head_id:+.3f}")
    print(f"  CRB-ceil − net-MMSE (total headroom): {head_crb:+.3f}")
    verdict = ('HEADROOM: net-MMSE well below perfect-ID ceiling -> better ID/estimator could gain' if head_id > 0.05
               else 'CLOSED: net-MMSE ~= perfect-ID ceiling -> remaining gap is info(ID)-limited, little learning headroom' if head_id < 0.02
               else 'MODEST headroom')
    print(f"\n=== VERDICT === {verdict}")
    json.dump(dict(episodes=res, summary=dict(
        relax=med('relax'), ridge=med('ridge'), netmmse=med('netmmse'), oracle=med('oracle'),
        corr_ceil_crb=med('corr_ceil_crb'), headroom_id=head_id, headroom_crb=head_crb,
        net_over_relax=net_over_relax, verdict=verdict)),
        open('results/direction3/gap_ladder.json', 'w'), indent=1, default=float)
    print(f"saved -> results/direction3/gap_ladder.json ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
