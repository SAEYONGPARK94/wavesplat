"""Dense-bank ceiling test — before training a 'better net', determine whether the
net-MMSE bank ARCHITECTURE (dense 51x51 ring grid + gamma prior + closed MMSE) can
even REACH the oracle ceiling. Place ORACLE gamma (true component energies on nearest
bank cells) and run the same MMSE. If ~= oracle-LS(1.0): bank OK -> a better learned
gamma helps. If << oracle: dense-bank plateau -> need off-grid/sparse representation.
Under-det seas (15comp), sparse obs. Usage: python3 tmp/bank_ceiling.py [--n 16]
Out: results/direction3/bank_ceiling.json
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from scripts.d1_crb import DEVICE, DT, load_ep, collect_obs, predict, omega_of_k
from d3_prior_net_eval import get_net, net_gamma_mmse

SIGMA = 0.02; NGRID = 51; KREL = 2.5
_net = None
def net():
    global _net
    if _net is None: _net = get_net()
    return _net

def dom_kp(th):
    w = th[:, 0].abs(); i = int(torch.argmax(w)); return float(torch.hypot(th[i, 1], th[i, 2]))

def bank_cells(kp):
    ax = np.linspace(-KREL*kp, KREL*kp, NGRID); KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY); sel = (kk > 0.35*kp) & (kk < KREL*kp)
    return KX[sel].astype(float), KY[sel].astype(float), ax

def oracle_gamma_on_bank(theta, kp):
    """Deposit true component energies A^2/2 onto nearest bank cell -> g (per cell)."""
    kxb, kyb, ax = bank_cells(kp)
    step = ax[1]-ax[0]
    kx, ky = theta[:, 1].cpu().numpy(), theta[:, 2].cpu().numpy(); A = theta[:, 0].cpu().numpy()
    g = np.full(len(kxb), 1e-8)
    for i in range(len(kx)):
        d = (kxb - kx[i])**2 + (kyb - ky[i])**2
        g[int(np.argmin(d))] += A[i]**2/2
    return kxb, kyb, g

def mmse_bank(kxb, kyb, g, x, y, t, z, qx, qy, qt, depth):
    """net_gamma_mmse's MMSE with a GIVEN per-cell prior g (variance-scaled)."""
    kxb_t = torch.tensor(kxb, dtype=DT, device=DEVICE); kyb_t = torch.tensor(kyb, dtype=DT, device=DEVICE)
    om = omega_of_k(torch.hypot(kxb_t, kyb_t)+1e-12, depth)
    def des(X, Y, T):
        psi = kxb_t[None]*X[:, None] + kyb_t[None]*Y[:, None] - om[None]*T[:, None]
        return torch.cat([torch.cos(psi), torch.sin(psi)], 1)
    gg = torch.tensor(g/np.sum(g), dtype=DT, device=DEVICE)
    gam = torch.cat([gg, gg]) * torch.var(z)
    Phi = des(x, y, t); s2 = SIGMA**2
    K = (Phi*gam[None])@Phi.T + s2*torch.eye(len(z), dtype=DT, device=DEVICE)
    alpha = torch.linalg.solve(K, z)
    return (des(qx, qy, qt)*gam[None])@(Phi.T@alpha)

def corr(a, b):
    a, b = a.cpu().numpy(), b.cpu().numpy(); m = np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan

def ls_oracle(theta, x, y, t, z, qx, qy, qt, depth):
    kx, ky = theta[:, 1], theta[:, 2]; k = torch.hypot(kx, ky)+1e-12; om = omega_of_k(k, depth)
    def des(X, Y, T):
        psi = kx[None]*X[:, None] + ky[None]*Y[:, None] - om[None]*T[:, None]
        return torch.cat([torch.cos(psi), torch.sin(psi)], 1)
    Phi = des(x, y, t); A = Phi.T@Phi; lam = 1e-4*torch.trace(A)/A.shape[0]
    c = torch.linalg.solve(A+lam*torch.eye(A.shape[0], dtype=DT, device=DEVICE), Phi.T@z)
    return des(qx, qy, qt)@c

def run_ep(ep, t_now, kp, R=6, nobs=250):
    depth = ep['depth']; th0 = ep['theta']; N = th0.shape[0]
    x, y, t = collect_obs(ep, t_now); g = torch.Generator(device=DEVICE).manual_seed(0)
    if x.numel() > nobs:
        idx = torch.randperm(x.numel(), device=DEVICE, generator=g)[:nobs]; x, y, t = x[idx], y[idx], t[idx]
    gx, gy = ep['gx'], ep['gy']; stride = max(1, (gx.numel()*gy.numel())//400)
    Gx, Gy = torch.meshgrid(gx, gy, indexing='xy'); qx, qy = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
    acc = {k: [] for k in ('netmmse', 'oracle_gamma_bank', 'oracle_ls')}
    for _ in range(R):
        thr = th0.clone(); thr[:, 3] = torch.rand(N, dtype=DT, device=DEVICE, generator=g)*2*np.pi
        qt = torch.full_like(qx, float(t_now)); truth = predict(thr, qx, qy, qt, depth)
        s = predict(thr, x, y, t, depth); z = s + SIGMA*torch.randn(s.shape, dtype=DT, device=DEVICE, generator=g)
        P = torch.stack([x, y, t], 1).cpu().numpy(); Q = torch.stack([qx, qy, qt], 1).cpu().numpy()
        pn = np.asarray(net_gamma_mmse(net(), P, z.cpu().numpy(), depth, Q, SIGMA, seed=0, kp_hint=None, use_em=False, smult_fixed=3.25))
        acc['netmmse'].append(corr(torch.tensor(pn, dtype=DT, device=DEVICE), truth))
        kxb, kyb, gg = oracle_gamma_on_bank(thr, kp)
        acc['oracle_gamma_bank'].append(corr(mmse_bank(kxb, kyb, gg, x, y, t, z, qx, qy, qt, depth), truth))
        acc['oracle_ls'].append(corr(ls_oracle(thr, x, y, t, z, qx, qy, qt, depth), truth))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=16); ap.add_argument('--R', type=int, default=6); a = ap.parse_args()
    paths = sorted(glob.glob('data/episode_BF4_15comp_*.h5') + glob.glob('data/episode_BF4_30comp_*.h5'))[:a.n]
    res = []; t0 = time.time()
    for p in paths:
        try:
            ep = load_ep(p); t_now = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
            r = run_ep(ep, t_now, kp, a.R); r['ep'] = os.path.basename(p); res.append(r)
            print(f"  {r['ep']:28s} netMMSE {r['netmmse']:.3f}  oracle-γ-BANK {r['oracle_gamma_bank']:.3f}  oracle-LS {r['oracle_ls']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback; print(f"  {os.path.basename(p)} ERR {ex}"); traceback.print_exc()
    med = lambda k: float(np.nanmedian([r[k] for r in res]))
    print(f"\n=== dense-bank ceiling (n={len(res)}) ===")
    print(f"  net-MMSE (learned γ)        {med('netmmse'):.3f}")
    print(f"  ORACLE-γ on same bank       {med('oracle_gamma_bank'):.3f}   <- ceiling of the dense-bank architecture")
    print(f"  oracle-LS (true sparse)     {med('oracle_ls'):.3f}")
    bank_room = med('oracle_gamma_bank') - med('netmmse')
    bank_ceiling_gap = med('oracle_ls') - med('oracle_gamma_bank')
    print(f"\n  learnable room in bank (oracle-γ − net-MMSE): {bank_room:+.3f}")
    print(f"  dense-bank architectural ceiling gap (oracle-LS − oracle-γ-bank): {bank_ceiling_gap:+.3f}")
    if bank_room >= 0.15 and bank_ceiling_gap < 0.10:
        v = "TRAIN BETTER NET: bank reaches oracle with perfect γ -> a better learned γ can close most of the gap (same architecture)"
    elif bank_ceiling_gap >= 0.15:
        v = "REDESIGN NEEDED: dense-bank plateaus below oracle even with perfect γ -> better net must use off-grid/sparse representation"
    else:
        v = "MIXED: modest learnable room + some architectural ceiling"
    print(f"\n=== VERDICT === {v}")
    json.dump(dict(episodes=res, summary=dict(netmmse=med('netmmse'), oracle_gamma_bank=med('oracle_gamma_bank'),
        oracle_ls=med('oracle_ls'), bank_room=bank_room, bank_ceiling_gap=bank_ceiling_gap, verdict=v)),
        open('results/direction3/bank_ceiling.json', 'w'), indent=1, default=float)
    print(f"saved ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
