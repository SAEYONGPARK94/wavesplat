"""Better-net multi-regime eval. Compares a candidate net checkpoint vs prior_net_v3
on SPARSE (15-comp episodes) and BROADBAND (200-comp bwtask) regimes via net-MMSE
achieved reconstruction corr. Prereg tmp/BETTERNET_prereg_2026-07-22.md.
PASS = sparse Δ(cand−v3)>=+0.05 (CI excl 0) AND broadband Δ>=−0.02.
Usage: python3 tmp/betternet_eval.py --ckpt results/direction3/prior_net_compdiverse.pt
Out: results/direction3/betternet_eval.json
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from scripts.d1_crb import DEVICE, DT, load_ep, collect_obs, predict
from d3_prior_net_eval import get_net, net_gamma_mmse

SIGMA = 0.02; V3 = 'results/direction3/prior_net_v3.pt'
def dom_kp(th):
    w = th[:, 0].abs(); i = int(torch.argmax(w)); return float(torch.hypot(th[i, 1], th[i, 2]))
def corr(a, b):
    a, b = np.asarray(a), np.asarray(b); m = np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan

def eval_sparse(net_v3, net_c, n=16, R=6, nobs=250):
    """15-comp episodes, sparse obs, synthetic theta -> net-MMSE corr."""
    paths = sorted(glob.glob('data/episode_BF4_15comp_*.h5'))[:n]
    rv, rc = [], []
    for p in paths:
        ep = load_ep(p); t_now = float(ep['ts'][len(ep['ts'])//2]); kp = dom_kp(ep['theta'])
        th0 = ep['theta']; N = th0.shape[0]; depth = ep['depth']
        x, y, t = collect_obs(ep, t_now); g = torch.Generator(device=DEVICE).manual_seed(0)
        if x.numel() > nobs:
            idx = torch.randperm(x.numel(), device=DEVICE, generator=g)[:nobs]; x, y, t = x[idx], y[idx], t[idx]
        gx, gy = ep['gx'], ep['gy']; stride = max(1, (gx.numel()*gy.numel())//400)
        Gx, Gy = torch.meshgrid(gx, gy, indexing='xy'); qx, qy = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
        cv, cc = [], []
        for _ in range(R):
            thr = th0.clone(); thr[:, 3] = torch.rand(N, dtype=DT, device=DEVICE, generator=g)*2*np.pi
            qt = torch.full_like(qx, float(t_now)); truth = predict(thr, qx, qy, qt, depth).cpu().numpy()
            s = predict(thr, x, y, t, depth); z = (s + SIGMA*torch.randn(s.shape, dtype=DT, device=DEVICE, generator=g)).cpu().numpy()
            P = torch.stack([x, y, t], 1).cpu().numpy(); Q = torch.stack([qx, qy, qt], 1).cpu().numpy()
            cv.append(corr(net_gamma_mmse(net_v3, P, z, depth, Q, SIGMA, seed=0, kp_hint=None, use_em=False, smult_fixed=3.25), truth))
            cc.append(corr(net_gamma_mmse(net_c, P, z, depth, Q, SIGMA, seed=0, kp_hint=None, use_em=False, smult_fixed=3.25), truth))
        rv.append(np.nanmean(cv)); rc.append(np.nanmean(cc))
    return np.array(rv), np.array(rc)

def eval_broadband(net_v3, net_c, nshards=3, per=40):
    """200-comp bwtask held-out scenes -> net-MMSE corr vs gt_q."""
    fs = sorted(glob.glob('tmp/prior_train/bwtask_s*.npz'))[-nshards:]
    rv, rc = [], []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        for i in range(min(per, len(d['P']))):
            P, z, Q, gt = d['P'][i], d['z'][i], d['Q'][i], d['gt_q'][i]; kp = d['meta'][i][0]
            pv = net_gamma_mmse(net_v3, P, z, 50.0, Q, SIGMA, seed=0, kp_hint=kp, use_em=False, smult_fixed=3.25)
            pc = net_gamma_mmse(net_c, P, z, 50.0, Q, SIGMA, seed=0, kp_hint=kp, use_em=False, smult_fixed=3.25)
            rv.append(corr(pv, gt)); rc.append(corr(pc, gt))
    return np.array(rv), np.array(rc)

def ci(d, B=4000):
    d = d[np.isfinite(d)]; rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(d, len(d))) for _ in range(B)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((d > 0).sum()), len(d)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); a = ap.parse_args()
    net_v3 = get_net(V3); net_c = get_net(a.ckpt)
    t0 = time.time()
    sv, sc = eval_sparse(net_v3, net_c)
    bv, bc = eval_broadband(net_v3, net_c)
    print(f"=== Better-net eval ({a.ckpt}) vs prior_net_v3 ===")
    print(f"  SPARSE (15comp, n={len(sv)}):    v3 {np.nanmedian(sv):.3f}   cand {np.nanmedian(sc):.3f}")
    dm, lo, hi, nb, nn = ci(sc - sv)
    print(f"     Δ(cand−v3) {dm:+.3f} CI[{lo:+.3f},{hi:+.3f}] {nb}/{nn}")
    print(f"  BROADBAND (200comp, n={len(bv)}): v3 {np.nanmedian(bv):.3f}   cand {np.nanmedian(bc):.3f}")
    dmb, lob, hib, nbb, nnb = ci(bc - bv)
    print(f"     Δ(cand−v3) {dmb:+.3f} CI[{lob:+.3f},{hib:+.3f}] {nbb}/{nnb}")
    pass_ = dm >= 0.05 and lo > 0 and dmb >= -0.02
    v = ("PASS: better net improves SPARSE without hurting broadband -> comp-diverse training WORKS (realizable demo)"
         if pass_ else "FAIL: no sparse gain (CI) or broadband regression")
    print(f"\n=== VERDICT === {v}  [{time.time()-t0:.0f}s]")
    json.dump(dict(sparse_v3=float(np.nanmedian(sv)), sparse_cand=float(np.nanmedian(sc)),
                   sparse_delta=dm, sparse_ci=[lo, hi], broadband_v3=float(np.nanmedian(bv)),
                   broadband_cand=float(np.nanmedian(bc)), broadband_delta=dmb, broadband_ci=[lob, hib],
                   verdict=v, ckpt=a.ckpt), open('results/direction3/betternet_eval.json', 'w'), indent=1, default=float)

if __name__ == '__main__':
    main()
