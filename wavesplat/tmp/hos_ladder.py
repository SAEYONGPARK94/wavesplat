"""HOS re-validation of ① / better-net (fixes fable's inverse-crime + HOS-directive
violation). On NONLINEAR HOS seas (real z, not predict(theta)+noise; no oracle
wavenumbers), reconstruct with RELAX / net-MMSE(v3) / net-MMSE(better-net cd_task)
and a rich-linear-fit ceiling, corr vs HOS truth field. Q1: is there learnable room
on HOS (ceiling − net-MMSE)? Q2: does better-net beat v3 on HOS?
Usage: python3 tmp/hos_ladder.py [--n 12]
Out: results/direction3/hos_ladder.json
"""
import json, os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from e15_hos_multidraw import (SEAS, build_sea, T_MANUAL_50, N_OBS, LEADS, DEPTH,
                               NOISE, R_MIN, FOV_H, z_at, DEVICE)
from d3_prior_net_eval import get_net, net_gamma_mmse, omega as omega_np

V3 = 'results/direction3/prior_net_v3.pt'
CD = 'results/direction3/prior_net_cd_task.pt'
RING = (0.35, 1.75); DT64 = torch.float64

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
    om = omega_np(np.hypot(kx, ky)+1e-12, DEPTH)
    psi = kx[None]*P[:, 0:1] + ky[None]*P[:, 1:2] - om[None]*P[:, 2:3]
    return np.concatenate([np.cos(psi), np.sin(psi)], 1)

def relax_fit(P, z, Q, kp, K=15):
    kx, ky = kgrid(kp, 25); C = design_np(kx, ky, P); N = len(kx)
    cos_c, sin_c = C[:, :N], C[:, N:]; r = z.copy(); sel = []
    for _ in range(min(K, N)):
        pw = (cos_c.T@r)**2 + (sin_c.T@r)**2
        if sel: pw[sel] = -1
        j = int(np.argmax(pw)); sel.append(j)
        Ps = np.concatenate([cos_c[:, sel], sin_c[:, sel]], 1)
        coef = np.linalg.lstsq(Ps, z, rcond=None)[0]; r = z - Ps@coef
    Cq = design_np(kx[sel], ky[sel], Q); return Cq@coef

def rich_fit(P, z, Q, kp, ridge=1e-2):
    """dense beamform-bank ridge = best achievable linear-atom fit (NOT oracle)."""
    kx, ky = kgrid(kp, 21); Phi = design_np(kx, ky, P); A = Phi.T@Phi
    lam = ridge*np.trace(A)/A.shape[0]
    c = np.linalg.solve(A + lam*np.eye(A.shape[0]), Phi.T@z)
    return design_np(kx, ky, Q)@c

def corr(a, b):
    a, b = np.asarray(a), np.asarray(b); m = np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan

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
    rf = np.sqrt(rng.uniform((15./200.)**2, 1.0, 800))*200.; af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf*np.cos(af), rf*np.sin(af)
    Q, gt = [], []
    for li, l_ in enumerate(LEADS):
        gt.append(z_at(sea['leads'][li], qx, qy)); Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
    return Ps, zs, np.vstack(Q), np.concatenate(gt)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=12); a = ap.parse_args()
    v3, cd = get_net(V3), get_net(CD)
    rows = []; t0 = time.time()
    for si, (Tp, Hs) in enumerate(SEAS[:a.n]):
        try:
            sea = build_sea(Tp, Hs, 0)
            Ps, zs, Q, gt = draw(sea, np.random.default_rng(900000 + 31*si))
            kp = two_stage_kp(Ps, zs)
            r = dict(sea=f'Tp{Tp:g}_Hs{Hs:g}',
                     relax=corr(relax_fit(Ps, zs, Q, kp), gt),
                     rich=corr(rich_fit(Ps, zs, Q, kp), gt),
                     net_v3=corr(np.asarray(net_gamma_mmse(v3, Ps, zs, DEPTH, Q, NOISE, seed=0, kp_hint=None, smult_fixed=3.25)), gt),
                     net_cd=corr(np.asarray(net_gamma_mmse(cd, Ps, zs, DEPTH, Q, NOISE, seed=0, kp_hint=None, smult_fixed=3.25)), gt))
            rows.append(r)
            print(f"  {r['sea']:12s} relax {r['relax']:.3f} net_v3 {r['net_v3']:.3f} net_cd {r['net_cd']:.3f} rich-ceil {r['rich']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback; print(f"  sea {si} ERR {ex}"); traceback.print_exc()
    def med(k): return float(np.nanmedian([r[k] for r in rows]))
    print(f"\n=== HOS ladder (corr vs HOS truth, n={len(rows)} seas — NO inverse crime, real HOS z) ===")
    for k in ['relax', 'net_v3', 'net_cd', 'rich']:
        print(f"  {k:10s} {med(k):.3f}")
    room = med('rich') - med('net_v3'); better = med('net_cd') - med('net_v3')
    print(f"\n  Q1 learnable room on HOS (rich-ceil − net_v3): {room:+.3f}  (synthetic was +0.33)")
    print(f"  Q2 better-net − v3 on HOS: {better:+.3f}")
    print(f"\n  ⟹ HOS verdict: room {'SMALL(model-mismatch dominates=fable confirmed)' if room < 0.1 else 'still present'}; "
          f"better-net {'no gain' if better < 0.02 else 'helps'}")
    json.dump(dict(rows=rows, relax=med('relax'), net_v3=med('net_v3'), net_cd=med('net_cd'),
                   rich=med('rich'), room_hos=room, better_hos=better), open('results/direction3/hos_ladder.json', 'w'), indent=1, default=float)
    print(f"saved ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main()
