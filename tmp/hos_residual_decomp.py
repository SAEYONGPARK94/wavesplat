"""
Decompose OUR estimator's residual on an HOS(M=3) field:
  Q1 (energy): does Hs/variance match even when shape doesn't?
  Q2 (fine waves): is the residual dominated by high-k bound harmonics
      ("자잘한 파"), or by in-band reconstruction errors on the resolvable waves?
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
os.chdir('/home/vexit/wavesplat/wavesplat')
sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d23_main_results import panel
from d26_relax_iterative import iterative_relax
from d40_learned_acquisition import pool_xy

G = 9.81; DEPTH = 50.0; DT = 0.1

def radial_power(field, dx, dy):
    """2D power spectrum + radial |k| coordinate (rad/m)."""
    F = np.fft.fftshift(np.fft.fft2(field - field.mean()))
    P = (np.abs(F)**2)
    ny, nx = field.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2*np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    return P, K

def band_energy(field, dx, dy, k_split):
    P, K = radial_power(field, dx, dy)
    tot = P.sum()
    inband = P[K <= k_split].sum()
    high  = P[K >  k_split].sum()
    return tot, inband/tot, high/tot

def run(Hs, Tp, label):
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    dx = gx[1]-gx[0]; dy = gy[1]-gy[0]
    kp = (2*np.pi/Tp)**2 / G                    # deep-water peak wavenumber
    k_split = 1.5*kp                             # separate fundamental band from 2nd-harmonic (bound) band
    pool = pool_xy()
    cfg = HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3)
    sim = HOSSimulator(cfg); sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=0)
    n_back = int(round(2.2*Tp/0.1)); Xg, Yg = np.meshgrid(gx, gy)
    frames = []
    for i in range(n_back+4):
        if i > 0: sim.step(DT)
        Zg, *_ = sim.evaluate_at(Xg, Yg)
        oi = np.random.default_rng(1000+i).choice(len(pool), 60, replace=False)
        ox, oy = pool[oi,0], pool[oi,1]
        Zo, *_ = sim.evaluate_at(ox.reshape(1,-1), oy.reshape(1,-1))
        frames.append((i*DT, Zg.astype(np.float32), ox, oy,
                       Zo.ravel()+np.random.default_rng(i).normal(0,0.02,len(ox))))
    i_star = len(frames)-2
    t, gt, *_ = frames[i_star]
    Hs_eff = float(gt.std()*4)
    P = np.vstack([np.column_stack([f[2], f[3], np.full_like(f[2], f[0])]) for f in frames[:i_star+1]])
    z = np.concatenate([f[4] for f in frames[:i_star+1]]).astype(np.float32)
    target = {'gx': gx, 'gy': gy, 't': t, 'gt': gt, 'Hs': Hs_eff}
    est = iterative_relax(P.astype(np.float32), z, DEPTH, target, 3)[0]
    est = np.asarray(est, np.float32)

    resid = est - gt
    nrmse = panel(est, gt, Hs_eff)['nrmse']
    # energy match
    Hs_est = float(est.std()*4)
    var_ratio = float(est.var()/gt.var())
    corr = float(np.corrcoef(est.ravel(), gt.ravel())[0,1])
    # spectral bands
    _, gt_in, gt_hi = band_energy(gt, dx, dy, k_split)
    _, es_in, es_hi = band_energy(est, dx, dy, k_split)
    # residual band split (how much of the ERROR is fine/high-k vs in-band)
    Pr, K = radial_power(resid, dx, dy)
    r_tot = Pr.sum(); r_in = Pr[K<=k_split].sum()/r_tot; r_hi = Pr[K>k_split].sum()/r_tot

    steep = Hs*kp/2
    print(f"\n===== {label}: Hs={Hs} Tp={Tp}  (ε≈{steep:.3f}, kp={kp:.3f}, k_split={k_split:.3f}) =====")
    print(f"  nRMSE (est vs HOS-GT)      : {nrmse:.3f}")
    print(f"  --- ENERGY (Q1) ---")
    print(f"  Hs   GT={Hs_eff:.3f}  est={Hs_est:.3f}   -> Hs err {abs(Hs_est-Hs_eff)/Hs_eff*100:5.1f}%")
    print(f"  variance ratio est/GT      : {var_ratio:.3f}   correlation: {corr:.3f}")
    print(f"  --- WHERE IS THE ENERGY (Q2) ---")
    print(f"  GT  energy: in-band={gt_in*100:5.1f}%   high-k(bound/fine)={gt_hi*100:5.1f}%")
    print(f"  est energy: in-band={es_in*100:5.1f}%   high-k           ={es_hi*100:5.1f}%")
    print(f"  --- RESIDUAL (est-GT) energy split ---")
    print(f"  residual  : in-band={r_in*100:5.1f}%   high-k(fine)     ={r_hi*100:5.1f}%")
    return dict(label=label, nrmse=nrmse, Hs_eff=Hs_eff, Hs_est=Hs_est,
                var_ratio=var_ratio, corr=corr, gt_hi=gt_hi, resid_hi=r_hi, resid_in=r_in)

if __name__ == '__main__':
    for Hs, Tp, lab in [(2.0, 8.0, 'moderate'), (3.0, 7.0, 'steep'), (3.5, 6.0, 'very_steep')]:
        try:
            run(Hs, Tp, lab)
        except Exception as e:
            import traceback; traceback.print_exc()
