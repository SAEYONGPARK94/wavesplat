"""
Simulator-data realism review.
Characterize OUR synthetic episodes vs realistic ocean wind-sea norms and vs
the real Bergamasco/Guimaraes 2020 Black-Sea stereo record.

wave_components columns (verified by dispersion omega=sqrt(g*k)): [A, omega, kx, ky, phi, Q]
"""
import h5py, numpy as np, glob, os, json
G = 9.81
os.chdir('/home/vexit/wavesplat/wavesplat')

def circ_spread_deg(theta_rad, w):
    C = np.sum(w*np.cos(theta_rad))/np.sum(w)
    S = np.sum(w*np.sin(theta_rad))/np.sum(w)
    R = np.hypot(C, S)
    return np.degrees(np.sqrt(max(-2*np.log(max(R,1e-12)),0.0))), np.degrees(np.arctan2(S,C))

def analyze(f):
    with h5py.File(f,'r') as h:
        wc = h['config/wave_components'][:]        # [A, omega, kx, ky, phi, Q]
        A, om, kx, ky = wc[:,0], wc[:,1], wc[:,2], wc[:,3]
        gx = h['frames/gt_grid_x'][:]; gy = h['frames/gt_grid_y'][:]
        fld = h['frames/gt_wave_field'][0]         # first frame
        fullstd = float(h['frames/gt_wave_field'][:].std())
    k = np.hypot(kx, ky); theta = np.arctan2(ky, kx)
    wgt = A**2
    spr, mdir = circ_spread_deg(theta, wgt)
    # peak (max-energy) component
    ip = int(np.argmax(wgt)); kp = k[ip]; omp = om[ip]
    Tp = 2*np.pi/omp if omp>0 else np.nan
    lam_p = 2*np.pi/kp if kp>0 else np.nan
    Hs = 4*fullstd
    eps = Hs*kp/2                                    # characteristic steepness
    Lx = gx.max()-gx.min(); Ly = gy.max()-gy.min()
    ap_x = Lx/lam_p; ap_y = Ly/lam_p                # aperture in peak wavelengths
    # high-frequency (k>2kp) energy fraction from the discrete spectrum
    hf = float(np.sum(wgt[k>2*kp])/np.sum(wgt))
    return dict(N=len(A), Hs=Hs, Tp=Tp, lam_p=lam_p, spread_deg=spr, mean_dir=mdir,
                eps=eps, ap_x=ap_x, ap_y=ap_y, hf_frac=hf)

files = sorted(glob.glob('data/episode_*.h5'))
# sample representative subset: one per (BF, comp) at seed0, both uniform/slope where present
rows = []
seen = set()
for f in files:
    b = os.path.basename(f)
    key = b.replace('_slope','').replace('_uniform','')
    if 'seed0' not in b:
        continue
    try:
        r = analyze(f); r['file'] = b; rows.append(r)
    except Exception as e:
        pass

print(f"{'file':46s} {'N':>3} {'Hs':>5} {'Tp':>5} {'lam':>5} {'spr°':>6} {'mdir°':>6} {'eps':>6} {'apX':>5} {'apY':>5} {'hf%':>5}")
for r in rows:
    print(f"{r['file']:46s} {r['N']:3d} {r['Hs']:5.2f} {r['Tp']:5.1f} {r['lam_p']:5.1f} "
          f"{r['spread_deg']:6.1f} {r['mean_dir']:6.1f} {r['eps']:6.3f} {r['ap_x']:5.1f} {r['ap_y']:5.1f} {r['hf_frac']*100:5.1f}")

# aggregate by sea state
print("\n--- aggregate by sea-state (all sampled seeds/comps) ---")
import collections
agg = collections.defaultdict(list)
for r in rows:
    bf = r['file'].split('_')[1]
    agg[bf].append(r)
summ = {}
for bf, rs in sorted(agg.items()):
    spr = np.mean([r['spread_deg'] for r in rs]); Hs = np.mean([r['Hs'] for r in rs])
    Tp = np.mean([r['Tp'] for r in rs]); eps = np.mean([r['eps'] for r in rs])
    apx = np.mean([r['ap_x'] for r in rs])
    print(f"{bf}: Hs={Hs:.2f} Tp={Tp:.1f} spread={spr:.1f}° eps={eps:.3f} apX={apx:.1f}wl  (n={len(rs)})")
    summ[bf] = dict(Hs=Hs, Tp=Tp, spread_deg=spr, eps=eps, ap_x=apx, n=len(rs))

json.dump({'rows': rows, 'by_sea_state': summ}, open('/home/vexit/wavesplat/tmp/simulator_realism.json','w'), indent=2)
print("\nsaved /home/vexit/wavesplat/tmp/simulator_realism.json")
