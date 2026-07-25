"""Decisive: on cross-Tp scenes, add the ORACLE-gamma arm (true spectrum) alongside
hand/v3/retrained. If oracle-gamma ALSO ties hand -> genuinely no gap (A). If oracle-gamma
BEATS hand but retrained doesn't -> there IS a gap, the net failed to capture it (B).
Also inspect whether the retrained net's predicted gamma is bimodal on a bimodal input.
"""
import numpy as np, torch
import d3_compare_gifs as cg
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse

G, DEPTH, SCAN_DT, M_PER_SCAN, NOISE = cg.G, cg.DEPTH, cg.SCAN_DT, cg.M_PER_SCAN, cg.NOISE


def bimodal_scene(Hs, Tp, seed, U=3.0, W=20.0):
    rng = np.random.default_rng(50000 + seed)
    r2 = rng.uniform(0.2, 0.5)
    tp2 = float(np.clip(Tp * rng.uniform(0.6, 0.85), 4.0, 12.0))
    doff = rng.uniform(40, 140)
    kp = (2 * np.pi / Tp) ** 2 / G
    c1 = cg.jonswap_components(Hs=Hs*np.sqrt(1-r2), Tp=Tp, n_components=120, depth=DEPTH,
                              direction_deg=0.0, direction_spread_deg=25.0, seed=50000+seed)
    c2 = cg.jonswap_components(Hs=Hs*np.sqrt(r2), Tp=tp2, n_components=60, depth=DEPTH,
                              direction_deg=doff, direction_spread_deg=25.0, seed=57000+seed)
    comps = list(c1)+list(c2)
    kx=np.array([c.kx for c in comps]); ky=np.array([c.ky for c in comps])
    A=np.array([c.A for c in comps]); ph=np.array([c.phi for c in comps])
    om=cg.omega_of(kx,ky,DEPTH)
    def eta(x,y,t):
        ps=(kx[None]*np.atleast_1d(x)[:,None]+ky[None]*np.atleast_1d(y)[:,None]
            -om[None]*np.atleast_1d(t)[:,None]+ph[None]); return (A[None]*np.cos(ps)).sum(1)
    t0,cx,cy=W,0.0,0.0
    def ship(t): return np.array([cx-U*(t-t0),cy])
    Pa,za=[],[]
    for ts in np.arange(t0-W,t0+1e-9,SCAN_DT):
        sp=ship(ts); rr=rng.uniform(20,100,M_PER_SCAN); aa=rng.uniform(-np.pi/3,np.pi/3,M_PER_SCAN)
        xs=sp[0]-rr*np.cos(aa); ys=sp[1]+rr*np.sin(aa)
        Pa.append(np.column_stack([xs,ys,np.full(M_PER_SCAN,ts)])); za.append(eta(xs,ys,ts)+rng.normal(0,NOISE,M_PER_SCAN))
    P,z=np.vstack(Pa),np.concatenate(za)
    gxq=np.arange(ship(t0)[0]-30,ship(t0)[0]+110,4.0); gyq=np.arange(cy-90,cy+90,4.0)
    Xq,Yq=np.meshgrid(gxq,gyq); npx=Xq.size
    Pq=np.column_stack([Xq.ravel(),Yq.ravel(),np.full(npx,t0)])
    gt=eta(Xq.ravel(),Yq.ravel(),t0).reshape(Xq.shape)
    Ctrue=np.column_stack([kx,ky,A])
    return dict(P=P,z=z,Pq=Pq,Xq=Xq,depth=DEPTH,Hs=Hs,kp=kp,seed=seed,gt=gt,Ctrue=Ctrue,r2=r2,tp2=tp2)


def nr(fq,sc):
    m=fq.reshape(sc['Xq'].shape); return float(np.sqrt(np.mean((m-sc['gt'])**2))/np.std(sc['gt']))


nv=get_net('results/direction3/prior_net_v3.pt')
nr40=get_net('results/direction3/prior_net_multimodal_big.pt')
H,V,R,O=[],[],[],[]
for s in range(12):
    sc=bimodal_scene(2.0,8.0,s); s0=0.05*sc['Hs']; P,z,Pq=sc['P'],sc['z'],sc['Pq']
    H.append(nr(mmse_bank_estimate(P,z,sc['depth'],Pq,sigma0=s0,kp_hint=sc['kp'],seed=s),sc))
    V.append(nr(net_gamma_mmse(nv,P,z,sc['depth'],Pq,sigma0=s0,seed=s,kp_hint=sc['kp']),sc))
    R.append(nr(net_gamma_mmse(nr40,P,z,sc['depth'],Pq,sigma0=s0,seed=s,kp_hint=sc['kp']),sc))
    O.append(nr(net_gamma_mmse(nr40,P,z,sc['depth'],Pq,sigma0=s0,seed=s,kp_hint=sc['kp'],oracle_components=sc['Ctrue']),sc))
    print(f"[{s:2d}] hand {H[-1]:.3f} v3 {V[-1]:.3f} rt {R[-1]:.3f} ORACLE {O[-1]:.3f}  (r2={sc['r2']:.2f} tp2={sc['tp2']:.1f})",flush=True)
H,V,R,O=map(np.array,(H,V,R,O))
print(f"\nMEANS: hand {H.mean():.3f}  v3 {V.mean():.3f}  retrained {R.mean():.3f}  ORACLE-gamma {O.mean():.3f}")
print(f"oracle vs hand: {(H-O).mean():+.3f}  (oracle better on {int((O<H).sum())}/12)")
if (H-O).mean() <= 0.02:
    print("=> (A) oracle-gamma ALSO ties hand on cross-Tp => GENUINELY NO GAP => thesis dead")
else:
    print(f"=> oracle-gamma BEATS hand by {(H-O).mean():+.3f} => there IS a gap; retrained didn't capture it => (B) net-learning failure, potentially fixable => thesis REVIVABLE")
