"""Confirm the POOLED (ensemble) linear along-ray slope reproduces the frozen
reference: sigma_s ~ 0.0305 and bits {25:.000 50:.0005 100:.052 150:.148 200:.231}
and weighted budget 0.70%. Test dir=180 vs dir=0, jonswap-analytic, 8 seeds."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
import numpy as np
from wavesim.wave_generator import jonswap_components
from d3_door1b_field_bias import ray_grid, eta_dr as d1b_eta_dr, R_MIN, R_MAX

HS,TP,DEPTH,SPREAD=1.2,8.0,50.0,30.0
H=5.0
o,d,foot=ray_grid(); fr0=np.hypot(foot[:,0],foot[:,1])
inb=(fr0>R_MIN)&(fr0<R_MAX); foot=foot[inb]; fr=fr0[inb]
RG=np.linspace(R_MIN,R_MAX,400); DENS=H/(RG**2+H**2); DENS/=np.trapezoid(DENS,RG)
RTAB=(25,50,100,150,200)

def budget(s, sigeta):
    ss=np.sort(s)
    gt=np.arctan(H/np.array(RTAB,float))
    bits=-np.log(np.clip(np.searchsorted(ss,gt,'right')/len(ss),1e-12,1))
    gg=np.arctan(H/RG); pg=np.searchsorted(ss,gg,'right')/len(ss)
    tot=np.trapezoid(-np.log(np.clip(pg,1e-12,1))*DENS,RG)
    amp=0.5*np.log(1+sigeta**2/0.05**2)
    return bits,tot,tot/amp

for wdir in (180.0, 0.0):
    S=[]; SE=[]
    for seed in range(8):
        for t in (16.0,18.0,20.0):
            cs=jonswap_components(HS,TP,n_components=60,depth=DEPTH,direction_deg=wdir,
                                  direction_spread_deg=SPREAD,seed=seed)
            Scmp=(np.array([c.kx for c in cs]),np.array([c.ky for c in cs]),
                  np.array([c.omega for c in cs]),np.array([c.A for c in cs]),
                  np.array([c.phi for c in cs]))
            S.append(d1b_eta_dr(foot[:,0],foot[:,1],t,Scmp))
            # eta at foot for sigma_eta
            kx,ky,om,A,ph=Scmp
            psi=np.outer(foot[:,0],kx)+np.outer(foot[:,1],ky)-t*om[None]+ph
            SE.append((A[None]*np.cos(psi)).sum(1))
    S=np.concatenate(S); sigeta=np.concatenate(SE).std()
    bits,tot,ratio=budget(S,sigeta)
    print(f"dir={wdir:5.0f}  pooled sigma_s={S.std():.4f}  sigma_eta={sigeta:.4f}")
    print(f"   bits: "+" ".join(f"{r}:{b:.4f}" for r,b in zip(RTAB,bits)))
    print(f"   weighted budget = {tot:.4f} nats = {100*ratio:.2f}% of amp  "
          f"(frozen: 0.0127 nats, 0.70%)")
