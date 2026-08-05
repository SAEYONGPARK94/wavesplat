"""Diagnose the sigma_s discrepancy: fast hos_eval grad vs evaluate_at, and
jonswap-analytic FD vs door1b eta_dr, on one M1 field / one sea."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
import numpy as np
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesim.wave_generator import jonswap_components
from d3_door1b_field_bias import ray_grid, eta as d1b_eta, eta_dr as d1b_eta_dr, R_MIN, R_MAX

HS,TP,DEPTH,SPREAD,WDIR=1.2,8.0,50.0,30.0,180.0
CX,CY=156.0,256.0
o,d,foot=ray_grid(); fr=np.hypot(foot[:,0],foot[:,1])
inb=(fr>R_MIN)&(fr<R_MAX); foot=foot[inb]; fr=fr[inb]
ux,uy=foot[:,0]/fr, foot[:,1]/fr

# ---- HOS M1 field, sample at t=16
sim=HOSSimulator(HOSConfig(Lx=512,Ly=512,Nx=128,Ny=128,depth=DEPTH,M=1))
sim.initialize_jonswap(Hs=HS,Tp=TP,direction_deg=WDIR,direction_spread_deg=SPREAD,seed=0)
while sim.t<16-1e-9: sim.step(min(0.05,16-sim.t))
xs,ys=CX+foot[:,0],CY+foot[:,1]

# (A) evaluate_at
Z,NX,NY,NZ=sim.evaluate_at(xs,ys)
ex_a=-NX/NZ; ey_a=-NY/NZ; s_a=ux*ex_a+uy*ey_a
# (B) fast grad, several thresholds
for rel in (1e-3,1e-5,1e-8,0.0):
    a=sim.eta_hat; m=np.abs(a)>rel*np.abs(a).max()
    kx,ky,av=sim.KX[m],sim.KY[m],a[m]
    ex=np.zeros(len(xs));ey=np.zeros(len(xs))
    B=1000
    for i in range(0,len(kx),B):
        e=np.exp(1j*(np.outer(kx[i:i+B],xs)+np.outer(ky[i:i+B],ys)))
        ex+=np.real(1j*kx[i:i+B,None]*av[i:i+B,None]*e).sum(0)
        ey+=np.real(1j*ky[i:i+B,None]*av[i:i+B,None]*e).sum(0)
    s_b=ux*ex+uy*ey
    print(f"HOS M1  rel={rel:<6}  n_modes={m.sum():6d}  sd(fast)={s_b.std():.5f}  "
          f"sd(eval_at)={s_a.std():.5f}  maxdiff={np.abs(s_b-s_a).max():.2e}")

# ---- jonswap analytic: FD vs eta_dr, and door1b geometry
cs=jonswap_components(HS,TP,n_components=60,depth=DEPTH,direction_deg=WDIR,
                      direction_spread_deg=SPREAD,seed=0)
kx=np.array([c.kx for c in cs]);ky=np.array([c.ky for c in cs])
om=np.array([c.omega for c in cs]);A=np.array([c.A for c in cs]);ph=np.array([c.phi for c in cs])
S=(kx,ky,om,A,ph)
# door1b eta_dr expects sensor-relative (x,y); use foot (sensor origin)
sdr=d1b_eta_dr(foot[:,0],foot[:,1],16.0,S)
# my FD on absolute coords
def eta_lin(x,y,t):
    psi=np.outer(x,kx)+np.outer(y,ky)-t*om[None]+ph
    return (A[None]*np.cos(psi)).sum(1)
FD=0.2
zf=eta_lin(CX+foot[:,0]+FD*ux, CY+foot[:,1]+FD*uy,16.0)
zb=eta_lin(CX+foot[:,0]-FD*ux, CY+foot[:,1]-FD*uy,16.0)
sfd=(zf-zb)/(2*FD)
print(f"\njonswap60  eta_dr sd={sdr.std():.5f}  FD sd={sfd.std():.5f}  "
      f"corr={np.corrcoef(sdr,sfd)[0,1]:.4f}")
# also direction default (door1b uses direction_deg=0)
cs0=jonswap_components(HS,TP,n_components=60,depth=DEPTH,direction_spread_deg=SPREAD,seed=0)
S0=(np.array([c.kx for c in cs0]),np.array([c.ky for c in cs0]),
    np.array([c.omega for c in cs0]),np.array([c.A for c in cs0]),np.array([c.phi for c in cs0]))
sdr0=d1b_eta_dr(foot[:,0],foot[:,1],16.0,S0)
print(f"jonswap60 dir=0 (door1b default): eta_dr sd={sdr0.std():.5f}")
# whole-domain slope sd (direction-agnostic) for reference
gx=np.linspace(0,300,200); X,Y=np.meshgrid(gx,gx)
exg=(-A[None]*kx[None]*np.sin(np.outer(X.ravel(),kx)+np.outer(Y.ravel(),ky)-16*om[None]+ph)).sum(1)
print(f"whole-domain eta_x sd (dir180)={exg.std():.5f}  (|grad| isotropic proxy)")
