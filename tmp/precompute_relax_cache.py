"""Precompute RELAX outputs (components) for N episodes -> cache for distillation training.
Target for distillation = RELAX-rendered field (structure-preserving, deterministic fn of obs)."""
import os, sys, numpy as np, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60)
N=300; pool=pool_xy(); rows=[]; t0=time.time()
for i in range(N):
    rng=np.random.default_rng(10000+i)
    Hs=rng.uniform(0.6,2.0); Tp=rng.uniform(5.5,9.5)
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=int(rng.integers(1e9)))
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp; oi=rng.integers(0,len(pool),200); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,200)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,200)
    P=np.column_stack([ox,oy,ot]).astype(np.float32); z=oz.astype(np.float32)
    # RELAX
    target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':np.zeros((60,80),np.float32),'Hs':Hs}
    try:
        klim=estimate_klim(P,z,DEPTH); rkx,rky=relax_cv(P,z,DEPTH,klim)
        rkx,rky,_=refine_params(rkx,rky,P,z,DEPTH,target,iters=400)
        rA,rphi=fit_amplitudes_phases(rkx,rky,np.column_stack([P,z]).astype(np.float32),DEPTH,lambda_reg=0.5)
    except Exception as e:
        print(f"  ep{i} RELAX fail {str(e)[:40]}",flush=True); continue
    rows.append(dict(P=P,z=z,Tobs=float(Tobs),kx=rkx.astype(np.float32),ky=rky.astype(np.float32),
                     A=rA.astype(np.float32),phi=rphi.astype(np.float32),Hs=float(Hs)))
    if i%50==0: print(f"  ep {i}/{N}  N_comp={len(rkx)}  ({time.time()-t0:.0f}s)",flush=True)
np.savez('results/direction3/_relax_distill_cache.npz', rows=np.array(rows,dtype=object))
print(f"saved {len(rows)} episodes -> _relax_distill_cache.npz ({time.time()-t0:.0f}s)")
