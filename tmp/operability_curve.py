"""Step 3: forecast/extrapolation operability curve across real records.
Per record: aperture-per-wavelength (dispersion-based), directional spread, forecast skill vs persistence,
extrapolation vs naive. -> operability: aperture-ratio vs forecast skill (physics value where naive/persist can't)."""
import os, sys, numpy as np
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from scripts.realdata_analysis import load_surface_nc, directional_spread, peak_period
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
G=9.81
RECS=[('BS_2011','data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc',30.0),
      ('BS_2013','data/real_bergamasco/BS_2013/Surfaces_20130922_130001_short.nc',30.0),
      ('AA_2014','data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc',17.0),
      ('YS_2017','data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc',50.0),
      ('LJ_2018','data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc',40.0)]
LEADS=[0.5,1.0,2.0,3.0]
def disp_wavelength(Tp,depth):
    om=2*np.pi/Tp; k=om**2/G
    for _ in range(50): k=om**2/(G*np.tanh(k*depth))
    return 2*np.pi/k
def analyze(name,path,depth):
    Z,gx,gy,t,meta=load_surface_nc(path,tmax_frames=700)
    T,Ny,Nx=Z.shape; dt=float(np.median(np.diff(t))); dx=meta['dx']; dy=meta['dy']
    Xg,Yg=np.meshgrid(gx,gy); rng=np.random.default_rng(0)
    Tp,_=peak_period(Z,t); Tp=max(Tp,2.0)
    lam=disp_wavelength(Tp,depth); Lx=gx.max()-gx.min(); Ly=gy.max()-gy.min(); apr=min(Lx,Ly)/lam
    ic=int(np.argmin(np.mean(np.isnan(Z.reshape(T,-1)),axis=1))); spr=directional_spread(Z[ic],dx,dy)[0]; Hs=4*np.nanstd(Z[ic])
    n_back=min(int(4*Tp/dt),T-int(max(LEADS)/dt)-2); n_back=max(8,n_back)
    frames=np.linspace(n_back+1,T-int(max(LEADS)/dt)-2,3).astype(int)
    fc={dl:[] for dl in LEADS}; pers={dl:[] for dl in LEADS}; ex_w=[]; ex_n=[]
    for i_star in frames:
        gt_now=Z[i_star]; Hsw=4*np.nanstd(gt_now); Ps,zs=[],[]
        for i in range(i_star-n_back,i_star+1):
            fr=Z[i]; vm=~np.isnan(fr); idx=np.flatnonzero(vm.ravel())
            if len(idx)==0: continue
            pick=rng.choice(idx,min(60,len(idx)),replace=False)
            Ps.append(np.column_stack([Xg.ravel()[pick],Yg.ravel()[pick],np.full(len(pick),i*dt)])); zs.append(fr.ravel()[pick])
        P=np.vstack(Ps).astype(np.float32); z=np.concatenate(zs).astype(np.float32)
        target={'gx':gx,'gy':gy,'t':float(i_star*dt),'gt':np.nan_to_num(gt_now),'Hs':Hsw}
        klim=estimate_klim(P,z,depth); kx,ky=relax_cv(P,z,depth,klim); kx,ky,_=refine_params(kx,ky,P,z,depth,target,iters=400)
        A,phi=fit_amplitudes_phases(kx,ky,np.column_stack([P,z]).astype(np.float32),depth,lambda_reg=0.5)
        for dl in LEADS:
            j=i_star+int(round(dl/dt))
            if j>=T: continue
            gt=Z[j]; v=~np.isnan(gt); pred=render_wave_field(kx,ky,A,phi,depth,gx,gy,float(j*dt))
            fc[dl].append(np.sqrt(np.mean((pred[v]-gt[v])**2))/Hsw); pers[dl].append(np.sqrt(np.mean((gt_now[v]-gt[v])**2))/Hsw)
        # extrapolation outside hull
        try:
            hull=Delaunay(Ps[-1][:,:2]); out=(hull.find_simplex(np.column_stack([Xg.ravel(),Yg.ravel()]))<0).reshape(gt_now.shape)&(~np.isnan(gt_now))
            if out.sum()>50:
                ws=render_wave_field(kx,ky,A,phi,depth,gx,gy,float(i_star*dt))
                nv=griddata(Ps[-1][:,:2],z[-len(zs[-1]):],(Xg,Yg),method='linear'); nv=np.nan_to_num(nv,nan=np.nanmean(z))
                ex_w.append(np.sqrt(np.mean((ws[out]-gt_now[out])**2))/Hsw); ex_n.append(np.sqrt(np.mean((nv[out]-gt_now[out])**2))/Hsw)
        except Exception: pass
    skill={dl:(1-(np.mean(fc[dl])/np.mean(pers[dl]))**2 if fc[dl] and np.mean(pers[dl])>0 else np.nan) for dl in LEADS}
    best=max([s for s in skill.values() if not np.isnan(s)],default=np.nan)
    exm=(np.mean(ex_n)-np.mean(ex_w)) if ex_w else np.nan
    print(f"{name:8s} patch{min(Lx,Ly):.0f}m Tp{Tp:.1f} lam{lam:.0f}m aper{apr:.2f}λ spr{spr:.0f}° | "
          f"fc-skill "+" ".join(f"{dl}s:{skill[dl]:+.2f}" for dl in LEADS)+f" | best{best:+.2f} extrap+{exm:+.3f}",flush=True)
    return dict(name=name,aperture_ratio=float(apr),spread=float(spr),Tp=float(Tp),best_skill=float(best),extrap_margin=float(exm))
print("=== forecast/extrapolation operability across real records ===")
out=[]
for name,path,depth in RECS:
    try: out.append(analyze(name,path,depth))
    except Exception as e: print(f"{name}: FAIL {str(e)[:60]}",flush=True)
import json; json.dump(out,open('results/operability_curve.json','w'),indent=2)
print("\naperture-ratio vs best forecast-skill:")
for r in sorted(out,key=lambda x:x['aperture_ratio']): print(f"  aper {r['aperture_ratio']:.2f}λ  spr {r['spread']:.0f}°  -> best skill {r['best_skill']:+.2f}  extrap +{r['extrap_margin']:+.3f}  ({r['name']})")
print("saved results/operability_curve.json")
