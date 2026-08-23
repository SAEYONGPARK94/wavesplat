"""RV-1: 제로필드 기준(ZERO_REF=1.0)의 실제 값을 원자료에서 재계산. 추정 런 0회."""
import os, sys, json, numpy as np
ROOT='/home/vexit/wavesplat/wavesplat'
os.chdir(ROOT)
for p in (ROOT, os.path.join(ROOT,'scripts'), os.path.join(ROOT,'tmp','B2_REAL_2026-08-23')):
    sys.path.insert(0,p)
from scene_station import RECORDS_12
from d3_sa_multirecord import MRecord
from realdata_analysis import directional_spread

out={}
for record,cfg in RECORDS_12.items():
    rec=MRecord(os.path.join(ROOT,cfg['path']), stride=cfg['stride'])
    W_max=40.0; nfw=int(round(W_max/rec.dt)); lm=14.0
    n_lead=int(round(lm/rec.dt))
    if rec.T_all < nfw+n_lead+4:
        n_lead=max(2,(rec.T_all-nfw-4)); lm=n_lead*rec.dt
    lo,hi=nfw+1, rec.T_all-n_lead-2
    i0=int(np.linspace(lo,hi,6).astype(int)[3])
    Zmem=rec.window(i0-nfw+1,nfw); Zfut=rec.window(i0,n_lead+1)
    sg=float(np.nanstd(Zmem))
    sq=max(1,int(round(4.0/rec.dx)))
    leads=np.arange(0.0,lm+1e-9,1.0)
    maps_gt=np.array([Zfut[int(round(l_/rec.dt))][::sq,::sq] for l_ in leads])
    g=np.asarray(maps_gt[0],float); mm=np.isfinite(g)
    zf_now=float(np.sqrt(np.mean(g[mm]**2))/sg)
    zf_curve=[]
    for li,l_ in enumerate(leads):
        gg=np.asarray(maps_gt[li],float); m2=np.isfinite(gg)
        zf_curve.append(float(np.sqrt(np.mean(gg[m2]**2))/sg) if m2.any() else float('nan'))
    # mean-removed variant
    gm=float(np.nanmean(g[mm]))
    _,lam_p,_,kp=directional_spread(np.nan_to_num(Zmem[-1]),rec.dx,rec.dy)
    out[record]=dict(sg=sg, zf_nowcast=zf_now, zf_leadmean=float(np.nanmean(zf_curve)),
                     npx=int(g.size), n_valid=int(mm.sum()),
                     mean_g=gm, std_g=float(np.std(g[mm])),
                     std_fut_over_sg=float(np.std(g[mm])/sg),
                     nfw=nfw, n_lead=n_lead, lm=float(lm), T_all=int(rec.T_all),
                     dt=float(rec.dt), dx=float(rec.dx), i0=i0,
                     box_x=float(rec.gx.max()-rec.gx.min()), box_y=float(rec.gy.max()-rec.gy.min()),
                     lam_p=float(2*np.pi/kp))
    print(record, json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in out[record].items()}))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'zf.json'),'w'), indent=1)
