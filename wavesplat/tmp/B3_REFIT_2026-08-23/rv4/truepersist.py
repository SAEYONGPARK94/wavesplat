"""RV-4 원자료 진단 — 추정기 호출 0회. scene_station 만 부른다."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
ROOT='/home/vexit/wavesplat/wavesplat'
for p in (ROOT, ROOT+'/scripts', ROOT+'/tmp/B2_REAL_2026-08-23', ROOT+'/tmp/B3_REFIT_2026-08-23'):
    sys.path.insert(0,p)
os.chdir(ROOT)
from scene_station import scene_station
from gates_b3 import RECORDS, WINDOWS, NOISE

out={}
for rec in RECORDS:
    for win in WINDOWS:
        sc=scene_station(rec, win=win, noise=NOISE, with_future_obs=True)
        sg=float(sc['sg']); leads=list(map(float,sc['leads']))
        g=np.asarray(sc['maps_gt'],float)
        g0=g[0]; m0=np.isfinite(g0)
        gxq=np.asarray(sc['gxq']); gyq=np.asarray(sc['gyq'])
        Xq,Yq=np.meshgrid(gxq,gyq)
        # 관측 지속: t0 프레임 근처 관측으로 최근접(NN) 보간
        P=np.asarray(sc['P'],float); z=np.asarray(sc['z'],float)
        t0=float(sc['t0'])
        d=dict(sg=sg, npx=int(sc['npx']), nx=len(gxq), ny=len(gyq),
               box_x=float(gxq.max()-gxq.min()), box_y=float(gyq.max()-gyq.min()),
               kp=float(sc['kp']), Hs=float(sc['Hs']), depth=float(sc['depth']),
               n_obs_mem=int(len(z)))
        for wsec,tag in ((1.0,'obs1'),(2.0,'obs2'),(4.0,'obs4')):
            sel=P[:,2] >= t0-wsec-1e-9
            d['n_'+tag]=int(sel.sum())
        res={}
        for wsec,tag in ((1.0,'obs1'),(2.0,'obs2'),(4.0,'obs4')):
            sel=P[:,2] >= t0-wsec-1e-9
            if sel.sum()==0: continue
            Ps,zs=P[sel],z[sel]
            dd=(Xq.ravel()[:,None]-Ps[None,:,0])**2+(Yq.ravel()[:,None]-Ps[None,:,1])**2
            nn=zs[np.argmin(dd,axis=1)].reshape(Xq.shape)
            res[tag]=nn
        per={}
        for li,l_ in enumerate(leads):
            gl=g[li]; mm=np.isfinite(gl)
            zero=float(np.sqrt(np.mean(gl[mm]**2))/sg)
            mb=mm & m0
            tp=float(np.sqrt(np.mean((g0[mb]-gl[mb])**2))/sg) if mb.sum() else None
            row=dict(zero=zero, tp_oracle=tp, cov=float(mb.sum())/max(1,mm.sum()),
                     sd_gt=float(np.std(gl[mm]))/sg, mean_gt=float(np.mean(gl[mm]))/sg,
                     npts=int(mm.sum()))
            if mb.sum()>3:
                c=np.corrcoef(g0[mb],gl[mb])[0,1]
                row['corr_g0_gl']=float(c)
            for tag,nn in res.items():
                row['tp_'+tag]=float(np.sqrt(np.mean((nn[mm]-gl[mm])**2))/sg)
            per[str(l_)]=row
        d['per_lead']=per
        out[f'{rec}|{win}']=d
        print(rec,win,'npx',sc['npx'],'box %.1fx%.1f'%(d['box_x'],d['box_y']),
              'sg %.3f'%sg,'nobs1',d['n_obs1'],'tp1 %.3f'%per['1.0']['tp_oracle'],
              'tp14 %.3f'%per['14.0']['tp_oracle'], flush=True)
json.dump(out, open(ROOT+'/tmp/B3_REFIT_2026-08-23/rv4/true_persist.json','w'), indent=1)
print('WROTE')
