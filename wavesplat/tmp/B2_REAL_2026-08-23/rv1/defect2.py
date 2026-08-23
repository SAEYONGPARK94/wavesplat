"""RV-1 청구 3: 12 레코드 원자료 결함 점검 (int16 포화·결측·죽은픽셀·중복프레임). 추정 런 0회."""
import os,sys,numpy as np
ROOT='/home/vexit/wavesplat/wavesplat'; os.chdir(ROOT)
for p in (ROOT,os.path.join(ROOT,'scripts'),os.path.join(ROOT,'tmp','B2_REAL_2026-08-23')): sys.path.insert(0,p)
from scene_station import RECORDS_12
from d3_sa_multirecord import MRecord
from realdata_analysis import _unpack
print(f"{'rec':8s}{'grid':>12s}{'validmask%':>11s}{'robust죽인%':>12s}{'finite%':>9s}"
      f"{'최빈상수%':>10s}{'int16포화%':>11s}{'중복프레임':>10s}{'질의유효%':>10s}")
for record,cfg in RECORDS_12.items():
    import io,contextlib
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec_pre=MRecord(os.path.join(ROOT,cfg['path']), stride=cfg['stride'], robust=False)
        rec=MRecord(os.path.join(ROOT,cfg['path']), stride=cfg['stride'])
    killed=int((rec_pre.valid & ~rec.valid).sum())
    s=cfg['stride']
    idx=np.linspace(0,rec.T_all-1,80).astype(int)
    raw=np.stack([rec.zd[int(i)][::s,::s] for i in idx])
    a=rec.zd.attrs
    fill=a['_FillValue'][0] if '_FillValue' in a else None
    rmin,rmax=int(raw.min()),int(raw.max())
    nonfill=raw[raw!=fill] if fill is not None else raw.ravel()
    sat=float(((nonfill==np.iinfo(np.int16).min)|(nonfill==np.iinfo(np.int16).max)|
               (nonfill==rmin)|(nonfill==rmax)).mean()*100)
    fr=np.stack([_unpack(rec.zd,np.s_[int(i)])[::s,::s] for i in idx])
    fr[:,~rec.valid]=np.nan
    v=fr[np.isfinite(fr)]
    vals,cnt=np.unique(np.round(v,3),return_counts=True)
    top=cnt.max()/v.size*100
    nfw=int(round(40.0/rec.dt)); lm=14.0; nl=int(round(lm/rec.dt))
    if rec.T_all<nfw+nl+4: nl=max(2,(rec.T_all-nfw-4))
    lo,hi=nfw+1,rec.T_all-nl-2; i0=int(np.linspace(lo,hi,6).astype(int)[3])
    Zw=rec.window(i0-nfw+1,nfw)
    dup=sum(1 for j in range(1,len(Zw)) if np.allclose(np.nan_to_num(Zw[j]),np.nan_to_num(Zw[j-1]),atol=1e-9))
    sq=max(1,int(round(4.0/rec.dx)))
    Zf=rec.window(i0,nl+1); g=Zf[0][::sq,::sq]
    print(f"{record:8s}{str(rec.valid.shape):>12s}{float(rec_pre.valid.mean()*100):11.1f}"
          f"{killed/rec.valid.size*100:12.2f}{float(np.isfinite(fr).mean()*100):9.1f}"
          f"{top:10.2f}{sat:11.2f}{dup:10d}{float(np.isfinite(g).mean()*100):10.1f}")
