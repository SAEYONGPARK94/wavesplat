"""RV-1: (a) Zmem/Zfut 가 서로 다른 창으로 demean 된다 -> 채점 GT 에 계통 편향.
(b) noise 를 바꿔도 rng 추출 위치가 그대로인가. 추정 런 0회."""
import os, sys, json, numpy as np
ROOT='/home/vexit/wavesplat/wavesplat'; os.chdir(ROOT)
for p in (ROOT, os.path.join(ROOT,'scripts'), os.path.join(ROOT,'tmp','B2_REAL_2026-08-23')):
    sys.path.insert(0,p)
from scene_station import RECORDS_12, scene_station
from d3_sa_multirecord import MRecord

print('=== (a) demean 창 불일치가 만드는 계통 편향 (질의격자, lead 0) ===')
print(f"{'rec':8s}{'sg':>8s}{'|B|rms':>9s}{'B/sg':>8s}{'ZF':>8s}{'ZF_nobias':>10s}")
for record,cfg in RECORDS_12.items():
    rec=MRecord(os.path.join(ROOT,cfg['path']), stride=cfg['stride'])
    nfw=int(round(40.0/rec.dt)); lm=14.0; n_lead=int(round(lm/rec.dt))
    if rec.T_all < nfw+n_lead+4: n_lead=max(2,(rec.T_all-nfw-4)); lm=n_lead*rec.dt
    lo,hi=nfw+1, rec.T_all-n_lead-2; i0=int(np.linspace(lo,hi,6).astype(int)[3])
    Zmem=rec.window(i0-nfw+1,nfw); Zfut=rec.window(i0,n_lead+1)
    sg=float(np.nanstd(Zmem))
    sq=max(1,int(round(4.0/rec.dx)))
    # 같은 물리 프레임 i0 을 두 규약으로 본 차이
    B = Zmem[-1][::sq,::sq] - Zfut[0][::sq,::sq]
    m=np.isfinite(B)
    brms=float(np.sqrt(np.nanmean(B[m]**2)))
    g=Zfut[0][::sq,::sq]; mm=np.isfinite(g)
    zf=float(np.sqrt(np.mean(g[mm]**2))/sg)
    g2=Zmem[-1][::sq,::sq]; m2=np.isfinite(g2)
    zf2=float(np.sqrt(np.mean(g2[m2]**2))/sg)
    print(f'{record:8s}{sg:8.4f}{brms:9.4f}{brms/sg:8.3f}{zf:8.4f}{zf2:10.4f}')

print()
print('=== (b) noise 를 바꾸면 관측 위치가 바뀌는가 ===')
for record in ['BS13c','AA14','LJ18a']:
    a=scene_station(record,win=3,noise=0.02); b=scene_station(record,win=3,noise=0.005); c=scene_station(record,win=3,noise=0.0)
    same_ab=np.array_equal(a['P'],b['P']); same_ac=np.array_equal(a['P'],c['P'])
    print(f"{record:8s} P 동일 0.02vs0.005={same_ab} 0.02vs0.0={same_ac} "
          f"sg 동일={a['sg']==b['sg']==c['sg']} kp 동일={a['kp']==b['kp']==c['kp']} "
          f"maps_gt 동일={np.array_equal(np.nan_to_num(a['maps_gt']),np.nan_to_num(c['maps_gt']))}")
    dz=a['z']-c['z']
    print(f"          실제 주입 잡음 rms(0.02)={float(np.std(dz)):.4f}  z(0.005)-z(0)rms={float(np.std(b['z']-c['z'])):.4f}")
