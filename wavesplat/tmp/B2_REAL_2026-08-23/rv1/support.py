"""RV-1 청구 1: 팬 제거가 '미관측 질의 격자' 종점을 유지하는가.
질의점 -> 최근접 관측점 거리(공간)를 두 프로토콜에서 비교. 추정 런 0회."""
import os,sys,io,contextlib,numpy as np
ROOT='/home/vexit/wavesplat/wavesplat'; os.chdir(ROOT)
for p in (ROOT,os.path.join(ROOT,'scripts'),os.path.join(ROOT,'tmp','B2_REAL_2026-08-23')): sys.path.insert(0,p)
from scene_station import scene_station, RECORDS_12
buf=io.StringIO()
print('=== 팬없는 고정관측소(B-2) ===')
print(f"{'rec':8s}{'npx':>6s}{'nobs':>6s}{'d_med[m]':>10s}{'d_max[m]':>10s}{'lam_p':>8s}{'d_max/lam':>10s}{'질의점당obs<2m':>14s}")
st={}
for r in RECORDS_12:
    with contextlib.redirect_stdout(buf):
        sc=scene_station(r,win=3)
    P=np.asarray(sc['P']); Q=np.column_stack([sc['Xq'].ravel(),sc['Yq'].ravel()])
    D=np.hypot(Q[:,None,0]-P[None,:,0],Q[:,None,1]-P[None,:,1])
    dmin=D.min(axis=1); lam=2*np.pi/sc['kp']
    st[r]=(float(np.median(dmin)),float(dmin.max()),float(lam))
    print(f"{r:8s}{sc['npx']:6d}{len(P):6d}{np.median(dmin):10.3f}{dmin.max():10.3f}{lam:8.2f}"
          f"{dmin.max()/lam:10.3f}{float((D<2.0).sum(axis=1).mean()):14.1f}")
print()
print('=== 원 팬 프로토콜(d3_compare_gifs.scene_real) — 같은 레코드 5개 ===')
sys.path.insert(0,os.path.join(ROOT,'scripts'))
from d3_compare_gifs import scene_real
print(f"{'rec':8s}{'npx':>6s}{'nobs':>6s}{'d_med[m]':>10s}{'d_max[m]':>10s}{'lam_p':>8s}{'d_max/lam':>10s}{'질의점당obs<2m':>14s}")
for r in ['YS17','AA15a','AA14','AA15b','LJ18g']:
    try:
        with contextlib.redirect_stdout(buf):
            sc=scene_real(record=r,win=3)
    except Exception as e:
        print(f'{r:8s} 실패 {e!r}'); continue
    P=np.asarray(sc['P'])
    Xq,Yq=np.meshgrid(sc['gxq'],sc['gyq'])          # scene_real 은 Yq 를 반환하지 않는다
    Q=np.column_stack([Xq.ravel(),Yq.ravel()])
    D=np.hypot(Q[:,None,0]-P[None,:,0],Q[:,None,1]-P[None,:,1])
    dmin=D.min(axis=1); lam=2*np.pi/sc['kp']
    print(f"{r:8s}{sc['npx']:6d}{len(P):6d}{np.median(dmin):10.3f}{dmin.max():10.3f}{lam:8.2f}"
          f"{dmin.max()/lam:10.3f}{float((D<2.0).sum(axis=1).mean()):14.1f}")
