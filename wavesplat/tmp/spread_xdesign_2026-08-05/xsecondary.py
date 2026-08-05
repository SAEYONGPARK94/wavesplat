"""사전등록 §7 SECONDARY 패널의 상호작용 대비. 탐색 아님 — 사전 지정된 지표들이다."""
import json, os
from math import comb, erfc
import numpy as np
HERE='tmp/spread_xdesign_2026-08-05'
C=json.load(open(os.path.join(HERE,'xdesign.json')))['cells']
def st(w,n):
    k=max(w,n-w); return min(1.0,2*sum(comb(n,i) for i in range(k,n+1))/2**n)
def wil(d):
    d=np.asarray([x for x in d if x!=0],float); n=len(d)
    if n==0: return None
    r=np.argsort(np.argsort(np.abs(d)))+1.0
    w=min(r[d>0].sum(), r[d<0].sum()); sd=np.sqrt(n*(n+1)*(2*n+1)/24)
    return float(erfc(abs((w-n*(n+1)/4)/sd)/np.sqrt(2))) if sd else None
def v(arm,nc,s,m,k):
    c=C.get(f'{arm}|{nc}comp|seed{s}')
    x=c[m].get(k)
    return x['median'] if isinstance(x,dict) else x
# 열화 방향: nRMSE·Hs_error 는 클수록 나쁨(+), SS·DirR2·rho 는 작을수록 나쁨(부호 반전)
KEYS={'nRMSE':+1,'Skill_Score':-1,'Directional_R2':-1,'Pearson_rho':-1,'Variance_Ratio':None}
print(f"{'metric':16s} {'중앙 ΔWS':>10s} {'중앙 ΔLK':>10s} {'중앙 I':>10s} {'I>0':>6s} {'sign p':>8s} {'wilcox p':>9s}")
out={}
for k,sgn in KEYS.items():
    I=[];dw=[];dl=[]
    for nc in (5,30):
        for s in range(5):
            a,b=v('ctrl30',nc,s,'WS',k), v('treat60',nc,s,'WS',k)
            p,q=v('ctrl30',nc,s,'LETKF',k), v('treat60',nc,s,'LETKF',k)
            if None in (a,b,p,q): continue
            s_ = sgn if sgn else 1
            dw.append(s_*(b-a)); dl.append(s_*(q-p)); I.append(s_*((b-a)-(q-p)))
    w=sum(1 for x in I if x>0)
    note='' if sgn else '  (VR 은 방향 미지정 — 원시 부호)'
    print(f'{k:16s} {np.median(dw):+10.4f} {np.median(dl):+10.4f} {np.median(I):+10.4f} '
          f'{w:>3d}/{len(I)} {st(w,len(I)):8.4f} {wil(I):9.4f}{note}')
    out[k]={'median_dWS':float(np.median(dw)),'median_dLETKF':float(np.median(dl)),
            'median_I':float(np.median(I)),'wins':w,'n':len(I),
            'p_sign':st(w,len(I)),'p_wilcoxon':wil(I)}
print('\n★ 30comp 층만 (서술 전용, n=5 바닥 0.0625)')
for k,sgn in KEYS.items():
    s_=sgn if sgn else 1
    I=[s_*((v('treat60',30,s,'WS',k)-v('ctrl30',30,s,'WS',k))-(v('treat60',30,s,'LETKF',k)-v('ctrl30',30,s,'LETKF',k))) for s in range(5)]
    dws=[s_*(v('treat60',30,s,'WS',k)-v('ctrl30',30,s,'WS',k)) for s in range(5)]
    print(f'  {k:16s} 중앙 ΔWS {np.median(dws):+.4f}  중앙 I {np.median(I):+.4f}  I>0 {sum(1 for x in I if x>0)}/5')
json.dump(out, open(os.path.join(HERE,'xsecondary.json'),'w'), indent=2)
