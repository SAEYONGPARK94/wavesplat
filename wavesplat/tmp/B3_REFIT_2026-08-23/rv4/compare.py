import json, os, statistics as st
from math import comb
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def sgn(w,n): return sum(comb(n,k) for k in range(w,n+1))/2**n
rows=[json.loads(l) for l in open(H+'/results_b4.jsonl')]
rows+=[json.loads(l) for l in open(H+'/results_b3.jsonl')]
ok=[r for r in rows if 'error' not in r]
tp=json.load(open(H+'/rv4/true_persist.json'))
K=lambda r:(r['record'],r['win'])
per={}
for r in ok: per.setdefault(r['cell'],{})[K(r)]=r['per_lead']
keys=sorted(per['FROZEN']); leads=[float(x) for x in sorted(per['FROZEN'][keys[0]],key=float)]
def T(k,l,f): return tp[f'{k[0]}|{k[1]}']['per_lead'][str(l)][f]

print('=== A. lead 별: 참지속(오라클) vs 제로 vs PERSIST vs FROZEN  (36창 중앙) ===')
print(f"{'lead':>4} {'zero':>6} {'TPora':>6} {'TPobs1':>6} {'PERS':>6} {'FROZ':>6} | {'TPora<zero':>10} {'FRO<TPora':>10} {'FRO<TPobs1':>11} {'PERS<TPora':>10}")
agg={}
for l in leads:
    s=str(l)
    z={k:T(k,l,'zero') for k in keys}; to={k:T(k,l,'tp_oracle') for k in keys}
    t1={k:T(k,l,'tp_obs1') for k in keys}
    P={k:per['PERSIST'][k][s] for k in keys}; F={k:per['FROZEN'][k][s] for k in keys}
    a=sum(1 for k in keys if to[k]<z[k]); b=sum(1 for k in keys if F[k]<to[k])
    c=sum(1 for k in keys if F[k]<t1[k]); d=sum(1 for k in keys if P[k]<to[k])
    agg[l]=(a,b,c,d)
    print(f"{l:4.0f} {st.median(z.values()):6.3f} {st.median(to.values()):6.3f} {st.median(t1.values()):6.3f} "
          f"{st.median(P.values()):6.3f} {st.median(F.values()):6.3f} | {a:5d}/36 {b:7d}/36 {c:8d}/36 {d:7d}/36")

print('\n=== B. fc_mean (lead 2-14 평균) 짝 비교 ===')
def fcm(d,k,f=None):
    v=[d[k][str(l)] if f is None else T(k,l,f) for l in leads if l>=2.0]
    return sum(v)/len(v)
for nm,g in (('FROZEN',lambda k:fcm(per['FROZEN'],k)),):
    pass
FC={c:{k:fcm(per[c],k) for k in keys} for c in ('FROZEN','PERSIST','REFIT2')}
FC['TP_ORACLE']={k:fcm(None,k,'tp_oracle') for k in keys}
FC['TP_OBS1']={k:fcm(None,k,'tp_obs1') for k in keys}
FC['TP_OBS4']={k:fcm(None,k,'tp_obs4') for k in keys}
FC['ZERO']={k:fcm(None,k,'zero') for k in keys}
def pair(a,b):
    w=sum(1 for k in keys if FC[a][k]<FC[b][k])
    rel=st.median([(FC[a][k]-FC[b][k])/FC[b][k]*100 for k in keys])
    return f"{a:9s} < {b:9s} : {w:2d}/36  rel_med {rel:+7.2f}%  p={sgn(w,36):.6g}"
for a,b in (('FROZEN','PERSIST'),('FROZEN','TP_ORACLE'),('FROZEN','TP_OBS1'),('FROZEN','TP_OBS4'),
            ('FROZEN','ZERO'),('REFIT2','TP_ORACLE'),('REFIT2','ZERO'),('PERSIST','ZERO'),
            ('TP_ORACLE','ZERO'),('TP_ORACLE','PERSIST'),('TP_OBS1','ZERO')):
    print('  '+pair(a,b))
print('  medians:', {k:round(st.median(v.values()),4) for k,v in FC.items()})

print('\n=== C. 진폭 회계 — 무상관 가정에서 예측장 분산비 v/var(g) = (nRMSE/zero)^2 - 1 ===')
print('    (양성대조: TP_ORACLE 는 정의상 v/var(g) ~= 1 이어야 한다)')
print(f"{'lead':>4} {'TPora':>8} {'PERSIST':>8} {'FROZEN':>8} {'REFIT2':>8}   {'FRO<PER(분산)':>12}")
for l in leads:
    s=str(l); z={k:T(k,l,'zero') for k in keys}
    def vr(d,f=None):
        return st.median([ (( (d[k][s] if f is None else T(k,l,f))/z[k])**2 -1) for k in keys])
    vF=[((per['FROZEN'][k][s]/z[k])**2-1) for k in keys]
    vP=[((per['PERSIST'][k][s]/z[k])**2-1) for k in keys]
    w=sum(1 for a,b in zip(vF,vP) if a<b)
    print(f"{l:4.0f} {vr(None,'tp_oracle'):8.3f} {vr(per['PERSIST']):8.3f} {vr(per['FROZEN']):8.3f} {vr(per['REFIT2']):8.3f}   {w:3d}/36 p={sgn(w,36):.2g}")

print('\n=== D. 참 필드 자기상관 corr(g0,gl) 중앙 ===')
print({int(l): round(st.median([T(k,l,'corr_g0_gl') for k in keys]),3) for l in leads})
