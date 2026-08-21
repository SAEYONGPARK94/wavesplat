import json, statistics as st, math
from math import comb
R=[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl')]
EPS=sorted(set(r['episode'] for r in R)); SEEDS=sorted(set(r['seed'] for r in R))
def sign_p(d,direction=-1):
    nz=[x for x in d if x!=0]; n=len(nz); k=sum(1 for x in nz if (x<0 if direction<0 else x>0))
    p=sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(k-n/2))/2**n if n else 1.0
    return k,n,min(1.0,p)
def med(c,f,e,sub=None):
    v=[(r[f] if sub is None else r[f][sub]) for r in R if r['cell']==c and r['episode']==e]
    return st.median(v)

print("=== CON: spec fields identical between D-BASE and D-NOTRACK? (exact per-run) ===")
flds=['k_mag_rel_error_mean','direction_error_deg_mean','detection_rate','n_est','n_matched','amp_rel_error_mean','sum_w_est']
for f in flds:
    exact_same=True; vals=[]
    for e in EPS:
        for s in SEEDS:
            a=[r for r in R if r['cell']=='D-BASE' and r['episode']==e and r['seed']==s][0]['spec'][f]
            b=[r for r in R if r['cell']=='D-NOTRACK' and r['episode']==e and r['seed']==s][0]['spec'][f]
            if a!=b: exact_same=False; vals.append((e,s,a,b))
    mA=st.median([med('D-BASE','spec',e,f) for e in EPS]); mB=st.median([med('D-NOTRACK','spec',e,f) for e in EPS])
    print(f"  {f:24s} bit-identical={exact_same!s:5s}  D-BASE med={mA:.6f}  D-NOTRACK med={mB:.6f}")
    if vals[:1]: print(f"      first mismatch: {vals[0]}")

print()
print("=== CON: decorr / shrink / m / nRMSE deltas (prereg convention) ===")
for A,B,lbl in [('D-NOTRACK','D-BASE','deploy'),('X-NOTRACK','X-BASE','oracle')]:
    for f in ['decorr_framemean','shrink_framemean','m_framemean','nRMSE','absVR1_framemean']:
        d=[med(A,f,e)-med(B,f,e) for e in EPS]; k,n,p=sign_p(d)
        print(f"  {lbl:7s} {A:10s} d{f:20s} {st.median(d):+.4f}  {k}/{n} improving  p={p:.4f}")
    print()

print("=== CON: rho-bar, VR, MMSE-optimal v*=rho^2 ===")
for c in ['D-BASE','D-NOTRACK','X-BASE','K4']:
    rho=[]; v=[]
    for r in [x for x in R if x['cell']==c]:
        rho.append(st.mean(r['rho_frames'])); v.append(st.mean(r['vr_frames']))
    rb=st.median(rho); vb=st.median(v)
    print(f"  {c:10s} rho_bar={rb:.4f}  v_bar={vb:.4f}  v*=rho^2={rb**2:.4f}  ratio v/v*={vb/rb**2:.2f}x  sqrt(v)={math.sqrt(vb):.4f}")

print()
print("=== CON: global-rescale ceiling  mean sqrt(1-rho^2) / mean sqrt(m) per run ===")
for c in ['D-BASE','D-NOTRACK','X-BASE']:
    fr=[]
    for r in [x for x in R if x['cell']==c]:
        num=st.mean([math.sqrt(max(0.0,1-rho**2)) for rho in r['rho_frames']])
        den=st.mean([math.sqrt(max(0.0,(1-rho**2)+(math.sqrt(max(v,0))-rho)**2)) for rho,v in zip(r['rho_frames'],r['vr_frames'])])
        fr.append(1-num/den)
    print(f"  {c:10s} gain-from-perfect-rescale: median={100*st.median(fr):.1f}%  min={100*min(fr):.1f}%  max={100*max(fr):.1f}%")

print()
print("=== CON: mean-reversion identity for harm slope (tracking-ON cells, cell x episode) ===")
ON=[c for c in sorted(set(r['cell'] for r in R)) if c not in ('X-NOTRACK','D-NOTRACK')]
X=[];Y=[]
for c in ON:
    for e in EPS:
        X.append(med(c,'init_nrmse',e)); Y.append(med(c,'nRMSE',e))
n=len(X); mx=st.mean(X); my=st.mean(Y)
sxy=sum((a-mx)*(b-my) for a,b in zip(X,Y)); sxx=sum((a-mx)**2 for a in X); syy=sum((b-my)**2 for b in Y)
b=sxy/sxx; a=my-b*mx; r=sxy/math.sqrt(sxx*syy)
H=[y-x for x,y in zip(X,Y)]; mh=st.mean(H)
bh=sum((x-mx)*(h-mh) for x,h in zip(X,H))/sxx; ah=mh-bh*mx
print(f"  n={n} cells(ON) tracked = {a:+.4f} + {b:.4f}*init   r={r:+.4f}")
print(f"  harm = {ah:+.4f} + {bh:.4f}*init")
print(f"  mean-reversion null slope = r*sd(Y)/sd(X) - 1 = {r*math.sqrt(syy/sxx)-1:+.4f}   [identity check: {bh:+.4f}]")
print(f"  init range: {min(X):.4f}-{max(X):.4f}")
