import json, statistics as st, math
R2=[json.loads(l) for l in open('tmp/rx2_init_2026-08-05/results.jsonl')]
R1=[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl')]
EPS=sorted(set(r['episode'] for r in R2))
CELLS=sorted(set(r['cell'] for r in R2))
def med(R,c,f,e): 
    v=[r[f] for r in R if r['cell']==c and r['episode']==e]; return st.median(v)

print("=== NEUTRAL: within-episode contraction fit tracked = a + b*init over rx2 cells ===")
bs=[];fps=[]
for e in EPS:
    X=[med(R2,c,'init_nrmse',e) for c in CELLS]; Y=[med(R2,c,'nRMSE',e) for c in CELLS]
    n=len(X); mx=st.mean(X); my=st.mean(Y)
    sxx=sum((a-mx)**2 for a in X); sxy=sum((a-mx)*(b-my) for a,b in zip(X,Y)); syy=sum((b-my)**2 for b in Y)
    b=sxy/sxx; a=my-b*mx; r=sxy/math.sqrt(sxx*syy); fp=a/(1-b)
    bs.append(b); fps.append(fp)
    print(f"  {e[:28]:28s} n={n} b={b:.4f} a={a:+.4f} r={r:.4f} fixedpoint={fp:.4f} initrange={min(X):.3f}-{max(X):.3f}")
print(f"  median b={st.median(bs):.4f}  median fixedpoint={st.median(fps):.4f}   b range {min(bs):.3f}-{max(bs):.3f}")

print()
print("=== NEUTRAL: predicted harm=(1-b)(fp - init) vs measured ON-OFF ===")
# oracle: rx1 X-BASE init, harm measured = X-BASE nRMSE - X-NOTRACK nRMSE
for lbl,base,notrack,R in [('oracle','X-BASE','X-NOTRACK',R1),('deploy','D-BASE','D-NOTRACK',R1)]:
    meas=[med(R,base,'nRMSE',e)-med(R,notrack,'nRMSE',e) for e in EPS]
    pred=[]
    for e,bb,ff in zip(EPS,bs,fps):
        init=med(R,base,'init_nrmse',e)
        pred.append((1-bb)*(ff-init))
    print(f"  {lbl:7s} predicted median={st.median(pred):+.4f}  measured median={st.median(meas):+.4f}  err={st.median(pred)-st.median(meas):+.4f}")

print()
print("=== NEUTRAL: NOTRACK cells' own (framemean - init) bias floor, n=12 runs ===")
for c,R in [('X-NOTRACK',R1),('D-NOTRACK',R1)]:
    d=[r['nRMSE']-r['init_nrmse'] for r in R if r['cell']==c]
    print(f"  {c:10s} median(n={len(d)})={st.median(d):+.4f}  mean={st.mean(d):+.4f}")

print()
print("=== PRO: frame-based survival via sqrt(m) window mean vs X-BASE ===")
def sqrtm_windowmean(r):
    return st.mean([math.sqrt(max(0.0,(1-rho**2)+(math.sqrt(max(v,0))-rho)**2)) for rho,v in zip(r['rho_frames'],r['vr_frames'])])
def sqrtm_frame0(r):
    rho=r['rho_frames'][0]; v=r['vr_frames'][0]
    return math.sqrt(max(0.0,(1-rho**2)+(math.sqrt(max(v,0))-rho)**2))
for c in ['AUTO','G1P1','G1','G1P3','ORACLE-K','ORACLE']:
    d0=[];dw=[]
    for e in EPS:
        a0=st.median([sqrtm_frame0(r) for r in R2 if r['cell']==c and r['episode']==e])
        b0=st.median([sqrtm_frame0(r) for r in R2 if r['cell']=='X-BASE' and r['episode']==e])
        aw=st.median([sqrtm_windowmean(r) for r in R2 if r['cell']==c and r['episode']==e])
        bw=st.median([sqrtm_windowmean(r) for r in R2 if r['cell']=='X-BASE' and r['episode']==e])
        d0.append(a0-b0); dw.append(aw-bw)
    s=st.median(dw)/st.median(d0)
    print(f"  {c:9s} frame0 gain={st.median(d0):+.4f}  window gain={st.median(dw):+.4f}  retention={100*s:.1f}%  eaten={100*(1-s):.1f}%")
