import json, statistics as st, math
from math import comb
R2=[json.loads(l) for l in open('tmp/rx2_init_2026-08-05/results.jsonl')]
R1=[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl')]
EPS=sorted(set(r['episode'] for r in R2)); CELLS=sorted(set(r['cell'] for r in R2))
def med(R,c,f,e): return st.median([r[f] for r in R if r['cell']==c and r['episode']==e])
def sp(d,direction=+1):
    nz=[x for x in d if x!=0]; n=len(nz); k=sum(1 for x in nz if (x>0 if direction>0 else x<0))
    return k,n,min(1.0,sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(k-n/2))/2**n)

print("=== PRO per-episode gap values (nRMSE - init_nrmse, seed median first) ===")
for R,c,lbl in [(R2,'AUTO','rx2 AUTO'),(R2,'G1P1','rx2 G1P1'),(R1,'X-BASE','rx1 X-BASE'),(R1,'X-NOTRACK','rx1 X-NOTRACK'),(R1,'D-BASE','rx1 D-BASE'),(R1,'D-NOTRACK','rx1 D-NOTRACK'),(R1,'K4','rx1 K4')]:
    g=[med(R,c,'nRMSE',e)-med(R,c,'init_nrmse',e) for e in EPS]
    k,n,p=sp(g)
    print(f"  {lbl:16s} per-ep={[round(x,4) for x in g]} median={st.median(g):+.4f} {k}/{n} sd={st.stdev(g):.4f}")

print()
print("=== NEUTRAL pooled-convention contraction prediction (median b / median fp / medEpSeedMed init) ===")
bs=[];fps=[]
for e in EPS:
    X=[med(R2,c,'init_nrmse',e) for c in CELLS]; Y=[med(R2,c,'nRMSE',e) for c in CELLS]
    mx=st.mean(X);my=st.mean(Y);sxx=sum((a-mx)**2 for a in X);sxy=sum((a-mx)*(b-my) for a,b in zip(X,Y))
    b=sxy/sxx; bs.append(b); fps.append((my-b*mx)/(1-b))
B=st.median(bs); F=st.median(fps)
for lbl,base,nt in [('oracle','X-BASE','X-NOTRACK'),('deploy','D-BASE','D-NOTRACK')]:
    init=st.median([med(R1,base,'init_nrmse',e) for e in EPS])
    meas=st.median([med(R1,base,'nRMSE',e)-med(R1,nt,'nRMSE',e) for e in EPS])
    print(f"  {lbl}: init={init:.4f}  pooled pred=(1-{B:.4f})({F:.4f}-{init:.4f})={(1-B)*(F-init):+.5f}  measured={meas:+.5f}")
print(f"  per-episode fixed points: {[round(x,4) for x in fps]}   per-ep b: {[round(x,4) for x in bs]}")
print(f"  per-ep D-BASE init: {[round(med(R1,'D-BASE','init_nrmse',e),4) for e in EPS]}")
print(f"  per-ep predicted deploy harm: {[round((1-b)*(f-med(R1,'D-BASE','init_nrmse',e)),4) for e,b,f in zip(EPS,bs,fps)]}")
print(f"  per-ep MEASURED deploy harm: {[round(med(R1,'D-BASE','nRMSE',e)-med(R1,'D-NOTRACK','nRMSE',e),4) for e in EPS]}")

print()
print("=== PRO: retention in m (not sqrt m) ===")
def mwin(r): return st.mean([(1-rho**2)+(math.sqrt(max(v,0))-rho)**2 for rho,v in zip(r['rho_frames'],r['vr_frames'])])
def m0(r):
    rho=r['rho_frames'][0]; v=r['vr_frames'][0]; return (1-rho**2)+(math.sqrt(max(v,0))-rho)**2
for c in ['AUTO','G1P1','G1','G1P3','ORACLE-K','ORACLE']:
    d0=[st.median([m0(r) for r in R2 if r['cell']==c and r['episode']==e])-st.median([m0(r) for r in R2 if r['cell']=='X-BASE' and r['episode']==e]) for e in EPS]
    dw=[st.median([mwin(r) for r in R2 if r['cell']==c and r['episode']==e])-st.median([mwin(r) for r in R2 if r['cell']=='X-BASE' and r['episode']==e]) for e in EPS]
    print(f"  {c:9s} m retention={100*st.median(dw)/st.median(d0):.1f}%")

print()
print("=== sign-test MDE (80% power, requiring all-improve) vs t-based MDE ===")
from statistics import NormalDist
def t_mde(sd,n): return (NormalDist().inv_cdf(0.975)+NormalDist().inv_cdf(0.80))*sd/math.sqrt(n)
def sign_mde(sd,n,k_req):
    # smallest true effect (in sd units) s.t. P(>=k_req of n improve) >= 0.80
    lo,hi=0.0,6.0
    for _ in range(80):
        mid=(lo+hi)/2; p=NormalDist().cdf(mid)
        pw=sum(comb(n,i)*p**i*(1-p)**(n-i) for i in range(k_req,n+1))
        if pw<0.80: lo=mid
        else: hi=mid
    return hi*sd
sd=0.02152
for n,kreq in [(4,4),(6,6),(8,8),(12,10),(12,12),(20,16)]:
    floorp=min(1.0,sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(kreq-n/2))/2**n)
    print(f"  n={n:2d} require {kreq}/{n} (two-sided p={floorp:.4f})  t-MDE={t_mde(sd,n):.4f}  SIGN-TEST MDE={sign_mde(sd,n,kreq):.4f}")
