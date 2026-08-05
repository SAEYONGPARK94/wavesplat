import json, statistics as st, math
rows=[json.loads(l) for l in open('tmp/s1s2_2026-08-05/results.jsonl')]
print("=== Identity check: is nRMSE = sqrt(1+v-2*rho*sqrt(v))/C ? ===")
Cs=[]
for r in rows:
    v=r['Variance_Ratio']; rho=r['Pearson_rho']; n=r['nRMSE']
    pred=math.sqrt(max(1+v-2*rho*math.sqrt(v),0))
    Cs.append(pred/n)
print(f"  ratio pred/nRMSE over all {len(Cs)} runs: median={st.median(Cs):.4f} mean={st.mean(Cs):.4f} min={min(Cs):.4f} max={max(Cs):.4f} sd={st.pstdev(Cs):.4f}")
C=st.median(Cs)
err=[abs(math.sqrt(max(1+r['Variance_Ratio']-2*r['Pearson_rho']*math.sqrt(r['Variance_Ratio']),0))/C - r['nRMSE'])/r['nRMSE'] for r in rows]
print(f"  with C={C:.4f}: median |rel error| = {st.median(err)*100:.2f}%  p90={sorted(err)[int(.9*len(err))]*100:.2f}%  max={max(err)*100:.2f}%")

EPS=sorted(set(r['episode'] for r in rows))
def med(arm,ep,m): return st.median([r[m] for r in rows if r['arm']==arm and r['episode']==ep])

print("\n=== SHRINKAGE ARITHMETIC (per arm, mean over 4 episodes of per-episode medians) ===")
print("  nMSE(v) = 1 + v - 2*rho*sqrt(v)  ->  minimized at v* = rho^2, min value = 1-rho^2")
print(f"  F  = shape-limited floor  = sqrt(1-rho^2)/C   (best nRMSE reachable by rescaling alone, C={C:.3f})")
print("  E  = nRMSE - F            = miscalibration excess\n")
hdr=f"{'arm':16s}{'rho':>7s}{'VR':>8s}{'VR*=rho^2':>11s}{'VR/VR*':>8s}{'nRMSE':>8s}{'F(floor)':>10s}{'E(excess)':>11s}"
print(hdr); print('-'*len(hdr))
res={}
for a in ['S1-A0','X-A0-notrack','S2-T1f','X-T1f-notrack']:
    rho=sum(med(a,e,'Pearson_rho') for e in EPS)/4
    v  =sum(med(a,e,'Variance_Ratio') for e in EPS)/4
    n  =sum(med(a,e,'nRMSE') for e in EPS)/4
    F=math.sqrt(1-rho**2)/C; res[a]=(rho,v,n,F)
    print(f"{a:16s}{rho:7.3f}{v:8.3f}{rho**2:11.3f}{v/rho**2:8.2f}{n:8.4f}{F:10.4f}{n-F:+11.4f}")

print("\n=== DECOMPOSITION of the tracker's effect (ON minus OFF) ===")
for on,off,lab in [('S1-A0','X-A0-notrack','BEAMFORM (deployed)'),('S2-T1f','X-T1f-notrack','ORACLE')]:
    r1,v1,n1,F1=res[on]; r0,v0,n0,F0=res[off]
    print(f"\n {lab}:")
    print(f"   d(rho)   = {r1-r0:+.4f}   -> shape/information change")
    print(f"   d(floor F)= {F1-F0:+.4f}  = irreducible cost of what tracking did to correlation")
    print(f"   d(excess E)={(n1-F1)-(n0-F0):+.4f}  = calibration change")
    print(f"   d(nRMSE) = {n1-n0:+.4f}  (check: {F1-F0:+.4f} {(n1-F1)-(n0-F0):+.4f} = {(F1-F0)+((n1-F1)-(n0-F0)):+.4f})")

print("\n=== COUNTERFACTUAL: rescale the notrack field by the MSE-optimal scalar (alpha=rho) ===")
for off,on,lab in [('X-A0-notrack','S1-A0','BEAMFORM'),('X-T1f-notrack','S2-T1f','ORACLE')]:
    r0,v0,n0,F0=res[off]; r1,v1,n1,F1=res[on]
    print(f" {lab}: notrack as-measured {n0:.4f} -> optimally rescaled {F0:.4f} | tracking-ON {n1:.4f} | margin {n1-F0:+.4f}")
