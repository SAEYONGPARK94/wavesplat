import json, statistics as st, math
rows=[json.loads(l) for l in open('tmp/s1s2_2026-08-05/results.jsonl')]
EPS=sorted(set(r['episode'] for r in rows))
C=3.8630
def med(arm,ep,m): return st.median([r[m] for r in rows if r['arm']==arm and r['episode']==ep])
def nrmse_of(rho,v): return math.sqrt(max(1+v-2*rho*math.sqrt(v),0))/C

def signtest(d):
    npos=sum(1 for x in d if x>0); k=max(npos,len(d)-npos)
    p=min(2*sum(math.comb(len(d),i) for i in range(k,len(d)+1))/2**len(d),1.0)
    return npos,p

print("=== COUNTERFACTUAL A: freeze the tracker, then apply ONE blind constant amplitude scalar alpha ===")
print("   (alpha = the SAME shrinkage magnitude the tracker itself applies, sqrt(VR_on/VR_off) --> GT-free constant)")
alphas=[med('S1-A0',e,'Variance_Ratio')/med('X-A0-notrack',e,'Variance_Ratio') for e in EPS]
alpha=math.sqrt(st.median(alphas))
print(f"   per-episode VR_on/VR_off = {[round(a,3) for a in alphas]}  -> alpha = sqrt(median) = {alpha:.3f}\n")
print(f"{'episode':16s}{'ON(track)':>11s}{'OFF(raw)':>10s}{'OFF*a':>10s}{'ON-OFF*a':>11s}")
d=[]
for e in EPS:
    rho0=med('X-A0-notrack',e,'Pearson_rho'); v0=med('X-A0-notrack',e,'Variance_Ratio')
    n_on=med('S1-A0',e,'nRMSE'); n_off=med('X-A0-notrack',e,'nRMSE')
    n_sc=nrmse_of(rho0, v0*alpha**2)
    d.append(n_on-n_sc)
    print(f"{e.split('_')[1]+'_'+e.split('_')[2]:16s}{n_on:11.4f}{n_off:10.4f}{n_sc:10.4f}{n_on-n_sc:+11.4f}")
npos,p=signtest(d)
print(f"{'mean4':16s}{sum(med('S1-A0',e,'nRMSE') for e in EPS)/4:11.4f}{sum(med('X-A0-notrack',e,'nRMSE') for e in EPS)/4:10.4f}{sum(nrmse_of(med('X-A0-notrack',e,'Pearson_rho'),med('X-A0-notrack',e,'Variance_Ratio')*alpha**2) for e in EPS)/4:10.4f}{st.median(d):+11.4f}")
print(f"   -> scaled-freeze beats tracking in {npos}/4 episodes (medD={st.median(d):+.4f}, sign p={p:.3f})")
print(f"   -> tracker's own margin over raw freeze was only -0.0074 (2/4, p=1.000)\n")

print("=== COUNTERFACTUAL B: alpha sweep on the FROZEN beamform field (no GT used to pick alpha) ===")
print(f"{'alpha':>7s}{'mean4 nRMSE':>13s}{'vs track-ON':>13s}{'beats ON n/4':>14s}")
for a in [1.0,0.9,0.8,0.728,0.7,0.6,0.5,0.451,0.4,0.3]:
    vals=[nrmse_of(med('X-A0-notrack',e,'Pearson_rho'), med('X-A0-notrack',e,'Variance_Ratio')*a*a) for e in EPS]
    dd=[med('S1-A0',e,'nRMSE')-v for e,v in zip(EPS,vals)]
    print(f"{a:7.3f}{sum(vals)/4:13.4f}{st.median(dd):+13.4f}{sum(1 for x in dd if x>0):>10d}/4")

print("\n=== ORACLE arm, same sweep (does ANY rescaling of the frozen field make tracking worthwhile?) ===")
print(f"{'alpha':>7s}{'mean4 nRMSE':>13s}{'vs track-ON(0.1671)':>21s}")
for a in [1.0,0.9,0.848,0.8,0.7,0.6,0.5]:
    vals=[nrmse_of(med('X-T1f-notrack',e,'Pearson_rho'), med('X-T1f-notrack',e,'Variance_Ratio')*a*a) for e in EPS]
    print(f"{a:7.3f}{sum(vals)/4:13.4f}{st.median([med('S2-T1f',e,'nRMSE')-v for e,v in zip(EPS,vals)]):+21.4f}")
