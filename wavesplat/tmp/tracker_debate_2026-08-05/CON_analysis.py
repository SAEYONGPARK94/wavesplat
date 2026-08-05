import json, statistics as st, math, itertools
rows=[json.loads(l) for l in open('tmp/s1s2_2026-08-05/results.jsonl')]
EPS=sorted(set(r['episode'] for r in rows))
METRICS=['nRMSE','Pearson_rho','Variance_Ratio','Hs_error','Skill_Score','avg_RMSE_unobserved','init_nrmse','Directional_R2']

def med(arm,ep,m):
    v=[r[m] for r in rows if r['arm']==arm and r['episode']==ep and r.get(m) is not None]
    return (st.median(v), max(v)-min(v), len(v)) if v else (None,None,0)

ARMS=['S1-A0','X-A0-notrack','S2-T1f','X-T1f-notrack']
print("=== per-episode median-of-3 (range in parens) ===")
for m in ['nRMSE','Pearson_rho','Variance_Ratio','Hs_error','init_nrmse']:
    print(f"\n-- {m} --")
    print(f"{'arm':16s}"+"".join(f"{e.split('_')[1]+'_'+e.split('_')[2]:>16s}" for e in EPS)+f"{'mean4':>10s}")
    for a in ARMS:
        vals=[med(a,e,m)[0] for e in EPS]; rng=[med(a,e,m)[1] for e in EPS]
        print(f"{a:16s}"+"".join(f"{v:8.4f}({r:.3f})" for v,r in zip(vals,rng))+f"{sum(vals)/4:10.4f}")

print("\n\n=== PAIRED CONTRASTS: tracking ON minus OFF (negative = tracking helps) ===")
for base,nt,label in [('S1-A0','X-A0-notrack','BEAMFORM (deployed)'),('S2-T1f','X-T1f-notrack','ORACLE')]:
    print(f"\n--- {label}: {base} (ON) vs {nt} (OFF) ---")
    for m in ['nRMSE','Pearson_rho','Variance_Ratio','Hs_error','Skill_Score','avg_RMSE_unobserved']:
        d=[med(base,e,m)[0]-med(nt,e,m)[0] for e in EPS]
        srange=st.median([max(med(base,e,m)[1],med(nt,e,m)[1]) for e in EPS])
        npos=sum(1 for x in d if x>0)
        # two-sided exact sign test n=4
        k=max(npos,4-npos); p=2*sum(math.comb(4,i) for i in range(k,5))/16; p=min(p,1.0)
        print(f"  {m:22s} medD={st.median(d):+8.4f}  per-ep={[round(x,4) for x in d]}  ON-worse={npos}/4  p={p:.3f}  seed_range_med={srange:.4f}  |eff|/noise={abs(st.median(d))/srange if srange else float('nan'):.2f}")
