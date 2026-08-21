import json, statistics as st, math, collections
from math import comb
R=[json.loads(l) for l in open('tmp/E_remeasure_2026-08-06/arm_runs_remeasure.jsonl')]
R=[r for r in R if r.get('err') is None and r.get('avg_nRMSE') is not None]
print("valid rows:",len(R), " (file is IN PROGRESS)")
def sp(d):
    nz=[x for x in d if x!=0]; n=len(nz); k=sum(1 for x in nz if x<0)
    return k,n,min(1.0,sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(k-n/2))/2**n) if n else (0,0,1.0)
for F in ['init_nrmse','raw_init_nrmse']:
    print(f"\n########## using {F} ##########")
    for arm in ['learned','v2_konly']:
        A=[r for r in R if r['arm']==arm]
        h=[r['avg_nRMSE']-r[F] for r in A]; i=[r[F] for r in A]
        # spearman
        def rank(v):
            s=sorted(range(len(v)),key=lambda j:v[j]); rk=[0]*len(v)
            for p,j in enumerate(s): rk[j]=p
            return rk
        ri,rh=rank(i),rank(h); n=len(i)
        mi=st.mean(ri);mh=st.mean(rh)
        num=sum((a-mi)*(b-mh) for a,b in zip(ri,rh)); den=math.sqrt(sum((a-mi)**2 for a in ri)*sum((b-mh)**2 for b in rh))
        rho=num/den if den else float('nan')
        t=rho*math.sqrt((n-2)/max(1e-12,1-rho**2))
        print(f"  {arm:9s} n={n:3d} init med={st.median(i):.4f} [{min(i):.3f},{max(i):.3f}] harm med={st.median(h):+.4f} Spearman(init,harm)={rho:+.4f} t={t:+.2f}")
    # paired
    key=lambda r:(r['episode'],r['est_seed'],r['diff_seed'])
    byk=collections.defaultdict(dict)
    for r in R: byk[key(r)][r['arm']]=r
    pairs=[(k,v['learned'],v['v2_konly']) for k,v in byk.items() if len(v)==2]
    print(f"  PAIRED n={len(pairs)} (episode,est_seed,diff_seed exact match)")
    di=[l[F]-v[F] for _,l,v in pairs]; dh=[(l['avg_nRMSE']-l[F])-(v['avg_nRMSE']-v[F]) for _,l,v in pairs]
    dn=[l['avg_nRMSE']-v['avg_nRMSE'] for _,l,v in pairs]
    k1,n1,p1=sp(dh); k2,n2,p2=sp(dn)
    print(f"    run-level: dinit med={st.median(di):+.4f}  dharm med={st.median(dh):+.4f} ({k1}/{n1} improving p={p1:.3f})  dnRMSE med={st.median(dn):+.4f} ({k2}/{n2} p={p2:.3g})")
    print(f"    pass-through = dnRMSE/dinit = {st.median(dn)/st.median(di):.3f}")
    # per-episode aggregate
    bye=collections.defaultdict(list)
    for (e,_,_),l,v in pairs: bye[e].append(((l[F]-v[F]),((l['avg_nRMSE']-l[F])-(v['avg_nRMSE']-v[F])),(l['avg_nRMSE']-v['avg_nRMSE'])))
    eps=sorted(bye)
    ei=[st.median([x[0] for x in bye[e]]) for e in eps]
    eh=[st.median([x[1] for x in bye[e]]) for e in eps]
    en=[st.median([x[2] for x in bye[e]]) for e in eps]
    k3,n3,p3=sp(eh); k4,n4,p4=sp(en)
    sd=st.stdev(eh)
    from statistics import NormalDist
    mde=(NormalDist().inv_cdf(0.975)+NormalDist().inv_cdf(0.8))*sd/math.sqrt(len(eh))
    print(f"    EPISODE-level n={len(eps)}: dinit med={st.median(ei):+.4f}  dharm med={st.median(eh):+.4f} ({k3}/{n3} improving p={p3:.3f}) sd={sd:.4f} t-MDE={mde:.4f}")
    print(f"                            dnRMSE med={st.median(en):+.4f} ({k4}/{n4} p={p4:.3g})")
    print(f"    final_n median: learned={st.median([l['final_n'] for _,l,_ in pairs]):.0f}  v2_konly={st.median([v['final_n'] for _,_,v in pairs]):.0f}")
