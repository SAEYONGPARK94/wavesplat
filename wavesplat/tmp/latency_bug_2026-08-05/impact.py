import json, re, numpy as np
STEP=1150.0; NF={'BF2':220,'BF3':190,'BF4':160,'BF5':120}
def sea(e): return re.search(r'BF\d', e).group(0)
for path,key in [('tmp/rerun_2026-07-25/batchB_ws.jsonl','elapsed_ms'),
                 ('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl','elapsed_ms'),
                 ('tmp/rerun_2026-07-24/ep100_runs.jsonl','elapsed_ms')]:
    rows=[json.loads(l) for l in open(path)]
    rows=[r for r in rows if isinstance(r.get(key),(int,float))]
    G={}
    for r in rows: G.setdefault(r['episode'],[]).append(r)
    clean=[]; dirty=[]
    for ep,rs in G.items():
        med=np.median([r[key] for r in rs]); sh=STEP/NF[sea(ep)]
        for r in rs:
            (dirty if (med-r[key])>=0.5*sh else clean).append(r[key])
    all_=np.array([r[key] for r in rows]); c=np.array(clean)
    # paper-style aggregation: per-episode median over seeds, then mean over episodes
    med_then_mean_all=np.mean([np.median([r[key] for r in rs]) for rs in G.values()])
    print(f"{path.split('/')[-1]:32s} raw_mean={all_.mean():6.3f}  clean_mean={c.mean():6.3f}  "
          f"bias={all_.mean()-c.mean():+.3f}ms ({(all_.mean()-c.mean())/c.mean()*100:+.1f}%)  "
          f"median-of-seeds-then-mean={med_then_mean_all:6.3f}  dirty={len(dirty)}/{len(rows)}")
