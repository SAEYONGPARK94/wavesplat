import json, os, numpy as np, collections
HERE='/home/vexit/wavesplat/wavesplat/tmp'
def load(p):
    rows=[]
    for line in open(p):
        try: r=json.loads(line)
        except Exception: continue
        if 'error' in r: continue
        rows.append(r)
    return rows
rows=load(os.path.join(HERE,'s1s2_2026-08-05','results.jsonl'))
print('s1s2 rows', len(rows))
# median-of-3 per (arm, episode)
by=collections.defaultdict(dict)
for r in rows:
    by[(r['arm'],r['episode'])][r['seed']]=r
arms=sorted({a for a,_ in by})
eps=sorted({e for _,e in by})
print(f"{'arm':<16s} {'init_nrmse mean4':>17s} {'nRMSE mean4':>12s}  per-ep init_nrmse")
for a in arms:
    med_i=[];med_n=[]
    detail=[]
    for e in eps:
        d=by[(a,e)]
        i=np.median([d[s]['init_nrmse'] for s in sorted(d)])
        n=np.median([d[s]['nRMSE'] for s in sorted(d)])
        med_i.append(i); med_n.append(n); detail.append(f'{i:.4f}')
    print(f'{a:<16s} {np.mean(med_i):>17.4f} {np.mean(med_n):>12.4f}  ' + ' '.join(detail))
print()
print('episodes order:', [e.replace('episode_','').replace('_seed0_uniform.h5','') for e in eps])
