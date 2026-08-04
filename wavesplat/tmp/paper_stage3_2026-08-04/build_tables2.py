import json, statistics as st
from pathlib import Path
from scipy import stats
R = Path('tmp/rerun_2026-08-03')

p1 = json.load(open(R / 'p1_comparison_rerun.json'))
print('=== adaptive N: 12 episodes, WS-P0 (fixed N, tw=1) vs WS-P1 (adaptive N, tw=3) ===')
rows, d = [], []
for ep, v in p1.items():
    a, b = v['WS-P0'], v['WS-P1']
    rows.append((ep, a['avg_nRMSE']['median'], a['avg_nRMSE']['std'],
                 b['avg_nRMSE']['median'], b['avg_nRMSE']['std'],
                 a['final_n']['median'], b['final_n']['median']))
    d.append(a['avg_nRMSE']['median'] - b['avg_nRMSE']['median'])
for r in rows:
    print(f"{r[0]:32s} P0={r[1]:.4f}+-{r[2]:.4f}  P1={r[3]:.4f}+-{r[4]:.4f}  d={r[1]-r[3]:+.4f}  n_fin {r[5]:.0f}/{r[6]:.0f}")
p0m = [r[1] for r in rows]; p1m = [r[3] for r in rows]
print(f"\nn={len(rows)}  mean P0={st.mean(p0m):.4f}  mean P1={st.mean(p1m):.4f}  gain={st.mean(d):+.4f}")
print(f"P1 better: {sum(x>0 for x in d)}/{len(d)}")
print('wilcoxon p=%.4f' % stats.wilcoxon(p0m, p1m).pvalue)
print('sign  p=%.4f' % stats.binomtest(sum(x>0 for x in d), len(d), 0.5).pvalue)
sd = [v['WS-P0']['avg_nRMSE']['std'] for v in p1.values()] + [v['WS-P1']['avg_nRMSE']['std'] for v in p1.values()]
print(f"median per-episode seed std = {st.median(sd):.4f}  (ratio to gain = {st.median(sd)/abs(st.mean(d)):.0f}x)")
print('final_n medians: P0', sorted(r[5] for r in rows), ' P1', sorted(r[6] for r in rows))

rd = json.load(open(R / 'raydrop_comparison_rerun.json'))
print('\n=== raydrop ===', type(rd), (list(rd)[:4] if isinstance(rd, dict) else len(rd)))
print(json.dumps(rd, indent=1)[:900])
