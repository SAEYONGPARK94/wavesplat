import json, statistics as st
from pathlib import Path
from scipy import stats
R = Path('tmp/rerun_2026-08-03')

rd = json.load(open(R / 'raydrop_comparison_rerun.json'))
print('=== raydrop: WS-P1 median-of-3, ETKF carried ===')
u_ws, s_ws, u_e, s_e = [], [], [], []
for ep, v in rd.items():
    wu = v['uniform']['WS-P1']['avg_nRMSE']['median']
    ws = v['slope']['WS-P1']['avg_nRMSE']['median']
    eu = v['uniform']['ETKF']['avg_nRMSE']; es = v['slope']['ETKF']['avg_nRMSE']
    u_ws.append(wu); s_ws.append(ws); u_e.append(eu); s_e.append(es)
    print(f"{ep:12s} WS U={wu:.4f} S={ws:.4f} d%={100*(ws-wu)/wu:+6.1f} | ETKF U={eu:.4f} S={es:.4f} d%={100*(es-eu)/eu:+6.1f}")
d = [s - u for s, u in zip(s_ws, u_ws)]
print(f"\nWS  mean U={st.mean(u_ws):.4f} S={st.mean(s_ws):.4f}  mean d={st.mean(d):+.4f} median d={st.median(d):+.4f}")
print(f"    rel = {100*(st.mean(s_ws)-st.mean(u_ws))/st.mean(u_ws):+.1f}%   slope worse: {sum(x>0 for x in d)}/{len(d)}")
print('    sign p=%.4f  wilcoxon p=%.4f' % (stats.binomtest(sum(x>0 for x in d), len(d), 0.5).pvalue,
                                            stats.wilcoxon(u_ws, s_ws).pvalue))
de = [s - u for s, u in zip(s_e, u_e)]
print(f"ETKF mean U={st.mean(u_e):.4f} S={st.mean(s_e):.4f} rel={100*(st.mean(s_e)-st.mean(u_e))/st.mean(u_e):+.1f}%  slope worse {sum(x>0 for x in de)}/{len(de)}")

# seed-std variants for the adaptive-N noise-floor statement
p1 = json.load(open(R / 'p1_comparison_rerun.json'))
per_ep_pooled = [st.mean([v['WS-P0']['avg_nRMSE']['std'], v['WS-P1']['avg_nRMSE']['std']]) for v in p1.values()]
print(f"\nadaptive-N seed std: median of per-episode pooled = {st.median(per_ep_pooled):.4f}")
print(f"  median of P1-arm only = {st.median([v['WS-P1']['avg_nRMSE']['std'] for v in p1.values()]):.4f}")
print(f"  median of P0-arm only = {st.median([v['WS-P0']['avg_nRMSE']['std'] for v in p1.values()]):.4f}")
