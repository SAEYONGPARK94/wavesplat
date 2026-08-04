"""Post-correction init-vs-tracked comparison, from the 2026-08-03 rerun (no new runs)."""
import json, statistics as st
from pathlib import Path
R = Path('tmp/rerun_2026-08-03')

print('=== ablation Full arm (v1 init), per-episode median over 3 seeds ===')
abl = [r for r in json.load(open(R / 'ablation_seeded_rerun.json')) if r['config'] == 'Full']
for r in sorted(abl, key=lambda x: x['episode']):
    i = r['init_nrmse']['median']; t = r['avg_nRMSE']['median']
    print(f"{r['episode']:30s} init={i:.4f} tracked={t:.4f}  {'WORSE' if t>i else 'better'} ({t-i:+.4f})")

print('\n=== adaptive-N arms: init vs tracked (median over 3 seeds) ===')
p1 = json.load(open(R / 'p1_comparison_rerun.json'))
for ep, v in p1.items():
    for arm in ('WS-P0', 'WS-P1'):
        sr = v[arm]['seed_results']
        i = st.median([s['init_nrmse'] for s in sr]); t = st.median([s['avg_nRMSE'] for s in sr])
        print(f"{ep:28s} {arm} init={i:.4f} tracked={t:.4f} {'WORSE' if t>i else 'better'} ({t-i:+.4f})")
