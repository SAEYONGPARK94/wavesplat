"""Stage-3 tex propagation: emit table bodies from the 2026-08-03 rerun raw data.
No new runs. Reads only tmp/rerun_2026-08-03/ and tmp/cdip_support_2026-08-03/."""
import json, statistics as st
from pathlib import Path

R = Path('tmp/rerun_2026-08-03')

# ---------- 1. ablation: 6 episodes x 4 configs, median +- std over 3 seeds ----------
abl = json.load(open(R / 'ablation_seeded_rerun.json'))
by = {(r['episode'], r['config']): r for r in abl}
eps = sorted({r['episode'] for r in abl})
cfgs = ['Full', 'A1_noDisp', 'A2_noConf', 'v2_disp']
print('=== ablation configs present ===', sorted({r['config'] for r in abl}))
for e in eps:
    row = [e]
    for c in sorted({r['config'] for r in abl}):
        r = by.get((e, c))
        row.append(f"{c}={r['avg_nRMSE']['median']:.4f}+-{r['avg_nRMSE']['std']:.4f}" if r else f"{c}=NA")
    print(' | '.join(row))
print('--- column medians / means over 6 episodes ---')
for c in sorted({r['config'] for r in abl}):
    v = [by[(e, c)]['avg_nRMSE']['median'] for e in eps if (e, c) in by]
    print(f"{c}: median={st.median(v):.4f} mean={st.mean(v):.4f} n={len(v)}")

# ---------- 2. adaptive N ----------
p1 = json.load(open(R / 'p1_comparison_rerun.json'))
print('\n=== p1 keys ===', type(p1), (list(p1)[:8] if isinstance(p1, dict) else len(p1)))
print(json.dumps(p1, indent=1)[:1200])
