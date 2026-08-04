"""Recompute the <=5-component subset panel from post-phasefix raw data (no new runs).
Protocol identical to the 2026-08-01 headline replacement: per-episode median over
5 seeds, then mean over episodes; SS CI by 10k bootstrap of per-episode medians."""
import json, re, statistics as st
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl')]
by = defaultdict(list)
for r in rows:
    by[r['episode']].append(r)

def ncomp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))

med = {ep: {k: st.median([r[k] for r in v]) for k in
            ('nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2')}
       for ep, v in by.items()}

rng = np.random.default_rng(20260804)
def panel(label, eps):
    ss = np.array([med[e]['Skill_Score'] for e in eps])
    boot = np.array([rng.choice(ss, len(ss), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{label:24s} n={len(eps):3d}  nRMSE={np.mean([med[e]['nRMSE'] for e in eps]):.4f}"
          f"  VR={np.mean([med[e]['Variance_Ratio'] for e in eps]):.4f}"
          f"  rho={np.mean([med[e]['Pearson_rho'] for e in eps]):.4f}"
          f"  SS={ss.mean():+.4f} CI[{lo:+.3f},{hi:+.3f}]"
          f"  DirR2={np.mean([med[e]['Directional_R2'] for e in eps]):+.4f}")

allep = sorted(med)
panel('ALL 100', allep)
panel('<=5 comp', [e for e in allep if ncomp(e) <= 5])
panel('>=15 comp', [e for e in allep if ncomp(e) >= 15])
for n in (1, 2, 5, 15, 30):
    panel(f'{n} comp', [e for e in allep if ncomp(e) == n])
