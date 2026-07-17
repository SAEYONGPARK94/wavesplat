"""e45 audit part 5 — N1 probe verification + e37 decomp (ring claim)."""
import json
import numpy as np
from scipy import stats

print('====== N1 blk400 geom (fresh block argmax theta errors) ======')
G = json.load(open('tmp/e44_probe_N1_blk400_geom.json'))
d1e = np.array([r['d1e'] for r in G])
d2e = np.array([r['d2e'] for r in G])
we = np.array([r['we'] for r in G])
gain = np.array([r['gain'] for r in G])
prize = np.array([r['prize'] for r in G])
sw = np.array([r['sw'] for r in G])
print(f'n={len(G)} swap {sw.sum()}/{len(G)}')
print(f'median d1e {np.median(d1e):.1f}  d2e {np.median(d2e):.1f}  we {np.median(we):.3f}')
near = (d1e < 25) & (d2e < 25)
print(f'NEAR(<25deg both): n={near.sum()}  gain mean {gain[near].mean():+.4f} '
      f'se {gain[near].std(ddof=1)/np.sqrt(near.sum()):.4f}')
print(f'corr(gain, d1e+d2e) {np.corrcoef(gain, d1e+d2e)[0,1]:+.2f}   '
      f'corr(gain, prize) {np.corrcoef(gain, prize)[0,1]:+.2f}')
print(f'gain overall {gain.mean():+.4f} (should be ~-0.0067)  prize {prize.mean():+.4f}')

print('\n====== N1 12-sea joint decomposition (si=300 block, selected) ======')
P = json.load(open('tmp/e44_probe_N1_geom.json'))
rows = P['rows'] if isinstance(P, dict) else P
gap = np.array([r['sc_node_orc'] - r['sc_full'] for r in rows])
recD = np.array([r['sc_hybD'] - r['sc_full'] for r in rows])
recR = np.array([r['sc_hybR'] - r['sc_full'] for r in rows])
print(f'n={len(rows)}  gap mean {gap.mean():+.4f}')
print(f'fix directions : {recD.mean():+.4f}  ({recD.mean()/gap.mean()*100:.0f}%)')
print(f'fix ring+w     : {recR.mean():+.4f}  ({recR.mean()/gap.mean()*100:.0f}%)')
sel = [r['si'] for r in rows]
print('selected sis:', sel)

print('\n====== e37 crosstp decomp (the "+0.102 ring" label) ======')
D = json.load(open('results/direction3/e37_crosstp_decomp.json'))
print(json.dumps(D, indent=1)[:2000])
