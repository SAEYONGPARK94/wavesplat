"""SS bootstrap CIs + regime splits + head-to-head for tab:strong_baseline prose (finemesh).
WS = ep100 median-of-5 per episode; LETKF = batchA single filter seed 42."""
import json
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R24 = os.path.join(HERE, '..', 'rerun_2026-07-24')


def load(path):
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def ncomp(ep):
    m = re.search(r'_(\d+)comp_', ep)
    return int(m.group(1)) if m else None


def boot_ci(vals, n=10000, seed=42):
    a = np.asarray(vals); rng = np.random.default_rng(seed)
    b = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# WS per-episode median-of-5 (nRMSE, SS, VR)
ws = {}
tmp = {}
for r in load(os.path.join(R24, 'ep100_runs.jsonl')):
    if 'error' in r:
        continue
    tmp.setdefault(r['episode'], []).append(r)
for ep, rs in tmp.items():
    ws[ep] = {'nRMSE': float(np.median([x['nRMSE'] for x in rs])),
              'SS': float(np.median([x['Skill_Score'] for x in rs])),
              'VR': float(np.median([x['Variance_Ratio'] for x in rs]))}

# LETKF per-episode (batchA)
lk = {}
for r in load(os.path.join(HERE, 'batchA_baselines.jsonl')):
    if r.get('method') == 'LETKF' and 'error' not in r:
        lk[r['episode']] = {'nRMSE': r['nRMSE'], 'SS': r['Skill_Score'], 'VR': r['Variance_Ratio']}

eps = sorted(set(ws) & set(lk))
print(f'paired episodes: {len(eps)}')


def regime(pred):
    return [e for e in eps if pred(ncomp(e))]


for label, sel in [('ALL', regime(lambda n: True)),
                   ('<=5comp', regime(lambda n: n <= 5)),
                   ('>=15comp', regime(lambda n: n >= 15))]:
    ws_ss = [ws[e]['SS'] for e in sel]
    lk_ss = [lk[e]['SS'] for e in sel]
    ws_vr = [ws[e]['VR'] for e in sel]
    lk_vr = [lk[e]['VR'] for e in sel]
    wc, hc = boot_ci(ws_ss); lc, hlc = boot_ci(lk_ss)
    print(f'\n[{label}] n={len(sel)}')
    print(f'  WS   SS mean={np.mean(ws_ss):+.3f} CI[{wc:+.3f},{hc:+.3f}]  VR={np.mean(ws_vr):.3f}')
    print(f'  LETKF SS mean={np.mean(lk_ss):+.3f} CI[{lc:+.3f},{hlc:+.3f}]  VR={np.mean(lk_vr):.3f}')

# head-to-head
win_nr = sum(1 for e in eps if ws[e]['nRMSE'] < lk[e]['nRMSE'])
win_ss = sum(1 for e in eps if ws[e]['SS'] > lk[e]['SS'])
print(f'\nHead-to-head (n={len(eps)}): WS wins nRMSE {win_nr}/{len(eps)}, WS wins SS {win_ss}/{len(eps)}')
