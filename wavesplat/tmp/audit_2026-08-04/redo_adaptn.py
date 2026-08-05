"""INDEPENDENT re-derivation of the adaptive-N NULL claim (2026-08-04 audit).

Rebuilt from progress.jsonl (the per-run raw log), not from the summary JSON.
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon, binomtest, t as tdist

ROOT = '/home/vexit/wavesplat/wavesplat'
P = os.path.join(ROOT, 'tmp/rerun_2026-08-03/progress.jsonl')

runs = [json.loads(l) for l in open(P)]
p1 = [r for r in runs if r['tag'].startswith('p1/')]
assert not any('error' in r for r in runs)

# episode -> arm -> seed -> value  (rebuilt from the raw per-run log)
byep = defaultdict(lambda: defaultdict(dict))
byn = defaultdict(lambda: defaultdict(dict))
for r in p1:
    _, ep, arm = r['tag'].split('/')
    byep[ep][arm][r['seed']] = r['avg_nRMSE']
    byn[ep][arm][r['seed']] = r['final_n']

eps = list(byep)
out = {'n_total_runs': len(runs), 'n_errors': 0, 'n_p1_runs': len(p1),
       'n_episodes': len(eps), 'seeds': sorted({r['seed'] for r in p1})}

# ---------- episode-level, median-of-3 (the reported unit) ----------
p0 = np.array([np.median(list(byep[e]['WS-P0'].values())) for e in eps])
pp1 = np.array([np.median(list(byep[e]['WS-P1'].values())) for e in eps])
d = p0 - pp1                                   # positive = adaptive arm better
out['episode_level'] = {
    'mean_P0': float(p0.mean()), 'mean_P1': float(pp1.mean()),
    'gain_mean': float(p0.mean() - pp1.mean()),
    'gain_median': float(np.median(d)),
    'P1_better': f'{int((d > 0).sum())}/{len(d)}',
    'wilcoxon_p': float(wilcoxon(p0, pp1).pvalue),
    'sign_test_p': float(binomtest(int((d > 0).sum()), len(d)).pvalue),
    'sd_of_paired_diff': float(d.std(ddof=1)),
}

# ---------- seed-level pairing (same seed in both arms => stronger design) ----
dd, cl = [], []
for e in eps:
    for s in byep[e]['WS-P0']:
        if s in byep[e]['WS-P1']:
            dd.append(byep[e]['WS-P0'][s] - byep[e]['WS-P1'][s]); cl.append(e)
dd = np.array(dd)
out['seed_level_paired'] = {
    'n_pairs': len(dd), 'mean_diff': float(dd.mean()),
    'median_diff': float(np.median(dd)), 'better': f'{int((dd > 0).sum())}/{len(dd)}',
    'wilcoxon_p': float(wilcoxon(dd).pvalue), 'sd': float(dd.std(ddof=1)),
    'note': 'not independent: 3 seeds share an episode',
}

# ---------- seed noise ----------
s0 = np.array([np.std(list(byep[e]['WS-P0'].values()), ddof=1) for e in eps])
s1 = np.array([np.std(list(byep[e]['WS-P1'].values()), ddof=1) for e in eps])
out['seed_noise'] = {
    'std_median_P0': float(np.median(s0)), 'std_median_P1': float(np.median(s1)),
    'std_median_pooled': float(np.median(np.concatenate([s0, s1]))),
    'ratio_noise_to_effect': float(np.median(np.concatenate([s0, s1])) /
                                   abs(p0.mean() - pp1.mean())),
}

# ---------- final_n ----------
n0 = np.array([np.median(list(byn[e]['WS-P0'].values())) for e in eps])
n1 = np.array([np.median(list(byn[e]['WS-P1'].values())) for e in eps])
allseed_n1 = np.array([v for e in eps for v in byn[e]['WS-P1'].values()])
out['final_n'] = {
    'median_P0': float(np.median(n0)), 'median_P1': float(np.median(n1)),
    'P0_values': n0.tolist(), 'P1_values': n1.tolist(),
    'P1_episodes_pruned_below_15': int((n1 < 15).sum()),
    'P1_episodes_at_15': int((n1 == 15).sum()),
    'P1_episodes_grown_above_15': int((n1 > 15).sum()),
    'P1_per_seed_runs_not_15': f'{int((allseed_n1 != 15).sum())}/{len(allseed_n1)}',
    'P1_per_seed_values': allseed_n1.tolist(),
    'P0_all_15': bool((np.array([v for e in eps for v in byn[e]['WS-P0'].values()]) == 15).all()),
}

# ---------- MDE (minimum detectable effect) ----------
sd = d.std(ddof=1)
n = len(d)
for alpha, power in [(0.05, 0.80), (0.05, 0.50)]:
    ta = tdist.ppf(1 - alpha / 2, n - 1)
    tb = tdist.ppf(power, n - 1)
    out.setdefault('MDE', {})[f'paired_t_a{alpha}_pow{power}'] = float((ta + tb) * sd / np.sqrt(n))

# Simulation MDE for the actual test used (Wilcoxon signed-rank), shifting the
# observed paired differences so noise structure is the empirical one.
rng = np.random.default_rng(0)


def power_at(delta, nsim=4000):
    hit = 0
    for _ in range(nsim):
        samp = rng.choice(d - d.mean(), size=n, replace=True) + delta
        if len(set(samp)) > 1 and wilcoxon(samp).pvalue < 0.05:
            hit += 1
    return hit / nsim


grid = np.arange(0.002, 0.030, 0.002)
pw = {float(g): power_at(g, 1500) for g in grid}
out['MDE']['wilcoxon_power_curve'] = pw
mde80 = next((g for g in grid if pw[float(g)] >= 0.80), None)
out['MDE']['wilcoxon_MDE_power80'] = float(mde80) if mde80 is not None else '>0.030'
out['MDE']['power_at_old_claimed_gain_0.0144'] = float(power_at(0.0144, 2000))

# ---------- CI on the effect; can we exclude the old +0.0144? ----------
ci = tdist.interval(0.95, n - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(n))
out['effect_CI'] = {
    'mean_diff': float(d.mean()), 'ci95': [float(ci[0]), float(ci[1])],
    'excludes_old_gain_0.0144': bool(ci[1] < 0.0144),
    'excludes_zero': bool(ci[0] > 0 or ci[1] < 0),
}
# Hodges-Lehmann (Wilcoxon point estimate) + walsh-average CI
w = np.array([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
out['effect_CI']['hodges_lehmann'] = float(np.median(w))

print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(ROOT, 'tmp/audit_2026-08-04/adaptn_redo.json'), 'w'), indent=2)
