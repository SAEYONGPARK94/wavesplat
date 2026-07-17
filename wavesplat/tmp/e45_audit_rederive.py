"""e45 audit — independent re-derivation from raw JSONs. READ-ONLY on results."""
import json
import numpy as np
from scipy import stats

D = 'results/direction3/'


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=d.mean(), se=se, t=float(t), p=float(p), lo=lo, hi=hi,
                win=int((d > 0).sum()), n=n)


def fmt(s):
    return (f"{s['mean']:+.4f} +-{s['se']:.4f} CI[{s['lo']:+.4f},{s['hi']:+.4f}] "
            f"t{s['t']:+.2f} p{s['p']:.2g} win {s['win']}/{s['n']}")


print('================ E-44 re-derivation ================')
e44 = json.load(open(D + 'e44_encounter_transfer.json'))
rows = e44['rows']
print('rows:', len(rows), ' uni:', len(e44['uni']))
# NaN / dropped analysis
bad = [r for r in rows if not all(np.isfinite(r.get(f'enc10_{a}', np.nan))
                                  for a in ['cur', 'm2', 'orc', 'buoy', 'ww3'])]
print('rows with any non-finite enc10:', len(bad))
for r in bad:
    print('  si', r['si'], 'U', round(r['U'], 3),
          {k: r[k] for k in r if isinstance(r[k], float) and not np.isfinite(r[k])})
for tag in ['fld', 'enc10', 'enc05', 'enc510']:
    for a, b in [('orc', 'cur'), ('orc', 'm2'), ('m2', 'cur'),
                 ('buoy', 'cur'), ('ww3', 'cur')]:
        ka, kb = f'{tag}_{a}', f'{tag}_{b}'
        if ka not in rows[0]:
            continue
        d = [r[ka] - r[kb] for r in rows
             if np.isfinite(r.get(ka, np.nan)) and np.isfinite(r.get(kb, np.nan))]
        s = paired(d)
        stored = e44['contrasts'].get(f'{tag}_{a}-{b}' if tag != 'fld'
                                      else f'anchor_fld_{a}-{b}')
        match = ''
        if stored:
            match = ('OK' if abs(stored['mean'] - s['mean']) < 1e-9 else
                     f"MISMATCH stored {stored['mean']:+.4f}")
        print(f'  {tag:6s} {a:4s}-{b:4s}: {fmt(s)}  {match}')

# horizon decay: paired per-sea (enc05_d - enc510_d)
print('\n-- horizon decay (per-sea paired: d05 - d510) --')
for a in ['orc', 'buoy', 'ww3']:
    d = [(r[f'enc05_{a}'] - r[f'enc05_cur']) - (r[f'enc510_{a}'] - r[f'enc510_cur'])
         for r in rows if all(np.isfinite(r.get(k, np.nan)) for k in
                              [f'enc05_{a}', 'enc05_cur', f'enc510_{a}', 'enc510_cur'])]
    print(f'  {a:4s}: {fmt(paired(d))}')

# wilcoxon check
d = np.array([r['enc10_orc'] - r['enc10_cur'] for r in rows])
d = d[np.isfinite(d)]
print('wilcoxon p:', stats.wilcoxon(d)[1], 'stored', e44['wilcoxon_p'])
print('heavy tails: kurtosis', stats.kurtosis(d), 'skew', stats.skew(d))
# uni arm coverage
print('uni row keys:', sorted(e44['uni'][0].keys()))

print('\n================ block heterogeneity (oracle - cur) ================')
blocks = {}
e37x = json.load(open(D + 'e37_crosstp_exploratory.json'))
print('e37 keys:', list(e37x.keys())[:20])
e42 = json.load(open(D + 'e42_blind_recovery.json'))
print('e42 keys:', list(e42.keys())[:20])
e43b = json.load(open(D + 'e43b_v3_pipeline.json'))
print('e43b keys:', list(e43b.keys())[:20])
e43c = json.load(open(D + 'e43c_true_node_rank.json'))
print('e43c keys:', list(e43c.keys())[:20])
e43e = json.load(open(D + 'e43e_fresh_battery.json'))
print('e43e keys:', list(e43e.keys())[:20])
