"""CON audit part 6: obs_resid_rms is MEMBER-0, every GT metric is the MEDIAN-OF-3 field.
sequential_update.py:1423 `MedianEnsembleUpdater.model -> self._primary.model`
run.py:234           `_zp = updater.model(...)`            <- member 0
run.py:201           `pred = updater.predict_field(...)`    <- torch.median over members
Quantify how many rows are affected and whether the verdicts move."""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
L = ['AUTO-NOTRACK', 'AUTO-LR1000', 'AUTO-LR100', 'AUTO-LR10', 'AUTO']


def sgn_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else 1.0


def spear(a, b):
    ra = [sorted(a).index(x) for x in a]
    rb = [sorted(b).index(x) for x in b]
    ma, mb = sum(ra) / 5, sum(rb) / 5
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float('nan')


rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_lane1.jsonl')) if l.strip()]
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean'] for n in json.load(
    open(os.path.join(HERE, '..', 'Rx7_tracker_deploy_2026-08-07', 'nulls_rx7.json')))}
bad = sorted({(r['episode'], r['seed']) for r in rows if r['ensemble_k'] != 1})
print(f'runs with ensemble_k=3 (obs_resid_rms measures member 0, GT metrics measure the '
      f'median-of-3 field): {sum(1 for r in rows if r["ensemble_k"]!=1)}/288')
for e, s in bad:
    print(f'   {e[8:-17]:24s} seed={s}   router picked v2_konly -> median-of-3')


def run(drop):
    I = {}
    for r in rows:
        if drop and (r['episode'], r['seed']) in bad:
            continue
        I.setdefault((r['cell'], r['episode']), []).append(r)

    def med(c, e, k):
        return st.median([x[k] for x in I[(c, e)]])
    eps = sorted({r['episode'] for r in rows})
    kept = [e for e in eps if med('AUTO', e, 'nRMSE') < NULL[e]]
    out = {}
    for k in ('avg_obs_loss', 'obs_resid_rms', 'nRMSE'):
        d = [med('AUTO', e, k) - med('AUTO-NOTRACK', e, k) for e in kept]
        d = [x for x in d if x != 0]
        w = sum(1 for x in d if x < 0)
        out[k] = (st.median(d), w, len(d), sgn_p(w, len(d)))
    rh = [spear([med(c, e, 'obs_resid_rms') for c in L],
                [med(c, e, 'avg_RMSE_observed') for c in L]) for e in kept]
    out['rho'] = (st.median(rh), sum(1 for x in rh if x >= .9), len(rh))
    out['kept'] = len(kept)
    return out


for lbl, drop in (('as reported     ', False), ('ensemble rows out', True)):
    o = run(drop)
    print(f'\n {lbl}  n_kept={o["kept"]}')
    for k in ('avg_obs_loss', 'obs_resid_rms', 'nRMSE'):
        m, w, n, p = o[k]
        print(f'   {k:16s} medD={m:+.6f}  {w:2d}/{n:<2d}  p={p:.2e}')
    m, h, n = o['rho']
    print(f'   PRIMARY-2 rho median={m:+.4f}  >=0.9: {h}/{n}')
print('\n=> the defect touches 2 of 48 (episode,seed) pairs; neither verdict moves.')
print('   It is still a real defect: on those rows the GT-free residual and the GT')
print('   metrics describe different fields, and avg_obs_loss is the MEDIAN of the three')
print('   member losses (sequential_update.py:1495) while obs_resid_rms is member 0 --')
print('   so the two GT-free numbers in the same row are not even mutually consistent.')
