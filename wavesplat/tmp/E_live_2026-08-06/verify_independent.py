"""E-LIVE independent re-derivation.

Written WITHOUT reusing score_e_live.py / score_confirm2.py code: the offline assembly,
the routing prediction and every slice mean are recomputed from the raw jsonl/h5 listing,
then compared to the live `--arms auto` rows.  Purpose = the dual-verification rule
(feedback_dual_verification_default.md): no headline number may rest on a single script.

    python3 tmp/E_live_2026-08-06/verify_independent.py
"""
import json
import os
import re
import sys
from collections import Counter
from math import comb
from statistics import median, mean

BASE = 'tmp/confirm2_2026-07-27'
DATA = 'data_confirm2_2026-07-27'
LIVE = 'tmp/E_live_2026-08-06/arm_runs_auto_live_confirm2.jsonl'
TH = 8
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
SHIP = {'all': 0.0877, 'sparse': 0.0175, 'moderate': 0.0315,
        'crowded': 0.1860, '2comp': 0.0160}


def ncomp(e):
    return int(re.search(r'_(\d+)comp_', e).group(1))


def sl(e):
    n = ncomp(e)
    return 'sparse' if n <= 2 else ('moderate' if n == 5 else 'crowded')


def read(path, dup_fatal=True):
    out, dups = {}, []
    for line in open(path):
        r = json.loads(line)
        k = (r['episode'], r['est_seed'], r['diff_seed'])
        if k in out:
            dups.append(k)
        out[k] = r
    if dups:
        print(f'!! {len(dups)} DUPLICATE lane rows in {path}: {dups[:5]}')
        if dup_fatal:
            sys.exit('duplicates are fatal (score_confirm2 Sec-2 rule)')
    return out


def main():
    eps = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
    assert len(eps) == 40, len(eps)
    L = read(f'{BASE}/arm_runs_confirm2_v1.jsonl')
    M = read(f'{BASE}/arm_runs_v2m3.jsonl')
    ns = {}
    for line in open(f'{BASE}/atoms_confirm2_v1.jsonl'):
        d = json.loads(line)
        if d.get('err') is None:
            ns[(d['episode'], d['diff_seed'])] = d['n_selected']

    off, route = {}, {}
    for e in eps:
        for es, ds in LANES:
            lrn = ns[(e, ds)] >= TH
            route[(e, es, ds)] = 'learned' if lrn else 'v2_konly'
            src = L if lrn else M
            assert src[(e, es, ds)].get('err') is None
            off[(e, es, ds)] = src[(e, es, ds)]['avg_nRMSE']

    if not os.path.exists(LIVE):
        sys.exit(f'no live file yet: {LIVE}')
    LV = read(LIVE)
    keys = [(e, es, ds) for e in eps for es, ds in LANES]
    miss = [k for k in keys if k not in LV]
    errs = [k for k in keys if k in LV and LV[k].get('err') is not None]
    print(f'live rows {len(LV)} / 200 ; missing {len(miss)} ; err rows {len(errs)}')
    if miss or errs:
        print('  INCOMPLETE -- stopping'); return

    # ---- routing ----
    bad = [(k, route[k], LV[k].get('auto_selected')) for k in keys
           if LV[k].get('auto_selected') != route[k]]
    print(f'\n[R1] routing agreement live vs predicted: {200 - len(bad)}/200')
    for b in bad[:10]:
        print('   ', b)
    print('     live auto_selected census :', Counter(LV[k]['auto_selected'] for k in keys))
    for s in ('sparse', 'moderate', 'crowded'):
        kk = [k for k in keys if sl(k[0]) == s]
        print(f'       {s:9s}', Counter(LV[k]['auto_selected'] for k in kk))
    nbad = [k for k in keys if LV[k].get('auto_n_selected') != ns[(k[0], k[2])]]
    print(f'     n_selected agreement      : {200 - len(nbad)}/200')

    # ---- lane exactness ----
    d = {k: LV[k]['avg_nRMSE'] - off[k] for k in keys}
    for tol in (1e-9, 1e-6, 1e-3):
        print(f'[R2] lanes within {tol:g}: {sum(1 for k in keys if abs(d[k]) <= tol)}/200')
    w = max(keys, key=lambda k: abs(d[k]))
    print(f'     max |lane diff| = {abs(d[w]):.6g} at {w} '
          f'(live {LV[w]["avg_nRMSE"]:.6f} vs off {off[w]:.6f}, route {route[w]})')

    # ---- episode / slice ----
    ep_live = {e: median([LV[(e, es, ds)]['avg_nRMSE'] for es, ds in LANES]) for e in eps}
    ep_off = {e: median([off[(e, es, ds)] for es, ds in LANES]) for e in eps}
    sls = {'all': eps, 'sparse': [e for e in eps if sl(e) == 'sparse'],
           'moderate': [e for e in eps if sl(e) == 'moderate'],
           'crowded': [e for e in eps if sl(e) == 'crowded'],
           '2comp': [e for e in eps if ncomp(e) == 2]}
    print('\n[R3] slice means (lane->episode MEDIAN, episode->slice MEAN)')
    print(f'     {"slice":9s} {"n":>2s} {"live":>9s} {"offline":>9s} {"delta":>9s} {"ship":>7s}')
    out_slices = {}
    for s, v in sls.items():
        a, b = mean([ep_live[e] for e in v]), mean([ep_off[e] for e in v])
        out_slices[s] = {'n': len(v), 'live': a, 'offline': b}
        print(f'     {s:9s} {len(v):2d} {a:9.6f} {b:9.6f} {a - b:+9.6f} {SHIP[s]:7.4f}')

    dd = [ep_live[e] - ep_off[e] for e in eps]
    pos, neg = sum(1 for x in dd if x > 0), sum(1 for x in dd if x < 0)
    n = pos + neg
    p = (min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)
         if n else 1.0)
    print(f'[R4] paired episode delta: median {median(dd):+.3e} '
          f'(+{pos}/-{neg}/={40 - pos - neg}) sign p={p:.4g}; '
          f'floor p at n=40 is 2*0.5**40 = {2 * 0.5 ** 40:.3e}')

    spl = [LV[k]['avg_nRMSE'] for k in keys if sl(k[0]) == 'sparse']
    print(f'[R5] sparse lanes >0.10: {sum(1 for v in spl if v > 0.10)}/{len(spl)} '
          f'(max {max(spl):.4f})')

    json.dump({'slices': out_slices, 'route_mismatch': len(bad),
               'lane_gt_1e-9': sum(1 for k in keys if abs(d[k]) > 1e-9),
               'max_abs_lane_diff': abs(d[w]),
               'median_episode_delta': median(dd), 'sign_pos': pos, 'sign_neg': neg,
               'sparse_lanes_gt_0.10': sum(1 for v in spl if v > 0.10),
               'route_census': dict(Counter(LV[k]['auto_selected'] for k in keys)),
               'ep_live': ep_live, 'ep_off': ep_off},
              open('tmp/E_live_2026-08-06/INDEPENDENT_CHECK.json', 'w'), indent=1)
    print('\n-> tmp/E_live_2026-08-06/INDEPENDENT_CHECK.json')


if __name__ == '__main__':
    main()
