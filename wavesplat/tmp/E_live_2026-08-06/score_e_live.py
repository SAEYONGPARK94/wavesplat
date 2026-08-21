"""E-LIVE -- does the SHIPPED configuration `auto_v1_m3` reproduce the numbers that
authorised it?

The four slice means written into the shipping default's comment (wavesplat/run.py:36-38,
"all 0.0877 / sparse 0.0175 / 5comp 0.0315 / crowded 0.1860 / 2comp 0.0160") are the output
of score_confirm2.py's OFFLINE lane assembly of arm='learned' and arm='v2_m3'.  No row with
arm='auto' exists anywhere in tmp/confirm2_2026-07-27/ (verified: 400/400 rows carry
auto_selected=None).  This script freezes that assembly as a PREDICTION, then scores a live
`--arms auto` run against it.

Usage (from /home/vexit/wavesplat/wavesplat):
    python3 tmp/E_live_2026-08-06/score_e_live.py --freeze
    python3 tmp/E_live_2026-08-06/score_e_live.py --score \
        --live tmp/E_live_2026-08-06/arm_runs_auto_live_confirm2.jsonl

Aggregation convention (frozen BEFORE results, discipline #5):
  lane -> episode = MEDIAN of the 5 lanes; episode -> slice = MEAN (this is exactly the
  convention score_confirm2.py used to produce the shipped comment, so the comparison is
  like-for-like).  Paired episode deltas, two-sided sign test.  n=40 episodes, floor
  p = 2 * 0.5**40 = 1.819e-12.  P1/P2/P4 are DETERMINISTIC -- no p-value applies to them.
"""
import os, sys, json, argparse, re, statistics as st

HERE = 'tmp/E_live_2026-08-06/'
REF = HERE + 'reference_frozen.json'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
TH = 8
ATOMS = 'tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl'
LEARNED = 'tmp/confirm2_2026-07-27/arm_runs_confirm2_v1.jsonl'
V2M3 = 'tmp/confirm2_2026-07-27/arm_runs_v2m3.jsonl'
NULLS = 'tmp/E_depth2_2026-08-06/D2_nulls_confirm2.json'

# tolerances: NOT invented here -- these are the bands already frozen in
# score_confirm2.py (P0/P2 gates: `m <= v + 0.005`, `median(diffs_v2) <= 0.002`).
TOL_EXACT = 1e-9
TOL_MEDIAN = 0.002
TOL_SLICE = 0.005
SPARSE_LANE_CAP = 0.10          # confirm2 P1c safety gate


def comp(e):
    return int(re.search(r'_(\d+)comp', e).group(1))


def slice_of(e):
    n = comp(e)
    return 'sparse' if n <= 2 else ('moderate' if n == 5 else 'crowded')


def ray(e):
    return 'slope' if 'slope' in e else 'uniform'


def load_rows(p):
    o = {}
    for ln in open(p):
        r = json.loads(ln)
        if r.get('err') is None and r.get('avg_nRMSE') is not None:
            o[(r['episode'], r['est_seed'], r['diff_seed'])] = r
    return o


def freeze():
    nsel = {}
    for ln in open(ATOMS):
        d = json.loads(ln)
        if d.get('err') is None:
            nsel[(d['episode'], d['diff_seed'])] = d['n_selected']
    L, M = load_rows(LEARNED), load_rows(V2M3)
    eps = sorted({k[0] for k in L})
    assert len(eps) == 40, len(eps)
    lane_pred, route_pred = {}, {}
    for e in eps:
        for es, ds in LANES:
            r = nsel[(e, ds)] >= TH
            route_pred['|'.join(map(str, (e, es, ds)))] = 'learned' if r else 'v2_konly'
            lane_pred['|'.join(map(str, (e, es, ds)))] = (L if r else M)[(e, es, ds)]['avg_nRMSE']
    ep_med = {e: st.median([lane_pred['|'.join(map(str, (e, es, ds)))] for es, ds in LANES])
              for e in eps}
    slices = {}
    for s in ('all', 'sparse', 'moderate', 'crowded'):
        v = eps if s == 'all' else [e for e in eps if slice_of(e) == s]
        slices[s] = {'n': len(v), 'mean': st.mean([ep_med[e] for e in v])}
    ref = {'note': 'OFFLINE assembly of learned/v2_m3 at th=8 -- the PREDICTION a live '
                   'auto run must reproduce. Frozen before any live run.',
           'lane_pred': lane_pred, 'route_pred': route_pred, 'ep_med': ep_med,
           'slices': slices, 'tol': {'exact': TOL_EXACT, 'median': TOL_MEDIAN,
                                     'slice': TOL_SLICE, 'sparse_cap': SPARSE_LANE_CAP}}
    os.makedirs(HERE, exist_ok=True)
    json.dump(ref, open(REF, 'w'), indent=1)
    print(f'frozen -> {REF}')
    for s, d in slices.items():
        print(f'  {s:9s} n={d["n"]:2d} mean={d["mean"]:.4f}')
    nr = sum(1 for v in route_pred.values() if v == 'learned')
    print(f'  routing prediction: learned {nr}/200, v2_konly {200 - nr}/200')


def signtest_two_sided(pos, neg):
    from math import comb
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def score(live_path):
    ref = json.load(open(REF))
    lane_pred, route_pred, ep_pred = ref['lane_pred'], ref['route_pred'], ref['ep_med']
    LV = load_rows(live_path)
    eps = sorted({k[0] for k in LV})
    keys = [(e, es, ds) for e in sorted(ep_pred) for es, ds in LANES]
    missing = [k for k in keys if k not in LV]
    print(f'live rows {len(LV)} / expected 200 ; missing {len(missing)}')
    if missing:
        print('  INCOMPLETE -- verdict withheld'); return

    # ---- P1 routing fidelity (deterministic, n=200 lanes) ----
    bad_route = []
    for k in keys:
        kk = '|'.join(map(str, k))
        got = LV[k].get('auto_selected')
        if got != route_pred[kk]:
            bad_route.append((kk, route_pred[kk], got))
    P1 = len(bad_route) == 0
    print(f'\nP1 routing agreement : {200 - len(bad_route)}/200  -> {"PASS" if P1 else "FAIL"}')
    for b in bad_route[:8]:
        print('   mismatch', b)

    # ---- P2 lane-exact accuracy (deterministic, n=200 lanes) ----
    diffs = {k: LV[k]['avg_nRMSE'] - lane_pred['|'.join(map(str, k))] for k in keys}
    bad_ex = [k for k in keys if abs(diffs[k]) > TOL_EXACT]
    P2 = len(bad_ex) == 0
    print(f'P2 lane exactness    : {200 - len(bad_ex)}/200 within {TOL_EXACT:g} '
          f'-> {"PASS" if P2 else "FAIL"}')
    if bad_ex:
        worst = max(bad_ex, key=lambda k: abs(diffs[k]))
        print(f'   max |diff| = {abs(diffs[worst]):.6f} at {worst}')

    # ---- P3 statistical fallback (only meaningful if P2 fails) ----
    ep_live = {e: st.median([LV[(e, es, ds)]['avg_nRMSE'] for es, ds in LANES]) for e in eps}
    d = {e: ep_live[e] - ep_pred[e] for e in eps}
    pos = sum(1 for v in d.values() if v > 0)
    neg = sum(1 for v in d.values() if v < 0)
    med = st.median(list(d.values()))
    p = signtest_two_sided(pos, neg)
    print(f'P3 paired episode delta (live - assembled): median={med:+.5f} '
          f'(+{pos}/-{neg}/={40 - pos - neg}) two-sided sign p={p:.4g} '
          f'[floor p at n=40 = 1.819e-12]')
    sl_ok = True
    print('   slice means (live vs frozen prediction vs shipping comment):')
    ship = {'all': 0.0877, 'sparse': 0.0175, 'moderate': 0.0315, 'crowded': 0.1860}
    for s in ('all', 'sparse', 'moderate', 'crowded'):
        v = eps if s == 'all' else [e for e in eps if slice_of(e) == s]
        mlive, mref = st.mean([ep_live[e] for e in v]), ref['slices'][s]['mean']
        ok = abs(mlive - mref) <= TOL_SLICE
        sl_ok &= ok
        print(f'     {s:9s} n={len(v):2d} live={mlive:.4f} ref={mref:.4f} '
              f'ship={ship[s]:.4f} d={mlive - mref:+.4f} {"ok" if ok else "OUT"}')
    P3 = (abs(med) <= TOL_MEDIAN) and sl_ok
    print(f'   P3 equivalence band -> {"PASS" if P3 else "FAIL"}')

    # ---- P4 sparse safety gate (deterministic, n=80 lanes) ----
    spl = [LV[k]['avg_nRMSE'] for k in keys if slice_of(k[0]) == 'sparse']
    viol = [v for v in spl if v > SPARSE_LANE_CAP]
    P4 = len(viol) == 0
    print(f'P4 sparse lane cap   : {len(viol)}/{len(spl)} lanes > {SPARSE_LANE_CAP} '
          f'(max {max(spl):.4f}) -> {"PASS" if P4 else "FAIL"}')

    # ---- mandatory reporting context (not pass lines) ----
    if os.path.exists(NULLS):
        nulls = json.load(open(NULLS))
        cr = [e for e in eps if slice_of(e) == 'crowded']
        w = sum(1 for e in cr if ep_live[e] < nulls[e])
        print(f'\n[context] crowded beats flat-sea null: {w}/{len(cr)} '
              f'(median margin {st.median([ep_live[e] - nulls[e] for e in cr]):+.4f})')
    for r_ in ('slope', 'uniform'):
        v = [e for e in eps if slice_of(e) == 'crowded' and ray(e) == r_]
        print(f'[context] crowded/{r_:7s} n={len(v)} live mean={st.mean([ep_live[e] for e in v]):.4f}')

    verdict = ('PASS' if (P1 and (P2 or P3) and P4) else 'FAIL')
    print(f'\n=== E-LIVE VERDICT: {verdict} ===')
    print(f'    P1={P1} P2={P2} P3={P3} P4={P4}')
    json.dump({'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'verdict': verdict,
               'median_delta': med, 'sign_p': p, 'bad_route': bad_route[:20],
               'ep_live': ep_live},
              open(HERE + 'VERDICT_E_LIVE.json', 'w'), indent=1)
    print(f'    -> {HERE}VERDICT_E_LIVE.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--live', default=HERE + 'arm_runs_auto_live_confirm2.jsonl')
    a = ap.parse_args()
    if a.freeze:
        freeze()
    elif a.score:
        score(a.live)
    else:
        ap.error('need --freeze or --score')
