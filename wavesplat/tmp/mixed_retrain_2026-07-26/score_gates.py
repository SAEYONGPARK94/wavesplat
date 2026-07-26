"""MIXED-RETRAIN gate scorer.  Bars are frozen in tmp/mixed_retrain_2026-07-26/PREREG.md Sec 5.

Scoring conventions are the ones the source protocols froze:
  * per (episode, arm): MEDIAN over the 5 paired lanes (42,0)(1,1)(2,2)(3,3)(4,4)
  * then MEAN over episodes
  * n_selected: per-episode MEDIAN over the 5 diffusion seeds; the gate statistic is the MAX of
    those per-episode medians (PREREG Sec 5, G2a)

  python3 tmp/mixed_retrain_2026-07-26/score_gates.py --gates G1 G2
"""
import argparse
import collections
import json
import os

import numpy as np
from scipy import stats as sps

HERE = 'tmp/mixed_retrain_2026-07-26/'
VER = 'v1'   # set from --ver

# ── frozen bars and old-net references (PREREG Sec 5) ────────────────────────
BAR_G1 = 0.17878          # old-net crowded 0.16878 + 0.010
OLD_G1 = 0.16878
BAR_G2A = 6               # max over the 12 one-comp episodes of per-episode median n_selected
OLD_G2A = 30
BAR_G2B = 0.01855         # pure-v2 one-comp mean 0.01355 + 0.005
OLD_G2B = 0.03567
V2_G2 = 0.01355
OLD_G3_MEAN, OLD_G3_MED, OLD_G3_NS = 0.2330, 0.2375, 22.5
OLD_G4_LEARNED_27 = 0.805
OLD_G4_HAND_27 = 0.744


def med(xs):
    xs = [v for v in xs if v is not None and np.isfinite(v)]
    return float(np.median(xs)) if xs else float('nan')


def load_arms(path):
    d = collections.defaultdict(list)
    if not os.path.exists(path):
        return d
    for line in open(path):
        r = json.loads(line)
        if r.get('err') is None and r.get('avg_nRMSE') is not None:
            d[r['episode']].append(float(r['avg_nRMSE']))
    return d


def load_ns(path):
    d = collections.defaultdict(list)
    if not os.path.exists(path):
        return d
    for line in open(path):
        r = json.loads(line)
        if r.get('err') is None and r.get('n_selected') is not None:
            d[r['episode']].append(int(r['n_selected']))
    return d


def gate_G1(out):
    arms = load_arms(HERE + f'arm_runs_mixed_{VER}.jsonl')
    eps = sorted(e for e in arms if '15comp' in e or '30comp' in e)
    per = {e: med(arms[e]) for e in eps}
    mean = float(np.mean(list(per.values())))
    old = json.loads(open('tmp/heldout40_2026-07-26/score_results.json').read())['per_episode']
    oldper = {e: old[e]['pure_learned'] for e in eps}
    v2per = {e: old[e]['pure_v2'] for e in eps}
    delta = np.array([per[e] - oldper[e] for e in eps])
    W = sps.wilcoxon(delta, alternative='two-sided') if np.any(delta != 0) else None
    r = dict(gate='G1', n=len(eps), new_mean=mean, old_mean=float(np.mean(list(oldper.values()))),
             v2_mean=float(np.mean(list(v2per.values()))),
             bar=BAR_G1, verdict='PASS' if mean <= BAR_G1 else 'FAIL',
             margin=float(BAR_G1 - mean),
             median_delta_vs_old=float(np.median(delta)),
             wins_vs_old=int(np.sum(delta < 0)),
             p_two_sided=float(W.pvalue) if W else None,
             per_episode={e: dict(new=per[e], old=oldper[e], v2=v2per[e]) for e in eps})
    out['G1'] = r
    print(f"\n=== G1 do-not-forget (crowded, n={r['n']}) ===")
    print(f"  {'episode':<44s} {'new':>8s} {'old':>8s} {'v2':>8s} {'delta':>8s}")
    for e in eps:
        print(f"  {e:<44s} {per[e]:8.4f} {oldper[e]:8.4f} {v2per[e]:8.4f} "
              f"{per[e]-oldper[e]:+8.4f}")
    print(f"  NEW mean {mean:.5f} | OLD {r['old_mean']:.5f} | v2 {r['v2_mean']:.5f}")
    print(f"  BAR <= {BAR_G1:.5f}  ->  {r['verdict']}  (margin {r['margin']:+.5f})")
    print(f"  vs old: medianD {r['median_delta_vs_old']:+.5f}, wins {r['wins_vs_old']}/{r['n']}, "
          f"p={r['p_two_sided']}")


def gate_G2(out):
    arms = load_arms(HERE + f'arm_runs_mixed_{VER}.jsonl')
    ns = load_ns(HERE + f'atoms_cache_mixed_{VER}.jsonl')
    eps = sorted(e for e in ns if '_1comp_' in e)
    ns_med = {e: med(ns[e]) for e in eps}
    g2a = float(np.max(list(ns_med.values())))
    per = {e: med(arms[e]) for e in eps if arms.get(e)}
    old = json.loads(open('tmp/heldout40_2026-07-26/score_results.json').read())['per_episode']
    oldper = {e: old[e]['pure_learned'] for e in eps}
    v2per = {e: old[e]['pure_v2'] for e in eps}
    oldns = {e: old[e]['ns_med'] for e in eps}
    g2b = float(np.mean([per[e] for e in eps])) if len(per) == len(eps) else float('nan')
    va = 'PASS' if g2a <= BAR_G2A else 'FAIL'
    vb = 'PASS' if g2b <= BAR_G2B else 'FAIL'
    r = dict(gate='G2', n=len(eps),
             G2a_max_ns=g2a, G2a_bar=BAR_G2A, G2a_verdict=va, G2a_old=OLD_G2A,
             G2a_pooled_median=float(np.median([v for e in eps for v in ns[e]])),
             G2b_mean=g2b, G2b_bar=BAR_G2B, G2b_verdict=vb,
             G2b_old=float(np.mean(list(oldper.values()))),
             v2_mean=float(np.mean(list(v2per.values()))),
             verdict='PASS' if va == 'PASS' and vb == 'PASS' else 'FAIL',
             per_episode={e: dict(new=per.get(e), old=oldper[e], v2=v2per[e],
                                  new_ns=ns_med[e], old_ns=oldns[e],
                                  new_ns_all=sorted(ns[e])) for e in eps})
    out['G2'] = r
    print(f"\n=== G2 OOD fix (1-comp, n={r['n']}) ===")
    print(f"  {'episode':<44s} {'ns_new':>7s} {'ns_old':>7s} {'new':>8s} {'old':>8s} {'v2':>8s}")
    for e in eps:
        print(f"  {e:<44s} {ns_med[e]:7.1f} {oldns[e]:7.1f} "
              f"{(per.get(e) if per.get(e) is not None else float('nan')):8.4f} "
              f"{oldper[e]:8.4f} {v2per[e]:8.4f}")
    print(f"  G2a max per-episode median n_selected = {g2a:.1f} (old {OLD_G2A}) "
          f"BAR <= {BAR_G2A}  ->  {va}")
    print(f"  G2b learned mean = {g2b:.5f} (old {r['G2b_old']:.5f}, v2 {r['v2_mean']:.5f}) "
          f"BAR <= {BAR_G2B:.5f}  ->  {vb}")
    print(f"  G2 OVERALL: {r['verdict']}")


def gate_G3(out):
    def load(p, key='cond_polish_nrmse_Hs'):
        d = collections.defaultdict(list)
        for line in open(p):
            r = json.loads(line)
            if r.get('err') is None and r.get(key) is not None and r.get('cell') == 'c':
                d[r['episode']].append(float(r[key]))
        return d
    new = load(HERE + f'cond_runs_mixed_{VER}.jsonl')
    old = load('tmp/discreteness_2026-07-26/cond_runs.jsonl')
    eps = sorted(set(new) & set(old))
    n = {e: med(new[e]) for e in eps}
    o = {e: med(old[e]) for e in eps}
    d = np.array([n[e] - o[e] for e in eps])
    W = sps.wilcoxon(d, alternative='two-sided') if len(d) and np.any(d != 0) else None
    r = dict(gate='G3', n=len(eps), new_mean=float(np.mean(list(n.values()))),
             new_median=float(np.median(list(n.values()))),
             old_mean=float(np.mean(list(o.values()))), old_mean_ref=OLD_G3_MEAN,
             median_delta=float(np.median(d)), wins=int(np.sum(d < 0)),
             p_two_sided=float(W.pvalue) if W else None,
             per_episode={e: dict(new=n[e], old=o[e]) for e in eps},
             note='report-only, no bar (PREREG Sec 0 / Sec 5 G3)')
    out['G3'] = r
    print(f"\n=== G3 continuum cell_c (n={r['n']}, REPORT ONLY) ===")
    for e in eps:
        print(f"  {e:<44s} new {n[e]:.4f}  old {o[e]:.4f}  d {n[e]-o[e]:+.4f}")
    print(f"  NEW mean {r['new_mean']:.4f} | OLD mean {r['old_mean']:.4f} "
          f"(recorded ref {OLD_G3_MEAN})")
    print(f"  medianD {r['median_delta']:+.4f}, new-better {r['wins']}/{r['n']}, "
          f"p={r['p_two_sided']}")


def gate_G4(out):
    r = dict(gate='G4', note='report-only (PREREG Sec 5 G4)')
    # ---- accuracy: 27 NORMAL windows ----
    pn = HERE + f'wp_real_mixed_{VER}.json'
    if os.path.exists(pn):
        new = json.loads(open(pn).read())['results']
        old = json.loads(open('tmp/wp_real_2026-07-25/wp_real.json').read())['results']
        key = lambda w: (w['record'], w['win'])
        on = {key(w): w for w in old}
        rows = []
        for w in new:
            k = key(w)
            if k not in on:
                continue
            rows.append(dict(rec=k[0], win=k[1], collapse=bool(w.get('collapse')),
                             new=w.get('learned_now'), old=on[k].get('learned_now'),
                             hand=on[k].get('hand_now')))
        norm = [x for x in rows if not x['collapse']
                and x['new'] is not None and x['old'] is not None]
        d = np.array([x['new'] - x['old'] for x in norm])
        W = sps.wilcoxon(d, alternative='two-sided') if len(d) and np.any(d != 0) else None
        r['accuracy'] = dict(
            n_normal=len(norm), new_mean=float(np.mean([x['new'] for x in norm])),
            old_mean=float(np.mean([x['old'] for x in norm])), old_ref=OLD_G4_LEARNED_27,
            hand_mean=float(np.mean([x['hand'] for x in norm])), hand_ref=OLD_G4_HAND_27,
            median_delta_new_minus_old=float(np.median(d)),
            new_better=int(np.sum(d < 0)), p_two_sided=float(W.pvalue) if W else None,
            new_better_than_hand=int(np.sum([x['new'] < x['hand'] for x in norm])),
            per_window=rows)
        a = r['accuracy']
        print(f"\n=== G4 real transfer, ACCURACY (normal windows n={a['n_normal']}, "
              f"REPORT ONLY) ===")
        print(f"  NEW learned {a['new_mean']:.3f} | OLD learned {a['old_mean']:.3f} "
              f"(ref {OLD_G4_LEARNED_27}) | hand {a['hand_mean']:.3f} (ref {OLD_G4_HAND_27})")
        print(f"  medianD(new-old) {a['median_delta_new_minus_old']:+.4f}, "
              f"new better {a['new_better']}/{a['n_normal']}, p={a['p_two_sided']}")
        print(f"  new beats hand on {a['new_better_than_hand']}/{a['n_normal']}")
    # ---- collapse: 18 harvested windows ----
    pc = HERE + f'collapse_learned_mixed_{VER}.json'
    po = 'tmp/collapse_controls_2026-07-26/learned_arm.json'
    if os.path.exists(pc) and os.path.exists(po):
        def rows_of(p):
            j = json.loads(open(p).read())
            rs = j['rows'] if isinstance(j, dict) and 'rows' in j else j
            return {(x['record'], x['win']): x for x in rs}
        nn, oo = rows_of(pc), rows_of(po)
        esc = rows_of('tmp/collapse_controls_2026-07-26/escapes.json')
        HELD = {('BS11b', 4), ('BS11c', 3), ('BS11c', 4), ('BS13a', 0), ('BS13a', 1),
                ('BS13a', 2), ('BS13a', 4), ('BS13a', 5), ('BS13c', 2), ('BS13c', 3),
                ('LJ18n', 0), ('LJ18n', 1), ('LJ18n', 3), ('LJ18n', 4), ('LJ18n', 5)}
        det = []
        for k in sorted(set(nn) & set(oo), key=str):
            a, b = nn[k], oo[k]
            e = esc.get(k, {})
            base = e.get('baseline_recheck')          # the analytic arm on this window
            e2d = (e.get('E2') or {}).get('kp_x2.0')  # the GT-free analytic escape control
            det.append(dict(rec=k[0], win=k[1], heldout=k in HELD,
                            new=a.get('learned_now'), old=b.get('learned_now'),
                            baseline=base, e2_double=e2d))
        ho = [x for x in det if x['heldout'] and None not in
              (x['new'], x['old'], x['baseline'])]
        allw = [x for x in det if None not in (x['new'], x['old'], x['baseline'])]
        r['collapse'] = dict(
            n_heldout=len(ho),
            new_repairs=int(sum(x['new'] < x['baseline'] for x in ho)),
            new_damages=int(sum(x['new'] > x['baseline'] for x in ho)),
            old_repairs=int(sum(x['old'] < x['baseline'] for x in ho)),
            old_damages=int(sum(x['old'] > x['baseline'] for x in ho)),
            n_all=len(allw),
            new_below_1=int(sum(x['new'] < 1.0 for x in allw)),
            old_below_1=int(sum(x['old'] < 1.0 for x in allw)),
            new_better_than_old=int(sum(x['new'] < x['old'] for x in allw)),
            per_window=det)
        c = r['collapse']
        print(f"\n=== G4 real transfer, COLLAPSE ({c['n_all']} windows, REPORT ONLY) ===")
        print(f"  held-out {c['n_heldout']}: NEW repairs {c['new_repairs']} / damages "
              f"{c['new_damages']}  |  OLD repairs {c['old_repairs']} / damages "
              f"{c['old_damages']} (recorded 8/7)")
        print(f"  all {c['n_all']}: below nRMSE 1.0 -- NEW {c['new_below_1']}, "
              f"OLD {c['old_below_1']} (recorded 6/18; analytic escape 11/18)")
        print(f"  NEW better than OLD on {c['new_better_than_old']}/{c['n_all']}")
    out['G4'] = r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gates', nargs='+', default=['G1', 'G2', 'G3', 'G4'])
    ap.add_argument('--ver', default='v1')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    VER = a.ver
    a.out = a.out or HERE + f'gate_results_{VER}.json'
    res = {}
    if os.path.exists(a.out):
        res = json.loads(open(a.out).read())
    for g in a.gates:
        try:
            globals()['gate_' + g](res)
        except Exception as e:
            print(f"\n!! {g} scoring failed: {e!r}")
            raise
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print(f"\nwrote {a.out}")
