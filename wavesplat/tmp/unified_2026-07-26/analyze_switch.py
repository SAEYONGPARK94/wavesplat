"""UNIFIED Part-B: 100ep switch scoreboard + non-inferiority + sensitivity + misroutes.

Sources (all reused except extra-48 learned):
  pure-v2      : tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl (100ep x 5 est_seeds)
  pure-learned : Stage-1 arm_runs.jsonl learned (52) + arm_runs_extra.jsonl (48)
  n_selected   : Stage-1 atoms_cache.jsonl (52) + atoms_cache_extra.jsonl (48)
  switch       : per-lane assembly (PRIMARY, faithful to live init_method='auto')
                 + per-episode-median-signal variant (secondary)

Frozen protocol: tmp/unified_2026-07-26/PREREG.md.  Writes switch_results.json.
"""
import json
import numpy as np
from scipy.stats import wilcoxon

BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
S1_ARMS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
S1_CACHE = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'
X_ARMS = 'tmp/unified_2026-07-26/arm_runs_extra.jsonl'
X_CACHE = 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def comp(ep):
    import re
    return int(re.search(r'_(\d+)comp', ep).group(1))


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


def load():
    # pure-v2: batchC keyed (episode, est_seed)
    v2 = {}
    eps_all = set()
    for l in open(BATCHC):
        d = json.loads(l)
        v2[(d['episode'], d['seed'])] = d['nRMSE']
        eps_all.add(d['episode'])
    # learned lane values keyed (episode, est_seed, diff_seed)
    learned = {}
    for src in (S1_ARMS, X_ARMS):
        for l in open(src):
            d = json.loads(l)
            if d.get('arm') == 'learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
                learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
    # n_selected keyed (episode, diff_seed)
    nsel = {}
    for src in (S1_CACHE, X_CACHE):
        for l in open(src):
            d = json.loads(l)
            if d.get('n_selected') is not None:
                nsel[(d['episode'], d['diff_seed'])] = d['n_selected']
    return sorted(eps_all), v2, learned, nsel


def episode_arms(eps, v2, learned, nsel, thresh):
    """Return per-episode dicts: pure_v2, pure_learned, switch_perlane, switch_perep, oracle,
    plus per-lane pick counts + per-episode median n_sel."""
    out = {}
    for ep in eps:
        v2_lane = [v2.get((ep, E)) for (E, D) in LANES]
        ln_lane = [learned.get((ep, E, D)) for (E, D) in LANES]
        ns_lane = [nsel.get((ep, D)) for (E, D) in LANES]
        pure_v2 = med(v2_lane)
        pure_learned = med(ln_lane)
        # per-lane switch
        sw_perlane = []
        picks = []
        for (E, D), vv, lv, ns in zip(LANES, v2_lane, ln_lane, ns_lane):
            pick = 'learned' if (ns is not None and ns >= thresh) else 'v2'
            picks.append(pick)
            sw_perlane.append(lv if pick == 'learned' else vv)
        switch_perlane = med(sw_perlane)
        # per-episode switch (median n_sel signal)
        ns_med = med([n for n in ns_lane if n is not None])
        pick_ep = 'learned' if (np.isfinite(ns_med) and ns_med >= thresh) else 'v2'
        switch_perep = pure_learned if pick_ep == 'learned' else pure_v2
        oracle = min(pure_v2, pure_learned)
        out[ep] = dict(pure_v2=pure_v2, pure_learned=pure_learned,
                       switch_perlane=switch_perlane, switch_perep=switch_perep,
                       oracle=oracle, ns_med=ns_med, pick_ep=pick_ep,
                       n_learned_lanes=sum(p == 'learned' for p in picks), comp=comp(ep))
    return out


def paired(a, b):
    """two-sided paired Wilcoxon on arrays a,b (a=test, b=ref). Δ = a-b."""
    a = np.array(a, float); b = np.array(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    try:
        p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p = float('nan')
    return dict(n=int(len(d)), mean_test=float(a.mean()), mean_ref=float(b.mean()),
                median_delta=float(np.median(d)), mean_delta=float(d.mean()),
                wins_test=int((d < 0).sum()), p=p)


def noninf(res, key_test, key_ref, subset):
    """non-inferiority: PASS iff NOT (median Δ>0 AND p<0.05). Δ=test-ref."""
    a = [res[e][key_test] for e in subset]
    b = [res[e][key_ref] for e in subset]
    st = paired(a, b)
    worse_sig = (st['median_delta'] > 0) and (st['p'] < 0.05)
    st['noninf_pass'] = bool(not worse_sig)
    return st


def main():
    eps, v2, learned, nsel = load()
    assert len(eps) == 100, len(eps)
    CROWDED = [e for e in eps if comp(e) in (15, 30)]
    SPARSE = [e for e in eps if comp(e) in (1, 2)]
    MODERATE = [e for e in eps if comp(e) == 5]

    report = {}
    for thresh in (6, 8, 10):
        res = episode_arms(eps, v2, learned, nsel, thresh)
        agg = lambda k, S=eps: float(np.nanmean([res[e][k] for e in S]))
        means = dict(
            pure_v2=agg('pure_v2'), pure_learned=agg('pure_learned'),
            switch_perlane=agg('switch_perlane'), switch_perep=agg('switch_perep'),
            oracle=agg('oracle'))
        best_pure = min(means['pure_v2'], means['pure_learned'])
        agg_pass = means['switch_perlane'] <= best_pure
        # non-inf slices (primary = per-lane switch)
        crowded_ni = noninf(res, 'switch_perlane', 'pure_learned', CROWDED)
        sparse_ni = noninf(res, 'switch_perlane', 'pure_v2', SPARSE)
        decision = bool(agg_pass and crowded_ni['noninf_pass'] and sparse_ni['noninf_pass'])
        # misroutes (per-episode route vs oracle)
        misroutes = []
        for e in eps:
            r = res[e]
            routed = r['pure_learned'] if r['pick_ep'] == 'learned' else r['pure_v2']
            cost = routed - r['oracle']
            if cost > 1e-9:
                misroutes.append(dict(episode=e, comp=r['comp'], pick=r['pick_ep'],
                                      ns_med=r['ns_med'], routed=routed,
                                      oracle=r['oracle'], cost=cost,
                                      pure_v2=r['pure_v2'], pure_learned=r['pure_learned']))
        misroutes.sort(key=lambda x: -x['cost'])
        # slice means
        slice_means = {}
        for name, S in (('crowded', CROWDED), ('sparse', SPARSE), ('moderate', MODERATE)):
            slice_means[name] = dict(
                n=len(S), pure_v2=agg('pure_v2', S), pure_learned=agg('pure_learned', S),
                switch_perlane=agg('switch_perlane', S), oracle=agg('oracle', S))
        # signal distribution
        ns_meds = [res[e]['ns_med'] for e in eps]
        near_boundary = sum(1 for e in eps if 6 <= (res[e]['ns_med'] or -1) <= 10)
        report[thresh] = dict(
            means=means, best_pure=best_pure, agg_pass=bool(agg_pass),
            crowded_noninf=crowded_ni, sparse_noninf=sparse_ni, decision=decision,
            n_misroutes=len(misroutes), misroute_total_cost=float(sum(m['cost'] for m in misroutes)),
            misroutes=misroutes[:20], slice_means=slice_means,
            near_boundary=int(near_boundary))

    # signal distribution (threshold-independent)
    res8 = episode_arms(eps, v2, learned, nsel, 8)
    dist = {name: sorted([round(res8[e]['ns_med'], 1) for e in S])
            for name, S in (('crowded', CROWDED), ('sparse', SPARSE), ('moderate', MODERATE))}

    out = dict(n_episodes=len(eps), n_crowded=len(CROWDED), n_sparse=len(SPARSE),
               n_moderate=len(MODERATE), by_threshold=report, ns_med_distribution=dist)
    json.dump(out, open('tmp/unified_2026-07-26/switch_results.json', 'w'), indent=1)

    # print
    for thresh in (8, 6, 10):
        R = report[thresh]
        print(f"\n{'='*70}\nTHRESHOLD n_selected >= {thresh}\n{'='*70}")
        m = R['means']
        print(f"  100ep MEANS: pure_v2={m['pure_v2']:.4f}  pure_learned={m['pure_learned']:.4f}"
              f"  switch(perlane)={m['switch_perlane']:.4f}  switch(perep)={m['switch_perep']:.4f}"
              f"  oracle={m['oracle']:.4f}")
        print(f"  best_pure={R['best_pure']:.4f}  -> AGG_PASS={R['agg_pass']}")
        c = R['crowded_noninf']; s = R['sparse_noninf']
        print(f"  crowded non-inf (switch vs pure_learned, n={c['n']}): medΔ={c['median_delta']:+.4f}"
              f" p={c['p']:.3g} PASS={c['noninf_pass']}")
        print(f"  sparse  non-inf (switch vs pure_v2,      n={s['n']}): medΔ={s['median_delta']:+.4f}"
              f" p={s['p']:.3g} PASS={s['noninf_pass']}")
        print(f"  DECISION = {R['decision']}")
        print(f"  misroutes={R['n_misroutes']} total_cost={R['misroute_total_cost']:.4f}"
              f"  near_boundary(6..10)={R['near_boundary']}")
        for name in ('crowded', 'sparse', 'moderate'):
            sm = R['slice_means'][name]
            print(f"    [{name:8s} n{sm['n']:2d}] v2={sm['pure_v2']:.3f} learned={sm['pure_learned']:.3f}"
                  f" switch={sm['switch_perlane']:.3f} oracle={sm['oracle']:.3f}")
        if thresh == 8 and R['misroutes']:
            print("  top misroutes (thresh=8):")
            for mr in R['misroutes'][:12]:
                print(f"    {mr['episode']:38s} c{mr['comp']:2d} pick={mr['pick']:7s}"
                      f" ns={mr['ns_med']:.1f} routed={mr['routed']:.3f} oracle={mr['oracle']:.3f}"
                      f" cost=+{mr['cost']:.3f}")
    print("\nns_med distribution:")
    for k, v in dist.items():
        print(f"  {k:8s}: {v}")


if __name__ == '__main__':
    main()
