"""TASK 1 scoring: guard-auto 100ep re-assembly + frozen primary criteria + sensitivity.

Guard flips are learned->v2 only; pure-v2 lane values exist (batchC), so the guard-auto score
is a pure re-assembly. bf_peak_ratio is computed via the FROZEN beamform_peak_ratio helper
(cached to guard_bf_ratio.json). Frozen protocol: tmp/unified_guard_2026-07-26/PREREG.md.
"""
import os, json, re, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np

FINE = 'data_finemesh_2026-07-23/'
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
S1_ARMS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
X_ARMS = 'tmp/unified_2026-07-26/arm_runs_extra.jsonl'
S1_CACHE = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'
X_CACHE = 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'
BF_JSON = 'tmp/unified_guard_2026-07-26/guard_bf_ratio.json'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
THRESH = 8


def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


def bf_ratios(eps):
    if os.path.exists(BF_JSON):
        return json.load(open(BF_JSON))
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.initializer import beamform_peak_ratio
    out = {}
    for i, ep in enumerate(eps):
        L = EpisodeDataLoader(FINE + ep)
        out[ep] = float(beamform_peak_ratio(L.get_init_window(2.0), L.depth))
        if (i + 1) % 20 == 0:
            print(f"  bf_ratio {i+1}/{len(eps)}", flush=True)
    json.dump(out, open(BF_JSON, 'w'), indent=0)
    return out


def load():
    v2, eps = {}, set()
    for l in open(BATCHC):
        d = json.loads(l); v2[(d['episode'], d['seed'])] = d['nRMSE']; eps.add(d['episode'])
    learned = {}
    for src in (S1_ARMS, X_ARMS):
        for l in open(src):
            d = json.loads(l)
            if d.get('arm') == 'learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
                learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
    nsel = {}
    for src in (S1_CACHE, X_CACHE):
        for l in open(src):
            d = json.loads(l)
            if d.get('n_selected') is not None:
                nsel[(d['episode'], d['diff_seed'])] = d['n_selected']
    return sorted(eps), v2, learned, nsel


def assemble(eps, v2, learned, nsel, bfr, guard_ratio):
    """Per-episode median-of-5 for: pure_v2, pure_learned, auto(no guard), guard-auto, oracle.
    Also record per-lane crowded flips vs no-guard."""
    res = {}
    crowded_flips = 0
    for ep in eps:
        v2l = [v2.get((ep, E)) for (E, D) in LANES]
        lnl = [learned.get((ep, E, D)) for (E, D) in LANES]
        nsl = [nsel.get((ep, D)) for (E, D) in LANES]
        auto_lane, guard_lane = [], []
        for (E, D), vv, lv, ns in zip(LANES, v2l, lnl, nsl):
            base = 'learned' if (ns is not None and ns >= THRESH) else 'v2'
            g = base
            if base == 'learned' and guard_ratio is not None and bfr.get(ep, 1.0) < guard_ratio:
                g = 'v2'
            auto_lane.append(lv if base == 'learned' else vv)
            guard_lane.append(lv if g == 'learned' else vv)
            if comp(ep) in (15, 30) and g != base:
                crowded_flips += 1
        pv, pl = med(v2l), med(lnl)
        res[ep] = dict(pure_v2=pv, pure_learned=pl, auto=med(auto_lane),
                       guard=med(guard_lane), oracle=min(pv, pl), comp=comp(ep),
                       bfr=bfr.get(ep))
    return res, crowded_flips


def main():
    eps, v2, learned, nsel = load()
    assert len(eps) == 100
    bfr = bf_ratios(eps)
    SPARSE = [e for e in eps if comp(e) in (1, 2)]
    CROWDED = [e for e in eps if comp(e) in (15, 30)]
    MOD = [e for e in eps if comp(e) == 5]
    agg = lambda res, k, S=eps: float(np.nanmean([res[e][k] for e in S]))

    out = {}
    for gr in (None, 0.04, 0.05, 0.055):
        res, flips = assemble(eps, v2, learned, nsel, bfr, gr)
        rec = dict(guard_ratio=gr, crowded_flips=flips,
                   mean_100=agg(res, 'guard' if gr else 'auto'),
                   sparse=agg(res, 'guard' if gr else 'auto', SPARSE),
                   crowded=agg(res, 'guard' if gr else 'auto', CROWDED),
                   moderate=agg(res, 'guard' if gr else 'auto', MOD))
        out[str(gr)] = rec

    base = out['None']
    pv2_sparse = agg(assemble(eps, v2, learned, nsel, bfr, None)[0], 'pure_v2', SPARSE)
    pl_sparse = agg(assemble(eps, v2, learned, nsel, bfr, None)[0], 'pure_learned', SPARSE)
    prim = out['0.05']
    c1 = prim['sparse'] <= pv2_sparse + 0.005
    c2 = prim['crowded_flips'] == 0
    c3 = prim['mean_100'] < base['mean_100']  # < current auto 0.0979
    decision = bool(c1 and c2 and c3)

    summary = dict(pure_v2_sparse=pv2_sparse, pure_learned_sparse=pl_sparse,
                   current_auto_100=base['mean_100'], current_auto_sparse=base['sparse'],
                   sparse_target=pv2_sparse + 0.005, by_guard_ratio=out,
                   primary=dict(c1_sparse_within_0005=bool(c1), sparse_val=prim['sparse'],
                                c2_zero_crowded_flips=bool(c2), flips=prim['crowded_flips'],
                                c3_agg_improves=bool(c3), agg_val=prim['mean_100'],
                                agg_baseline=base['mean_100'], DECISION=decision))
    json.dump(summary, open('tmp/unified_guard_2026-07-26/task1_results.json', 'w'), indent=1)

    print("\n" + "="*78)
    print("TASK 1 — guard-auto re-assembly (median-of-5, per-lane routing)")
    print("="*78)
    print(f"  pure-v2 sparse={pv2_sparse:.4f}  pure-learned sparse={pl_sparse:.4f}  "
          f"sparse target (<=v2+0.005)={pv2_sparse+0.005:.4f}")
    print(f"  current auto (no guard): 100ep={base['mean_100']:.4f} sparse={base['sparse']:.4f} "
          f"crowded={base['crowded']:.4f} mod={base['moderate']:.4f}")
    print(f"\n  {'guard_ratio':>11} | {'100ep':>7} | {'sparse':>7} | {'crowded':>7} | {'mod':>6} | flips")
    for gr in ('0.04', '0.05', '0.055'):
        r = out[gr]
        print(f"  {gr:>11} | {r['mean_100']:.4f} | {r['sparse']:.4f} | {r['crowded']:.4f} | "
              f"{r['moderate']:.4f} | {r['crowded_flips']}")
    p = summary['primary']
    print("\n  FROZEN PRIMARY (guard_ratio=0.05):")
    print(f"   [1] sparse {p['sparse_val']:.4f} <= v2+0.005 {pv2_sparse+0.005:.4f} : {p['c1_sparse_within_0005']}")
    print(f"   [2] crowded flips == 0                       : {p['c2_zero_crowded_flips']} (flips={p['flips']})")
    print(f"   [3] 100ep {p['agg_val']:.4f} < current auto {p['agg_baseline']:.4f} : {p['c3_agg_improves']}")
    print(f"   DECISION = {p['DECISION']}")


if __name__ == '__main__':
    main()
