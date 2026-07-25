"""TASK 2 scoring: merged-learned sparse parity + crowded no-harm. Frozen tiers/tests in
tmp/unified_guard_2026-07-26/PREREG.md. Reads merge_arms.jsonl + reused arm/v2 jsonls."""
import json, re
import numpy as np
from scipy.stats import wilcoxon

BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
S1_ARMS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
X_ARMS = 'tmp/unified_2026-07-26/arm_runs_extra.jsonl'
MERGE = 'tmp/unified_guard_2026-07-26/merge_arms.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
VARIANTS = [0.25, 0.375]
CROWDED_SPOT = ([f'episode_BF4_15comp_seed{s}_uniform.h5' for s in range(5)] +
                [f'episode_BF4_30comp_seed{s}_uniform.h5' for s in range(5)])


def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


def load():
    v2 = {}; eps = set()
    for l in open(BATCHC):
        d = json.loads(l); v2[(d['episode'], d['seed'])] = d['nRMSE']; eps.add(d['episode'])
    learned = {}
    for src in (S1_ARMS, X_ARMS):
        for l in open(src):
            d = json.loads(l)
            if d.get('arm') == 'learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
                learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
    merged = {}; mn = {}
    for l in open(MERGE):
        d = json.loads(l)
        if d.get('err') is None and d.get('avg_nRMSE') is not None:
            merged[(d['episode'], d['est_seed'], d['diff_seed'], d['merge_frac'])] = d['avg_nRMSE']
            mn[(d['episode'], d['est_seed'], d['diff_seed'], d['merge_frac'])] = d['final_n']
    return sorted(eps), v2, learned, merged, mn


def ep_med(d, ep, kind, mf=None):
    if kind == 'v2':
        return med([d.get((ep, E)) for (E, D) in LANES])
    if kind == 'learned':
        return med([d.get((ep, E, D)) for (E, D) in LANES])
    return med([d.get((ep, E, D, mf)) for (E, D) in LANES])


def main():
    eps, v2, learned, merged, mn = load()
    SPARSE = sorted([e for e in eps if comp(e) in (1, 2)])
    v2_sparse = float(np.mean([ep_med(v2, e, 'v2') for e in SPARSE]))
    ln_sparse = float(np.mean([ep_med(learned, e, 'learned') for e in SPARSE]))

    print("="*80)
    print("TASK 2 — merged-learned sparse parity (n=40, median-of-5)")
    print("="*80)
    print(f"  baselines: pure-v2 sparse={v2_sparse:.4f}  pure-learned sparse={ln_sparse:.4f}")
    print(f"  FULL PARITY <= {v2_sparse+0.005:.4f} (+0.005) ; PARTIAL <= 0.037 ; else FAIL")

    out = dict(v2_sparse=v2_sparse, learned_sparse=ln_sparse, variants={})
    best = None
    for mf in VARIANTS:
        # coverage check
        have = sum(1 for e in SPARSE for (E, D) in LANES if (e, E, D, mf) in merged)
        mg_sparse = float(np.mean([ep_med(merged, e, 'merged', mf) for e in SPARSE]))
        # paired wilcoxon merged vs v2 (episode medians)
        a = np.array([ep_med(merged, e, 'merged', mf) for e in SPARSE])
        b = np.array([ep_med(v2, e, 'v2') for e in SPARSE])
        m = np.isfinite(a) & np.isfinite(b)
        try:
            p = float(wilcoxon(a[m], b[m]).pvalue) if np.any(a[m] != b[m]) else 1.0
        except Exception:
            p = float('nan')
        tier = 'FULL' if mg_sparse <= v2_sparse + 0.005 else ('PARTIAL' if mg_sparse <= 0.037 else 'FAIL')
        avg_n = med([mn.get((e, E, D, mf)) for e in SPARSE for (E, D) in LANES])
        out['variants'][str(mf)] = dict(coverage=have, sparse_mean=mg_sparse, tier=tier,
                                        wilcoxon_p_vs_v2=p, median_delta_vs_v2=float(np.median(a[m]-b[m])),
                                        wins_vs_v2=int((a[m] < b[m]).sum()), n=int(m.sum()),
                                        median_final_n=avg_n)
        print(f"\n  merge_frac={mf} (coverage {have}/200 lanes):")
        print(f"    sparse mean = {mg_sparse:.4f}  -> TIER = {tier}")
        print(f"    vs v2: medianΔ={np.median(a[m]-b[m]):+.4f} wins {int((a[m]<b[m]).sum())}/{int(m.sum())} "
              f"Wilcoxon p={p:.3g}  median final_N={avg_n:.0f}")
        if best is None or mg_sparse < out['variants'][str(best)]['sparse_mean']:
            best = mf
    out['best_variant'] = best
    out['best_tier'] = out['variants'][str(best)]['tier']

    # No-harm crowded spot check: merged vs CURRENT learned (paired, 10 episodes)
    print("\n" + "="*80)
    print("No-harm crowded spot check (10 eps): merged-learned vs CURRENT learned (paired)")
    print("="*80)
    out['no_harm'] = {}
    for mf in VARIANTS:
        a = np.array([ep_med(merged, e, 'merged', mf) for e in CROWDED_SPOT])
        b = np.array([ep_med(learned, e, 'learned') for e in CROWDED_SPOT])
        m = np.isfinite(a) & np.isfinite(b)
        try:
            p = float(wilcoxon(a[m], b[m]).pvalue) if np.any(a[m] != b[m]) else 1.0
        except Exception:
            p = float('nan')
        harmless = (p >= 0.05) or (np.median(a[m]-b[m]) <= 0)
        out['no_harm'][str(mf)] = dict(merged_mean=float(a[m].mean()), learned_mean=float(b[m].mean()),
                                       median_delta=float(np.median(a[m]-b[m])), p=p,
                                       n_sig=int(m.sum()), no_harm_pass=bool(p >= 0.05))
        print(f"  mf={mf}: merged={a[m].mean():.4f} learned={b[m].mean():.4f} "
              f"medianΔ={np.median(a[m]-b[m]):+.4f} p={p:.3g} -> n.s.(no-harm)={p>=0.05}")

    print(f"\n  BEST VARIANT = {best}  TIER = {out['best_tier']}")
    json.dump(out, open('tmp/unified_guard_2026-07-26/task2_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
