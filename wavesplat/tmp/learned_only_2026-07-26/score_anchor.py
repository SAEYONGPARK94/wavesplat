"""CANDIDATE 1 scoring: peak-anchored-learned sparse parity + crowded no-harm. Frozen
tiers/tests in tmp/learned_only_2026-07-26/PREREG.md. Reads anchor_arms.jsonl + reused jsonls."""
import json, re
import numpy as np
from scipy.stats import wilcoxon

BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
S1_ARMS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
X_ARMS = 'tmp/unified_2026-07-26/arm_runs_extra.jsonl'
ANCHOR = 'tmp/learned_only_2026-07-26/anchor_arms.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
VARIANTS = [0.375, 0.60]
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
    anc = {}; an = {}
    for l in open(ANCHOR):
        d = json.loads(l)
        if d.get('err') is None and d.get('avg_nRMSE') is not None:
            anc[(d['episode'], d['est_seed'], d['diff_seed'], d['anchor_frac'])] = d['avg_nRMSE']
            an[(d['episode'], d['est_seed'], d['diff_seed'], d['anchor_frac'])] = d['final_n']
    return sorted(eps), v2, learned, anc, an


def ep_med(d, ep, kind, af=None):
    if kind == 'v2':
        return med([d.get((ep, E)) for (E, D) in LANES])
    if kind == 'learned':
        return med([d.get((ep, E, D)) for (E, D) in LANES])
    return med([d.get((ep, E, D, af)) for (E, D) in LANES])


def main():
    eps, v2, learned, anc, an = load()
    SPARSE = sorted([e for e in eps if comp(e) in (1, 2)])
    v2_sparse = float(np.mean([ep_med(v2, e, 'v2') for e in SPARSE]))
    ln_sparse = float(np.mean([ep_med(learned, e, 'learned') for e in SPARSE]))

    print("=" * 92)
    print("CANDIDATE 1 — peak-anchored-learned sparse parity (n=40, median-of-5)")
    print("=" * 92)
    print(f"  baselines: pure-v2 sparse={v2_sparse:.4f}  pure-learned sparse={ln_sparse:.4f}")
    print(f"  FULL PARITY <= {v2_sparse+0.005:.4f} (+0.005) ; PARTIAL <= 0.037 ; else FAIL")

    out = dict(v2_sparse=v2_sparse, learned_sparse=ln_sparse, variants={})
    best = None
    for af in VARIANTS:
        have = sum(1 for e in SPARSE for (E, D) in LANES if (e, E, D, af) in anc)
        ag_sparse = float(np.mean([ep_med(anc, e, 'anc', af) for e in SPARSE]))
        a = np.array([ep_med(anc, e, 'anc', af) for e in SPARSE])
        b = np.array([ep_med(v2, e, 'v2') for e in SPARSE])
        m = np.isfinite(a) & np.isfinite(b)
        try:
            p = float(wilcoxon(a[m], b[m]).pvalue) if np.any(a[m] != b[m]) else 1.0
        except Exception:
            p = float('nan')
        # paired vs pure-learned too (did anchoring help sparse at all?)
        c = np.array([ep_med(learned, e, 'learned') for e in SPARSE])
        m2 = np.isfinite(a) & np.isfinite(c)
        try:
            p_ln = float(wilcoxon(a[m2], c[m2]).pvalue) if np.any(a[m2] != c[m2]) else 1.0
        except Exception:
            p_ln = float('nan')
        tier = 'FULL' if ag_sparse <= v2_sparse + 0.005 else ('PARTIAL' if ag_sparse <= 0.037 else 'FAIL')
        avg_n = med([an.get((e, E, D, af)) for e in SPARSE for (E, D) in LANES])
        out['variants'][str(af)] = dict(coverage=have, sparse_mean=ag_sparse, tier=tier,
                                        wilcoxon_p_vs_v2=p, median_delta_vs_v2=float(np.median(a[m]-b[m])),
                                        wins_vs_v2=int((a[m] < b[m]).sum()),
                                        median_delta_vs_learned=float(np.median(a[m2]-c[m2])),
                                        wilcoxon_p_vs_learned=p_ln, n=int(m.sum()), median_final_n=avg_n)
        print(f"\n  anchor_frac={af} (coverage {have}/200 lanes):")
        print(f"    sparse mean = {ag_sparse:.4f}  -> TIER = {tier}")
        print(f"    vs v2:      medianΔ={np.median(a[m]-b[m]):+.4f} wins {int((a[m]<b[m]).sum())}/{int(m.sum())} p={p:.3g}")
        print(f"    vs learned: medianΔ={np.median(a[m2]-c[m2]):+.4f} p={p_ln:.3g}  median final_N={avg_n:.0f}")
        if best is None or ag_sparse < out['variants'][str(best)]['sparse_mean']:
            best = af
    out['best_variant'] = best
    out['best_tier'] = out['variants'][str(best)]['tier']

    print("\n" + "=" * 92)
    print("No-harm crowded spot check (10 eps): anchored-learned vs CURRENT learned (paired)")
    print("=" * 92)
    out['no_harm'] = {}
    for af in VARIANTS:
        a = np.array([ep_med(anc, e, 'anc', af) for e in CROWDED_SPOT])
        b = np.array([ep_med(learned, e, 'learned') for e in CROWDED_SPOT])
        m = np.isfinite(a) & np.isfinite(b)
        try:
            p = float(wilcoxon(a[m], b[m]).pvalue) if np.any(a[m] != b[m]) else 1.0
        except Exception:
            p = float('nan')
        harmless = bool((p >= 0.05) or (np.median(a[m]-b[m]) <= 0))
        out['no_harm'][str(af)] = dict(anchored_mean=float(a[m].mean()), learned_mean=float(b[m].mean()),
                                       median_delta=float(np.median(a[m]-b[m])), p=p,
                                       n=int(m.sum()), no_harm_pass=harmless)
        print(f"  af={af}: anchored={a[m].mean():.4f} learned={b[m].mean():.4f} "
              f"medianΔ={np.median(a[m]-b[m]):+.4f} p={p:.3g} -> no-harm={harmless}")

    print(f"\n  BEST VARIANT = {best}  TIER = {out['best_tier']}")
    # per-episode detail on the catastrophic BF2_1comp eps (best variant)
    print("\n  Per-episode (BF2_1comp, best variant vs v2/learned):")
    for e in SPARSE:
        if 'BF2_1comp' in e:
            print(f"    {e:38s} v2={ep_med(v2,e,'v2'):.3f} learned={ep_med(learned,e,'learned'):.3f} "
                  f"anchored={ep_med(anc,e,'anc',best):.3f}")
    json.dump(out, open('tmp/learned_only_2026-07-26/anchor_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
