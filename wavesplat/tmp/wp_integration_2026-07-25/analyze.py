"""STAGE-1 paired analysis: learned-init vs v2_konly-init under the phase-fixed tracker.

Median-of-5 per (episode, arm); paired one-sided Wilcoxon on the 40 crowded; panel; splits;
sparse control; init decomposition; naive regime-switch. Writes results.json + prints tables.
"""
import json, math
import numpy as np
from scipy.stats import wilcoxon, rankdata

RUNS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
CACHE = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'
H2H = 'tmp/wp_crowded_2026-07-25/results.json'

CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30)
           for s in range(5) for arm in ('uniform', 'slope')]
SPARSE = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_slope.h5',
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed1_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_uniform.h5',
]
PANEL = ['avg_nRMSE', 'avg_Variance_Ratio', 'avg_Pearson_rho', 'avg_Skill_Score',
         'avg_Directional_R2']
H2H_INIT_ADV = 0.1052   # head-to-head median paired Δ (cond vs ws_best), init/nowcast level


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


def load():
    rows = [json.loads(l) for l in open(RUNS)]
    rows = [r for r in rows if r.get('err') is None]
    # index: [arm][episode] -> {metric: [values over lanes]}
    D = {}
    for r in rows:
        D.setdefault(r['arm'], {}).setdefault(r['episode'], []).append(r)
    return D


def episode_median(D, arm, ep, key):
    return med([r.get(key) for r in D.get(arm, {}).get(ep, [])])


def rank_biserial(delta):
    """positive => 'learned' (delta<0) favored."""
    d = np.array([x for x in delta if np.isfinite(x) and x != 0.0])
    if len(d) == 0:
        return 0.0
    ar = rankdata(np.abs(d))
    Wneg = ar[d < 0].sum(); Wpos = ar[d > 0].sum()
    return float((Wneg - Wpos) / (Wneg + Wpos)) if (Wneg + Wpos) else 0.0


def paired_test(learned, v2):
    """arrays aligned per episode. H1: learned < v2 (improvement)."""
    learned = np.asarray(learned, float); v2 = np.asarray(v2, float)
    m = np.isfinite(learned) & np.isfinite(v2)
    learned, v2 = learned[m], v2[m]
    delta = learned - v2
    wins = int((delta < 0).sum()); n = len(delta)
    try:
        p = float(wilcoxon(learned, v2, alternative='less').pvalue)
    except Exception:
        p = float('nan')
    return dict(n=n, learned_mean=float(learned.mean()), v2_mean=float(v2.mean()),
                median_delta=float(np.median(delta)), mean_delta=float(delta.mean()),
                wins=wins, rank_biserial=rank_biserial(delta), p_onesided=p)


def main():
    D = load()
    arms = sorted(D.keys())
    print("arms present:", arms)

    # ---- primary: crowded 40, tracked avg_nRMSE ----
    L = [episode_median(D, 'learned', ep, 'avg_nRMSE') for ep in CROWDED]
    V = [episode_median(D, 'v2_konly', ep, 'avg_nRMSE') for ep in CROWDED]
    prim = paired_test(L, V)
    P1 = prim['p_onesided'] < 0.05
    P2 = prim['median_delta'] < 0
    retention = abs(prim['median_delta']) / H2H_INIT_ADV
    prim.update(P1=bool(P1), P2=bool(P2), decision='PASS' if (P1 and P2) else 'FAIL',
                retention_frac=retention, retention_pass=bool(retention >= 0.60))

    print("\n=== PRIMARY (crowded 40, tracked avg_nRMSE/Hs, learned vs v2_konly) ===")
    for k in ('n', 'learned_mean', 'v2_mean', 'median_delta', 'mean_delta', 'wins',
              'rank_biserial', 'p_onesided', 'decision', 'retention_frac',
              'retention_pass'):
        print(f"  {k:16s} {prim[k]}")

    # ---- context: vs ws_best (min init/tracked) ----
    def wsbest(ep):
        i = episode_median(D, 'v2_konly', ep, 'init_nrmse')
        t = episode_median(D, 'v2_konly', ep, 'avg_nRMSE')
        return min([x for x in (i, t) if np.isfinite(x)], default=float('nan'))
    Lb = [episode_median(D, 'learned', ep, 'avg_nRMSE') for ep in CROWDED]
    Vb = [wsbest(ep) for ep in CROWDED]
    vs_best = paired_test(Lb, Vb)

    # ---- panel (crowded, median-of-5, paired) ----
    panel = {}
    for key in PANEL:
        Lp = [episode_median(D, 'learned', ep, key) for ep in CROWDED]
        Vp = [episode_median(D, 'v2_konly', ep, key) for ep in CROWDED]
        Lm = np.nanmean(Lp); Vm = np.nanmean(Vp)
        # higher-is-better for VR(→1)/rho/SS/DirR2; nRMSE lower-is-better
        panel[key] = dict(learned=float(Lm), v2_konly=float(Vm),
                          delta=float(Lm - Vm))

    # ---- splits ----
    def split(subset):
        L = [episode_median(D, 'learned', ep, 'avg_nRMSE') for ep in subset]
        V = [episode_median(D, 'v2_konly', ep, 'avg_nRMSE') for ep in subset]
        return paired_test(L, V)
    splits = {
        '15comp': split([e for e in CROWDED if '15comp' in e]),
        '30comp': split([e for e in CROWDED if '30comp' in e]),
        'BF4': split([e for e in CROWDED if 'BF4' in e]),
        'BF5': split([e for e in CROWDED if 'BF5' in e]),
        'uniform': split([e for e in CROWDED if 'uniform' in e]),
        'slope': split([e for e in CROWDED if 'slope' in e]),
    }

    # ---- sparse control ----
    Ls = [episode_median(D, 'learned', ep, 'avg_nRMSE') for ep in SPARSE]
    Vs = [episode_median(D, 'v2_konly', ep, 'avg_nRMSE') for ep in SPARSE]
    sparse = paired_test(Ls, Vs)

    # ---- init decomposition (crowded) ----
    decomp = dict(
        learned_raw_init=med([episode_median(D, 'learned', ep, 'raw_init_nrmse')
                              for ep in CROWDED]),
        learned_post_init=med([episode_median(D, 'learned', ep, 'init_nrmse')
                               for ep in CROWDED]),
        learned_tracked=med([episode_median(D, 'learned', ep, 'avg_nRMSE')
                             for ep in CROWDED]),
        v2_init=med([episode_median(D, 'v2_konly', ep, 'init_nrmse') for ep in CROWDED]),
        v2_tracked=med([episode_median(D, 'v2_konly', ep, 'avg_nRMSE') for ep in CROWDED]),
        learned_final_n=med([episode_median(D, 'learned', ep, 'final_n')
                             for ep in CROWDED]),
        v2_final_n=med([episode_median(D, 'v2_konly', ep, 'final_n') for ep in CROWDED]),
    )

    # ---- learned_konly ablation (if present) ----
    konly = None
    if 'learned_konly' in D:
        Lk = [episode_median(D, 'learned_konly', ep, 'avg_nRMSE') for ep in CROWDED]
        konly = dict(vs_v2=paired_test(Lk, V),
                     vs_learned=paired_test(Lk, L),
                     mean=float(np.nanmean(Lk)))

    # ---- naive regime-switch (n_selected>=8 => learned) ----
    nsel = {}
    for line in open(CACHE):
        d = json.loads(line)
        if d.get('err') is None and d.get('n_selected') is not None:
            nsel.setdefault(d['episode'], []).append(d['n_selected'])
    nsel_med = {ep: med(v) for ep, v in nsel.items()}
    ALL = CROWDED + SPARSE
    THRESH = 8
    switch_vals, pure_L, pure_V, oracle, choice = [], [], [], [], {}
    for ep in ALL:
        lv = episode_median(D, 'learned', ep, 'avg_nRMSE')
        vv = episode_median(D, 'v2_konly', ep, 'avg_nRMSE')
        pure_L.append(lv); pure_V.append(vv)
        oracle.append(min(lv, vv))
        pick = 'learned' if nsel_med.get(ep, 0) >= THRESH else 'v2_konly'
        choice[ep] = pick
        switch_vals.append(lv if pick == 'learned' else vv)
    # separation of the signal by design regime
    sep = dict(
        sparse_nsel=[nsel_med.get(ep) for ep in SPARSE],
        crowded_nsel_min=float(np.nanmin([nsel_med.get(ep, np.nan) for ep in CROWDED])),
        crowded_nsel_med=float(np.nanmedian([nsel_med.get(ep, np.nan) for ep in CROWDED])),
        sparse_nsel_max=float(np.nanmax([nsel_med.get(ep, np.nan) for ep in SPARSE])),
        n_pick_learned=int(sum(1 for ep in ALL if choice[ep] == 'learned')),
    )
    regime = dict(
        threshold=THRESH,
        switch_mean=float(np.nanmean(switch_vals)),
        pure_learned_mean=float(np.nanmean(pure_L)),
        pure_v2_mean=float(np.nanmean(pure_V)),
        oracle_mean=float(np.nanmean(oracle)),
        n_episodes=len(ALL), separation=sep,
        crowded_switch_mean=float(np.nanmean(switch_vals[:40])),
        sparse_switch_mean=float(np.nanmean(switch_vals[40:])),
    )

    # ---- per-episode table (crowded) ----
    def sd(arm, ep):
        vs = [r['avg_nRMSE'] for r in D.get(arm, {}).get(ep, [])
              if r.get('avg_nRMSE') is not None]
        return float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
    table = []
    for ep in CROWDED:
        table.append(dict(
            episode=ep,
            learned=episode_median(D, 'learned', ep, 'avg_nRMSE'),
            learned_sd=sd('learned', ep),
            v2_konly=episode_median(D, 'v2_konly', ep, 'avg_nRMSE'),
            v2_sd=sd('v2_konly', ep),
            learned_init=episode_median(D, 'learned', ep, 'init_nrmse'),
            v2_init=episode_median(D, 'v2_konly', ep, 'init_nrmse'),
            learned_raw=episode_median(D, 'learned', ep, 'raw_init_nrmse'),
            nsel=nsel_med.get(ep),
        ))
    sparse_table = []
    for ep in SPARSE:
        sparse_table.append(dict(
            episode=ep,
            learned=episode_median(D, 'learned', ep, 'avg_nRMSE'),
            v2_konly=episode_median(D, 'v2_konly', ep, 'avg_nRMSE'),
            nsel=nsel_med.get(ep)))

    out = dict(primary=prim, vs_ws_best=vs_best, panel=panel, splits=splits,
               sparse=sparse, decomp=decomp, konly=konly, regime=regime,
               table=table, sparse_table=sparse_table)
    json.dump(out, open('tmp/wp_integration_2026-07-25/results.json', 'w'), indent=2)

    print("\n=== vs ws_best (context) ===");  print(" ", vs_best)
    print("\n=== PANEL (crowded means) ===")
    for k, v in panel.items():
        print(f"  {k:22s} learned {v['learned']:.4f}  v2 {v['v2_konly']:.4f}  Δ {v['delta']:+.4f}")
    print("\n=== SPLITS (median Δ, wins, p) ===")
    for k, v in splits.items():
        print(f"  {k:8s} n{v['n']} learned {v['learned_mean']:.3f} v2 {v['v2_mean']:.3f} "
              f"medΔ {v['median_delta']:+.3f} wins {v['wins']}/{v['n']} p {v['p_onesided']:.2g}")
    print("\n=== SPARSE control ===");  print(" ", sparse)
    print("\n=== INIT DECOMPOSITION (crowded medians) ===")
    for k, v in decomp.items():
        print(f"  {k:20s} {v}")
    if konly:
        print("\n=== learned_konly ablation ==="); print(" vs_v2:", konly['vs_v2']);
        print(" vs_learned:", konly['vs_learned'])
    print("\n=== REGIME-SWITCH (52 ep, n_selected>=8 => learned) ===")
    for k, v in regime.items():
        if k != 'separation':
            print(f"  {k:20s} {v}")
    print("  separation:", regime['separation'])


if __name__ == '__main__':
    main()
