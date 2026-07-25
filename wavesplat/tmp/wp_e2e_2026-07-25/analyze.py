"""Paired analysis for STAGE-2 (PREREG §4,§5). Prints + dumps results.json.

Primary: ft vs current, 40 crowded, median-of-5 cond_polish_nrmse_Hs, one-sided Wilcoxon (ft<current).
Sparse control (incl. OOD BF2_1comp_slope). Integrated spot check: ft-learned vs Stage-1 old-learned
tracked avg_nRMSE, 10 crowded, median-of-5.
"""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon

CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30) for s in range(5) for arm in ('uniform', 'slope')]
SPARSE = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_slope.h5',
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed1_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_uniform.h5',
]
SPOT10 = [
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_slope.h5',
    'episode_BF4_15comp_seed1_uniform.h5', 'episode_BF4_15comp_seed1_slope.h5',
    'episode_BF4_30comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_slope.h5',
    'episode_BF4_30comp_seed1_uniform.h5', 'episode_BF4_30comp_seed1_slope.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_slope.h5',
]


def med_by_ep(path, key, valid_eps=None):
    """median over seeds per episode from a jsonl (skip err/None)."""
    acc = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get('err') is not None:
            continue
        v = d.get(key)
        if v is None:
            continue
        ep = d['episode']
        if valid_eps is not None and ep not in valid_eps:
            continue
        acc.setdefault(ep, []).append(float(v))
    return {ep: float(np.median(vs)) for ep, vs in acc.items()}


def paired(ft, cur, eps):
    rows = []
    for ep in eps:
        if ep in ft and ep in cur:
            rows.append((ep, ft[ep], cur[ep], ft[ep] - cur[ep]))
    d = np.array([r[3] for r in rows])
    wins = int((d < 0).sum()); ties = int((d == 0).sum())
    out = dict(n=len(rows), ft_mean=float(np.mean([r[1] for r in rows])),
               cur_mean=float(np.mean([r[2] for r in rows])),
               median_delta=float(np.median(d)), mean_delta=float(np.mean(d)),
               wins=wins, losses=len(rows) - wins - ties, ties=ties, rows=rows)
    if len(rows) >= 5 and np.any(d != 0):
        try:
            w = wilcoxon(d, alternative='less')   # H1: ft < cur
            out['wilcoxon_p'] = float(w.pvalue)
            wl = wilcoxon(d, alternative='two-sided')
            out['wilcoxon_stat'] = float(wl.statistic)
        except Exception as e:
            out['wilcoxon_p'] = None; out['err'] = repr(e)
    # matched-pairs rank-biserial
    nz = d[d != 0]
    if len(nz):
        ranks = np.argsort(np.argsort(np.abs(nz))) + 1
        wp = ranks[nz > 0].sum(); wm = ranks[nz < 0].sum()
        out['rank_biserial'] = float((wm - wp) / (wp + wm))  # + = ft better (more neg deltas)
    return out


def main():
    res = {}
    FT = 'tmp/wp_e2e_2026-07-25/ft_cond_runs.jsonl'
    CUR = 'tmp/wp_crowded_2026-07-25/cond_runs.jsonl'

    ft = med_by_ep(FT, 'cond_polish_nrmse_Hs')
    cur = med_by_ep(CUR, 'cond_polish_nrmse_Hs')

    # ---- PRIMARY: 40 crowded ----
    prim = paired(ft, cur, CROWDED)
    res['primary'] = prim
    p = prim.get('wilcoxon_p')
    P1 = (p is not None and p < 0.05); P2 = prim['median_delta'] < 0
    prim['P1'] = P1; prim['P2'] = P2; prim['DECISION'] = 'PASS' if (P1 and P2) else 'FAIL'
    print("=== PRIMARY (ft vs current, 40 crowded, cond median-of-5) ===")
    print(f"  ft mean {prim['ft_mean']:.4f}  current mean {prim['cur_mean']:.4f}")
    print(f"  median Δ (ft−cur) {prim['median_delta']:+.4f}  mean Δ {prim['mean_delta']:+.4f}")
    print(f"  wins {prim['wins']}/{prim['n']}  ties {prim['ties']}  "
          f"rank-biserial {prim.get('rank_biserial')}")
    print(f"  one-sided Wilcoxon p {p}")
    print(f"  P1(p<0.05)={P1}  P2(medΔ<0)={P2}  ==> {prim['DECISION']}")

    # ---- SPARSE control (incl OOD) ----
    spa = paired(ft, cur, SPARSE)
    res['sparse'] = spa
    print("\n=== SPARSE control (ft vs current, 12) ===")
    print(f"  ft mean {spa['ft_mean']:.4f}  current mean {spa['cur_mean']:.4f}  "
          f"median Δ {spa['median_delta']:+.4f}  wins {spa['wins']}/{spa['n']}")
    for ep, f, c, dd in sorted(spa['rows'], key=lambda r: -r[3]):
        flag = '  <-- OOD' if 'BF2_1comp_seed0_slope' in ep else ''
        print(f"    {ep:38s} ft {f:.4f} cur {c:.4f} Δ {dd:+.4f}{flag}")

    # ---- INTEGRATED spot check: ft-learned vs old-learned tracked avg_nRMSE ----
    FT_ARM = 'tmp/wp_e2e_2026-07-25/ft_arm_runs.jsonl'
    OLD_ARM = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
    if os.path.exists(FT_ARM):
        def learned_med(path):
            acc = {}
            for line in open(path):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get('err') or d.get('arm') != 'learned' or d.get('avg_nRMSE') is None:
                    continue
                if d['episode'] not in SPOT10:
                    continue
                acc.setdefault(d['episode'], []).append(float(d['avg_nRMSE']))
            return {e: float(np.median(v)) for e, v in acc.items()}
        ftl = learned_med(FT_ARM); oldl = learned_med(OLD_ARM)
        integ = paired(ftl, oldl, SPOT10)
        res['integrated'] = integ
        print("\n=== INTEGRATED spot check (ft-learned vs old-learned tracked avg_nRMSE, 10) ===")
        print(f"  ft-learned mean {integ['ft_mean']:.4f}  old-learned mean {integ['cur_mean']:.4f}")
        print(f"  median Δ {integ['median_delta']:+.4f}  wins {integ['wins']}/{integ['n']}  "
              f"p(ft<old) {integ.get('wilcoxon_p')}")
        for ep, f, c, dd in sorted(integ['rows'], key=lambda r: r[3]):
            print(f"    {ep:38s} ftL {f:.4f} oldL {c:.4f} Δ {dd:+.4f}")
    else:
        print("\n(integrated spot check pending: ft_arm_runs.jsonl not found)")

    # full crowded table
    print("\n=== crowded per-episode (ft / cur / Δ) ===")
    for ep, f, c, dd in prim['rows']:
        print(f"  {ep:38s} ft {f:.4f} cur {c:.4f} Δ {dd:+.4f}")

    json.dump(res, open('tmp/wp_e2e_2026-07-25/results.json', 'w'), indent=2)
    print("\nwrote tmp/wp_e2e_2026-07-25/results.json")


if __name__ == '__main__':
    main()
