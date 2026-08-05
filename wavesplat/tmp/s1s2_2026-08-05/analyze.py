"""Aggregate tmp/s1s2_2026-08-05/results.jsonl exactly as PREREG.md section 2.1 fixes.

median-of-3 seeds per (arm, episode) -> paired difference vs the arm's designated control
-> median of the 4 paired differences + exact two-sided sign test (floor p = 0.125, n=4).
"""
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

EPISODES = [
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
SHORT = {e: e.replace('episode_', '').replace('_seed0_uniform.h5', '') for e in EPISODES}

# FROZEN reference (tmp/rerun_2026-07-25/batchA_baselines.jsonl, seed 42, same finemesh data)
LETKF = {
    'episode_BF4_15comp_seed0_uniform.h5': 0.2282,
    'episode_BF4_30comp_seed0_uniform.h5': 0.2347,
    'episode_BF5_15comp_seed0_uniform.h5': 0.2298,
    'episode_BF5_30comp_seed0_uniform.h5': 0.2651,
}

# metric -> +1 if HIGHER is better, -1 if LOWER is better
DIRECTION = {
    'nRMSE': -1, 'Pearson_rho': +1, 'Skill_Score': +1, 'Directional_R2': +1,
    'Hs_error': -1, 'avg_RMSE_unobserved': -1, 'VR_dev': -1,
    'detection_rate': +1, 'energy_detection_rate': +1,
    'k_mag_rel_error_mean': -1, 'omega_rel_error_mean': -1,
    'amp_rel_error_mean': -1, 'direction_error_deg_mean': -1,
}
FIELD_METRICS = ['nRMSE', 'Skill_Score', 'Pearson_rho', 'VR_dev', 'Variance_Ratio',
                 'Directional_R2', 'Hs_error', 'avg_RMSE_unobserved']
SPEC_METRICS = ['detection_rate', 'energy_detection_rate', 'k_mag_rel_error_mean',
                'omega_rel_error_mean', 'amp_rel_error_mean', 'direction_error_deg_mean']


def load():
    recs = []
    with open(os.path.join(HERE, 'results.jsonl')) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r:
                print('ERROR RUN:', r['job_id'], r['error'][:120])
                continue
            flat = dict(r)
            flat['VR_dev'] = abs(r['Variance_Ratio'] - 1.0)
            for k, v in (r.get('spec') or {}).items():
                flat[k] = v
            recs.append(flat)
    return recs


def med_table(recs, metric):
    """(arm, episode) -> median over seeds; also seed range."""
    med, rng, nseed = {}, {}, {}
    for arm in sorted({r['arm'] for r in recs}):
        for ep in EPISODES:
            vals = [r[metric] for r in recs
                    if r['arm'] == arm and r['episode'] == ep and metric in r
                    and np.isfinite(r[metric])]
            if vals:
                med[(arm, ep)] = float(np.median(vals))
                rng[(arm, ep)] = float(np.max(vals) - np.min(vals))
                nseed[(arm, ep)] = len(vals)
    return med, rng, nseed


def sign_test_p(diffs):
    """Exact two-sided sign test; ties dropped. n=4 floor is 0.125."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x > 0)
    kk = min(k, n - k)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n)
    return p, k, n


def contrast(recs, treat, ctrl, metric):
    med, rng, _ = med_table(recs, metric)
    rows, diffs, improves = [], [], []
    sgn = DIRECTION.get(metric, -1)
    for ep in EPISODES:
        if (treat, ep) not in med or (ctrl, ep) not in med:
            continue
        c, t = med[(ctrl, ep)], med[(treat, ep)]
        d = t - c
        rows.append((ep, c, t, d))
        diffs.append(d)
        improves.append(sgn * d)          # >0 means treatment is better
    if not diffs:
        return None
    p, k, n = sign_test_p(improves)
    return {
        'metric': metric, 'treat': treat, 'ctrl': ctrl, 'rows': rows,
        'median_diff': float(np.median(diffs)),
        'n_better': int(sum(1 for x in improves if x > 0)), 'n': len(improves),
        'sign_p': p,
        'seed_range_median_ctrl': float(np.median(
            [rng[(ctrl, ep)] for ep in EPISODES if (ctrl, ep) in rng])),
        'seed_range_median_treat': float(np.median(
            [rng[(treat, ep)] for ep in EPISODES if (treat, ep) in rng])),
    }


def recovery(recs, treat, ctrl):
    """nRMSE recovery fraction of the LETKF gap, per PREREG 3.1."""
    med, _, _ = med_table(recs, 'nRMSE')
    out = []
    for ep in EPISODES:
        if (treat, ep) not in med or (ctrl, ep) not in med:
            continue
        c, t, L = med[(ctrl, ep)], med[(treat, ep)], LETKF[ep]
        G = c - L
        out.append({'ep': SHORT[ep], 'ctrl': c, 'treat': t, 'letkf': L, 'gap': G,
                    'R': (c - t) / G if abs(G) > 1e-12 else float('nan'),
                    'resid_frac': (t - L) / G if abs(G) > 1e-12 else float('nan'),
                    'beats_letkf': bool(t < L)})
    return out


def verdict(R):
    R = [x for x in R if np.isfinite(x)]
    if not R:
        return 'NO DATA'
    m = float(np.median(R))
    if m >= 0.50 and all(x > 0 for x in R):
        return f'IMPLEMENTATION-LIMITED (median R={m:+.2f}, {len(R)}/{len(R)} positive)'
    if m <= 0.20 and all(x < 0.50 for x in R):
        return f'REPRESENTATION-LIMITED (median R={m:+.2f})'
    return f'INDETERMINATE (median R={m:+.2f}, {sum(1 for x in R if x>0)}/{len(R)} positive)'


def print_contrast(c, indent='  '):
    if c is None:
        return
    print(f"{indent}{c['metric']:<26s} medianΔ {c['median_diff']:+.4f}  "
          f"better {c['n_better']}/{c['n']}  sign p={c['sign_p']:.3f}  "
          f"(seed range med ctrl {c['seed_range_median_ctrl']:.4f} / "
          f"treat {c['seed_range_median_treat']:.4f})")


def main():
    recs = load()
    print(f'runs loaded: {len(recs)}')
    arms = sorted({r['arm'] for r in recs})
    print('arms:', arms)

    # ---- per-arm per-episode nRMSE table ----
    med, rng, nseed = med_table(recs, 'nRMSE')
    print('\n=== nRMSE, median-of-3 (range over seeds) ===')
    hdr = f"{'arm':<8s}" + ''.join(f'{SHORT[e]:>22s}' for e in EPISODES) + f"{'mean4':>9s}"
    print(hdr)
    for arm in arms:
        cells, vals = '', []
        for e in EPISODES:
            if (arm, e) in med:
                cells += f'{med[(arm,e)]:>13.4f} ({rng[(arm,e)]:.3f})'
                vals.append(med[(arm, e)])
            else:
                cells += f"{'--':>22s}"
        print(f'{arm:<8s}{cells}{np.mean(vals):>9.4f}' if vals else f'{arm:<8s}{cells}')
    print(f"{'LETKF':<8s}" + ''.join(f'{LETKF[e]:>22.4f}' for e in EPISODES)
          + f'{np.mean(list(LETKF.values())):>9.4f}')

    med_n, _, _ = med_table(recs, 'N')
    med_d, _, _ = med_table(recs, 'detection_rate')
    print('\n=== final_n (median) / detection_rate (median) ===')
    for arm in arms:
        cells = ''
        for e in EPISODES:
            if (arm, e) in med_n:
                cells += f'{med_n[(arm,e)]:>10.0f}/{med_d.get((arm,e),float("nan")):.3f}'
            else:
                cells += f"{'--':>16s}"
        print(f'{arm:<8s}{cells}')

    CONTRASTS = [
        ('S1', 'S1-A1', 'S1-A0'), ('S1', 'S1-A2', 'S1-A0'), ('S1', 'S1-A3', 'S1-A0'),
        ('S2', 'S2-C1', 'S1-A0'),
        ('S2', 'S2-T1', 'S2-C1'), ('S2', 'S2-T1f', 'S2-C1'),
        ('S2', 'S2-T1f', 'S1-A0'), ('S2', 'S2-T1f', 'S2-T1'),
        ('S2', 'S2-T2', 'S2-T1'), ('S2', 'S2-T4', 'S2-T1f'),
        ('S2', 'S2-T4', 'S1-A0'),
    ]
    print('\n=== paired contrasts (median of 4 episode-level paired diffs) ===')
    for tag, t, c in CONTRASTS:
        if t not in arms or c not in arms:
            continue
        print(f'\n[{tag}] {t}  vs  {c}')
        for m in FIELD_METRICS + SPEC_METRICS:
            print_contrast(contrast(recs, t, c, m))

    print('\n=== LETKF-gap recovery (PRIMARY = nRMSE) ===')
    for t, c in [('S1-A1', 'S1-A0'), ('S1-A2', 'S1-A0'), ('S1-A3', 'S1-A0'),
                 ('S2-C1', 'S1-A0'), ('S2-T1', 'S1-A0'), ('S2-T1f', 'S1-A0'),
                 ('S2-T2', 'S1-A0'), ('S2-T4', 'S1-A0'),
                 ('S2-T1', 'S2-C1'), ('S2-T1f', 'S2-C1')]:
        if t not in arms or c not in arms:
            continue
        rows = recovery(recs, t, c)
        print(f'\n{t} vs control {c}:')
        for r in rows:
            print(f"  {r['ep']:<12s} ctrl {r['ctrl']:.4f}  treat {r['treat']:.4f}  "
                  f"LETKF {r['letkf']:.4f}  gap {r['gap']:+.4f}  "
                  f"R {r['R']:+.3f}  resid/gap {r['resid_frac']:+.3f}  "
                  f"{'BEATS LETKF' if r['beats_letkf'] else ''}")
        print('  VERDICT:', verdict([r['R'] for r in rows]))

    # residual error of the best oracle arm, absolute and relative
    print('\n=== oracle residual error (absolute) ===')
    for arm in ['S2-T1', 'S2-T1f', 'S2-T4', 'S2-T2']:
        if arm not in arms:
            continue
        vals = [med[(arm, e)] for e in EPISODES if (arm, e) in med]
        base = [med[('S1-A0', e)] for e in EPISODES]
        L = [LETKF[e] for e in EPISODES]
        print(f'  {arm}: mean nRMSE {np.mean(vals):.4f} | shipped {np.mean(base):.4f} | '
              f'LETKF {np.mean(L):.4f} | '
              f'residual over LETKF {np.mean(vals)-np.mean(L):+.4f}')


if __name__ == '__main__':
    main()
