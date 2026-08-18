"""B2 — INDEPENDENT re-derivation of the five numbers PREREG.md §0.2 inherits.

ZERO estimator runs.  Nothing here imports or executes any prior agent's script
(`probe_postlasso.py`, `probe_varpro.py`, `analyze_w3.py`, `analyze_s4.py`,
 K6's `audit_*.py`).  Only raw artefacts are read:

    tmp/W3_SOLVER_2026-08-15/results.jsonl      (1212 rows)
    tmp/K6_run_2026-08-07/results.jsonl
    tmp/S4_phase_oracle_2026-08-08/results.jsonl  (stored final kx,ky)
    data_indep_crowded_2026-08-08/*.h5            (observations, for F1 only)

`probe_postlasso.jsonl` is read ONLY at the very end, as a cross-check target
(never as an input to any number I report).

Usage:  python3 B2_factors.py [f1|rest|all]
"""
import json
import math
import os
import sys
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))   # .../wavesplat/wavesplat
TMP = os.path.join(ROOT, 'tmp')
W3 = os.path.join(TMP, 'W3_SOLVER_2026-08-15', 'results.jsonl')
K6 = os.path.join(TMP, 'K6_run_2026-08-07', 'results.jsonl')
S4 = os.path.join(TMP, 'S4_phase_oracle_2026-08-08', 'results.jsonl')
G = 9.81
OUT = {}


# ----------------------------------------------------------------- statistics
def med(v):
    return float(np.median(np.asarray(v, float))) if len(v) else float('nan')


def sign_p_two(wins, n):
    """Exact two-sided binomial sign test, p=0.5."""
    if n == 0:
        return float('nan')
    c = lambda k: math.comb(n, k)
    tail = lambda k: sum(c(i) for i in range(k, n + 1)) / 2.0 ** n
    lo = min(wins, n - wins)
    return float(min(1.0, 2 * tail(n - lo)))


def sign_p_one(wins, n):
    if n == 0:
        return float('nan')
    return float(sum(math.comb(n, i) for i in range(wins, n + 1)) / 2.0 ** n)


def paired(treat, ctrl, label):
    """treat/ctrl: dict episode -> value.  Lower is better; 'win' = treat < ctrl."""
    keys = sorted(set(treat) & set(ctrl))
    t = np.array([treat[k] for k in keys], float)
    c = np.array([ctrl[k] for k in keys], float)
    ok = np.isfinite(t) & np.isfinite(c) & (t != c)
    n = int(ok.sum())
    wins = int((t[ok] < c[ok]).sum())
    ratios = t[ok] / c[ok]
    return dict(label=label, n=n, n_pairs_all=len(keys), wins=wins,
                p_one=sign_p_one(wins, n), p_two=sign_p_two(wins, n),
                median_ratio=med(ratios),
                median_rel_pct=100.0 * (med(ratios) - 1.0),
                median_treat=med(t[ok]), median_ctrl=med(c[ok]),
                median_abs_diff=med(t[ok] - c[ok]),
                ratio_of_medians=float(med(t[ok]) / med(c[ok])))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def norm_ep(e):
    """Strip dir/prefix/suffix so the three batches' episode ids are comparable."""
    e = e.split('/')[-1]
    if e.startswith('episode_'):
        e = e[len('episode_'):]
    if e.endswith('.h5'):
        e = e[:-3]
    return e


def med3(rows, key='avg_nRMSE', epkey='episode'):
    """episode -> median over estimator seeds (PREREG §6 aggregation)."""
    d = collections.defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            d[norm_ep(r[epkey])].append(float(v))
    return {k: med(v) for k, v in d.items()}, {k: len(v) for k, v in d.items()}


# =============================================================== F1: PROBE-A
def f1_closed_form_ab(specs=((('AUTO-NT',), (1557, 1558, 1559)),
                             (('ORA-WPHI-0STEP',), (1557,))),
                      regimes=('crowded', 'sparse')):
    """CLAIM 1: on the k the estimator actually ended with, does an exact global
    closed-form (a,b) refit help?  My own lstsq + my own scoring loop."""
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.evaluation.metrics import compute_all_metrics

    rows = [r for r in load(S4) if 'error' not in r and r['regime'] in regimes
            and any(r['cell'] in c and r['seed'] in s for c, s in specs)]
    cache = {}
    out = []
    for i, r in enumerate(sorted(rows, key=lambda z: (z['cell'], z['seed'], z['episode'])), 1):
        kx = np.asarray(r['kx'], float)
        ky = np.asarray(r['ky'], float)
        if kx.size == 0:
            continue
        ep = r['episode']
        if ep not in cache:
            cache[ep] = EpisodeDataLoader(os.path.join(ROOT, ep))
        L = cache[ep]
        kmag = np.hypot(kx, ky)
        om = np.sqrt(G * kmag * np.tanh(kmag * L.depth))

        # --- fit: exact global least squares on the init window ---------------
        win = L.get_init_window(tp_factor=2.0)
        Xw = np.concatenate([f.x_obs.numpy() for f in win]).astype(np.float64)
        Yw = np.concatenate([f.y_obs.numpy() for f in win]).astype(np.float64)
        Zw = np.concatenate([f.z_obs.numpy() for f in win]).astype(np.float64)
        Tw = np.concatenate([np.full(len(f.x_obs), f.t) for f in win]).astype(np.float64)
        th = np.outer(Xw, kx) + np.outer(Yw, ky) - np.outer(Tw, om)
        A = np.hstack([np.cos(th), np.sin(th)])
        coef, _res, rank, sv = np.linalg.lstsq(A, Zw, rcond=None)
        a, b = coef[:kx.size], coef[kx.size:]
        win_resid = float(np.sqrt(np.mean((A @ coef - Zw) ** 2)))

        # --- score: analytic propagation over the tracking frames -------------
        gx = L.gt_grid_x.numpy().astype(np.float64)
        gy = L.gt_grid_y.numpy().astype(np.float64)
        MX, MY = np.meshgrid(gx, gy)
        px, py = MX.ravel(), MY.ravel()
        base = np.outer(px, kx) + np.outer(py, ky)
        nrs = []
        for f in L.get_tracking_frames(tp_factor=2.0):
            ph = base - om[None, :] * float(f.t)
            fld = (np.cos(ph) @ a + np.sin(ph) @ b).reshape(MX.shape)
            ox = f.x_obs.numpy(); oy = f.y_obs.numpy()
            m = compute_all_metrics(fld, f.gt_field.numpy(), gx, gy, L.Hs_true,
                                    obs_x=ox if len(ox) else None,
                                    obs_y=oy if len(oy) else None)
            nrs.append(float(m['nRMSE']))
        out.append(dict(cell=r['cell'], seed=r['seed'], regime=r['regime'],
                        episode=norm_ep(ep), n_atoms=int(kx.size),
                        stored_nRMSE=float(r['nRMSE']),
                        stored_obs_resid=r.get('avg_obs_resid_rms'),
                        b2_refit_nRMSE=float(np.mean(nrs)),
                        b2_win_resid=win_resid,
                        b2_cond=float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float('inf'),
                        b2_rank=int(rank), n_frames=len(nrs)))
        print('[%3d] %-14s s%d %-42s stored %.5f -> B2 refit %.5f'
              % (i, r['cell'], r['seed'], out[-1]['episode'][:42],
                 out[-1]['stored_nRMSE'], out[-1]['b2_refit_nRMSE']), flush=True)
    with open(os.path.join(HERE, 'B2_f1_refit.jsonl'), 'w') as fh:
        for o in out:
            fh.write(json.dumps(o) + '\n')
    return out


def f1_report():
    rows = load(os.path.join(HERE, 'B2_f1_refit.jsonl'))
    rep = {}
    for cell in sorted({r['cell'] for r in rows}):
        for reg in sorted({r['regime'] for r in rows}):
            for seed in sorted({r['seed'] for r in rows}):
                sub = [r for r in rows if r['cell'] == cell and r['regime'] == reg
                       and r['seed'] == seed]
                if not sub:
                    continue
                t = {r['episode']: r['b2_refit_nRMSE'] for r in sub}
                c = {r['episode']: r['stored_nRMSE'] for r in sub}
                rep['%s|%s|seed%d' % (cell, reg, seed)] = paired(
                    t, c, 'closed-form (a,b) refit vs estimator (a,b)')
            # pooled over seeds: per-episode median-of-3 on BOTH sides
            sub = [r for r in rows if r['cell'] == cell and r['regime'] == reg]
            if len({r['seed'] for r in sub}) > 1:
                t, _ = med3(sub, key='b2_refit_nRMSE')
                c, _ = med3(sub, key='stored_nRMSE')
                rep['%s|%s|med3' % (cell, reg)] = paired(t, c, 'med-of-3 both sides')
    # harness calibration: my closed-form on ORACLE (kx,ky) vs W3 ORA-K-CVX
    cvx, _ = med3([r for r in load(W3) if r['arm'] == 'ORA-K-CVX'])
    for reg in ('crowded', 'sparse'):
        mine = {r['episode']: r['b2_refit_nRMSE'] for r in rows
                if r['cell'] == 'ORA-WPHI-0STEP' and r['regime'] == reg
                and r['seed'] == min(x['seed'] for x in rows)}
        shared = sorted(set(mine) & set(cvx))
        if shared:
            rel = [abs(mine[k] - cvx[k]) / cvx[k] for k in shared]
            rep['CALIB|0STEP_vs_W3CVX|' + reg] = dict(
                n=len(shared), median_mine=med([mine[k] for k in shared]),
                median_w3cvx=med([cvx[k] for k in shared]),
                max_rel_diff=float(max(rel)))
    return rep


# =============================================================== F2: K6
def f2_k6():
    rows = [r for r in load(K6) if 'error' not in r and 'nRMSE' in r]
    cells = {c: med3([r for r in rows if r['cell'] == c], key='nRMSE')[0]
             for c in sorted({r['cell'] for r in rows})}
    rep = {}
    for treat, ctrl in [('ORACLE', 'ORACLE-K'), ('ORACLE', 'AUTO'),
                        ('ORACLE-K', 'AUTO'), ('ORACLE-K', 'K6-NSCHED'),
                        ('K6-NSCHED', 'AUTO'), ('K6-KLR00', 'AUTO'),
                        ('K6-KLR03', 'AUTO'), ('AUTO-DUP', 'AUTO')]:
        rep['%s_vs_%s' % (treat, ctrl)] = paired(cells[treat], cells[ctrl],
                                                 '%s < %s' % (treat, ctrl))
    rep['_medians'] = {c: med(list(v.values())) for c, v in cells.items()}
    rep['_n_seeds'] = sorted({r['seed'] for r in rows})
    rep['_episodes'] = sorted(cells['ORACLE'])
    return rep


# =============================================================== F3/F4: W3
def f3_w3_ladder():
    rows = [r for r in load(W3) if 'avg_nRMSE' in r]
    rep = {}
    for stratum in ('crowded', 'sparse'):
        s = [r for r in rows if r['stratum'] == stratum]
        arms = {a: med3([r for r in s if r['arm'] == a])[0]
                for a in sorted({r['arm'] for r in s})}
        nseed = {a: collections.Counter(
            med3([r for r in s if r['arm'] == a])[1].values()) for a in arms}
        rep[stratum] = {}
        rep[stratum]['_medians'] = {a: med(list(v.values())) for a, v in arms.items()}
        rep[stratum]['_n_ep'] = {a: len(v) for a, v in arms.items()}
        rep[stratum]['_seeds_per_ep'] = {a: dict(c) for a, c in nseed.items()}
        cvx = arms.get('ORA-K-CVX', {})
        for a, v in arms.items():
            if a == 'ORA-K-CVX' or not cvx:
                continue
            shared = sorted(set(v) & set(cvx))
            if shared:
                rep[stratum]['ratio_vs_CVX|' + a] = dict(
                    n=len(shared),
                    ratio_of_medians=float(med([v[k] for k in shared])
                                           / med([cvx[k] for k in shared])),
                    median_paired_ratio=med([v[k] / cvx[k] for k in shared]))
        for treat, ctrl in [('ORA-K@L4', 'SHIP'), ('ORA-K-NOREG', 'ORA-K@L4'),
                            ('ORA-K-CVX', 'ORA-K-NOREG'), ('ORA-K-CVX', 'SHIP'),
                            ('ORA-K-NOREG', 'ORA-K@L3'), ('SHIP', 'ZERO')]:
            if treat in arms and ctrl in arms:
                rep[stratum]['%s_vs_%s' % (treat, ctrl)] = paired(
                    arms[treat], arms[ctrl], '%s < %s' % (treat, ctrl))
    return rep


# =============================================================== F5: k drift
def f5_drift():
    rows = [r for r in load(W3) if 'k_drift_rel_median' in r]
    rep = {'by_arm': {}, 'correlation': {}}
    for arm in sorted({r['arm'] for r in rows}):
        e = {}
        for stratum in ('crowded', 'sparse'):
            v = [r['k_drift_rel_median'] for r in rows
                 if r['arm'] == arm and r['stratum'] == stratum
                 and r.get('k_drift_valid') and r['k_drift_rel_median'] is not None]
            if v:
                e[stratum] = dict(n=len(v), median=med(v),
                                  p25=float(np.percentile(v, 25)),
                                  p75=float(np.percentile(v, 75)),
                                  max=float(np.max(v)))
        pool = [r['k_drift_rel_median'] for r in rows
                if r['arm'] == arm and r.get('k_drift_valid')
                and r['k_drift_rel_median'] is not None]
        allrows = [r for r in rows if r['arm'] == arm]
        e['pooled'] = dict(n=len(pool), median=med(pool),
                           max=float(np.max(pool)) if pool else float('nan'))
        e['_valid_rows'] = '%d/%d' % (len(pool), len(allrows))
        rep['by_arm'][arm] = e
    # pooled-over-arms figure (what a naive G-12 style print would give)
    allv = [r['k_drift_rel_median'] for r in rows
            if r.get('k_drift_valid') and r['k_drift_rel_median'] is not None]
    oracle = [r['k_drift_rel_median'] for r in rows
              if r.get('k_drift_valid') and r['k_drift_rel_median'] is not None
              and r['arm'].startswith('ORA-K')]
    rep['pooled_all_arms'] = dict(n=len(allv), median=med(allv), max=float(np.max(allv)))
    rep['pooled_oracle_arms'] = dict(n=len(oracle), median=med(oracle),
                                     max=float(np.max(oracle)))
    for arm in ('ORA-K-NOREG', 'ORA-K@L4', 'ORA-K@L3', 'SHIP', 'KONLY'):
        for stratum in ('crowded', 'sparse'):
            sub = [r for r in rows if r['arm'] == arm and r['stratum'] == stratum
                   and r.get('k_drift_valid') and r['k_drift_rel_median'] is not None
                   and np.isfinite(r.get('avg_nRMSE', np.nan))]
            if len(sub) < 8:
                continue
            d = [r['k_drift_rel_median'] for r in sub]
            e = [r['avg_nRMSE'] for r in sub]
            q = np.percentile(d, [25, 75])
            lo = [e[i] for i in range(len(d)) if d[i] <= q[0]]
            hi = [e[i] for i in range(len(d)) if d[i] >= q[1]]
            rep['correlation']['%s|%s' % (arm, stratum)] = dict(
                n=len(sub), pearson=float(np.corrcoef(d, e)[0, 1]),
                spearman=spearman(d, e),
                median_nRMSE_lowest_drift_quartile=med(lo),
                median_nRMSE_highest_drift_quartile=med(hi))
    return rep


# =============================================================== sample overlap
def f6_overlap():
    w3 = {norm_ep(r['episode']) for r in load(W3) if r['arm'] == 'SHIP'
          and r['stratum'] == 'crowded'}
    w3cvx = {norm_ep(r['episode']) for r in load(W3) if r['arm'] == 'ORA-K-CVX'
             and r['stratum'] == 'crowded'}
    s4 = {norm_ep(r['episode']) for r in load(S4) if r['cell'] == 'AUTO-NT'
          and r['regime'] == 'crowded'}
    k6 = {norm_ep(r['episode']) for r in load(K6) if r['cell'] == 'ORACLE'}
    seeds = dict(
        W3=sorted({r['est_seed'] for r in load(W3) if r['arm'] == 'SHIP'}),
        S4=sorted({r['seed'] for r in load(S4)}),
        K6=sorted({r['seed'] for r in load(K6)}))
    corpora = dict(
        W3=sorted({r['corpus'] for r in load(W3) if r['stratum'] == 'crowded'}),
        S4=sorted({r['episode'].split('/')[0] for r in load(S4)
                   if r['regime'] == 'crowded'}),
        K6='data_indep2_2026-08-07 (run_k6.py:74 K6_DATA default)')
    return dict(n=dict(W3_SHIP_crowded=len(w3), W3_CVX_crowded=len(w3cvx),
                       S4_crowded=len(s4), K6=len(k6)),
                overlap=dict(S4_x_K6=len(s4 & k6), W3_x_K6=len(w3 & k6),
                             W3_x_S4=len(w3 & s4), W3cvx_x_S4=len(w3cvx & s4)),
                estimator_seeds=seeds, corpora=corpora,
                K6_episodes=sorted(k6)[:4], S4_episodes=sorted(s4)[:4])


# =============================================================== product test
def f7_product_test(f1, f2, f3):
    """Is the joint gap the PRODUCT of the two singles, as PREREG §0.2 asserts?"""
    r_ab = f1['AUTO-NT|crowded|seed1557']['median_ratio']       # (a,b) only
    r_k = f2['ORACLE_vs_ORACLE-K']['median_ratio']              # k only (freeze)
    joint = f3['crowded']['ratio_vs_CVX|SHIP']['ratio_of_medians']
    return dict(
        factor_ab_only=r_ab, factor_k_only=r_k,
        product_of_singles=r_ab * r_k,
        implied_joint_ratio_from_product=1.0 / (r_ab * r_k),
        measured_joint_ratio_SHIP_over_CVX=joint,
        discrepancy_x=joint / (1.0 / (r_ab * r_k)),
        log_share_of_gap_explained_by_product=(
            math.log(1.0 / (r_ab * r_k)) / math.log(joint)))


def f8_within_sample():
    """The ONE factorial contrast that lives inside a single sample.

    Same 16 crowded episodes, same S4 seed, same lambda era, same tracking
    setting, same scorer, same closed-form (a,b) solver.  Only k changes:
        refit on the estimator's own final k   (PROBE-A `AUTO-NT`)
        refit on GT k                          (PROBE-A `ORA-WPHI-0STEP`)
    ⟹ isolates the k factor *while holding the (a,b) solver at closed form*.
    The complementary cell (frozen GT k + gradient-descent (a,b)) does NOT
    exist in this sample -- it exists only in K6, a disjoint corpus.
    """
    rows = load(os.path.join(HERE, 'B2_f1_refit.jsonl'))
    rep = {}
    for reg in ('crowded', 'sparse'):
        s = min(r['seed'] for r in rows)
        gt = {r['episode']: r['b2_refit_nRMSE'] for r in rows
              if r['cell'] == 'ORA-WPHI-0STEP' and r['regime'] == reg and r['seed'] == s}
        est = {r['episode']: r['b2_refit_nRMSE'] for r in rows
               if r['cell'] == 'AUTO-NT' and r['regime'] == reg and r['seed'] == s}
        raw = {r['episode']: r['stored_nRMSE'] for r in rows
               if r['cell'] == 'AUTO-NT' and r['regime'] == reg and r['seed'] == s}
        rep['k_factor_given_closedform_ab|' + reg] = paired(gt, est, 'GT k < est k')
        rep['ab_factor_given_est_k|' + reg] = paired(est, raw, 'closed form < GD')
        rep['joint_within_sample|' + reg] = paired(gt, raw, 'GT k+closed form < ship')
        # 2x2 with the missing cell filled from K6 (CROSS-BATCH -- flagged)
        rep['_note|' + reg] = ('missing cell (frozen GT k + GD (a,b)) is NOT in this '
                               'sample; K6 supplies it from a disjoint corpus')
    return rep


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('f1', 'all'):
        f1_closed_form_ab()
    if mode in ('rest', 'all'):
        f1 = f1_report()
        f2 = f2_k6()
        f3 = f3_w3_ladder()
        OUT.update(dict(F1_closed_form_ab=f1, F2_K6=f2, F3_W3_ladder=f3,
                        F5_k_drift=f5_drift(), F6_sample_overlap=f6_overlap(),
                        F8_within_sample=f8_within_sample(),
                        F7_product_test=f7_product_test(f1, f2, f3)))
        with open(os.path.join(HERE, 'B2_factors.json'), 'w') as fh:
            json.dump(OUT, fh, indent=1, sort_keys=True)
        print(json.dumps(OUT, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
