"""Analysis for the 2026-08-05 timing re-measurement. Pre-registration: ./PREREG.md

Every aggregation rule here was fixed in PREREG.md BEFORE the run:
  * median is PRIMARY (the CLOCK_REALTIME step contaminates means, not medians)
  * PRIMARY frame set = (frame index >= 5) AND (frame_t >= t_init_end)
  * ratios are paired per EPISODE (n=12), CI by percentile bootstrap over episodes
"""
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.jsonl')
ARMS = ['WS-P1', 'ETKF', 'LETKF', 'GPR-1', 'GPR-disp']
WARMUP = 5
BUDGET_MS = 100.0
RNG = np.random.default_rng(0)
NBOOT = 10000


def load():
    prov, runs = None, []
    for line in open(RES):
        r = json.loads(line)
        if r.get('kind') == 'provenance':
            prov = r
        elif 'error' not in r:
            runs.append(r)
        else:
            print('ERROR RUN:', r.get('job_id'), r.get('error'))
    return prov, runs


def primary_mask(r):
    t = np.asarray(r['frame_t'], float)
    idx = np.arange(len(t))
    return (idx >= WARMUP) & (t >= r['t_init_end'] - 1e-9)


def sign_p(k, n):
    """Two-sided exact sign test p-value for k successes out of n."""
    if n == 0:
        return float('nan')
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(vals, stat=np.median, n=NBOOT):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float('nan'), float('nan'))
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    b = stat(v[idx], axis=1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def main():
    prov, runs = load()
    eps = sorted({r['episode'] for r in runs})
    out = {'provenance': prov, 'n_runs': len(runs), 'n_episodes': len(eps),
           'warmup_frames_excluded': WARMUP, 'budget_ms': BUDGET_MS}

    # ---- per-run summaries ---------------------------------------------------
    per_run = []
    for r in runs:
        m = primary_mask(r)
        upd = np.asarray(r['t_update_ms'], float)[m]
        frm = np.asarray(r['t_frame_ms'], float)[m]
        prd = np.asarray(r['t_predict_ms'], float)[m]
        itp = np.asarray(r['t_interp_ms'], float)[m]
        prod = np.asarray(r['prod_elapsed_ms'], float)[m]
        per_run.append({
            'arm': r['arm'], 'episode': r['episode'], 'rep': r['rep'],
            'n_frames_primary': int(m.sum()), 'n_frames_run': len(r['frame_t']),
            'upd_med': float(np.median(upd)), 'upd_p95': float(np.percentile(upd, 95)),
            'upd_mean': float(np.mean(upd)),
            'frame_med': float(np.median(frm)), 'frame_p95': float(np.percentile(frm, 95)),
            'frame_mean': float(np.mean(frm)),
            'pred_med': float(np.median(prd)), 'interp_med': float(np.median(itp)),
            'prod_med': float(np.median(prod)), 'prod_mean': float(np.mean(prod)),
            'init_s': float(r['init_monotonic_s']),
            'init_real_s': float(r['init_realtime_s']),
            'init_prod_s': r.get('init_prod_reported_s'),
            'run_clock_bias_pct': float(r['run_clock_bias_pct']),
            'n_steps': int(r['n_steps_swallowed']),
            'final_n': r.get('final_n'), 'wall_s': r['wall_s'],
            'n_pts_med': float(np.median(np.asarray(r['n_pts'], float)[m])),
        })
    out['per_run'] = per_run

    def ep_val(arm, ep, key):
        v = [p[key] for p in per_run if p['arm'] == arm and p['episode'] == ep]
        return float(np.median(v)) if v else float('nan')

    # ---- M1: arm-level per-frame latency ------------------------------------
    m1 = {}
    for arm in ARMS:
        rr = [r for r in runs if r['arm'] == arm]
        pool_u, pool_f = [], []
        for r in rr:
            m = primary_mask(r)
            pool_u.append(np.asarray(r['t_update_ms'], float)[m])
            pool_f.append(np.asarray(r['t_frame_ms'], float)[m])
        pu = np.concatenate(pool_u)
        pf = np.concatenate(pool_f)
        ep_med = [ep_val(arm, e, 'upd_med') for e in eps]
        ep_fmed = [ep_val(arm, e, 'frame_med') for e in eps]
        ep_p95 = [ep_val(arm, e, 'frame_p95') for e in eps]
        lo, hi = boot_ci(ep_med)
        m1[arm] = {
            'n_frames_pooled': int(pu.size),
            'update_median_pooled': float(np.median(pu)),
            'update_p95_pooled': float(np.percentile(pu, 95)),
            'update_mean_pooled': float(np.mean(pu)),
            'update_median_of_episode_medians': float(np.median(ep_med)),
            'update_median_ci95': [lo, hi],
            'update_episode_median_range': [float(np.min(ep_med)), float(np.max(ep_med))],
            'frame_median_pooled': float(np.median(pf)),
            'frame_p95_pooled': float(np.percentile(pf, 95)),
            'frame_mean_pooled': float(np.mean(pf)),
            'frame_median_of_episode_medians': float(np.median(ep_fmed)),
            'frame_p95_max_over_episodes': float(np.max(ep_p95)),
            'predict_median': float(np.median([ep_val(arm, e, 'pred_med') for e in eps])),
            'interp_median': float(np.median([ep_val(arm, e, 'interp_med') for e in eps])),
        }
    out['M1_per_frame'] = m1

    # ---- M2: paired per-episode ratio vs WS ---------------------------------
    m2 = {}
    for arm in ARMS:
        if arm == 'WS-P1':
            continue
        for key, lbl in (('upd_med', 'update_only'), ('frame_med', 'update_plus_field')):
            ratios = np.array([ep_val(arm, e, key) / ep_val('WS-P1', e, key) for e in eps])
            lo, hi = boot_ci(ratios)
            k = int(np.sum(ratios > 1.0))
            m2.setdefault(arm, {})[lbl] = {
                'ratio_median': float(np.median(ratios)),
                'ratio_ci95': [lo, hi],
                'ratio_min': float(ratios.min()), 'ratio_max': float(ratios.max()),
                'n_episodes_slower_than_WS': k, 'n': len(ratios),
                'sign_test_p_two_sided': sign_p(k, len(ratios)),
                'per_episode': {e: float(v) for e, v in zip(eps, ratios)},
            }
    out['M2_ratio_vs_WS'] = m2

    # ---- M3: 100 ms budget ---------------------------------------------------
    m3 = {'criterion': 'per-frame (update + field production) p95 <= 100 ms', 'arms': {}}
    npass = 0
    for arm in ARMS:
        p95 = m1[arm]['frame_p95_pooled']
        worst = m1[arm]['frame_p95_max_over_episodes']
        ok = worst <= BUDGET_MS
        npass += int(ok)
        pool = np.concatenate([np.asarray(r['t_frame_ms'], float)[primary_mask(r)]
                               for r in runs if r['arm'] == arm])
        m3['arms'][arm] = {'frame_p95_pooled_ms': p95,
                           'frame_p95_worst_episode_ms': worst,
                           'update_p95_pooled_ms': m1[arm]['update_p95_pooled'],
                           'frame_max_pooled_ms': float(pool.max()),
                           'frac_frames_over_budget': float((pool > BUDGET_MS).mean()),
                           'passes': bool(ok)}
    m3['n_pass'] = npass
    m3['n_arms'] = len(ARMS)
    m3['discriminative'] = npass not in (0, len(ARMS))
    out['M3_budget'] = m3

    # ---- M4: initialisation latency -----------------------------------------
    m4 = {}
    for arm in ARMS:
        v = np.array([ep_val(arm, e, 'init_s') for e in eps])
        lo, hi = boot_ci(v)
        m4[arm] = {'init_s_median': float(np.median(v)),
                   'init_s_ci95': [lo, hi],
                   'init_s_min': float(v.min()), 'init_s_max': float(v.max()),
                   'per_episode': {e: float(x) for e, x in zip(eps, v)}}
    ws = m4['WS-P1']['init_s_median']
    for arm in ARMS:
        if arm != 'WS-P1':
            m4[arm]['ws_init_over_this_arm'] = (ws / m4[arm]['init_s_median']
                                                if m4[arm]['init_s_median'] > 0 else float('inf'))
    m4['WS_init_over_own_frame_cost'] = ws * 1000.0 / m1['WS-P1']['frame_median_of_episode_medians']
    out['M4_init'] = m4

    # ---- M5: clock bias (time.time vs perf_counter) -------------------------
    m5 = {'predicted_pct': -3.86, 'arms': {}}
    for arm in ARMS:
        b = np.array([p['run_clock_bias_pct'] for p in per_run if p['arm'] == arm])
        lo, hi = boot_ci(b, stat=np.mean)
        # contamination of the paper's aggregator: mean vs median of production ms
        pm = np.array([p['prod_mean'] for p in per_run if p['arm'] == arm])
        pmed = np.array([p['prod_med'] for p in per_run if p['arm'] == arm])
        tm = np.array([p['upd_mean'] for p in per_run if p['arm'] == arm])
        tmed = np.array([p['upd_med'] for p in per_run if p['arm'] == arm])
        m5['arms'][arm] = {
            'run_clock_bias_pct_mean': float(b.mean()),
            'run_clock_bias_pct_ci95': [lo, hi],
            'run_clock_bias_pct_min': float(b.min()), 'max': float(b.max()),
            'prod_mean_vs_perf_mean_pct': float(100 * np.mean(pm / tm - 1)),
            'prod_median_vs_perf_median_pct': float(100 * np.mean(pmed / tmed - 1)),
        }
    allb = np.array([p['run_clock_bias_pct'] for p in per_run])
    m5['all_runs_mean_pct'] = float(allb.mean())
    m5['all_runs_ci95'] = list(boot_ci(allb, stat=np.mean))
    # between-arm spread: is the bias arm-independent (A-1) ?
    means = [m5['arms'][a]['run_clock_bias_pct_mean'] for a in ARMS]
    m5['between_arm_spread_pct'] = float(np.max(means) - np.min(means))
    out['M5_clock_bias'] = m5

    # ---- M6: step swallowing -------------------------------------------------
    steps = np.array([p['n_steps'] for p in per_run])
    out['M6_steps'] = {
        'total_steps_detected': int(steps.sum()),
        'n_runs': len(steps),
        'n_runs_with_at_least_one': int((steps > 0).sum()),
        'per_arm': {a: int(sum(p['n_steps'] for p in per_run if p['arm'] == a))
                    for a in ARMS},
        'total_monotonic_s': float(sum(p['wall_s'] for p in per_run)),
    }

    # ---- paper-convention number (run.py:274 mean of time.time ms) ----------
    out['paper_convention'] = {
        a: {'mean_of_production_time_time_ms':
            float(np.median([p['prod_mean'] for p in per_run if p['arm'] == a])),
            'corrected_median_perf_counter_ms': m1[a]['update_median_of_episode_medians']}
        for a in ARMS}

    with open(os.path.join(HERE, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # ---------------------------- console report -----------------------------
    P = print
    P('=' * 86)
    P('TIMING RE-MEASUREMENT 2026-08-05 | perf_counter | 1 thread | 1 process | serial')
    P('=' * 86)
    P(f"runs {len(runs)} | episodes {len(eps)} | reps 3 | warmup excluded {WARMUP} frames")
    P(f"CPU {prov['cpu_model']} ({prov['nproc']} logical) | torch threads {prov['torch_num_threads']}")
    P(f"python {prov['python']} numpy {prov['numpy']} torch {prov['torch']} "
      f"scipy {prov['scipy']} sklearn {prov['sklearn']}")
    P('')
    P('--- M1  per-frame update() latency [ms], PRIMARY = median ---')
    P(f"{'arm':<10}{'median':>9}{'CI95':>18}{'p95':>9}{'mean':>9}{'ep-range':>18}{'frames':>9}")
    for a in ARMS:
        d = m1[a]
        P(f"{a:<10}{d['update_median_of_episode_medians']:9.2f}"
          f"{'[%.2f, %.2f]' % tuple(d['update_median_ci95']):>18}"
          f"{d['update_p95_pooled']:9.2f}{d['update_mean_pooled']:9.2f}"
          f"{'%.2f-%.2f' % tuple(d['update_episode_median_range']):>18}"
          f"{d['n_frames_pooled']:9d}")
    P('')
    P('--- M1b per-frame TOTAL (update + field on GT grid) [ms] ---')
    P(f"{'arm':<10}{'median':>9}{'p95':>9}{'mean':>9}{'predict':>9}{'interp':>9}")
    for a in ARMS:
        d = m1[a]
        P(f"{a:<10}{d['frame_median_of_episode_medians']:9.2f}{d['frame_p95_pooled']:9.2f}"
          f"{d['frame_mean_pooled']:9.2f}{d['predict_median']:9.2f}{d['interp_median']:9.2f}")
    P('')
    P('--- M2  ratio vs WS-P1 (paired per episode, n=12) ---')
    for lbl in ('update_only', 'update_plus_field'):
        P(f"  [{lbl}]")
        P(f"  {'arm':<10}{'ratio':>8}{'CI95':>18}{'range':>16}{'slower':>9}{'sign p':>10}")
        for a in ARMS:
            if a == 'WS-P1':
                continue
            d = m2[a][lbl]
            P(f"  {a:<10}{d['ratio_median']:8.2f}"
              f"{'[%.2f, %.2f]' % tuple(d['ratio_ci95']):>18}"
              f"{'%.2f-%.2f' % (d['ratio_min'], d['ratio_max']):>16}"
              f"{'%d/%d' % (d['n_episodes_slower_than_WS'], d['n']):>9}"
              f"{d['sign_test_p_two_sided']:10.2e}")
    P('')
    P(f"--- M3  {BUDGET_MS:.0f} ms budget (criterion: {m3['criterion']}) ---")
    for a in ARMS:
        d = m3['arms'][a]
        P(f"  {a:<10} p95 {d['frame_p95_pooled_ms']:7.2f} ms | worst-ep p95 "
          f"{d['frame_p95_worst_episode_ms']:7.2f} ms | max {d['frame_max_pooled_ms']:8.2f} ms"
          f" | over-budget {100*d['frac_frames_over_budget']:6.2f}% | "
          f"{'PASS' if d['passes'] else 'FAIL'}")
    P(f"  -> {m3['n_pass']}/{m3['n_arms']} arms pass. "
      f"DISCRIMINATIVE = {m3['discriminative']}")
    P('')
    P('--- M4  initialisation latency [s] ---')
    P(f"{'arm':<10}{'median':>10}{'CI95':>20}{'min':>10}{'max':>10}")
    for a in ARMS:
        d = m4[a]
        P(f"{a:<10}{d['init_s_median']:10.3f}"
          f"{'[%.3f, %.3f]' % tuple(d['init_s_ci95']):>20}"
          f"{d['init_s_min']:10.3f}{d['init_s_max']:10.3f}")
    P(f"  WS init / WS per-frame cost = {m4['WS_init_over_own_frame_cost']:.0f}x")
    P('')
    P('--- M5  clock bias: time.time vs perf_counter (predicted -3.86%) ---')
    for a in ARMS:
        d = m5['arms'][a]
        P(f"  {a:<10} run bias {d['run_clock_bias_pct_mean']:+7.3f}% "
          f"CI [{d['run_clock_bias_pct_ci95'][0]:+.3f}, {d['run_clock_bias_pct_ci95'][1]:+.3f}] | "
          f"prod MEAN {d['prod_mean_vs_perf_mean_pct']:+7.3f}% | "
          f"prod MEDIAN {d['prod_median_vs_perf_median_pct']:+7.3f}%")
    P(f"  all runs {m5['all_runs_mean_pct']:+.3f}% "
      f"CI [{m5['all_runs_ci95'][0]:+.3f}, {m5['all_runs_ci95'][1]:+.3f}] | "
      f"between-arm spread {m5['between_arm_spread_pct']:.3f} pp")
    P('')
    P('--- M6  CLOCK_REALTIME backward steps swallowed ---')
    d = out['M6_steps']
    P(f"  {d['total_steps_detected']} steps in {d['n_runs']} runs "
      f"({d['n_runs_with_at_least_one']} runs affected), "
      f"{d['total_monotonic_s']/60:.1f} min total | per arm {d['per_arm']}")
    P('')
    P('--- paper convention vs corrected ---')
    P(f"{'arm':<10}{'mean(time.time)':>18}{'median(perf_counter)':>22}")
    for a in ARMS:
        d = out['paper_convention'][a]
        P(f"{a:<10}{d['mean_of_production_time_time_ms']:18.2f}"
          f"{d['corrected_median_perf_counter_ms']:22.2f}")
    P('=' * 86)


if __name__ == '__main__':
    main()
