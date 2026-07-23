"""Assemble the paper tables from the 2026-07-24 finemesh + median-of-k re-run.

Reporting rule (docs/REPORTING_CONVENTION_median_of_k_2026-07-23.md, binding):
every per-episode number is the MEDIAN over k>=5 estimator seeds, quoted WITH its spread.
A median without a spread is not citable, so every cell here carries one.
Aggregates over episodes are exempt from the spread requirement in practice, but we report
the seed-noise floor alongside them so the reader can see which deltas are resolvable.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'ws_runs.jsonl')
ETKF = os.path.join(HERE, 'etkf_aspublished.json')
GPR = os.path.join(HERE, 'gpr_runs.jsonl')
OUT_JSON = os.path.join(HERE, 'tables_finemesh_medk.json')

PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
TABLE1_EPS = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS] + [
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_slope.h5']


def load_runs():
    by = defaultdict(list)
    with open(RUNS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r:
                continue
            by[(r['episode'], r['cfg_name'])].append(r)
    return by


def stat(rows, key='avg_nRMSE'):
    a = np.array([r[key] for r in rows], float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {'k': int(a.size), 'median': float(np.median(a)),
            'mean': float(a.mean()),
            'sd': float(a.std(ddof=1)) if a.size > 1 else 0.0,
            'min': float(a.min()), 'max': float(a.max()),
            'cv_pct': float(100 * a.std(ddof=1) / a.mean())
            if a.size > 1 and a.mean() else 0.0}


def cell(s, prec=3):
    if s is None:
        return '---'
    return f"{s['median']:.{prec}f}\\,$\\pm$\\,{s['sd']:.{prec}f}"


def short(ep):
    return (ep.replace('episode_', '').replace('.h5', '').replace('_seed0_', ' ')
            .replace('uniform', 'U').replace('slope', 'S').replace('comp', 'c'))


def main():
    by = load_runs()
    etkf = json.load(open(ETKF)) if os.path.exists(ETKF) else {}
    gpr = {}
    if os.path.exists(GPR):
        with open(GPR) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if 'error' not in r:
                        gpr[r['episode']] = r
                except json.JSONDecodeError:
                    pass

    out = {'per_episode': {}, 'aggregate': {}}

    # ---------------------------------------------------------------- Table 1
    print('=' * 100)
    print('TABLE 1 (finemesh, median of k seeds +- sd): WaveSplat vs ETKF')
    print('=' * 100)
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{WaveSplat vs ETKF-grid on 12 representative episodes, re-run on the')
    print(r'Nyquist-adequate regenerated dataset. Each WaveSplat entry is the \textbf{median')
    print(r'over 5 estimator seeds $\pm$ standard deviation}; single-seed numbers are not')
    print(r'reported because the run-to-run distribution is bimodal (Sec.~\ref{sec:limitations}).')
    print(r'ETKF uses the as-published configuration. nRMSE $=$ RMSE$/H_s$, lower is better.}')
    print(r'\label{tab:main-comparison}')
    print(r'\begin{tabular}{lcccc}')
    print(r'\toprule')
    print(r'Episode & WS-P0 (fixed $N$) & WS-P1 (adaptive $N$) & ETKF & $N_{\text{adapt}}$ \\')
    print(r'\midrule')

    agg = defaultdict(list)
    for ep in TABLE1_EPS:
        s0 = stat(by.get((ep, 'P0'), []))
        s1 = stat(by.get((ep, 'P1'), []))
        et = etkf.get(ep, {}).get('finemesh', {}).get('avg_nRMSE')
        ns = [r['n_gaussians'] for r in by.get((ep, 'P1'), []) if r.get('n_gaussians')]
        nadapt = f'{int(np.median(ns))}' if ns else '?'
        c0, c1 = cell(s0), cell(s1)
        if s0 and s1:
            if s1['median'] < s0['median']:
                c1 = f'\\textbf{{{c1}}}'
            else:
                c0 = f'\\textbf{{{c0}}}'
        print(f'{short(ep):<18} & {c0} & {c1} & '
              f'{et:.3f} & {nadapt} \\\\' if et is not None
              else f'{short(ep):<18} & {c0} & {c1} & --- & {nadapt} \\\\')
        out['per_episode'][ep] = {'WS-P0': s0, 'WS-P1': s1, 'ETKF_aspublished': et,
                                  'N_adapt_median': nadapt}
        if s0: agg['P0'].append(s0['median'])
        if s1: agg['P1'].append(s1['median'])
        if et is not None: agg['ETKF'].append(et)
        if s1: agg['P1_sd'].append(s1['sd'])

    print(r'\midrule')
    print(f"\\textbf{{Average}} & \\textbf{{{np.mean(agg['P0']):.3f}}} & "
          f"\\textbf{{{np.mean(agg['P1']):.3f}}} & "
          f"{np.mean(agg['ETKF']):.3f} & --- \\\\")
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')

    for k in ('P0', 'P1', 'ETKF'):
        out['aggregate'][k] = float(np.mean(agg[k]))
    out['aggregate']['seed_noise_sd_mean'] = float(np.mean(agg['P1_sd']))
    out['aggregate']['seed_noise_on_12ep_mean'] = float(
        np.mean(agg['P1_sd']) / np.sqrt(len(agg['P1'])))

    print(f"\n--- aggregate ---")
    print(f"WS-P0 {np.mean(agg['P0']):.4f} | WS-P1 {np.mean(agg['P1']):.4f} | "
          f"ETKF {np.mean(agg['ETKF']):.4f}")
    print(f"P1 vs ETKF: {(np.mean(agg['ETKF'])-np.mean(agg['P1']))/np.mean(agg['ETKF'])*100:+.1f}%")
    print(f"P1 vs P0:   {(np.mean(agg['P0'])-np.mean(agg['P1']))/np.mean(agg['P0'])*100:+.1f}%")
    print(f"mean per-episode seed sd = {np.mean(agg['P1_sd']):.4f} "
          f"=> noise floor on a 12-episode mean = "
          f"{np.mean(agg['P1_sd'])/np.sqrt(len(agg['P1'])):.4f}")
    print(f"WS-P1 wins vs ETKF: "
          f"{sum(1 for a, b in zip(agg['P1'], agg['ETKF']) if a < b)}/{len(agg['ETKF'])}")

    # ---------------------------------------------------------------- Table 2
    print('\n' + '=' * 100)
    print('TABLE 2 (finemesh, median of k seeds +- sd): ablation')
    print('=' * 100)
    abl_eps = [f'episode_BF3_{c}_seed0_{cond}.h5'
               for c in ('1comp', '2comp', '5comp') for cond in ('uniform', 'slope')]
    configs = ['Full', 'A1_noDisp', 'A2_noConf', 'v2+disp']
    print(f"{'episode':<16} " + ' '.join(f'{c:>22}' for c in configs))
    abl_agg = defaultdict(list)
    out['ablation'] = {}
    for ep in abl_eps:
        cells = []
        out['ablation'][ep] = {}
        for c in configs:
            s = stat(by.get((ep, c), []))
            out['ablation'][ep][c] = s
            cells.append('---' if s is None
                         else f"{s['median']:.4f}+-{s['sd']:.4f}")
            if s:
                abl_agg[c].append(s['median'])
        print(f'{short(ep):<16} ' + ' '.join(f'{x:>22}' for x in cells))
    print('-' * 100)
    print(f"{'AVERAGE':<16} " + ' '.join(
        f"{np.mean(abl_agg[c]):>22.4f}" if abl_agg[c] else f"{'---':>22}"
        for c in configs))
    if abl_agg['Full']:
        base = np.mean(abl_agg['Full'])
        for c in configs[1:]:
            if abl_agg[c]:
                print(f'  {c}: {np.mean(abl_agg[c]):.4f} '
                      f'({(np.mean(abl_agg[c])-base)/base*100:+.1f}% vs Full)')
        out['aggregate']['ablation'] = {c: float(np.mean(abl_agg[c]))
                                        for c in configs if abl_agg[c]}

    # ---------------------------------------------------------------- Table 4
    print('\n' + '=' * 100)
    print('TABLE 4 (finemesh, median of k seeds): raydrop = geometry-weighted vs unweighted')
    print('=' * 100)
    print(f"{'pair':<14}{'WS weighted':>20}{'WS unweighted':>20}{'WS d%':>9}")
    out['raydrop'] = {}
    du, ds = [], []
    for bf, c in PAIRS:
        u = stat(by.get((f'episode_{bf}_{c}_seed0_uniform.h5', 'P1'), []))
        s = stat(by.get((f'episode_{bf}_{c}_seed0_slope.h5', 'P1'), []))
        out['raydrop'][f'{bf}_{c}'] = {'weighted': u, 'unweighted': s}
        if u and s:
            d = (s['median'] - u['median']) / u['median'] * 100
            du.append(u['median'])
            ds.append(s['median'])
            print(f'{bf}_{c:<9}{u["median"]:>13.4f}+-{u["sd"]:.3f}'
                  f'{s["median"]:>13.4f}+-{s["sd"]:.3f}{d:>+8.1f}%')
    if du:
        print('-' * 63)
        print(f'{"AVERAGE":<14}{np.mean(du):>20.4f}{np.mean(ds):>20.4f}'
              f'{(np.mean(ds)-np.mean(du))/np.mean(du)*100:>+8.1f}%')
        print(f'unweighted better on {sum(1 for a, b in zip(du, ds) if b < a)}/{len(du)}')

    # ---------------------------------------------------------------- GPR
    if gpr:
        print('\n' + '=' * 100)
        print('GPR baseline (finemesh)')
        print('=' * 100)
        vals = []
        for ep, r in sorted(gpr.items()):
            print(f'{short(ep):<18} {r["avg_nRMSE"]:.4f}  ({r.get("avg_inference_ms", 0):.0f} ms/f)')
            vals.append(r['avg_nRMSE'])
        print(f'AVERAGE {np.mean(vals):.4f}')
        out['gpr_mean'] = float(np.mean(vals))
    else:
        print('\n(GPR not yet run)')

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {OUT_JSON}')


if __name__ == '__main__':
    main()
