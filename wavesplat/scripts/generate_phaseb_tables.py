"""
generate_phaseb_tables.py
Phase C: Generate LaTeX tables from Phase B results (6-metric panel, sweep, ablation).

Tables:
  T1. Main 6-metric comparison (nRMSE / Hs_err / VR / rho / SS / Directional_R2)
      across methods on the full 100-episode dataset (from b1_full_comparison.json).
  T2. Summary by sea state (BF2/3/4/5 averages).
  T3. ETKF tuning sweep top-10 configs (from etkf_tuning_sweep.json).
  T4. BF4-5 dispersion ablation (from ablation_bf45.json).
  T5. Extended multi-seed Wilcoxon (from multiseed_extended.json).

Prints both a plaintext console summary and LaTeX snippets.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"[missing] {name}")
        return None
    with open(path) as f:
        return json.load(f)


def t1_main_metric_panel(data):
    """6-metric panel: mean across all episodes per method."""
    print("\n" + "=" * 90)
    print("T1. Pre-registered 6-metric panel — full-dataset mean per method")
    print("=" * 90)
    if not data:
        return
    results = data.get('results', data)  # handle both wrapper and raw

    methods = ['WaveSplat', 'ETKF', 'GPR-1', 'GPR-3']
    metric_keys = [
        ('avg_nRMSE', 'nRMSE↓'),
        ('avg_Hs_error', 'Hs_err↓'),
        ('avg_Variance_Ratio', 'VR→1'),
        ('avg_Pearson_rho', 'ρ↑'),
        ('avg_Skill_Score', 'SS↑'),
        ('avg_Directional_R2', 'Dir.R²↑'),
        ('avg_RMSE_far', 'RMSE_far↓'),
        ('avg_inference_ms', 'ms/f'),
    ]
    header = f"{'Method':<12}" + ''.join(f"{l:>12}" for _, l in metric_keys)
    print(header)
    print('-' * len(header))
    for meth in methods:
        row = f"{meth:<12}"
        for key, _ in metric_keys:
            vals = []
            for ep, res in results.items():
                m = res.get(meth, {})
                v = m.get(key)
                if isinstance(v, (int, float)) and not np.isnan(v):
                    vals.append(v)
            mean = np.mean(vals) if vals else float('nan')
            row += f"{mean:>12.4f}"
        print(row)

    # LaTeX
    print(r"""
\begin{table*}[t]
\centering
\caption{Pre-registered six-metric evaluation on the full 100-episode dataset.
VR = variance ratio (target 1), $\rho$ = Pearson correlation,
SS = skill score vs zero predictor, Dir.\ $R^2$ = directional
spectral agreement. Lower nRMSE, $H_s$ error, and RMSE$_{\text{far}}$ are better.}
\label{tab:6metric}
\begin{tabular}{lcccccc}
\toprule
Method & nRMSE $\downarrow$ & $H_s$ err $\downarrow$ & VR & $\rho$ $\uparrow$ &
SS $\uparrow$ & Dir.\ $R^2$ $\uparrow$ \\
\midrule""")
    for meth in methods:
        cols = [meth]
        for key, _ in metric_keys[:6]:
            vals = [res.get(meth, {}).get(key) for res in results.values()]
            vals = [v for v in vals if isinstance(v, (int, float))
                    and not np.isnan(v)]
            mean = np.mean(vals) if vals else float('nan')
            cols.append(f"{mean:.3f}")
        print(' & '.join(cols) + r' \\')
    print(r"""\bottomrule
\end{tabular}
\end{table*}""")


def t2_by_sea_state(data):
    print("\n" + "=" * 90)
    print("T2. Summary by sea state (nRMSE mean per BF)")
    print("=" * 90)
    if not data:
        return
    results = data.get('results', data)
    methods = ['WaveSplat', 'ETKF', 'GPR-1', 'GPR-3']
    by_bf = defaultdict(lambda: defaultdict(list))
    for ep, res in results.items():
        bf = ep.split('_')[0]
        for meth in methods:
            v = res.get(meth, {}).get('avg_nRMSE')
            if isinstance(v, (int, float)) and not np.isnan(v):
                by_bf[bf][meth].append(v)
    bfs = sorted(by_bf.keys())
    header = f"{'BF':<6}" + ''.join(f"{m:>12}" for m in methods) + f"{'N_ep':>8}"
    print(header)
    for bf in bfs:
        row = f"{bf:<6}"
        n_ep = 0
        for meth in methods:
            vs = by_bf[bf][meth]
            row += f"{np.mean(vs) if vs else float('nan'):>12.4f}"
            n_ep = max(n_ep, len(vs))
        row += f"{n_ep:>8}"
        print(row)


def t3_etkf_sweep_top(data):
    print("\n" + "=" * 90)
    print("T3. ETKF sweep — top 10 configs")
    print("=" * 90)
    if not data:
        return
    ranked = sorted(
        data.items(),
        key=lambda kv: kv[1].get('_mean_nRMSE', 1e9))
    print(f"{'Rank':<6}{'Config':<45}{'mean nRMSE':>12}")
    for rank, (key, cfg) in enumerate(ranked[:10], 1):
        print(f"{rank:<6}{key:<45}{cfg.get('_mean_nRMSE', float('nan')):>12.4f}")


def t4_bf45_ablation(data):
    print("\n" + "=" * 90)
    print("T4. BF4-5 dispersion ablation")
    print("=" * 90)
    if not data:
        return
    cfg_means = defaultdict(list)
    cfg_hs = defaultdict(list)
    for r in data:
        cfg = r['config']
        valid = [m for m in r['seed_results'] if 'error' not in m]
        if valid:
            cfg_means[cfg].append(np.mean([m['avg_nRMSE'] for m in valid]))
            cfg_hs[cfg].append(np.mean([m['avg_Hs_error'] for m in valid]))
    print(f"{'Config':<15}{'mean nRMSE':>14}{'std':>10}{'mean Hs_err':>14}{'n_ep':>8}")
    for cfg in ['Full', 'A1_noDisp', 'A2_noConf']:
        vs = cfg_means.get(cfg, [])
        hss = cfg_hs.get(cfg, [])
        if vs:
            print(f"{cfg:<15}{np.mean(vs):>14.4f}{np.std(vs):>10.4f}"
                  f"{np.mean(hss):>14.4f}{len(vs):>8}")

    # Breakdown by BF4 vs BF5
    print("\nBreakdown by sea state:")
    for bf in ['BF4', 'BF5']:
        print(f"\n  {bf}:")
        for cfg in ['Full', 'A1_noDisp', 'A2_noConf']:
            vs = []
            hss = []
            for r in data:
                if r['config'] == cfg and r['episode'].startswith(bf):
                    valid = [m for m in r['seed_results'] if 'error' not in m]
                    if valid:
                        vs.append(np.mean([m['avg_nRMSE'] for m in valid]))
                        hss.append(np.mean([m['avg_Hs_error'] for m in valid]))
            if vs:
                print(f"    {cfg:<12} nRMSE={np.mean(vs):.4f}±{np.std(vs):.4f} "
                      f"Hs={np.mean(hss):.3f} (n={len(vs)})")


def t5_multiseed_extended(data):
    print("\n" + "=" * 90)
    print("T5. Extended multi-seed statistics")
    print("=" * 90)
    if not data:
        return
    results = data.get('results', data)
    ws_n, et_n, gp_n = [], [], []
    ws_vr, et_vr, gp_vr = [], [], []
    for ep, res in results.items():
        if 'error' in res:
            continue
        ws = res.get('WS', {})
        et = res.get('ETKF', {})
        gp = res.get('GPR-3', {})
        if ws and et and gp:
            ws_n.append(ws.get('nRMSE', np.nan))
            et_n.append(et.get('nRMSE', np.nan))
            gp_n.append(gp.get('nRMSE', np.nan))
            ws_vr.append(ws.get('Variance_Ratio', np.nan))
            et_vr.append(et.get('Variance_Ratio', np.nan))
            gp_vr.append(gp.get('Variance_Ratio', np.nan))
    print(f"Paired runs: {len(ws_n)}")
    print(f"  WS   nRMSE={np.nanmean(ws_n):.4f}±{np.nanstd(ws_n):.4f}  "
          f"VR={np.nanmean(ws_vr):.3f}")
    print(f"  ETKF nRMSE={np.nanmean(et_n):.4f}±{np.nanstd(et_n):.4f}  "
          f"VR={np.nanmean(et_vr):.3f}")
    print(f"  GPR3 nRMSE={np.nanmean(gp_n):.4f}±{np.nanstd(gp_n):.4f}  "
          f"VR={np.nanmean(gp_vr):.3f}")

    from scipy.stats import wilcoxon
    try:
        ws_a, et_a, gp_a = map(np.array, (ws_n, et_n, gp_n))
        mask = ~(np.isnan(ws_a) | np.isnan(et_a) | np.isnan(gp_a))
        if mask.sum() > 5:
            w_e, p_e = wilcoxon(ws_a[mask], et_a[mask])
            w_g, p_g = wilcoxon(ws_a[mask], gp_a[mask])
            print(f"\nWilcoxon signed-rank (n={mask.sum()}):")
            print(f"  WS vs ETKF:  W={w_e:.1f} p={p_e:.4g}")
            print(f"  WS vs GPR-3: W={w_g:.1f} p={p_g:.4g}")
    except Exception as e:
        print(f"  wilcoxon error: {e}")


def main():
    b1 = load('b1_full_comparison.json')
    sweep = load('etkf_tuning_sweep.json')
    abl = load('ablation_bf45.json')
    mext = load('multiseed_extended.json')

    t1_main_metric_panel(b1)
    t2_by_sea_state(b1)
    t3_etkf_sweep_top(sweep)
    t4_bf45_ablation(abl)
    t5_multiseed_extended(mext)


if __name__ == '__main__':
    main()
