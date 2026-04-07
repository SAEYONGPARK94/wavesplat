"""
generate_paper_tables.py
Generate LaTeX tables and summary statistics for the WaveSplat paper.

Tables:
1. Table 1: WaveSplat vs ETKF (main comparison, P1 results)
2. Table 2: Ablation study (A1: no dispersion, A2: no confidence)
3. Table 3: Adaptive N behavior (N_final per episode)
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {name} not found")
        return None
    with open(path) as f:
        return json.load(f)


def table1_main_comparison():
    """Table 1: WaveSplat-P1 vs ETKF-medium — 12 episodes."""
    data = load_json('p1_comparison.json')
    if data is None:
        return

    print("\n" + "="*80)
    print("TABLE 1: WaveSplat vs ETKF-grid (12 representative episodes)")
    print("="*80)

    # LaTeX table
    print(r"""
\begin{table}[t]
\centering
\caption{WaveSplat vs ETKF-grid comparison on 12 representative episodes
spanning BF2--BF5 sea states with 1--30 wave components.
nRMSE = RMSE / $H_s$, lower is better.}
\label{tab:main-comparison}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{WaveSplat (ours)} & \multicolumn{2}{c}{ETKF-grid} & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Episode & nRMSE & ms/f & nRMSE & ms/f & $N_{\text{adapt}}$ \\
\midrule""")

    ws_nrmses, etkf_nrmses, ws_speeds, etkf_speeds = [], [], [], []
    ws_wins = 0

    for ep, res in data.items():
        p1 = res.get('WS-P1', {})
        et = res.get('ETKF', {})

        p1_n = p1.get('avg_nRMSE', float('nan'))
        et_n = et.get('avg_nRMSE', float('nan'))
        p1_s = p1.get('avg_inference_ms', float('nan'))
        et_s = et.get('avg_inference_ms', float('nan'))
        p1_nn = p1.get('final_n', '?')

        # Bold the better nRMSE
        if not np.isnan(p1_n) and not np.isnan(et_n):
            if p1_n < et_n:
                p1_str = f"\\textbf{{{p1_n:.3f}}}"
                et_str = f"{et_n:.3f}"
                ws_wins += 1
            else:
                p1_str = f"{p1_n:.3f}"
                et_str = f"\\textbf{{{et_n:.3f}}}"
        else:
            p1_str = f"{p1_n:.3f}"
            et_str = f"{et_n:.3f}"

        # Format episode name
        ep_fmt = ep.replace('_seed0_', ' ').replace('uniform', 'U').replace('slope', 'S')

        print(f"{ep_fmt:<25} & {p1_str} & {p1_s:.1f} & {et_str} & {et_s:.1f} & {p1_nn} \\\\")

        if not np.isnan(p1_n): ws_nrmses.append(p1_n)
        if not np.isnan(et_n): etkf_nrmses.append(et_n)
        if not np.isnan(p1_s): ws_speeds.append(p1_s)
        if not np.isnan(et_s): etkf_speeds.append(et_s)

    ws_avg = np.mean(ws_nrmses) if ws_nrmses else float('nan')
    et_avg = np.mean(etkf_nrmses) if etkf_nrmses else float('nan')
    ws_s_avg = np.mean(ws_speeds) if ws_speeds else float('nan')
    et_s_avg = np.mean(etkf_speeds) if etkf_speeds else float('nan')

    print(r"\midrule")
    print(f"\\textbf{{Average}} & \\textbf{{{ws_avg:.3f}}} & {ws_s_avg:.1f} & "
          f"{et_avg:.3f} & {et_s_avg:.1f} & --- \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    improvement = (et_avg - ws_avg) / et_avg * 100
    speedup = et_s_avg / ws_s_avg
    print(f"\n--- Summary Stats ---")
    print(f"WaveSplat avg nRMSE: {ws_avg:.3f}")
    print(f"ETKF avg nRMSE: {et_avg:.3f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Speed: {ws_s_avg:.1f} vs {et_s_avg:.1f} ms/frame ({speedup:.1f}x faster)")
    print(f"WaveSplat wins: {ws_wins}/{len(data)} episodes")


def table2_ablation():
    """Table 2: Ablation study (3-seed average)."""
    data = load_json('ablation_seeded.json')
    if data is None:
        return

    print("\n" + "="*80)
    print("TABLE 2: Ablation Study (3-seed average over 6 BF3 episodes)")
    print("="*80)

    # Organize by episode and config
    configs = ['Full', 'A1_noDisp', 'A2_noConf']
    config_labels = {
        'Full': 'WaveSplat (full)',
        'A1_noDisp': 'w/o dispersion',
        'A2_noConf': 'w/o confidence',
    }

    episodes = {}
    for entry in data:
        ep = entry['episode']
        cfg = entry['config']
        if cfg not in configs:
            continue

        seeds = entry['seed_results']
        avg_nrmse = np.mean([s['avg_nRMSE'] for s in seeds])
        std_nrmse = np.std([s['avg_nRMSE'] for s in seeds])

        if ep not in episodes:
            episodes[ep] = {}
        episodes[ep][cfg] = (avg_nrmse, std_nrmse)

    print(r"""
\begin{table}[t]
\centering
\caption{Ablation study on BF3 episodes (3-seed average $\pm$ std).
A1 removes dispersion constraint ($\omega$ as free parameter),
A2 removes confidence weighting (uniform observation weights).}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Episode & Full & w/o Dispersion & w/o Confidence \\
\midrule""")

    full_vals, a1_vals, a2_vals = [], [], []

    for ep in sorted(episodes.keys()):
        cfg_data = episodes[ep]
        ep_fmt = ep.replace('episode_', '').replace('_seed0_', ' ').replace('.h5', '') \
                    .replace('uniform', 'U').replace('slope', 'S')

        cells = []
        for cfg in configs:
            if cfg in cfg_data:
                avg, std = cfg_data[cfg]
                cells.append(f"{avg:.3f}$\\pm${std:.3f}")
                if cfg == 'Full': full_vals.append(avg)
                elif cfg == 'A1_noDisp': a1_vals.append(avg)
                elif cfg == 'A2_noConf': a2_vals.append(avg)
            else:
                cells.append("---")

        print(f"{ep_fmt:<25} & {' & '.join(cells)} \\\\")

    print(r"\midrule")
    print(f"\\textbf{{Average}} & \\textbf{{{np.mean(full_vals):.3f}}} "
          f"& {np.mean(a1_vals):.3f} & {np.mean(a2_vals):.3f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print(f"\n--- Ablation Summary ---")
    print(f"Full: {np.mean(full_vals):.3f}")
    print(f"A1 (no dispersion): {np.mean(a1_vals):.3f} "
          f"(+{(np.mean(a1_vals)-np.mean(full_vals))/np.mean(full_vals)*100:.1f}%)")
    print(f"A2 (no confidence): {np.mean(a2_vals):.3f} "
          f"(+{(np.mean(a2_vals)-np.mean(full_vals))/np.mean(full_vals)*100:.1f}%)")


def table3_adaptive_n():
    """Table 3: Adaptive N behavior — shows how N adapts to scene complexity."""
    data = load_json('p1_comparison.json')
    if data is None:
        return

    print("\n" + "="*80)
    print("TABLE 3: Adaptive N — Model Complexity Adapts to Scene")
    print("="*80)

    print(r"""
\begin{table}[t]
\centering
\caption{Adaptive N behavior: the model automatically adjusts complexity
based on wave scene. $N_{\text{true}}$ is the ground-truth number of wave
components, $N_{\text{adapt}}$ is the model's final Gaussian count.}
\label{tab:adaptive-n}
\begin{tabular}{lcccc}
\toprule
Episode & $N_{\text{true}}$ & $N_{\text{adapt}}$ & nRMSE (P0, $N$=15) & nRMSE (P1, adaptive) \\
\midrule""")

    for ep, res in data.items():
        p0 = res.get('WS-P0', {})
        p1 = res.get('WS-P1', {})

        # Extract true N from episode name
        parts = ep.split('_')
        n_true = '?'
        for p in parts:
            if p.endswith('comp'):
                n_true = p.replace('comp', '')
                break

        p0_n = p0.get('avg_nRMSE', float('nan'))
        p1_n = p1.get('avg_nRMSE', float('nan'))
        p1_nn = p1.get('final_n', '?')

        ep_fmt = ep.replace('_seed0_', ' ').replace('uniform', 'U').replace('slope', 'S')

        # Bold if P1 is better
        if not np.isnan(p0_n) and not np.isnan(p1_n):
            if p1_n < p0_n:
                p1_str = f"\\textbf{{{p1_n:.3f}}}"
                p0_str = f"{p0_n:.3f}"
            else:
                p1_str = f"{p1_n:.3f}"
                p0_str = f"\\textbf{{{p0_n:.3f}}}"
        else:
            p0_str = f"{p0_n:.3f}"
            p1_str = f"{p1_n:.3f}"

        print(f"{ep_fmt:<25} & {n_true:>3} & {p1_nn:>3} & {p0_str} & {p1_str} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def summary_statistics():
    """Print key numbers for paper text."""
    data = load_json('p1_comparison.json')
    if data is None:
        return

    print("\n" + "="*80)
    print("KEY NUMBERS FOR PAPER TEXT")
    print("="*80)

    p0_vals, p1_vals, etkf_vals = [], [], []
    p0_speeds, p1_speeds, etkf_speeds = [], [], []

    for ep, res in data.items():
        p0 = res.get('WS-P0', {})
        p1 = res.get('WS-P1', {})
        et = res.get('ETKF', {})

        if 'avg_nRMSE' in p0: p0_vals.append(p0['avg_nRMSE'])
        if 'avg_nRMSE' in p1: p1_vals.append(p1['avg_nRMSE'])
        if 'avg_nRMSE' in et: etkf_vals.append(et['avg_nRMSE'])

        if 'avg_inference_ms' in p0: p0_speeds.append(p0['avg_inference_ms'])
        if 'avg_inference_ms' in p1: p1_speeds.append(p1['avg_inference_ms'])
        if 'avg_inference_ms' in et: etkf_speeds.append(et['avg_inference_ms'])

    print(f"\nP0 (fixed N=15) avg nRMSE: {np.mean(p0_vals):.3f}")
    print(f"P1 (adaptive N) avg nRMSE: {np.mean(p1_vals):.3f}")
    print(f"ETKF-medium avg nRMSE: {np.mean(etkf_vals):.3f}")

    print(f"\nP1 vs ETKF: {(np.mean(etkf_vals) - np.mean(p1_vals))/np.mean(etkf_vals)*100:.1f}% better")
    print(f"P1 vs P0: {(np.mean(p0_vals) - np.mean(p1_vals))/np.mean(p0_vals)*100:.1f}% better")

    print(f"\nSpeed: P0={np.mean(p0_speeds):.1f}ms, P1={np.mean(p1_speeds):.1f}ms, ETKF={np.mean(etkf_speeds):.1f}ms")
    print(f"Speedup vs ETKF: {np.mean(etkf_speeds)/np.mean(p1_speeds):.1f}x")

    # Best/worst cases
    p1_data = {ep: res['WS-P1']['avg_nRMSE'] for ep, res in data.items()
               if 'WS-P1' in res and 'avg_nRMSE' in res['WS-P1']}
    etkf_data = {ep: res['ETKF']['avg_nRMSE'] for ep, res in data.items()
                 if 'ETKF' in res and 'avg_nRMSE' in res['ETKF']}

    best_ep = min(p1_data, key=p1_data.get)
    worst_ep = max(p1_data, key=p1_data.get)
    print(f"\nBest WaveSplat episode: {best_ep} (nRMSE={p1_data[best_ep]:.3f})")
    print(f"Worst WaveSplat episode: {worst_ep} (nRMSE={p1_data[worst_ep]:.3f})")

    # Biggest advantage over ETKF
    advantages = {}
    for ep in p1_data:
        if ep in etkf_data:
            advantages[ep] = etkf_data[ep] - p1_data[ep]
    best_adv = max(advantages, key=advantages.get)
    worst_adv = min(advantages, key=advantages.get)
    print(f"\nBiggest advantage over ETKF: {best_adv} "
          f"(WS={p1_data[best_adv]:.3f} vs ETKF={etkf_data[best_adv]:.3f})")
    if advantages[worst_adv] < 0:
        print(f"ETKF wins on: {worst_adv} "
              f"(WS={p1_data[worst_adv]:.3f} vs ETKF={etkf_data[worst_adv]:.3f})")


def table4_raydrop():
    """Table 4: Raydrop model comparison (uniform vs slope)."""
    data = load_json('raydrop_comparison.json')
    if data is None:
        return

    print("\n" + "=" * 80)
    print("TABLE 4: Raydrop Model Comparison (Uniform vs Slope)")
    print("=" * 80)

    ws_u_all, ws_s_all, et_u_all, et_s_all = [], [], [], []

    print(f"\n{'Pair':<15} {'WS-U':>8} {'WS-S':>8} {'WS Δ%':>8} "
          f"{'ETKF-U':>8} {'ETKF-S':>8} {'ETKF Δ%':>9}")
    print("-" * 70)

    for pair_key, v in data.items():
        u = v.get('uniform', {})
        s = v.get('slope', {})

        ws_u = u.get('WS-P1', {}).get('avg_nRMSE', float('nan'))
        ws_s = s.get('WS-P1', {}).get('avg_nRMSE', float('nan'))
        et_u = u.get('ETKF', {}).get('avg_nRMSE', float('nan'))
        et_s = s.get('ETKF', {}).get('avg_nRMSE', float('nan'))

        ws_d = (ws_s - ws_u) / ws_u * 100 if ws_u > 0 else float('nan')
        et_d = (et_s - et_u) / et_u * 100 if et_u > 0 else float('nan')

        print(f"{pair_key:<15} {ws_u:>8.3f} {ws_s:>8.3f} {ws_d:>+7.1f}% "
              f"{et_u:>8.3f} {et_s:>8.3f} {et_d:>+8.1f}%")

        ws_u_all.append(ws_u)
        ws_s_all.append(ws_s)
        et_u_all.append(et_u)
        et_s_all.append(et_s)

    print("-" * 70)
    ws_u_avg, ws_s_avg = np.mean(ws_u_all), np.mean(ws_s_all)
    et_u_avg, et_s_avg = np.mean(et_u_all), np.mean(et_s_all)
    print(f"{'AVERAGE':<15} {ws_u_avg:>8.3f} {ws_s_avg:>8.3f} "
          f"{(ws_s_avg-ws_u_avg)/ws_u_avg*100:>+7.1f}% "
          f"{et_u_avg:>8.3f} {et_s_avg:>8.3f} "
          f"{(et_s_avg-et_u_avg)/et_u_avg*100:>+8.1f}%")

    print(f"\n--- Raydrop Summary ---")
    print(f"WS slope better: {sum(1 for u,s in zip(ws_u_all,ws_s_all) if s<u)}/{len(ws_u_all)}")
    print(f"ETKF slope better: {sum(1 for u,s in zip(et_u_all,et_s_all) if s<u)}/{len(et_u_all)}")
    print(f"WS wins on uniform: {sum(1 for w,e in zip(ws_u_all,et_u_all) if w<e)}/{len(ws_u_all)}")
    print(f"WS wins on slope: {sum(1 for w,e in zip(ws_s_all,et_s_all) if w<e)}/{len(ws_s_all)}")


def main():
    table1_main_comparison()
    table2_ablation()
    table3_adaptive_n()
    table4_raydrop()
    summary_statistics()


if __name__ == '__main__':
    main()
