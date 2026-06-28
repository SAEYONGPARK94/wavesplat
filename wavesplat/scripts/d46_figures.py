"""
Comparison figures for the paper — produced from the committed result JSONs.

Panels:
  A. Equal-budget comparison on the DISCRETE benchmark (RELAX vs DPS vs GP vs oracle).
  B. THE story flip: RELAX vs DPS on DISCRETE benchmark vs CONTINUOUS (HOS) spectra.
  C. Observation-scarcity crossover (learned prior vs RELAX).
  D. Free-aim active sensing budget curve (random vs learned vs greedy) — honest negative.

Run: python3 scripts/d46_figures.py
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

R = 'results/direction3/'


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def sea_order(keys):
    def k(s):
        bf = int(s.split('_')[0][2]); nc = int(s.split('_')[1].replace('c', ''))
        return (bf, nc)
    return sorted(keys, key=k)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- A. equal-budget discrete benchmark ---
    gp = load(R + 'symmetric_gp.json'); dps = load(R + 'symmetric_dps.json')
    iri = load(R + 'main_results_iter.json')
    ax = axes[0, 0]
    if gp and dps and iri:
        tags = sea_order(gp.keys()); seas = sorted(set('_'.join(t.split('_')[:2]) for t in tags),
                                                   key=lambda s: (int(s.split('_')[0][2]), int(s.split('_')[1][:-1])))
        def perasea(d, f):
            return [np.mean([f(d[t]) for t in tags if '_'.join(t.split('_')[:2]) == s]) for s in seas]
        relax = perasea(iri, lambda r: r['deployable']['nrmse'][0])
        dpsv = perasea(dps, lambda r: r['dps']['nrmse'][0])
        gpv = perasea(gp, lambda r: r['gp_win3000']['nrmse'][0])
        ora = perasea(gp, lambda r: r['oracle'])
        x = np.arange(len(seas)); w = 0.2
        ax.bar(x - 1.5*w, relax, w, label='RELAX (classical, discrete)', color='tab:red')
        ax.bar(x - 0.5*w, dpsv, w, label='DPS (learned, m≤200)', color='tab:green')
        ax.bar(x + 0.5*w, gpv, w, label='GP (equal budget)', color='tab:gray')
        ax.bar(x + 1.5*w, ora, w, label='oracle', color='k', alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(seas, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('field nRMSE'); ax.set_title('A. Equal-budget — DISCRETE benchmark\n(classical RELAX wins; learning over-smooths)')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.25)

    # --- B. story flip: discrete vs continuous ---
    hos = load(R + 'dps_on_hos.json')
    ax = axes[0, 1]
    if dps and iri and hos:
        # discrete benchmark overall
        d_relax = np.mean([iri[t]['deployable']['nrmse'][0] for t in iri])
        d_dps = np.mean([dps[t]['dps']['nrmse'][0] for t in dps])
        # continuous HOS overall
        c_relax = np.mean([hos[t]['relax_mean'] for t in hos]); c_relax_s = np.mean([hos[t]['relax_std'] for t in hos])
        c_dps = np.mean([hos[t]['dps_mean'] for t in hos]); c_dps_s = np.mean([hos[t]['dps_std'] for t in hos])
        x = np.array([0, 1]); w = 0.35
        ax.bar(x - w/2, [d_relax, c_relax], w, yerr=[0, c_relax_s], label='RELAX (discrete-atom)', color='tab:red', capsize=4)
        ax.bar(x + w/2, [d_dps, c_dps], w, yerr=[0, c_dps_s], label='DPS (learned continuous)', color='tab:green', capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(['DISCRETE benchmark\n(15–30 components)', 'CONTINUOUS HOS\n(realistic spectrum)'])
        ax.set_ylabel('field nRMSE')
        ax.set_title('B. The flip: realism reverses the winner\nRELAX wins on discrete; DPS (learning) wins on realistic continuous')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25)
        for xi, (a, b) in zip(x, [(d_relax, d_dps), (c_relax, c_dps)]):
            win = 'RELAX' if a < b else 'DPS'
            ax.annotate(f'{win} wins', (xi, max(a, b) + 0.01), ha='center', fontsize=9, fontweight='bold')

    # --- C. scarcity crossover ---
    sc = load(R + 'scarcity_sweep.json')
    ax = axes[1, 0]
    if sc:
        caps = [50, 100, 150, 200, 400, 700, 1500, 3000]
        dv = [np.mean([sc[t]['dps'][str(m)] for t in sc if str(m) in sc[t]['dps']]) for m in [50,100,150,200,400]]
        rv = [np.mean([sc[t]['relax'][str(m)] for t in sc if str(m) in sc[t]['relax']]) for m in caps]
        ax.axvspan(30, 200, color='tab:green', alpha=0.07); ax.axvspan(200, 3500, color='tab:gray', alpha=0.07)
        ax.plot([50,100,150,200,400], dv, 'o-', color='tab:green', label='DPS (learned prior)')
        ax.plot(caps, rv, 's-', color='tab:red', label='RELAX (off-grid)')
        ax.axvline(200, ls='--', c='0.5'); ax.set_xscale('log')
        ax.set_xlabel('observations per window'); ax.set_ylabel('field nRMSE')
        ax.set_title('C. Observation-scarcity crossover\n(learned prior = graceful floor when data scarce)')
        ax.legend(fontsize=9); ax.grid(alpha=0.25, which='both')

    # --- D. free-aim active sensing budget curve ---
    bs = load(R + 'beam_steering.json')
    ax = axes[1, 1]
    if bs:
        bud = bs[list(bs.keys())[0]]['budgets']
        for mth, col, lab in [('rand', 'tab:blue', 'random uniform aim'),
                              ('learned', 'tab:green', 'learned policy'),
                              ('greedy', 'tab:red', 'greedy D-optimal')]:
            cur = np.mean([bs[t][mth] for t in bs], axis=0)
            ax.plot(bud, cur, 'o-', color=col, label=lab)
        ax.set_xlabel('measurement budget (free aim)'); ax.set_ylabel('field nRMSE')
        ax.set_title('D. Free-aim active sensing\n(random uniform ≥ active selection — no sensing lever)')
        ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle('WaveSplat — data comparisons (classical vs learning across regimes)', fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(R + 'fig_comparisons.png', dpi=140, bbox_inches='tight')
    print('saved ' + R + 'fig_comparisons.png')


if __name__ == '__main__':
    main()
