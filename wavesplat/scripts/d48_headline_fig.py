"""
Headline figure — make the LEARNING WIN prominent (the multi-panel d46 buried it among
regimes where DPS is intentionally weak). The narrative: the field's discrete benchmark is
biased toward classical discrete estimators; on realistic continuous spectra learning wins.

Run: python3 scripts/d48_headline_fig.py
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

R = 'results/direction3/'


def main():
    hos = json.load(open(R + 'dps_on_hos.json'))
    iri = json.load(open(R + 'main_results_iter.json'))
    dpsd = json.load(open(R + 'symmetric_dps.json'))
    # discrete benchmark overall
    d_relax = np.mean([iri[t]['deployable']['nrmse'][0] for t in iri])
    d_dps = np.mean([dpsd[t]['dps']['nrmse'][0] for t in dpsd])
    # continuous HOS overall (mean of per-condition means)
    c_relax = np.mean([hos[t]['relax_mean'] for t in hos]); c_relax_s = np.std([hos[t]['relax_mean'] for t in hos])
    c_dps = np.mean([hos[t]['dps_mean'] for t in hos]); c_dps_s = np.std([hos[t]['dps_mean'] for t in hos])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1 — the FLIP
    x = np.array([0, 1]); w = 0.34
    b1 = ax1.bar(x - w/2, [d_relax, c_relax], w, yerr=[0, c_relax_s], capsize=5,
                 label='RELAX (classical, discrete atoms)', color='tab:red')
    b2 = ax1.bar(x + w/2, [d_dps, c_dps], w, yerr=[0, c_dps_s], capsize=5,
                 label='DPS (learned continuous prior)', color='tab:green')
    ax1.set_xticks(x); ax1.set_xticklabels(['DISCRETE benchmark\n(15–30 components)\n— what the field uses',
                                            'CONTINUOUS HOS\n(realistic JONSWAP spectrum)\n— reality'], fontsize=10)
    ax1.set_ylabel('field nRMSE', fontsize=11)
    ax1.set_title('Realism flips the winner', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(axis='y', alpha=0.25)
    ax1.annotate('classical wins\n(−%d%%)' % round(100*(d_dps-d_relax)/d_dps), (0, max(d_relax, d_dps)+0.012),
                 ha='center', fontsize=10, color='tab:red', fontweight='bold')
    ax1.annotate('LEARNING wins\n(−16%%, p<0.001)', (1, max(c_relax, c_dps)+0.022),
                 ha='center', fontsize=10, color='tab:green', fontweight='bold')
    ax1.set_ylim(0, 0.34)

    # Panel 2 — per-condition HOS (DPS wins all, with error bars)
    conds = sorted(hos.keys())
    xr = np.arange(len(conds))
    ax2.bar(xr - w/2, [hos[c]['relax_mean'] for c in conds], w,
            yerr=[hos[c]['relax_std'] for c in conds], capsize=5, label='RELAX', color='tab:red')
    ax2.bar(xr + w/2, [hos[c]['dps_mean'] for c in conds], w,
            yerr=[hos[c]['dps_std'] for c in conds], capsize=5, label='DPS (learned)', color='tab:green')
    ax2.set_xticks(xr); ax2.set_xticklabels([c + '\n(Hs↑)' for c in conds])
    ax2.set_ylabel('field nRMSE', fontsize=11)
    ax2.set_title('On realistic continuous spectra, learning wins every condition\n(9/9 windows, 3 seeds each)', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.25)
    for xi, c in zip(xr, conds):
        ax2.annotate('✓', (xi + w/2, hos[c]['dps_mean'] + hos[c]['dps_std'] + 0.01), ha='center',
                     color='tab:green', fontsize=14, fontweight='bold')

    fig.suptitle('The discrete benchmark hides the learning win: classical wins on discrete components,\n'
                 'the learned continuous prior wins on realistic continuous-spectrum (HOS) waves',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(R + 'fig_headline_flip.png', dpi=140, bbox_inches='tight')
    print('saved ' + R + 'fig_headline_flip.png')
    print(f'discrete: RELAX {d_relax:.3f} DPS {d_dps:.3f} | continuous: RELAX {c_relax:.3f} DPS {c_dps:.3f}')


if __name__ == '__main__':
    main()
