"""GT vs deployed(analytic) fields on GOOD/TYPICAL real windows — the counterpart to
real_collapse_repair_fields.png, to show what the 27 normal windows actually look like.
Hand arm only (that is what the deployed system runs on normal windows), protocol
verbatim from run_wp_real.py. Run from wavesplat/ package root."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

# best / good / median-ish windows by hand nowcast nRMSE (from hand_amp.json)
WINDOWS = [('AA15a', 5, 'best window'), ('AA14', 2, 'good window'),
           ('YS17', 3, 'median-level window')]
INK, SEC, SURF = '#0b0b0b', '#52514e', '#fcfcfb'


def main():
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6))
    fig.patch.set_facecolor(SURF)
    for ax in axes.ravel():
        ax.set_facecolor(SURF)

    for col, (rec, win, tag) in enumerate(WINDOWS):
        sc = cg.scene_real(rec, win=win)
        npx = sc['npx']; nl = len(sc['leads'])
        s0 = 0.05 * sc['Hs']
        fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'], sigma0=s0,
                                kp_hint=sc['kp'], seed=sc['seed'])
        maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
        gt0 = sc['maps_gt'][0]; pr0 = maps[0]
        m = np.isfinite(gt0) & np.isfinite(pr0)
        sg = float(np.std(sc['maps_gt'][np.isfinite(sc['maps_gt'])]))
        nr = float(np.sqrt(np.mean((pr0[m] - gt0[m]) ** 2)) / sg)
        cc = float(np.corrcoef(pr0[m], gt0[m])[0, 1])
        vm = 2.5 * sg
        ext = [sc['gxq'].min(), sc['gxq'].max(), sc['gyq'].min(), sc['gyq'].max()]
        for row, (f, lab) in enumerate([(gt0, 'stereo GT'), (pr0, 'deployed (analytic)')]):
            ax = axes[row, col]
            ax.imshow(np.where(np.isfinite(f), f, np.nan), origin='lower', extent=ext,
                      cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='equal')
            ax.tick_params(labelsize=7, colors=SEC)
            if row == 0:
                ax.set_title(f'{rec} w{win} — {tag}\nnRMSE {nr:.2f}   corr {cc:.2f}',
                             fontsize=10.5, color=INK)
            else:
                ax.set_xlabel('x [m]', fontsize=8, color=SEC)
        axes[0, col].set_xticklabels([])
        print(f'{rec}_w{win}: nRMSE {nr:.3f} corr {cc:.3f}', flush=True)
    axes[0, 0].set_ylabel('stereo GT', fontsize=10, color=INK)
    axes[1, 0].set_ylabel('deployed (analytic)', fontsize=10, color=INK)
    fig.suptitle('The 27 NORMAL real windows are not the collapse GIF — '
                 'typical deployed fields, nowcast (frozen protocol, zero-shot)',
                 fontsize=12.5, color=INK)
    fig.text(0.5, 0.925,
             "best-ever real field nRMSE (07-01 pilot): 0.47 — today's best window "
             'matches it (0.46). The collapse GIF = the WORST window (2.93), '
             'chosen to demonstrate the failure detector.',
             ha='center', fontsize=9, color=SEC)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = 'results/gifs/real_normalwin_fields.png'
    fig.savefig(out, dpi=150, facecolor=SURF)
    print('saved', out)


if __name__ == '__main__':
    main()
