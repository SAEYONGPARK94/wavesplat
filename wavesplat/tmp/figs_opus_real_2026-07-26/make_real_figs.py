"""REAL-OCEAN-DATA summary figures (Bergamasco stereo, 30 windows).

Framing is bound by docs/HANDOFF_OPUS_2026-07-26.md §5: on real data the learned arm's
role is COLLAPSE INSURANCE, not accuracy. Nothing here claims a real-data accuracy win.

  fig1 real_collapse_repair_fields.png : GT | analytic | learned fields, the 2 severe
                                         analytic collapse windows, 3 time instants
  fig2 real_perwindow_paired.png       : all 30 windows, analytic vs deployed + amp signal
  fig3 real_amp_signal.png             : GT-free detector honestly (incl. the miss)
  fig4 real_normal27_loss.png          : the honest negative — learned LOSES the typical 27

Sources (verified, no re-derivation of any published number):
  tmp/wp_real_2026-07-25/wp_real.json          per-window hand/learned (frozen prereg)
  tmp/unified_2026-07-26/hand_amp.json         per-window GT-free amp_ratio_grid
  tmp/unified_2026-07-26/fallback_results.json deployed fallback aggregate
  tmp/figs_opus_real_2026-07-26/collapse_fields.npz  re-run fields (bit-identical, see .json)

Run from wavesplat/ package root:  python3 tmp/figs_opus_real_2026-07-26/make_real_figs.py
"""
import json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wilcoxon, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = 'results/gifs'
WP_REAL = 'tmp/wp_real_2026-07-25/wp_real.json'
HAND_AMP = 'tmp/unified_2026-07-26/hand_amp.json'
FALLBACK = 'tmp/unified_2026-07-26/fallback_results.json'
NPZ = os.path.join(HERE, 'collapse_fields.npz')
META = os.path.join(HERE, 'collapse_fields.json')
TAU = 1.5

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, SEC, MUT, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
PROTO = ('30 real stereo windows (Bergamasco / Guimaraes 2020 IFREMER; 5 records) - '
         'frozen pre-registered GT-free amplitude guard, tau = 1.5')

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.edgecolor': MUT, 'axes.labelcolor': SEC,
    'xtick.color': SEC, 'ytick.color': SEC, 'axes.grid': False,
    'font.family': 'sans-serif', 'font.size': 11,
})


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def wrap(s, n=150):
    return textwrap.fill(' '.join(s.split()), n)


def load_rows():
    """Merge wp_real (hand/learned) with hand_amp (GT-free amp signal). Cross-checked."""
    wp = {(r['record'], r['win']): r for r in json.load(open(WP_REAL))['results']}
    amp = json.load(open(HAND_AMP))['rows']
    assert abs(json.load(open(HAND_AMP))['hand_now_maxdiff']) < 1e-9
    rows = []
    for a in amp:
        k = (a['record'], a['win'])
        w = wp[k]
        assert abs(w['hand_now'] - a['hand_now']) < 1e-9
        assert abs(w['learned_now'] - a['learned_now']) < 1e-12
        fires = a['amp_ratio_grid'] > TAU
        rows.append(dict(name=f"{a['record']}_w{a['win']}", record=a['record'],
                         win=a['win'], collapse=a['collapse'],
                         hand=a['hand_now'], learned=a['learned_now'],
                         amp=a['amp_ratio_grid'], fires=fires,
                         deployed=a['learned_now'] if fires else a['hand_now']))
    assert len(rows) == 30
    return rows


# =============================================================== fig 1 : the fields
def fig_fields():
    d = np.load(NPZ)
    meta = json.load(open(META))
    wins = [('LJ18g_w3', [0, 6, 12]), ('AA14_w0', [0, 6, 12])]
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#eeece6')

    fig = plt.figure(figsize=(13.0, 9.0))
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 0.35, 1, 1, 1],
                          left=0.075, right=0.985, top=0.785, bottom=0.072,
                          wspace=0.10, hspace=0.13)
    rows = [('Stereo ground truth', 'gt'), ('Analytic estimator', 'hand'),
            ('Learned estimator', 'learned')]
    rowcol = {'gt': INK, 'hand': BLUE, 'learned': ORANGE}

    for wi, (key, leads) in enumerate(wins):
        gt = d[key + '_gt']; gx = d[key + '_gxq']; gy = d[key + '_gyq']
        ext = [gx.min(), gx.max(), gy.min(), gy.max()]
        vmax = float(np.nanpercentile(np.abs(gt), 99))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        c0 = 0 if wi == 0 else 4
        for ri, (rlab, akey) in enumerate(rows):
            for ci, li in enumerate(leads):
                ax = fig.add_subplot(gs[ri, c0 + ci])
                m = np.isfinite(gt[li])          # exactly the scored footprint
                fld = np.where(m, d[key + '_' + akey][li], np.nan)
                ax.imshow(np.ma.masked_invalid(fld), origin='lower', extent=ext,
                          cmap=cmap, norm=norm, aspect='equal',
                          interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(MUT); s.set_linewidth(0.6)
                # amplitude ratio vs GT on the same scored pixels
                sg = float(np.std(gt[li][m]))
                sf = float(np.std(fld[m]))
                if ri == 0:
                    ax.set_title(f"t = {int(d[key+'_leads'][li])} s"
                                 + ('  (nowcast)' if li == 0 else ''),
                                 fontsize=9.5, color=SEC, pad=4)
                else:
                    e = meta[key]['hand_per_lead' if akey == 'hand'
                                  else 'learned_per_lead'][li]
                    ax.text(0.035, 0.965, f'nRMSE {e:.2f}', transform=ax.transAxes,
                            fontsize=9, color=INK, va='top', ha='left',
                            bbox=dict(fc=SURF, ec='none', alpha=0.82, pad=1.6))
                ax.text(0.965, 0.028, f'amp {sf/sg:.1f}x GT', transform=ax.transAxes,
                        fontsize=8, color=SEC, va='bottom', ha='right',
                        bbox=dict(fc=SURF, ec='none', alpha=0.82, pad=1.4))
                if ci == 0 and wi == 0:
                    ax.set_ylabel(rlab, fontsize=11, color=rowcol[akey], labelpad=6)

        gtxt = ("LJ18g w3  -  storm swell, worst analytic collapse\n"
                "analytic 2.93  ->  learned 0.96 nRMSE (nowcast, below the collapse line)"
                if wi == 0 else
                "AA14 w0  -  second severe collapse\n"
                "analytic 2.31  ->  learned 1.07 nRMSE (nowcast, halved but still >= 1)")
        fig.text(0.285 if wi == 0 else 0.775, 0.859, gtxt, fontsize=10.5,
                 color=INK, ha='center', va='top', linespacing=1.45)

    cax = fig.add_axes([0.435, 0.020, 0.13, 0.013])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(['trough', '0', 'crest'])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5)
    cax.text(0.5, 2.6, 'surface elevation (scale shared within a window)',
             transform=cax.transAxes, fontsize=8.5, color=SEC, ha='center', va='bottom')

    fig.suptitle('Learning as collapse insurance on real data: the analytic estimator '
                 'blows up, the learned prior does not', fontsize=13.5, x=0.075,
                 ha='left', y=0.984)
    fig.text(0.075, 0.966, wrap(
        PROTO + '. Both arms zero-shot: same observations, same solver back-end, no '
        'training on real data. Every panel is masked to the stereo-valid pixels the '
        'metric actually scores, and the colour scale is shared within a window - so the '
        'analytic panels clip, and that clipping IS the failure. On the other 27 normal '
        'windows the analytic arm is the one used, and the more accurate one. The learned arm reduces the error on both windows and never itself collapses, but only LJ18g w3 ends below nRMSE = 1.', 152),
        fontsize=9.5, color=SEC, ha='left', va='top', linespacing=1.55)
    fig.savefig(f'{OUT}/real_collapse_repair_fields.png', dpi=155)
    plt.close(fig)
    print('fig1 saved')


# ====================================================== fig 2 : per-window paired
def fig_perwindow():
    rows = sorted(load_rows(), key=lambda r: r['hand'])
    fb = json.load(open(FALLBACK))
    n = len(rows); xs = np.arange(n)
    hand = np.array([r['hand'] for r in rows])
    dep = np.array([r['deployed'] for r in rows])
    amp = np.array([r['amp'] for r in rows])
    coll = np.array([r['collapse'] for r in rows])
    fires = np.array([r['fires'] for r in rows])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.6, 8.5), sharex=True,
                                  gridspec_kw={'height_ratios': [1.75, 1],
                                               'hspace': 0.10})
    # ---- top: paired outcome
    for i in np.flatnonzero(fires):
        ax.annotate('', xy=(i, dep[i]), xytext=(i, hand[i]),
                    arrowprops=dict(arrowstyle='-|>', color=AQUA, lw=2.0,
                                    shrinkA=2, shrinkB=0))
    ax.scatter(xs, hand, s=46, color=BLUE, zorder=4, label='analytic estimator (hand)')
    ax.scatter(xs, dep, s=104, facecolors='none', edgecolors=AQUA, lw=1.9, zorder=5,
               label='DEPLOYED: analytic + GT-free collapse guard')
    ax.axhline(1.0, color=MUT, ls=(0, (4, 3)), lw=1.1, zorder=1)
    ax.text(-0.4, 1.03, 'nRMSE = 1  (no better than predicting a flat sea)',
            fontsize=9, color=SEC, va='bottom')
    for i in np.flatnonzero(coll):
        ax.axvspan(i - 0.45, i + 0.45, color='#f6ece6', zorder=0)
        lab = rows[i]['name']
        if fires[i]:
            ax.text(i, dep[i] - 0.10, f'{dep[i]:.2f}', ha='center', va='top',
                    fontsize=9.5, color=AQUA)
            ax.text(i, hand[i] + 0.09, f'{hand[i]:.2f}', ha='center', va='bottom',
                    fontsize=9.5, color=BLUE)
        else:
            ax.annotate(f'{lab}: guard did NOT fire\n(mild collapse, {hand[i]:.2f}) - the '
                        'learned arm\nwould have given 0.79, but is not called',
                        xy=(i, hand[i]), xytext=(i - 17.0, 2.55), fontsize=9.5,
                        color=ORANGE, ha='left', va='top', linespacing=1.4,
                        arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.3,
                                        connectionstyle='arc3,rad=-0.16'))
    ax.text(n - 2.7, 1.72, 'guard fires ->\ndeployed switches\nto the learned arm',
            fontsize=9.5, color=AQUA, ha='right', va='center', linespacing=1.4)
    ax.set_ylabel('nowcast map nRMSE  (lower = better)')
    ax.set_ylim(0, 3.30)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    style(ax)
    agg = fb['aggregate']
    ax.legend(frameon=False, fontsize=10, loc='center left',
              bbox_to_anchor=(0.015, 0.63))
    ax.text(0.115, 0.055,
            f"30-window MEAN {agg['hand_mean']:.3f} -> {agg['fallback_mean']:.3f}   |   "
            f"MEDIAN unchanged: {agg['hand_median']:.3f} -> {agg['fallback_median']:.3f}",
            transform=ax.transAxes, fontsize=10.5, color=INK, ha='left', va='bottom')

    # ---- bottom: GT-free signal
    cols = [ORANGE if c else BLUE for c in coll]
    b = ax2.bar(xs, amp, 0.66, color=cols, zorder=3)
    for i in np.flatnonzero(fires):
        b[i].set_edgecolor(AQUA); b[i].set_linewidth(1.8)
    ax2.axhline(TAU, color=INK, ls=(0, (5, 3)), lw=1.3, zorder=4)
    ax2.text(n - 3.6, TAU + 0.07, f'frozen threshold  tau = {TAU}', fontsize=9.5,
             color=INK, ha='right', va='bottom')
    for i in np.flatnonzero(coll):
        ax2.text(i, amp[i] + 0.06, f'{amp[i]:.2f}', ha='center', va='bottom',
                 fontsize=9, color=INK)
    ax2.set_ylabel('amp_ratio_grid\n(GT-free)', fontsize=10.5)
    ax2.set_ylim(0, 3.0)
    ax2.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    style(ax2)
    ax2.set_xticks(xs, [r['name'] for r in rows], rotation=90, fontsize=8)
    for t, c in zip(ax2.get_xticklabels(), coll):
        t.set_color(ORANGE if c else SEC)
    ax2.set_xlim(-0.8, n - 0.2)
    ax2.text(0.005, 0.93, 'orange = the 3 windows where the analytic path collapses '
             '(nRMSE >= 1); blue = 27 normal windows',
             transform=ax2.transAxes, fontsize=9.5, color=SEC, va='top')

    fig.suptitle('Real ocean data: the deployed estimator changes 2 windows out of 30 - '
                 'and only where the analytic path had collapsed',
                 fontsize=13.5, x=0.02, ha='left', y=0.985)
    fig.text(0.02, 0.958, wrap(
        PROTO + '. Windows sorted by analytic nRMSE. Deployed = the analytic estimator '
        'everywhere, learned only when the guard fires. Learning here buys robustness, '
        'not typical-window accuracy. CAVEATS, all measured: the 30-window mean improves '
        'but the MEDIAN is unchanged (0.809 -> 0.809), so the mean shift is two windows '
        'and not a general gain; false alarms are 0/27 but tau was set on these same 30 '
        'windows (in-sample); and end-to-end only 1 of the 3 collapse windows ends below '
        'nRMSE = 1 - reducing the error is not the same as fixing the window.', 150),
        fontsize=9.5, color=SEC, ha='left', va='top', linespacing=1.55)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.795, bottom=0.135)
    fig.savefig(f'{OUT}/real_perwindow_paired.png', dpi=155)
    plt.close(fig)
    print('fig2 saved')


# ================================================== fig 3 : the detector, honestly
def fig_amp_signal():
    rows = load_rows()
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    norm = [r for r in rows if not r['collapse']]
    coll = [r for r in rows if r['collapse']]

    ax.axvspan(TAU, 3.0, color='#eaf6f1', zorder=0)
    ax.axvline(TAU, color=INK, ls=(0, (5, 3)), lw=1.4, zorder=2)
    ax.axhline(1.0, color=MUT, ls=(0, (4, 3)), lw=1.1, zorder=2)
    ax.text(2.98, 0.94, 'collapse threshold: nRMSE = 1 (worse than a flat sea)',
            fontsize=9.5, color=SEC, va='top', ha='right')
    ax.text(2.98, 3.32, f'guard fires  (amp > tau = {TAU})', fontsize=10,
            color='#0f8f63', ha='right', va='top')

    ax.scatter([r['amp'] for r in norm], [r['hand'] for r in norm], s=54,
               color=BLUE, alpha=0.9, zorder=4,
               label='27 normal windows (analytic works)')
    for r in coll:
        c = AQUA if r['fires'] else ORANGE
        ax.scatter([r['amp']], [r['hand']], s=190, marker='D', color=c, zorder=6,
                   edgecolor=INK, lw=0.8)
        ax.annotate('', xy=(r['amp'], r['learned']), xytext=(r['amp'], r['hand']),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=2.0,
                                    ls='-' if r['fires'] else (0, (3, 2)),
                                    shrinkA=5, shrinkB=0), zorder=5)
        ax.scatter([r['amp']], [r['learned']], s=64, marker='o', facecolors=SURF,
                   edgecolors=c, lw=1.9, zorder=6)

    ax.text(2.45, 2.93, 'LJ18g_w3   analytic 2.93\nguard FIRES -> deployed 0.96',
            fontsize=10, color='#0f8f63', ha='right', va='center', linespacing=1.4)
    ax.text(2.32, 2.31, 'AA14_w0   analytic 2.31\nguard FIRES -> deployed 1.07',
            fontsize=10, color='#0f8f63', ha='right', va='center', linespacing=1.4)
    ax.annotate('THE MISS - AA15a_w3\namp 0.89 < tau, so the guard stays silent.\n'
                'The mild collapse (1.12) is left uncorrected;\n'
                'the learned arm would have given 0.79.',
                xy=(0.89, 1.12), xytext=(0.33, 2.10), fontsize=10, color=ORANGE,
                ha='left', va='top', linespacing=1.45,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.5,
                                connectionstyle='arc3,rad=0.20'))
    ax.text(1.03, 0.325, 'no false alarm fires here:\n0 / 27 normal windows exceed tau',
            fontsize=10, color=SEC, ha='center', va='center', linespacing=1.4)

    ax.set_xlabel('amp_ratio_grid  =  std(analytic nowcast field) / rms(observed elevation)'
                  '   -   computed WITHOUT ground truth')
    ax.set_ylabel('nowcast map nRMSE of the analytic estimator')
    ax.set_xlim(0.3, 3.0); ax.set_ylim(0.18, 3.4)
    ax.grid(True, color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    style(ax)
    ax.set_title('The GT-free collapse detector, shown together with its miss\n'
                 + wrap(PROTO + '. Filled diamond = analytic outcome; arrow to hollow '
                        'circle = what the learned arm delivers. Solid arrow = guard '
                        'fired (this is the deployed value); dashed = repair was '
                        'available but not taken. Error is reduced on all three, but '
                        'end-to-end only LJ18g_w3 ends below nRMSE = 1, and tau was set '
                        'on these same 30 windows (in-sample).', 108),
                 fontsize=11.5, loc='left', pad=12, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(f'{OUT}/real_amp_signal.png', dpi=155)
    plt.close(fig)
    print('fig3 saved')


# ============================================ fig 4 : the honest negative (why a guard)
def fig_normal27():
    rows = [r for r in load_rows() if not r['collapse']]
    delta = np.array([r['hand'] - r['learned'] for r in rows])   # >0 = learned better
    order = np.argsort(delta)
    delta = delta[order]; names = [rows[i]['name'] for i in order]
    res = wilcoxon(delta)
    ar = rankdata(np.abs(delta))
    rb = (ar[delta > 0].sum() - ar[delta < 0].sum()) / ar.sum()
    hm = np.mean([r['hand'] for r in rows]); lm = np.mean([r['learned'] for r in rows])

    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    xs = np.arange(len(delta))
    cols = [ORANGE if v > 0 else BLUE for v in delta]
    bars = ax.bar(xs, delta, 0.68, color=cols, zorder=3)
    ax.axhline(0, color=INK, lw=1.0, zorder=4)
    ax.axhline(np.median(delta), color=SEC, ls=(0, (4, 3)), lw=1.2, zorder=4)
    ax.text(len(delta) - 0.5, np.median(delta) - 0.012,
            f'median  {np.median(delta):+.3f}', fontsize=9.5, color=SEC,
            ha='right', va='top')
    ax.set_xticks(xs, names, rotation=90, fontsize=8)
    ax.set_xlim(-0.8, len(delta) - 0.2)
    ax.set_ylabel('nRMSE(analytic) - nRMSE(learned)\n<- analytic better    learned better ->',
                  fontsize=10.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    style(ax)
    ax.text(0.015, 0.055,
            f'learned better in only {int((delta>0).sum())} / {len(delta)} normal windows\n'
            f'mean delta {delta.mean():+.4f}   median {np.median(delta):+.4f}\n'
            f'two-sided Wilcoxon W={res.statistic:.0f}, p={res.pvalue:.4f}   '
            f'rank-biserial {rb:+.3f}\n'
            f'means: analytic {hm:.3f}  vs  learned {lm:.3f}',
            transform=ax.transAxes, fontsize=10, color=INK, va='bottom', ha='left',
            linespacing=1.6, bbox=dict(fc=SURF, ec=GRID, pad=6))
    ax.set_title('The honest negative that forces the guard design: on typical real windows '
                 'the learned prior is significantly WORSE\n'
                 + wrap('27 of the 30 real stereo windows (Bergamasco / Guimaraes 2020 '
                        'IFREMER, 5 records); the 3 analytic collapse windows are '
                        'pre-excluded by the frozen pre-registration. Pre-registered '
                        'primary endpoint, decision = LOSS-significant. This is why the '
                        'deployed system runs the analytic estimator by default and calls '
                        'the learned arm only as collapse insurance.', 124),
                 fontsize=11.5, loc='left', pad=12, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(f'{OUT}/real_normal27_loss.png', dpi=155)
    plt.close(fig)
    print('fig4 saved', delta.mean(), np.median(delta), (delta > 0).sum(),
          res.statistic, res.pvalue, rb)


if __name__ == '__main__':
    fig_fields()
    fig_perwindow()
    fig_amp_signal()
    fig_normal27()
