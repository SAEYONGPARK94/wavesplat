"""Animated version of fig1 on LJ18g_w3: stereo GT | analytic | learned over the
21 forecast leads (0..20 s, 1 s apart). Fields come from collapse_fields.npz
(re-run of the frozen protocol, bit-identical to wp_real.json -- see the run log).

Output: results/gifs/real_collapse_repair.gif  (<= 9 MB, ~900 px wide, 128 colours)
Run from wavesplat/ package root.
"""
import json, os, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = 'LJ18g_w3'
OUTGIF = 'results/gifs/real_collapse_repair.gif'
W_PX = 900

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, SEC, MUT, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.edgecolor': MUT, 'font.family': 'sans-serif',
})


def main():
    d = np.load(os.path.join(HERE, 'collapse_fields.npz'))
    meta = json.load(open(os.path.join(HERE, 'collapse_fields.json')))[KEY]
    gt = d[KEY + '_gt']; hand = d[KEY + '_hand']; learn = d[KEY + '_learned']
    leads = d[KEY + '_leads']; gx = d[KEY + '_gxq']; gy = d[KEY + '_gyq']
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]
    vmax = float(np.nanpercentile(np.abs(gt), 99))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#eeece6')

    dpi = 100
    figw = W_PX / dpi
    frames = []
    for li in range(len(leads)):
        fig = plt.figure(figsize=(figw, figw * 0.62), dpi=dpi)
        gs = fig.add_gridspec(1, 3, left=0.015, right=0.985, top=0.735, bottom=0.10,
                              wspace=0.06)
        m = np.isfinite(gt[li])
        panels = [('Stereo ground truth', gt[li], INK, None),
                  ('Analytic estimator', hand[li], BLUE,
                   meta['hand_per_lead'][li]),
                  ('Learned estimator', learn[li], ORANGE,
                   meta['learned_per_lead'][li])]
        for ci, (lab, fld, col, err) in enumerate(panels):
            ax = fig.add_subplot(gs[0, ci])
            ax.imshow(np.ma.masked_invalid(np.where(m, fld, np.nan)), origin='lower',
                      extent=ext, cmap=cmap, norm=norm, aspect='equal',
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(MUT); s.set_linewidth(0.6)
            ax.set_title(lab, fontsize=11, color=col, pad=5)
            if err is not None:
                ax.text(0.04, 0.96, f'nRMSE {err:.2f}', transform=ax.transAxes,
                        fontsize=10, color=INK, va='top', ha='left',
                        bbox=dict(fc=SURF, ec='none', alpha=0.85, pad=2))
            sg = float(np.std(gt[li][m])); sf = float(np.std(fld[m]))
            ax.text(0.96, 0.03, f'amp {sf/sg:.1f}x GT', transform=ax.transAxes,
                    fontsize=9, color=SEC, va='bottom', ha='right',
                    bbox=dict(fc=SURF, ec='none', alpha=0.85, pad=2))
        fig.suptitle('Real ocean data, LJ18g window 3 - the worst analytic collapse: '
                     'learning as insurance', fontsize=12.5, x=0.015, ha='left', y=0.975)
        fig.text(0.015, 0.925,
                 '30 real stereo windows (Bergamasco / Guimaraes 2020 IFREMER); frozen '
                 'pre-registered GT-free amplitude guard, tau = 1.5 - it fires here.\n'
                 'Nowcast: analytic 2.93  ->  learned 0.96. On the other 27 normal windows '
                 'the analytic arm is used and is the more accurate one.',
                 fontsize=8.8, color=SEC, ha='left', va='top', linespacing=1.55)
        fig.text(0.985, 0.055, f't = +{int(leads[li]):d} s'
                 + ('   (nowcast)' if li == 0 else '   (forecast lead)'),
                 fontsize=11, color=INK, ha='right', va='bottom')
        # progress bar
        axp = fig.add_axes([0.015, 0.045, 0.45, 0.012])
        axp.set_xlim(0, leads[-1]); axp.set_ylim(0, 1)
        axp.barh(0.5, leads[li], height=1.0, color=AQUA)
        axp.set_xticks([]); axp.set_yticks([])
        for s in axp.spines.values():
            s.set_color(GRID)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))

    pal = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE)
           for f in frames]
    pal[0].save(OUTGIF, save_all=True, append_images=pal[1:], loop=0,
                duration=420, optimize=True, disposal=2)
    sz = os.path.getsize(OUTGIF)
    print(f'wrote {OUTGIF}: {len(pal)} frames, {frames[0].size[0]}x{frames[0].size[1]} px, '
          f'{sz/1e6:.2f} MB')
    assert sz < 9e6, 'GIF too large'


if __name__ == '__main__':
    main()
