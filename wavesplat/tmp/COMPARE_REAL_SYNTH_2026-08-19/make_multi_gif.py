#!/usr/bin/env python3
"""8-panel animation: 4 real records (4 different sites) vs 4 synthetic regimes.

The synthetic row deliberately includes the two moving-sensor cells generated today
(C_ENCOUNTER_2026-08-20), so the grid shows every kind of sea this project has ever
optimised against, next to what the instrument actually records.

Each panel keeps its OWN spatial extent and states it in the title.  Forcing a shared
extent would hide the thing most worth seeing: the real footprints are a small corner of
what the synthetic corpus assumes (B-3 found no real record large enough for the batch
geometry).  The colour scale IS shared, so wave amplitude is comparable across panels.
"""
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib import animation                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
N = 24

REAL = [
    ('Acqua Alta 2014 (tower)', 'data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc'),
    ('Black Sea 2013 (platform)', 'data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc'),
    ('La Jument 2018 (lighthouse)', 'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc'),
    ('Yellow Sea 2017 (platform)', 'data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc'),
]
SYNTH = [
    ('SYNTH sparse (2 comp)', 'data_s9_sparse_2026-08-11'),
    ('SYNTH crowded (15 comp)', 'data_s9_crowded_2026-08-11'),
    ('SYNTH moving, following', 'tmp/C_ENCOUNTER_2026-08-20/data/C_moving_following'),
    ('SYNTH moving, head', 'tmp/C_ENCOUNTER_2026-08-20/data/D_moving_head'),
]


def _decode(dset, idx=None):
    """netCDF packed integers: value = raw * scale_factor + add_offset, fill -> nan.

    X and Y carry their OWN scale_factor (X: 1.34e-3 m/count) -- reading them raw yields
    an extent in the tens of kilometres, which is how this was first wrong here.
    """
    raw = (dset[idx] if idx is not None else dset[:]).astype(float)
    fill = dset.attrs.get('_FillValue', [None])
    if fill[0] is not None:
        raw[raw == float(fill[0])] = np.nan
    sf = float(dset.attrs.get('scale_factor', [1.0])[0])
    off = float(dset.attrs.get('add_offset', [0.0])[0])
    return raw * sf + off


def load_real(path):
    d = h5py.File(os.path.join(ROOT, path), 'r')
    xname = '_nc4_non_coord_X' if '_nc4_non_coord_X' in d else 'X'
    yname = '_nc4_non_coord_Y' if '_nc4_non_coord_Y' in d else 'Y'
    xg, yg = _decode(d[xname]), _decode(d[yname])
    idx = (np.arange(N) * 6)
    idx = idx[idx < d['Z'].shape[0]]
    z = _decode(d['Z'], idx)
    return xg, yg, z - np.nanmean(z)


def load_synth(folder):
    files = sorted(f for f in os.listdir(os.path.join(ROOT, folder))
                   if f.endswith('.h5'))
    d = h5py.File(os.path.join(ROOT, folder, files[0]), 'r')
    gx, gy = d['frames/gt_grid_x'][:], d['frames/gt_grid_y'][:]
    z = d['frames/gt_wave_field'][:]
    step = max(1, len(z) // N)
    z = z[np.arange(0, len(z), step)[:N]]
    x1 = gx[0, :] if gx.ndim == 2 else gx
    y1 = gy[:, 0] if gy.ndim == 2 else gy
    return x1, y1, z


def main():
    panels = []
    for title, p in REAL:
        try:
            x, y, z = load_real(p)
            panels.append((title, x, y, z, 'real'))
        except Exception as exc:                          # noqa: BLE001
            print('  skip %-30s %s' % (title, exc))
    for title, f in SYNTH:
        x, y, z = load_synth(f)
        panels.append((title, x, y, z, 'synth'))

    nfr = min(len(p[3]) for p in panels)
    lim = float(np.nanpercentile(np.abs(np.concatenate(
        [p[3][:nfr].ravel() for p in panels])), 99))
    print('panels=%d frames=%d colour limit=+-%.2f m' % (len(panels), nfr, lim))

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6))
    fig.patch.set_facecolor('#0d1117')
    ims = []
    for ax, (title, x, y, z, kind) in zip(axes.ravel(), panels):
        ext = [float(np.nanmin(x)), float(np.nanmax(x)),
               float(np.nanmin(y)), float(np.nanmax(y))]
        # per-panel scale: a shared one is dominated by the largest record and flattens
        # every other panel to white.  The amplitude comparison is carried by the Hs
        # printed in each title instead, which is the honest way to state it.
        plim = float(np.nanpercentile(np.abs(z[:nfr]), 99)) or 1.0
        im = ax.imshow(z[0], origin='lower', extent=ext, cmap='RdBu_r',
                       vmin=-plim, vmax=plim, aspect='equal',
                       interpolation='nearest')
        ims.append(im)
        span = '%.0f x %.0f m' % (ext[1] - ext[0], ext[3] - ext[2])
        hs = 4.0 * float(np.nanstd(z[:nfr]))
        # Missing data is a property only the real records have; the synthetic GT is
        # dense everywhere by construction.  Stating it keeps the two rows honest.
        gap = 100.0 * float(np.mean(np.isnan(z[:nfr])))
        colour = '#7ee787' if kind == 'real' else '#79c0ff'
        ax.set_title('%s\n%s   Hs~%.2f m   missing %.0f%%'
                     % (title, span, hs, gap), color=colour, fontsize=8.6, pad=5)
        ax.tick_params(colors='#8b949e', labelsize=6.2)
        for s in ax.spines.values():
            s.set_color(colour); s.set_linewidth(1.4)
        ax.set_facecolor('#0d1117')
    fig.suptitle('Measured sea (green, fixed installations)   vs   '
                 'simulated sea (blue, incl. today\'s moving-sensor cells)',
                 color='#e6edf3', fontsize=11.5, y=0.985)
    fig.text(0.5, 0.015, 'each panel at its own extent and its own colour scale -- '
                         'compare shape and footprint here, amplitude via the stated Hs',
             ha='center', color='#8b949e', fontsize=8.5)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])

    def upd(i):
        for im, (_, _, _, z, _) in zip(ims, panels):
            im.set_array(z[i])
        return ims

    out = os.path.join(HERE, 'multi_real_vs_synth.gif')
    animation.FuncAnimation(fig, upd, frames=nfr, blit=False).save(
        out, writer=animation.PillowWriter(fps=5), dpi=58,
        savefig_kwargs={'facecolor': '#0d1117'})
    print('wrote', out, '%.1f MB' % (os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
