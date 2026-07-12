"""External-evidence figures: does the wave estimate actually match the sea?

Three publication-grade PNGs from the frozen 60-window real-data protocol
(gate_promotion_eval.json arms, windows rebuilt verbatim):

  fig_ext1  encounter-point time series, truth vs deployed learned arm,
            3 showcase windows from 3 sites (LJ18g w8 / YS17 w2 / AA15b w10).
  fig_ext2  the field-vs-track paradox window (AA14 w10): hand's field LOOKS
            better (pooled field corr) while its encounter track is wrong.
  fig_ext3  aggregate: 8-record fixed-aperture bars (gamma-net) +
            60-window hand-vs-learned scatter.

Every annotated number is recomputed here and cross-checked against the
stored JSONs (tolerance printed); fig_ext2 additionally verifies the
documented field-corr claim (hand +0.72 vs head +0.66, PAPER outline C.2b).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from d3_falcon_real_operator import RECORDS, bilinear
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_gate_promotion_eval import window_data
from d3_l1_s2head_train import SmultHead
from d3_gpu_mmse import mmse_gpu

# validated reference palette (dataviz skill, fixed slot order)
C_HAND = '#2a78d6'    # slot 1 blue  — physics-prior (hand) arm
C_NET = '#1baf7a'     # slot 2 aqua  — learned arm (net + s2-head)
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_MUT = '#898781'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'
C_BADBG = '#f5ede2'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb',
})

OUT = 'results/direction3'
DATE = '2026-07-12'
ROWS = json.load(open(f'{OUT}/gate_promotion_eval.json'))


def json_row(name, wi):
    return next(r for r in ROWS if r['record'] == name and r['wi'] == wi)


class Window:
    """Rebuild one frozen protocol window + both arms on arbitrary queries."""

    def __init__(self, name, wi):
        self.name, self.wi = name, wi
        cfg = RECORDS[name]
        self.cfg = cfg
        self.rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / self.rec.dt))
        n_lead = int(round(cfg['lead_max'] / self.rec.dt))
        i0s = np.linspace(nfw + 1, self.rec.T_all - n_lead - 2, 12).astype(int)
        self.i0 = int(i0s[wi])
        (self.P, self.z, self.Pq, self.gt_q, self.leads, self.Hs,
         self.kp, _) = window_data(self.rec, cfg, self.i0, nfw, n_lead, wi)
        self.Zfut = self.rec.window(self.i0, n_lead + 1)
        self.t0 = float(self.rec.t[self.i0])
        self.sig0 = 0.05 * self.Hs
        self.net = get_net('results/direction3/prior_net_l1s2bw.pt')
        self.head = SmultHead().to(DEVICE)
        self.head.load_state_dict(torch.load(
            'results/direction3/prior_net_l1s2bw.pt',
            map_location=DEVICE)['head'])
        self.head.eval()

    def predict(self, Q):
        with torch.no_grad():
            ph = mmse_gpu(
                torch.tensor(self.P, device=DEVICE, dtype=torch.float32),
                torch.tensor(self.z, device=DEVICE, dtype=torch.float32),
                self.cfg['depth'],
                torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                self.sig0, kp_hint=self.kp).cpu().numpy()
        pn = net_gamma_mmse(self.net, self.P, self.z, self.cfg['depth'],
                            Q, self.sig0, seed=self.i0, kp_hint=self.kp,
                            use_em=False, smult_head=self.head)
        return np.asarray(ph), np.asarray(pn)

    def ship_pos(self, t):
        """recompute the deterministic straight track from stored Pq."""
        # Pq rows are [x, y, t] at integer leads; interpolate linearly
        return (np.interp(t, self.Pq[:, 2], self.Pq[:, 0]),
                np.interp(t, self.Pq[:, 2], self.Pq[:, 1]))

    def track_series(self, lead_max=10.0):
        """truth + both arms at the encounter point, every truth frame."""
        js = np.arange(0, int(np.floor(lead_max / self.rec.dt)) + 1)
        leads = js * self.rec.dt
        pts = np.array([[*self.ship_pos(self.t0 + l_), self.t0 + l_]
                        for l_ in leads])
        gt = np.array([bilinear(self.Zfut[j], self.rec.gx, self.rec.gy,
                                px, py)
                       for j, (px, py, _t) in zip(js, pts)])
        ph, pn = self.predict(pts)
        return leads, gt, ph, pn

    def protocol_corr(self):
        """verbatim eval-protocol corr (1 s leads, <=10 s) for fact-check."""
        v = np.isfinite(self.gt_q) & (self.leads <= 10.0)
        ph, pn = self.predict(self.Pq)
        return (float(np.corrcoef(ph[v], self.gt_q[v])[0, 1]),
                float(np.corrcoef(pn[v], self.gt_q[v])[0, 1]))

    def fields(self, leads, stride=2):
        gx, gy = self.rec.gx[::stride], self.rec.gy[::stride]
        Xg, Yg = np.meshgrid(gx, gy)
        truth, fh, fn = [], [], []
        for l_ in leads:
            Zt = self.Zfut[int(round(l_ / self.rec.dt))][::stride, ::stride]
            Q = np.column_stack([Xg.ravel(), Yg.ravel(),
                                 np.full(Xg.size, self.t0 + l_)])
            ph, pn = self.predict(Q)
            truth.append(Zt)
            fh.append(ph.reshape(Xg.shape))
            fn.append(pn.reshape(Xg.shape))
        return gx, gy, truth, fh, fn


def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.6)
    ax.grid(axis='x', visible=False)


def fig1(showcase):
    fig, axs = plt.subplots(len(showcase), 1, figsize=(8.6, 7.4),
                            sharex=False, dpi=200)
    site = {'LJ18g': 'La Jument 2018 (Atlantic, storm)',
            'YS17': 'Yellow Sea 2017 (wind sea)',
            'AA15b': 'Acqua Alta 2015 (Adriatic)'}
    checks = []
    for ax, (name, wi) in zip(axs, showcase):
        w = Window(name, wi)
        leads, gt, ph, pn = w.track_series()
        ch, cn = w.protocol_corr()
        jr = json_row(name, wi)
        checks.append((name, wi, cn, jr['bw_s0_head'], ch, jr['hand']))
        m = np.isfinite(gt)
        ax.plot(leads[m], gt[m], '-', color=C_INK, lw=2, marker='o', ms=4,
                label='truth (stereo camera)')
        ax.plot(leads, pn, '-', color=C_NET, lw=2,
                label='prediction (learned)')
        cc = float(np.corrcoef(pn[m], gt[m])[0, 1])
        ax.text(0.995, 0.04,
                f"$\\rho$ = {cn:+.2f} (protocol)   Hs {w.Hs:.2f} m",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color=C_INK2,
                bbox=dict(facecolor='#fcfcfb', alpha=0.85,
                          edgecolor='none', pad=1.5))
        ax.set_title(f"{site[name]} — window {wi}", fontsize=10,
                     loc='left', color=C_INK)
        ax.set_ylabel('elevation [m]')
        style_ax(ax)
        print(f"fig1 {name} w{wi}: protocol net+head {cn:+.3f} "
              f"(json {jr['bw_s0_head']:+.3f})  fine-track {cc:+.3f}")
    axs[0].legend(loc='upper right', frameon=False, fontsize=9)
    axs[-1].set_xlabel('forecast lead at the ship encounter point [s]')
    fig.suptitle('Real sea, unseen future: wave elevation at the ship '
                 'encounter point,\npredicted from 40 s of sparse '
                 'LiDAR-like samples (three ocean sites)',
                 fontsize=11.5, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = f'{OUT}/fig_ext1_track_match_{DATE}.png'
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)
    return checks


def fig2(name='AA14', wi=10):
    w = Window(name, wi)
    jr = json_row(name, wi)
    ch, cn = w.protocol_corr()
    print(f"fig2 {name} w{wi}: protocol hand {ch:+.3f} (json "
          f"{jr['hand']:+.3f})  net+head {cn:+.3f} (json "
          f"{jr['bw_s0_head']:+.3f})")

    show = [l_ for l_ in w.leads if l_ <= 10.0]
    gx, gy, tr, fh, fn = w.fields(show, stride=2)

    def fcorr(pred, i):
        mi = np.isfinite(tr[i])
        return float(np.corrcoef(pred[i][mi], tr[i][mi])[0, 1])

    cl_h = [fcorr(fh, i) for i in range(len(show))]
    cl_n = [fcorr(fn, i) for i in range(len(show))]
    print(f"fig2 lead-0 FIELD corr: hand {cl_h[0]:+.3f} net {cl_n[0]:+.3f}"
          f"  (doc claim +0.72 / +0.66)")
    trs = np.array(tr)
    m3 = np.isfinite(trs)
    sd_t = float(np.nanstd(trs[m3]))
    sd_h = float(np.std(np.array(fh)[m3]))
    sd_n = float(np.std(np.array(fn)[m3]))
    print(f"fig2 amplitude ratio: hand {sd_h/sd_t:.2f}x  net {sd_n/sd_t:.2f}x")

    mask0 = ~np.isfinite(tr[0])
    Zt = tr[0]
    Zh = np.where(mask0, np.nan, fh[0])
    Zn = np.where(mask0, np.nan, fn[0])
    vmax = np.nanpercentile(np.abs(trs), 97)
    sp = w.ship_pos(w.t0 + show[0])

    fig = plt.figure(figsize=(11.2, 7.8), dpi=200)
    gs = fig.add_gridspec(2, 6, height_ratios=(1.12, 1), hspace=0.30,
                          wspace=0.55)
    panels = [(Zt, 'truth (stereo camera)', None),
              (Zh, 'physics prior ("hand")', cl_h[0]),
              (Zn, 'learned (net + s2-head)', cl_n[0])]
    for k, (Z, ttl, cf) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        ax.pcolormesh(gx, gy, Z, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      shading='auto')
        ax.plot(*sp, marker='*', ms=15, mec=C_INK, mfc='yellow', mew=0.8)
        sub = '' if cf is None else f"  field corr {cf:+.2f}"
        ax.set_title(ttl + sub, fontsize=9.5, color=C_INK)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    axl = fig.add_subplot(gs[1, :3])
    axl.axhline(0, color=C_AXIS, lw=1)
    axl.plot(show, cl_h, '-o', color=C_HAND, lw=2, ms=4.5)
    axl.plot(show, cl_n, '-o', color=C_NET, lw=2, ms=4.5)
    axl.text(show[1], cl_h[1] + 0.05, 'hand', color=C_HAND, fontsize=9)
    axl.text(show[3], cl_n[3] - 0.09, 'learned', color=C_NET, fontsize=9)
    axl.set_xlabel('forecast lead [s]')
    axl.set_ylabel('whole-field corr')
    axl.set_title('hand starts ahead, decays faster with lead\n'
                  '(negative by 10 s; learned crosses above at ~7 s)',
                  fontsize=10, loc='left', color=C_INK)
    style_ax(axl)

    axb = fig.add_subplot(gs[1, 3:])
    leads, gt, ph, pn = w.track_series()
    m = np.isfinite(gt)
    # the encounter point leaves the stereo-valid region after some lead;
    # show only the span where truth exists (the corr uses the same span)
    iend = int(np.max(np.flatnonzero(m)))
    leads, gt, ph, pn = (a[:iend + 1] for a in (leads, gt, ph, pn))
    m = m[:iend + 1]
    axb.plot(leads[m], gt[m], '-', color=C_INK, lw=2, marker='o', ms=4)
    axb.plot(leads, ph, '-', color=C_HAND, lw=2)
    axb.plot(leads, pn, '-', color=C_NET, lw=2)
    axb.text(leads[-1] + 0.1, gt[m][-1], 'truth', color=C_INK, fontsize=9,
             va='center')
    axb.text(leads[-1] + 0.1, ph[-1], f'hand  $\\rho$ {ch:+.2f}',
             color=C_HAND, fontsize=9, va='center')
    axb.text(leads[-1] + 0.1, pn[-1] - 0.02,
             f'learned  $\\rho$ {cn:+.2f}', color=C_NET, fontsize=9,
             va='center')
    axb.set_xlim(-0.15, leads[-1] + 2.2)
    axb.set_xlabel('forecast lead at the encounter point [s]')
    axb.set_ylabel('elevation [m]')
    axb.set_title('at the point the ship will meet the wave (★),\n'
                  'hand is anti-phase — learned holds', fontsize=10,
                  loc='left', color=C_INK)
    style_ax(axb)
    fig.suptitle(
        f'"Looks better" is not "is better" — {name} window {wi}\n'
        f'hand does look better now (field corr {cl_h[0]:+.2f} vs '
        f'{cl_n[0]:+.2f}, amplitude {sd_h/sd_t:.2f}x vs {sd_n/sd_t:.2f}x)\n'
        f'— but at the ship encounter point: hand $\\rho$ {ch:+.2f}, '
        f'learned $\\rho$ {cn:+.2f}',
        fontsize=11, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    p = f'{OUT}/fig_ext2_field_vs_track_{DATE}.png'
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)
    return dict(hand=ch, net=cn, field0_hand=cl_h[0], field0_net=cl_n[0],
                amp_hand=sd_h / sd_t, amp_net=sd_n / sd_t)


def fig3():
    h8 = json.load(open(f'{OUT}/sa_multirecord_mmse.json'))
    n8 = json.load(open(f'{OUT}/sa_multirecord_mmse_net.json'))
    recs = ['BS11a', 'AA14', 'AA15a', 'BS11b', 'BS13a', 'BS13b', 'AA15b',
            'YS17']

    def fixed(d, r):
        return float(np.mean(
            [w['conditions']['A']['mean']['common_unobs']['corr_bl']
             for w in d[r]['windows']]))

    hv = np.array([fixed(h8, r) for r in recs])
    nv = np.array([fixed(n8, r) for r in recs])
    print(f"fig3 A: hand {hv.mean():.3f} net {nv.mean():.3f} "
          f"({nv.mean()/hv.mean()-1:+.0%}, wins {(nv>hv).sum()}/8)")

    hand = np.array([r['hand'] for r in ROWS])
    net = np.array([r['bw_s0_head'] for r in ROWS])
    d = net - hand
    bad = hand < 0.5
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"fig3 B: all d {d.mean():+.3f} (t {d.mean()/se:.2f}), "
          f"bad d {d[bad].mean():+.3f} (n={bad.sum()}), "
          f"wins {(d>0).sum()}/60")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=200)
    x = np.arange(len(recs))
    a.bar(x - 0.19, hv, 0.36, color=C_HAND, label='physics prior (hand)')
    a.bar(x + 0.19, nv, 0.36, color=C_NET, label='learned prior '
          '($\\gamma$-net)')
    for xi, (l_, m_) in enumerate(zip(hv, nv)):
        a.text(xi + 0.19, m_ + 0.012, f'+{m_-l_:.2f}', ha='center',
               fontsize=8, color=C_INK2)
    a.set_xticks(x, recs, rotation=30, ha='right')
    a.set_ylim(0, 0.78)
    a.set_ylabel('nowcast corr (fixed aperture, real records)')
    a.legend(loc='upper left', frameon=False, fontsize=9)
    a.set_title(f'A   learned spectral prior: 8/8 records\n'
                f'(mean {hv.mean():.2f} $\\rightarrow$ {nv.mean():.2f}, '
                f'{nv.mean()/hv.mean()-1:+.0%})', loc='left', fontsize=10.5,
                color=C_INK)
    style_ax(a)

    b.axvspan(-0.65, 0.5, color=C_BADBG, zorder=0)
    lims = (-0.65, 1.02)
    b.plot(lims, lims, color=C_AXIS, lw=1, zorder=1)
    b.scatter(hand[~bad], net[~bad], s=34, color=C_NET, alpha=0.85,
              edgecolors='#fcfcfb', linewidths=0.8, zorder=3)
    b.scatter(hand[bad], net[bad], s=34, color=C_NET, alpha=0.85,
              edgecolors=C_INK, linewidths=0.8, zorder=3)
    for nm, wi_ in (('AA14', 10), ('LJ18g', 10)):
        r = json_row(nm, wi_)
        b.annotate(f'{nm} w{wi_}', (r['hand'], r['bw_s0_head']),
                   textcoords='offset points', xytext=(8, -4), fontsize=8,
                   color=C_INK2)
    b.set_xlim(*lims)
    b.set_ylim(*lims)
    b.set_xlabel('encounter corr, physics prior (hand)')
    b.set_ylabel('encounter corr, learned (net + s2-head)')
    b.text(0.03, 0.96, 'hand-failure region (hand < 0.5):\n'
           f'learned +{d[bad].mean():.2f} mean gain',
           transform=b.transAxes, fontsize=8.5, color='#7a5c2e',
           va='top')
    b.text(0.97, 0.05, 'above diagonal = learned wins '
           f'({(d>0).sum()}/60)', transform=b.transAxes, fontsize=8.5,
           color=C_MUT, ha='right')
    b.set_title(f'B   60 frozen real windows, paired\n'
                f'(all +{d.mean():.3f}, t {d.mean()/se:.1f}; '
                f'failure windows +{d[bad].mean():.2f})', loc='left',
                fontsize=10.5, color=C_INK)
    style_ax(b)
    b.grid(axis='x', linewidth=0.6)

    fig.suptitle('Where learning wins on real seas — same data, same '
                 'physics, learned prior + learned noise floor',
                 fontsize=12, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = f'{OUT}/fig_ext3_aggregate_{DATE}.png'
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


def main():
    fig3()
    checks = fig1([('LJ18g', 8), ('YS17', 2), ('AA15b', 10)])
    res2 = fig2()
    print('\n=== fact-check summary ===')
    for name, wi, cn, cn_j, ch, ch_j in checks:
        print(f"{name} w{wi}: net {cn:+.3f} vs json {cn_j:+.3f} "
              f"(d {abs(cn-cn_j):.3f}); hand {ch:+.3f} vs json {ch_j:+.3f}")
    print('AA14 w10:', {k: round(v, 3) for k, v in res2.items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
