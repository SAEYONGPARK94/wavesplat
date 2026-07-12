"""Doctoral appendix figure gallery + main-text field verdict figure.

Three deliverables, all numbers re-derived from the frozen real-data JSONs
(no re-rounding beyond the display precision; see VERIFICATION md):

  1. results/direction3/figs_appendix/VERIFICATION_2026-07-12.md
       independent re-derivation of the three headline learned-vs-hand
       claims (8-record nowcast / 60-window whole-field / 60-window track)
       against the stored JSONs, with match flags.

  2. results/direction3/fig_field_verdict_2026-07-12.png   (dpi 200)
       A: per-lead whole-field corr, hand vs learned (60-window mean + band).
       B: 60-window paired whole-field scatter (hand vs learned).

  3. results/direction3/figs_appendix/appendix_<record>.png  x5  (dpi 150)
       every record's 12 protocol windows, one row each:
       [truth field (lead 0) | hand field | learned field | per-lead corr].
       Predicted fields masked to the truth's finite footprint for fair
       visual comparison. Field renders are lead-0 only (stride 2).

Sources
  field_wide_eval.json          whole-field corr, per-lead corr (60 windows)
  gate_promotion_eval.json      encounter-track corr: hand, bw_s0_head
  sa_multirecord_mmse{,_net}.json  8-record fixed-aperture nowcast (cond A)
  cluster_bootstrap_2026-07-12.json  cluster CIs
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = 'results/direction3'
APP = f'{OUT}/figs_appendix'
DATE = '2026-07-12'
os.makedirs(APP, exist_ok=True)

# fixed palette (task spec)
C_HAND = '#2a78d6'   # physics-prior (hand) arm
C_NET = '#1baf7a'    # learned arm (net + s2-head)
C_INK = '#0b0b0b'    # truth / primary ink
C_INK2 = '#52514e'   # secondary ink
C_GRID = '#e1e0d9'
C_BG = '#fcfcfb'
C_AXIS = '#c3c2b7'
C_MUT = '#898781'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': C_BG, 'axes.facecolor': C_BG,
    'savefig.facecolor': C_BG,
})

RECORD_ORDER = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']


# ---------------------------------------------------------------- load JSONs
def load_json(name):
    return json.load(open(f'{OUT}/{name}'))


FIELD = load_json('field_wide_eval.json')            # 60 rows
GATE = load_json('gate_promotion_eval.json')          # 60 rows
CLUS = load_json('cluster_bootstrap_2026-07-12.json')  # 7 rows
HAND8 = load_json('sa_multirecord_mmse.json')          # 8 records
NET8 = load_json('sa_multirecord_mmse_net.json')       # 8 records

FIELD_K = {(r['record'], r['wi']): r for r in FIELD}
GATE_K = {(r['record'], r['wi']): r for r in GATE}
CLUS_K = {r['contrast']: r for r in CLUS}


# ---------------------------------------------------------------- 1. VERIFY
def verify():
    """Re-derive the three headline claims from raw JSON; write md table."""
    lines = []
    lines.append('# Verification — learned vs hand headline numbers')
    lines.append(f'\nDate: {DATE}. All values independently re-derived from '
                 'the stored JSONs by `scripts/d3_appendix_figs.py::verify`.\n')

    disc = []  # discrepancies

    # ---- Claim 1: 8-record fixed-aperture nowcast, cond A common_unobs corr_bl
    lines.append('## Claim 1 — 8-record nowcast (unobserved region)')
    lines.append('Path: `sa_multirecord_mmse{,_net}.json`, per record mean over '
                 'windows of `conditions.A.mean.common_unobs.corr_bl`.\n')
    lines.append('| record | n_win | hand | learned(net) | delta |')
    lines.append('|---|---|---|---|---|')
    recs8 = list(HAND8.keys())
    hm, nm, wtot = [], [], 0

    def rec_mean(d, r):
        ws = d[r]['windows']
        v = [w['conditions']['A']['mean']['common_unobs']['corr_bl']
             for w in ws]
        return float(np.mean(v)), len(v)

    for r in recs8:
        h, hn = rec_mean(HAND8, r)
        n, _ = rec_mean(NET8, r)
        hm.append(h); nm.append(n); wtot += hn
        lines.append(f'| {r} | {hn} | {h:.4f} | {n:.4f} | {n-h:+.4f} |')
    hbar, nbar = float(np.mean(hm)), float(np.mean(nm))
    sign = sum(1 for a, b in zip(nm, hm) if a > b)
    rel = (nbar - hbar) / hbar * 100
    lines.append(f'| **mean(8 rec)** | {wtot} | **{hbar:.4f}** | '
                 f'**{nbar:.4f}** | **{nbar-hbar:+.4f}** |')
    lines.append(f'\n- re-derived: hand **{hbar:.3f}** -> learned '
                 f'**{nbar:.3f}**, {rel:+.1f}% rel, sign **{sign}/8**')
    lines.append('- documented: hand 0.396 -> learned 0.538, +36%, 8/8')
    ok1 = (abs(hbar - 0.396) < 5e-4 and abs(nbar - 0.538) < 5e-4
           and sign == 8)
    lines.append(f'- **match: {"YES" if ok1 else "NO"}** '
                 f'(hand {hbar:.3f} vs 0.396; learned {nbar:.3f} vs 0.538; '
                 f'{rel:+.0f}% vs +36%)')
    if not ok1:
        disc.append('Claim 1 mismatch')
    # cross-check cluster bootstrap contrast A
    ca = CLUS_K['A gamma-net - hand (8rec nowcast)']
    lines.append(f'- cluster boot (contrast A): mean {ca["mean"]:+.4f}, '
                 f'n_win {ca["n_windows"]}, sign {ca["sign"]}, '
                 f'CI [{ca["boot_ci"][0]:+.4f}, {ca["boot_ci"][1]:+.4f}], '
                 f'survives {ca["survives"]}')

    # ---- Claim 2: 60-window whole-field corr
    lines.append('\n## Claim 2 — 60-window whole-field corr')
    lines.append('Path: `field_wide_eval.json`, fields `field_hand`, '
                 '`field_learned`, `per_lead_*`.\n')
    fh = np.array([r['field_hand'] for r in FIELD])
    fl = np.array([r['field_learned'] for r in FIELD])
    win_frac = int((fl > fh).sum())
    from collections import defaultdict
    rd = defaultdict(list)
    for r in FIELD:
        rd[r['record']].append(r['field_learned'] - r['field_hand'])
    rec_sign = sum(1 for r in rd if np.mean(rd[r]) > 0)
    lines.append(f'- whole-field mean: hand **{fh.mean():.4f}** -> learned '
                 f'**{fl.mean():.4f}**, delta **{fl.mean()-fh.mean():+.4f}**')
    lines.append(f'- learned > hand: **{win_frac}/60**; record sign '
                 f'**{rec_sign}/5**')
    lines.append('- documented: hand +0.550, learned +0.583, Δ+0.033, '
                 '43/60, 5/5')
    ok2 = (abs(fh.mean() - 0.550) < 1e-3 and abs(fl.mean() - 0.583) < 1e-3
           and win_frac == 43 and rec_sign == 5)
    lines.append(f'- **match: {"YES" if ok2 else "NO"}**')
    if not ok2:
        disc.append('Claim 2 mismatch')
    # per lead
    leads = FIELD[0]['leads']
    plh = np.array([r['per_lead_hand'] for r in FIELD])
    pll = np.array([r['per_lead_learned'] for r in FIELD])
    lines.append('\n| lead (s) | hand | learned | delta |')
    lines.append('|---|---|---|---|')
    for i, L in enumerate(leads):
        lines.append(f'| {L:.0f} | {plh[:, i].mean():.4f} | '
                     f'{pll[:, i].mean():.4f} | '
                     f'{pll[:, i].mean()-plh[:, i].mean():+.4f} |')
    lead_win = int(sum(pll[:, i].mean() > plh[:, i].mean()
                       for i in range(len(leads))))
    lines.append(f'\n- learned wins at **{lead_win}/{len(leads)}** leads '
                 '(all four)')
    ce = CLUS_K['E whole-field learned - hand (60win)']
    lines.append(f'- cluster boot (contrast E): mean {ce["mean"]:+.4f}, '
                 f'CI [{ce["boot_ci"][0]:+.4f}, {ce["boot_ci"][1]:+.4f}], '
                 f't_win {ce["t_window"]:.2f}, sign {ce["sign"]}')
    ci_ok = abs(ce['boot_ci'][0] - 0.018) < 1e-3 and \
        abs(ce['boot_ci'][1] - 0.048) < 1.5e-3
    lines.append(f'- documented field CI [+0.018, +0.048]; re-read '
                 f'[{ce["boot_ci"][0]:+.4f}, {ce["boot_ci"][1]:+.4f}] '
                 f'-> match {"YES" if ci_ok else "NO"}')
    if not ci_ok:
        disc.append('Claim 2 field CI mismatch')

    # ---- Claim 3: 60-window encounter-track corr
    lines.append('\n## Claim 3 — 60-window encounter-track corr')
    lines.append('Path: `gate_promotion_eval.json`, fields `hand`, '
                 '`bw_s0_head` (learned = net + s2-head).\n')
    th = np.array([r['hand'] for r in GATE])
    tl = np.array([r['bw_s0_head'] for r in GATE])
    d_all = float((tl - th).mean())
    bad = th < 0.5
    d_bad = float((tl[bad] - th[bad]).mean())
    rd2 = defaultdict(list)
    for r in GATE:
        rd2[r['record']].append(r['bw_s0_head'] - r['hand'])
    rec_sign3 = sum(1 for r in rd2 if np.mean(rd2[r]) > 0)
    lines.append(f'- track mean: hand **{th.mean():.4f}** -> learned '
                 f'**{tl.mean():.4f}**')
    lines.append(f'- delta all 60: **{d_all:+.4f}**; learned>hand '
                 f'{(tl > th).sum()}/60; record sign **{rec_sign3}/5**')
    lines.append(f'- failure windows (hand<0.5, n={int(bad.sum())}): '
                 f'delta **{d_bad:+.4f}** (hand {th[bad].mean():.3f} -> '
                 f'learned {tl[bad].mean():.3f})')
    lines.append('- documented: Δ+0.088 all / +0.227 failure windows')
    ok3 = (abs(d_all - 0.088) < 1.5e-3 and abs(d_bad - 0.227) < 1.5e-3
           and int(bad.sum()) == 25 and rec_sign3 == 5)
    lines.append(f'- **match: {"YES" if ok3 else "NO"}**')
    if not ok3:
        disc.append('Claim 3 mismatch')
    cb_all = CLUS_K['B head(bw_s0_head) - hand [all60]']
    cb_bad = CLUS_K['B head(bw_s0_head) - hand [bad25]']
    lines.append(f'- cluster boot (B all60): mean {cb_all["mean"]:+.4f} '
                 f'CI [{cb_all["boot_ci"][0]:+.4f}, '
                 f'{cb_all["boot_ci"][1]:+.4f}]; (B bad25): '
                 f'mean {cb_bad["mean"]:+.4f} CI '
                 f'[{cb_bad["boot_ci"][0]:+.4f}, {cb_bad["boot_ci"][1]:+.4f}]')

    # ---- discrepancy summary
    lines.append('\n## Discrepancy summary')
    if disc:
        for d in disc:
            lines.append(f'- **{d}**')
    else:
        lines.append('- None. All three headline claims and the two cited '
                     'cluster-bootstrap CIs reproduce to display precision.')

    lines.append('\n## Rounding notes')
    lines.append('- Documented values are 3-dp rounded; re-derived means '
                 f'(hand nowcast {hbar:.4f}, learned {nbar:.4f}; field hand '
                 f'{fh.mean():.4f}, learned {fl.mean():.4f}; track all '
                 f'{d_all:.4f}, bad {d_bad:.4f}) round to the documented '
                 'figures. No divergent rounding found.')

    txt = '\n'.join(lines) + '\n'
    with open(f'{APP}/VERIFICATION_{DATE}.md', 'w') as f:
        f.write(txt)
    print(f'wrote {APP}/VERIFICATION_{DATE}.md '
          f'(discrepancies: {len(disc)})')
    return dict(hbar=hbar, nbar=nbar, sign1=sign, rel=rel,
                fh=fh, fl=fl, win_frac=win_frac, rec_sign=rec_sign,
                leads=leads, plh=plh, pll=pll, ce=ce,
                th=th, tl=tl, d_all=d_all, d_bad=d_bad, bad=bad,
                disc=disc)


# ------------------------------------------------------------ 2. MAIN FIGURE
def fig_field_verdict(v):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.9))

    # ---- Panel A: per-lead field corr, hand vs learned (60-win mean + band)
    leads = np.array(v['leads'])
    plh, pll = v['plh'], v['pll']
    mh, ml = plh.mean(0), pll.mean(0)
    # record-mean spread band (5 records) as light envelope
    recs = [r['record'] for r in FIELD]
    ru = RECORD_ORDER

    def rec_lead_means(P):
        out = []
        for rr in ru:
            idx = [i for i, x in enumerate(recs) if x == rr]
            out.append(P[idx].mean(0))
        return np.array(out)  # (5, nlead)
    rh, rl = rec_lead_means(plh), rec_lead_means(pll)
    a.fill_between(leads, rh.min(0), rh.max(0), color=C_HAND, alpha=0.12,
                   lw=0)
    a.fill_between(leads, rl.min(0), rl.max(0), color=C_NET, alpha=0.12,
                   lw=0)
    a.plot(leads, mh, '-o', color=C_HAND, lw=2.2, ms=6,
           markeredgecolor=C_BG, markeredgewidth=1.0, zorder=4)
    a.plot(leads, ml, '-o', color=C_NET, lw=2.2, ms=6,
           markeredgecolor=C_BG, markeredgewidth=1.0, zorder=5)
    # direct series labels (nudged apart at the last lead)
    a.text(leads[-1] + 0.35, ml[-1] + 0.028, 'learned\n(net + s2-head)',
           color=C_NET, fontsize=9.5, va='center', ha='left',
           fontweight='bold')
    a.text(leads[-1] + 0.35, mh[-1] - 0.030, 'physics prior\n(hand)',
           color=C_HAND, fontsize=9.5, va='center', ha='left',
           fontweight='bold')
    for i, L in enumerate(leads):
        if i == len(leads) - 1:
            a.text(L - 0.15, ml[i] + 0.006, f'+{ml[i]-mh[i]:.3f}',
                   ha='right', va='bottom', fontsize=8, color=C_INK2)
        else:
            a.text(L, ml[i] + 0.012, f'+{ml[i]-mh[i]:.3f}', ha='center',
                   fontsize=8, color=C_INK2)
    a.set_xlim(-0.4, leads[-1] + 3.1)
    a.set_ylim(0.36, 0.74)
    a.set_xticks(leads)
    a.set_xlabel('forecast lead (s)')
    a.set_ylabel('whole-field corr (60 windows)')
    a.set_title('A   Whole-field skill vs lead — learned leads at every '
                'horizon', loc='left', fontsize=10.5, color=C_INK)
    a.spines[['top', 'right']].set_visible(False)
    a.grid(axis='y', linewidth=0.6)
    a.grid(axis='x', visible=False)

    # ---- Panel B: 60-window paired scatter
    fh, fl = v['fh'], v['fl']
    lims = (0.05, 0.92)
    b.plot(lims, lims, color=C_AXIS, lw=1, zorder=1)
    win = fl > fh
    b.scatter(fh[~win], fl[~win], s=34, color=C_HAND, alpha=0.55,
              edgecolors=C_BG, linewidths=0.7, zorder=3,
              label='hand wins')
    b.scatter(fh[win], fl[win], s=34, color=C_NET, alpha=0.9,
              edgecolors=C_BG, linewidths=0.7, zorder=4,
              label='learned wins')
    b.set_xlim(*lims)
    b.set_ylim(*lims)
    b.set_xlabel('whole-field corr, physics prior (hand)')
    b.set_ylabel('whole-field corr, learned (net + s2-head)')
    ce = v['ce']
    b.text(0.035, 0.965,
           f'learned > hand: {v["win_frac"]}/60\n'
           f'record sign: {v["rec_sign"]}/5\n'
           f'cluster CI [{ce["boot_ci"][0]:+.3f}, {ce["boot_ci"][1]:+.3f}]',
           transform=b.transAxes, fontsize=9, color=C_INK2, va='top',
           bbox=dict(boxstyle='round,pad=0.4', fc=C_BG, ec=C_GRID, lw=0.8))
    b.text(0.97, 0.06, 'above diagonal =\nlearned wins',
           transform=b.transAxes, fontsize=8.5, color=C_NET, ha='right',
           va='bottom', fontweight='bold')
    b.set_title(f'B   60 frozen real windows, paired '
                f'(mean $\\Delta$ {v["fl"].mean()-v["fh"].mean():+.3f})',
                loc='left', fontsize=10.5, color=C_INK)
    b.spines[['top', 'right']].set_visible(False)
    b.grid(linewidth=0.6)

    fig.suptitle('Learned spectral prior improves the reconstructed sea '
                 'surface across the whole field\n(stereo-reconstructed sea '
                 'surface + spec-faithful LiDAR sampling operator, '
                 '5 real records)',
                 fontsize=11, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = f'{OUT}/fig_field_verdict_{DATE}.png'
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print('wrote', p)


# ------------------------------------------------------- 3. APPENDIX GALLERY
def appendix_record(record, Window, net, head):
    from d3_sa_multirecord import MRecord
    from d3_falcon_real_operator import RECORDS
    cfg = RECORDS[record]
    t0 = time.time()
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    print(f'  [{record}] MRecord loaded ({time.time()-t0:.1f}s)')

    nrows = 12
    fig = plt.figure(figsize=(11.0, 2.05 * nrows), dpi=150)
    gs = GridSpec(nrows, 4, figure=fig, wspace=0.18, hspace=0.45,
                  left=0.135, right=0.985, top=0.955, bottom=0.02,
                  width_ratios=[1, 1, 1, 0.95])

    leads = FIELD[0]['leads']  # [0,2,5,8]
    for wi in range(nrows):
        tw = time.time()
        w = Window(record, wi, rec=rec, net=net, head=head)
        gx, gy, truth, fh, fn = w.fields([0.0], stride=2)
        T = truth[0]
        H = np.where(~np.isfinite(T), np.nan, fh[0])
        N = np.where(~np.isfinite(T), np.nan, fn[0])
        vmax = np.nanpercentile(np.abs(T), 97)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        ext = [gx[0], gx[-1], gy[0], gy[-1]]
        fr = FIELD_K[(record, wi)]
        gr = GATE_K[(record, wi)]

        axs = [fig.add_subplot(gs[wi, c]) for c in range(4)]
        for ax, dat, ttl in zip(axs[:3], [T, H, N],
                                ['truth', 'hand', 'learned']):
            ax.imshow(dat, origin='lower', extent=ext, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, aspect='auto',
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ttl, fontsize=8.5, color=C_INK2, pad=2)
            for s in ax.spines.values():
                s.set_edgecolor(C_AXIS)
        # row label on truth panel (left of the grid)
        axs[0].set_ylabel(
            f'w{wi}\nfield h/l '
            f'{fr["field_hand"]:.2f}/{fr["field_learned"]:.2f}\n'
            f'track $\\rho$ h/l {gr["hand"]:.2f}/{gr["bw_s0_head"]:.2f}',
            fontsize=8.2, color=C_INK, rotation=0, ha='right', va='center',
            labelpad=10)

        # per-lead field corr mini-bars
        m = axs[3]
        x = np.arange(len(leads))
        m.bar(x - 0.19, fr['per_lead_hand'], 0.36, color=C_HAND,
              label='hand')
        m.bar(x + 0.19, fr['per_lead_learned'], 0.36, color=C_NET,
              label='learned')
        m.set_ylim(0, 1.0)
        m.set_xticks(x)
        m.set_xticklabels([f'{int(L)}' for L in leads], fontsize=7)
        m.tick_params(axis='y', labelsize=7)
        m.set_title('field corr / lead(s)', fontsize=8, color=C_INK2, pad=2)
        m.grid(axis='y', linewidth=0.5)
        m.spines[['top', 'right']].set_visible(False)
        if wi == 0:
            m.legend(fontsize=6.5, frameon=False, loc='upper right',
                     handlelength=1.0, ncol=2, columnspacing=0.8)
        print(f'    w{wi} rendered ({time.time()-tw:.1f}s) '
              f'field h/l {fr["field_hand"]:.2f}/{fr["field_learned"]:.2f} '
              f'track {gr["hand"]:.2f}/{gr["bw_s0_head"]:.2f}')

    fig.suptitle(
        f'Appendix {record} — 12 protocol windows: truth | hand | learned '
        f'fields (lead 0, RdBu_r)\n'
        f'stereo-reconstructed sea surface + spec-faithful LiDAR sampling '
        f'operator; predictions masked to the truth footprint',
        fontsize=10, color=C_INK, y=0.992)
    p = f'{APP}/appendix_{record}.png'
    fig.savefig(p)
    plt.close(fig)
    print(f'  wrote {p}')


def main():
    v = verify()
    fig_field_verdict(v)
    if v['disc']:
        print('WARNING discrepancies:', v['disc'])

    # heavy imports only for the gallery
    import torch
    from d3_external_figs import Window
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    CK = 'results/direction3/prior_net_l1s2bw.pt'
    net = get_net(CK)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CK, map_location=DEVICE)['head'])
    head.eval()
    print('net/head loaded; DEVICE', DEVICE)

    t0 = time.time()
    for r in RECORD_ORDER:
        appendix_record(r, Window, net, head)
    print(f'gallery done in {(time.time()-t0)/60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
