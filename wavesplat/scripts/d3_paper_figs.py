"""
d3_paper_figs.py — paper-quality figure samples (2026-07-06, Q1 item 4;
a few to check look/feel before full typesetting pass). PNG (check) +
PDF (vector, paper) into results/figs/.

  fig_horizon    per-lead encounter corr at the design point (W=5/10/20)
                 with the information-horizon prediction R/(U+c_g)
  fig_robust     FF-band (0-2 s) corr across all stress configs
  fig_gate_roc   vector re-render of the gate reliability/availability

Usage: python3 scripts/d3_paper_figs.py
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

G = 9.81
FD = 'results/figs'
C1, C2, C3 = '#5B7EC9', '#D08770', '#7CA982'


def save(fig, name):
    fig.savefig(f'{FD}/{name}.png', dpi=160)
    fig.savefig(f'{FD}/{name}.pdf')
    print(f'saved {FD}/{name}.png/.pdf')


def fig_horizon():
    src = 'results/direction3/falcon_hos_probe_design22kt_n12.json'
    if not os.path.exists(src):
        src = 'results/direction3/falcon_hos_probe_design22kt.json'
    d = json.load(open(src))
    leads = np.array(d['leads'])
    U, Tp = d['U'], d['Tp']
    cp = G * Tp / (2 * np.pi)
    lead_star = 100.0 / (U + cp / 2)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=110)
    cols = {5.0: C3, 10.0: C1, 20.0: C2}
    for W, pairs in sorted(d['raw'].items(), key=lambda kv: float(kv[0])):
        preds = np.array([p for p, _ in pairs])
        gts = np.array([g for _, g in pairs])
        # pooled over a +-1 s lead band (n=4 seas per single lead is too
        # noisy for a per-lead point estimate)
        cc = []
        for l_ in leads:
            m = np.abs(leads - l_) <= 1.0
            cc.append(np.corrcoef(preds[:, m].ravel(),
                                  gts[:, m].ravel())[0, 1])
        ax.plot(leads, cc, '-o', ms=4, lw=1.8, color=cols[float(W)],
                label=f'memory {float(W):.0f} s')
    ax.axvline(lead_star, color='0.35', lw=1.2, ls='--')
    ax.annotate(r'$\ell^* = R_{max}/(U+c_g)$' + f' ≈ {lead_star:.1f} s',
                xy=(lead_star, 0.55), xytext=(lead_star + 0.3, 0.62),
                fontsize=9, color='0.25')
    ax.axvspan(0, 2, color='0.93', zorder=0)
    ax.annotate('FF preview band', xy=(1.0, 1.01), fontsize=8,
                ha='center', color='0.35')
    ax.set_xlabel('forecast lead [s]', fontsize=10)
    ax.set_ylabel('encounter corr', fontsize=10)
    ax.set_title('Design point (22 kt, BF4, HOS M=3): skill vs lead '
                 'and the information horizon', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.grid(axis='y', color='0.92', lw=0.7)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, 'fig_horizon')


def fig_robust():
    import glob
    cfgs = [('baseline', 'design22kt'), ('pose 0.05°', 'pose005deg'),
            ('pose 0.1°', 'pose01deg'), ('pose 0.3°', 'pose03deg'),
            ('crest shadowing', 'shadow3m'), ('heading 30°', 'head30'),
            ('heading 60°', 'head60'), ('BF2 (Hs 0.3 m)', 'BF2'),
            ('BF5 (Hs 2.0 m)', 'BF5')]
    names, vals = [], []
    for label, tag in cfgs:
        d = json.load(open(
            f'results/direction3/falcon_hos_probe_{tag}.json'))
        leads = np.array(d['leads'])
        W = '10.0' if '10.0' in d['raw'] else list(d['raw'])[0]
        pairs = d['raw'][W]
        preds = np.array([p for p, _ in pairs])
        gts = np.array([g for _, g in pairs])
        m = leads <= 2.0
        names.append(label)
        vals.append(float(np.corrcoef(preds[:, m].ravel(),
                                      gts[:, m].ravel())[0, 1]))
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=110)
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, height=0.62, color=C1)
    ax.barh(ypos[0], vals[0], height=0.62, color=C2)
    for y, v in zip(ypos, vals):
        ax.annotate(f'{v:.2f}', xy=(v - 0.005, y), va='center',
                    ha='right', fontsize=8, color='white')
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0.9, color='0.4', lw=1.0, ls=':')
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel('FF-band (0–2 s) encounter corr', fontsize=10)
    ax.set_title('Robustness at the design point: FF-band skill under '
                 'stress configs', fontsize=10)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, 'fig_robust')


def fig_gate_roc():
    d = json.load(open('results/direction3/gate_roc.json'))
    c = d['curve']
    av = [o['avail'] * 100 for o in c]
    sk = [o['skill'] for o in c]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=110)
    ax.plot(av, sk, '-o', ms=5, lw=2.0, color=C1)
    ax.axhline(sk[-1], color='0.6', lw=1.0, ls='--')
    ax.annotate(f'ungated: {sk[-1]:+.2f}', xy=(22, sk[-1] + 0.008),
                fontsize=9, color='0.35')
    for q, dxy in ((0.3, (7, -0.005)), (0.5, (5, 0.02))):
        o = [x for x in c if abs(x['avail'] - q) < 1e-9][0]
        ax.annotate(f"{int(q*100)}% avail → {o['skill']:+.2f}",
                    xy=(q * 100, o['skill']),
                    xytext=(q * 100 + dxy[0], o['skill'] + dxy[1]),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='-', color='0.5', lw=0.8))
    ax.set_xlabel('gate availability [% windows accepted]', fontsize=10)
    ax.set_ylabel('conditional encounter corr', fontsize=10)
    ax.set_title('Self-diagnosed reliability gate '
                 '(3 real records, 36 windows)', fontsize=10)
    ax.set_ylim(0.45, 0.76)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.grid(axis='y', color='0.92', lw=0.7)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, 'fig_gate_roc')


if __name__ == '__main__':
    os.makedirs(FD, exist_ok=True)
    fig_horizon()
    fig_robust()
    fig_gate_roc()
