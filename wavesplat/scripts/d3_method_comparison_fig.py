"""Method comparison on FRESH held-out data + the real 24-window eval.

User request 2026-07-11: after the G-REAL verdict, compare the methodologies
on a RANDOM dataset never used in training, and produce the comparison
figure.

Held-out synthetic: rng stream seed 777 (training shards used generator
seeds 0-stream for fan0/bwtask; 777 was never used), fan-mode scenes
(forecast queries), two families x 12 scenes:
  clean  bw_inject = 0    (well-specified linear -- the G-ZERO family)
  bw     bw_inject = 0.5  (off-shell bound-wave energy -- misspecified)
Methods, identical inputs per scene (paired):
  hand    mmse_gpu (grid+EM production path)
  v3      prior_net_v3 gamma via net_gamma_mmse
  l1s2bw  prior_net_l1s2bw gamma (run 2a, bw-misspecification-trained)
Metric: corr(pred, gt) on the forecast-lead queries (tq > T).

Figure: (A) held-out synthetic, per-scene dots + mean bar per family x
method; (B) real 24 windows (l1s2bw_real_eval.json), this-vs-hand paired
scatter with the bad-window region (hand < 0.5) shaded. Okabe-Ito CVD-safe
trio + distinct markers (identity never color-alone); no dual axes.
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

from d3_prior_net_data import gen_sample
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_gpu_mmse import mmse_gpu

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_JSON = 'results/direction3/method_comparison_heldout.json'
OUT_PNG = 'results/direction3/fig_method_comparison_2026-07-11.png'
N_PER = 12
COL = {'hand': '#0072B2', 'v3': '#E69F00', 'l1s2bw': '#009E73'}
MRK = {'hand': 'o', 'v3': 's', 'l1s2bw': '^'}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 else 0.0


def main():
    net_v3 = get_net('results/direction3/prior_net_v3.pt')
    net_l1 = get_net('results/direction3/prior_net_l1s2bw.pt')
    rng = np.random.default_rng(777)
    rows = []
    for fam, bw in (('clean', 0.0), ('bw', 0.5)):
        made = 0
        while made < N_PER:
            s = gen_sample(rng, fan_frac=1.0, bimodal_frac=0.0, bw_inject=bw)
            Q, gt = s['Q'], s['gt_q']
            fut = Q[:, 2] > s['P'][:, 2].max() + 0.05
            if fut.sum() < 40:
                continue
            P32 = torch.tensor(s['P'], device=DEV)
            z32 = torch.tensor(s['z'], device=DEV)
            Q32 = torch.tensor(Q, device=DEV)
            sig0 = s['nfrac'] * s['Hs']
            with torch.no_grad():
                ph = mmse_gpu(P32, z32, 50.0, Q32, sig0,
                              kp_hint=s['kp_e']).cpu().numpy()
                pv = net_gamma_mmse(net_v3, s['P'], s['z'], 50.0, Q, sig0,
                                    kp_hint=s['kp_e'])
                pl = net_gamma_mmse(net_l1, s['P'], s['z'], 50.0, Q, sig0,
                                    kp_hint=s['kp_e'])
            pv = pv.cpu().numpy() if torch.is_tensor(pv) else np.asarray(pv)
            pl = pl.cpu().numpy() if torch.is_tensor(pl) else np.asarray(pl)
            rows.append({'family': fam,
                         'hand': corr(ph[fut], gt[fut]),
                         'v3': corr(pv[fut], gt[fut]),
                         'l1s2bw': corr(pl[fut], gt[fut])})
            made += 1
            print(f"{fam} {made:2d}/12 hand {rows[-1]['hand']:+.2f} "
                  f"v3 {rows[-1]['v3']:+.2f} l1 {rows[-1]['l1s2bw']:+.2f}",
                  flush=True)
    json.dump(rows, open(OUT_JSON, 'w'), indent=2)

    real = json.load(open('results/direction3/l1s2bw_real_eval.json'))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    fig.patch.set_facecolor('white')

    # -- panel A: held-out synthetic ---------------------------------------
    METH = ['hand', 'v3', 'l1s2bw']
    for fi, fam in enumerate(('clean', 'bw')):
        sub = [r for r in rows if r['family'] == fam]
        for mi, m in enumerate(METH):
            x0 = fi * 4 + mi
            vals = np.array([r[m] for r in sub])
            xs = x0 + rng.uniform(-0.16, 0.16, len(vals))
            axA.scatter(xs, vals, s=22, marker=MRK[m], color=COL[m],
                        alpha=0.55, linewidths=0, zorder=3)
            axA.hlines(vals.mean(), x0 - 0.3, x0 + 0.3, color=COL[m],
                       lw=2.5, zorder=4)
            axA.annotate(f"{vals.mean():+.2f}", (x0, vals.mean()),
                         textcoords='offset points', xytext=(0, 7),
                         ha='center', fontsize=8, color='#333333')
    axA.set_xticks([1, 5])
    axA.set_xticklabels(['clean (well-specified)\nheld-out, n=12',
                         'bound-wave injected\nheld-out, n=12'], fontsize=9)
    axA.set_ylabel('forecast corr (lead queries)', fontsize=9)
    axA.set_title('A  Held-out synthetic (seed 777, never trained on)',
                  fontsize=10, loc='left')
    axA.grid(axis='y', color='#e8e8e8', lw=0.7, zorder=0)
    axA.set_axisbelow(True)
    for sp in ('top', 'right'):
        axA.spines[sp].set_visible(False)
    hd = [plt.Line2D([], [], color=COL[m], marker=MRK[m], ls='none',
                     label={'hand': 'hand (grid+EM)', 'v3': 'v3 net',
                            'l1s2bw': 'l1s2bw net (bw-trained)'}[m])
          for m in METH]
    axA.legend(handles=hd, fontsize=8, frameon=False, loc='lower left')

    # -- panel B: real 24 windows, paired ----------------------------------
    h = np.array([r['hand'] for r in real])
    t_ = np.array([r['this'] for r in real])
    v = np.array([r['v3'] for r in real])
    axB.axvspan(-0.6, 0.5, color='#f5eee6', zorder=0)
    axB.text(-0.52, 1.02, 'hand-bad windows', fontsize=8, color='#8a6d3b')
    lim = (-0.6, 1.1)
    axB.plot(lim, lim, color='#bbbbbb', lw=1, zorder=1)
    axB.scatter(h, v, s=30, marker=MRK['v3'], color=COL['v3'], alpha=0.6,
                label='v3 net', zorder=2)
    axB.scatter(h, t_, s=34, marker=MRK['l1s2bw'], color=COL['l1s2bw'],
                alpha=0.85, label='l1s2bw net', zorder=3)
    axB.set_xlim(lim)
    axB.set_ylim(lim)
    axB.set_xlabel('hand forecast corr (per window)', fontsize=9)
    axB.set_ylabel('net forecast corr', fontsize=9)
    axB.set_title('B  Real records, 24 fwd-fan windows (YS17+AA15a)',
                  fontsize=10, loc='left')
    axB.legend(fontsize=8, frameon=False, loc='lower right')
    for sp in ('top', 'right'):
        axB.spines[sp].set_visible(False)
    axB.grid(color='#e8e8e8', lw=0.7, zorder=0)
    axB.set_axisbelow(True)
    axB.annotate('above diagonal = net wins', (0.62, 1.02), fontsize=8,
                 color='#555555')

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches='tight')
    print(f"wrote {OUT_PNG} and {OUT_JSON}")

    for fam in ('clean', 'bw'):
        sub = [r for r in rows if r['family'] == fam]
        line = '  '.join(f"{m} {np.mean([r[m] for r in sub]):+.3f}"
                         for m in METH)
        print(f"held-out {fam:6s}: {line}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
