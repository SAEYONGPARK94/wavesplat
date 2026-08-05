"""Summary comparison figure from today's HOS results (honest, same-setup comparisons).
(A) HOS ladder: learned net-MMSE vs WEAK classical (RELAX) [hos_ladder setup].
(B) analytic prior (WaveSplat) vs learned prior (WavePrior), SAME MMSE, across cases [compare_metrics].
(C) §8 operability-envelope skill prediction: CRB vs empirical-UQ [hos_s8].
Out: tmp/fig_summary_hos.png
"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

lad = json.load(open('results/direction3/hos_ladder.json'))
s8 = json.load(open('results/direction3/hos_s8_envelope.json'))
cm = json.load(open('results/direction3/compare_metrics.json'))

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5))

# (A) HOS ladder — learned vs WEAK classical (same hos_ladder setup, n=8, 600 obs)
methods = ['RELAX\n(weak\nclassical)', 'linear-fit\nceiling', 'net-MMSE\n(learned)', 'better-net\n(comp-div)']
vals = [lad['relax'], lad['rich'], lad['net_v3'], lad['net_cd']]
a1.bar(range(4), vals, color=['#888', '#bbb', '#2a7', '#7a2'], edgecolor='k')
a1.set_xticks(range(4)); a1.set_xticklabels(methods, fontsize=8.5)
a1.set_ylabel('recon corr vs HOS truth', fontsize=11); a1.set_ylim(0, 1.0); a1.grid(axis='y', alpha=0.3)
a1.set_title('(A) HOS recon (n=8): learned beats\nWEAK classical (RELAX). "better net" no gain', fontsize=9.5)
for i, v in enumerate(vals): a1.text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=9.5, fontweight='bold')

# (B) analytic prior vs learned prior — SAME MMSE, same setup per case (the user's point)
cases = ['linear', 'hos', 'real']; xl = ['linear', 'HOS', 'real stereo']
ws = [cm[c]['WaveSplat']['map_corr_nowcast'] for c in cases]
wp = [cm[c]['WavePrior']['map_corr_nowcast'] for c in cases]
xx = np.arange(3); w = 0.36
a2.bar(xx-w/2, ws, w, label='WaveSplat (analytic prior)', color='#48c', edgecolor='k')
a2.bar(xx+w/2, wp, w, label='WavePrior (learned prior)', color='#2a7', edgecolor='k')
a2.set_xticks(xx); a2.set_xticklabels(xl, fontsize=10); a2.set_ylim(0, 1.0); a2.grid(axis='y', alpha=0.3)
a2.set_ylabel('nowcast corr (η)', fontsize=11); a2.legend(fontsize=8.5, loc='lower left')
a2.set_title('(B) SAME MMSE, analytic vs learned PRIOR:\n≈ TIE (analytic wins linear/real, learned wins HOS)', fontsize=9.5)
for i in range(3):
    a2.text(i-w/2, ws[i]+0.015, f'{ws[i]:.2f}', ha='center', fontsize=8.5)
    a2.text(i+w/2, wp[i]+0.015, f'{wp[i]:.2f}', ha='center', fontsize=8.5)

# (C) §8 envelope skill prediction on HOS
preds = ['CRB envelope\n(ours)', 'empirical-UQ\n(Zhao proxy)', 'distance\nonly']
pv = [s8['crb_act'], s8['emp_act'], s8['dist_act']]
a3.bar(range(3), pv, color=['#25a', '#c72', '#bbb'], edgecolor='k')
a3.set_xticks(range(3)); a3.set_xticklabels(preds, fontsize=9)
a3.set_ylabel('Spearman(predictor, actual skill)', fontsize=11); a3.set_ylim(0, 0.55); a3.grid(axis='y', alpha=0.3)
a3.set_title('(C) HOS operability-envelope prediction (n=8):\nCRB envelope beats empirical-UQ (8/8 seas)', fontsize=9.5)
for i, v in enumerate(pv): a3.text(i, v+0.01, f'{v:.2f}', ha='center', fontsize=9.5, fontweight='bold')
a3.axhline(0.5, color='r', ls=':', alpha=0.5); a3.text(1.9, 0.51, 'PASS bar', color='r', fontsize=8)

fig.suptitle('WaveSplat — HOS-validated results (2026-07-22, honest same-setup comparisons)', fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('tmp/fig_summary_hos.png', dpi=135, bbox_inches='tight')
print('saved tmp/fig_summary_hos.png')
print(f"  (B) WaveSplat vs WavePrior: lin {ws[0]:.2f}/{wp[0]:.2f}  hos {ws[1]:.2f}/{wp[1]:.2f}  real {ws[2]:.2f}/{wp[2]:.2f}")
