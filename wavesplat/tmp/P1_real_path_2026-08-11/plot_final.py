import json, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
d = json.load(open('tmp/P1_real_path_2026-08-11/final_compare.json'))
recs = [r for r in d if 'error' not in d[r]]
fig, axes = plt.subplots(1, len(recs), figsize=(3.0*len(recs), 3.6), sharey=True)
SYN = 0.178   # synthetic crowded baseline, Hs units (S9 AUTO)
for ax, r in zip(np.atleast_1d(axes), recs):
    e = d[r]; hs = e['Hs_over_sigma']
    dep = np.array(e['deployed'])/hs; res = np.array(e['research'])/hs
    t = np.arange(len(dep))
    ax.plot(t, dep, '-o', ms=3, color='#1f77b4', label='deployed path')
    ax.plot(t, res, '--s', ms=3, color='#d62728', label='research path')
    ax.axhline(1.0/hs, color='k', lw=1.2, ls=':', label='zero-predictor (=1σ)')
    ax.axhline(SYN, color='green', lw=1.2, ls='-.', label='synthetic crowded')
    ax.set_title(f'{r}  (w{e["win"]})', fontsize=10)
    ax.set_xlabel('forecast lead [s]', fontsize=9); ax.grid(alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel('nRMSE  [Hs units]', fontsize=9)
np.atleast_1d(axes)[0].legend(fontsize=7, loc='upper left')
fig.suptitle('Real-sea wave estimation, all 5 records — deployed vs research path '
             '(normalised in Hs units so synthetic is comparable)', fontsize=11)
fig.tight_layout()
out='results/gifs/p1_real_5records.png'; fig.savefig(out, dpi=130, bbox_inches='tight')
print('wrote', out)
print()
print(f'{"record":8s} {"Hs/sig":>7s} {"nowcast(Hs)":>12s} {"leadmean(Hs)":>13s} {"vs synth 0.178":>15s}')
for r in recs:
    e=d[r]; hs=e['Hs_over_sigma']
    print(f'{r:8s} {hs:7.3f} {e["dep_nowcast"]/hs:12.3f} {e["dep_leadmean"]/hs:13.3f} '
          f'{e["dep_nowcast"]/hs/SYN:14.2f}x')
