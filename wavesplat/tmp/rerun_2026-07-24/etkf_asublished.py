"""ETKF on the finemesh data with the *as-published* configuration.

This is deliberately NOT a re-tune -- it is the reference point that shows what the paper's
own ETKF settings do on Nyquist-adequate data. The re-tuned configuration is produced
separately (tmp/ETKF_FINEMESH_2026-07-24.md); reporting both is what makes the comparison
fair, since the published tuning was done on the artifact-laden coarse render.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from run_p1_comparison import run_etkf, EPISODES

FINE = os.path.join(REPO, 'data_finemesh_2026-07-23')
OLD = os.path.join(REPO, 'data')
OUT = os.path.join(HERE, 'etkf_aspublished.json')

out = {}
for ep in EPISODES:
    row = {}
    for tag, d in (('finemesh', FINE), ('oldmesh', OLD)):
        p = os.path.join(d, ep)
        if not os.path.exists(p):
            continue
        t0 = time.time()
        try:
            r = run_etkf(p, 'medium')
            row[tag] = {**r, 'wall_s': time.time() - t0}
        except Exception as e:
            row[tag] = {'error': repr(e)}
    out[ep] = row
    f = row.get('finemesh', {}).get('avg_nRMSE')
    o = row.get('oldmesh', {}).get('avg_nRMSE')
    d = f'{(f - o) / o * 100:+.1f}%' if (f and o) else '--'
    print(f'{ep:<45s} old={o if o is None else f"{o:.4f}"} '
          f'fine={f if f is None else f"{f:.4f}"}  {d}', flush=True)

with open(OUT, 'w') as fh:
    json.dump(out, fh, indent=2)
print(f'\nwrote {OUT}')
