"""Rx-4 step 1: independent re-derivation of B3 headline numbers from raw jsonl.
Written from scratch (does not import B3's analyze/rederive scripts)."""
import json, statistics as st
from collections import defaultdict

P = 'tmp/B3_rx3_2026-08-06/results.jsonl'
rows = [json.loads(l) for l in open(P) if l.strip()]
print(f'rows={len(rows)}')

CROWD = ['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
         'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
eps = sorted({r['episode'] for r in rows})
print('episodes:', len(eps))
for e in eps: print('   ', e)
cells = sorted({r['cell'] for r in rows}); print('cells:', cells)
seeds = sorted({r['seed'] for r in rows}); print('seeds:', seeds)

# median over seeds per (cell, episode)
def med(cell, ep, key, fn=None):
    v = [ (fn(r) if fn else r[key]) for r in rows if r['cell']==cell and r['episode']==ep ]
    v = [x for x in v if x is not None]
    return st.median(v) if v else None

print('\n=== A. crowded stratum: medOfMed nRMSE per cell ===')
for c in cells:
    per = [med(c,e,'nRMSE') for e in CROWD]
    print(f'  {c:15s} medOfMed={st.median(per):.4f}   per-ep={[round(x,4) for x in per]}')

print('\n=== B. paired Delta vs AUTO (crowded), median of per-episode deltas ===')
for c in cells:
    if c=='AUTO': continue
    d = [med(c,e,'nRMSE')-med('AUTO',e,'nRMSE') for e in CROWD]
    rel = [dd/med('AUTO',e,'nRMSE') for dd,e in zip(d,CROWD)]
    wins = sum(1 for x in d if x<0)
    print(f'  {c:15s} dMed={st.median(d):+.4f}  rel={st.median(rel)*100:+.1f}%  wins={wins}/4  per-ep={[round(x,4) for x in d]}')

print('\n=== C. final - init (crowded), per cell ===')
for c in cells:
    fi = [med(c,e,None,fn=lambda r: r['nRMSE']-r['init_nrmse']) for e in CROWD]
    print(f'  {c:15s} medOfMed(final-init)={st.median(fi):+.4f}  per-ep={[round(x,4) for x in fi]}')

print('\n=== D. panel: rho / |VR-1| deltas vs AUTO (crowded) ===')
for c in cells:
    if c=='AUTO': continue
    drho = st.median([med(c,e,'Pearson_rho')-med('AUTO',e,'Pearson_rho') for e in CROWD])
    dvr  = st.median([med(c,e,'absVR_framewise')-med('AUTO',e,'absVR_framewise') for e in CROWD])
    dm   = st.median([med(c,e,'m_mean')-med('AUTO',e,'m_mean') for e in CROWD])
    dde  = st.median([med(c,e,'decorr_mean')-med('AUTO',e,'decorr_mean') for e in CROWD])
    dsh  = st.median([med(c,e,'shrink_mean')-med('AUTO',e,'shrink_mean') for e in CROWD])
    print(f'  {c:15s} drho={drho:+.4f} d|VR-1|={dvr:+.4f} dm={dm:+.4f} ddecorr={dde:+.4f} dshrink={dsh:+.4f}')

print('\n=== E. partial-vs-full recovery %, NOTRACK as floor (crowded) ===')
base = st.median([med('AUTO-NOTRACK',e,'nRMSE')-med('AUTO',e,'nRMSE') for e in CROWD])
print(f'  NOTRACK floor delta = {base:+.4f}')
for c in cells:
    if c in ('AUTO','AUTO-NOTRACK'): continue
    d = st.median([med(c,e,'nRMSE')-med('AUTO',e,'nRMSE') for e in CROWD])
    print(f'  {c:15s} d={d:+.4f}  recovery={100*d/base:.1f}% of NOTRACK floor')

print('\n=== F. Phase-1 invariance check (init_nrmse/auto_selected/N identical across cells) ===')
bad=0
for e in eps:
    for s in seeds:
        vals = {}
        for c in cells:
            rr = [r for r in rows if r['cell']==c and r['episode']==e and r['seed']==s]
            if not rr: continue
            r=rr[0]
            vals[c]=(round(r['init_nrmse'],12), r['auto_selected'], r['auto_n_selected'], r['ensemble_k'], r['N'])
        uniq=set(vals.values())
        if len(uniq)>1:
            bad+=1
            print(f'  MISMATCH {e} seed{s}: {vals}')
print(f'  mismatches={bad}')
