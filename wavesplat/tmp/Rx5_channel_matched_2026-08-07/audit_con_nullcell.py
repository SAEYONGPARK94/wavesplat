#!/usr/bin/env python3
"""CON audit part 4 — per-cell null gate (prereg 4.6), config identity vs K6,
integrity completeness."""
import json, os, math
HERE=os.path.dirname(os.path.abspath(__file__))
K6='/home/vexit/wavesplat/wavesplat/tmp/K6_run_2026-08-07'
nulls={d['episode']:d for d in json.load(open(os.path.join(K6,'nulls_k6.json')))}

recs=[json.loads(l) for l in open(os.path.join(HERE,'results.jsonl'))]
eps=sorted({r['episode'] for r in recs}); seeds=sorted({r['seed'] for r in recs})
cells=sorted({r['cell'] for r in recs})
by={(r['cell'],r['episode'],r['seed']):r for r in recs}
def med(x):
    x=sorted(x);n=len(x);return x[n//2] if n%2 else .5*(x[n//2-1]+x[n//2])

print("A. PER-CELL NULL GATE (prereg 4.6: a cell below the zero-null may not be cited)")
print(f"   {'cell':14s} {'blk med':>8s} {'null med':>8s} {'ep beating null':>16s} {'run-level 48':>13s}")
nm=med([nulls[e]['zero_null_tracking_nRMSE_mean'] for e in eps])
for c in cells:
    b={e:med([by[(c,e,s)]['nRMSE'] for s in seeds]) for e in eps}
    w=sum(1 for e in eps if b[e]<nulls[e]['zero_null_tracking_nRMSE_mean'])
    wr=sum(1 for e in eps for s in seeds if by[(c,e,s)]['nRMSE']<nulls[e]['zero_null_tracking_nRMSE_mean'])
    print(f"   {c:14s} {med(list(b.values())):8.4f} {nm:8.4f} {w:12d}/16 {wr:10d}/48")

print("\nB. INTEGRITY COMPLETENESS")
print(f"   runs={len(recs)} expected={14*16*3}  unique keys={len(by)}")
bad=[k for k,r in by.items() if r.get('nRMSE') is None or (isinstance(r.get('nRMSE'),float) and math.isnan(r['nRMSE']))]
print(f"   NaN/missing nRMSE: {len(bad)}")
print(f"   'error' key present: {sum(1 for r in recs if 'error' in r)}")

print("\nC. CONFIG IDENTITY: Rx5 AUTO vs K6 AUTO")
k6=[json.loads(l) for l in open(os.path.join(K6,'results.jsonl'))]
k6a=[r for r in k6 if r['cell']=='AUTO'][0]
rx5a=[r for r in recs if r['cell']=='AUTO'][0]
ck,cr=k6a.get('cfg',{}),rx5a.get('cfg',{})
diff={k:(ck.get(k,'<absent>'),cr.get(k,'<absent>')) for k in set(ck)|set(cr) if ck.get(k,'<absent>')!=cr.get(k,'<absent>')}
print(f"   cfg diff: {diff if diff else 'NONE — identical config, only the seed differs'}")

print("\nD. W-MATCH: how much of the w_only tracking damage has been switched off?")
NT={e:med([by[('AUTO-NOTRACK',e,s)]['nRMSE'] for s in seeds]) for e in eps}
W3={e:med([by[('W-5e3',e,s)]['nRMSE'] for s in seeds]) for e in eps}
WM={e:med([by[('W-MATCH',e,s)]['nRMSE'] for s in seeds]) for e in eps}
W6={e:med([by[('W-5e6',e,s)]['nRMSE'] for s in seeds]) for e in eps}
AU={e:med([by[('AUTO',e,s)]['nRMSE'] for s in seeds]) for e in eps}
for lbl,B in [('W-5e3',W3),('W-MATCH',WM),('W-5e6',W6)]:
    frac=[1-(B[e]-NT[e])/(W3[e]-NT[e]) for e in eps if abs(W3[e]-NT[e])>1e-9]
    print(f"   {lbl:8s}: median per-episode fraction of w_only damage removed = {100*med(frac):6.1f}%")
print(f"   (AUTO's own damage over floor = {med([AU[e]-NT[e] for e in eps]):+.5f};"
      f" W-MATCH's residual = {med([WM[e]-NT[e] for e in eps]):+.5f}"
      f" = {100*med([WM[e]-NT[e] for e in eps])/med([AU[e]-NT[e] for e in eps]):.1f}% of it)")
print("DONE")
