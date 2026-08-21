#!/usr/bin/env python3
"""CON: is T-PHIONLY vs T-WONLY a clean CHANNEL contrast, or is it confounded with
effective step size?  Adam with a single lr=0.005 moves an amplitude in metres and a phase
in radians by the SAME absolute amount per step, but those are not the same relative move."""
import json, math, os, statistics as st
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__))
ROWS = [json.loads(l) for l in open(os.path.join(D, 'results.jsonl')) if l.strip()]
NULLS = {r['episode']: r['zero_null_tracking_nRMSE_mean']
         for r in json.load(open(os.path.join(D, 'nulls_full.json')))}
CROWDED = [f'episode_BF{b}_{c}comp_seed{s}_uniform.h5'
           for b in (4, 5) for c in (15, 30) for s in range(4)]
IDX = {(r['cell'], r['episode'], r['seed']): r for r in ROWS}
BY = defaultdict(list)
for r in ROWS: BY[(r['cell'], r['episode'])].append(r)
def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n//2] if n % 2 else .5*(xs[n//2-1]+xs[n//2])
KEEP = [e for e in CROWDED if med([x['nRMSE'] for x in BY[('AUTO', e)]]) < NULLS[e]]

def nrm(v): return math.sqrt(sum(x*x for x in v))

print('AUTO-NOTRACK never updates anything, so its final w IS the Phase-1 w0.')
print('measure ||w_cell - w0|| / ||w0|| and sum|w|/sum|w0| on block C keep (per run, then median).')
print()
print(f'{"cell":14s} {"med ||dw||/||w0||":>18s} {"med sum|w|/sum|w0|":>19s} {"w frozen?":>10s}')
for c in ['AUTO', 'T-PHIONLY', 'T-WONLY', 'T-EARLY20', 'T-K4REF', 'AUTO-NOTRACK']:
    rel, sm, frozen = [], [], True
    for e in KEEP:
        for s in (1524, 1525, 1526):
            a = IDX.get(('AUTO-NOTRACK', e, s)); b = IDX.get((c, e, s))
            if not a or not b: continue
            w0, w = a['w'], b['w']
            if len(w0) != len(w): continue
            dw = [x - y for x, y in zip(w, w0)]
            rel.append(nrm(dw) / max(nrm(w0), 1e-12))
            sm.append(sum(abs(x) for x in w) / max(sum(abs(x) for x in w0), 1e-12))
            if nrm(dw) > 1e-9: frozen = False
    print(f'{c:14s} {med(rel):18.4f} {med(sm):19.4f} {str(frozen):>10s}')
print()
print('=> T-PHIONLY has w bit-frozen (||dw||=0), confirming the cell does what it says.')
print('   AUTO and T-WONLY move the amplitude vector by ~the whole norm of w0 and shed')
print('   20-40% of the total |w| mass.  diag_analysis.txt sec.5 measured rms(dphi) at')
print('   0.71-0.98 rad = 11-16% of 2*pi over the same window.')
print('   So at a SINGLE Adam lr the amplitude channel is driven far harder in RELATIVE terms')
print('   than the phase channel.  tracking_mode therefore changes TWO things at once:')
print('     (i) which channel is free, and (ii) how big the effective relative step is.')
print('   => D1 (either direction) cannot separate "the damage lives in the amplitude channel"')
print('      from "lr=0.005 is a much larger relative step for w than for phi".')
print('      That is a BUNDLE hiding inside a one-key cell.  A clean D1 needs an lr sweep,')
print('      or per-channel lr matched by relative step; neither exists in this batch.')
