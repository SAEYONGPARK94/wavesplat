#!/usr/bin/env python3
"""Rederive rx2 'tracker eats 22-56% of placement gain' from raw results.jsonl."""
import json, statistics as st
from math import comb

rows = [json.loads(l) for l in open('tmp/rx2_init_2026-08-05/results.jsonl') if l.strip()]
rows = [r for r in rows if 'error' not in r]
EPS = sorted({r['episode'] for r in rows})

def sign_p(k, n):
    lo = min(k, n-k)
    return min(1.0, 2*sum(comb(n,i) for i in range(lo+1))/2**n)

def tab(metric, agg=st.median):
    d = {}
    for r in rows:
        d.setdefault((r['cell'], r['episode']), []).append(r[metric])
    return {k: agg(v) for k, v in d.items()}

def pdelta(cell, base, metric, seedagg=st.median, epagg=st.median):
    t = tab(metric, seedagg)
    dl = [t[(cell,e)] - t[(base,e)] for e in EPS]
    k = sum(1 for x in dl if x < 0)
    return epagg(dl), k, len(dl), sign_p(k, len(dl)), dl

CELLS = ['P1','P2','P3','S1','S2','W1','W2','W3','G1','G1P1','G1P3','AUTO','ORACLE-K','ORACLE']
print('CONVENTION: seed median-of-3 -> per-episode paired delta vs X-BASE -> median over 4 eps')
print('n=4 two-sided sign-test floor p=0.125\n')
print(f"{'cell':10s} {'D init_nrmse':>13s} {'k/n':>5s} {'D nRMSE(frmean)':>16s} {'k/n':>5s} {'p':>6s} {'retention':>10s}")
for c in CELLS:
    di,ki,n,pi,_ = pdelta(c,'X-BASE','init_nrmse')
    dn,kn,_,pn,_ = pdelta(c,'X-BASE','nRMSE')
    ret = dn/di if abs(di)>1e-9 else float('nan')
    print(f"{c:10s} {di:+13.4f} {ki}/{n:<3d} {dn:+16.4f} {kn}/{n:<3d} {pn:6.3f} {ret*100:9.1f}%")

print('\n--- Baseline levels (absolute, seed-median then ep-median) ---')
ti, tn = tab('init_nrmse'), tab('nRMSE')
for c in ['X-BASE','X-BASE2','X-BASE3','X-BASE4','G1','G1P1','AUTO','ORACLE-K','ORACLE']:
    print(f"  {c:10s} init={st.median([ti[(c,e)] for e in EPS]):.4f}  frame-mean nRMSE={st.median([tn[(c,e)] for e in EPS]):.4f}"
          f"  raw diff(track phase cost)={st.median([tn[(c,e)]-ti[(c,e)] for e in EPS]):+.4f}")

print('\n--- CONVENTION SENSITIVITY on retention (does the sign/size flip?) ---')
for c in ['P3','G1','G1P1','AUTO','ORACLE-K','ORACLE']:
    line = f"  {c:9s}"
    for sa, san in [(st.median,'medS'), (st.mean,'meanS')]:
        for ea, ean in [(st.median,'medE'), (st.mean,'meanE')]:
            di,_,_,_,_ = pdelta(c,'X-BASE','init_nrmse',sa,ea)
            dn,_,_,_,_ = pdelta(c,'X-BASE','nRMSE',sa,ea)
            line += f"  {san}/{ean}={dn/di*100:6.1f}%"
    print(line)

print('\n--- Per-episode retention spread (is 44-78% stable across episodes?) ---')
for c in ['P3','G1','G1P1','AUTO','ORACLE-K','ORACLE']:
    ti2, tn2 = tab('init_nrmse'), tab('nRMSE')
    rets = []
    for e in EPS:
        di = ti2[(c,e)] - ti2[('X-BASE',e)]
        dn = tn2[(c,e)] - tn2[('X-BASE',e)]
        rets.append(dn/di*100 if abs(di) > 1e-6 else float('nan'))
    print(f"  {c:9s} " + '  '.join(f"{'_'.join(e.split('_')[1:3])}={r:7.1f}%" for e, r in zip(EPS, rets)))

print('\n--- ORACLE cell absolute: does tracking phase HURT (rx1 claim) in rx2 data too? ---')
for c in ['ORACLE','ORACLE-K','X-BASE','AUTO','G1P1']:
    d,k,n,p,dl = None,None,None,None,None
    ti2, tn2 = tab('init_nrmse'), tab('nRMSE')
    dl = [tn2[(c,e)] - ti2[(c,e)] for e in EPS]
    k = sum(1 for x in dl if x < 0)
    print(f"  {c:9s} frame-mean minus init: med={st.median(dl):+.4f} k_improve/n={k}/{len(dl)} p={sign_p(k,len(dl)):.3f} per-ep={[round(x,4) for x in dl]}")
