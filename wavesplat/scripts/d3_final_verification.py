"""
d3_final_verification.py — end-to-end fact audit (2026-07-06, user
directive: full-session re-verification against the JSONs, mindful of
past WaveSplat failure modes). Every headline claim in the draft is
recomputed from its result file and compared to the claimed value.

Usage: python3 scripts/d3_final_verification.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

D = 'results/direction3'
PASS, FAIL = 0, 0


def check(name, got, expect, tol):
    global PASS, FAIL
    ok = abs(got - expect) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:+.3f} "
          f"(claimed {expect:+.3f} ±{tol})")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def pooled_corr(pairs, leads, lo, hi):
    m = (leads >= lo) & (leads <= hi)
    p = np.array([x for x, _ in pairs])[:, m].ravel()
    g = np.array([y for _, y in pairs])[:, m].ravel()
    return float(np.corrcoef(p, g)[0, 1])


print("=== C1: design point / horizon ===")
d = json.load(open(f'{D}/falcon_hos_probe_design22kt_n12.json'))
leads = np.array(d['leads'])
check("FF band 0-2s (W=10, n12)", pooled_corr(d['raw']['10.0'], leads,
                                              0, 2), 0.915, 0.005)
check("0-10s (W=10, n12)", pooled_corr(d['raw']['10.0'], leads, 0, 10),
      0.66, 0.05)

print("=== robustness (FF band, W=10) ===")
for tag, exp in (('pose03deg', 0.92), ('shadow3m', 0.96),
                 ('head60', 0.95), ('BF2', 0.90), ('BF5', 0.95)):
    dd = json.load(open(f'{D}/falcon_hos_probe_{tag}.json'))
    ll = np.array(dd['leads'])
    W = '10.0' if '10.0' in dd['raw'] else list(dd['raw'])[0]
    check(tag, pooled_corr(dd['raw'][W], ll, 0, 2), exp, 0.01)

print("=== real sites (best 0-10s band over W) ===")
for rec, exp in (('YS17', 0.71), ('AA15a', 0.71), ('LJ18g', 0.62),
                 ('AA15b', 0.57), ('AA14', 0.43)):
    dd = json.load(open(f'{D}/falcon_real_operator_{rec}.json'))
    ll = np.array(dd['leads'])
    best = -9
    for W in dd['w_mems']:
        preds = np.array([w['by_W'][str(W)]['pred']
                          for w in dd['windows']])
        gts = np.array([w['gt'] for w in dd['windows']])
        m = ll <= 10.0
        p, g = preds[:, m].ravel(), gts[:, m].ravel()
        v = np.isfinite(g)
        best = max(best, float(np.corrcoef(p[v], g[v])[0, 1]))
    check(rec, best, exp, 0.01)

print("=== streaming ===")
s = json.load(open(f'{D}/stream_seqmmse.json'))
check("equivalence err (log10)", np.log10(s['equivalence_err']), -13, 2)
s = json.load(open(f'{D}/stream_design_pipeline_hos_m48solo.json'))
check("latency median m48 solo (design)", s['lat_ms_median'], 18.5, 1.0)
check("latency p95 m48 solo (design)", s['lat_ms_p95'], 30.8, 2.0)
s = json.load(open(f'{D}/stream_design_pipeline_ys17_m48solo.json'))
check("real streaming lat median", s['lat_ms_median'], 17.6, 1.0)

print("=== information limits (clean testbed n2000) ===")
s = json.load(open(f'{D}/slope_oracle_gap_m48n2000.json'))
check("achieved slope", s['achieved']['slope']['corr'], 0.652, 0.005)
check("oracle slope", s['oracle']['slope']['corr'], 0.648, 0.005)
check("oracle gap (slope)", s['oracle']['slope']['corr']
      - s['achieved']['slope']['corr'], 0.0, 0.02)
s = json.load(open(f'{D}/slope_oracle_gap_lines_n2000.json'))
check("adjacency retraction (lines slope, clean)",
      s['achieved']['slope']['corr'], 0.512, 0.005)
s = json.load(open(f'{D}/stokes2_probe.json'))
check("bound-wave delta", s['stokes2']['slope']['corr']
      - s['linear']['slope']['corr'], 0.0, 0.005)

print("=== gate ledger ===")
g = json.load(open(f'{D}/gate_confirmatory.json'))
check("conf#1 spearman", g['spearman'], -0.111, 0.005)
check("conf#1 split", g['split_delta'], 0.172, 0.005)
g = json.load(open(f'{D}/gate_confirmatory2.json'))
check("conf#2 spearman", g['spearman'], -0.209, 0.005)
check("conf#2 split", g['split_delta'], 0.084, 0.005)
r = json.load(open(f'{D}/gate_roc.json'))
c50 = [o for o in r['curve'] if abs(o['avail'] - 0.5) < 1e-9][0]
check("gate 50% avail skill", c50['skill'], 0.68, 0.01)
c100 = [o for o in r['curve'] if abs(o['avail'] - 1.0) < 1e-9][0]
check("ungated skill", c100['skill'], 0.49, 0.01)

print("=== CI ===")
b = json.load(open(f'{D}/bootstrap_ci.json'))
check("FF CI center", b['ff_band_design']['est'], 0.915, 0.005)
check("real pooled CI center", b['real_pooled_W20']['est'], 0.703, 0.005)

print("=== E2E ===")
e = json.load(open(f'{D}/e2e_ctrl_tune.json'))
red = [r['red_gt'] for r in e['stage2']]
check("tuned GT-FF mean reduction", float(np.mean(red)), 0.18, 0.05)
fr = [r['frac'] for r in e['stage2']]
check("stream fraction mean", float(np.nanmean(fr)), 0.14, 0.08)

print(f"\nTOTAL: {PASS} PASS / {FAIL} FAIL")
