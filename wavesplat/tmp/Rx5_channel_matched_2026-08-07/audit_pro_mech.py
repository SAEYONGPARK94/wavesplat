#!/usr/bin/env python3
"""PRO part 4: mechanism check -- how far did each arm actually drive the amplitude vector?
AUTO-NOTRACK has tracking_lr=0, so its final w IS the Phase-1 w (Phase-1 invariance verified).
=> |w_cell - w_NOTRACK| is a GT-free measurement of realised amplitude drive.
Also: what fraction of the shipped-lr channel asymmetry does the lr knob alone remove?"""
import json, os, math
from statistics import median
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
idx = {(r["cell"], r["episode"], r["seed"]): r for r in rows}
EPS = sorted(set(r["episode"] for r in rows)); SEEDS = sorted(set(r["seed"] for r in rows))

def C(n,k): return math.comb(n,k)
def sgn_p(n,w):
    k=max(w,n-w); return min(1.0, 2.0*sum(C(n,i) for i in range(k,n+1))/2**n)

print("=" * 78)
print("A) realised amplitude drive: median_j |w_cell[j] - w_NOTRACK[j]| / median_j |w_NOTRACK[j]|")
print("   (NOTRACK w == Phase-1 w, frozen; Phase-1 invariance already verified 0 violations)")
print("=" * 78)
drive = defaultdict(list)
for c in ["AUTO","W-5e3","W-5e4","W-MATCH","W-5e6","PHI-5e3","E-2","E-10","E-20","E-80"]:
    per_ep = []
    for e in EPS:
        vals=[]
        for s in SEEDS:
            a=idx[(c,e,s)]["w"]; b=idx[("AUTO-NOTRACK",e,s)]["w"]
            assert len(a)==len(b)
            d=median([abs(x-y) for x,y in zip(a,b)])
            scale=median([abs(y) for y in b])
            vals.append(d/scale if scale>0 else float('nan'))
        per_ep.append(median(vals))
    drive[c]=per_ep
    print(f"  {c:10s} block-median relative |dw| = {median(per_ep)*100:8.3f} %   "
          f"(min {min(per_ep)*100:6.3f} max {max(per_ep)*100:7.3f})")
print()
print("  ratio to W-5e3 (lr ratio in brackets):")
base = median(drive["W-5e3"])
for c,lr in [("W-5e4",5e-4/5e-3),("W-MATCH",3.508993677838184e-05/5e-3),("W-5e6",5e-6/5e-3)]:
    print(f"    {c:8s} drive ratio {median(drive[c])/base:8.4f}   lr ratio {lr:8.4f}")
print(f"    {'PHI-5e3':8s} drive ratio {median(drive['PHI-5e3'])/base:8.4f}   (phase-only: w frozen -> expect 0)")
print(f"    {'AUTO':8s} drive ratio {median(drive['AUTO'])/base:8.4f}   (w_phi at shipped lr)")

print()
print("=" * 78)
print("B) how much of the shipped-lr channel asymmetry does tracking_lr alone remove?")
print("=" * 78)
def sm(metric="nRMSE"):
    b=defaultdict(list)
    for r in rows: b[(r["cell"],r["episode"])].append(r[metric])
    return {k:median(v) for k,v in b.items()}
M=sm()
asym_ship = {e: M[("W-5e3",e)]-M[("PHI-5e3",e)] for e in EPS}      # >0 = amplitude worse
asym_match= {e: M[("W-MATCH",e)]-M[("PHI-5e3",e)] for e in EPS}
lr_move   = {e: M[("W-MATCH",e)]-M[("W-5e3",e)] for e in EPS}      # exact 1-knob move
print(f"  median asymmetry @ shipped lr  (W-5e3   - PHI-5e3) = {median(list(asym_ship.values())):+.5f}  "
      f"amp-worse in {sum(1 for v in asym_ship.values() if v>0)}/16, p={sgn_p(16,sum(1 for v in asym_ship.values() if v>0)):.4f}")
print(f"  median asymmetry @ matched lr  (W-MATCH - PHI-5e3) = {median(list(asym_match.values())):+.5f}  "
      f"amp-worse in {sum(1 for v in asym_match.values() if v>0)}/16, p={sgn_p(16,sum(1 for v in asym_match.values() if v>0)):.4f}")
print(f"  the 1-knob move                (W-MATCH - W-5e3)   = {median(list(lr_move.values())):+.5f}  "
      f"better in {sum(1 for v in lr_move.values() if v<0)}/16, p={sgn_p(16,sum(1 for v in lr_move.values() if v<0)):.5f}")
frac = [(-lr_move[e])/asym_ship[e] for e in EPS if asym_ship[e] > 0]
print(f"  per-episode fraction of the shipped-lr asymmetry removed by the lr knob "
      f"(only the {len(frac)}/16 eps with positive asymmetry): median {median(frac)*100:.1f}%")
over = sum(1 for f in frac if f > 1.0)
print(f"    fraction > 100% (i.e. lr knob overshoots and reverses the asymmetry) in {over}/{len(frac)} eps")

print()
print("=" * 78)
print("C) is the surviving W-MATCH damage real?  W-MATCH vs the two floors")
print("=" * 78)
for a,b in [("W-MATCH","AUTO-NOTRACK"),("W-MATCH","W-5e6"),("W-5e6","AUTO-NOTRACK"),
            ("E-2","AUTO-NOTRACK"),("E-5","AUTO-NOTRACK"),("E-10","AUTO-NOTRACK")]:
    d=[M[(a,e)]-M[(b,e)] for e in EPS]; w=sum(1 for x in d if x<0)
    print(f"  {a:8s} - {b:13s} med={median(d):+.5f}  {a} better in {w:2d}/16  p={sgn_p(16,w):.4f}"
          f"   {'PASS(b better)' if 16-w>=13 else ('PASS(a better)' if w>=13 else 'below threshold')}")
