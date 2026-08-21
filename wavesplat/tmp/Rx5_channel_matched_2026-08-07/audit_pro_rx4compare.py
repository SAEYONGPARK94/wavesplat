#!/usr/bin/env python3
"""PRO part 3: does Rx-5 'W-5e3' reproduce Rx-4 'T-WONLY'?  Read-only, no writes to Rx-4."""
import json, os, math
from statistics import median
from collections import defaultdict

R4 = "/home/vexit/wavesplat/wavesplat/tmp/Rx4_tracker_init_2026-08-06"
R5 = os.path.dirname(os.path.abspath(__file__))

def load(p):
    return [json.loads(l) for l in open(p)]

def seedmed(rows, metric="nRMSE"):
    b = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            b[(r["cell"], r["episode"])].append(r[metric])
    return {k: median(v) for k, v in b.items()}

def C(n, k): return math.comb(n, k)
def sgn_p(n, w):
    k = max(w, n - w)
    return min(1.0, 2.0 * sum(C(n, i) for i in range(k, n + 1)) / 2 ** n)

r4 = load(os.path.join(R4, "results.jsonl"))
r5 = load(os.path.join(R5, "results.jsonl"))
M4 = seedmed(r4); M5 = seedmed(r5)

CROWD = [f"episode_BF{b}_{c}comp_seed{s}_uniform" for b in (4, 5) for c in (15, 30) for s in range(4)]
E4 = [e + ".h5" for e in CROWD]
E5 = [e + "_ra_rf_ds.h5" for e in CROWD]
assert all((("AUTO", e) in M4) for e in E4)
assert all((("AUTO", e) in M5) for e in E5)

def rep(M, eps, a, b, tag):
    d = [M[(a, e)] - M[(b, e)] for e in eps]
    w = sum(1 for x in d if x < 0); n = len(d)
    print(f"  {tag:36s} med={median(d):+.5f}  better={w:2d}/{n}  p={sgn_p(n,w):.4f}")
    return median(d)

print("=" * 78)
print("Rx-4 block C (16 crowded, seeds 1524-26, OLD generator)")
print("=" * 78)
print(f"  levels: AUTO {median([M4[('AUTO',e)] for e in E4]):.4f}  "
      f"NOTRACK {median([M4[('AUTO-NOTRACK',e)] for e in E4]):.4f}  "
      f"T-WONLY {median([M4[('T-WONLY',e)] for e in E4]):.4f}  "
      f"T-PHIONLY {median([M4[('T-PHIONLY',e)] for e in E4]):.4f}  "
      f"T-EARLY20 {median([M4[('T-EARLY20',e)] for e in E4]):.4f}")
g4 = rep(M4, E4, "AUTO-NOTRACK", "AUTO", "NOTRACK - AUTO (floor gain)")
rep(M4, E4, "T-WONLY", "AUTO", "T-WONLY - AUTO")
rep(M4, E4, "T-PHIONLY", "AUTO", "T-PHIONLY - AUTO")
rep(M4, E4, "T-EARLY20", "AUTO", "T-EARLY20 - AUTO")
rep(M4, E4, "T-WONLY", "T-PHIONLY", "T-WONLY - T-PHIONLY (signed D1)")
gap4 = {e: M4[("AUTO-NOTRACK", e)] - M4[("AUTO", e)] for e in E4}
for c in ["T-WONLY", "T-PHIONLY", "T-EARLY20"]:
    r = median([(M4[(c, e)] - M4[("AUTO", e)]) / gap4[e] for e in E4])
    print(f"  floor-recovery R({c}) = {r*100:+.1f}%")

print()
print("=" * 78)
print("Rx-5 block (16 crowded, seeds 1530-32, NEW independent generator _ra_rf_ds)")
print("=" * 78)
print(f"  levels: AUTO {median([M5[('AUTO',e)] for e in E5]):.4f}  "
      f"NOTRACK {median([M5[('AUTO-NOTRACK',e)] for e in E5]):.4f}  "
      f"W-5e3 {median([M5[('W-5e3',e)] for e in E5]):.4f}  "
      f"PHI-5e3 {median([M5[('PHI-5e3',e)] for e in E5]):.4f}  "
      f"E-20 {median([M5[('E-20',e)] for e in E5]):.4f}")
rep(M5, E5, "AUTO-NOTRACK", "AUTO", "NOTRACK - AUTO (floor gain)")
rep(M5, E5, "W-5e3", "AUTO", "W-5e3   - AUTO   (== T-WONLY cfg)")
rep(M5, E5, "PHI-5e3", "AUTO", "PHI-5e3 - AUTO   (== T-PHIONLY cfg)")
rep(M5, E5, "E-20", "AUTO", "E-20    - AUTO   (== T-EARLY20 cfg)")
rep(M5, E5, "W-5e3", "PHI-5e3", "W-5e3 - PHI-5e3 (signed D1, shipped lr)")
gap5 = {e: M5[("AUTO-NOTRACK", e)] - M5[("AUTO", e)] for e in E5}
for c in ["W-5e3", "PHI-5e3", "E-20"]:
    r = median([(M5[(c, e)] - M5[("AUTO", e)]) / gap5[e] for e in E5])
    print(f"  floor-recovery R({c}) = {r*100:+.1f}%")

print()
print("=" * 78)
print("REPRODUCTION VERDICT TABLE (sign + rank, NOT magnitude; different generators)")
print("=" * 78)
print(f"{'quantity':34s} {'Rx-4':>12s} {'Rx-5':>12s} {'sign match':>11s}")
pairs = [("floor gain (NOTRACK-AUTO)", ("AUTO-NOTRACK", "AUTO"), ("AUTO-NOTRACK", "AUTO")),
         ("w-only - AUTO", ("T-WONLY", "AUTO"), ("W-5e3", "AUTO")),
         ("phi-only - AUTO", ("T-PHIONLY", "AUTO"), ("PHI-5e3", "AUTO")),
         ("early20 - AUTO", ("T-EARLY20", "AUTO"), ("E-20", "AUTO")),
         ("w-only - phi-only", ("T-WONLY", "T-PHIONLY"), ("W-5e3", "PHI-5e3"))]
for name, p4, p5 in pairs:
    m4 = median([M4[(p4[0], e)] - M4[(p4[1], e)] for e in E4])
    m5 = median([M5[(p5[0], e)] - M5[(p5[1], e)] for e in E5])
    print(f"{name:34s} {m4:+12.5f} {m5:+12.5f} {'YES' if (m4>0)==(m5>0) else 'NO':>11s}")
