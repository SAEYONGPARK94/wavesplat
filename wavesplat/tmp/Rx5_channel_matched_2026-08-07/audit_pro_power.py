#!/usr/bin/env python3
"""PRO part 2: floor-recovery, power/MDE, multiplicity, E-sweep aggregation forensics,
and the Rx-4 T-WONLY reproduction check. Read-only."""
import json, os, math, random
from statistics import median
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
nulls = {d["episode"]: d for d in json.load(open(
    "/home/vexit/wavesplat/wavesplat/tmp/K6_run_2026-08-07/nulls_k6.json"))}
EPS = sorted(set(r["episode"] for r in rows))

def seedmed(metric, rr=rows):
    b = defaultdict(list)
    for r in rr:
        if r.get(metric) is not None:
            b[(r["cell"], r["episode"])].append(r[metric])
    return {k: median(v) for k, v in b.items()}
M = seedmed("nRMSE")
THRESH = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}

def C(n, k): return math.comb(n, k)
def sgn_p(n, wins):
    k = max(wins, n - wins)
    return min(1.0, 2.0 * sum(C(n, i) for i in range(k, n + 1)) / 2 ** n)

def paired(a, b, eps=EPS): return [M[(a, e)] - M[(b, e)] for e in eps]

print("=" * 78); print("1) FLOOR RECOVERY  R = (cell - AUTO)/(NOTRACK - AUTO), per episode")
print("=" * 78)
gap = {e: M[("AUTO-NOTRACK", e)] - M[("AUTO", e)] for e in EPS}
neg = [e for e in EPS if gap[e] < 0]
print(f"  gap<0 (floor better) in {len(neg)}/16 episodes; the 2 exceptions:")
for e in EPS:
    if gap[e] >= 0: print(f"    {e}  gap={gap[e]:+.5f}")
print()
print(f"{'cell':14s} {'medR(all16)':>12s} {'medR(gap<0 14)':>15s} {'medDelta vs AUTO':>17s}")
for c in ["W-5e3", "W-5e4", "W-MATCH", "W-5e6", "PHI-5e3",
          "E-2", "E-5", "E-10", "E-20", "E-40", "E-80"]:
    obs = {e: M[(c, e)] - M[("AUTO", e)] for e in EPS}
    r16 = median([obs[e] / gap[e] for e in EPS])
    r14 = median([obs[e] / gap[e] for e in neg])
    print(f"{c:14s} {r16*100:11.1f}% {r14*100:14.1f}% {median(list(obs.values())):+17.5f}")

print()
print("=" * 78); print("2) POWER / MDE for the frozen bar (n=16, need >=13 wins)")
print("=" * 78)
# win-probability needed for 80% power
def power(n, thr, p):
    return sum(C(n, i) * p**i * (1-p)**(n-i) for i in range(thr, n+1))
lo, hi = 0.5, 1.0
for _ in range(200):
    mid = (lo+hi)/2
    if power(16, 13, mid) < 0.80: lo = mid
    else: hi = mid
p80 = (lo+hi)/2
print(f"  win-probability needed for 80% power at n=16, thr=13 : p* = {p80:.4f}")
for pp in [0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]:
    print(f"    p={pp:.2f} -> power {power(16,13,pp)*100:5.1f}%")

def mde_shift(deltas, thr=13, n=16, target=0.80, B=4000, seed=0):
    """Smallest constant shift m such that shifting the observed (centred) deltas by -m
    yields >=target power under a paired bootstrap over episodes."""
    rnd = random.Random(seed)
    c = median(deltas)
    cent = [d - c for d in deltas]
    lo, hi = 0.0, max(abs(x) for x in cent)*4 + 1e-9
    for _ in range(40):
        m = (lo+hi)/2
        hit = 0
        for _b in range(B):
            samp = [cent[rnd.randrange(n)] - m for _ in range(n)]
            if sum(1 for x in samp if x < 0) >= thr: hit += 1
        if hit/B < target: lo = m
        else: hi = m
    return (lo+hi)/2

dA = paired("W-MATCH", "PHI-5e3")
mdeA = mde_shift(dA)
print(f"\n  PRIMARY-A  observed median = {median(dA):+.5f}, wins(phase worse) = "
      f"{sum(1 for d in dA if d<0)}/16, p = {sgn_p(16, sum(1 for d in dA if d<0)):.4f}")
print(f"  PRIMARY-A  MDE (80% power, bootstrap over episodes) = {mdeA:.5f} nRMSE")
print(f"  |observed| / MDE = {abs(median(dA))/mdeA:.2f}x")
headroom = abs(median(paired("AUTO-NOTRACK", "AUTO")))
print(f"  tracker headroom (|median(NOTRACK-AUTO)|) = {headroom:.5f};  "
      f"MDE/headroom = {mdeA/headroom*100:.1f}%")

dF = paired("W-MATCH", "AUTO-NOTRACK")
mdeF = mde_shift(dF)
print(f"\n  W-MATCH vs FLOOR observed median = {median(dF):+.5f}, "
      f"wins(W-MATCH better) = {sum(1 for d in dF if d<0)}/16, p={sgn_p(16,sum(1 for d in dF if d<0)):.4f}")
print(f"  MDE = {mdeF:.5f};  |observed|/MDE = {abs(median(dF))/mdeF:.2f}x; "
      f"MDE/headroom = {mdeF/headroom*100:.1f}%")

# cluster bootstrap CI (resample episodes) for medians
def boot_ci(deltas, B=20000, seed=1):
    rnd = random.Random(seed); n=len(deltas); out=[]
    for _ in range(B):
        out.append(median([deltas[rnd.randrange(n)] for _ in range(n)]))
    out.sort(); return out[int(0.025*B)], out[int(0.975*B)]
for name, d in [("W-MATCH - PHI-5e3", dA), ("W-MATCH - AUTO", paired("W-MATCH","AUTO")),
                ("W-MATCH - NOTRACK", dF), ("NOTRACK - AUTO", paired("AUTO-NOTRACK","AUTO")),
                ("W-5e6 - W-MATCH", paired("W-5e6","W-MATCH"))]:
    lo_, hi_ = boot_ci(d)
    print(f"  95% episode-bootstrap CI  {name:20s} median {median(d):+.5f}  [{lo_:+.5f}, {hi_:+.5f}]")

print()
print("=" * 78); print("3) MULTIPLICITY (Benjamini-Hochberg over the contrasts actually run)")
print("=" * 78)
tests = []
for c in ["W-5e4","W-MATCH","W-5e6"]:
    d = paired(c,"W-5e3"); tests.append((f"{c}-W-5e3", sgn_p(16,sum(1 for x in d if x<0))))
for c in ["W-5e3","W-5e4","W-MATCH","W-5e6","PHI-5e3","AUTO-NOTRACK",
          "E-2","E-5","E-10","E-20","E-40","E-80"]:
    d = paired(c,"AUTO"); tests.append((f"{c}-AUTO", sgn_p(16,sum(1 for x in d if x<0))))
for c in ["W-5e3","W-5e4","W-MATCH","W-5e6","PHI-5e3","E-2","E-5","E-10","E-20","E-40","E-80"]:
    d = paired(c,"AUTO-NOTRACK"); tests.append((f"{c}-NOTRACK", sgn_p(16,sum(1 for x in d if x<0))))
d = paired("W-MATCH","PHI-5e3"); tests.append(("PRIMARY-A", sgn_p(16,sum(1 for x in d if x<0))))
tests.sort(key=lambda t: t[1])
m = len(tests)
print(f"  m = {m} tests")
for i,(name,p) in enumerate(tests,1):
    print(f"    {i:2d} {name:22s} p={p:.5f}  BH crit={0.05*i/m:.5f}  "
          f"{'survives' if p <= 0.05*i/m else ''}   Bonf={0.05/m:.5f} "
          f"{'survives-B' if p <= 0.05/m else ''}")

print()
print("=" * 78); print("4) E-SWEEP AGGREGATION FORENSICS")
print("=" * 78)
print("  ratio-of-medians vs median-of-ratios (the reported 125.9% peak):")
print(f"{'E':>4s} {'predR':>7s} {'ratio-of-med':>13s} {'med-of-ratio(16)':>17s} {'med-of-ratio(14)':>17s}")
for E in (2,5,10,20,40,80):
    c=f"E-{E}"; obs={e:M[(c,e)]-M[("AUTO",e)] for e in EPS}
    rom = median(list(obs.values()))/median([gap[e] for e in EPS])
    mor16 = median([obs[e]/gap[e] for e in EPS])
    mor14 = median([obs[e]/gap[e] for e in neg])
    print(f"{E:4d} {(1-E/160)*100:6.1f}% {rom*100:12.1f}% {mor16*100:16.1f}% {mor14*100:16.1f}%")

print()
print("  paired test observed vs uniform-damage prediction (PREREG 5 direction restated):")
print("  dif = obs - pred  (in nRMSE units).  dif>0  <=>  observed R LOWER than predicted")
print("                                              <=>  FRONT-LOADED per PREREG 5 main clause")
print(f"{'E':>4s} {'n':>3s} {'dif>0':>6s} {'p':>8s} {'med(dif)':>10s} {'verdict':>20s}")
for E in (2,5,10,20,40,80):
    c=f"E-{E}"
    dif=[(M[(c,e)]-M[("AUTO",e)]) - (1-E/160)*gap[e] for e in EPS]
    kept=[x for x in dif if x!=0.0]; n=len(kept); up=sum(1 for x in kept if x>0)
    p=sgn_p(n,up); thr=THRESH[n]
    v = "FRONT-LOADED" if up>=thr else ("BACK-LOADED" if n-up>=thr else "not distinguished")
    print(f"{E:4d} {n:3d} {up:4d}/{n:2d} {p:8.4f} {median(dif):+10.5f} {v:>20s}")

print()
print("  same test restricted to the 14 episodes with gap<0:")
for E in (2,5,10,20,40,80):
    c=f"E-{E}"
    dif=[(M[(c,e)]-M[("AUTO",e)]) - (1-E/160)*gap[e] for e in neg]
    n=len(dif); up=sum(1 for x in dif if x>0); p=sgn_p(n,up); thr=THRESH[n]
    v = "FRONT-LOADED" if up>=thr else ("BACK-LOADED" if n-up>=thr else "not distinguished")
    print(f"    E={E:3d}  {up:2d}/{n} dif>0  p={p:.4f}  med={median(dif):+.5f}  {v}")

print()
print("  is R(E) non-monotone?  per-episode paired R differences (E=10 as the claimed peak):")
Rep={E:{e:(M[(f'E-{E}',e)]-M[("AUTO",e)])/gap[e] for e in EPS} for E in (2,5,10,20,40,80)}
for a,b in [(2,5),(5,10),(2,10),(10,20),(10,40),(10,80)]:
    d=[Rep[b][e]-Rep[a][e] for e in EPS]
    up=sum(1 for x in d if x>0); print(f"    R({b})-R({a}): med={median(d):+.4f} up={up}/16 p={sgn_p(16,up):.4f}")
print("  raw-nRMSE version of the same question (no ratio):")
for a,b in [("E-2","E-5"),("E-5","E-10"),("E-2","E-10"),("E-10","E-20"),("E-10","E-40")]:
    d=paired(b,a); w=sum(1 for x in d if x<0)
    print(f"    {b}-{a}: med={median(d):+.5f} better={w}/16 p={sgn_p(16,w):.4f}")

print()
print("=" * 78); print("5) rx1-trap check: m_mean levels (m>1 = worse than flat sea)")
print("=" * 78)
MM = seedmed("m_mean")
for c in ["AUTO","AUTO-NOTRACK","W-5e3","W-5e4","W-MATCH","W-5e6","PHI-5e3","E-10","E-80"]:
    v=[MM[(c,e)] for e in EPS]
    print(f"  {c:14s} block-median m_mean={median(v):.4f}  m>1 in {sum(1 for x in v if x>1)}/16 eps")

print()
print("=" * 78); print("6) null-screen at CELL level (PREREG 4.6)")
print("=" * 78)
for c in ["W-5e3","W-5e4","W-MATCH","W-5e6","PHI-5e3","E-2","E-5","E-10","E-20","E-40","E-80",
          "AUTO","AUTO-NOTRACK"]:
    fails=[e for e in EPS if M[(c,e)] >= nulls[e]["zero_null_tracking_nRMSE_mean"]]
    print(f"  {c:14s} beats null {16-len(fails):2d}/16   fails: {[f.split('_uniform')[0] for f in fails]}")
