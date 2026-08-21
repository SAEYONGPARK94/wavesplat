#!/usr/bin/env python3
"""PRO part 5: sample-size needed, displacement-budget mechanism, and the
magnitude-sensitive supplementary (NOT the frozen test -- reported as such)."""
import json, os, math, itertools
from statistics import median
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
EPS = sorted(set(r["episode"] for r in rows))
b = defaultdict(list)
for r in rows: b[(r["cell"], r["episode"])].append(r["nRMSE"])
M = {k: median(v) for k, v in b.items()}

def C(n,k): return math.comb(n,k)
def crit(n, alpha=0.05):
    """smallest k such that two-sided exact sign test p<=alpha"""
    for k in range(n, n//2, -1):
        p = min(1.0, 2*sum(C(n,i) for i in range(k,n+1))/2**n)
        if p > alpha: return k+1
    return n
def power(n,k,p): return sum(C(n,i)*p**i*(1-p)**(n-i) for i in range(k,n+1))

print("=" * 78)
print("A) sample size needed for the observed PRIMARY-A win-rate (12/16 -> p_hat = 0.75)")
print("=" * 78)
print(f"{'n':>4s} {'crit':>5s} {'power@p=0.75':>13s} {'power@p=0.70':>13s} {'power@p=0.80':>13s}")
for n in [16,20,24,28,32,36,40,48,56,64,80]:
    k=crit(n)
    print(f"{n:4d} {k:5d} {power(n,k,0.75)*100:12.1f}% {power(n,k,0.70)*100:12.1f}% {power(n,k,0.80)*100:12.1f}%")
for target,pw in [(0.75,0.80),(0.75,0.90)]:
    n=16
    while n<400 and power(n,crit(n),target)<pw: n+=1
    print(f"  -> n needed for {pw*100:.0f}% power at true win-rate {target}: n = {n}")

print()
print("=" * 78)
print("B) displacement-budget mechanism  (steps x lr, relative to the parameter's own scale)")
print("=" * 78)
FRAMES=160; STEPS=10   # tracking_steps default; budget = FRAMES*STEPS*lr
WSCALE=0.044095; PHISCALE=2*math.pi
cells=[("W-5e3",0.005,WSCALE),("W-5e4",5e-4,WSCALE),
       ("W-MATCH",3.508993677838184e-05,WSCALE),("W-5e6",5e-6,WSCALE),
       ("PHI-5e3",0.005,PHISCALE)]
print(f"{'cell':10s} {'lr':>12s} {'per-step rel':>13s} {'budget/scale':>13s} {'measured |dw| rel':>18s} {'nRMSE':>8s}")
drv={"W-5e3":0.96340,"W-5e4":0.91814,"W-MATCH":0.37426,"W-5e6":0.05824,"PHI-5e3":0.0}
for c,lr,sc in cells:
    print(f"{c:10s} {lr:12.3e} {lr/sc*100:12.3f}% {FRAMES*STEPS*lr/sc:13.3f} "
          f"{drv[c]*100:17.2f}% {median([M[(c,e)] for e in EPS]):8.4f}")
print("  note: W-MATCH and PHI-5e3 sit at the SAME budget/scale (1.273) by construction.")

print()
print("=" * 78)
print("C) magnitude-sensitive supplementary -- NOT the frozen test (PREREG 4.1 froze the sign test)")
print("   reported only to characterise the magnitude structure the sign test discards")
print("=" * 78)
d=[M[("W-MATCH",e)]-M[("PHI-5e3",e)] for e in EPS]
pos=[x for x in d if x>0]; negs=[x for x in d if x<0]
print(f"  favorable (phase worse) n={len(negs)}  largest {min(negs):+.5f}  median {median(negs):+.5f}")
print(f"  adverse   (amp   worse) n={len(pos)}  largest {max(pos):+.5f}  median {median(pos):+.5f}")
print(f"  sum favorable {sum(negs):+.5f}  sum adverse {sum(pos):+.5f}  net {sum(d):+.5f}  mean {sum(d)/16:+.5f}")
# exact Wilcoxon signed rank, two-sided (small n exact by enumeration of sign flips)
absd=sorted((abs(x),i) for i,x in enumerate(d))
rank={}
i=0
while i<len(absd):
    j=i
    while j+1<len(absd) and absd[j+1][0]==absd[i][0]: j+=1
    r=(i+j)/2+1
    for k in range(i,j+1): rank[absd[k][1]]=r
    i=j+1
Wplus=sum(rank[i] for i,x in enumerate(d) if x>0)
n=len(d); tot=n*(n+1)/2
# exact null distribution by DP
dp=[1]+[0]*int(tot)
for i in range(1,n+1):
    for s in range(int(tot),i-1,-1): dp[s]+=dp[s-i]
tot_ways=2**n
cum_le=sum(dp[:int(Wplus)+1]); cum_ge=sum(dp[int(Wplus):])
pw=min(1.0,2*min(cum_le,cum_ge)/tot_ways)
print(f"  exact Wilcoxon signed-rank W+ = {Wplus:.1f} / {tot:.0f}  two-sided p = {pw:.5f}   "
      f"[SUPPLEMENTARY ONLY -- protocol froze the sign test]")
# permutation test on the median (sign-flip)
import random
rnd=random.Random(7); obs=median(d); cnt=0; B=200000
for _ in range(B):
    s=[x if rnd.random()<0.5 else -x for x in d]
    if abs(median(s))>=abs(obs): cnt+=1
print(f"  sign-flip permutation test on the median: p = {(cnt+1)/(B+1):.5f}   [SUPPLEMENTARY ONLY]")

print()
print("=" * 78)
print("D) leave-one-out stability of PRIMARY-A (how fragile is the 12/16?)")
print("=" * 78)
for drop in range(16):
    dd=[x for i,x in enumerate(d) if i!=drop]
    w=sum(1 for x in dd if x<0)
    print(f"  drop {EPS[drop].split('_uniform')[0]:26s} -> {w:2d}/15 wins, med {median(dd):+.5f}, "
          f"{'PASS(thr 12)' if w>=12 else 'below'}")
