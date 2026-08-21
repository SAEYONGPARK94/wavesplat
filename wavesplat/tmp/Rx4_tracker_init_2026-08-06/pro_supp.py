#!/usr/bin/env python3
"""PRO reviewer: supplementary robustness / power / asset-hunting. Read-only."""
import json, statistics as st, math
from collections import defaultdict, Counter
from math import comb

D = "/home/vexit/wavesplat/wavesplat/tmp/Rx4_tracker_init_2026-08-06/"
rows = [json.loads(l) for l in open(D + "results.jsonl")]
nulls = {n["episode"]: n["zero_null_tracking_nRMSE_mean"] for n in json.load(open(D + "nulls_full.json"))}
CELLS = ["AUTO", "AUTO-DUP", "AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"]
CROWDED = [f"episode_BF{b}_{c}comp_seed{s}_uniform.h5" for b in (4, 5) for c in (15, 30) for s in range(4)]
ALL = sorted({r["episode"] for r in rows}); SPARSE = [e for e in ALL if e not in CROWDED]
IDX = defaultdict(dict)
for r in rows: IDX[(r["cell"], r["episode"])][r["seed"]] = r
SEEDS = sorted({r["seed"] for r in rows})
HIGH = {"Pearson_rho", "Skill_Score", "Directional_R2", "Variance_Ratio"}

def v(r, m): return r["nRMSE"] - r["init_nrmse"] if m == "final_minus_init" else r[m]
def epmed(c, e, m): return st.median([v(IDX[(c, e)][s], m) for s in SEEDS])
def sgn(k, n):
    lo = min(k, n - k); return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)
C14 = [e for e in CROWDED if epmed("AUTO", e, "nRMSE") < nulls[e]]

print("### 1. PER-SEED ROBUSTNESS of every cell vs AUTO (nRMSE, crowded n=14, no median-of-3) ###")
for c in CELLS[2:]:
    line = f"  {c:14s}"
    for s in SEEDS:
        ds = [IDX[(c, e)][s]["nRMSE"] - IDX[("AUTO", e)][s]["nRMSE"] for e in C14]
        k = sum(1 for d in ds if d < 0)
        line += f" | seed{s}: med={st.median(ds):+.4f} {k:2d}/14 p={sgn(k,14):.3f}"
    print(line)

print()
print("### 2. ALTERNATIVE AGGREGATIONS (sensitivity of the T-EARLY20 result) ###")
for c in ["AUTO-NOTRACK", "T-EARLY20", "T-PHIONLY", "T-K4REF", "T-WONLY"]:
    d_med = [epmed(c, e, "nRMSE") - epmed("AUTO", e, "nRMSE") for e in C14]
    d_mean = [st.mean([IDX[(c, e)][s]["nRMSE"] for s in SEEDS]) - st.mean([IDX[("AUTO", e)][s]["nRMSE"] for s in SEEDS]) for e in C14]
    pooled = [IDX[(c, e)][s]["nRMSE"] - IDX[("AUTO", e)][s]["nRMSE"] for e in C14 for s in SEEDS]
    print(f"  {c:14s} prereg(medseed,medΔ)={st.median(d_med):+.4f} {sum(1 for x in d_med if x<0)}/14 | "
          f"mean-seed medΔ={st.median(d_mean):+.4f} {sum(1 for x in d_mean if x<0)}/14 | "
          f"mean-of-Δ={st.mean(d_med):+.4f} | pooled-run {sum(1 for x in pooled if x<0)}/42 p={sgn(sum(1 for x in pooled if x<0),42):.4f}")

print()
print("### 3. POWER / MDE (paired, n=14). MDE at alpha=.05 two-sided, power .80, using observed sd(Δ) ###")
for c in ["AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"]:
    ds = [epmed(c, e, "nRMSE") - epmed("AUTO", e, "nRMSE") for e in C14]
    sd = st.stdev(ds); mde = 3.03 * sd / math.sqrt(14)
    print(f"  {c:14s} obs medΔ={st.median(ds):+.4f} meanΔ={st.mean(ds):+.4f} sd={sd:.4f} "
          f"MDE~{mde:.4f}  |obs|/MDE={abs(st.mean(ds))/mde:5.2f}  floor-sized(-0.0205)/MDE={0.0205/mde:5.2f}")

print()
print("### 4. T-EARLY20 vs AUTO-NOTRACK per episode (does it ever beat the floor?) ###")
for e in C14:
    a, f_, x = epmed("AUTO", e, "nRMSE"), epmed("AUTO-NOTRACK", e, "nRMSE"), epmed("T-EARLY20", e, "nRMSE")
    print(f"  {e:40s} AUTO={a:.4f} FLOOR={f_:.4f} E20={x:.4f}  E20-FLOOR={x-f_:+.4f} {'BEATS FLOOR' if x<f_ else ''}")

print()
print("### 5. Does the intervention help most where AUTO is worst? (Spearman-ish rank check) ###")
def spearman(x, y):
    def rk(a):
        o = sorted(range(len(a)), key=lambda i: a[i]); r = [0]*len(a)
        for j, i in enumerate(o): r[i] = j
        return r
    rx, ry = rk(x), rk(y); n = len(x)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    den = math.sqrt(sum((rx[i]-mx)**2 for i in range(n))*sum((ry[i]-my)**2 for i in range(n)))
    return num/den
lev = [epmed("AUTO", e, "nRMSE") for e in CROWDED]
gap = [epmed("AUTO", e, "nRMSE") - epmed("AUTO-NOTRACK", e, "nRMSE") for e in CROWDED]
for c in ["T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF", "AUTO-NOTRACK"]:
    d = [epmed(c, e, "nRMSE") - epmed("AUTO", e, "nRMSE") for e in CROWDED]
    print(f"  {c:14s} rho(AUTO level, Δ)={spearman(lev,d):+.3f}   rho(tracker damage AUTO-FLOOR, Δ)={spearman(gap,d):+.3f}   (n=16, all crowded)")

print()
print("### 6. BLOCK S detail, T-K4REF and others (levels are tiny -> report relative too) ###")
for c in CELLS[2:]:
    ds = [epmed(c, e, "nRMSE") - epmed("AUTO", e, "nRMSE") for e in SPARSE]
    rel = [d / epmed("AUTO", e, "nRMSE") for d, e in zip(ds, SPARSE)]
    k = sum(1 for d in ds if d < 0)
    print(f"  {c:14s} medΔ={st.median(ds):+.5f} medΔrel={st.median(rel)*100:+6.1f}%  {k}/8 p={sgn(k,8):.4f}  "
          f"per-ep: {['%+.4f'%d for d in ds]}")

print()
print("### 7. FINAL-INIT: which intervention actually stops Phase-2 from wrecking the init? ###")
print("  (B3 mechanism: crowded Phase-2 makes the field WORSE than the initialisation)")
for c in CELLS:
    fi = [epmed(c, e, "final_minus_init") for e in C14]
    worse = sum(1 for x in fi if x > 0)
    print(f"  {c:14s} med(final-init)={st.median(fi):+.4f}  episodes where Phase2 HURTS init: {worse}/14")

print()
print("### 8. PANEL CONSISTENCY SCORE (how many of 6 panel metrics does each cell win at the n=14 bar 12/14?) ###")
PANEL = ["nRMSE", "Pearson_rho", "absVR_framewise", "m_mean", "decorr_mean", "shrink_mean"]
for c in CELLS[2:]:
    good = []
    for m in PANEL:
        ds = [epmed(c, e, m) - epmed("AUTO", e, m) for e in C14]
        k = sum(1 for d in ds if (d > 0 if m in HIGH else d < 0))
        good.append(f"{m}:{k}/14" + ("*" if k >= 12 else ""))
    nb = sum(1 for g in good if g.endswith("*"))
    print(f"  {c:14s} at-bar {nb}/6   " + "  ".join(good))

print()
print("### 9. D1 head-to-head + w/phi asymmetry ###")
ds = [epmed("T-PHIONLY", e, "nRMSE") - epmed("T-WONLY", e, "nRMSE") for e in C14]
k = sum(1 for d in ds if d < 0)
print(f"  T-PHIONLY vs T-WONLY (nRMSE, C14): medΔ={st.median(ds):+.4f} PHIONLY wins {k}/14 p={sgn(k,14):.4f}")
ds16 = [epmed("T-PHIONLY", e, "nRMSE") - epmed("T-WONLY", e, "nRMSE") for e in CROWDED]
k16 = sum(1 for d in ds16 if d < 0)
print(f"  same on all 16                : medΔ={st.median(ds16):+.4f} PHIONLY wins {k16}/16 p={sgn(k16,16):.4f}")
dsS = [epmed("T-PHIONLY", e, "nRMSE") - epmed("T-WONLY", e, "nRMSE") for e in SPARSE]
kS = sum(1 for d in dsS if d < 0)
print(f"  block S                      : medΔ={st.median(dsS):+.5f} PHIONLY wins {kS}/8 p={sgn(kS,8):.4f}")
print("  NOTE prereg D1 rule is |medΔ vs AUTO| comparison -> PHIONLY 0.0013 < WONLY 0.0106 -> REFUTED (literal)")

print()
print("### 10. ROUTER composition per episode (AUTO is a router) ###")
for e in CROWDED:
    sel = [IDX[("AUTO", e)][s]["auto_selected"] for s in SEEDS]
    if len(set(sel)) > 1 or sel[0] != "learned":
        print(f"  {e:40s} {sel}")
print("  (only lines printed are episodes where any seed routed to something other than 'learned')")

print()
print("### 11. wall/elapsed record only (NOT a performance claim) ###")
for c in CELLS:
    ms = st.median([IDX[(c, e)][s]["elapsed_ms"] for e in C14 for s in SEEDS])
    print(f"  {c:14s} median elapsed_ms/frame = {ms:.2f}")
