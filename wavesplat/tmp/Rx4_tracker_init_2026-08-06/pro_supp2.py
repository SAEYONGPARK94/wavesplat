#!/usr/bin/env python3
"""PRO reviewer: mechanism checks (m = decorr + shrink), K4REF trade, E20 dose-response."""
import json, statistics as st, math
from collections import defaultdict
from math import comb
D = "/home/vexit/wavesplat/wavesplat/tmp/Rx4_tracker_init_2026-08-06/"
rows = [json.loads(l) for l in open(D + "results.jsonl")]
nulls = {n["episode"]: n["zero_null_tracking_nRMSE_mean"] for n in json.load(open(D + "nulls_full.json"))}
CROWDED = [f"episode_BF{b}_{c}comp_seed{s}_uniform.h5" for b in (4, 5) for c in (15, 30) for s in range(4)]
IDX = defaultdict(dict)
for r in rows: IDX[(r["cell"], r["episode"])][r["seed"]] = r
SEEDS = sorted({r["seed"] for r in rows})
def epmed(c, e, m): return st.median([IDX[(c, e)][s][m] for s in SEEDS])
def sgn(k, n):
    lo = min(k, n - k); return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)
C14 = [e for e in CROWDED if epmed("AUTO", e, "nRMSE") < nulls[e]]

print("### A. identity check: m_mean == decorr_mean + shrink_mean ? (all 504 runs) ###")
w = max(abs(r["m_mean"] - r["decorr_mean"] - r["shrink_mean"]) for r in rows)
print(f"  worst |m - (decorr+shrink)| = {w:.3e}  -> the panel IS the exact MSE/var decomposition")
w2 = max(abs(r["nRMSE"] - 0.5 * math.sqrt(max(r["m_mean"], 0))) for r in rows)
print(f"  worst |nRMSE - 0.5*sqrt(m)| = {w2:.3e}   (nRMSE normalisation check)")

print()
print("### B. T-K4REF trade, crowded n=14: implied VR and rho levels ###")
for c in ["AUTO", "AUTO-NOTRACK", "T-K4REF", "T-EARLY20", "T-PHIONLY", "T-WONLY"]:
    rho = st.median([epmed(c, e, "Pearson_rho") for e in C14])
    avr = st.median([epmed(c, e, "absVR_framewise") for e in C14])
    vr = st.median([epmed(c, e, "Variance_Ratio") for e in C14])
    dec = st.median([epmed(c, e, "decorr_mean") for e in C14])
    sh = st.median([epmed(c, e, "shrink_mean") for e in C14])
    m = st.median([epmed(c, e, "m_mean") for e in C14])
    print(f"  {c:14s} rho={rho:.4f} VR(final)={vr:.4f} |VR-1|fw={avr:.4f} decorr={dec:.4f} shrink={sh:.4f} m={m:.4f}")

print()
print("### C. T-EARLY20: is its gain exactly damage-avoidance? regress delta on tracker damage ###")
dam = [epmed("AUTO", e, "nRMSE") - epmed("AUTO-NOTRACK", e, "nRMSE") for e in C14]
d20 = [epmed("T-EARLY20", e, "nRMSE") - epmed("AUTO", e, "nRMSE") for e in C14]
mx, my = st.mean(dam), st.mean(d20)
b = sum((dam[i]-mx)*(d20[i]-my) for i in range(14)) / sum((x-mx)**2 for x in dam)
print(f"  OLS slope d(E20-AUTO) on d(AUTO-FLOOR) = {b:+.3f}  (a slope of -1 == E20 recovers the whole damage)")
print(f"  intercept = {my - b*mx:+.5f}   (an intercept != 0 would be a gain NOT explained by damage-avoidance)")

print()
print("### D. E20 fraction of episode: rx4_track_frames=20 of 160 frames = 12.5% ###")
print("     -> the tracker runs for 12.5% of the record and is frozen for 87.5%")

print()
print("### E. what does the shipped tracker ever BUY? cells where AUTO < AUTO-NOTRACK ###")
for e in CROWDED:
    a, f = epmed("AUTO", e, "nRMSE"), epmed("AUTO-NOTRACK", e, "nRMSE")
    if a < f: print(f"  {e:40s} AUTO={a:.4f} < FLOOR={f:.4f}  ({a-f:+.4f})")
print("  (blank above = the shipped tracker never wins on nRMSE in crowded)")

print()
print("### F. block-S: EVERY Phase-2 perturbation points the same way (is this cell-specific?) ###")
SP = [e for e in sorted({r['episode'] for r in rows}) if e not in CROWDED]
for c in ["AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"]:
    ds = [epmed(c, e, "nRMSE") - epmed("AUTO", e, "nRMSE") for e in SP]
    print(f"  {c:14s} {sum(1 for d in ds if d<0)}/8 better than AUTO, medΔ={st.median(ds):+.5f}")
print("  -> if all five point the same way, the sparse signal is 'shipped Phase 2 is slightly net-negative',")
print("     not 'this particular knob is good'.")

print()
print("### G. K4REF crowded: is it better than the FLOOR on the panel? ###")
for m, hi in [("nRMSE", 0), ("Pearson_rho", 1), ("absVR_framewise", 0), ("m_mean", 0), ("decorr_mean", 0), ("shrink_mean", 0)]:
    ds = [epmed("T-K4REF", e, m) - epmed("AUTO-NOTRACK", e, m) for e in C14]
    k = sum(1 for d in ds if (d > 0 if hi else d < 0))
    print(f"  {m:20s} medΔ(K4REF-FLOOR)={st.median(ds):+.4f}  K4REF better {k}/14 p={sgn(k,14):.4f}")
