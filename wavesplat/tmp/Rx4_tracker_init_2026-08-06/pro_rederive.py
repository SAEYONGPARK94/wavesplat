#!/usr/bin/env python3
"""PRO reviewer independent re-derivation of Rx-4 headline. Written from scratch.
Read-only on results.jsonl / nulls_full.json.
Conventions: PREREG.md Section 4 (frozen).
  1. seed -> episode : median of 3
  2. level           : median of episode medians (medOfMed)
  3. effect          : median over episodes of paired delta
  4. direction       : nRMSE/absVR/m/decorr/shrink : lower better ; rho : higher better
  5. sign test, two-sided, ties counted UNFAVOURABLY (not a win, stays in n)
"""
import json, math, statistics as st
from collections import defaultdict
from math import comb

D = "/home/vexit/wavesplat/wavesplat/tmp/Rx4_tracker_init_2026-08-06/"

rows = [json.loads(l) for l in open(D + "results.jsonl")]
nulls = {n["episode"]: n["zero_null_tracking_nRMSE_mean"] for n in json.load(open(D + "nulls_full.json"))}

CELLS = ["AUTO", "AUTO-DUP", "AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"]
CROWDED = [f"episode_BF{b}_{c}comp_seed{s}_uniform.h5"
           for b in (4, 5) for c in (15, 30) for s in range(4)]
ALL_EPS = sorted({r["episode"] for r in rows})
SPARSE = [e for e in ALL_EPS if e not in CROWDED]
assert len(CROWDED) == 16 and len(SPARSE) == 8, (len(CROWDED), len(SPARSE))

MET_LOWER = ["nRMSE", "absVR_framewise", "m_mean", "decorr_mean", "shrink_mean",
             "avg_RMSE_unobserved", "Hs_error", "init_nrmse", "final_minus_init"]
MET_HIGHER = ["Pearson_rho", "Skill_Score", "Directional_R2", "Variance_Ratio"]


def val(r, m):
    if m == "final_minus_init":
        return r["nRMSE"] - r["init_nrmse"]
    return r[m]


# index: (cell, episode) -> {seed: row}
IDX = defaultdict(dict)
for r in rows:
    IDX[(r["cell"], r["episode"])][r["seed"]] = r
SEEDS = sorted({r["seed"] for r in rows})


def epmed(cell, ep, m):
    d = IDX[(cell, ep)]
    assert len(d) == 3, (cell, ep, len(d))
    return st.median([val(d[s], m) for s in SEEDS])


def signtest(k, n):
    """two-sided exact binomial p, p0=0.5"""
    if n == 0:
        return float("nan")
    tot = 2 ** n
    def tail(x):
        return sum(comb(n, i) for i in range(x, n + 1)) / tot
    lo = min(k, n - k)
    p = 2 * sum(comb(n, i) for i in range(0, lo + 1)) / tot
    return min(1.0, p)


def paired(cellA, cellB, eps, m):
    """delta = A - B, per episode. returns (median delta, wins, n, p, deltas)"""
    ds = [epmed(cellA, e, m) - epmed(cellB, e, m) for e in eps]
    if m in MET_HIGHER:
        wins = sum(1 for d in ds if d > 0)
    else:
        wins = sum(1 for d in ds if d < 0)
    return st.median(ds), wins, len(ds), signtest(wins, len(ds)), ds


def medofmed(cell, eps, m):
    return st.median([epmed(cell, e, m) for e in eps])


# ---------------------------------------------------------------- integrity
print("=" * 100)
print("INTEGRITY")
print("=" * 100)

# 6.2 Phase-1 invariance
bad1 = []
for ep in ALL_EPS:
    for s in SEEDS:
        base = None
        for c in CELLS:
            r = IDX[(c, ep)][s]
            key = (round(r["init_nrmse"], 12), r["auto_selected"], r["auto_n_selected"],
                   r["ensemble_k"], r["N"])
            if base is None:
                base = key
            elif key != base:
                bad1.append((ep, s, c, key, base))
print(f"6.2 Phase-1 invariance mismatches: {len(bad1)}  (checked {len(ALL_EPS)*len(SEEDS)*len(CELLS)} runs)")
for b in bad1[:10]:
    print("   ", b)

# 6.3 worker-leak: AUTO-DUP vs AUTO bit-identity on ALL numeric fields
NUMF = ["nRMSE", "Variance_Ratio", "Pearson_rho", "Skill_Score", "Directional_R2", "Hs_error",
        "absVR_framewise", "decorr_mean", "shrink_mean", "m_mean", "avg_RMSE_unobserved",
        "N", "init_nrmse", "init_rmse"]
ndiff = 0
ncmp = 0
worst = 0.0
for ep in ALL_EPS:
    for s in SEEDS:
        a, d = IDX[("AUTO", ep)][s], IDX[("AUTO-DUP", ep)][s]
        for f in NUMF:
            ncmp += 1
            if a[f] != d[f]:
                ndiff += 1
                worst = max(worst, abs(a[f] - d[f]))
print(f"6.3 AUTO vs AUTO-DUP: {ncmp} numeric comparisons, mismatches={ndiff}, worst|diff|={worst:.3e}")
# also full vector fields
vdiff = 0
for ep in ALL_EPS:
    for s in SEEDS:
        a, d = IDX[("AUTO", ep)][s], IDX[("AUTO-DUP", ep)][s]
        for f in ("kx", "ky", "w", "vr_frames", "rho_frames"):
            if a[f] != d[f]:
                vdiff += 1
print(f"     vector fields (kx,ky,w,vr_frames,rho_frames): mismatches={vdiff}")

# repro lane vs B3
try:
    repro = [json.loads(l) for l in open(D + "repro.jsonl")]
    b3 = [json.loads(l) for l in open(D + "../B3_rx3_2026-08-06/results.jsonl")]
    b3i = {(r["cell"], r["episode"], r["seed"]): r for r in b3}
    nm = 0; nc = 0
    for r in repro:
        k = (r["cell"], r["episode"], r["seed"])
        if k not in b3i:
            k = ("AUTO", r["episode"], r["seed"])
        o = b3i.get(k)
        if o is None:
            print("   missing in B3:", k); continue
        for f in NUMF:
            if f in o:
                nc += 1
                if r[f] != o[f]:
                    nm += 1
    print(f"6.1 repro lane vs B3: {nc} comparisons, mismatches={nm}  (n_repro={len(repro)})")
except Exception as e:
    print("6.1 repro check error:", e)

# ---------------------------------------------------------------- null exclusion
print()
print("=" * 100)
print("NULL EXCLUSION (rule 4.a: control cell AUTO episode-median nRMSE >= episode null)")
print("=" * 100)
excl = []
for e in CROWDED:
    a = epmed("AUTO", e, "nRMSE")
    if a >= nulls[e]:
        excl.append(e)
    print(f"  {e:42s} AUTO={a:.4f}  null={nulls[e]:.4f}  {'EXCLUDED' if a>=nulls[e] else ''}")
C14 = [e for e in CROWDED if e not in excl]
print(f"  -> excluded {len(excl)}: {excl}")
print(f"  -> block C after exclusion n={len(C14)}")

print()
for e in SPARSE:
    a = epmed("AUTO", e, "nRMSE")
    print(f"  S {e:42s} AUTO={a:.4f}  null={nulls[e]:.4f}  {'EXCLUDED' if a>=nulls[e] else ''}")
S8 = [e for e in SPARSE if epmed("AUTO", e, "nRMSE") < nulls[e]]
print(f"  -> block S after exclusion n={len(S8)}")


def block_table(eps, label, m="nRMSE"):
    print()
    print("=" * 118)
    print(f"{label}   metric={m}   n={len(eps)}")
    print("=" * 118)
    nullmed = st.median([nulls[e] for e in eps])
    hdr = f"{'cell':14s} {'medOfMed':>9s} {'medD vs AUTO':>13s} {'w/n':>7s} {'p':>8s} {'medD vs NOTR':>13s} {'w/n':>7s} {'p':>8s}  {'NULLfail':>8s}"
    print(hdr)
    for c in CELLS:
        lv = medofmed(c, eps, m)
        if c == "AUTO":
            dA = (0.0, "-", "-", "-")
        else:
            md, k, n, p, _ = paired(c, "AUTO", eps, m)
            dA = (md, f"{k}/{n}", f"{p:.4f}", "")
        if c == "AUTO-NOTRACK":
            dN = (0.0, "-", "-")
        else:
            md2, k2, n2, p2, _ = paired(c, "AUTO-NOTRACK", eps, m)
            dN = (md2, f"{k2}/{n2}", f"{p2:.4f}")
        nf = "YES" if (m == "nRMSE" and lv >= nullmed) else ""
        print(f"{c:14s} {lv:9.4f} {dA[0]:13.4f} {str(dA[1]):>7s} {str(dA[2]):>8s} "
              f"{dN[0]:13.4f} {str(dN[1]):>7s} {str(dN[2]):>8s}  {nf:>8s}")
    print(f"  block null median (zero_null_tracking_nRMSE_mean) = {nullmed:.4f}")


for m in ["nRMSE", "final_minus_init", "Pearson_rho", "absVR_framewise", "m_mean",
          "decorr_mean", "shrink_mean", "avg_RMSE_unobserved", "Skill_Score"]:
    block_table(C14, "BLOCK C (crowded, null-excluded)", m)

block_table(CROWDED, "BLOCK C16 (crowded, NO exclusion) [prereg requires reporting]", "nRMSE")
block_table(S8, "BLOCK S (sparse, do-no-harm)", "nRMSE")
block_table(S8, "BLOCK S (sparse, do-no-harm)", "final_minus_init")

# ---------------------------------------------------------------- n=4 config-level
print()
print("=" * 100)
print("CONFIG-LEVEL n=4 (fold 4 episode-seeds by median, then 4 configs)  [NO 'significant' allowed]")
print("=" * 100)
CFG = {"BF4_15": [e for e in C14 if "BF4_15" in e], "BF4_30": [e for e in C14 if "BF4_30" in e],
       "BF5_15": [e for e in C14 if "BF5_15" in e], "BF5_30": [e for e in C14 if "BF5_30" in e]}
for c in CELLS:
    if c == "AUTO":
        continue
    ds = []
    for g, eps in CFG.items():
        ds.append(st.median([epmed(c, e, "nRMSE") - epmed("AUTO", e, "nRMSE") for e in eps]))
    dn = []
    for g, eps in CFG.items():
        dn.append(st.median([epmed(c, e, "nRMSE") - epmed("AUTO-NOTRACK", e, "nRMSE") for e in eps]))
    print(f"{c:14s} vs AUTO med={st.median(ds):+.4f} wins={sum(1 for d in ds if d<0)}/4  "
          f"[{', '.join(f'{d:+.4f}' for d in ds)}]   vs NOTR med={st.median(dn):+.4f} "
          f"wins={sum(1 for d in dn if d<0)}/4")
print("  group sizes:", {g: len(e) for g, e in CFG.items()})

# ---------------------------------------------------------------- per-episode dump
print()
print("=" * 100)
print("PER-EPISODE nRMSE (episode medians over seeds 1524/1525/1526), block C")
print("=" * 100)
print(f"{'episode':42s} " + " ".join(f"{c:>13s}" for c in CELLS) + "   null")
for e in CROWDED:
    tag = "  <-EXCL" if e in excl else ""
    print(f"{e:42s} " + " ".join(f"{epmed(c,e,'nRMSE'):13.4f}" for c in CELLS) +
          f"   {nulls[e]:.4f}{tag}")

print()
print("=" * 100)
print("PER-EPISODE final-init, block C")
print("=" * 100)
print(f"{'episode':42s} " + " ".join(f"{c:>13s}" for c in CELLS))
for e in CROWDED:
    print(f"{e:42s} " + " ".join(f"{epmed(c,e,'final_minus_init'):13.4f}" for c in CELLS))

# ---------------------------------------------------------------- router
print()
print("=" * 100)
print("AUTO ROUTER composition (auto_selected), all cells share Phase-1")
print("=" * 100)
from collections import Counter
cnt = Counter()
for e in CROWDED:
    for s in SEEDS:
        cnt[IDX[("AUTO", e)][s]["auto_selected"]] += 1
print("  block C (16ep x 3seed):", dict(cnt))
cnt = Counter()
for e in SPARSE:
    for s in SEEDS:
        cnt[IDX[("AUTO", e)][s]["auto_selected"]] += 1
print("  block S ( 8ep x 3seed):", dict(cnt))

# ---------------------------------------------------------------- recovery
print()
print("=" * 100)
print("RECOVERY RATE R = med(delta cell-AUTO)/med(delta NOTRACK-AUTO)  [DESCRIPTIVE ONLY]")
print("=" * 100)
base = paired("AUTO-NOTRACK", "AUTO", C14, "nRMSE")[0]
print(f"  floor med delta (NOTRACK-AUTO) = {base:+.4f}")
for c in ["T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"]:
    md = paired(c, "AUTO", C14, "nRMSE")[0]
    print(f"  {c:12s} med delta={md:+.4f}  R={md/base*100:6.1f}%")

# ---------------------------------------------------------------- MDE
print()
print("=" * 100)
print("EFFECT SIZE vs MDE-ish reference")
print("=" * 100)
for c in ["T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF", "AUTO-NOTRACK"]:
    md, k, n, p, ds = paired(c, "AUTO", C14, "nRMSE")
    print(f"  {c:14s} medD={md:+.4f}  mean={st.mean(ds):+.4f}  sd={st.stdev(ds):.4f}  "
          f"wins={k}/{n} p={p:.4f}  min={min(ds):+.4f} max={max(ds):+.4f}")
