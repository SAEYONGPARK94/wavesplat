import sys, math, statistics as st
sys.path.insert(0, "/home/vexit/wavesplat/wavesplat/tmp/Rx6_channel_curve_2026-08-07")
from audit_con_load import *

print("=" * 92)
print("A. INTEGRITY RE-CHECK (my own code)")
print("=" * 92)

# A1 AUTO-DUP bit identity
bad = 0
scal = ["nRMSE", "Variance_Ratio", "Pearson_rho", "Skill_Score", "Directional_R2",
        "Hs_error", "absVR_framewise", "decorr_mean", "shrink_mean", "m_mean",
        "avg_RMSE_unobserved", "N", "init_nrmse", "init_rmse"]
n_chk = 0
for ep in EPS:
    for s in SEEDS:
        a, b = IDX[("AUTO", ep, s)], IDX[("AUTO-DUP", ep, s)]
        for k in scal:
            n_chk += 1
            if a[k] != b[k]:
                bad += 1
        for k in ["w", "phi", "kx", "ky"]:
            n_chk += 1
            if a[k] != b[k]:
                bad += 1
print(f"A1 AUTO-DUP == AUTO : {n_chk - bad}/{n_chk} identical  (mismatch={bad})")

# A2 Phase-1 invariance across all 13 cells
bad1 = []
for ep in EPS:
    for s in SEEDS:
        ref = IDX[("AUTO", ep, s)]
        for c in CELLS:
            r = IDX[(c, ep, s)]
            for k in ["init_nrmse", "init_rmse", "auto_selected", "N", "ensemble_k", "auto_n_selected"]:
                if r[k] != ref[k]:
                    bad1.append((c, ep, s, k, r[k], ref[k]))
print(f"A2 Phase-1 invariance across 13 cells x 96 (ep,seed): mismatches={len(bad1)}")
if bad1[:5]:
    for x in bad1[:5]:
        print("   ", x)

# A3 auto_selected distribution
from collections import Counter
print("A3 auto_selected:", Counter(r["auto_selected"] for r in RECS if r["cell"] == "AUTO"))
print("A3 N distribution:", sorted(Counter(IDX[("AUTO", e, SEEDS[0])]["N"] for e in EPS).items()))

# A4 cfg knobs per cell
print("A4 cfg per cell:")
for c in CELLS:
    cfg = IDX[(c, EPS[0], SEEDS[0])]["cfg"]
    print(f"   {c:<14} " + "  ".join(f"{k}={v!r}" for k, v in sorted(cfg.items())))

# A5 wall time -> did every run recompute phase 1?
ws = [r["wall_s"] for r in RECS]
print(f"A5 wall_s: min={min(ws):.2f} med={st.median(ws):.2f} max={max(ws):.2f}")

print()
print("=" * 92)
print("B. NULLS  (PREREG 4.4/4.5)")
print("=" * 92)
# null: per PREREG, zero_null_tracking_nRMSE_mean is the tracking null.
passfail = {}
for c in CELLS:
    b = block(c, "nRMSE")
    npass = sum(1 for e in EPS if b[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"])
    mgt1 = sum(1 for e in EPS if block(c, "m_mean")[e] > 1.0)
    passfail[c] = (npass, mgt1)
    print(f"   {c:<14} null-pass {npass:2d}/32   m>1 {mgt1:2d}/32   blockmed nRMSE={st.median(list(b.values())):.5f}")

AUTOb = block("AUTO", "nRMSE")
auto_fail = [e for e in EPS if not (AUTOb[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"])]
print(f"   AUTO null-fail episodes ({len(auto_fail)}): {[e[8:-18] for e in auto_fail]}")
EPS_PRIM = [e for e in EPS if e not in auto_fail]
print(f"   PRIMARY episode set n={len(EPS_PRIM)}")

print()
print("=" * 92)
print("C. PRIMARY  C(S) = median[ (W-S) - (P-S) ]   (positive = amplitude worse)")
print("=" * 92)
NEED = {32: 22, 31: 22, 30: 21, 29: 21, 28: 20, 27: 20, 26: 19}
for S in "12345":
    r = report(f"W-S{S}", f"P-S{S}", eps=EPS_PRIM, label=f"C(S{S}) W-S{S} - P-S{S}")
    need = NEED.get(r["n"], "?")
    mx = max(r["pos"], r["neg"])
    print(fmt(r), f"   maxwin={mx} need={need} -> {'PASS' if isinstance(need,int) and mx>=need else 'fail'}")
print()
print("  [unexcluded, all 32]")
for S in "12345":
    r = report(f"W-S{S}", f"P-S{S}", eps=EPS, label=f"C(S{S}) all32")
    mx = max(r["pos"], r["neg"])
    print(fmt(r), f"   maxwin={mx} need={NEED[32]}")

print()
print("=" * 92)
print("D. LEVELS + DEGENERACY (each cell vs AUTO and vs NOTRACK)")
print("=" * 92)
for c in [f"{ch}-S{i}" for ch in "WP" for i in "12345"] + ["AUTO-NOTRACK"]:
    rA = report(c, "AUTO", eps=EPS_PRIM)
    rB = report(c, "AUTO-NOTRACK", eps=EPS_PRIM)
    print(f"{c:<14} vs AUTO   medD={rA['medD']:+.5f} {rA['pos']:2d}/{rA['n']:2d} p={rA['p']:.4g}"
          f"   | vs FLOOR medD={rB['medD']:+.5f} {rB['pos']:2d}/{rB['n']:2d} p={rB['p']:.4g}")

print()
print("=" * 92)
print("E. AUTO-NOTRACK - AUTO  (3rd independent measurement)")
print("=" * 92)
for eps_, tag in [(EPS_PRIM, "primary"), (EPS, "all32")]:
    r = report("AUTO-NOTRACK", "AUTO", eps=eps_, label=f"NOTRACK-AUTO [{tag}]")
    print(fmt(r))
# per-seed (no median-of-3) to check seed robustness
for s in SEEDS:
    d = [IDX[("AUTO-NOTRACK", e, s)]["nRMSE"] - IDX[("AUTO", e, s)]["nRMSE"] for e in EPS]
    n, pos, p = sign_test(d)
    print(f"   seed {s}: medD={st.median(d):+.5f} pos={pos}/{n} p={p:.4g}")
