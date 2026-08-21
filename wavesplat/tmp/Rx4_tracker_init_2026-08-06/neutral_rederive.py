#!/usr/bin/env python3
"""NEUTRAL reviewer independent re-derivation of Rx-4.
Written from scratch against PREREG.md sections 4/5/6. Read-only on all inputs.
"""
import json, math, statistics as st, itertools, collections, os

D = os.path.dirname(os.path.abspath(__file__))
B3 = os.path.join(D, "..", "B3_rx3_2026-08-06", "results.jsonl")

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

RES = load(os.path.join(D, "results.jsonl"))
REPRO = load(os.path.join(D, "repro.jsonl"))
B3ROWS = load(B3)
NULLS = {d["episode"]: d for d in json.load(open(os.path.join(D, "nulls_full.json")))}

# ---------------- block definitions (PREREG §3) ----------------
CROWDED = [f"episode_{b}_{c}comp_seed{s}_uniform.h5"
           for b in ("BF4", "BF5") for c in (15, 30) for s in range(4)]
ALLEPS = sorted({r["episode"] for r in RES})
SPARSE = sorted(set(ALLEPS) - set(CROWDED))
CELLS = ["AUTO", "AUTO-DUP", "AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"]

assert len(CROWDED) == 16 and set(CROWDED) <= set(ALLEPS), "crowded block mismatch"
assert len(SPARSE) == 8, SPARSE

# ---------------- §4.1 seed -> episode median ----------------
METRICS = ["nRMSE", "Pearson_rho", "absVR_framewise", "m_mean", "decorr_mean",
           "shrink_mean", "init_nrmse"]
# direction: True = lower is better (Delta<0 is a win)
LOWER_BETTER = {"nRMSE": True, "Pearson_rho": False, "absVR_framewise": True,
                "m_mean": True, "decorr_mean": True, "shrink_mean": True,
                "init_nrmse": True}

byslot = collections.defaultdict(list)          # (cell,ep) -> [row,...]
for r in RES:
    byslot[(r["cell"], r["episode"])].append(r)

MED = {}                                        # (cell,ep,metric) -> median over 3 seeds
for (cell, ep), rows in byslot.items():
    assert len(rows) == 3, (cell, ep, len(rows))
    for m in METRICS:
        MED[(cell, ep, m)] = st.median([r[m] for r in rows])

# ---------------- sign test ----------------
def binom_cdf_ge(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n

def binom_cdf_le(k, n):
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n

def signtest_two_sided(k, n):
    """k = wins, n = total (ties left in n, counted as non-wins per PREREG §4.5)."""
    if n == 0:
        return float("nan")
    return min(1.0, 2.0 * min(binom_cdf_le(k, n), binom_cdf_ge(k, n)))

def wins_needed(n, alpha=0.05):
    """Smallest k in the UPPER tail (k >= n/2) with two-sided p <= alpha."""
    for k in range(math.ceil(n / 2), n + 1):
        if signtest_two_sided(k, n) <= alpha:
            return k
    return None

# ---------------- §4.3 paired delta ----------------
def paired(cell, ref, eps, metric="nRMSE"):
    """Returns dict with median delta, wins, ties, n, p, per-episode deltas."""
    d = {e: MED[(cell, e, metric)] - MED[(ref, e, metric)] for e in eps}
    vals = list(d.values())
    lb = LOWER_BETTER[metric]
    wins = sum(1 for v in vals if (v < 0 if lb else v > 0))
    ties = sum(1 for v in vals if v == 0.0)
    n = len(vals)
    return dict(med=st.median(vals) if vals else float("nan"), wins=wins, ties=ties,
                n=n, p=signtest_two_sided(wins, n), deltas=d)

def medofmed(cell, eps, metric="nRMSE"):
    return st.median([MED[(cell, e, metric)] for e in eps])

def nullmed(eps):
    return st.median([NULLS[e]["zero_null_tracking_nRMSE_mean"] for e in eps])

# ================= §6 INTEGRITY =================
print("=" * 78)
print("§6 INTEGRITY CHECKS (re-run by NEUTRAL)")
print("=" * 78)

NUMFIELDS = ["nRMSE", "Variance_Ratio", "Pearson_rho", "Skill_Score", "Directional_R2",
             "Hs_error", "absVR_framewise", "decorr_mean", "shrink_mean", "m_mean",
             "avg_RMSE_unobserved", "N", "init_nrmse", "init_rmse", "auto_selected",
             "auto_n_selected", "ensemble_k", "final_n"]
LISTFIELDS = ["kx", "ky", "w", "vr_frames", "rho_frames"]

# --- 6.1 repro vs B3 bit-identity ---
b3idx = {(r["cell"], r["episode"], r["seed"]): r for r in B3ROWS}
n61 = n61bad = 0
bad61 = []
for r in REPRO:
    key = (r["cell"], r["episode"], r["seed"])
    o = b3idx.get(key)
    if o is None:
        bad61.append((key, "MISSING IN B3")); n61bad += 1; continue
    n61 += 1
    for f in NUMFIELDS:
        if r[f] != o[f]:
            bad61.append((key, f, r[f], o[f])); n61bad += 1
    for f in LISTFIELDS:
        if r[f] != o[f]:
            bad61.append((key, f, "LIST DIFF")); n61bad += 1
    # spec dict
    if r["spec"] != o["spec"]:
        bad61.append((key, "spec", "DICT DIFF")); n61bad += 1
print(f"6.1 repro(12) vs B3 : compared {n61} runs x {len(NUMFIELDS)+len(LISTFIELDS)+1} fields "
      f"-> mismatches {n61bad}  => {'PASS' if n61bad == 0 else 'FAIL'}")
for b in bad61[:20]:
    print("     ", b)

# --- 6.2 Phase-1 invariance ---
P1 = ["init_nrmse", "auto_selected", "auto_n_selected", "ensemble_k", "N"]
idx = {(r["cell"], r["episode"], r["seed"]): r for r in RES}
bad62 = collections.defaultdict(list)
n62 = 0
for ep in ALLEPS:
    for sd in (1524, 1525, 1526):
        base = idx[("AUTO", ep, sd)]
        for c in CELLS:
            if c == "AUTO":
                continue
            r = idx[(c, ep, sd)]
            for f in P1:
                n62 += 1
                if r[f] != base[f]:
                    bad62[c].append((ep, sd, f, r[f], base[f]))
tot62 = sum(len(v) for v in bad62.values())
print(f"6.2 Phase-1 invariance: {n62} comparisons -> mismatches {tot62}  "
      f"=> {'PASS' if tot62 == 0 else 'FAIL'}")
for c, v in bad62.items():
    print(f"     cell {c}: {len(v)} mismatches; e.g. {v[:3]}")

# --- 6.3 AUTO-DUP worker-leak probe ---
n63 = n63bad = 0
bad63 = []
for ep in ALLEPS:
    for sd in (1524, 1525, 1526):
        a, b = idx[("AUTO", ep, sd)], idx[("AUTO-DUP", ep, sd)]
        for f in NUMFIELDS:
            n63 += 1
            if a[f] != b[f]:
                bad63.append((ep, sd, f, a[f], b[f])); n63bad += 1
        for f in LISTFIELDS:
            n63 += 1
            if a[f] != b[f]:
                bad63.append((ep, sd, f, "LIST DIFF")); n63bad += 1
print(f"6.3 AUTO-DUP == AUTO : {n63} field comparisons over 72 (ep,seed) "
      f"-> mismatches {n63bad}  => {'PASS' if n63bad == 0 else 'FAIL'}")
for b in bad63[:20]:
    print("     ", b)

# ================= §4 NULL RULE =================
print()
print("=" * 78)
print("§4 NULL RULE")
print("=" * 78)
excl = [e for e in CROWDED
        if MED[("AUTO", e, "nRMSE")] >= NULLS[e]["zero_null_tracking_nRMSE_mean"]]
KEEP = [e for e in CROWDED if e not in excl]
print(f"(a) episode exclusion by control cell AUTO: {len(excl)} excluded -> n={len(KEEP)}")
for e in CROWDED:
    v = MED[("AUTO", e, "nRMSE")]; nl = NULLS[e]["zero_null_tracking_nRMSE_mean"]
    print(f"    {'EXCL' if e in excl else '    '} {e:44s} AUTO_med={v:.4f}  null={nl:.4f}  "
          f"margin={v-nl:+.4f}")

exclS = [e for e in SPARSE
         if MED[("AUTO", e, "nRMSE")] >= NULLS[e]["zero_null_tracking_nRMSE_mean"]]
KEEPS = [e for e in SPARSE if e not in exclS]
print(f"\n    block S: {len(exclS)} excluded -> n={len(KEEPS)}")
for e in SPARSE:
    v = MED[("AUTO", e, "nRMSE")]; nl = NULLS[e]["zero_null_tracking_nRMSE_mean"]
    print(f"    {'EXCL' if e in exclS else '    '} {e:44s} AUTO_med={v:.4f}  null={nl:.4f}  "
          f"margin={v-nl:+.4f}")

print("\n(b) cell NULL-FAILING check (medOfMed >= median null of the block)")
for tag, eps in (("C-kept", KEEP), ("C-all16", CROWDED), ("S-kept", KEEPS), ("S-all8", SPARSE)):
    nl = nullmed(eps)
    print(f"  block {tag} (n={len(eps)}) null_median={nl:.4f}")
    for c in CELLS:
        mm = medofmed(c, eps)
        print(f"     {c:14s} medOfMed={mm:.4f}  {'NULL-FAILING' if mm >= nl else 'ok'}")

# ================= HEADLINE TABLES =================
def table(title, eps, metric="nRMSE"):
    nl = nullmed(eps)
    print()
    print("=" * 100)
    print(f"{title}   (n={len(eps)} episodes, metric={metric}, null_median={nl:.4f})")
    print("=" * 100)
    hdr = (f"{'cell':14s} {'medOfMed':>9s} | {'medD_vsAUTO':>11s} {'w/n':>7s} {'ties':>4s} "
           f"{'p':>8s} | {'medD_vsNOTR':>11s} {'w/n':>7s} {'ties':>4s} {'p':>8s} | nullfail")
    print(hdr); print("-" * len(hdr))
    for c in CELLS:
        mm = medofmed(c, eps, metric)
        a = paired(c, "AUTO", eps, metric)
        b = paired(c, "AUTO-NOTRACK", eps, metric)
        print(f"{c:14s} {mm:9.4f} | {a['med']:+11.4f} {a['wins']:3d}/{a['n']:<3d} "
              f"{a['ties']:4d} {a['p']:8.4f} | {b['med']:+11.4f} {b['wins']:3d}/{b['n']:<3d} "
              f"{b['ties']:4d} {b['p']:8.4f} | "
              f"{'YES' if mm >= nl else 'no'}")

table("BLOCK C  CROWDED, NULL-EXCLUDED  (PRIMARY)", KEEP)
table("BLOCK C  CROWDED, ALL 16 (no exclusion) [companion]", CROWDED)
table("BLOCK S  SPARSE, NULL-EXCLUDED (do-no-harm)", KEEPS)
table("BLOCK S  SPARSE, ALL 8 [companion]", SPARSE)

# ================= tie census =================
print()
print("=" * 78)
print("EXACT-TIE CENSUS (Delta == 0.0 exactly), nRMSE")
print("=" * 78)
for tag, eps in (("C-kept", KEEP), ("C-all16", CROWDED), ("S-all8", SPARSE)):
    print(f"-- block {tag} (n={len(eps)})")
    for c in CELLS:
        a = paired(c, "AUTO", eps); b = paired(c, "AUTO-NOTRACK", eps)
        print(f"   {c:14s} ties_vsAUTO={a['ties']:2d}/{a['n']}  ties_vsNOTRACK={b['ties']:2d}/{b['n']}")

# sign-test threshold table
print()
print("SIGN TEST: wins needed for two-sided p<=0.05")
for n in (4, 8, 10, 12, 13, 14, 15, 16):
    k = wins_needed(n)
    if k is None:
        print(f"   n={n:2d}: impossible (floor p={signtest_two_sided(n,n):.4f})")
    else:
        prev = f"k={k-1} -> p={signtest_two_sided(k-1,n):.4f}; " if k >= 1 else ""
        print(f"   n={n:2d}: k>={k}  (p={signtest_two_sided(k,n):.4f}); {prev}"
              f"floor p={signtest_two_sided(n,n):.2e}")

# tie-robust alternative (NOT prereg): drop exact ties from n
print()
print("ALTERNATIVE (NOT PREREG) -- exact ties DROPPED from n, block C-kept:")
for c in CELLS:
    a = paired(c, "AUTO", KEEP)
    n2 = a["n"] - a["ties"]
    p2 = signtest_two_sided(a["wins"], n2) if n2 > 0 else float("nan")
    print(f"   {c:14s} vsAUTO wins={a['wins']}/{n2} (dropped {a['ties']} ties) p={p2:.4f}"
          f"   [prereg: {a['wins']}/{a['n']} p={a['p']:.4f}]")

# ================= n=4 setting-level fold =================
print()
print("=" * 78)
print("SETTING-LEVEL n=4 (fold 4 episode-seeds by median) -- NO 'significant' allowed")
print("=" * 78)
SETTINGS = {b + "_" + str(c) + "comp": [f"episode_{b}_{c}comp_seed{s}_uniform.h5" for s in range(4)]
            for b in ("BF4", "BF5") for c in (15, 30)}
for c in CELLS:
    dd = {}
    for name, eps in SETTINGS.items():
        eps_k = [e for e in eps if e in KEEP]
        if not eps_k:
            dd[name] = float("nan"); continue
        dd[name] = st.median([MED[(c, e, "nRMSE")] - MED[("AUTO", e, "nRMSE")] for e in eps_k])
    vals = [v for v in dd.values() if v == v]
    wins = sum(1 for v in vals if v < 0)
    print(f"   {c:14s} med={st.median(vals):+.4f}  wins={wins}/{len(vals)}  "
          f"per-setting=" + " ".join(f"{k}:{v:+.4f}" for k, v in dd.items()))

# ================= D1 / D2 =================
print()
print("=" * 78)
print("D1 / D2")
print("=" * 78)
for eps, tag in ((KEEP, "C-kept n=%d" % len(KEEP)), (CROWDED, "C-all16")):
    ph = paired("T-PHIONLY", "AUTO", eps); wo = paired("T-WONLY", "AUTO", eps)
    ea = paired("T-EARLY20", "AUTO", eps); nt = paired("AUTO-NOTRACK", "AUTO", eps)
    print(f"[{tag}]")
    print(f"   |medD(T-PHIONLY-AUTO)| = {abs(ph['med']):.4f}   (med {ph['med']:+.4f})")
    print(f"   |medD(T-WONLY  -AUTO)| = {abs(wo['med']):.4f}   (med {wo['med']:+.4f})")
    print(f"   D1 => {'DIAGNOSIS SUPPORTED' if abs(ph['med'])>abs(wo['med']) else 'DIAGNOSIS REFUTED'}")
    print(f"   floor medD(NOTRACK-AUTO) = {nt['med']:+.4f}  wins {nt['wins']}/{nt['n']} p={nt['p']:.4f}")
    print(f"   medD(T-EARLY20-AUTO)     = {ea['med']:+.4f}  wins {ea['wins']}/{ea['n']} p={ea['p']:.4f}")
    frac = ea['med'] / nt['med'] if nt['med'] else float('nan')
    print(f"   D2 => EARLY20 between AUTO(0) and floor? "
          f"{'YES' if (nt['med'] < ea['med'] < 0) else 'NO'}  (fraction of floor = {frac:.3f})")
    # per-episode D2
    ok = sum(1 for e in eps if nt['deltas'][e] < ea['deltas'][e] < 0)
    print(f"      per-episode strictly between: {ok}/{len(eps)}")

# ================= effect sizes / power =================
print()
print("=" * 78)
print("EFFECT SIZE vs MDE (block C-kept)")
print("=" * 78)
n = len(KEEP)
kneed = wins_needed(n)
print(f"n={n}, wins needed = {kneed}/{n} for two-sided p<=0.05")
print("Per-episode paired deltas vs AUTO (nRMSE):")
for c in CELLS:
    a = paired(c, "AUTO", KEEP)
    v = sorted(a["deltas"].values())
    print(f"  {c:14s} med={a['med']:+.4f} min={v[0]:+.4f} max={v[-1]:+.4f} "
          f"IQR=[{v[n//4]:+.4f},{v[(3*n)//4]:+.4f}] wins={a['wins']}/{n} p={a['p']:.4f}")

# ================= auto router census =================
print()
print("=" * 78)
print("AUTO ROUTER CENSUS (auto_selected), block C")
print("=" * 78)
cnt = collections.Counter()
for e in CROWDED:
    for sd in (1524, 1525, 1526):
        cnt[idx[("AUTO", e, sd)]["auto_selected"]] += 1
print("  crowded16 x 3 seeds:", dict(cnt))
cnt = collections.Counter()
for e in SPARSE:
    for sd in (1524, 1525, 1526):
        cnt[idx[("AUTO", e, sd)]["auto_selected"]] += 1
print("  sparse8   x 3 seeds:", dict(cnt))

# ================= secondary metrics on PRIMARY cell =================
print()
print("=" * 78)
print("SECONDARY METRICS, block C-kept, vs AUTO")
print("=" * 78)
for m in ("Pearson_rho", "absVR_framewise", "m_mean", "decorr_mean", "shrink_mean"):
    print(f"-- {m} (better = {'lower' if LOWER_BETTER[m] else 'higher'})")
    for c in ("AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"):
        a = paired(c, "AUTO", KEEP, m)
        print(f"     {c:14s} medD={a['med']:+.4f} wins={a['wins']}/{a['n']} p={a['p']:.4f}")
