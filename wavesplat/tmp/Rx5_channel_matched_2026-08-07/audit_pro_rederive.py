#!/usr/bin/env python3
"""PRO independent re-derivation of Rx-5 + E-sweep.
Written from scratch against PREREG.md sections 4/5/6. Reads results.jsonl only.
No writes outside this directory. No GPU.
"""
import json, os, sys, itertools, math
from statistics import median
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results.jsonl")
NULLS = "/home/vexit/wavesplat/wavesplat/tmp/K6_run_2026-08-07/nulls_k6.json"

rows = [json.loads(l) for l in open(RES)]
nulls = {d["episode"]: d for d in json.load(open(NULLS))}

CELLS = ["AUTO", "AUTO-DUP", "AUTO-NOTRACK", "W-5e3", "W-5e4", "W-MATCH", "W-5e6",
         "PHI-5e3", "E-2", "E-5", "E-10", "E-20", "E-40", "E-80"]
EPS = sorted(set(r["episode"] for r in rows))
SEEDS = sorted(set(r["seed"] for r in rows))

# ---------- exact two-sided sign test ----------
def C(n, k):
    return math.comb(n, k)

def sign_test(deltas, tol=0.0):
    """deltas: list of signed paired differences (intervention - reference).
    'win' = delta < 0 (improvement, since nRMSE larger is worse).
    Ties (|d| <= tol) excluded from n per PREREG 4.1."""
    kept = [d for d in deltas if abs(d) > tol]
    ties = len(deltas) - len(kept)
    n = len(kept)
    wins = sum(1 for d in kept if d < 0)
    if n == 0:
        return dict(n=0, wins=0, ties=ties, p=1.0, med=median(deltas) if deltas else 0.0)
    k = max(wins, n - wins)
    p = min(1.0, 2.0 * sum(C(n, i) for i in range(k, n + 1)) / 2 ** n)
    return dict(n=n, wins=wins, ties=ties, p=p, med=median(deltas))

# PREREG 4.4 frozen pass thresholds
THRESH = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}

# ---------- seed-median aggregation (PREREG 4: seed median -> paired signed delta) ----------
def seedmed(metric):
    """returns {(cell, episode): median over seeds}"""
    buck = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        buck[(r["cell"], r["episode"])].append(v)
    return {k: median(v) for k, v in buck.items()}, {k: v for k, v in buck.items()}

NRMSE, NRMSE_RAW = seedmed("nRMSE")

def paired(cell_a, cell_b, table=None, eps=None):
    """signed delta = a - b, per episode"""
    table = table if table is not None else NRMSE
    eps = eps if eps is not None else EPS
    return [table[(cell_a, e)] - table[(cell_b, e)] for e in eps]

# ================= INTEGRITY (PREREG 6) =================
print("=" * 78)
print("INTEGRITY (PREREG 6)")
print("=" * 78)

# 6.1 worker leak: AUTO-DUP bit-identical to AUTO across every field
BITFIELDS = ["nRMSE", "Variance_Ratio", "Pearson_rho", "Skill_Score", "Directional_R2",
             "Hs_error", "absVR_framewise", "decorr_mean", "shrink_mean", "m_mean",
             "avg_RMSE_unobserved", "N", "init_nrmse", "init_rmse", "auto_selected",
             "auto_n_selected", "ensemble_k"]
idx = {(r["cell"], r["episode"], r["seed"]): r for r in rows}
ndiff = 0; ncmp = 0
worst = 0.0
for e in EPS:
    for s in SEEDS:
        a = idx[("AUTO", e, s)]; b = idx[("AUTO-DUP", e, s)]
        for f in BITFIELDS:
            ncmp += 1
            if a[f] != b[f]:
                ndiff += 1
            if isinstance(a[f], float) and isinstance(b[f], float):
                worst = max(worst, abs(a[f] - b[f]))
        # vector fields
        for f in ["kx", "ky", "w", "vr_frames", "rho_frames"]:
            ncmp += 1
            if a.get(f) != b.get(f):
                ndiff += 1
print(f"6.1 AUTO vs AUTO-DUP: {ncmp} field comparisons, mismatches={ndiff}, worst|float diff|={worst:.3e}")

# AUTO-DUP as a sign test (should be 16 ties)
d = paired("AUTO-DUP", "AUTO")
st = sign_test(d)
print(f"    AUTO-DUP sign test on nRMSE: ties={st['ties']}/16, n={st['n']}  -> {'PASS' if st['ties']==16 else 'FAIL'}")

# 6.2 Phase-1 invariance across all 14 cells
inv_fields = ["init_nrmse", "auto_selected", "N", "ensemble_k", "init_rmse", "auto_n_selected"]
viol = defaultdict(int); tot = 0
for e in EPS:
    for s in SEEDS:
        ref = idx[("AUTO", e, s)]
        for c in CELLS:
            r = idx[(c, e, s)]
            for f in inv_fields:
                tot += 1
                if r[f] != ref[f]:
                    viol[(c, f)] += 1
print(f"6.2 Phase-1 invariance: {tot} comparisons across {len(CELLS)} cells, violations={sum(viol.values())}")
if viol:
    for k, v in sorted(viol.items()):
        print("     VIOLATION", k, v)

# 6.3 wiring: one knob per label, from cfg records
def cfgdiff(ca, cb):
    diffs = {}
    for e in EPS:
        for s in SEEDS:
            A = idx[(ca, e, s)]["cfg"]; B = idx[(cb, e, s)]["cfg"]
            for k in set(A) | set(B):
                if A.get(k, "<absent>") != B.get(k, "<absent>"):
                    diffs[k] = (A.get(k, "<absent>"), B.get(k, "<absent>"))
    return diffs

print("6.3 wiring (independent from cfg records):")
for pair in [("W-5e3", "W-5e4"), ("W-5e3", "W-MATCH"), ("W-5e3", "W-5e6"),
             ("AUTO", "W-5e3"), ("AUTO", "PHI-5e3"), ("AUTO", "AUTO-NOTRACK"),
             ("AUTO", "AUTO-DUP")] + [("AUTO", f"E-{E}") for E in (2, 5, 10, 20, 40, 80)]:
    dd = cfgdiff(*pair)
    print(f"    {pair[0]:14s} vs {pair[1]:14s} nknob={len(dd)}  {dd}")

# missing / NaN
bad = [r for r in rows if r.get("nRMSE") is None or (isinstance(r["nRMSE"], float) and math.isnan(r["nRMSE"]))]
print(f"    missing/NaN nRMSE runs: {len(bad)} ; total runs {len(rows)} (expected 672)")

# ================= NULLS (PREREG 4.5) =================
print()
print("=" * 78)
print("NULL SCREEN (PREREG 4.5/4.6)")
print("=" * 78)
lvl = {c: {e: NRMSE[(c, e)] for e in EPS} for c in CELLS}
excl = []
for e in EPS:
    nl = nulls[e]["zero_null_tracking_nRMSE_mean"]
    if lvl["AUTO"][e] >= nl:
        excl.append(e)
print(f"AUTO fails null in {len(excl)} episodes: {excl}")
KEEP = [e for e in EPS if e not in excl]
n_keep = len(KEEP)
print(f"n = {n_keep}, frozen required wins = {THRESH[n_keep]}")
print()
print(f"{'cell':14s} {'blockmed':>9s} {'meanlvl':>9s} {'beats null':>11s}")
for c in CELLS:
    bn = sum(1 for e in EPS if lvl[c][e] < nulls[e]["zero_null_tracking_nRMSE_mean"])
    print(f"{c:14s} {median([lvl[c][e] for e in EPS]):9.4f} "
          f"{sum(lvl[c][e] for e in EPS)/16:9.4f} {bn:>8d}/16")
print(f"{'NULL':14s} {median([nulls[e]['zero_null_tracking_nRMSE_mean'] for e in EPS]):9.4f} "
      f"{sum(nulls[e]['zero_null_tracking_nRMSE_mean'] for e in EPS)/16:9.4f}")

# ================= PAIRED CONTRASTS =================
def report(a, b, label, eps=None, table=None):
    d = paired(a, b, table=table, eps=eps)
    st = sign_test(d)
    thr = THRESH.get(st["n"], None)
    verdict = ""
    if thr is not None:
        verdict = "PASS(a better)" if st["wins"] >= thr else (
            "PASS(b better)" if (st["n"] - st["wins"]) >= thr else "below threshold")
    print(f"{label:44s} med={st['med']:+.4f}  wins={st['wins']:2d}/{st['n']:2d} "
          f"p={st['p']:.4f}  {verdict}")
    return st

print()
print("=" * 78)
print("A) CAPACITY-RESPONSE  (reference = W-5e3, one knob: tracking_lr)")
print("=" * 78)
for c in ["W-5e4", "W-MATCH", "W-5e6"]:
    report(c, "W-5e3", f"{c} - W-5e3", eps=KEEP)
print()
report("W-MATCH", "W-5e4", "W-MATCH - W-5e4  (adjacent)", eps=KEEP)
report("W-5e6", "W-MATCH", "W-5e6  - W-MATCH (adjacent)", eps=KEEP)

print()
print("=" * 78)
print("vs AUTO (descriptive; PREREG 2 note: W-* vs AUTO is 2 knobs)")
print("=" * 78)
for c in ["W-5e3", "W-5e4", "W-MATCH", "W-5e6", "PHI-5e3", "AUTO-NOTRACK",
          "E-2", "E-5", "E-10", "E-20", "E-40", "E-80"]:
    report(c, "AUTO", f"{c} - AUTO", eps=KEEP)

print()
print("=" * 78)
print("vs FLOOR (AUTO-NOTRACK)")
print("=" * 78)
for c in ["W-5e3", "W-5e4", "W-MATCH", "W-5e6", "PHI-5e3",
          "E-2", "E-5", "E-10", "E-20", "E-40", "E-80"]:
    report(c, "AUTO-NOTRACK", f"{c} - NOTRACK", eps=KEEP)

# ================= PRIMARY-A =================
print()
print("=" * 78)
print("PRIMARY-A  (PREREG 5): median[ D(W-MATCH - AUTO) - D(PHI-5e3 - AUTO) ]")
print("=" * 78)
dA = [(NRMSE[("W-MATCH", e)] - NRMSE[("AUTO", e)]) -
      (NRMSE[("PHI-5e3", e)] - NRMSE[("AUTO", e)]) for e in KEEP]
# identical to W-MATCH - PHI-5e3 per episode; verify
dA2 = paired("W-MATCH", "PHI-5e3", eps=KEEP)
assert max(abs(x - y) for x, y in zip(dA, dA2)) < 1e-15
stA = sign_test(dA)
thrA = THRESH[stA["n"]]
pos = sum(1 for d in dA if d > 0)
print(f"  median = {stA['med']:+.6f}   (>0 => amplitude more fragile = CHANNEL-REAL)")
print(f"  episodes with positive delta (amp worse) = {pos}/{stA['n']}")
print(f"  episodes with negative delta (phase worse) = {stA['n']-pos}/{stA['n']}")
print(f"  exact two-sided sign test p = {stA['p']:.4f}; threshold = {thrA}/{stA['n']}")
if stA["med"] > 0 and pos >= thrA:
    print("  VERDICT = CHANNEL-REAL")
elif stA["med"] < 0 and (stA["n"] - pos) >= thrA:
    print("  VERDICT = INVERTED")
else:
    print("  VERDICT = NO-DIFFERENCE  (neither direction reaches threshold)")
print("  per-episode deltas:")
for e, d in zip(KEEP, dA):
    print(f"    {e:52s} {d:+.5f}")

# ================= PRIMARY-B =================
print()
print("=" * 78)
print("PRIMARY-B  (PREREG 5): E-sweep vs pre-declared uniform-damage null R(E)=1-E/160")
print("=" * 78)
FRAMES = 160
gap = {e: NRMSE[("AUTO-NOTRACK", e)] - NRMSE[("AUTO", e)] for e in KEEP}  # negative = floor better
print(f"  gap(NOTRACK - AUTO): median = {median(list(gap.values())):+.5f}, "
      f"n_negative = {sum(1 for v in gap.values() if v<0)}/{len(KEEP)}")
print()
print(f"{'E':>4s} {'pred R':>7s} {'R(med of ratios)':>17s} {'R(ratio of meds)':>17s} "
      f"{'obs-pred wins':>14s} {'p':>8s} {'med(obs-pred)':>14s}")
for E in (2, 5, 10, 20, 40, 80):
    c = f"E-{E}"
    predR = 1 - E / FRAMES
    obs = {e: NRMSE[(c, e)] - NRMSE[("AUTO", e)] for e in KEEP}
    ratios = [obs[e] / gap[e] for e in KEEP if abs(gap[e]) > 1e-12]
    Rmed = median(ratios)
    Rrat = median(list(obs.values())) / median(list(gap.values()))
    # paired test: observed improvement vs predicted improvement
    pred = {e: predR * gap[e] for e in KEEP}
    dif = [obs[e] - pred[e] for e in KEEP]   # <0 => recovered MORE than uniform predicts (front-loaded)
    st = sign_test(dif)
    thr = THRESH[st["n"]]
    tag = ""
    if st["wins"] >= thr:
        tag = "FRONT-LOADED"
    elif (st["n"] - st["wins"]) >= thr:
        tag = "BACK-LOADED"
    else:
        tag = "not distinguished"
    print(f"{E:4d} {predR*100:6.1f}% {Rmed*100:16.1f}% {Rrat*100:16.1f}% "
          f"{st['wins']:6d}/{st['n']:2d} {st['p']:8.4f} {st['med']:+13.5f}   {tag}")

# non-monotonicity of R(E): paired episode-level test E-10 vs its neighbours
print()
print("  Non-monotonicity probe (episode-level paired recovery R_ep):")
Rep = {}
for E in (2, 5, 10, 20, 40, 80):
    c = f"E-{E}"
    Rep[E] = {e: (NRMSE[(c, e)] - NRMSE[("AUTO", e)]) / gap[e] for e in KEEP}
for a, b in [(2, 5), (5, 10), (10, 20), (20, 40), (40, 80), (2, 10), (10, 40)]:
    d = [Rep[b][e] - Rep[a][e] for e in KEEP]
    st = sign_test(d)
    up = sum(1 for x in d if x > 0)
    print(f"    R(E={b}) - R(E={a}): med={st['med']:+.4f}  higher-at-larger-E in {up}/{st['n']} eps  p={st['p']:.4f}")

# also: raw nRMSE monotonicity  E-10 vs E-2/E-20
print()
print("  Raw nRMSE monotonicity of the E cells (paired):")
for a, b in [("E-2", "E-5"), ("E-5", "E-10"), ("E-10", "E-20"), ("E-20", "E-40"), ("E-40", "E-80")]:
    report(b, a, f"{b} - {a}", eps=KEEP)

# ================= PANEL of secondary metrics (PREREG 4 table) =================
print()
print("=" * 78)
print("SECONDARY PANEL (seed-median -> paired signed delta -> median)")
print("=" * 78)
PANEL = [("Pearson_rho", -1), ("absVR_framewise", +1), ("m_mean", +1),
         ("decorr_mean", +1), ("shrink_mean", +1), ("init_nrmse", +1)]
tables = {m: seedmed(m)[0] for m, _ in PANEL}
for cell in ["W-MATCH", "W-5e6", "PHI-5e3", "W-5e3", "AUTO-NOTRACK", "E-10", "E-20"]:
    out = [f"{cell:14s}"]
    for m, sgn in PANEL:
        T = tables[m]
        d = [(T[(cell, e)] - T[("AUTO", e)]) * sgn for e in KEEP]  # sgn makes <0 = better
        st = sign_test(d)
        out.append(f"{m}: med={st['med']:+.4f} w={st['wins']}/{st['n']}")
    print("  " + " | ".join(out))
print()
print("  same panel vs FLOOR (AUTO-NOTRACK):")
for cell in ["W-MATCH", "W-5e6", "E-10"]:
    out = [f"{cell:14s}"]
    for m, sgn in PANEL:
        T = tables[m]
        d = [(T[(cell, e)] - T[("AUTO-NOTRACK", e)]) * sgn for e in KEEP]
        st = sign_test(d)
        out.append(f"{m}: med={st['med']:+.4f} w={st['wins']}/{st['n']}")
    print("  " + " | ".join(out))

# ================= CONFIG FOLD (n=4, PREREG 4.7: report, no significance, no effect size) =================
print()
print("=" * 78)
print("CONFIG FOLD n=4  (PREREG 4.7: report direction only; NO significance, NO effect size)")
print("=" * 78)
FOLDS = defaultdict(list)
for e in EPS:
    key = e.split("_seed")[0]          # BF4_15comp etc
    FOLDS[key].append(e)
EPSEED = defaultdict(list)
for e in EPS:
    key = "seed" + e.split("_seed")[1].split("_")[0]
    EPSEED[key].append(e)
for name, groups in [("config", FOLDS), ("episode-seed", EPSEED)]:
    print(f"  --- {name} folds ({len(groups)}) ---")
    for a, b in [("W-MATCH", "PHI-5e3"), ("W-MATCH", "AUTO"), ("W-MATCH", "AUTO-NOTRACK"),
                 ("E-10", "AUTO"), ("E-10", "AUTO-NOTRACK")]:
        sgn = []
        for g, es in sorted(groups.items()):
            m = median([NRMSE[(a, e)] - NRMSE[(b, e)] for e in es if e in KEEP])
            sgn.append(m < 0)
        print(f"    {a:9s} - {b:13s} : better in {sum(sgn)}/{len(sgn)} folds  (direction only)")
