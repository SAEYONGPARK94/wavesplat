import sys, math, statistics as st, random, itertools
sys.path.insert(0, "/home/vexit/wavesplat/wavesplat/tmp/Rx6_channel_curve_2026-08-07")
from audit_con_load import *

TWOPI = 2 * math.pi
wrap = lambda x: (x + math.pi) % TWOPI - math.pi
rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
AUTOb = block("AUTO", "nRMSE")
EPS_PRIM = [e for e in EPS if AUTOb[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"]]
CH = [f"{c}-S{i}" for c in "WP" for i in "12345"]

print("=" * 100)
print("H1. Is `spectrum_params` the SCORED model?  (ensemble_k>1 -> member-0 only)")
print("=" * 100)
ens = sorted({(r["episode"], r["seed"]) for r in RECS if r["ensemble_k"] != 1})
print(f"  (ep,seed) with ensemble_k=3: {len(ens)} -> {[(e[8:-18], s) for e, s in ens]}")
epc = sorted({e for e, s in ens})
print(f"  => D for those runs describes MEMBER 0, not the median-of-3 field that was scored.")
print(f"  => episodes touched: {len(epc)}/32; one of them (BF5_30comp_seed6) has 2/3 seeds ensembled")

print()
print("=" * 100)
print("H2. Does the k-grid stay frozen?  (D only meaningful if kx,ky identical to NOTRACK)")
print("=" * 100)
bad = 0
tot = 0
for c in CH:
    for e in EPS:
        for s in SEEDS:
            tot += 1
            if IDX[(c, e, s)]["kx"] != IDX[("AUTO-NOTRACK", e, s)]["kx"]:
                bad += 1
            if IDX[(c, e, s)]["ky"] != IDX[("AUTO-NOTRACK", e, s)]["ky"]:
                bad += 1
print(f"  kx/ky identical to NOTRACK in {tot - bad}/{tot} (mismatch={bad})")
# and phase channel must leave w untouched, amplitude channel must leave phi untouched
for c in CH:
    nw = sum(1 for e in EPS for s in SEEDS if IDX[(c, e, s)]["w"] != IDX[("AUTO-NOTRACK", e, s)]["w"])
    npz = sum(1 for e in EPS for s in SEEDS if IDX[(c, e, s)]["phi"] != IDX[("AUTO-NOTRACK", e, s)]["phi"])
    print(f"  {c:<7} w moved in {nw:3d}/96   phi moved in {npz:3d}/96")

print()
print("=" * 100)
print("H3. THE MATCHING CONSTANT: median|w| vs the L2 statistic the field actually uses")
print("=" * 100)
meds, rmss = [], []
for e in EPS:
    for s in SEEDS:
        w = IDX[("AUTO-NOTRACK", e, s)]["w"]
        meds.append(st.median([abs(x) for x in w]))
        rmss.append(rms(w))
print(f"  median|w| over 96 NOTRACK runs : median={st.median(meds):.6f}  (PREREG froze 0.044095)")
print(f"  rms(w)    over 96 NOTRACK runs : median={st.median(rmss):.6f}")
print(f"  ratio rms/median = {st.median(rmss)/st.median(meds):.3f}")
print("  => phase field gain is L2-weighted (dominated by the big atoms); PREREG normalised")
print("     by a robust CENTRAL statistic.  The grid is offset by ~this factor.")

print()
print("=" * 100)
print("H4. MECHANISM: where does each channel put the error?  (decorr vs shrink)")
print("=" * 100)
print(f"{'cell':<14}{'m_mean':>10}{'decorr':>10}{'shrink':>10}{'absVR':>10}{'rho':>9}{'VR':>9}{'nRMSE':>9}")
for c in ["AUTO-NOTRACK"] + CH + ["AUTO"]:
    row = [st.median(list(block(c, k).values())) for k in
           ["m_mean", "decorr_mean", "shrink_mean", "absVR_framewise", "Pearson_rho", "Variance_Ratio", "nRMSE"]]
    print(f"{c:<14}" + "".join(f"{v:>10.5f}" for v in row[:4]) + f"{row[4]:>9.4f}{row[5]:>9.4f}{row[6]:>9.5f}")

print()
print("  paired deltas vs FLOOR (AUTO-NOTRACK):")
print(f"{'cell':<10}{'d_decorr':>12}{'d_shrink':>12}{'shrink share':>14}")
for c in CH:
    dd = med(paired(c, "AUTO-NOTRACK", "decorr_mean", EPS_PRIM))
    ds = med(paired(c, "AUTO-NOTRACK", "shrink_mean", EPS_PRIM))
    tot_ = abs(dd) + abs(ds)
    print(f"{c:<10}{dd:>12.5f}{ds:>12.5f}{(abs(ds)/tot_ if tot_ else 0):>13.1%}")

print()
print("=" * 100)
print("H5. IS THE PHASE ARM SATURATED WHILE THE AMPLITUDE ARM IS NOT?")
print("=" * 100)
for ch in "WP":
    print(f"  channel {ch}: lr x10 steps S5->S1")
    prev = None
    for i in "54321":
        c = f"{ch}-S{i}"
        nr = st.median(list(block(c, "nRMSE").values()))
        r = report(c, f"{ch}-S{int(i)+1}" if i != "5" else "AUTO-NOTRACK", eps=EPS_PRIM)
        print(f"    {c}: nRMSE={nr:.5f}   vs next-lower-lr cell: medD={r['medD']:+.5f} "
              f"{r['pos']}/{r['n']} p={r['p']:.4g}")

print()
print("=" * 100)
print("H6. NULL-CLEAN RESTRICTION of C(S)  (PREREG 4.5: null-failing cells not citable)")
print("=" * 100)
for i in "12345":
    W, P = f"W-S{i}", f"P-S{i}"
    bw, bp = block(W, "nRMSE"), block(P, "nRMSE")
    clean = [e for e in EPS_PRIM
             if bw[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"]
             and bp[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"]]
    r = report(W, P, eps=clean, label=f"C(S{i}) null-clean")
    print(fmt(r), f"   (kept {len(clean)}/31)")

print()
print("=" * 100)
print("H7. SURROGATE SUBSTITUTION (Rx-5's killer): replace a cell with the FLOOR")
print("=" * 100)
for i in "12345":
    W, P = f"W-S{i}", f"P-S{i}"
    real = paired(W, P, eps=EPS_PRIM)
    surW = paired("AUTO-NOTRACK", P, eps=EPS_PRIM)     # W -> floor
    surP = paired(W, "AUTO-NOTRACK", eps=EPS_PRIM)     # P -> floor

    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = math.sqrt(sum((x - ma) ** 2 for x in a))
        db = math.sqrt(sum((y - mb) ** 2 for y in b))
        return num / (da * db) if da * db else float("nan")
    nW, pW, ppW = sign_test(surW)
    nP, pP, ppP = sign_test(surP)
    print(f"  S{i}: real med={med(real):+.5f}    "
          f"[W:=floor] med={med(surW):+.5f} {pW}/{nW} p={ppW:.3g} r={corr(real,surW):+.3f}   "
          f"[P:=floor] med={med(surP):+.5f} {pP}/{nP} p={ppP:.3g} r={corr(real,surP):+.3f}")

print()
print("=" * 100)
print("H8. MULTIPLICITY: 5 S x 7 metrics = 35 pre-registered C(S) tests, BH + Bonferroni")
print("=" * 100)
MET = ["nRMSE", "Pearson_rho", "absVR_framewise", "m_mean", "decorr_mean", "shrink_mean", "Variance_Ratio"]
tests = []
for i in "12345":
    for m in MET:
        r = report(f"W-S{i}", f"P-S{i}", metric=m, eps=EPS_PRIM, label=f"C(S{i})/{m}")
        tests.append(r)
tests_sorted = sorted(tests, key=lambda r: r["p"])
M = len(tests)
print(f"  M={M} tests. Bonferroni alpha = {0.05/M:.5f}")
crit = 0.0
for rank, r in enumerate(tests_sorted, 1):
    if r["p"] <= 0.05 * rank / M:
        crit = 0.05 * rank / M
for rank, r in enumerate(tests_sorted[:12], 1):
    bh = "BH-PASS" if r["p"] <= crit else "bh-fail"
    bf = "BONF-PASS" if r["p"] <= 0.05 / M else "bonf-fail"
    print(f"   {rank:2d} {r['label']:<26} medD={r['medD']:+.6f} {r['pos']:2d}/{r['n']} "
          f"p={r['p']:.3g}   {bh}  {bf}")

print()
print("=" * 100)
print("H9. SEED VARIABILITY vs EFFECT SIZE")
print("=" * 100)
for i in ["1", "4"]:
    print(f"  C(S{i}):")
    vals = []
    for s in SEEDS:
        d = [IDX[(f"W-S{i}", e, s)]["nRMSE"] - IDX[(f"P-S{i}", e, s)]["nRMSE"] for e in EPS_PRIM]
        n, pos, p = sign_test(d)
        vals.append(med(d))
        print(f"    seed {s} alone: med={med(d):+.5f} {pos}/{n} p={p:.4g}")
    print(f"    spread across single seeds = {max(vals)-min(vals):+.5f}  "
          f"(median-of-3 value = {med(paired(f'W-S{i}', f'P-S{i}', eps=EPS_PRIM)):+.5f})")
# per-episode seed noise on the SAME cell
noise = []
for c in ["AUTO", "W-S1", "P-S1"]:
    per = []
    for e in EPS:
        v = [IDX[(c, e, s)]["nRMSE"] for s in SEEDS]
        per.append(max(v) - min(v))
    print(f"  within-cell seed range of nRMSE, {c}: median={st.median(per):.5f} max={max(per):.5f}")
# leave-one-out on C(S1)
d1 = paired("W-S1", "P-S1", eps=EPS_PRIM)
loo_fail = 0
for j in range(len(d1)):
    dd = d1[:j] + d1[j + 1:]
    n, pos, p = sign_test(dd)
    need = {31: 22, 30: 21, 29: 21}.get(n, 99)
    if max(pos, n - pos) < need:
        loo_fail += 1
print(f"  LOO on C(S1): {loo_fail}/{len(d1)} deletions drop it below the frozen threshold")
d4 = paired("W-S4", "P-S4", eps=EPS_PRIM)
loo_fail4 = sum(1 for j in range(len(d4))
                if max(sign_test(d4[:j] + d4[j + 1:])[1],
                       sign_test(d4[:j] + d4[j + 1:])[0] - sign_test(d4[:j] + d4[j + 1:])[1])
                < {31: 22, 30: 21}.get(sign_test(d4[:j] + d4[j + 1:])[0], 99))
print(f"  LOO on C(S4): {loo_fail4}/{len(d4)} deletions drop it below threshold")

print()
print("=" * 100)
print("H10. POWER / MDE for the frozen sign test (n=31, need 22)")
print("=" * 100)


def power(n, need, p):
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(need, n + 1))


for pw in [0.5, 0.6, 0.65, 0.70, 0.71, 0.75, 0.8]:
    print(f"   true win-rate {pw:.2f} -> power(one-sided,n=31,need 22) = {power(31,22,pw):.3f}")
absd = sorted(abs(x) for x in d1)
print(f"   |paired delta| median = {st.median(absd):.5f}")
