import sys, math, statistics as st, random
sys.path.insert(0, "/home/vexit/wavesplat/wavesplat/tmp/Rx6_channel_curve_2026-08-07")
from audit_con_load import *

TWOPI = 2 * math.pi
wrap = lambda x: (x + math.pi) % TWOPI - math.pi
rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
AUTOb = block("AUTO", "nRMSE")
EPS_PRIM = [e for e in EPS if AUTOb[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"]]
CH = [f"{c}-S{i}" for c in "WP" for i in "12345"]

print("=" * 100)
print("J1. DIRECT NUMERIC FIELD DISPLACEMENT rms_xy( z_cell - z_NOTRACK )  (no linearisation,")
print("    no incoherence assumption).  Grid 81x61 over +-100 m, t=0, dispersion omega.")
print("=" * 100)
G = 9.81
DEPTH = 20.0  # deep-water regime for these k; tanh(kd)->1 anyway, and omega cancels at t=0
XS = [-100 + 200 * i / 80 for i in range(81)]
YS = [-75 + 150 * i / 60 for i in range(61)]
PTS = [(x, y) for x in XS[::2] for y in YS[::2]]


def field_disp(cell, ep, seed):
    r = IDX[(cell, ep, seed)]
    b = IDX[("AUTO-NOTRACK", ep, seed)]
    kx, ky = b["kx"], b["ky"]
    w0, p0, w1, p1 = b["w"], b["phi"], r["w"], r["phi"]
    n = len(w0)
    acc = 0.0
    accb = 0.0
    for (x, y) in PTS:
        d = 0.0
        z0 = 0.0
        for i in range(n):
            th = kx[i] * x + ky[i] * y
            c0 = math.cos(th + p0[i])
            c1 = math.cos(th + p1[i])
            d += w1[i] * c1 - w0[i] * c0
            z0 += w0[i] * c0
        acc += d * d
        accb += z0 * z0
    return math.sqrt(acc / len(PTS)), math.sqrt(accb / len(PTS))


print(f"{'cell':<8}{'rms(dz)[m]':>13}{'rms(z0)[m]':>13}{'dz/z0':>9}{'D_exact(anal)':>15}{'num/anal':>10}")
NUM = {}
random.seed(0)
SUB = EPS[::2]  # 16 episodes, one seed, to keep it cheap but unbiased by config
for c in CH:
    vals, base = [], []
    for ep in SUB:
        a, b = field_disp(c, ep, SEEDS[0])
        vals.append(a); base.append(b)
    v = st.median(vals)
    NUM[c] = v
    # analytic exact on the same subset
    an = []
    for ep in SUB:
        r = IDX[(c, ep, SEEDS[0])]; bb = IDX[("AUTO-NOTRACK", ep, SEEDS[0])]
        n = len(bb["w"])
        if c.startswith("W"):
            an.append(rms([r["w"][i] - bb["w"][i] for i in range(n)]) / math.sqrt(2))
        else:
            an.append(rms([2 * bb["w"][i] * math.sin(wrap(r["phi"][i] - bb["phi"][i]) / 2)
                           for i in range(n)]) / math.sqrt(2))
    a_ = st.median(an)
    print(f"{c:<8}{v:>13.5f}{st.median(base):>13.5f}{v/st.median(base):>9.3f}{a_:>15.5f}{v/a_:>10.3f}")

print()
print("  DIRECT-FIELD matching ratio W/P per S:")
for i in "12345":
    print(f"    S{i}: rms(dz) W={NUM[f'W-S{i}']:.5f}  P={NUM[f'P-S{i}']:.5f}  W/P={NUM[f'W-S{i}']/NUM[f'P-S{i}']:.3f}")

print()
print("=" * 100)
print("J2. D RATIO computed the way PREREG sec.4.2 ALLOWS (paired ratio, then median)")
print("    -- 'ratio of aggregates' is explicitly forbidden by the PREREG itself.")
print("=" * 100)


def Dcell(c, ep, seed):
    r = IDX[(c, ep, seed)]; b = IDX[("AUTO-NOTRACK", ep, seed)]
    n = len(b["w"])
    if c.startswith("W"):
        return rms([r["w"][i] - b["w"][i] for i in range(n)])
    return rms([b["w"][i] * wrap(r["phi"][i] - b["phi"][i]) for i in range(n)])


for i in "12345":
    per_ep = []
    for ep in EPS:
        rr = [Dcell(f"W-S{i}", ep, s) / Dcell(f"P-S{i}", ep, s) for s in SEEDS]
        per_ep.append(st.median(rr))
    ratio_of_agg = (st.median([st.median([Dcell(f"W-S{i}", ep, s) for s in SEEDS]) for ep in EPS]) /
                    st.median([st.median([Dcell(f"P-S{i}", ep, s) for s in SEEDS]) for ep in EPS]))
    print(f"  S{i}: median of PAIRED ratio W/P = {st.median(per_ep):.3f}   "
          f"(P/W = {1/st.median(per_ep):.3f})   | ratio-of-medians = {ratio_of_agg:.3f} "
          f"(P/W={1/ratio_of_agg:.3f})  [forbidden form]")

print()
print("  predicted mismatch from the median-vs-rms normalisation error:")
rr = []
for ep in EPS:
    for s in SEEDS:
        w = IDX[("AUTO-NOTRACK", ep, s)]["w"]
        rr.append(rms(w) / st.median([abs(x) for x in w]))
print(f"    median over 96 runs of  rms(w)/median|w| = {st.median(rr):.3f}")
print("    (this is exactly the factor by which D_phi/D_w overshoots in the UNSATURATED rows)")

print()
print("=" * 100)
print("J3. CAN THE FLOOR WIN BY PREDICTING LESS?  (zero-predictor / shrinkage check)")
print("=" * 100)
for c in ["AUTO", "AUTO-NOTRACK"]:
    row = {k: st.median(list(block(c, k).values())) for k in
           ["nRMSE", "Variance_Ratio", "Pearson_rho", "m_mean", "decorr_mean", "shrink_mean",
            "absVR_framewise", "Skill_Score", "Hs_error", "avg_RMSE_unobserved"]}
    print(f"  {c:<14}" + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))
print("  paired NOTRACK - AUTO on every panel metric:")
for m in ["nRMSE", "Pearson_rho", "Variance_Ratio", "absVR_framewise", "m_mean", "decorr_mean",
          "shrink_mean", "Skill_Score", "Hs_error", "avg_RMSE_unobserved", "Directional_R2"]:
    r = report("AUTO-NOTRACK", "AUTO", metric=m, eps=EPS_PRIM, label=f"NOTRACK-AUTO/{m}")
    print("   " + fmt(r))
# how many episodes does NOTRACK beat the flat-sea null vs AUTO
nn = sum(1 for e in EPS if block("AUTO-NOTRACK", "nRMSE")[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"])
na = sum(1 for e in EPS if AUTOb[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"])
print(f"  beats flat-sea null: NOTRACK {nn}/32  AUTO {na}/32")

print()
print("=" * 100)
print("J4. DEGENERACY LEDGER: which of the 10 grid cells are distinguishable from BOTH ends?")
print("=" * 100)
NEED = {31: 22, 30: 21, 32: 22}
info = []
for c in CH:
    rA = report(c, "AUTO", eps=EPS_PRIM)
    rB = report(c, "AUTO-NOTRACK", eps=EPS_PRIM)
    okA = max(rA["pos"], rA["neg"]) >= NEED[rA["n"]]
    okB = max(rB["pos"], rB["neg"]) >= NEED[rB["n"]]
    tag = "INFORMATIVE" if (okA and okB) else ("== AUTO" if not okA else "== FLOOR")
    if okA and okB and abs(rB["medD"]) < 0.002:
        tag = "== FLOOR (trivial size)"
    info.append((c, tag))
    print(f"  {c:<7} vsAUTO {'sep' if okA else 'DEGEN':<5} (p={rA['p']:.3g}, d={rA['medD']:+.5f})   "
          f"vsFLOOR {'sep' if okB else 'DEGEN':<5} (p={rB['p']:.3g}, d={rB['medD']:+.5f})   -> {tag}")
ni = sum(1 for _, t in info if t == "INFORMATIVE")
print(f"  ==> INFORMATIVE cells: {ni}/10")
print("  ==> rows with BOTH cells informative:",
      [i for i in "12345" if dict(info)[f"W-S{i}"] == "INFORMATIVE" and dict(info)[f"P-S{i}"] == "INFORMATIVE"] or "NONE")
