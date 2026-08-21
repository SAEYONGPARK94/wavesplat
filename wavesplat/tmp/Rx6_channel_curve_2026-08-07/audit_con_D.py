import sys, math, statistics as st
sys.path.insert(0, "/home/vexit/wavesplat/wavesplat/tmp/Rx6_channel_curve_2026-08-07")
from audit_con_load import *

TWOPI = 2 * math.pi


def wrap(x):
    return (x + math.pi) % TWOPI - math.pi


def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v))


def Dpair(cell, ep, seed):
    """returns (D_w_lin, D_phi_lin, D_w_exactRMS, D_phi_exactRMS, maxabsdphi, relwdrift)"""
    r = IDX[(cell, ep, seed)]
    b = IDX[("AUTO-NOTRACK", ep, seed)]
    w0 = b["w"]; p0 = b["phi"]
    w1 = r["w"]; p1 = r["phi"]
    n = len(w0)
    dw = [w1[i] - w0[i] for i in range(n)]
    dp = [wrap(p1[i] - p0[i]) for i in range(n)]
    Dw = rms(dw)
    Dp = rms([w0[i] * dp[i] for i in range(n)])
    # exact field-RMS of the per-component perturbation (over the phase argument):
    #   amplitude: dw*cos(t)                 -> rms_t = |dw|/sqrt2
    #   phase:     w0*(cos(t+dp)-cos(t))     -> rms_t = |2*w0*sin(dp/2)|/sqrt2
    Ew = rms(dw) / math.sqrt(2)
    Ep = rms([2 * w0[i] * math.sin(dp[i] / 2) for i in range(n)]) / math.sqrt(2)
    mx = max(abs(x) for x in dp)
    relw = rms(dw) / rms(w0)
    return Dw, Dp, Ew, Ep, mx, relw


CH = [f"{c}-S{i}" for c in "WP" for i in "12345"]
print("=" * 108)
print("F. REALIZED FIELD DISPLACEMENT  D  (block median of seed-median), PREREG 1.1 + my exact version")
print("=" * 108)
print(f"{'cell':<8}{'D_lin[m]':>11}{'D_exact[m]':>12}{'exact/lin':>11}{'rms|dphi|rad':>14}"
      f"{'max|dphi|':>11}{'rel_w_drift':>13}{'nRMSE':>10}")
DTAB = {}
for c in CH:
    rows = []
    for ep in EPS:
        per = [Dpair(c, ep, s) for s in SEEDS]
        rows.append([st.median([p[i] for p in per]) for i in range(6)])
    agg = [st.median([r[i] for r in rows]) for i in range(6)]
    lin = agg[0] if c.startswith("W") else agg[1]
    ex = agg[2] if c.startswith("W") else agg[3]
    # rms|dphi| for phase cells
    dphis = []
    for ep in EPS:
        per = []
        for s in SEEDS:
            b = IDX[("AUTO-NOTRACK", ep, s)]
            r = IDX[(c, ep, s)]
            per.append(rms([wrap(r["phi"][i] - b["phi"][i]) for i in range(len(b["phi"]))]))
        dphis.append(st.median(per))
    nrm = st.median(list(block(c, "nRMSE").values()))
    DTAB[c] = dict(lin=lin, ex=ex, nrmse=nrm)
    print(f"{c:<8}{lin:>11.5f}{ex:>12.5f}{ex/lin:>11.3f}{st.median(dphis):>14.4f}"
          f"{agg[4]:>11.4f}{agg[5]:>13.4f}{nrm:>10.5f}")

print()
print("PREREG matching check: ratio of D_lin within each S (W/P) and my exact ratio")
for i in "12345":
    W, P = DTAB[f"W-S{i}"], DTAB[f"P-S{i}"]
    print(f"  S{i}: D_lin  W={W['lin']:.5f} P={P['lin']:.5f} ratio W/P={W['lin']/P['lin']:.3f}"
          f"   | D_exact W={W['ex']:.5f} P={P['ex']:.5f} ratio W/P={W['ex']/P['ex']:.3f}")

print()
print("=" * 108)
print("G. SAME-CURVE TEST — is nRMSE a single function of realized displacement D?")
print("=" * 108)
# Per episode: build the 5-point curve for each channel in (D_exact, nRMSE) and ask,
# at the D realized by the OTHER channel's cell, what does this channel's curve predict?


def curve(ch, ep):
    pts = []
    for i in "12345":
        c = f"{ch}-S{i}"
        per = [Dpair(c, ep, s) for s in SEEDS]
        ex = st.median([p[2] if ch == "W" else p[3] for p in per])
        nr = st.median([IDX[(c, ep, s)]["nRMSE"] for s in SEEDS])
        pts.append((ex, nr))
    pts.sort()
    return pts


def interp(pts, x):
    xs = [p[0] for p in pts]
    if x <= xs[0]:
        return pts[0][1]
    if x >= xs[-1]:
        return pts[-1][1]
    for a, b in zip(pts, pts[1:]):
        if a[0] <= x <= b[0]:
            if b[0] == a[0]:
                return a[1]
            t = (x - a[0]) / (b[0] - a[0])
            return a[1] + t * (b[1] - a[1])
    return pts[-1][1]


AUTOb = block("AUTO", "nRMSE")
EPS_PRIM = [e for e in EPS if AUTOb[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"]]

for i in "12345":
    resid = []
    extrap = 0
    for ep in EPS_PRIM:
        cw, cp = curve("W", ep), curve("P", ep)
        perW = [Dpair(f"W-S{i}", ep, s) for s in SEEDS]
        Dw = st.median([p[2] for p in perW])
        nW = st.median([IDX[(f"W-S{i}", ep, s)]["nRMSE"] for s in SEEDS])
        pred = interp(cp, Dw)          # what the PHASE curve says at this same D
        if Dw > max(p[0] for p in cp) or Dw < min(p[0] for p in cp):
            extrap += 1
        resid.append(nW - pred)
    n, pos, p = sign_test(resid)
    print(f"  S{i}: nRMSE(W-S{i}) - PHASE-curve@sameD  med={st.median(resid):+.5f} "
          f"pos={pos}/{n} p={p:.4g}   (extrapolated eps={extrap}/{len(EPS_PRIM)})")

print()
print("  reverse direction (phase point vs amplitude curve at same D):")
for i in "12345":
    resid = []
    extrap = 0
    for ep in EPS_PRIM:
        cw = curve("W", ep)
        perP = [Dpair(f"P-S{i}", ep, s) for s in SEEDS]
        Dp = st.median([p[3] for p in perP])
        nP = st.median([IDX[(f"P-S{i}", ep, s)]["nRMSE"] for s in SEEDS])
        pred = interp(cw, Dp)
        if Dp > max(p[0] for p in cw) or Dp < min(p[0] for p in cw):
            extrap += 1
        resid.append(nP - pred)
    n, pos, p = sign_test(resid)
    print(f"  S{i}: nRMSE(P-S{i}) - AMPL-curve@sameD   med={st.median(resid):+.5f} "
          f"pos={pos}/{n} p={p:.4g}   (extrapolated eps={extrap}/{len(EPS_PRIM)})")
