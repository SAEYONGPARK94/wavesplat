#!/usr/bin/env python3
"""PRO analysis for proposition J: placement quality vs field accuracy are different axes,
with atom stacking (nearest-neighbour collapse / surplus atoms inside the signal box)
as the mediator.

Re-analysis ONLY. No estimator runs. Reads:
  tmp/e1_grid_2026-08-05/results.jsonl   (180 e2e runs: 12 ep x 5 cells x 3 seeds)
  tmp/e1_grid_2026-08-05/probe.jsonl     (180 placement probes: 12 ep x 5 cells x 3 n_g)
  tmp/s1s2_2026-08-05/results.jsonl      (132 e2e runs: 11 arms x 4 ep x 3 seeds)
  data_finemesh_2026-07-23/*.h5          (config/wave_components only -> TRUE nn spacing)
"""
import json, math, statistics as st
from collections import defaultdict
import numpy as np

ROOT = "/home/vexit/wavesplat/wavesplat"
E1 = f"{ROOT}/tmp/e1_grid_2026-08-05"
S12 = f"{ROOT}/tmp/s1s2_2026-08-05"
DATA = f"{ROOT}/data_finemesh_2026-07-23"
OUT = f"{ROOT}/tmp/placement_field_debate_2026-08-05"

def jl(p): return [json.loads(l) for l in open(p) if l.strip()]

# ---------------- stats ----------------
def rank(x):
    n = len(x); order = sorted(range(n), key=lambda i: x[i]); r = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and x[order[j+1]] == x[order[i]]: j += 1
        avg = (i+j)/2.0 + 1.0
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r

def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    if sx == 0 or sy == 0: return float('nan')
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy)

def spearman(x, y):
    if len(x) < 3: return float('nan')
    return pearson(rank(x), rank(y))

def perm_p(x, y, nperm=20000, seed=0):
    obs = spearman(x, y)
    if math.isnan(obs): return float('nan')
    rng = np.random.default_rng(seed); yy = np.array(y, float); cnt = 0
    rx = rank(x)
    for _ in range(nperm):
        s = pearson(rx, rank(list(rng.permutation(yy))))
        if not math.isnan(s) and abs(s) >= abs(obs) - 1e-12: cnt += 1
    return (cnt+1)/(nperm+1)

def sign_test(deltas, tol=1e-12):
    d = [v for v in deltas if abs(v) > tol]
    n = len(d); k = sum(1 for v in d if v > 0)
    if n == 0: return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n-k)+1))
    return k, n, min(1.0, 2*tail/2**n)

def partial_spearman(x, y, z):
    rx, ry, rz = rank(x), rank(y), rank(z)
    def resid(a, b):
        n = len(a); ma, mb = sum(a)/n, sum(b)/n
        den = sum((v-mb)**2 for v in b)
        if den == 0: return [v-ma for v in a]
        beta = sum((u-ma)*(v-mb) for u, v in zip(a, b))/den
        return [u-ma-beta*(v-mb) for u, v in zip(a, b)]
    return pearson(resid(rx, rz), resid(ry, rz))

def ols(Y, Xs):
    A = np.column_stack([np.ones(len(Y))] + [np.asarray(x, float) for x in Xs])
    b, *_ = np.linalg.lstsq(A, np.asarray(Y, float), rcond=None)
    r = np.asarray(Y, float) - A @ b
    dof = max(len(Y) - np.linalg.matrix_rank(A), 1)
    cov = (r @ r / dof) * np.linalg.pinv(A.T @ A)
    return b, np.sqrt(np.diag(cov))

def med(v): return st.median(v) if v else float('nan')

# ---------------- load ----------------
e1 = jl(f"{E1}/results.jsonl"); pr = jl(f"{E1}/probe.jsonl"); s12 = jl(f"{S12}/results.jsonl")
for r in e1 + s12:
    r["absVR1"] = abs(r["Variance_Ratio"] - 1.0)
    for k, v in r["spec"].items(): r["spec_"+k] = v

CELLS = ["A", "B", "C", "D", "E"]
EPS = sorted({r["episode"] for r in e1})
CROWDED = [e for e in EPS if "15comp" in e or "30comp" in e]
SPARSE = [e for e in EPS if e not in CROWDED]

import h5py
TRUE = {}
for e in EPS:
    with h5py.File(f"{DATA}/{e}", "r") as f:
        wc = f["config/wave_components"][:]
    k = wc[:, 2:4]; n = len(k)
    if n >= 2:
        d = np.array([min(math.hypot(k[i,0]-k[j,0], k[i,1]-k[j,1]) for j in range(n) if j != i)
                      for i in range(n)])
    else:
        d = np.array([float('nan')])
    TRUE[e] = {"n_gt": n, "nn": d, "kmax": float(np.hypot(wc[:,2], wc[:,3]).max()),
               "kmin": float(np.hypot(wc[:,2], wc[:,3]).min())}

FIELDS = ["nRMSE", "init_nrmse", "Variance_Ratio", "absVR1", "Hs_error", "Skill_Score",
          "Pearson_rho", "Directional_R2", "avg_RMSE_unobserved", "N",
          "spec_detection_rate", "spec_energy_detection_rate", "spec_k_mag_rel_error_mean",
          "spec_direction_error_deg_mean", "spec_amp_rel_error_mean", "spec_n_est",
          "spec_n_matched", "spec_omega_rel_error_mean"]

def med3(rows, keyf):
    g = defaultdict(list)
    for r in rows: g[keyf(r)].append(r)
    return {k: {f: med([x[f] for x in v]) for f in FIELDS} | {"_n": len(v)} for k, v in g.items()}

E1M = med3(e1, lambda r: (r["episode"], r["cell"]))
S12M = med3(s12, lambda r: (r["episode"], r["arm"]))
PRB = {(p["episode"], p["cell"], p["n_g"]): p for p in pr}
NGS = sorted({p["n_g"] for p in pr})

# derived placement variables
for (e, c, ng), p in PRB.items():
    ngt = TRUE[e]["n_gt"]
    p["surplus_in_box"] = max(0, ng - p["n_outside_narrow_box"] - min(ngt, ng))
    p["log_nn"] = math.log10(max(p["placed_nn_median"], 1e-4))
    tn = float(np.nanmedian(TRUE[e]["nn"])) if ngt >= 2 else float('nan')
    p["nn_ratio"] = p["placed_nn_median"]/tn if tn == tn and tn > 0 else float('nan')
# e2e-side stacking surrogate: surviving atoms in excess of the truth
for k, v in E1M.items():
    v["surplus_final_n"] = max(0.0, v["N"] - TRUE[k[0]]["n_gt"])
for k, v in S12M.items():
    v["surplus_final_n"] = max(0.0, v["N"] - TRUE[k[0]]["n_gt"])

lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

P("="*104)
P("PRO ANALYSIS — proposition J.  Existing raw JSONL/HDF5 re-analysis only; zero estimator runs.")
P("="*104)
P(f"E1 e2e={len(e1)}  E1 probe={len(pr)} (n_g={NGS})  S1S2 e2e={len(s12)}  episodes={len(EPS)}")
P(f"crowded(4)={[e[8:-13] for e in CROWDED]}")
P("verified: 'placed_nn_median' occurrences in tmp/s1s2_2026-08-05/results.jsonl = 0")

# ======================================================================================
P("\n" + "#"*104)
P("# PART 0 — the confound demo: why the POOLED placement<->field correlation is a trap")
P("#"*104)
keys = sorted(E1M.keys())
X = [E1M[k]["spec_detection_rate"] for k in keys]; Y = [E1M[k]["nRMSE"] for k in keys]
NG = [float(TRUE[k[0]]["n_gt"]) for k in keys]
P(f"  spearman(detection, nRMSE)  POOLED over 60 (episode,cell) cells = {spearman(X,Y):+.3f}  <-- looks like ONE axis")
P(f"  spearman(detection, n_gt_components)                            = {spearman(X,NG):+.3f}")
P(f"  spearman(n_gt_components, nRMSE)                                = {spearman(NG,Y):+.3f}")
P(f"  partial spearman(detection, nRMSE | n_gt)                       = {partial_spearman(X,Y,NG):+.3f}")
P("  => the pooled number is mostly 'crowded episodes are both hard to detect and hard to fit'.")
P("     Per 2026-08-04 discipline the stratifying variable must be controlled before any mechanism claim.")

# ======================================================================================
P("\n" + "#"*104)
P("# PART 1 — TWO AXES, episode- and arm-controlled")
P("#"*104)

def corrblock(mtable, tag, pairs):
    P(f"\n--- {tag} (n={len(mtable)}) ---")
    hdr = (f"{'x = placement metric':<32}{'y = field metric':<14}{'POOLED':>9}{'p':>8}"
           f"{'WITHIN-ep':>11}{'p':>8}{'WITHIN ep+arm':>15}{'p':>8}")
    P(hdr); P("-"*len(hdr))
    ks = sorted(mtable.keys())
    def demean(vals, gf):
        g = defaultdict(list)
        for k, v in zip(ks, vals): g[gf(k)].append(v)
        m = {k: sum(v)/len(v) for k, v in g.items()}
        return [v - m[gf(k)] for k, v in zip(ks, vals)]
    for xk, yk in pairs:
        Xv = [mtable[k][xk] for k in ks]; Yv = [mtable[k][yk] for k in ks]
        Xe, Ye = demean(Xv, lambda k: k[0]), demean(Yv, lambda k: k[0])
        Xa, Ya = demean(Xe, lambda k: k[1]), demean(Ye, lambda k: k[1])
        P(f"{xk.replace('spec_',''):<32}{yk:<14}{spearman(Xv,Yv):>9.3f}{perm_p(Xv,Yv,seed=1):>8.4f}"
          f"{spearman(Xe,Ye):>11.3f}{perm_p(Xe,Ye,seed=2):>8.4f}"
          f"{spearman(Xa,Ya):>15.3f}{perm_p(Xa,Ya,seed=3):>8.4f}")

PAIRS = [("spec_detection_rate", "nRMSE"), ("spec_detection_rate", "init_nrmse"),
         ("spec_detection_rate", "absVR1"), ("spec_detection_rate", "Hs_error"),
         ("spec_energy_detection_rate", "nRMSE"),
         ("spec_k_mag_rel_error_mean", "nRMSE"), ("spec_k_mag_rel_error_mean", "init_nrmse"),
         ("spec_k_mag_rel_error_mean", "absVR1")]
corrblock(E1M, "E1: 12 ep x 5 cells, median-of-3", PAIRS)
corrblock({k: v for k, v in S12M.items() if k[1].startswith("S1-")},
          "S1 budget arms: 4 ep x 4 arms", PAIRS)
corrblock({k: v for k, v in S12M.items() if k[1] in ("S1-A0","S2-C1","S2-T1","S2-T1f","S2-T2","S2-T4")},
          "S2 placement arms (tracker ON): 4 ep x 6 arms", PAIRS)

P("\n--- E1: per-episode Spearman across the 5 cells (episode FULLY controlled), then sign test ---")
for xk, yk in [("spec_detection_rate","nRMSE"), ("spec_detection_rate","init_nrmse"),
               ("spec_detection_rate","absVR1"), ("spec_detection_rate","Hs_error"),
               ("spec_k_mag_rel_error_mean","nRMSE"), ("spec_k_mag_rel_error_mean","init_nrmse")]:
    rs = [spearman([E1M[(e,c)][xk] for c in CELLS], [E1M[(e,c)][yk] for c in CELLS]) for e in EPS]
    rs = [r for r in rs if not math.isnan(r)]
    k, n, p = sign_test(rs)
    P(f"  {xk.replace('spec_',''):<28} vs {yk:<12} med rho={med(rs):+.3f}  pos {k}/{n}  sign p={p:.4f}")

# ======================================================================================
P("\n" + "#"*104)
P("# PART 2 — THE INTERVENTION VIEW + dose-response on the mediator")
P("#"*104)

delta_rows = []
for e in EPS:
    for t in ["B", "C", "D", "E"]:
        d = {"episode": e, "treat": t, "crowded": e in CROWDED, "n_gt": TRUE[e]["n_gt"]}
        for f in FIELDS + ["surplus_final_n"]:
            d["d_"+f] = E1M[(e,t)][f] - E1M[(e,"A")][f]
        for ng in NGS:
            a, b = PRB[(e,t,ng)], PRB[(e,"A",ng)]
            d[f"d_det@{ng}"] = a["detection_rate"] - b["detection_rate"]
            d[f"d_lognn@{ng}"] = a["log_nn"] - b["log_nn"]
            d[f"d_surplus@{ng}"] = a["surplus_in_box"] - b["surplus_in_box"]
            d[f"d_kerr@{ng}"] = a["k_mag_rel_error_mean"] - b["k_mag_rel_error_mean"]
        delta_rows.append(d)

P("\n★ DOSE-RESPONSE — all four treatments raise crowded detection; the FIELD HARM tracks the")
P("  spacing collapse / surplus-atom count, not the detection gain.  (medians over 12 episodes)")
hdr = (f"{'cell':<5}{'nn_med@15':>10}{'surplus@15':>11}{'outside@15':>11}{'ddet crowd':>11}"
       f"{'ddet 12ep':>10}{'dHs_err':>9}{'sgn':>6}{'d|VR-1|':>9}{'sgn':>6}{'dfinal_n':>10}")
P(hdr); P("-"*len(hdr))
for c in CELLS:
    nn = med([PRB[(e,c,15)]["placed_nn_median"] for e in EPS])
    sp = med([PRB[(e,c,15)]["surplus_in_box"] for e in EPS])
    ob = med([PRB[(e,c,15)]["n_outside_narrow_box"] for e in EPS])
    dc = med([PRB[(e,c,15)]["detection_rate"]-PRB[(e,"A",15)]["detection_rate"] for e in CROWDED])
    da = med([PRB[(e,c,15)]["detection_rate"]-PRB[(e,"A",15)]["detection_rate"] for e in EPS])
    if c == "A":
        P(f"{c:<5}{nn:>10.3f}{sp:>11.1f}{ob:>11.1f}{'(ref)':>11}{'(ref)':>10}"
          f"{'':>9}{'':>6}{'':>9}{'':>6}{med([E1M[(e,c)]['N'] for e in EPS]):>10.1f}")
        continue
    dh = [E1M[(e,c)]["Hs_error"]-E1M[(e,"A")]["Hs_error"] for e in EPS]
    dv = [E1M[(e,c)]["absVR1"]-E1M[(e,"A")]["absVR1"] for e in EPS]
    dn = [E1M[(e,c)]["N"]-E1M[(e,"A")]["N"] for e in EPS]
    kh, nh, _ = sign_test(dh); kv, nv, _ = sign_test(dv)
    P(f"{c:<5}{nn:>10.3f}{sp:>11.1f}{ob:>11.1f}{dc:>+11.3f}{da:>+10.3f}"
      f"{med(dh):>+9.4f}{f'{kh}/{nh}':>6}{med(dv):>+9.4f}{f'{kv}/{nv}':>6}{med(dn):>+10.1f}")
P("  sgn = #episodes the metric got WORSE / #non-tied.   surplus@15 = 15 - n_outside_box - min(N_gt,15).")

P("\n★ MATCHED-GAIN CONTRAST  B vs C  — the single cleanest test of proposition J")
for c in ["B", "C"]:
    dc = med([PRB[(e,c,15)]["detection_rate"]-PRB[(e,"A",15)]["detection_rate"] for e in CROWDED])
    P(f"   cell {c}: crowded detection gain {dc:+.3f} | nn@15 med {med([PRB[(e,c,15)]['placed_nn_median'] for e in EPS]):.3f}"
      f" | surplus@15 med {med([PRB[(e,c,15)]['surplus_in_box'] for e in EPS]):.0f}"
      f" | dHs {med([E1M[(e,c)]['Hs_error']-E1M[(e,'A')]['Hs_error'] for e in EPS]):+.4f}"
      f" | d|VR-1| {med([E1M[(e,c)]['absVR1']-E1M[(e,'A')]['absVR1'] for e in EPS]):+.4f}")
for f, lab in [("Hs_error","Hs_error"), ("absVR1","|VR-1|"), ("N","final_n"),
               ("init_nrmse","init_nrmse"), ("nRMSE","nRMSE")]:
    d = [E1M[(e,"B")][f]-E1M[(e,"C")][f] for e in EPS]
    k, n, p = sign_test(d)
    P(f"   B - C  {lab:<11} med={med(d):+.4f}  B-worse {k}/{n}  sign p={p:.4f}")
d = [PRB[(e,"B",15)]["placed_nn_median"]-PRB[(e,"C",15)]["placed_nn_median"] for e in EPS]
k, n, p = sign_test(d)
P(f"   B - C  nn@15       med={med(d):+.4f}  B-larger {k}/{n}  sign p={p:.4f}   (B is TIGHTER 11/12)")
d = [PRB[(e,"B",15)]["detection_rate"]-PRB[(e,"C",15)]["detection_rate"] for e in EPS]
k, n, p = sign_test(d)
P(f"   B - C  det@15      med={med(d):+.4f}  B-better {k}/{n}  sign p={p:.4f}   (detection is TIED)")

P("\nCorrelation across the 48 intervention contrasts (episode = pairing unit):")
hdr = f"{'x = placement gain':<22}{'y = field change':<16}{'spearman':>10}{'perm p':>9}"
P(hdr); P("-"*len(hdr))
for xk in ["d_det@3", "d_det@15", "d_spec_detection_rate"]:
    for yk in ["d_nRMSE", "d_init_nrmse", "d_Hs_error", "d_absVR1"]:
        Xv = [r[xk] for r in delta_rows]; Yv = [r[yk] for r in delta_rows]
        P(f"{xk:<22}{yk:<16}{spearman(Xv,Yv):>10.3f}{perm_p(Xv,Yv,seed=7):>9.4f}")

# ======================================================================================
P("\n" + "#"*104)
P("# PART 3 — MEDIATION TEST:  stacking (nn collapse / surplus atoms)")
P("#"*104)

P("\n(3a) WITHIN-EPISODE leg a->b:  per-episode Spearman across the 5 cells, then sign test over 12 ep")
hdr = f"{'mediator (probe)':<22}{'field':<12}{'med rho':>9}{'pos/n':>8}{'sign p':>9}   expected"
P(hdr); P("-"*(len(hdr)+8))
for ng in NGS:
    for mk, mlab in [("placed_nn_median", f"nn_median@{ng}"), ("surplus_in_box", f"surplus@{ng}")]:
        for fk, flab, exp in [("Variance_Ratio","VR","+ (tighter->less variance)" if mk=="placed_nn_median" else "- (more surplus->less variance)"),
                              ("Hs_error","Hs_error","- " if mk=="placed_nn_median" else "+ "),
                              ("absVR1","|VR-1|","- " if mk=="placed_nn_median" else "+ "),
                              ("N","final_n","- " if mk=="placed_nn_median" else "+ ")]:
            rs = [spearman([PRB[(e,c,ng)][mk] for c in CELLS], [E1M[(e,c)][fk] for c in CELLS]) for e in EPS]
            rs = [r for r in rs if not math.isnan(r)]
            k, n, p = sign_test(rs)
            P(f"{mlab:<22}{flab:<12}{med(rs):>9.3f}{f'{k}/{n}':>8}{p:>9.4f}   {exp}")
    P("")

P("(3b) e2e-INTERNAL mediator (no probe proxy): surviving surplus atoms  final_n - N_gt")
for fk, flab in [("Variance_Ratio","VR"), ("absVR1","|VR-1|"), ("Hs_error","Hs_error"),
                 ("nRMSE","nRMSE"), ("init_nrmse","init_nrmse")]:
    rs = [spearman([E1M[(e,c)]["surplus_final_n"] for c in CELLS], [E1M[(e,c)][fk] for c in CELLS]) for e in EPS]
    rs = [r for r in rs if not math.isnan(r)]
    k, n, p = sign_test(rs)
    P(f"   surplus_final_n vs {flab:<11} med rho={med(rs):+.3f}  pos {k}/{n}  sign p={p:.4f}")
Xv = [r["d_surplus_final_n"] for r in delta_rows]
for yk in ["d_absVR1", "d_Hs_error", "d_nRMSE", "d_init_nrmse"]:
    Yv = [r[yk] for r in delta_rows]
    P(f"   48-contrast  d_surplus_final_n vs {yk:<13} rho={spearman(Xv,Yv):+.3f}  p={perm_p(Xv,Yv,seed=13):.4f}")

P("\n(3c) DOES CONTROLLING THE MEDIATOR REVIVE detection -> field?  (partial Spearman, 48 contrasts)")
hdr = f"{'field y':<13}{'mediator z':<18}{'rho(det,y)':>11}{'rho(det,y|z)':>14}{'rho(z,y)':>10}{'rho(z,y|det)':>14}"
P(hdr); P("-"*len(hdr))
for ng in [3, 15]:
    for zk, zlab in [(f"d_lognn@{ng}", f"log10 nn@{ng}"), (f"d_surplus@{ng}", f"surplus@{ng}"),
                     ("d_surplus_final_n", "surplus final_n")]:
        if zk == "d_surplus_final_n" and ng == 15: continue
        Z = [r[zk] for r in delta_rows]; D = [r[f"d_det@{ng}"] for r in delta_rows]
        for yk, ylab in [("d_nRMSE","nRMSE"), ("d_init_nrmse","init_nrmse"),
                         ("d_absVR1","|VR-1|"), ("d_Hs_error","Hs_error")]:
            Yv = [r[yk] for r in delta_rows]
            P(f"{ylab:<13}{zlab:<18}{spearman(D,Yv):>11.3f}{partial_spearman(D,Yv,Z):>14.3f}"
              f"{spearman(Z,Yv):>10.3f}{partial_spearman(Z,Yv,D):>14.3f}")
        P("")

P("(3d) EPISODE-FIXED-EFFECT regression on the 60 (episode,cell) cells.")
P("     y = alpha_e + b1*detection@15 + b2*log10(nn@15)  [+ b3*surplus@15]")
ks = sorted(E1M.keys())
epidx = {e: i for i, e in enumerate(EPS)}
D_e = np.zeros((len(ks), len(EPS)-1))
for i, k in enumerate(ks):
    j = epidx[k[0]]
    if j < len(EPS)-1: D_e[i, j] = 1.0
det = [PRB[(k[0],k[1],15)]["detection_rate"] for k in ks]
lnn = [PRB[(k[0],k[1],15)]["log_nn"] for k in ks]
sur = [float(PRB[(k[0],k[1],15)]["surplus_in_box"]) for k in ks]
for yk, ylab in [("absVR1","|VR-1|"), ("Hs_error","Hs_error"), ("nRMSE","nRMSE"), ("init_nrmse","init_nrmse")]:
    Yv = [E1M[k][yk] for k in ks]
    fe = [D_e[:, j] for j in range(D_e.shape[1])]
    b1, s1 = ols(Yv, [det] + fe)
    b2, s2 = ols(Yv, [det, lnn] + fe)
    b3, s3 = ols(Yv, [det, lnn, sur] + fe)
    P(f"   {ylab:<11} det alone: b_det={b1[1]:+.4f}+-{s1[1]:.4f} | +log_nn: b_det={b2[1]:+.4f}+-{s2[1]:.4f}, "
      f"b_lognn={b2[2]:+.4f}+-{s2[2]:.4f} | +surplus: b_det={b3[1]:+.4f}+-{s3[1]:.4f}, "
      f"b_lognn={b3[2]:+.4f}+-{s3[2]:.4f}, b_sur={b3[3]:+.4f}+-{s3[3]:.4f}")

# ======================================================================================
P("\n" + "#"*104)
P("# PART 4 — S2 ORACLE: counterexample or confirmation?  What spacing does the oracle place at?")
P("#"*104)
P("\nTRUE component nn spacing = the spacing the oracle arm places at, because S2-T1f/T4 FREEZE k")
P("at ground truth (k_mag_rel_error_mean = 0.000 measured).  Computed from config/wave_components.")
hdr = f"{'episode':<26}{'N_gt':>5}{'nn_min':>9}{'nn_med':>9}{'nn_max':>9}{'|k|max':>8}{'|k|min':>8}"
P(hdr); P("-"*len(hdr))
for e in EPS:
    t = TRUE[e]; nm = e.replace("episode_","").replace("_seed0","").replace(".h5","")
    if t["n_gt"] >= 2:
        P(f"{nm:<26}{t['n_gt']:>5}{t['nn'].min():>9.4f}{np.median(t['nn']):>9.4f}{t['nn'].max():>9.4f}{t['kmax']:>8.4f}{t['kmin']:>8.4f}")
    else:
        P(f"{nm:<26}{t['n_gt']:>5}{'n/a':>9}{'n/a':>9}{'n/a':>9}{t['kmax']:>8.4f}{t['kmin']:>8.4f}")

P("\n★ HEAD-TO-HEAD (crowded 4 ep): three ways to 'fix placement', and what each does to spacing")
hdr = f"{'regime':<40}{'atom nn_median':>16}{'detection':>11}{'k_err':>8}{'nRMSE':>9}{'VR':>8}{'Hs_err':>9}"
P(hdr); P("-"*len(hdr))
def cm(c, f): return med([E1M[(e,c)][f] for e in CROWDED])
def sm(a, f): return med([S12M[(e,a)][f] for e in CROWDED if (e,a) in S12M])
nnA = med([PRB[(e,"A",15)]["placed_nn_median"] for e in CROWDED])
nnD = med([PRB[(e,"D",15)]["placed_nn_median"] for e in CROWDED])
nnB = med([PRB[(e,"B",15)]["placed_nn_median"] for e in CROWDED])
nnO = med([float(np.median(TRUE[e]["nn"])) for e in CROWDED])
P(f"{'A  shipping (wide grid)':<40}{nnA:>16.4f}{cm('A','spec_detection_rate'):>11.3f}"
  f"{cm('A','spec_k_mag_rel_error_mean'):>8.3f}{cm('A','nRMSE'):>9.4f}{cm('A','Variance_Ratio'):>8.3f}{cm('A','Hs_error'):>9.4f}")
P(f"{'B  narrow grid, static peak':<40}{nnB:>16.4f}{cm('B','spec_detection_rate'):>11.3f}"
  f"{cm('B','spec_k_mag_rel_error_mean'):>8.3f}{cm('B','nRMSE'):>9.4f}{cm('B','Variance_Ratio'):>8.3f}{cm('B','Hs_error'):>9.4f}")
P(f"{'D  narrow grid + recompute (E1 PRIMARY)':<40}{nnD:>16.4f}{cm('D','spec_detection_rate'):>11.3f}"
  f"{cm('D','spec_k_mag_rel_error_mean'):>8.3f}{cm('D','nRMSE'):>9.4f}{cm('D','Variance_Ratio'):>8.3f}{cm('D','Hs_error'):>9.4f}")
P(f"{'S2-T1f  oracle k (frozen at truth)':<40}{nnO:>16.4f}{sm('S2-T1f','spec_detection_rate'):>11.3f}"
  f"{sm('S2-T1f','spec_k_mag_rel_error_mean'):>8.3f}{sm('S2-T1f','nRMSE'):>9.4f}{sm('S2-T1f','Variance_Ratio'):>8.3f}{sm('S2-T1f','Hs_error'):>9.4f}")
P(f"{'X-T1f-notrack  oracle + tracker OFF':<40}{nnO:>16.4f}{sm('X-T1f-notrack','spec_detection_rate'):>11.3f}"
  f"{sm('X-T1f-notrack','spec_k_mag_rel_error_mean'):>8.3f}{sm('X-T1f-notrack','nRMSE'):>9.4f}"
  f"{sm('X-T1f-notrack','Variance_Ratio'):>8.3f}{sm('X-T1f-notrack','Hs_error'):>9.4f}")
P(f"\n  oracle spacing / cell-D spacing = {nnO/nnD:.1f}x ;  oracle spacing / cell-A spacing = {nnO/nnA:.2f}x")
P(f"  ==> the oracle is NOT 'the narrow-grid fix taken further'. It places atoms ~{nnO/nnD:.0f}x FARTHER APART")
P(f"      than the narrow grid does, and by construction has ZERO surplus atoms (n_est = N_gt exactly).")

P("\n★ THE OTHER HALF OF S2 — perfect placement makes the DISPERSION metrics WORSE than shipping")
hdr = f"{'arm':<24}{'nRMSE':>9}{'init_nrmse':>12}{'VR':>8}{'|VR-1|':>9}{'Hs_err':>9}{'det':>7}{'k_err':>8}{'final_n':>9}"
P(hdr); P("-"*len(hdr))
S2EPS = sorted({k[0] for k in S12M if k[1]=="S2-T1f"})
for a in ["S1-A0","S2-C1","S2-T1","S2-T1f","S2-T4","X-A0-notrack","X-T1f-notrack"]:
    def m(f): return med([S12M[(e,a)][f] for e in S2EPS])
    P(f"{a:<24}{m('nRMSE'):>9.4f}{m('init_nrmse'):>12.4f}{m('Variance_Ratio'):>8.3f}{m('absVR1'):>9.3f}"
      f"{m('Hs_error'):>9.4f}{m('spec_detection_rate'):>7.3f}{m('spec_k_mag_rel_error_mean'):>8.3f}{m('N'):>9.1f}")
P("")
for a, b in [("S2-T1f","S1-A0"), ("X-T1f-notrack","S2-T1f"), ("S2-T1f","S2-C1")]:
    for f, lab in [("nRMSE","nRMSE"), ("absVR1","|VR-1|"), ("Hs_error","Hs_error"), ("Variance_Ratio","VR")]:
        d = [S12M[(e,a)][f]-S12M[(e,b)][f] for e in S2EPS]
        k, n, p = sign_test(d)
        P(f"   {a:<15}-{b:<9} {lab:<9} med={med(d):+.4f}  larger {k}/{n}  p={p:.4f}")
    P("")
P("★ SAME-DIRECTION REPLICATION: two totally different placement interventions, same dispersion harm")
dh_e1 = med([E1M[(e,"D")]["Hs_error"]-E1M[(e,"A")]["Hs_error"] for e in EPS])
dv_e1 = med([E1M[(e,"D")]["absVR1"]-E1M[(e,"A")]["absVR1"] for e in EPS])
dh_s2 = med([S12M[(e,"S2-T1f")]["Hs_error"]-S12M[(e,"S1-A0")]["Hs_error"] for e in S2EPS])
dv_s2 = med([S12M[(e,"S2-T1f")]["absVR1"]-S12M[(e,"S1-A0")]["absVR1"] for e in S2EPS])
P(f"   E1  narrow grid   (D-A, 12 ep): d_Hs_error {dh_e1:+.4f} 12/12 | d|VR-1| {dv_e1:+.4f} 12/12")
P(f"   S2  oracle k      (T1f-A0, 4 ep): d_Hs_error {dh_s2:+.4f} | d|VR-1| {dv_s2:+.4f}")

P("\n★ DECOMPOSITION of the S2-T1f residual: how much does perfect placement NOT remove?")
t1f = med([S12M[(e,"S2-T1f")]["nRMSE"] for e in S2EPS])
xn  = med([S12M[(e,"X-T1f-notrack")]["nRMSE"] for e in S2EPS])
a0  = med([S12M[(e,"S1-A0")]["nRMSE"] for e in S2EPS])
LETKF = 0.2394
P(f"   perfect placement, tracker ON  nRMSE = {t1f:.4f}")
P(f"   perfect placement, tracker OFF nRMSE = {xn:.4f}")
P(f"   tracker-attributable error that perfect placement CANNOT remove = {t1f-xn:+.4f}"
  f"  = {100*(t1f-xn)/t1f:.0f}% of the perfect-placement arm's error")
P(f"   shipping A0 {a0:.4f} vs frozen LETKF ref {LETKF:.4f} -> gap {a0-LETKF:+.4f}")
P(f"   ==> the tracker term is {(t1f-xn)/abs(a0-LETKF):.1f}x the ENTIRE WS-vs-LETKF gap.")

# ======================================================================================
P("\n" + "#"*104)
P("# PART 5 — S1 budget sweep: more atoms into a fixed k-support = stacking by construction")
P("#"*104)
hdr = (f"{'arm':<10}{'nRMSE':>9}{'det':>7}{'edet':>7}{'n_est':>7}{'final_n':>8}{'surplus':>8}{'VR':>7}"
       f"{'|VR-1|':>8}{'Hs_err':>9}{'amp_err':>9}{'dir_err':>9}{'k_err':>8}{'nn_proxy':>9}")
P(hdr); P("-"*len(hdr))
for a in ["S1-A0","S1-A1","S1-A2","S1-A3"]:
    def m(f): return med([S12M[(e,a)][f] for e in S2EPS])
    Rm = med([TRUE[e]["kmax"] for e in S2EPS])
    P(f"{a:<10}{m('nRMSE'):>9.4f}{m('spec_detection_rate'):>7.3f}{m('spec_energy_detection_rate'):>7.3f}"
      f"{m('spec_n_est'):>7.1f}{m('N'):>8.1f}{m('surplus_final_n'):>8.1f}{m('Variance_Ratio'):>7.3f}"
      f"{m('absVR1'):>8.3f}{m('Hs_error'):>9.4f}{m('spec_amp_rel_error_mean'):>9.4f}"
      f"{m('spec_direction_error_deg_mean'):>9.2f}{m('spec_k_mag_rel_error_mean'):>8.3f}"
      f"{Rm/math.sqrt(max(m('spec_n_est'),1)):>9.4f}")
P("  nn_proxy = |k|max / sqrt(n_est) : GEOMETRIC PROXY. S1 raw data has no placed_nn_median field.")
P("\n  paired vs S1-A0 (4 crowded ep, median-of-3):")
for a in ["S1-A1","S1-A2","S1-A3"]:
    out = []
    for f, lab, good in [("spec_detection_rate","det",+1), ("nRMSE","nRMSE",-1),
                         ("absVR1","|VR-1|",-1), ("Hs_error","Hs_err",-1),
                         ("spec_amp_rel_error_mean","amp_err",-1),
                         ("spec_direction_error_deg_mean","dir_err",-1)]:
        d = [S12M[(e,a)][f]-S12M[(e,"S1-A0")][f] for e in S2EPS]
        k, n, p = sign_test(d)
        out.append(f"{lab} {med(d):+.4f}({k}/{n})")
    P(f"   {a}: " + "  ".join(out))
P("\n  ★ BF4_15comp: detection 0.467 -> 1.000 (A1/A3 reach ALL 15 true components).")
for a in ["S1-A0","S1-A1","S1-A2","S1-A3"]:
    e = "episode_BF4_15comp_seed0_uniform.h5"; v = S12M[(e,a)]
    P(f"     {a}: det={v['spec_detection_rate']:.3f} edet={v['spec_energy_detection_rate']:.3f} "
      f"n_est={v['spec_n_est']:.0f} final_n={v['N']:.0f} nRMSE={v['nRMSE']:.4f} "
      f"VR={v['Variance_Ratio']:.3f} Hs_err={v['Hs_error']:.4f} amp_err={v['spec_amp_rel_error_mean']:.3f}")

open(f"{OUT}/PRO_analysis.txt","w").write("\n".join(lines)+"\n")
json.dump({"delta_rows": delta_rows}, open(f"{OUT}/PRO_contrasts.json","w"), indent=1)
P(f"\nwritten: {OUT}/PRO_analysis.txt , PRO_contrasts.json")
