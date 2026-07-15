"""E-24 (novelty lever #1) — DECOUPLE the field-reconstruction CRB from the raw
aperture ratio, and show the CRB does INDEPENDENT work predicting Delta_lin.

THE OBJECTION WE MUST KILL
--------------------------
Prior work (E-19) found the field-reconstruction Cramer-Rao bound predicts the
sign of the LINEAR reconstruction gain  Delta_lin = corr(net) - corr(hand-MMSE)
with Spearman +0.725: Delta_lin < 0 where the field is well-determined (short
wavelength, low CRB) and Delta_lin > 0 where under-determined (long wavelength,
high CRB). BUT across E-19's grid the CRB is collinear with the raw aperture
ratio  ar = R_max/lambda  (Spearman CRB~ar ~ -0.67 in that grid). A hostile
examiner therefore says: "the CRB is a re-labelled aperture ratio; it does no
independent work. You varied lambda, the CRB moved because lambda moved, full
stop."

THE INTERVENTION (this experiment)
----------------------------------
Break the collinearity by HOLDING THE APERTURE GEOMETRY FIXED and moving the CRB
through OBSERVATION INFORMATIVENESS instead of geometry:

  * SAME fan:      +/-60 deg, R in [15, 200] m, 12 scans over [-20,0]s   (unchanged)
  * SAME wavelength within a group:  Tp in {6,7,8} -> lambda ~ {56,76,100} m
    (all short / well-determined; aperture ratio ar = 200/lambda is CONSTANT
     within each Tp group).
  * VARY ONLY the measurement noise sigma (primary axis) over the IDENTICAL point
    cloud. Within one (Tp, seed, draw) the exact same points are selected and the
    exact same unit-noise vector is used; only sigma scales it and only sigma is
    handed to the estimators and to the CRB. So the geometry is byte-identical
    across noise levels: nothing about the aperture changes, yet
        field-nRMSE CRB  ∝  sigma   (Fisher F = J^T J / sigma^2, G fixed)
    sweeps from well below to well above the E-19 ~0.11 flip threshold.
  * SECONDARY (reported, not gated) axis: point density n_obs at fixed sigma=0.05,
    a DIFFERENT way to degrade informativeness (sparser scan), aperture still fixed.

If, at FIXED aperture (within a Tp group), pushing sigma up moves the CRB up and
FLIPS Delta_lin from negative to positive, then the CRB — not the geometry —
carried the information. The aperture ratio was constant and did zero work.

Linear field ONLY (build_sea_linear, the analytic free-wave = the correct-physics
arm where the sign-flip lives). No HOS, so NO NaN risk. Short lambda is REQUIRED:
we need a baseline-negative Delta_lin (well-determined) that we can flip positive
by starving the observation of information.

CRB is computed with the EXACT E-19 pipeline (field_nrmse_crb over the +/-60 fan
reconstruction grid at t*=0, theta = the 60 JONSWAP free-wave modes) so the ~0.11
threshold and the +0.725 comparison carry over verbatim — the ONLY change vs E-19
is that the obs points and sigma come from the parametrized draw.

=================  PRE-REGISTERED GATES  =====================================
Analysis unit for the gates: sea = (Tp, seed). k = 3 lambda x n_seed seas. Within
a sea the aperture ratio is CONSTANT, so any Delta_lin variation across the noise
sweep is aperture-free by construction.

GATE A  (CRB does INDEPENDENT work — the decoupling proof).
  For each sea, fit the within-sea slope  b_sea = d(Delta_lin)/d(log CRB)  over
  the 5 noise levels (draw-averaged Delta_lin and CRB per level). Because lambda
  (hence aperture) is fixed within a sea, b_sea is a PURE informativeness->CRB->
  Delta_lin effect with the aperture partialled out exactly.
  PASS = mean_sea(b_sea) > 0  with BOTH  (i) sea-level t(k-1) p < 0.05  AND
         (ii) Webb wild cluster bootstrap-t p < 0.05  (cluster = sea, null imposed).
  [This is what proves the CRB is not a re-parameterization of aperture: at
   constant aperture it still predicts Delta_lin.]

GATE B  (the sign-flip happens at a CRB threshold, at fixed aperture).
  Pooling seeds within each lambda group:
  (b1) pooled Delta_lin at the lowest sigma (0.02, lowest CRB) < 0  AND at the
       highest sigma (0.40, highest CRB) > 0;
  (b2) >= 2 of the 3 lambda groups individually show mean Delta_lin cross 0->+
       as sigma rises;
  (b3) the median per-lambda crossing CRB lies in [0.05, 0.20] (brackets the
       E-19 ~0.11 threshold).
  PASS = b1 AND b2 AND b3.

OVERALL PASS = GATE A and GATE B.

Exploratory (reported, not gated):
  * pooled partial Spearman(CRB, Delta_lin | lambda) and per-lambda Spearman;
  * joint OLS  Delta_lin ~ z(logCRB) + z(ar): expect z(logCRB) significant and,
    the mirror of the examiner's complaint, z(ar) NON-significant once CRB is in
    (aperture does no independent work) — underpowered at 3 lambda, so reported
    not gated;
  * the raw confound Spearman(ar, Delta_lin) at baseline sigma (should be < 0,
    i.e. the collinearity the examiner points at IS present) — then Gate A shows
    the CRB survives it;
  * SECONDARY density axis (n_obs in {600,300,150} at sigma=0.05): does the same
    CRB->Delta_lin law hold when informativeness is degraded by SPARSITY instead
    of noise, aperture still fixed? (Confirmatory; n_obs=150 < 4N=240 so its CRB
    is ridge-sensitive and flagged.)

  python3 tmp/e24_crb_decouple.py            # full (GPU; ~10-20 min)
  python3 tmp/e24_crb_decouple.py --smoke    # 1 lambda x 1 seed x 1 draw, 2 sigmas
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# parallel CPU raycast builds (order=1 -> build_sea_linear) BEFORE any GPU init
from e18_decorrelated_2factor import build_cell, _lam, R_APERTURE  # noqa: E402

OUT = 'results/direction3/e24_crb_decouple.json'

# --- the fixed-aperture grid ------------------------------------------------
TPS = [6.0, 7.0, 8.0]              # lambda ~ 56 / 76 / 100 m, all well-determined
STEEP = 0.02                        # single modest steepness -> Hs = STEEP*lambda
NOISES = [0.02, 0.05, 0.10, 0.20, 0.40]    # primary informativeness axis (sigma)
NOBS_ROBUST = [600, 300, 150]              # secondary axis, at sigma = 0.05
SIG_BASE = 0.05                            # E-19 baseline noise (for confound / density)
CRB_THRESH_BAND = (0.05, 0.20)             # brackets E-19 ~0.11 flip threshold

# fan geometry (hardcoded to match d3_door1b_field_bias so this module stays
# torch-free at top level -> the ProcessPoolExecutor fork never inherits a live
# CUDA context; a post-fork assert in main() guards against drift).
R_MIN, R_MAX, FOV_H = 15.0, 200.0, 60.0
# fixed reconstruction grid for the CRB (SAME +/-60 fan at t*=0 for every sea, so
# the field design G depends only on theta; CRB moves only through F(points,sigma))
_GR = np.random.default_rng(1)
_GA = np.random.default_rng(2)
_GRID_RR = np.sqrt(_GR.uniform((R_MIN / R_MAX) ** 2, 1.0, 400)) * R_MAX
_GRID_AA = _GA.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 400)


def _corr(p, gt):
    return float(np.corrcoef(p, gt)[0, 1]) if np.std(p) > 1e-9 else np.nan


def make_draw(sea, rng, n_obs):
    """Select ONE observation point cloud + unit-noise vector + query set. No
    sigma applied yet -> the identical geometry is reused across every sigma."""
    from wavesim.raydrop_threshold import T_MANUAL_50
    from d3_h7_hos_arena import z_at
    from e15_hos_multidraw import LEADS
    P, zc, rr = [], [], []
    for sc in sea['scans']:                              # return-power lottery
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep],
                                  np.full(int(keep.sum()), sc['trel'])]))
        zc.append(sc['z'][keep])
        rr.append(sc['r'][keep])
    P, zc, rr = np.vstack(P), np.concatenate(zc), np.concatenate(rr)
    eps = rng.standard_normal(len(zc))                   # UNIT noise, scaled later
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)      # range-stratified n_obs
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    per = max(-(-n_obs // 12), 1)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), per)
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Ps, zcs, epss = P[sel], zc[sel], eps[sel]
    rf = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, 800)) * R_MAX  # queries
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf * np.cos(af), rf * np.sin(af)
    Q, gt = [], []
    for li, l_ in enumerate(LEADS):
        gt.append(z_at(sea['leads'][li], qx, qy))
        Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
    Q, gt = np.vstack(Q), np.concatenate(gt)
    return Ps, zcs, epss, Q, gt


def crb_from_obs(Ps, theta, Gmat, sigma, Hs):
    """E-19 field-nRMSE CRB from the ACTUAL selected obs points + this sigma.
    Gmat (field design over the fixed t*=0 fan grid) is precomputed per sea."""
    import torch
    import d1_crb
    from d1_crb import fisher, field_nrmse_crb, DT
    from d3_f1_hos_door1 import CX, CY, DEPTH
    dev = d1_crb.DEVICE
    x = torch.tensor(CX + Ps[:, 0], dtype=DT, device=dev)
    y = torch.tensor(CY + Ps[:, 1], dtype=DT, device=dev)
    t = torch.tensor(Ps[:, 2], dtype=DT, device=dev)
    F = fisher(theta, x, y, t, DEPTH, sigma)
    crb, _ = field_nrmse_crb(F, Gmat, Hs)
    return crb


def field_design_for_sea(theta):
    """G = <g g^T> over the FIXED +/-60 fan grid at t*=0 (E-19 geometry)."""
    import torch
    import d1_crb
    from d1_crb import jacobian, DT
    from d3_f1_hos_door1 import CX, CY, DEPTH
    dev = d1_crb.DEVICE
    gx = torch.tensor(CX + _GRID_RR * np.cos(_GRID_AA), dtype=DT, device=dev)
    gy = torch.tensor(CY + _GRID_RR * np.sin(_GRID_AA), dtype=DT, device=dev)
    Jg = jacobian(theta, gx, gy, torch.zeros_like(gx), DEPTH)
    return (Jg.T @ Jg) / Jg.shape[0]


def eval_cell(Ps, zcs, epss, Q, gt, theta, Gmat, sigma, kp, Hs, net, head):
    """hand-MMSE, net, and CRB on the IDENTICAL geometry at this sigma."""
    import torch
    from d3_gpu_mmse import mmse_gpu
    from d3_prior_net_eval import net_gamma_mmse, DEVICE
    from d3_f1_hos_door1 import DEPTH as DEP
    zs = zcs + sigma * epss
    with torch.no_grad():
        ph = mmse_gpu(torch.tensor(Ps, device=DEVICE, dtype=torch.float32),
                      torch.tensor(zs, device=DEVICE, dtype=torch.float32), DEP,
                      torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                      sigma, kp_hint=kp).cpu().numpy()
    pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEP, Q, sigma, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head))
    hand, netc = _corr(ph, gt), _corr(pn, gt)
    crb = float(crb_from_obs(Ps, theta, Gmat, sigma, Hs))
    return hand, netc, crb


# ---------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------
def wild_sea(y, seed=0, nboot=10000):
    """Webb wild cluster bootstrap-t on k independent sea-level values, NULL
    IMPOSED (yb = y*w has E[yb]=0), identical to e16_physics_operator.wild_sea."""
    rng = np.random.default_rng(seed)
    WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5),
                     np.sqrt(.5), 1., np.sqrt(1.5)])
    mu = y.mean()
    se = y.std(ddof=1) / np.sqrt(len(y))
    t_o = mu / se if se > 0 else np.inf
    boot = []
    for _ in range(nboot):
        yb = y * rng.choice(WEBB, len(y))
        seb = yb.std(ddof=1) / np.sqrt(len(y))
        boot.append(yb.mean() / seb if seb > 0 else 0.0)
    boot = np.array(boot)
    return float(t_o), float((np.abs(boot) >= abs(t_o)).mean())


def crossing_crb(sig_crb, sig_dlin):
    """CRB at which the pooled Delta_lin curve crosses 0, by linear interp in
    (CRB, Delta_lin) between the adjacent noise levels that straddle 0."""
    crb = np.asarray(sig_crb, float)
    dl = np.asarray(sig_dlin, float)
    o = np.argsort(crb)
    crb, dl = crb[o], dl[o]
    for i in range(len(dl) - 1):
        if dl[i] < 0 <= dl[i + 1] or dl[i] <= 0 < dl[i + 1]:
            f = (0.0 - dl[i]) / (dl[i + 1] - dl[i]) if dl[i + 1] != dl[i] else 0.0
            return float(crb[i] + f * (crb[i + 1] - crb[i]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--noises', type=float, nargs='+', default=NOISES)
    ap.add_argument('--density', action='store_true', default=True,
                    help='also run the secondary n_obs density axis')
    ap.add_argument('--no-density', dest='density', action='store_false')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    tps = TPS[:1] if a.smoke else TPS
    seeds = a.seeds[:1] if a.smoke else a.seeds
    draws = 1 if a.smoke else a.draws
    noises = [0.05, 0.40] if a.smoke else a.noises
    nobs_levels = [] if a.smoke else (NOBS_ROBUST if a.density else [])

    seas = [(Tp, round(STEEP * _lam(Tp), 3)) for Tp in tps]
    tasks = [(Tp, Hs, sd, 1) for (Tp, Hs) in seas for sd in seeds]   # order=1 only
    print(f"E-24 CRB-decouple: {len(seas)} lambda x {len(seeds)} seeds = "
          f"{len(tasks)} linear seas; sigma sweep {noises}; "
          f"density {nobs_levels or 'off'}", flush=True)

    # ---- parallel CPU linear builds (before any GPU / cuda init) ----
    t0 = time.time()
    built = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for task, sea, note, dt in ex.map(build_cell, tasks):
            built[task] = sea
            if sea is None:
                print(f"  !! build {task} -> {note} [{dt:.0f}s]", flush=True)
    n_ok = sum(v is not None for v in built.values())
    print(f"  built {n_ok}/{len(tasks)} seas in {(time.time()-t0)/60:.1f} min",
          flush=True)

    # ---- GPU estimators + CRB (serial, main process) ----
    import torch
    import d3_f1_hos_door1 as F1
    from d3_f1_hos_door1 import sea_components
    from e15_hos_multidraw import CKPT
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    from d1_crb import DT
    import d1_crb
    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    # drift guard: hardcoded fan geometry must equal the source of truth
    import d3_door1b_field_bias as DB
    assert (R_MIN, R_MAX, FOV_H) == (DB.R_MIN, DB.R_MAX, DB.FOV_H), \
        f"fan geometry drift: {(R_MIN, R_MAX, FOV_H)} != " \
        f"{(DB.R_MIN, DB.R_MAX, DB.FOV_H)}"

    rows = []           # one row per (Tp, seed, draw, axis, level)
    for (Tp, Hs) in seas:
        lam = _lam(Tp)
        ar = R_APERTURE / lam
        F1.TP = float(Tp)                      # sea_components uses the module TP
        for sd in seeds:
            sea = built[(Tp, Hs, sd, 1)]
            if sea is None:
                continue
            S = sea_components(sd, Hs=Hs)
            theta = torch.tensor(np.stack([S['A'], S['kx'], S['ky'], S['ph']], 1),
                                 dtype=DT, device=d1_crb.DEVICE)
            Gmat = field_design_for_sea(theta)
            kp = sea['kp']
            for dr in range(draws):
                # PRIMARY: fixed n_obs=600, identical geometry, sweep sigma
                rng = np.random.default_rng(240000 + 9973 * sd + 31 * dr)
                Ps, zcs, epss, Q, gt = make_draw(sea, rng, 600)
                for sig in noises:
                    h, n_, crb = eval_cell(Ps, zcs, epss, Q, gt, theta, Gmat,
                                           sig, kp, Hs, net, head)
                    rows.append(dict(axis='noise', Tp=Tp, lam=lam,
                                     aperture_ratio=ar, seed=sd, draw=dr,
                                     sigma=sig, n_obs=int(len(zcs)), crb=crb,
                                     hand=h, net=n_, d_lin=n_ - h))
                # SECONDARY: fixed sigma=SIG_BASE, sweep n_obs (density)
                for nob in nobs_levels:
                    rng = np.random.default_rng(240000 + 9973 * sd + 31 * dr)
                    Ps2, zc2, ep2, Q2, gt2 = make_draw(sea, rng, nob)
                    h, n_, crb = eval_cell(Ps2, zc2, ep2, Q2, gt2, theta, Gmat,
                                           SIG_BASE, kp, Hs, net, head)
                    rows.append(dict(axis='density', Tp=Tp, lam=lam,
                                     aperture_ratio=ar, seed=sd, draw=dr,
                                     sigma=SIG_BASE, n_obs=int(len(zc2)), crb=crb,
                                     hand=h, net=n_, d_lin=n_ - h))
            # progress line: draw-averaged Delta_lin / CRB across the sigma sweep
            nz = [r for r in rows if r['axis'] == 'noise' and r['Tp'] == Tp
                  and r['seed'] == sd]
            byS = {s: np.mean([r['d_lin'] for r in nz if r['sigma'] == s])
                   for s in noises}
            byC = {s: np.mean([r['crb'] for r in nz if r['sigma'] == s])
                   for s in noises}
            print(f"[Tp{Tp:g} lam{lam:.0f} ar{ar:.2f} seed{sd}] "
                  + " ".join(f"s{s:g}:CRB{byC[s]:.3f}/D{byS[s]:+.3f}"
                             for s in noises), flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    if a.smoke:
        print("\n[smoke] rows:", len(rows), "-> logic OK, skipping inference")
        return 0

    # =================== inference ==========================================
    noise_rows = [r for r in rows if r['axis'] == 'noise'
                  and np.isfinite(r['d_lin']) and np.isfinite(r['crb'])]
    seas_u = sorted({(r['Tp'], r['seed']) for r in noise_rows})
    k = len(seas_u)

    def cellmean(Tp, sd, sig, key):
        v = [r[key] for r in noise_rows if r['Tp'] == Tp and r['seed'] == sd
             and r['sigma'] == sig]
        return float(np.mean(v)) if v else np.nan

    # ---- GATE A: within-sea slope of Delta_lin vs log(CRB) (aperture fixed) --
    slopes = []
    for (Tp, sd) in seas_u:
        xc = np.array([np.log(cellmean(Tp, sd, s, 'crb')) for s in noises])
        yd = np.array([cellmean(Tp, sd, s, 'd_lin') for s in noises])
        m = np.isfinite(xc) & np.isfinite(yd)
        if m.sum() >= 3:
            slopes.append(np.polyfit(xc[m], yd[m], 1)[0])
    slopes = np.array(slopes)
    t_s, p_s = stats.ttest_1samp(slopes, 0.0)
    t_w, p_w = wild_sea(slopes)
    gate_a = bool(slopes.mean() > 0 and p_s < 0.05 and p_w < 0.05)

    # ---- GATE B: within-lambda sign-flip + crossing CRB near 0.11 -----------
    lo, hi = min(noises), max(noises)
    d_lo = np.nanmean([cellmean(Tp, sd, lo, 'd_lin') for (Tp, sd) in seas_u])
    d_hi = np.nanmean([cellmean(Tp, sd, hi, 'd_lin') for (Tp, sd) in seas_u])
    b1 = bool(d_lo < 0 and d_hi > 0)
    per_lambda, crossings = {}, []
    for Tp in tps:
        sds = sorted({sd for (T, sd) in seas_u if T == Tp})
        sig_dl = [np.nanmean([cellmean(Tp, sd, s, 'd_lin') for sd in sds])
                  for s in noises]
        sig_cr = [np.nanmean([cellmean(Tp, sd, s, 'crb') for sd in sds])
                  for s in noises]
        flip = bool(sig_dl[np.argmin(noises)] < 0 and sig_dl[np.argmax(noises)] > 0)
        xcr = crossing_crb(sig_cr, sig_dl)
        if xcr is not None:
            crossings.append(xcr)
        per_lambda[Tp] = dict(lam=_lam(Tp), aperture_ratio=R_APERTURE / _lam(Tp),
                              sig_dlin=sig_dl, sig_crb=sig_cr, flip=flip,
                              crossing_crb=xcr)
    n_flip = sum(v['flip'] for v in per_lambda.values())
    b2 = bool(n_flip >= 2)
    med_cross = float(np.median(crossings)) if crossings else np.nan
    b3 = bool(np.isfinite(med_cross)
              and CRB_THRESH_BAND[0] <= med_cross <= CRB_THRESH_BAND[1])
    gate_b = bool(b1 and b2 and b3)

    overall = bool(gate_a and gate_b)

    # ---- exploratory: pooled partial Spearman(CRB, Dlin | lambda) -----------
    cells = [(Tp, sd, s) for (Tp, sd) in seas_u for s in noises]
    cd = np.array([cellmean(Tp, sd, s, 'd_lin') for (Tp, sd, s) in cells])
    cc = np.log(np.array([cellmean(Tp, sd, s, 'crb') for (Tp, sd, s) in cells]))
    car = np.array([R_APERTURE / _lam(Tp) for (Tp, sd, s) in cells])
    clam = np.array([Tp for (Tp, sd, s) in cells])
    good = np.isfinite(cd) & np.isfinite(cc)
    cd, cc, car, clam = cd[good], cc[good], car[good], clam[good]

    def resid_rank_within(v, grp):
        r = stats.rankdata(v)
        out = np.zeros_like(r, float)
        for g in np.unique(grp):
            mm = grp == g
            out[mm] = r[mm] - r[mm].mean()
        return out

    partial_rho = float(np.corrcoef(resid_rank_within(cc, clam),
                                    resid_rank_within(cd, clam))[0, 1])
    per_lam_spear = {float(Tp): float(stats.spearmanr(
        cc[clam == Tp], cd[clam == Tp])[0]) for Tp in tps}

    # joint OLS Delta_lin ~ z(logCRB) + z(ar): does ar still do work given CRB?
    def ols(y, cols):
        X = np.column_stack([np.ones(len(y))] + cols)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        dof = len(y) - X.shape[1]
        se = np.sqrt(np.diag((r @ r) / dof * np.linalg.inv(X.T @ X)))
        return b, b / se, 2 * stats.t.sf(np.abs(b / se), dof)
    zc_ = stats.zscore(cc)
    zar = stats.zscore(car) if np.std(car) > 0 else np.zeros_like(car)
    bj, tj, pj = ols(cd, [zc_, zar])

    # raw confound at baseline sigma: Spearman(ar, Dlin) across lambda
    base = [(Tp, sd) for (Tp, sd) in seas_u]
    bdl = np.array([cellmean(Tp, sd, SIG_BASE, 'd_lin') for (Tp, sd) in base])
    bar = np.array([R_APERTURE / _lam(Tp) for (Tp, sd) in base])
    mm = np.isfinite(bdl)
    rho_conf, p_conf = stats.spearmanr(bar[mm], bdl[mm])

    # ---- secondary density axis (confirmatory) -----------------------------
    dens = {}
    drows = [r for r in rows if r['axis'] == 'density'
             and np.isfinite(r['d_lin']) and np.isfinite(r['crb'])]
    if drows:
        d_slopes = []
        for (Tp, sd) in seas_u:
            xc, yd = [], []
            for nob in NOBS_ROBUST:
                v = [r for r in drows if r['Tp'] == Tp and r['seed'] == sd
                     and abs(r['n_obs'] - nob) <= max(12, nob // 12)]
                if v:
                    xc.append(np.log(np.mean([r['crb'] for r in v])))
                    yd.append(np.mean([r['d_lin'] for r in v]))
            if len(xc) >= 2:
                d_slopes.append(np.polyfit(xc, yd, 1)[0])
        d_slopes = np.array(d_slopes)
        dens = dict(n_seas=len(d_slopes),
                    mean_slope=float(d_slopes.mean()) if len(d_slopes) else np.nan,
                    frac_pos=float((d_slopes > 0).mean()) if len(d_slopes) else np.nan)

    # ======================= report ========================================
    print(f"\n{'='*76}\nE-24 CRB-decouple  ({k} seas = {len(tps)} lambda x "
          f"{len(seeds)} seeds, {draws} draws, |sigma|={len(noises)})\n{'='*76}")
    print("GATE A  — CRB does INDEPENDENT work (within-sea slope, aperture fixed)")
    print(f"  within-sea d(Dlin)/d(logCRB): mean {slopes.mean():+.4f}  "
          f"(n={len(slopes)}, {(slopes>0).sum()}/{len(slopes)} positive)")
    print(f"    sea-level t({k-1}) {t_s:+.2f} p {p_s:.4f}   "
          f"wild-cluster boot t {t_w:+.2f} p {p_w:.4f}")
    print(f"  *** GATE A: {'PASS' if gate_a else 'FAIL'}")
    print("\nGATE B  — sign-flip at a CRB threshold, aperture fixed")
    print(f"  pooled Dlin: lowest sigma={lo:g} -> {d_lo:+.4f}   "
          f"highest sigma={hi:g} -> {d_hi:+.4f}   (b1 {b1})")
    print(f"  per-lambda flips {n_flip}/{len(tps)} (b2 {b2}); "
          f"crossing CRB median {med_cross:.4f} band {CRB_THRESH_BAND} (b3 {b3})")
    for Tp, v in per_lambda.items():
        print(f"    Tp{Tp:g} lam{v['lam']:.0f} ar{v['aperture_ratio']:.2f} "
              f"flip {v['flip']} crossCRB {v['crossing_crb']}")
        print("        " + "  ".join(
            f"CRB{c:.3f}:D{d:+.3f}" for c, d in zip(v['sig_crb'], v['sig_dlin'])))
    print(f"  *** GATE B: {'PASS' if gate_b else 'FAIL'}")
    print(f"\n*** OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"\nexploratory:")
    print(f"  partial Spearman(logCRB, Dlin | lambda) {partial_rho:+.3f}  "
          f"(aperture removed by within-lambda ranking)")
    print(f"  per-lambda Spearman(logCRB, Dlin): "
          + "  ".join(f"Tp{t:g}:{r:+.2f}" for t, r in per_lam_spear.items()))
    print(f"  joint OLS Dlin ~ z(logCRB)+z(ar): "
          f"logCRB beta {bj[1]:+.4f} p {pj[1]:.4f} | "
          f"ar beta {bj[2]:+.4f} p {pj[2]:.4f}  "
          f"(mirror: ar should go NON-significant once CRB is in)")
    print(f"  raw confound Spearman(ar, Dlin)@sigma{SIG_BASE:g}: "
          f"{rho_conf:+.3f} p {p_conf:.4f}  (the collinearity the examiner cites)")
    if dens:
        print(f"  SECONDARY density axis: within-sea d(Dlin)/d(logCRB) mean "
              f"{dens['mean_slope']:+.4f}  {dens['frac_pos']*100:.0f}% positive "
              f"(n={dens['n_seas']}) — same CRB law via sparsity, aperture fixed")

    json.dump({'rows': rows, 'summary': dict(
        k_seas=k, n_lambda=len(tps), n_seeds=len(seeds), draws=draws,
        noises=noises, gate_a=gate_a, gate_b=gate_b, overall=overall,
        slope_mean=float(slopes.mean()), slope_t=float(t_s), slope_p=float(p_s),
        slope_wild_t=float(t_w), slope_wild_p=float(p_w),
        d_lin_lowsigma=float(d_lo), d_lin_highsigma=float(d_hi), b1=b1, b2=b2,
        b3=b3, n_flip=int(n_flip), crossing_crb_median=med_cross,
        per_lambda={str(t): {kk: (vv if not isinstance(vv, np.ndarray)
                                  else vv.tolist()) for kk, vv in v.items()}
                    for t, v in per_lambda.items()},
        partial_spearman_crb_dlin=partial_rho, per_lambda_spearman=per_lam_spear,
        ols_logcrb_beta=float(bj[1]), ols_logcrb_p=float(pj[1]),
        ols_ar_beta=float(bj[2]), ols_ar_p=float(pj[2]),
        confound_spearman_ar_dlin=[float(rho_conf), float(p_conf)],
        density=dens)},
        open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
