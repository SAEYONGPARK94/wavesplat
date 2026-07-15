"""E-25 (deployable method) — the CRB-GATED HYBRID estimator.

Turns the session's DIAGNOSIS into an operational METHOD. The diagnosis (E-15..
E-24): the learned prior (net) beats the classical hand-MMSE ONLY where the field
is UNDER-determined (long wavelength / small aperture ratio / high
field-reconstruction CRB) and LOSES where WELL-determined (short lambda / low
CRB). So a principled estimator should trust the CLASSICAL arm where the CRB is
low and the LEARNED arm where the CRB is high. The Cramer-Rao bound is computable
from the observation geometry alone (no ground truth), so it is a *deployable*
gate, not an oracle.

  hybrid(cell) = hand_corr  if  crb(cell) <  tau
                 net_corr   if  crb(cell) >= tau

This makes the CRB OPERATIONALLY LOAD-BEARING: it decides which estimator to
trust, per observation window, rather than merely describing the regime.

ARENA. Fixed HOS-arena fan operator (E-15/E-16). Aperture axis spans BOTH regimes:
  Tp in {6,7,8,9,10,11,12,13}  ->  lambda ~56..231 m, aperture_ratio R/lam 3.56..0.87
(the spec's Tp {6,7,8,9,11,13} span, FILLED to 8 distinct wavelengths: the G1
significance clusters on the SEA, so cluster count is the binding constraint on
power -> 8 clusters not 6, still within the same aperture span, consistent with
the E-15 grid.) At fixed steepness ~0.025 (short seas well-det, long seas under-det),
Hs = round(0.025*lambda) CAPPED at HS_CAP to keep the longest-lambda HOS M3 finite.
>=4 phase realisations per sea. A "cell" = (sea, phase). Score = corr on the M3
(realistic nonlinear) forecast field, draw-averaged over D observation draws.

Per cell we record: hand_corr, net_corr (draw-averaged and per-draw), the
field-reconstruction nRMSE CRB (deterministic, from a fixed-seed 600-pt draw
geometry + JONSWAP linear modes via d1_crb), oracle_corr = max(hand,net), and
aperture_ratio (a control gate variable).

CPU raycast builds (build_cell, order=3 only) run in a ProcessPool BEFORE any GPU
init; the CRB (d1_crb, tiny 240x240 FIM) and the two estimator draws run serially
in the main process afterwards.

--- PRE-REGISTERED GATES ----------------------------------------------------
Units: N_cells = 8 seas x 4 phases = 32 (nan-filtered). Gate variable g = crb;
convention low-g -> hand. tau-grid = midpoints of the sorted per-cell CRBs plus
+/-inf sentinels (so the in-sample optimum may collapse to a PURE estimator ->
"beats both pure" is a nontrivial claim). All corrs are draw-averaged (D draws)
unless a per-draw split is named.

G1  DEPLOYABLE HYBRID BEATS BOTH PURE, out of sample.
    Leave-one-SEA-out CV: for each held-out sea, pick tau* = argmax pooled hybrid
    skill on the OTHER 5 seas' cells; apply that tau* to the held-out sea's cells;
    collect the held-out hybrid_cv corrs. better_pure := the pure arm (hand or
    net) with the higher POOLED mean (a single global choice).
    PASS iff BOTH:
      (i)  pooled(hybrid_cv) > max(pooled_hand, pooled_net)          [beats both]
      (ii) one-sided p < 0.05 for (hybrid_cv - better_pure) across the 8 SEA-level
           paired means (t(7)). ALSO reported (not gating): a Webb wild CLUSTER
           bootstrap-t on the cell-level diffs clustered on sea, Wilcoxon, sign
           count, and the draw-noise SE of the pooled margin.
    Rationale: beating the *better* pure arm requires exploiting the crossover
    (hand wins some seas, net wins others). The gain concentrates in the few most
    under-determined seas, so with better_pure absorbing the rest as ~0 diffs the
    sea-clustered test is the WEAKEST link -> see self-review #1.

G2  THE CRB IS LOAD-BEARING (it predicts the crossover the gate needs).
    PASS iff BOTH:
      (a) Spearman(crb, net_corr - hand_corr) > 0 with p < 0.05 over cells (higher
          CRB -> net increasingly favoured). Sea-level Spearman (8 pts) reported.
      (b) The skill-optimal tau* (in-sample, pooled) and the ORACLE-AGREEMENT
          optimal tau_x (the tau maximising per-cell agreement between the gate's
          choice and the actually-better arm) select the SAME partition:
          agreement(tau*) >= 0.70 AND |tau* - tau_x| <= median inter-cell CRB gap.

OVERALL PASS = G1 AND G2.

G3  (reported, NOT gating — the two ways this could mislead, quantified)
    (a) ORACLE APPROACH. recovery := (pooled(hybrid_cv) - better_pure_pooled) /
        (oracle_splithalf_pooled - better_pure_pooled). oracle_splithalf DECIDES
        the arm on the first D/2 draws and EVALUATES on the last D/2 (de-biases the
        max()-capitalisation of the naive oracle). Report recovery >= 0.5 target.
    (b) CRB vs APERTURE gate. Re-run the identical LOSO-CV gating on g = -aperture
        _ratio (the raw wavelength split). Report pooled(hybrid_cv) for CRB gate
        vs aperture gate, and Spearman(crb, aperture_ratio). The CRB gate need only
        be NOT WORSE than the aperture gate; their collinearity is a MECHANISM
        note, not a refutation (at deployment you compute the CRB from geometry;
        it also resolves *within*-sea differences the pure-Tp split cannot).

  python3 tmp/e25_crb_gated.py                 # FULL run (heavy: GPU + builds)
  python3 tmp/e25_crb_gated.py --selftest      # analysis logic only, no GPU (<1s)
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
os.environ.setdefault('OMP_NUM_THREADS', '1')          # keep build workers 1-core

# e18 import is torch-free (only numpy/scipy at its top level) -> safe for
# --selftest, which never touches the GPU.
from e18_decorrelated_2factor import build_cell, _lam, R_APERTURE  # noqa: E402

OUT = 'results/direction3/e25_crb_gated.json'
TPS = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]   # spec span, filled to 8 seas
STEEP = 0.025
HS_CAP = 4.5                    # cap Hs at the long-lambda end so M3 stays finite
NOISE = 0.05                    # elevation noise (matches E-15/E-19 pipeline)
DEPTH = 50.0
AGREE_MIN = 0.70               # G2(b) threshold
RECOVERY_TGT = 0.50            # G3(a) target


# ==========================================================================
#  Per-cell CRB (deterministic): fixed-seed 600-pt draw geometry + JONSWAP modes
# ==========================================================================
def _draw_obs_geom(sea, rng, n_obs):
    """The E-15/E-19 observation selection, positions+times only (no z, no
    estimator). Return-power lottery -> range-stratified n_obs points."""
    from d3_door1b_field_bias import R_MIN, R_MAX
    from wavesim.raydrop_threshold import T_MANUAL_50
    P, rr = [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep],
                                  np.full(int(keep.sum()), sc['trel'])]))
        rr.append(sc['r'][keep])
    P, rr = np.vstack(P), np.concatenate(rr)
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-n_obs // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    return P[np.concatenate(take)]


def cell_crb(sea, seed, Hs, n_obs, rng_seed=700000):
    """Field-reconstruction nRMSE CRB (any unbiased estimator) from the fan obs
    geometry + JONSWAP linear free-wave modes. Property of the OPERATOR + field,
    independent of the estimator (d1_crb). Deterministic per cell."""
    import torch
    import d1_crb
    from d3_f1_hos_door1 import CX, CY, sea_components
    from d3_door1b_field_bias import R_MIN, R_MAX, FOV_H
    DT, DEV = d1_crb.DT, d1_crb.DEVICE
    P = _draw_obs_geom(sea, np.random.default_rng(rng_seed), n_obs)
    S = sea_components(seed, Hs=Hs)                        # [A, kx, ky, phi] modes
    theta = torch.tensor(np.stack([S['A'], S['kx'], S['ky'], S['ph']], 1),
                         dtype=DT, device=DEV)
    x = torch.tensor(CX + P[:, 0], dtype=DT, device=DEV)
    y = torch.tensor(CY + P[:, 1], dtype=DT, device=DEV)
    t = torch.tensor(P[:, 2], dtype=DT, device=DEV)
    F = d1_crb.fisher(theta, x, y, t, DEPTH, NOISE)
    # reconstruction design G = <g g^T> over the fan region at t*=0 (E-19 convention)
    rr = np.sqrt(np.random.default_rng(1).uniform((R_MIN / R_MAX) ** 2, 1, 400)) * R_MAX
    aa = np.random.default_rng(2).uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 400)
    gx = torch.tensor(CX + rr * np.cos(aa), dtype=DT, device=DEV)
    gy = torch.tensor(CY + rr * np.sin(aa), dtype=DT, device=DEV)
    Jg = d1_crb.jacobian(theta, gx, gy, torch.zeros_like(gx), DEPTH)
    G = (Jg.T @ Jg) / Jg.shape[0]
    crb, _ = d1_crb.field_nrmse_crb(F, G, Hs)
    return float(crb)


# ==========================================================================
#  Analysis (PURE numpy/scipy — importable without torch; drives --selftest)
# ==========================================================================
def _tau_grid(g):
    cs = np.unique(g)
    if len(cs) < 2:
        return np.array([cs[0] - 1e-9, cs[0] + 1e-9])
    mids = (cs[:-1] + cs[1:]) / 2.0
    return np.concatenate([[cs.min() - 1e-9], mids, [cs.max() + 1e-9]])


def _hybrid(g, H, Nn, tau):
    return np.where(g < tau, H, Nn)


def _best_tau_skill(g, H, Nn):
    grid = _tau_grid(g)
    sk = np.array([_hybrid(g, H, Nn, t).mean() for t in grid])
    i = int(np.argmax(sk))
    return float(grid[i]), float(sk[i])


def _agreement(g, H, Nn, tau):
    return float(np.mean((g < tau) == (H >= Nn)))


def _best_tau_agree(g, H, Nn):
    grid = _tau_grid(g)
    ag = np.array([_agreement(g, H, Nn, t) for t in grid])
    i = int(np.argmax(ag))
    return float(grid[i]), float(ag[i])


def _loso_gate(g, H, Nn, sea):
    """Leave-one-SEA-out CV gating. tau chosen on training seas' skill, applied
    out-of-sample to the held-out sea. Returns per-cell hybrid_cv corrs."""
    seas = sorted(set(sea))
    hyb = np.full(len(g), np.nan)
    tau_used = {}
    for s in seas:
        tr, te = (sea != s), (sea == s)
        tau, _ = _best_tau_skill(g[tr], H[tr], Nn[tr])
        hyb[te] = _hybrid(g[te], H[te], Nn[te], tau)
        tau_used[s] = tau
    return hyb, tau_used


def _sea_means(v, sea, seas):
    return np.array([np.mean(v[sea == s]) for s in seas])


def _cr_t(y, sea, seas):
    """Cluster-robust t (clusters = seas), null-imposed form for the wild boot."""
    k, n = len(seas), len(y)
    mu = y.mean()
    acc = sum((y[sea == s].sum() - (sea == s).sum() * mu) ** 2 for s in seas)
    c = (k / (k - 1)) * ((n - 1) / (n - k)) if (k > 1 and n > k) else 1.0
    se = np.sqrt(c * acc / n ** 2)
    return mu / se if se > 0 else np.inf


def _wild_cluster_p(y, sea, seas, nboot=10000, seed=0):
    """Webb wild CLUSTER bootstrap-t (one weight per sea), null imposed (yb=y*w).
    Two-sided p. Valid but low-powered at small #clusters (see e16 note)."""
    rng = np.random.default_rng(seed)
    WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])
    idx = {s: i for i, s in enumerate(seas)}
    ci = np.array([idx[s] for s in sea])
    t_obs = _cr_t(y, sea, seas)
    boot = np.array([_cr_t(y * rng.choice(WEBB, len(seas))[ci], sea, seas)
                     for _ in range(nboot)])
    return float(t_obs), float((np.abs(boot) >= abs(t_obs)).mean())


def _draw_boot_margin_se(rows, tau_used, better, seed=0, nboot=2000):
    """SE of the pooled (hybrid_cv - better_pure) margin from DRAW noise only:
    resample the D draws within each cell, recompute per-cell hand/net means and
    the CV-gated pick, recompute the pooled margin. Isolates the reducible Monte-
    Carlo component (orthogonal to the sea-sampling uncertainty the CV probes)."""
    if not all('draws_hand' in r for r in rows):
        return np.nan
    rng = np.random.default_rng(seed)
    DH = [np.array(r['draws_hand'], float) for r in rows]
    DN = [np.array(r['draws_net'], float) for r in rows]
    C = np.array([r['crb'] for r in rows])
    sea = [r['sea'] for r in rows]
    margins = []
    for _ in range(nboot):
        hyb, bp = [], []
        for i, r in enumerate(rows):
            D = len(DH[i])
            bi = rng.integers(0, D, D)
            h, nn = DH[i][bi].mean(), DN[i][bi].mean()
            tau = tau_used[sea[i]]
            hyb.append(h if C[i] < tau else nn)
            bp.append(h if better == 'hand' else nn)
        margins.append(np.mean(hyb) - np.mean(bp))
    return float(np.std(margins, ddof=1))


def analyse(rows, verbose=True):
    rows = [r for r in rows if np.isfinite(r['hand']) and np.isfinite(r['net'])
            and np.isfinite(r['crb'])]
    H = np.array([r['hand'] for r in rows])
    Nn = np.array([r['net'] for r in rows])
    C = np.array([r['crb'] for r in rows])
    AR = np.array([r['aperture_ratio'] for r in rows])
    sea = np.array([r['sea'] for r in rows])
    seas = sorted(set(sea))
    n = len(rows)

    pooled_hand, pooled_net = float(H.mean()), float(Nn.mean())
    better = 'hand' if pooled_hand >= pooled_net else 'net'
    better_pooled = max(pooled_hand, pooled_net)
    better_v = H if better == 'hand' else Nn

    # ---- G1: LOSO-CV CRB gate vs both pure -------------------------------
    hyb_cv, tau_used = _loso_gate(C, H, Nn, sea)
    pooled_hyb_cv = float(np.nanmean(hyb_cv))
    d_sea = _sea_means(hyb_cv, sea, seas) - _sea_means(better_v, sea, seas)
    t_g1, p_two = stats.ttest_1samp(d_sea, 0.0)
    p_one = p_two / 2.0 if t_g1 > 0 else 1.0 - p_two / 2.0
    try:
        w_stat, w_p = stats.wilcoxon(d_sea, alternative='greater')
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    sign_ge = int(np.sum(d_sea >= -1e-12))
    # reported robustness: wild cluster bootstrap-t + draw-noise SE of the margin
    diff_cell = hyb_cv - better_v
    t_wild, p_wild = _wild_cluster_p(diff_cell, sea, seas)
    margin_draw_se = _draw_boot_margin_se(rows, tau_used, better)
    g1 = bool(pooled_hyb_cv > better_pooled and p_one < 0.05)

    # ---- G2: CRB predicts the crossover ---------------------------------
    delta = Nn - H
    rho_c, p_c = stats.spearmanr(C, delta)
    # sea-level (6 pts): mean crb vs mean delta per sea
    csea = _sea_means(C, sea, seas)
    dsea = _sea_means(delta, sea, seas)
    rho_cs, p_cs = stats.spearmanr(csea, dsea)
    tau_skill, skill_at = _best_tau_skill(C, H, Nn)      # in-sample optimum
    tau_agree, agree_at = _best_tau_agree(C, H, Nn)
    agree_at_skill = _agreement(C, H, Nn, tau_skill)
    gap = float(np.median(np.diff(np.sort(np.unique(C))))) if len(set(C)) > 1 else np.inf
    g2b = bool(agree_at_skill >= AGREE_MIN and abs(tau_skill - tau_agree) <= gap + 1e-12)
    g2 = bool(rho_c > 0 and p_c < 0.05 and g2b)

    # ---- G3(a): oracle approach (split-half de-biased) ------------------
    oracle_full = float(np.mean(np.maximum(H, Nn)))
    osh = []
    have_draws = all('draws_hand' in r and 'draws_net' in r for r in rows)
    if have_draws:
        for r in rows:
            dh = np.array(r['draws_hand'], float)
            dn = np.array(r['draws_net'], float)
            h = len(dh) // 2
            if h >= 1:
                pick_hand = dh[:h].mean() >= dn[:h].mean()   # decide on first half
                osh.append(dh[h:].mean() if pick_hand else dn[h:].mean())  # eval 2nd
        oracle_sh = float(np.mean(osh)) if osh else np.nan
    else:
        oracle_sh = np.nan
    denom = (oracle_sh if np.isfinite(oracle_sh) else oracle_full) - better_pooled
    recovery = float((pooled_hyb_cv - better_pooled) / denom) if abs(denom) > 1e-9 else np.nan

    # ---- G3(b): aperture-ratio gate control -----------------------------
    # low-g -> hand; well-determined = HIGH aperture_ratio -> gate on g=-AR.
    hyb_ar, _ = _loso_gate(-AR, H, Nn, sea)
    pooled_hyb_ar = float(np.nanmean(hyb_ar))
    rho_ca, p_ca = stats.spearmanr(C, AR)

    S = dict(
        n_cells=n, n_seas=len(seas), seas=seas,
        pooled_hand=pooled_hand, pooled_net=pooled_net,
        better_pure=better, better_pooled=better_pooled,
        pooled_hybrid_cv=pooled_hyb_cv,
        margin_vs_hand=pooled_hyb_cv - pooled_hand,
        margin_vs_net=pooled_hyb_cv - pooled_net,
        margin_vs_better=pooled_hyb_cv - better_pooled,
        g1_t=float(t_g1), g1_p_onesided=float(p_one),
        g1_wilcoxon_p=float(w_p), g1_sign_ge=sign_ge,
        g1_wildcluster_t=float(t_wild), g1_wildcluster_p=float(p_wild),
        g1_margin_draw_se=float(margin_draw_se), G1_pass=g1,
        spearman_crb_delta=[float(rho_c), float(p_c)],
        spearman_crb_delta_sea=[float(rho_cs), float(p_cs)],
        tau_skill=tau_skill, tau_agree=tau_agree,
        agreement_at_tau_skill=float(agree_at_skill),
        best_agreement=float(agree_at), crb_gap=float(gap), G2b_pass=g2b,
        G2_pass=g2,
        oracle_full=oracle_full, oracle_splithalf=float(oracle_sh),
        recovery_fraction=recovery, recovery_target=RECOVERY_TGT,
        pooled_hybrid_cv_aperture_gate=pooled_hyb_ar,
        crb_gate_ge_aperture_gate=bool(pooled_hyb_cv >= pooled_hyb_ar - 1e-9),
        spearman_crb_aperture=[float(rho_ca), float(p_ca)],
        tau_used_per_fold={k: float(v) for k, v in tau_used.items()},
        OVERALL_PASS=bool(g1 and g2),
    )

    if verbose:
        _print_report(rows, S, H, Nn, C, AR, sea, seas, hyb_cv, better)
    return S


def _print_report(rows, S, H, Nn, C, AR, sea, seas, hyb_cv, better):
    print(f"\n{'='*80}\nE-25 CRB-gated hybrid   ({S['n_cells']} cells, "
          f"{S['n_seas']} seas)\n{'='*80}")
    print(f"  {'sea':<13}{'ar':>6}{'crb':>8}{'hand':>8}{'net':>8}"
          f"{'net-hand':>10}{'hyb_cv':>9}")
    for s in seas:
        m = sea == s
        print(f"  {s:<13}{AR[m][0]:6.2f}{C[m].mean():8.3f}{H[m].mean():+8.3f}"
              f"{Nn[m].mean():+8.3f}{(Nn[m]-H[m]).mean():+10.3f}"
              f"{np.nanmean(hyb_cv[m]):+9.3f}")
    print(f"\n  pooled  hand {S['pooled_hand']:+.4f}   net {S['pooled_net']:+.4f}"
          f"   (better pure = {better})")
    print(f"  pooled  hybrid_cv (CRB gate)  {S['pooled_hybrid_cv']:+.4f}   "
          f"margin vs better {S['margin_vs_better']:+.4f}")
    print(f"  G1 sea-level paired (hyb_cv - {better}): t {S['g1_t']:+.2f} "
          f"p1 {S['g1_p_onesided']:.4f}  Wilcoxon p {S['g1_wilcoxon_p']:.4f}  "
          f"sign>=0 {S['g1_sign_ge']}/{S['n_seas']}")
    print(f"     robustness: wild-cluster t {S['g1_wildcluster_t']:+.2f} "
          f"p {S['g1_wildcluster_p']:.4f}  |  margin {S['margin_vs_better']:+.4f} "
          f"+/- {S['g1_margin_draw_se']:.4f} (draw-noise SE)")
    print(f"  *** G1 (hybrid beats BOTH pure, out-of-sample): "
          f"{'PASS' if S['G1_pass'] else 'FAIL'}")
    print(f"\n  Spearman(crb, net-hand) cell {S['spearman_crb_delta'][0]:+.3f} "
          f"p {S['spearman_crb_delta'][1]:.4f}  | sea "
          f"{S['spearman_crb_delta_sea'][0]:+.3f} p {S['spearman_crb_delta_sea'][1]:.4f}")
    print(f"  tau*(skill) {S['tau_skill']:.3f}  tau_x(agree) {S['tau_agree']:.3f}  "
          f"gap {S['crb_gap']:.3f}  agree@tau* {S['agreement_at_tau_skill']:.2f}")
    print(f"  *** G2 (CRB predicts the crossover): "
          f"{'PASS' if S['G2_pass'] else 'FAIL'}")
    print(f"\n  G3(a) oracle full {S['oracle_full']:+.4f}  split-half "
          f"{S['oracle_splithalf']:+.4f}  recovery {S['recovery_fraction']:.2f} "
          f"(tgt {S['recovery_target']})")
    print(f"  G3(b) CRB gate {S['pooled_hybrid_cv']:+.4f}  vs aperture gate "
          f"{S['pooled_hybrid_cv_aperture_gate']:+.4f}  "
          f"(CRB>=aper: {S['crb_gate_ge_aperture_gate']})  "
          f"rho(crb,ar) {S['spearman_crb_aperture'][0]:+.3f}")
    print(f"\n  *** OVERALL {'PASS' if S['OVERALL_PASS'] else 'FAIL'} "
          f"(G1 AND G2)\n")


# ==========================================================================
#  Self-test: fabricate a crossover dataset, exercise the full analysis (no GPU)
# ==========================================================================
def selftest():
    rng = np.random.default_rng(0)
    D = 6
    rows = []
    for i, Tp in enumerate(TPS):
        lam = _lam(Tp)
        ar = R_APERTURE / lam
        x = i / (len(TPS) - 1)                       # 0 (short lam) .. 1 (long lam)
        crb0 = 0.05 + 0.9 * x                        # CRB rises with lambda
        hand0 = 0.90 - 0.70 * x                      # hand: strong when well-det
        net0 = 0.60 - 0.10 * x                        # net: flatter -> wins at high x
        for seed in range(4):
            dh = list(hand0 + rng.normal(0, 0.02, D))
            dn = list(net0 + rng.normal(0, 0.02, D))
            crb = crb0 + rng.normal(0, 0.01)
            rows.append(dict(sea=f'Tp{Tp:g}', Tp=Tp, Hs=0.0, lam=lam,
                             steepness=STEEP, aperture_ratio=ar, seed=seed,
                             hand=float(np.mean(dh)), net=float(np.mean(dn)),
                             crb=float(crb), oracle=float(max(np.mean(dh), np.mean(dn))),
                             draws_hand=dh, draws_net=dn))
    print("SELFTEST — synthetic crossover (should PASS G1 & G2; logic check only)")
    S = analyse(rows, verbose=True)
    ok = S['G1_pass'] and S['G2_pass']
    print(f"SELFTEST analysis logic exercised end-to-end; synthetic OVERALL "
          f"{'PASS' if ok else 'FAIL'} (expected PASS).")
    return 0


# ==========================================================================
#  Full run (heavy: parallel CPU builds -> GPU draws -> CRB)
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--draws', type=int, default=6)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    n_obs = int(os.environ['A1_N_OBS'])
    tps = TPS[:2] if a.smoke else TPS
    seeds = a.seeds[:1] if a.smoke else a.seeds
    draws = 2 if a.smoke else a.draws
    seas = [(Tp, min(round(STEEP * _lam(Tp), 3), HS_CAP)) for Tp in tps]
    tasks = [(Tp, Hs, sd, 3) for (Tp, Hs) in seas for sd in seeds]
    print(f"E-25: {len(seas)} seas x {len(seeds)} phases = {len(tasks)} M3 cells; "
          f"{draws} draws/cell; {a.workers} build workers; n_obs {n_obs}",
          flush=True)

    # ---- parallel CPU raycast builds (order=3 only), before any GPU init ----
    t0 = time.time()
    built = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for task, sea, note, dt in ex.map(build_cell, tasks):
            built[task] = sea
            if sea is None:
                print(f"  !! build {task} -> {note} [{dt:.0f}s]", flush=True)
    n_ok = sum(v is not None for v in built.values())
    print(f"  built {n_ok}/{len(tasks)} M3 cells in {(time.time()-t0)/60:.1f} min",
          flush=True)

    # ---- GPU estimators + CRB (serial, main process) ----
    import torch
    from e15_hos_multidraw import one_draw, CKPT
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp)
        ar = R_APERTURE / lam
        for sd in seeds:
            sea = built[(Tp, Hs, sd, 3)]
            if sea is None:
                continue
            dh, dn = [], []
            for r in range(draws):
                rng = np.random.default_rng(250000 + 7919 * sd + 31 * r)
                h, n_, _ = one_draw(sea, rng, net, head)
                dh.append(float(h)); dn.append(float(n_))
            crb = cell_crb(sea, sd, Hs, n_obs)
            hand_c, net_c = float(np.nanmean(dh)), float(np.nanmean(dn))
            rows.append(dict(sea=f'Tp{Tp:g}', Tp=float(Tp), Hs=float(Hs),
                             lam=lam, steepness=Hs / lam, aperture_ratio=ar,
                             seed=sd, hand=hand_c, net=net_c, crb=crb,
                             oracle=max(hand_c, net_c), draws_hand=dh, draws_net=dn))
            print(f"[Tp{Tp:g} lam{lam:.0f} ar{ar:.2f} ph{sd}] crb {crb:.3f}  "
                  f"hand {hand_c:+.3f}  net {net_c:+.3f}  net-hand {net_c-hand_c:+.3f}",
                  flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    n_all = len(rows)
    rows = [r for r in rows if np.isfinite(r['hand']) and np.isfinite(r['net'])
            and np.isfinite(r['crb'])]
    if n_all - len(rows):
        print(f"  nan-filtered {n_all-len(rows)} cells (M3 blow-up / degenerate)")
    if len({r['sea'] for r in rows}) < 3:
        print("too few seas for LOSO analysis"); return 0

    S = analyse(rows, verbose=True)
    json.dump({'rows': rows, 'summary': S}, open(a.out, 'w'), indent=2)
    print(f"-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
