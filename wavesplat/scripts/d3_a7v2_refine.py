"""A-7 v2 -- continuous-k refine-select hybrid (method candidate B-1, 1st demo).
Frozen prereg: docs/A7_D20_prereg_2026-07-12.md ("A-7 v2 사전등록" section).

A-7 v1 FAILED for two reasons:
  1. detect statistic (beamform top-1 cell fraction) is kp-dependent -> BF2 never
     fires with the BF3-5-tuned theta.
  2. even force-fired, the beamform ARGMAX sits on a sidelobe in 3/5 seeds
     (|k0| 1.0-1.76 vs true 0.77); only seed1 (argmax correct) recovered
     0.008->0.84. The oracle single true-k atom gives corr 0.993 -> the
     information exists, the peak super-resolution is the bottleneck.

v2 design (FROZEN):
  1. REFINE-SELECT: beamform power map -> top M=8 local peaks (non-max
     suppression to separate them); refine each candidate by continuous-k
     single-component least squares (dispersion-constrained, closed-form 2-coeff
     cos/sin, (kx,ky) by Nelder-Mead); SELECT by held-out (20%) observation
     residual, NOT by peak power. Sidelobe candidates should fit the holdout
     worse after refinement.
  2. DETECT v2 (kp-invariant): a fit-gap statistic on the refined candidates
     (single-component explained variance EV1 / 1st-vs-2nd holdout residual
     ratio). Threshold calibrated on BF3/BF4 1comp-vs-5comp ONLY (BF2 = judgment
     set, never used to calibrate).
  3. HYBRID: on detection splice dk=kp/40 fine atoms around the REFINED k into
     solve_mmse_bank; else the existing hand path verbatim (delta=0 harmless).

Gates (FROZEN, inherited from A-7):
  G-A7v2-1 recovery : BF2 1comp fixedT x5, field corr(lead0)>=0.5 AND encounter
                      rho>=0.5, in the MAJORITY (>=3/5).
  G-A7v2-2 harmless : 6 synth dfield>=-0.01 AND YS17 12 windows dencounter>=-0.02.
  G-A7v2-3 mechanism: per-seed selected-k error |dk|/k_true + sidelobe rejection
                      table (diagnostic, not a gate).

python3 scripts/d3_a7v2_refine.py            # full run
python3 scripts/d3_a7v2_refine.py --smoke    # BF2 + synth only (skip real spot)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, DEV, build_bank, beamform_prior, design, G
# reuse all of v1's data / query / scoring / augmented-solve plumbing verbatim
from d3_a7_finebank import (load_synth, build_queries, field_track_corr,
                            solve_mmse_bank, hand_predict, APP_EPISODES,
                            U_SHIP, LEADS)

OUT = 'results/direction3/a7v2_refine_eval.json'
DATE = '2026-07-12'
M_PEAKS = 8              # top-M beamform candidates (frozen)
VAL_FRAC = 0.20         # holdout for refine-SELECT (frozen: "20% 홀드아웃")
KREL, ANGDEG, FINE_DIV = 0.25, 15.0, 40    # fine-bank geometry (v1 spec)
NMS_SUPP = 0.40         # non-max suppression radius (fraction of kp)
DEDUP = 0.08            # merge refined candidates within this |dk| (abs rad/m)

# Data root, overridable with --data_dir. Default 'data' preserves historical behaviour.
# The 2026-07-23 retry runs against data_finemesh_fixedT_2026-07-23/, which regenerates the
# gate set AND the calibration sets at the Nyquist-adequate mesh with the SAME 8000
# obs/frame cap -- the observation-count confound that invalidated the July retry
# (docs/E2_MESH_AUDIT_prereg_2026-07-13.md:94-101) is thereby controlled, verified by
# comparing per-frame return counts (8000 in both, cap binds).
DATA_DIR = 'data'

CAL_POS = ['{d}/episode_BF3_1comp_seed%d_fixedT.h5',
           '{d}/episode_BF4_1comp_seed%d_fixedT.h5']
CAL_NEG = ['{d}/episode_BF3_5comp_seed%d_fixedT.h5',
           '{d}/episode_BF4_5comp_seed%d_fixedT.h5']


# ===================================================== continuous-k refine-select
def _single_fit(kx, ky, P, z, depth, lam=1e-8):
    """Closed-form 2-coeff (cos,sin) LS of ONE dispersion-constrained component.
    Returns (coef[2], residual)."""
    k = math.hypot(kx, ky)
    om = math.sqrt(G * k * math.tanh(k * depth) + 1e-12)
    psi = kx * P[:, 0] + ky * P[:, 1] - om * P[:, 2]
    D = np.stack([np.cos(psi), np.sin(psi)], 1)
    A = D.T @ D + lam * np.eye(2)
    coef = np.linalg.solve(A, D.T @ z)
    return coef, z - D @ coef


def _eval_resid(kx, ky, coef, P, z, depth):
    """Residual of a FIXED-coeff single component on a held-out split."""
    k = math.hypot(kx, ky)
    om = math.sqrt(G * k * math.tanh(k * depth) + 1e-12)
    psi = kx * P[:, 0] + ky * P[:, 1] - om * P[:, 2]
    return z - (coef[0] * np.cos(psi) + coef[1] * np.sin(psi))


def beamform_grid(Pt, zt, kp, depth, b_hi=2.5, n=61):
    """Regular 2D beamform power map over [-b_hi kp, b_hi kp]^2, out-of-band
    (annulus 0.35-2.5 kp) zeroed. Returns KX,KY,PW as n x n numpy arrays."""
    ax = torch.linspace(-b_hi * kp, b_hi * kp, n, device=DEV)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy')
    kx, ky = KX.reshape(-1), KY.reshape(-1)
    k = torch.hypot(kx, ky)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-12)
    x, y, t = Pt[:, 0], Pt[:, 1], Pt[:, 2]
    pw = torch.empty_like(kx)
    B = 2048
    for i in range(0, len(kx), B):
        ps = (kx[i:i + B, None] * x[None] + ky[i:i + B, None] * y[None]
              - om[i:i + B, None] * t[None])
        pw[i:i + B] = ((zt[None] * torch.cos(ps)).sum(1) ** 2
                       + (zt[None] * torch.sin(ps)).sum(1) ** 2)
    inband = (k > 0.35 * kp) & (k < 2.5 * kp)
    pw = torch.where(inband, pw, torch.zeros_like(pw))
    return (KX.cpu().numpy(), KY.cpu().numpy(), pw.reshape(n, n).cpu().numpy())


def top_m_peaks(KXg, KYg, PWg, kp, M=M_PEAKS, supp=NMS_SUPP):
    """M strongest local maxima by greedy argmax + neighbourhood suppression."""
    kxf, kyf, pf = KXg.ravel(), KYg.ravel(), PWg.ravel().copy()
    peaks = []
    for _ in range(M):
        j = int(np.argmax(pf))
        if pf[j] <= 0:
            break
        peaks.append((float(kxf[j]), float(kyf[j])))
        d = np.hypot(kxf - kxf[j], kyf - kyf[j])
        pf[d < supp * kp] = -1.0
    return peaks


def _refine_single(kx0, ky0, P, z, depth):
    """Continuous-k refine of ONE component: minimise train residual RMS over
    (kx,ky) (dispersion constrained; coeffs closed-form each eval)."""
    def obj(p):
        _, r = _single_fit(p[0], p[1], P, z, depth)
        return float(np.sqrt(np.mean(r ** 2)))
    res = minimize(obj, np.array([kx0, ky0]), method='Nelder-Mead',
                   options=dict(xatol=1e-4, fatol=1e-7, maxiter=300))
    return float(res.x[0]), float(res.x[1])


def refine_select(P, z, depth, kp, M=M_PEAKS, val_frac=VAL_FRAC, seed=0):
    """Full refine-select: beamform peaks -> continuous refine each -> select by
    holdout residual. Returns winner k, ranked candidates, detect statistics."""
    n = len(z)
    g = np.random.default_rng(seed)
    vi = g.choice(n, max(8, int(val_frac * n)), replace=False)
    vm = np.zeros(n, dtype=bool); vm[vi] = True
    Ptr, ztr = P[~vm], z[~vm]
    Pv, zv = P[vm], z[vm]
    Pt = torch.tensor(Ptr, device=DEV, dtype=torch.float32)
    zt = torch.tensor(ztr, device=DEV, dtype=torch.float32)

    KXg, KYg, PWg = beamform_grid(Pt, zt, kp, depth)
    peaks = top_m_peaks(KXg, KYg, PWg, kp, M)
    # beamform argmax (v1's peak) for the diagnostic
    argmax_k = peaks[0] if peaks else (0.0, 0.0)
    argmax_kabs = float(math.hypot(*argmax_k))

    cands = []
    for (px, py) in peaks:
        rx, ry = _refine_single(px, py, Ptr, ztr, depth)
        coef, _ = _single_fit(rx, ry, Ptr, ztr, depth)   # train coeffs
        rv = _eval_resid(rx, ry, coef, Pv, zv, depth)      # holdout residual
        vr = float(np.sqrt(np.mean(rv ** 2)))
        cands.append(dict(kx=rx, ky=ry, kabs=float(math.hypot(rx, ry)),
                          val_resid=vr, amp=float(math.hypot(*coef))))
    cands.sort(key=lambda c: c['val_resid'])
    # dedup refined candidates that converged to the same component
    kept = []
    for c in cands:
        if all(math.hypot(c['kx'] - k['kx'], c['ky'] - k['ky']) > DEDUP
               for k in kept):
            kept.append(c)
    while len(kept) < 2:                                   # guarantee a runner-up
        kept.append(dict(kx=0.0, ky=0.0, kabs=0.0,
                         val_resid=float(np.sqrt(np.mean(zv ** 2))), amp=0.0))

    win, run = kept[0], kept[1]
    var_zv = float(np.var(zv)) + 1e-12
    ev1 = 1.0 - win['val_resid'] ** 2 / var_zv             # explained variance
    ratio = win['val_resid'] / (run['val_resid'] + 1e-12)  # 1st/2nd resid ratio
    return dict(
        winner=(win['kx'], win['ky']), winner_kabs=win['kabs'],
        runner=(run['kx'], run['ky']),
        argmax_k=argmax_k, argmax_kabs=argmax_kabs,
        val_resid_win=win['val_resid'], val_resid_run=run['val_resid'],
        score_ev1=float(ev1), score_ratio=float(1.0 - ratio),
        resid_ratio=float(ratio), n_cand=len(kept),
        cands=[dict(kx=c['kx'], ky=c['ky'], kabs=c['kabs'],
                    val_resid=c['val_resid']) for c in kept[:M]])


# ==================================================================== fine bank
def augment_finebank_v2(Pt, zt, depth, kp, k0x, k0y):
    """Splice a dk=kp/40 grid around the REFINED k (v2 centre); the combined
    bank's prior is the beamform power recomputed over all atoms (consistent
    scale). mmse_gpu is untouched -- solve_mmse_bank consumes this."""
    kxb, kyb, om = build_bank(kp, (0.35, 2.5), kp / 10.0, depth, DEV,
                              torch.float32)
    k0 = math.hypot(k0x, k0y)
    dk = kp / FINE_DIV
    half = KREL * k0
    axf = np.arange(-half, half + dk / 2, dk)
    FX, FY = np.meshgrid(axf, axf)
    fkx, fky = k0x + FX.ravel(), k0y + FY.ravel()
    fk = np.hypot(fkx, fky)
    ang0 = math.atan2(k0y, k0x)
    ang = np.arctan2(fky, fkx)
    dang = np.abs(np.arctan2(np.sin(ang - ang0), np.cos(ang - ang0)))
    keep = (np.abs(fk - k0) / k0 <= KREL) & (dang <= np.deg2rad(ANGDEG)) \
        & (fk > 1e-6)
    fkx, fky = fkx[keep], fky[keep]
    fk = np.hypot(fkx, fky)
    fom = np.sqrt(G * fk * np.tanh(fk * depth))
    kxb2 = torch.cat([kxb, torch.tensor(fkx, device=DEV, dtype=torch.float32)])
    kyb2 = torch.cat([kyb, torch.tensor(fky, device=DEV, dtype=torch.float32)])
    om2 = torch.cat([om, torch.tensor(fom, device=DEV, dtype=torch.float32)])
    pw2 = beamform_prior(Pt, zt, kxb2, kyb2, om2)   # consistent, coarse+fine
    return kxb2, kyb2, om2, pw2, dict(n_fine=int(keep.sum()), k0abs=k0)


def a7v2_predict(P, z, depth, Q, sigma0, kp, theta, stat='ev1', force=False):
    """v2 arm: refine-select -> detect -> (fire) fine-bank solve / (else) hand."""
    sel = refine_select(P, z, depth, kp)
    score = sel['score_ev1'] if stat == 'ev1' else sel['score_ratio']
    fired = force or (score >= theta)
    Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
    if not fired:
        pred = mmse_gpu(Pt, zt, depth, Qt, sigma0, kp_hint=kp).cpu().numpy()
        return pred, sel, False, dict(n_fine=0)
    k0x, k0y = sel['winner']
    kxb2, kyb2, om2, pw2, info = augment_finebank_v2(Pt, zt, depth, kp, k0x, k0y)
    pred = solve_mmse_bank(Pt, zt, depth, Qt, sigma0, kxb2, kyb2, om2,
                           pw2).cpu().numpy()
    info['score'] = score
    return pred, sel, True, info


# =================================================================== detect cal
def _score_of(path, stat_return='both'):
    rng = np.random.default_rng(11)
    ep = load_synth(path, rng)
    sel = refine_select(ep['P'], ep['z'], ep['depth'], ep['kp'])
    return sel['score_ev1'], sel['score_ratio']


def _youden(pos, neg):
    """Threshold maximising TPR-FPR; ties -> largest theta (harmless-favouring)."""
    cands = np.unique(np.concatenate([pos, neg]))
    mids = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
    best = None
    for th in mids:
        tpr = float((pos >= th).mean()); fpr = float((neg >= th).mean())
        j = tpr - fpr
        if best is None or j > best[0] + 1e-12 or (abs(j - best[0]) < 1e-12
                                                   and th > best[1]):
            best = (j, float(th), tpr, fpr)
    return best


def calibrate_detect():
    """Calibrate on BF3/BF4 1comp (fire) vs 5comp (no-fire), NON-BF2. Compute
    both candidate statistics, pick the one with higher Youden-J."""
    pe, pr, ne, nr = [], [], [], []
    for tmpl in CAL_POS:
        for s in range(5):
            e, r = _score_of((tmpl % s).format(d=DATA_DIR)); pe.append(e); pr.append(r)
    for tmpl in CAL_NEG:
        for s in range(5):
            e, r = _score_of((tmpl % s).format(d=DATA_DIR)); ne.append(e); nr.append(r)
    pe, pr, ne, nr = map(np.array, (pe, pr, ne, nr))
    j_ev, th_ev, tpr_ev, fpr_ev = _youden(pe, ne)
    j_ra, th_ra, tpr_ra, fpr_ra = _youden(pr, nr)
    stat = 'ev1' if j_ev >= j_ra else 'ratio'
    theta = th_ev if stat == 'ev1' else th_ra
    return dict(
        stat=stat, theta=float(theta),
        ev1=dict(theta=th_ev, youden=j_ev, tpr=tpr_ev, fpr=fpr_ev,
                 pos=pe.tolist(), neg=ne.tolist()),
        ratio=dict(theta=th_ra, youden=j_ra, tpr=tpr_ra, fpr=fpr_ra,
                   pos=pr.tolist(), neg=nr.tolist()))


# ====================================================================== gate 1
def gate1(theta, stat):
    rows = []
    for seed in range(5):
        path = DATA_DIR + '/episode_BF2_1comp_seed%d_fixedT.h5' % seed
        rng = np.random.default_rng(11)
        ep = load_synth(path, rng)
        tk = ep['kp'] * ep['khat']                          # true k vector
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        pn, sel, fired, info = a7v2_predict(ep['P'], ep['z'], ep['depth'], Q,
                                            sig0, ep['kp'], theta, stat)
        pf, _, _, finfo = a7v2_predict(ep['P'], ep['z'], ep['depth'], Q, sig0,
                                       ep['kp'], theta, stat, force=True)
        ch, rh, _ = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, rn, _ = field_track_corr(pn, ep, npix, truth_tr, nf)
        cf, rf, _ = field_track_corr(pf, ep, npix, truth_tr, nf)
        wx, wy = sel['winner']
        dk_rel = float(math.hypot(wx - tk[0], wy - tk[1]) / ep['kp'])
        argmax_rel = float(abs(sel['argmax_kabs'] - ep['kp']) / ep['kp'])
        rows.append(dict(
            seed=seed, kp=ep['kp'], true_kabs=float(ep['kp']),
            score_ev1=sel['score_ev1'], score_ratio=sel['score_ratio'],
            resid_ratio=sel['resid_ratio'], fired=fired,
            argmax_kabs=sel['argmax_kabs'], argmax_rel_err=argmax_rel,
            winner_kabs=sel['winner_kabs'], winner=[wx, wy],
            dk_rel=dk_rel, sidelobe_rejected=bool(dk_rel < 0.10),
            hand_field0=ch[0], hand_rho=rh,
            hybrid_field0=cn[0], hybrid_rho=rn,
            forced_field0=cf[0], forced_rho=rf, n_fine=finfo.get('n_fine'),
            hand_field=ch, hybrid_field=cn, forced_field=cf))
        print(f"[G1 BF2 s{seed}] ev1 {sel['score_ev1']:+.3f} ratio "
              f"{sel['score_ratio']:+.3f} fire {fired}  "
              f"argmax|k| {sel['argmax_kabs']:.2f} -> win|k| "
              f"{sel['winner_kabs']:.2f} (true {ep['kp']:.2f}, dk/k "
              f"{dk_rel:.3f}{' REJECT' if dk_rel < 0.10 else ' MISS'})  "
              f"hand {ch[0]:+.2f}/{rh:+.2f}  hyb {cn[0]:+.2f}/{rn:+.2f}  "
              f"forced {cf[0]:+.2f}/{rf:+.2f}", flush=True)
    hy0 = np.array([r['hybrid_field0'] for r in rows])
    hyr = np.array([r['hybrid_rho'] for r in rows])
    n_pass = int(((hy0 >= 0.5) & (hyr >= 0.5)).sum())
    # forced (isolate refine-select quality from the detector)
    fo0 = np.array([r['forced_field0'] for r in rows])
    forr = np.array([r['forced_rho'] for r in rows])
    n_forced = int(((fo0 >= 0.5) & (forr >= 0.5)).sum())
    n_reject = int(sum(r['sidelobe_rejected'] for r in rows))
    return rows, dict(n_pass=n_pass, n_forced_pass=n_forced,
                      n_sidelobe_rejected=n_reject, n_total=5,
                      passed=n_pass >= 3,
                      criterion='field0>=0.5 AND rho>=0.5 in >=3/5')


# ====================================================================== gate 2
def gate2_synth(theta, stat):
    rows = []
    for fname in APP_EPISODES:
        rng = np.random.default_rng(11)
        ep = load_synth(DATA_DIR + '/' + fname, rng)
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        pn, sel, fired, info = a7v2_predict(ep['P'], ep['z'], ep['depth'], Q,
                                            sig0, ep['kp'], theta, stat)
        ch, _, _ = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, _, _ = field_track_corr(pn, ep, npix, truth_tr, nf)
        dfield = [cn[k] - ch[k] for k in range(len(LEADS))]
        rows.append(dict(episode=fname, sea=ep['sea'], ncomp=ep['ncomp'],
                         kp=ep['kp'], score_ev1=sel['score_ev1'],
                         score_ratio=sel['score_ratio'], fired=fired,
                         dfield=dfield, dfield_min=float(min(dfield))))
        print(f"[G2 synth {fname}] ev1 {sel['score_ev1']:+.3f} fire {fired}  "
              f"dfield {[f'{d:+.3f}' for d in dfield]}", flush=True)
    return rows


def gate2_real(theta, stat, smoke=False):
    if smoke:
        return [], dict(skipped=True)
    from d3_sa_multirecord import MRecord
    from d3_gate_promotion_eval import window_data
    from d3_falcon_real_operator import RECORDS
    name = 'YS17'
    cfg = RECORDS[name]
    print(f"[G2 real] loading {name} ...", flush=True)
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
    rows = []
    for wi, i0 in enumerate(i0s):
        P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(rec, cfg, int(i0),
                                                       nfw, n_lead, wi)
        v = np.isfinite(gt_q) & (leads <= 10.0)
        if v.sum() < 5:
            continue
        sig0 = 0.05 * Hs
        ph = hand_predict(P, z, cfg['depth'], Pq, sig0, kp)
        pn, sel, fired, info = a7v2_predict(P, z, cfg['depth'], Pq, sig0, kp,
                                            theta, stat)
        ch = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
        cn = float(np.corrcoef(pn[v], gt_q[v])[0, 1])
        rows.append(dict(record=name, wi=wi, kp=kp, score_ev1=sel['score_ev1'],
                         fired=fired, hand_enc=ch, hybrid_enc=cn, denc=cn - ch))
        print(f"[G2 real {name} w{wi}] ev1 {sel['score_ev1']:+.3f} fire {fired}"
              f"  hand {ch:+.2f} hyb {cn:+.2f}  d {cn-ch:+.3f}", flush=True)
    d = np.array([r['denc'] for r in rows])
    return rows, dict(n=len(rows), min_denc=float(d.min()) if len(d) else None,
                      mean_denc=float(d.mean()) if len(d) else None)


# ========================================================================= verify
def verify_solve(ep):
    """solve_mmse_bank on the coarse bank must reproduce mmse_gpu (sanity)."""
    Q = np.column_stack([np.linspace(20, 240, 200), np.linspace(-50, 50, 200),
                         np.full(200, ep['t_end'])]).astype(np.float32)
    Pt = torch.tensor(ep['P'], device=DEV, dtype=torch.float32)
    zt = torch.tensor(ep['z'], device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
    sig0 = 0.05 * ep['Hs']
    a = mmse_gpu(Pt, zt, ep['depth'], Qt, sig0, kp_hint=ep['kp']).cpu().numpy()
    kxb, kyb, om = build_bank(ep['kp'], (0.35, 2.5), ep['kp'] / 10.0,
                              ep['depth'], DEV, torch.float32)
    pw = beamform_prior(Pt, zt, kxb, kyb, om)
    b = solve_mmse_bank(Pt, zt, ep['depth'], Qt, sig0, kxb, kyb, om,
                        pw).cpu().numpy()
    return float(np.max(np.abs(a - b)))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--data_dir', default='data')
    a = ap.parse_args()
    global DATA_DIR
    DATA_DIR = a.data_dir
    t0 = time.time()
    print(f"device {DEV}", flush=True)

    rng = np.random.default_rng(11)
    ep0 = load_synth(DATA_DIR + '/episode_BF2_1comp_seed0_fixedT.h5', rng)
    vdiff = verify_solve(ep0)
    print(f"verify solve_mmse_bank == mmse_gpu (coarse): max|diff| {vdiff:.2e} "
          f"({'OK' if vdiff < 1e-3 else 'MISMATCH'})", flush=True)

    print("\n--- detect v2 calibration (1comp vs 5comp, non-BF2) ---", flush=True)
    cal = calibrate_detect()
    stat, theta = cal['stat'], cal['theta']
    for s in ('ev1', 'ratio'):
        c = cal[s]
        print(f"  [{s}] theta {c['theta']:+.3f}  Youden {c['youden']:.2f}  "
              f"TPR {c['tpr']:.2f} FPR {c['fpr']:.2f}", flush=True)
        print(f"       pos(1comp) {['%+.2f' % v for v in c['pos']]}")
        print(f"       neg(5comp) {['%+.2f' % v for v in c['neg']]}")
    print(f"  CHOSEN statistic = {stat}, theta = {theta:+.4f}", flush=True)

    print("\n--- G-A7v2-1 recovery (BF2 1comp x5) ---", flush=True)
    g1_rows, g1 = gate1(theta, stat)
    fire_rate = float(np.mean([r['fired'] for r in g1_rows]))
    print(f"G-A7v2-1 {'PASS' if g1['passed'] else 'FAIL'}  "
          f"({g1['n_pass']}/5 recover; forced {g1['n_forced_pass']}/5; "
          f"sidelobe-rejected {g1['n_sidelobe_rejected']}/5; "
          f"fire-rate {fire_rate:.0%})", flush=True)

    print("\n--- G-A7v2-2 harmless: synth 6 ---", flush=True)
    g2s = gate2_synth(theta, stat)
    dfield_all = [d for r in g2s for d in r['dfield']]
    g2s_min = float(min(dfield_all))
    print(f"synth min dfield {g2s_min:+.4f}  "
          f"({'PASS' if g2s_min >= -0.01 else 'FAIL'} @ -0.01)", flush=True)

    print("\n--- G-A7v2-2 harmless: real YS17 12 windows ---", flush=True)
    g2r, g2r_sum = gate2_real(theta, stat, smoke=a.smoke)
    if not a.smoke:
        real_min = g2r_sum['min_denc']
        g2_passed = (g2s_min >= -0.01) and (real_min is not None
                                            and real_min >= -0.02)
        print(f"real min denc {real_min:+.4f}  "
              f"({'PASS' if real_min >= -0.02 else 'FAIL'} @ -0.02)", flush=True)
    else:
        g2_passed = g2s_min >= -0.01
        real_min = None

    # G-A7v2-3 diagnostic table
    g3 = [dict(seed=r['seed'], true_kabs=r['true_kabs'],
               argmax_kabs=r['argmax_kabs'], argmax_rel_err=r['argmax_rel_err'],
               winner_kabs=r['winner_kabs'], dk_rel=r['dk_rel'],
               sidelobe_rejected=r['sidelobe_rejected'],
               forced_field0=r['forced_field0'], forced_rho=r['forced_rho'])
          for r in g1_rows]

    out = dict(
        date=DATE, prereg='docs/A7_D20_prereg_2026-07-12.md (A-7 v2)',
        method='B-1 refine-select hybrid',
        design=dict(M_peaks=M_PEAKS, val_frac=VAL_FRAC, nms_supp=NMS_SUPP,
                    finebank=f'dk=kp/{FINE_DIV}, |k-k0|/k0<={KREL}, +-{ANGDEG}deg',
                    refine='beamform peak -> Nelder-Mead single-comp LS '
                           '(dispersion-constrained) -> select by 20% holdout '
                           'residual', U_ship=U_SHIP,
                    note='mmse_gpu unmodified; fired arm = solve_mmse_bank on '
                         'coarse+fine bank (verified equal on coarse)'),
        verify_solve_maxdiff=vdiff,
        detect_calibration=cal,
        gate1=dict(summary=g1, bf2_fire_rate=fire_rate, episodes=g1_rows),
        gate2=dict(synth=g2s, synth_min_dfield=g2s_min, real=g2r,
                   real_summary=g2r_sum if not a.smoke else None,
                   passed=g2_passed),
        gate3_mechanism=g3,
        gates=dict(
            G_A7v2_1_recovery=dict(passed=g1['passed'], n_pass=g1['n_pass'],
                                   n_forced_pass=g1['n_forced_pass'],
                                   n_sidelobe_rejected=g1['n_sidelobe_rejected'],
                                   bf2_fire_rate=fire_rate),
            G_A7v2_2_harmless=dict(passed=g2_passed, synth_min_dfield=g2s_min,
                                   real_min_denc=real_min),
            G_A7v2_3_mechanism=dict(
                n_sidelobe_rejected=g1['n_sidelobe_rejected'], table=g3)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n================= GATES =================", flush=True)
    print(f"G-A7v2-1 recovery : {'PASS' if g1['passed'] else 'FAIL'}  "
          f"({g1['n_pass']}/5 recover, forced {g1['n_forced_pass']}/5, "
          f"sidelobe-rej {g1['n_sidelobe_rejected']}/5, fire {fire_rate:.0%})")
    print(f"G-A7v2-2 harmless : {'PASS' if g2_passed else 'FAIL'}  "
          f"(synth min dfield {g2s_min:+.4f}"
          f"{'' if a.smoke else f', real min denc {real_min:+.4f}'})")
    print(f"G-A7v2-3 mechanism: sidelobe rejection "
          f"{g1['n_sidelobe_rejected']}/5 seeds (diagnostic)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
