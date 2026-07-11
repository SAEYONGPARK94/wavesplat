"""H1 -- WHERE does the mask's 17% go? Slope / near-field / encounter-forecast.

G0b (docs/G0B_THE_GAP_IS_THE_RESULT_2026-07-11.md) established, paired on
identical observations, that the mask's 17.13% (F2b, gate-PASS) buys ZERO
far-field elevation accuracy: gain -0.0000 +- 0.0020 (n=16). Its autopsy named
two mechanisms: (1) the fan task is dead at this operating point (naive 1.31 >
zero-predictor; threshold top-k concentrates returns inside r90 ~ 20 m), and
(2) FUNCTIONAL MISMATCH -- mask bits are along-ray SLOPE at the ray positions,
the fan task wants far-field PHASE. The shadow literature (Gangeskar) recovers
Hs and slope statistics from masks, never fields.

This script measures the one remaining accuracy question of the map
(MAP_what_is_learnable_2026-07-10.md): the mask's DESTINATION. Same paired
G0b machinery -- identical scene generator, identical EP, identical prior --
only the scored functionals change:

  ETA_INF   elevation on the informed annulus r in [R_MIN, r90], 400 dense
            queries (G0b's informed scoring had a median of FIVE fan queries
            -- degenerate; #48). Live operating point: naive ~ 0.58.
  SLP_INF   along-ray slope on the same annulus (grad_design functionals).
            This is the functional the mask bits actually speak.
  ENC       encounter forecast: eta at the SENSOR (0,0) at leads 0-10 s,
            scored in two bands (0,3] and (3,10] s. Causal in HEAD SEAS:
            the main arm generates seas with direction 180 deg (waves travel
            toward the sensor; the forward fan is upwave), the 07-05
            "forward-scan encounter forecast" configuration and G0b's own
            recommendation ("the time axis -- the waves come to the sensor").
  SIG_INF   report-only: posterior-based sigma_eta estimate on the annulus
            (uses the accounting solves' pointwise variances; Gangeskar axis).

ACCOUNTING FIRST (design law #1). For every functional family c the script
computes, inline, the Gaussian posterior-variance contraction
    pred_gain_frac = 1 - sqrt( sum_q c_q' A_aware^-1 c_q / sum_q c_q' A0^-1 c_q )
with A_aware = A0 + G' diag(w tau) G at the EP-converged tau -- BEFORE looking
at errors. This is the per-functional ceiling map; the paired errors then test
it. (F2b's MI was task-agnostic; this is the task-resolved version.)

The estimator KNOWS (as in G0b): fired-ray geometry, per-scan T, radiometric
law A, kp. It never sees true slopes or the GT field.

================================ PRE-REGISTRATION ============================
Frozen before the first full run. Main arm n >= 16 seas, paired by sea,
clustered SE. Smoke runs (n <= 2) are for machinery checks only.

  R1  REDUCTION.  sweeps=0 => naive, same code path (inherited from G0b).
  HA  ACCOUNTING MAP printed per functional BEFORE errors. A functional with
      pred_gain_frac < 0.005 is predicted-dead; measured |gain| > 2x its
      accounting prediction + 0.005 flags a leak and BLOCKS a positive verdict
      on that functional until resolved (two-sided, #45 and mirror).
  HG1 DESTINATION (per functional, gated on main arm): paired gain > 0 with
      t >= 3 AND mean gain >= 0.005 (normalized units). Any functional
      passing HG1 (+HG3, HG4, HA-consistency) is "the destination".
  HG3 PERMUTATION (n >= 8): miss footprints shuffled within range bins;
      |gain| <= 0.005 on every functional. Permuted masks helping =
      regularisation, not selection information. NO VERDICT then.
  HG4 MCAR IDENTITY (n >= 4): uniform-random top-k + truthful constant-p
      aware arm: gain exactly 0 by construction; printed as identity check.
  HG5 CAUSALITY CONTRAST (n >= 8, dir=0 arm): following seas -- the sensor's
      future comes from the unobserved x<0 half-plane. ENC skill (1 - nrmse)
      must collapse toward 0 relative to the head-sea main arm. This gates
      the INTERPRETATION of ENC (propagation, not smoothing), not HG1.
      NOTE the dir0 arm changes the SCENE (wave direction), so it is a
      contrast, not a paired control.
  HG6 HOS REPRODUCTION (standing instruction): any HG1-positive functional
      must reproduce on HOS seas before it becomes a headline. Separate run.

DECISION, fixed now:
  - Some functional passes HG1 + HG3 + HG4 + HA => the mask's 17% has a
    measured destination; the N5 site head gets a TEACHER WITH NONZERO GAIN
    on that functional, and amortizing it is the learning chapter's accuracy
    piece (step 3 resumes with a live target).
  - All functionals null => the mask channel is closed at EVERY functional
    we can name, the map's last accuracy cell fills with a number, and the
    learning chapter stands on architecture + speed alone.
Both endings are results. The gap, if it persists, is itself the result (G0b).

CAPS -- DISCLOSED PROTOCOL CHANGE after the n=2 smoke (machinery check, as
pre-registered; no gate consumed). The first draft capped only ETA_INF and
reused its beta for ENC; the smoke then reproduced #49 verbatim on the other
functionals: slp naive 1.0-2.9, enc naive 3.8-8.5 -- all above the zero
predictor, i.e. B1's certificate was silently absent exactly where the
estimator extrapolates. The B1 proof requires beta computed at the SCORED
queries. So: every functional now carries its own cap,
    eta_inf: beta = min(1, std(z) / std(f_eta))          (observation anchor)
    enc:     beta = min(1, std(z) / std(f_enc))          (observation anchor)
    slp_inf: beta = min(1, sd_prior_slope / std(f_slp)),
             sd_prior_slope = sqrt(mean_q sum_b gam_b G_qb^2) -- the PRIOR
             slope sd; model-anchored, GT-free. Not a B1 certificate (no
             observed slope exists); an instrument, identical across arms.
Livness is reported per functional: a capped naive nRMSE ~ 1 means the
functional is dead at this operating point and its gain row is moot (G0b:
"117% of zero is zero").

SECOND DISCLOSED CHANGE (same smoke): slope was scored against the TOTAL true
slope, but 56% of slope variance lives OUTSIDE the bank (measured: eta
in-band 0.90, slope in-band 0.44 at band 0.35-2.5 kp; slope weights k^2 S(k))
-- an nRMSE floor of sqrt(0.56) = 0.75 no estimator with this bank can cross.
That is itself a structural finding: over half of what the mask's slope bits
describe has NO receiver in the model space. SLP_INF is therefore scored
against the IN-BAND true slope (slope of the band-limited true field, same
band as the bank, identical for both arms); the total-GT row is kept as a
report column. Correlations rho(pred, gt) are reported per functional as a
secondary, cap-free currency (no gate; HG1 stays on capped nRMSE as frozen).
=============================================================================
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import d3_g0b_selection_aware as g0b
from d3_g0b_selection_aware import (A_of_t, DEV, DT, N_PTS, K_MISS, SWEEPS,
                                    DAMP, GH_X, GH_W, BAND)
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import (eta_dr, H_SENS, R_MIN, R_MAX, SCAN_T,
                                  NOISE, DEPTH, FOV_H, HS, TP, NCOMP, SPREAD)
from wavesim.wave_generator import jonswap_components

OUT = 'results/direction3/h1_mask_destination'
N_QINF = 400
ENC_T = np.arange(0.25, 10.001, 0.25)          # 40 leads
ENC_SPLIT = 3.0                                 # (0,3] / (3,10] s


def sea_dir(direction_deg):
    """head-sea (180) or following-sea (0) scene family, else as door1b."""
    def _sea(seed, amp_scale=1.0):
        cs = jonswap_components(HS, TP, n_components=NCOMP, depth=DEPTH,
                                direction_spread_deg=SPREAD, seed=seed,
                                direction_deg=direction_deg)
        return (np.array([c.kx for c in cs]), np.array([c.ky for c in cs]),
                np.array([c.omega for c in cs]),
                np.array([c.A for c in cs]) * amp_scale,
                np.array([c.phi for c in cs]))
    return _sea


def eta_xt(x, y, t, S):
    """eta at per-query times (door1b's eta takes a scalar t)."""
    kx, ky, om, a, ph = S
    psi = (np.outer(x, kx) + np.outer(y, ky) - np.outer(t, om) + ph)
    return (a * np.cos(psi)).sum(1)


def ep_aware_x(Phi, z, s2, gam, kxb, kyb, om, sitesP, sites_r, sites_T,
               is_return, w_scale, sweeps=SWEEPS, mcar=False):
    """g0b.ep_aware, verbatim numerics, but returns (a, act, tau, G, w) so the
    accounting can rebuild A_aware. Kept in sync by the R1/identity checks."""
    A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    b0 = Phi.T @ z / s2
    if sweeps == 0 or len(sitesP) == 0 or mcar:
        return torch.linalg.solve(A0, b0), 0.0, None, None, None
    G = grad_design(sitesP, kxb, kyb, om)
    K = len(G)
    w = torch.tensor(w_scale, device=DEV, dtype=DT)
    tau = torch.zeros(K, device=DEV, dtype=DT)
    nu = torch.zeros(K, device=DEV, dtype=DT)
    gx = torch.tensor(GH_X, device=DEV, dtype=DT)
    gw = torch.tensor(GH_W / GH_W.sum(), device=DEV, dtype=DT)
    rT, rr = sites_T, sites_r

    for _ in range(sweeps):
        A = A0 + (G * (w * tau)[:, None]).T @ G
        S_ = torch.linalg.inv(A)
        mu = S_ @ (b0 + G.T @ (w * nu))
        v = ((G @ S_) * G).sum(1).clamp_min(1e-14)
        m = G @ mu
        vc_inv = (1.0 / v) - tau
        ok = vc_inv > 1e-12
        vc = torch.where(ok, 1.0 / vc_inv.clamp_min(1e-12), v * 1e6)
        mc = torch.where(ok, vc * (m / v - nu), m)

        sd = torch.sqrt(vc)
        sq = mc[:, None] + sd[:, None] * gx[None, :]
        p = torch.exp(-rT[:, None] / A_of_t(sq, rr[:, None]).clamp_min(1e-30))
        f = torch.where(is_return[:, None], p, 1.0 - p).clamp_min(1e-300)
        Z = (gw[None] * f).sum(1)
        m_h = (gw[None] * f * sq).sum(1) / Z
        v_h = ((gw[None] * f * sq ** 2).sum(1) / Z - m_h ** 2).clamp_min(1e-14)

        tau_n = ((1.0 / v_h) - (1.0 / vc)).clamp_min(0.0)
        nu_n = m_h / v_h - mc / vc
        bad = ~ok | ~torch.isfinite(tau_n) | ~torch.isfinite(nu_n)
        tau_n = torch.where(bad, tau, tau_n)
        nu_n = torch.where(bad, nu, nu_n)
        tau = DAMP * tau_n + (1 - DAMP) * tau
        nu = DAMP * nu_n + (1 - DAMP) * nu

    A = A0 + (G * (w * tau)[:, None]).T @ G
    a = torch.linalg.solve(A, b0 + G.T @ (w * nu))
    return a, float((tau > 1e-8).float().mean()), tau, G, w


def var_of(A, C):
    """pointwise posterior variances c_q' A^-1 c_q for functional rows C."""
    X = torch.linalg.solve(A, C.T)
    return (C.T * X).sum(0)


def run_sea(seed, amp_scale, rng, mcar=False, perm=False):
    S, kp, amps, missP, miss_r, Ts = g0b.gen_scene(seed, amp_scale, rng, mcar)
    if perm:                                             # HG3: break field link
        edges = np.linspace(R_MIN, R_MAX, 13)
        idx = np.clip(np.digitize(miss_r, edges) - 1, 0, 11)
        for b in range(12):
            m = idx == b
            if m.sum() > 1:
                sh = rng.permutation(np.where(m)[0])
                missP[m, :2] = missP[sh, :2]

    P, zval = amps[:, :3], amps[:, 3] + rng.normal(0, NOISE, len(amps))
    ret_r = amps[:, 4]
    r90 = float(np.quantile(ret_r, 0.90))

    # -- queries -------------------------------------------------------------
    rq = np.sqrt(rng.uniform((R_MIN / r90) ** 2, 1.0, N_QINF)) * r90
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), N_QINF)
    Qi = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(N_QINF)])
    gt_i = eta_xt(Qi[:, 0], Qi[:, 1], Qi[:, 2], S)
    gs_i = eta_dr(Qi[:, 0], Qi[:, 1], 0.0, S)
    kt = np.hypot(S[0], S[1])
    ib = (kt >= BAND[0] * kp) & (kt <= BAND[1] * kp)
    S_ib = tuple(c[ib] for c in S)
    gs_ib = eta_dr(Qi[:, 0], Qi[:, 1], 0.0, S_ib)
    ibf_eta = float((S[3][ib] ** 2).sum() / (S[3] ** 2).sum())
    ibf_slp = float(((kt * S[3])[ib] ** 2).sum() / ((kt * S[3]) ** 2).sum())
    Qe = np.column_stack([np.zeros(len(ENC_T)), np.zeros(len(ENC_T)), ENC_T])
    gt_e = eta_xt(Qe[:, 0], Qe[:, 1], Qe[:, 2], S)

    # sigma for normalisation: large GT sample on the annulus / global field
    sig_i = float(np.std(gt_i))
    sig_s = float(np.std(gs_ib))
    sig_st = float(np.std(gs_i))
    xs = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, 2000)) * R_MAX
    asz = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 2000)
    sig_g = float(np.std(eta_xt(xs * np.cos(asz), xs * np.sin(asz),
                                np.zeros(2000), S)))

    # -- model + designs (identical to G0b) -----------------------------------
    tP = torch.tensor(P, device=DEV, dtype=torch.float32)
    tz = torch.tensor(zval, device=DEV, dtype=torch.float32)
    tQ = torch.tensor(Qi, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(tP, tz, DEPTH, tQ, NOISE, kp, seed=seed,
                      return_model=True)
    gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])

    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
    Phi = design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om)
    Ci = design(torch.tensor(Qi, device=DEV, dtype=DT), kxb, kyb, om)
    Cs = grad_design(torch.tensor(Qi, device=DEV, dtype=DT), kxb, kyb, om)
    Ce = design(torch.tensor(Qe, device=DEV, dtype=DT), kxb, kyb, om)
    tz64 = torch.tensor(zval, device=DEV, dtype=DT)

    n_miss = len(missP)
    ksub = min(K_MISS, n_miss)
    sub = rng.choice(n_miss, ksub, replace=False)
    Tmap = {t: Ts[i] for i, t in enumerate(SCAN_T)}
    sT_ret = np.array([Tmap[t] for t in P[:, 2]])
    sT_mis = np.array([Tmap[t] for t in missP[sub, 2]])
    sitesP = torch.tensor(np.vstack([P, missP[sub]]), device=DEV, dtype=DT)
    sites_r = torch.tensor(np.concatenate([ret_r, miss_r[sub]]),
                           device=DEV, dtype=DT)
    sites_T = torch.tensor(np.concatenate([sT_ret, sT_mis]),
                           device=DEV, dtype=DT)
    is_ret = torch.tensor(np.r_[np.ones(len(P), bool), np.zeros(ksub, bool)],
                          device=DEV)
    w_scale = np.r_[np.ones(len(P)), np.full(ksub, n_miss / ksub)]

    a_naive, _, _, _, _ = ep_aware_x(Phi, tz64, s2, gam, kxb, kyb, om, sitesP,
                                     sites_r, sites_T, is_ret, w_scale,
                                     sweeps=0)
    a_aware, act, tau, G, w = ep_aware_x(Phi, tz64, s2, gam, kxb, kyb, om,
                                         sitesP, sites_r, sites_T, is_ret,
                                         w_scale, mcar=mcar)

    # -- HA accounting: variance contraction per functional, BEFORE errors ----
    A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    acct, v0i = {}, None
    if tau is not None:
        A1 = A0 + (G * (w * tau)[:, None]).T @ G
        for name, C in (('eta_inf', Ci), ('slp_inf', Cs), ('enc', Ce)):
            v0 = var_of(A0, C)
            v1 = var_of(A1, C)
            acct[name] = float(1.0 - torch.sqrt(v1.sum() / v0.sum()))
            if name == 'eta_inf':
                v0i, v1i = v0, v1
    else:
        v0i = var_of(A0, Ci)
        v1i = v0i

    # -- errors ----------------------------------------------------------------
    sd_prior_slp = float(torch.sqrt(((Cs ** 2) * gam[None]).sum(1).mean()))

    def rho(p, g):
        p, g = np.asarray(p, float), np.asarray(g, float)
        if p.std() < 1e-12 or g.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(p, g)[0, 1])

    def scores(a, v_eta):
        zsd = float(tz64.std())
        fi = Ci @ a
        beta = min(1.0, zsd / float(fi.std().clamp_min(1e-12)))
        gt_it = torch.tensor(gt_i, device=DEV, dtype=DT)
        e_i = float(torch.sqrt(((beta * fi - gt_it) ** 2).mean())) / sig_i
        fs = Cs @ a
        beta_s = min(1.0, sd_prior_slp / float(fs.std().clamp_min(1e-12)))
        fs_n = (beta_s * fs).cpu().numpy()
        e_s = float(np.sqrt(np.mean((fs_n - gs_ib) ** 2))) / sig_s
        e_st = float(np.sqrt(np.mean((fs_n - gs_i) ** 2))) / sig_st
        fe_r = Ce @ a
        beta_e = min(1.0, zsd / float(fe_r.std().clamp_min(1e-12)))
        fe = (beta_e * fe_r).cpu().numpy()
        m03 = ENC_T <= ENC_SPLIT
        e_e0 = float(np.sqrt(np.mean((fe[m03] - gt_e[m03]) ** 2))) / sig_g
        e_e1 = float(np.sqrt(np.mean((fe[~m03] - gt_e[~m03]) ** 2))) / sig_g
        # report-only sigma estimate: E[eta^2] = m^2 + var, B1-capped mean
        sg = float(torch.sqrt(((beta * fi) ** 2).mean()
                              + v_eta.mean())) / sig_i
        rr = {'eta_inf': rho(fi.cpu().numpy(), gt_i),
              'slp_inf': rho(fs_n, gs_ib), 'enc': rho(fe, gt_e)}
        return e_i, e_s, e_st, e_e0, e_e1, sg, beta, fe, rr

    ei0, es0, est0, ee00, ee10, sg0, b0_, fe0, rr0 = scores(a_naive, v0i)
    ei1, es1, est1, ee01, ee11, sg1, b1_, fe1, rr1 = scores(a_aware, v1i)
    return {'seed': seed, 'kp': kp, 'r90': r90, 'act': act,
            'n_amp': len(P), 'n_miss': int(n_miss),
            'sig_i': sig_i, 'sig_s': sig_s, 'sig_g': sig_g,
            'ibf_eta': ibf_eta, 'ibf_slp': ibf_slp,
            'acct': acct, 'beta': [b0_, b1_],
            'eta_inf': [ei0, ei1], 'slp_inf': [es0, es1],
            'slp_tot': [est0, est1],
            'enc03': [ee00, ee01], 'enc310': [ee10, ee11],
            'sig_ratio': [sg0, sg1], 'rho': [rr0, rr1],
            'enc_pred': [fe0.tolist(), fe1.tolist()], 'enc_gt': gt_e.tolist()}


def clustered(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.inf
    return float(v.mean()), float(se), float(v.mean() / se if se > 0 else 0)


FUNC = ['eta_inf', 'slp_inf', 'enc03', 'enc310']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=16)
    ap.add_argument('--amp-scale', type=float, default=3.33)
    ap.add_argument('--arm', choices=['main', 'perm', 'mcar', 'dir0'],
                    default='main')
    ap.add_argument('--tag', default=None)
    a_ = ap.parse_args()
    tag = a_.tag or a_.arm
    direction = 0.0 if a_.arm == 'dir0' else 180.0
    g0b.sea = sea_dir(direction)                 # scene family for this arm

    print(f"=== H1 mask destination  arm={a_.arm} dir={direction:.0f} "
          f"seas={a_.seas} amp={a_.amp_scale} ===")
    print("PRE-REGISTERED: R1 | HA accounting-first | HG1 gain>=0.005 t>=3 |")
    print("  HG3 |perm|<=0.005 | HG4 mcar identity | HG5 dir0 enc collapse |")
    print("  HG6 HOS reproduction before any headline\n")

    rng = np.random.default_rng(21)
    rows = []
    for s in range(a_.seas):
        t0 = time.time()
        r = run_sea(s, a_.amp_scale, rng, mcar=(a_.arm == 'mcar'),
                    perm=(a_.arm == 'perm'))
        rows.append(r)
        ac = r['acct']
        ac_s = ' '.join(f"{k} {v:.4f}" for k, v in ac.items()) or 'naive-only'
        print(f"sea {s:2d} r90 {r['r90']:5.1f} act {r['act']:.2f} "
              f"[{time.time()-t0:5.1f}s]\n"
              f"   HA acct: {ac_s}\n"
              f"   eta_inf {r['eta_inf'][0]:.4f}->{r['eta_inf'][1]:.4f} "
              f"slp_inf {r['slp_inf'][0]:.4f}->{r['slp_inf'][1]:.4f} "
              f"enc03 {r['enc03'][0]:.4f}->{r['enc03'][1]:.4f} "
              f"enc310 {r['enc310'][0]:.4f}->{r['enc310'][1]:.4f}", flush=True)

    print("\n" + "=" * 76)
    res = {'arm': a_.arm, 'direction': direction, 'amp_scale': a_.amp_scale,
           'n_sea': len(rows), 'rows': rows, 'summary': {}}
    for f in FUNC:
        naive = clustered([r[f][0] for r in rows])
        gain = clustered([r[f][0] - r[f][1] for r in rows])
        acc = float(np.mean([r['acct'].get(f.replace('enc03', 'enc')
                                           .replace('enc310', 'enc'), 0.0)
                             for r in rows]))
        res['summary'][f] = {'naive': naive, 'gain': gain, 'acct': acc}
        print(f"{f:8s} naive {naive[0]:.4f}+-{naive[1]:.4f}   "
              f"gain {gain[0]:+.4f}+-{gain[1]:.4f}  t {gain[2]:+.2f}   "
              f"HA-pred(frac) {acc:.4f}")
    st = clustered([r['slp_tot'][0] - r['slp_tot'][1] for r in rows])
    print(f"slp_tot  naive {np.mean([r['slp_tot'][0] for r in rows]):.4f} "
          f"(report; in-band frac slp "
          f"{np.mean([r['ibf_slp'] for r in rows]):.3f} eta "
          f"{np.mean([r['ibf_eta'] for r in rows]):.3f})   "
          f"gain {st[0]:+.4f}+-{st[1]:.4f}")
    for f in ('eta_inf', 'slp_inf', 'enc'):
        r0 = clustered([r['rho'][0][f] for r in rows])
        r1 = clustered([r['rho'][1][f] for r in rows])
        dr = clustered([r['rho'][1][f] - r['rho'][0][f] for r in rows])
        res['summary'][f'rho_{f}'] = {'naive': r0, 'aware': r1, 'd': dr}
        print(f"rho {f:8s} naive {r0[0]:+.3f}  aware {r1[0]:+.3f}  "
              f"d {dr[0]:+.4f}+-{dr[1]:.4f} t {dr[2]:+.2f}")
    sg = clustered([r['sig_ratio'][0] for r in rows])
    sa = clustered([r['sig_ratio'][1] for r in rows])
    print(f"sigma_hat/sigma (report-only): naive {sg[0]:.3f}+-{sg[1]:.3f}  "
          f"aware {sa[0]:.3f}+-{sa[1]:.3f}")
    res['summary']['sig_ratio'] = {'naive': sg, 'aware': sa}

    os.makedirs('results/direction3', exist_ok=True)
    with open(f'{OUT}_{tag}.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}_{tag}.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
