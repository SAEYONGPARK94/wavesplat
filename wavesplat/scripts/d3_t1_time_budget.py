"""T1: the accounting gate for the TIME channel. NO LEARNING happens here.

Today closed every accuracy channel for learning except one. The one measured
POSITIVE was a scalar exponential-forgetting weight: obs_weight = lam^age gained
+0.0242 +- 0.0058 nRMSE on 30 real cells (N3), passing its negative control (it
HURTS on stationary synthetic seas). Nobody has measured how much MORE the
time / non-stationarity channel holds BEYOND that scalar. A learned lam(k) net
can only ever recover the gap between the best SCALAR forgetting and the best
PER-BAND forgetting. This script computes that gap. Training is gated on it.

  Design law of the day: ACCOUNTING BEFORE TRAINING. The gate sits at the
  ceiling of what a lam(k) network could achieve, computed with GT oracles.

------------------------------------------------------------------ the seas
Non-stationary synthetic seas. Components are dispersion-consistent plane waves
with FIXED (k_i, omega_i, phi_i); non-stationarity is injected exactly as O1's
`nonstat` does -- a time-varying multiplicative envelope on the amplitudes --
but GENERALISED from O1's single band+direction-blind sinusoid to a
band-and-direction-selective Gaussian envelope whose PEAK DRIFTS:

    a_i(t) = a0_i * exp( -0.5 (ln(k_i/kp(t)) / s_k)^2 )
                  * exp( -0.5 (angle(theta_i - dir(t)) / s_d)^2 ) * c(t)

    kp(t)  = kp0 * (1 + rate_k * t),  rate_k = +-0.15/60 s^-1   (+-15% per 60 s)
    dir(t) = dir0 + rot * (t / 20),   rot    = +-(10..15) deg   (over the window)
    c(t) renormalises spatial variance to sigma^2 at every t (energy held fixed,
         so the ONLY non-stationarity is spectral SHAPE drift -- the honest hard
         test: a scalar lam cannot be helped by a bulk energy decay).

kp DRIFT is the mechanism that makes a scalar SUB-optimal: as the log-k envelope
slides, different k/kp bands become non-stationary at different rates, so the
optimal forgetting rate is band-dependent. Direction rotation adds
non-stationarity that a scalar already handles (it is not k/kp-band-selective).
The stationary control sets rate_k = rot = 0, so a_i(t) is constant.

Observations: trailing 20 s fan window (SF geometry), 600 points, noise 0.05,
query fan at t=0, nRMSE normalised by the GT std at t=0.

--------------------------------------------------------------- estimators
ALL through the same mmse_gpu, same (0.35, 2.5) kp bank, same 600-pt budget --
the pairing (same seas, same points) is the statistical power.

  E0  batch                uniform weights (lam = 1).
  E1  scalar-lam           obs_weight = lam^((t0-t_j)/1s), lam in
                           {0.80,0.85,0.90,0.95,1.00}. Picked two ways:
                             E1-val    : best on a held-out 25% of observations
                             E1-oracle : best on GT (upper bound for ANY scalar)
  E2  per-band-lam oracle  the bank split into 3 k/kp bands (0.35-0.7, 0.7-1.3,
                           1.3-2.5). obs_weight cannot differ per band inside one
                           solve. The LITERAL prescription -- three separate
                           prior_gain-masked mmse_gpu solves, summed -- was tried
                           and REJECTED by its own fidelity check: mmse_gpu
                           renormalises the prior to the full field variance over
                           whatever atoms survive the mask, so each of the three
                           solves tries to explain the WHOLE field with one band
                           and the sum is ~9x too large (|E2sum - E0| = 1.04).
                           Corrected superposition (documented, C3-verified):
                           take the batch prior gam and gram from ONE mmse_gpu
                           call (return_model=True), and give band b its own
                           NOISE weighting via
                               K_b = K + s2 (diag(1/w_b^2) - I),  w_b = lam_b^age,
                           field = sum_b (Phi_q*gam)[:,S_b] @ (Phi^T K_b^-1 z)[S_b].
                           At lam_b = 1 for all b, K_b = K and this collapses to
                           the batch field to fp precision -> C3 passes exactly.
                           E1 and E2 therefore share ONE prior and ONE machinery;
                           E1 is the diagonal (all lam equal) of E2's 5^3 grid, so
                           E2-oracle <= E1-oracle BY CONSTRUCTION and the gap is a
                           clean, non-negative ceiling on per-band forgetting.
                           lam per band picked on GT = ORACLE; also picked on val.
  E3  kp-tracking oracle   scalar lam (GT-picked) + kp_hint = TRUE kp(t=0) rather
                           than the window-average kp -- isolates how much of the
                           prize is just bank band-placement drift.

================================ PRE-REGISTRATION ==========================
Computed BEFORE running. Clustered by sea, n >= 20 non-stationary seas.

    gap := nRMSE(E1-oracle) - nRMSE(E2-oracle),  paired per sea.

  * gap < 0.005  OR  t(gap) < 2  =>  the H1 cell CLOSES: "a scalar saturates
      the time channel." Do NOT train d3_t2. Report the closed gate.
  * gap >= 0.005 AND t(gap) >= 2 =>  H1 has a target. Train d3_t2, gate the net
      BELOW this computed ceiling (recover >= 50% of the gap on held-out seas).

Controls (must all hold or the instrument is suspect):
  C1 stationary negative control: on stationary seas forgetting must not help --
     E1-oracle - E0 ~ 0 (the GT oracle can always fall back to lam=1), and the
     GT-oracle lam distribution concentrates at 1.0.
  C2 selectability without GT: E2-val (per-band lam picked on val, no GT) must
     NOT hurt more than 0.002 vs E0 on stationary seas.
  C3 superposition fidelity: |nRMSE(E2 sum, all lam=1) - nRMSE(E0)| < 0.005 on
     stationary seas (the per-band solve reduces to the batch solve).
=============================================================================
"""
import argparse
import json
import os
import sys
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_g3_adversarial_sea import geometry, DEV, DEPTH, KP, SIGMA

OUT = 'results/direction3/t1_time_budget.json'
G = 9.81
KP0 = float(KP)                       # 2*pi/80 m, reused so geometry is consistent
S0 = 0.05 * 4 * SIGMA                  # 0.20 m regularisation, project convention
NOISE = 0.05
BAND = (0.35, 2.5)
SUBBANDS = [0.35, 0.7, 1.3, 2.5]       # 3 k/kp bands partitioning the bank
LAMS = (0.80, 0.85, 0.90, 0.95, 1.00)
AGE_DT = 1.0                           # obs_weight = lam^((t0 - t_j)/AGE_DT)
HELD = 0.25
NCOMP = 100
S_K = 0.4                              # log-k envelope width (a peaky spectrum)
S_D = np.deg2rad(30.0)                 # directional envelope width
T0 = 0.0                               # the query / scoring instant


# --------------------------------------------------------- non-stationary sea
def make_sea(seed, drift=True, drift_scale=1.0):
    """Return eta(P) [numpy], kp(t), and drift metadata for one sea.

    drift_scale multiplies the spec drift rates. The pre-registered gate runs
    at drift_scale = 1.0 (kp +-15% per 60 s, direction 10-15 deg per window);
    other values are a SENSITIVITY diagnostic only.
    """
    rng = np.random.default_rng(seed)
    kmag = KP0 * np.exp(rng.uniform(np.log(0.4), np.log(2.4), NCOMP))
    dir0 = rng.uniform(-np.pi, np.pi)
    theta = rng.normal(dir0, np.deg2rad(25.0), NCOMP)
    kx, ky = kmag * np.cos(theta), kmag * np.sin(theta)
    om = np.sqrt(G * kmag * np.tanh(kmag * DEPTH))
    phi = rng.uniform(-np.pi, np.pi, NCOMP)
    a0 = np.abs(rng.normal(0.0, 1.0, NCOMP))

    if drift:
        rate_k = drift_scale * rng.choice([-1.0, 1.0]) * 0.15 / 60.0
        rot = drift_scale * rng.choice([-1.0, 1.0]) \
            * np.deg2rad(rng.uniform(10.0, 15.0))
    else:
        rate_k, rot = 0.0, 0.0

    def kp_t(t):
        return KP0 * (1.0 + rate_k * np.asarray(t, float))

    def amp_t(t):
        t = np.atleast_1d(t).astype(float)
        kpt = KP0 * (1.0 + rate_k * t)                       # (n,)
        dirt = dir0 + rot * (t / 20.0)                       # (n,)
        lw = np.log(kmag[None, :] / kpt[:, None])            # (n, N)
        dth = np.arctan2(np.sin(theta[None, :] - dirt[:, None]),
                         np.cos(theta[None, :] - dirt[:, None]))
        W = np.exp(-0.5 * (lw / S_K) ** 2 - 0.5 * (dth / S_D) ** 2)
        a = a0[None, :] * W                                  # (n, N)
        var = (a ** 2).sum(1) / 2.0
        c = SIGMA / np.sqrt(np.maximum(var, 1e-12))
        return a * c[:, None]

    def eta(P):
        x, y, t = P[:, 0], P[:, 1], P[:, 2]
        psi = (x[:, None] * kx[None] + y[:, None] * ky[None]
               - t[:, None] * om[None] + phi[None])
        return (amp_t(t) * np.cos(psi)).sum(1)

    return eta, kp_t, dict(rate_k=float(rate_k), rot_deg=float(np.rad2deg(rot)))


# ------------------------------------------------------------------- helpers
def w_of(lam, t):
    if lam >= 1.0:
        return None
    age = (T0 - t) / AGE_DT
    return torch.pow(torch.tensor(float(lam), device=DEV), age).clamp_min(1e-3)


def build_bank_and_masks(kp):
    """The bank mmse_gpu builds internally (kp, BAND, dk=kp/10), plus the 3
    band index masks over the 2B (cos|sin) coefficient vector."""
    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
    ratio = (torch.hypot(kxb, kyb) / kp).cpu().numpy()
    bmask2 = []
    for i in range(3):
        hi = SUBBANDS[i + 1] + (1e-6 if i == 2 else 0.0)
        m = (ratio >= SUBBANDS[i]) & (ratio < hi)
        mt = torch.tensor(m, device=DEV)
        bmask2.append(torch.cat([mt, mt]))            # cos and sin indices
    return (kxb, kyb, om), bmask2


def nested_contribs(model, Aq, Phi_obs, obs_t, z_obs, bmask2):
    """Per-band field contributions for each lam, sharing ONE batch prior.

    K_b = K + s2 (diag(1/w_b^2) - I); band b's contribution to the field is
    (Phi_q * gam)[:, S_b] @ (Phi_obs^T K_b^-1 z)[S_b]. K_b depends only on lam,
    so 5 solves cover all (band, lam). At lam=1, K_b = K -> the summed field is
    the batch field exactly.
    """
    gam, s2, K = model['gam'], float(model['s2']), model['K']
    n = len(z_obs)
    di = torch.arange(n, device=DEV)
    e1, contribs = {}, {}
    for lam in LAMS:
        if lam >= 1.0:
            Kb = K
        else:
            w = w_of(lam, obs_t)
            Kb = K.clone()
            Kb[di, di] = Kb[di, di] + s2 * (1.0 / w ** 2 - 1.0)
        alpha = torch.linalg.solve(Kb, z_obs)
        wpred = Phi_obs.t() @ alpha                    # (2B,)
        e1[lam] = (Aq @ wpred).detach().cpu().numpy()
        for b, bm in enumerate(bmask2):
            contribs[(b, lam)] = (Aq[:, bm] @ wpred[bm]).detach().cpu().numpy()
    return e1, contribs


def nrmse(field, gt, sg):
    f = field - field.mean()
    return float(np.sqrt(np.mean((f - gt) ** 2)) / sg)


def deploy_oracle(Pt, zt, Qt, kp, gt, sg, seed):
    """Deployable scalar forgetting: fresh mmse_gpu with obs_weight per lam
    (re-fits its prior each lam). GT-oracle over the lam grid. Context only /
    pair for E3; the gate uses the nested E1."""
    with torch.no_grad():
        nr = {lam: nrmse(mmse_gpu(Pt, zt, DEPTH, Qt, sigma0=S0, kp_hint=kp,
                                  band=BAND, obs_weight=w_of(lam, Pt[:, 2]),
                                  seed=seed).detach().cpu().numpy(), gt, sg)
              for lam in LAMS}
    lam = min(LAMS, key=lambda l: nr[l])
    return nr[lam], lam


# ------------------------------------------------------------------ one sea
def run_sea(seed, P, Pq, drift, drift_scale=1.0):
    eta, kp_t, meta = make_sea(seed, drift=drift, drift_scale=drift_scale)
    Pnp, Qnp = P.cpu().numpy(), Pq.cpu().numpy()

    gt = eta(Qnp)
    gt = gt - gt.mean()
    sg = float(gt.std())
    if sg < 1e-6:
        return None

    rng = np.random.default_rng(9000 + seed)
    z = eta(Pnp) + rng.normal(0.0, NOISE, len(Pnp))
    Pt, Qt = P, Pq
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)

    kp_win = float(np.mean(kp_t(Pnp[:, 2])))     # window-average kp (deployable)
    kp_true = float(kp_t(0.0))                   # kp at the scoring instant

    # held-out split of OBSERVATIONS (no GT), for val-selection
    n = len(z)
    ridx = np.random.default_rng(400 + seed).permutation(n)
    nh = int(HELD * n)
    H = torch.tensor(ridx[:nh], device=DEV)
    Tr = torch.tensor(ridx[nh:], device=DEV)
    zH = z[ridx[:nh]]

    (kxb, kyb, om), bmask2 = build_bank_and_masks(kp_win)

    with torch.no_grad():
        # ---- batch model (E0) + full-obs nested contribs -------------------
        out0, model0 = mmse_gpu(Pt, zt, DEPTH, Qt, sigma0=S0, kp_hint=kp_win,
                                band=BAND, seed=seed, return_model=True)
        E0 = nrmse(out0.detach().cpu().numpy(), gt, sg)
        Aq = design(Qt, kxb, kyb, om) * model0['gam'][None]
        e1_full, contribs_full = nested_contribs(
            model0, Aq, model0['Phi'], Pt[:, 2], zt, bmask2)

        # ---- train model + val-set nested contribs (predict held-out obs) --
        _, modelTr = mmse_gpu(Pt[Tr], zt[Tr], DEPTH, Pt[H], sigma0=S0,
                              kp_hint=kp_win, band=BAND, seed=seed,
                              return_model=True)
        AqV = design(Pt[H], kxb, kyb, om) * modelTr['gam'][None]
        e1_val, contribs_val = nested_contribs(
            modelTr, AqV, modelTr['Phi'], Pt[Tr, 2], zt[Tr], bmask2)

    # C3 fidelity: nested lam=1 field must equal the batch field
    c3_err = abs(nrmse(e1_full[1.0], gt, sg) - E0)

    # ---- E1 nested (the pre-registered scalar baseline) --------------------
    e1_nr = {lam: nrmse(e1_full[lam], gt, sg) for lam in LAMS}
    e1_valrms = {lam: float(np.sqrt(np.mean((e1_val[lam] - zH) ** 2)))
                 for lam in LAMS}
    lam_o = min(LAMS, key=lambda l: e1_nr[l])
    E1_oracle = e1_nr[lam_o]
    lam_v = min(LAMS, key=lambda l: e1_valrms[l])
    E1_val = e1_nr[lam_v]

    # ---- E2 per-band: oracle (GT) and val over the 5^3 combos --------------
    best_o, best_o_lams = np.inf, None
    best_v, best_vfield = np.inf, None
    for l0, l1, l2 in product(LAMS, LAMS, LAMS):
        field = contribs_full[(0, l0)] + contribs_full[(1, l1)] \
            + contribs_full[(2, l2)]
        nr = nrmse(field, gt, sg)
        if nr < best_o:
            best_o, best_o_lams = nr, (l0, l1, l2)
        vp = contribs_val[(0, l0)] + contribs_val[(1, l1)] \
            + contribs_val[(2, l2)]
        vr = float(np.sqrt(np.mean((vp - zH) ** 2)))
        if vr < best_v:
            best_v, best_vfield = vr, field
    E2_oracle = best_o
    E2_val = nrmse(best_vfield, gt, sg)

    # ---- E1 deploy (context) + E3 (kp band-placement) ---------------------
    E1_deploy, lam_dep = deploy_oracle(Pt, zt, Qt, kp_win, gt, sg, seed)
    E3_oracle, lam_3 = deploy_oracle(Pt, zt, Qt, kp_true, gt, sg, seed)

    return dict(
        seed=seed, sg=sg, kp_win=kp_win, kp_true=kp_true, meta=meta,
        E0=E0, E1_val=E1_val, E1_oracle=E1_oracle, lam_o=lam_o, lam_v=lam_v,
        E2_val=E2_val, E2_oracle=E2_oracle, E2_oracle_lams=best_o_lams,
        E1_deploy=E1_deploy, lam_dep=lam_dep, E3_oracle=E3_oracle, lam_3=lam_3,
        c3_err=c3_err, gap=E1_oracle - E2_oracle)


def clustered(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('inf')
    return m, se, float(m / se) if se > 0 else 0.0


def summarise(rows, label):
    keys = ['E0', 'E1_val', 'E1_oracle', 'E2_val', 'E2_oracle', 'E3_oracle',
            'E1_deploy', 'c3_err']
    s = {k: clustered([r[k] for r in rows]) for k in keys}
    s['gap'] = clustered([r['gap'] for r in rows])
    s['n'] = len(rows)
    s['lam_o'] = [r['lam_o'] for r in rows]
    s['lam_v'] = [r['lam_v'] for r in rows]
    s['label'] = label
    return s


def print_table(s):
    print(f"\n=== {s['label']}  (n = {s['n']} seas) ===")
    print(f"{'estimator':<28}{'nRMSE':>10}{'SE':>9}")
    names = [('E0  batch (lam=1)', 'E0'),
             ('E1  scalar-lam  val', 'E1_val'),
             ('E1  scalar-lam  ORACLE', 'E1_oracle'),
             ('E1  scalar deploy ORACLE', 'E1_deploy'),
             ('E2  per-band    val', 'E2_val'),
             ('E2  per-band    ORACLE', 'E2_oracle'),
             ('E3  kp(0)-track ORACLE', 'E3_oracle')]
    for nm, k in names:
        m, se, _ = s[k]
        print(f"{nm:<28}{m:>10.4f}{se:>9.4f}")
    gm, gse, gt = s['gap']
    print(f"\n  gap = E1-oracle - E2-oracle : {gm:+.4f} +- {gse:.4f} "
          f"(t = {gt:+.2f})")
    print(f"  C3 superposition |E1nested(1)-E0| : {s['c3_err'][0]:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=32)
    ap.add_argument('--drift-scale', type=float, default=1.0,
                    help='1.0 = the pre-registered spec drift (gate). Other '
                         'values are a sensitivity diagnostic only.')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.quick:
        a.seas = 3

    P, Pq = geometry()
    print(f"obs {len(P)} over [{float(P[:,2].min()):.0f}, "
          f"{float(P[:,2].max()):.0f}] s   query {len(Pq)} at t=0", flush=True)
    print(f"lam grid {LAMS}   bands {SUBBANDS}   sigma0 {S0}  noise {NOISE}\n")
    print("PRE-REGISTERED gate: gap = nRMSE(E1-oracle) - nRMSE(E2-oracle)")
    print("  gap < 0.005 OR t < 2  -> H1 CLOSES (scalar saturates), no training")
    print("  gap >= 0.005 AND t>=2 -> H1 has a target, train d3_t2 below ceiling")

    if a.drift_scale != 1.0:
        print(f"\n*** SENSITIVITY RUN drift_scale={a.drift_scale} "
              f"(NOT the pre-registered gate) ***")
    res = {'drift_scale': a.drift_scale}
    for label, drift in (('non-stationary', True), ('stationary', False)):
        rows = []
        for s in range(a.seas):
            r = run_sea(s, P, Pq, drift, drift_scale=a.drift_scale)
            if r is not None:
                rows.append(r)
            if (s + 1) % 8 == 0:
                print(f"  [{label}] {s+1}/{a.seas}", flush=True)
        summ = summarise(rows, label)
        print_table(summ)
        # store (drop bulky per-lam dicts)
        res[label] = {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in summ.items() if k != 'label'}
        res[label]['rows'] = [{k: r[k] for k in
                               ('seed', 'kp_win', 'kp_true', 'E0', 'E1_val',
                                'E1_oracle', 'E1_deploy', 'E2_val', 'E2_oracle',
                                'E3_oracle', 'gap', 'c3_err', 'lam_o', 'lam_v',
                                'lam_dep', 'lam_3', 'meta')}
                              for r in rows]

    # ------------------------------------------------ gate + controls
    ns = res['non-stationary']
    st = res['stationary']
    gm, gse, gt = ns['gap']
    gate_open = (gm >= 0.005) and (gt >= 2.0)

    # C1 stationary negative control
    st_e1 = st['E1_oracle'][0] - st['E0'][0]
    lam_o_stat = st['lam_o']
    frac_lam1 = float(np.mean([l == 1.0 for l in lam_o_stat]))
    # C2 val per-band must not hurt > 0.002 on stationary
    c2 = st['E2_val'][0] - st['E0'][0]
    # C3 superposition fidelity on stationary (nested lam=1 == batch)
    c3 = st['c3_err'][0]

    print("\n" + "=" * 70)
    print("STEP-0 ACCOUNTING VERDICT")
    print("=" * 70)
    print(f"  non-stationary gap E1o - E2o : {gm:+.4f} +- {gse:.4f} (t {gt:+.2f})")
    print(f"  GATE (gap>=0.005 AND t>=2)   : "
          f"{'OPEN -> train d3_t2' if gate_open else 'CLOSED -> H1 dead, no training'}")
    print(f"\n  controls:")
    print(f"   C1 stationary E1o-E0        : {st_e1:+.4f}  "
          f"(oracle lam=1 on {frac_lam1*100:.0f}% of stationary seas)")
    print(f"   C2 stationary E2val-E0<=.002: {c2:+.4f}  "
          f"{'PASS' if c2 <= 0.002 else 'FAIL'}")
    print(f"   C3 superposition |E1n(1)-E0|<.005: {c3:.2e}  "
          f"{'PASS' if c3 < 0.005 else 'FAIL'}")
    print(f"\n  scalar time channel is real (E0->E1-oracle on non-stat): "
          f"{ns['E0'][0]-ns['E1_oracle'][0]:+.4f}")
    print(f"  deployable scalar (E0->E1-deploy on non-stat)           : "
          f"{ns['E0'][0]-ns['E1_deploy'][0]:+.4f}")
    print(f"  kp band-placement share (E1deploy->E3o on non-stat)     : "
          f"{ns['E1_deploy'][0]-ns['E3_oracle'][0]:+.4f}")

    res['gate'] = dict(gap=gm, gap_se=gse, gap_t=gt, gate_open=bool(gate_open),
                       c1_stat_e1_minus_e0=st_e1, c1_frac_lam1=frac_lam1,
                       c2_stat_e2val_minus_e0=c2, c3_superpos_err=c3,
                       scalar_channel=ns['E0'][0] - ns['E1_oracle'][0],
                       scalar_deploy=ns['E0'][0] - ns['E1_deploy'][0],
                       kp_share=ns['E1_deploy'][0] - ns['E3_oracle'][0])

    os.makedirs('results/direction3', exist_ok=True)
    out = OUT if a.drift_scale == 1.0 else \
        OUT.replace('.json', f'_scale{a.drift_scale:g}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
