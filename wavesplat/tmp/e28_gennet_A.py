"""
e28_gennet_A.py — a learned wave-field prior designed to GENERALIZE by
MMSE-anchored do-no-harm gating + informativeness conditioning + a physics
(bound-wave) residual arm. Train + LORO-eval + selftest in one file.

--------------------------------------------------------------------------
DESIGN PRINCIPLE (project findings E-16..E-27)
  A learned prior helps in proportion to observation UNDER-DETERMINATION
  (aperture / field-reconstruction CRB). Where the field is WELL-determined
  its only residual value is bound-wave (nonlinear) misspecification.
  Failures this design targets:
    E-16  the current prior_net can LOSE on well-determined linear fields.
    E-24  CRB (via the obs-density axis) independently predicts the gain.
    E-25  a HARD CRB threshold-gate fails -> the gate must be SOFT/LEARNED.
    E-21  the residual value where the field is determined is bound-wave.

ARCHITECTURE (GenNet)
  classical anchor : beamform-power prior gamma0(k) fed to the linear MMSE
                     (mmse_gpu / mmse_bank_estimate), pexp=1. This is the
                     do-no-harm baseline the net must revert to.
  residual net r(k): a CNN (PriorNet backbone, OUTPUT ZERO-INIT) on the
                     3+3 ch normalized beamform-power maps -> a per-bank-atom
                     LOG10 residual (deviation of log-gamma from the classical
                     log-prior).  gamma_free = gamma0 * 10^(alpha_prior * r).
  info head        : an MLP on 7 cheap INFORMATIVENESS features -> three
                     scalars: alpha_prior in [0,1], alpha_bw in [0,1],
                     log_smult (scene noise floor).  alpha_prior GATES the
                     free-wave residual (=> revert to classical when it -> 0);
                     alpha_bw GATES a 2nd-order (2k,2w) OFF-SHELL bound-wave
                     arm whose prior rides on gamma_free.
  ANCHOR GUARANTEE : at alpha_prior=alpha_bw=0 the solve is EXACTLY the
                     classical free-wave beamform-prior MMSE (selftest checks
                     |field(a=0) - classical| < 1e-4).
  informativeness features f (all cheap, from P,z,kp,sigma0):
    f0 log10 n_obs
    f1 log10(1+ kp*apertureX/2pi)   aperture in wavelengths (x)
    f2 log10(1+ kp*apertureY/2pi)   aperture in wavelengths (y)
    f3 log10(1+ w_p*T/2pi)          temporal coverage in wave periods
    f4 log10(std(z)/sigma0)         SNR (high => well-determined)
    f5 log10(pw_max/pw_mean)        spectral-peak concentration
    f6 log10(1+ n/(kp^2 Ax Ay/(2pi)^2))  E-24 obs-density axis (looks/wl^2)
  Features are standardized by train-set mean/std (stored in the ckpt).

TRAINING / AUGMENTATION (widened vs the current net's Hs 1-3)
  Hs ~ U[1,5] ; Tp ~ U[5,11] ; JONSWAP gamma ~ U[1.5,6] ; spread U[15,45].
  obs conditions: fixed / track U/cp U[0.1,1.5] ; nfrac U[0.02,0.10] ;
  bound-wave dose bw_inject U[0,0.4] (2nd-order Stokes super-harmonics).
  loss = normalized field-MSE(task, held-out query pixels) through the
         differentiable MMSE  +  lambda_gate*(alpha_prior+alpha_bw)  (the
         gate sparsity term forces the gate CLOSED unless it earns task loss
         -> mitigation for the "gate collapses to always-on" failure).
  LORO: the TRAIN split withholds whole regimes (see loro_eval).

PRE-REGISTERED LORO SUCCESS CRITERION (all four must hold)
  Held-out seas are binned by the MEASURED field-reconstruction determinacy
  (oracle_ls2: true wavenumbers given, A/phi by ridge LS, corr with GT) -- NOT
  by an a-priori aperture label, because the fixed disk is LESS determined than
  a moving track at the unobserved pixels (synthetic-aperture effect; the
  smoke confirmed fixed det ~0.23 < track det ~0.37). "well" = top determinacy
  third, "under" = bottom third.
  (a) DO-NO-HARM  : mean(gennet - classical_MMSE) on the TOP-determinacy third
                    >= -0.01 corr.
  (b) UNDER-DET   : mean(gennet - current_net) on the BOTTOM-determinacy third
                    >= 0, with win-rate >= 1/2.
  (c) EXTRAPOLATE : mean(gennet - classical_MMSE) on a held-out
                    steepness+aperture never trained (Hs 4-5, U=1.5cp,
                    bw 0.3-0.5) >= -0.01 (no harm; ideally beats current net).
  (d) GATE SANITY : mean(alpha_prior|top-det) < mean(alpha_prior|bottom-det)
                    - 0.03  AND  std(alpha_prior) > 0.02  AND (reported)
                    corr(alpha_prior, 1-determinacy) > 0 -- the gate closes
                    where the field is determined and is not collapsed.

SELF-REVIEW (blunt) is printed by --selftest and summarized in the report.

Usage:
  python3 tmp/e28_gennet_A.py --selftest
  python3 tmp/e28_gennet_A.py --smoke              # tiny train + tiny LORO
  python3 tmp/e28_gennet_A.py --steps 8000 --n_train 4000 --eval_seas 12
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d18_relax_estimator import beamform_top, omega
from d3_sa_nomp import beamform_power, split_masks, mmse_bank_estimate
from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d3_prior_net_eval import net_gamma_mmse
from d3_l1_s2head_train import SmultHead
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, DEPTH)

G = 9.81
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FEAT = 7
CKPT = 'results/direction3/e28_gennet_A.pt'
INIT = 'results/direction3/prior_net_l1s2bw.pt'   # warm feature extractor

# ---- fixed normalized bank atoms (band annulus of the 51x51 grid) ----------
_MASK = band_mask()
_AXN = np.linspace(-KREL, KREL, NGRID)
_KXN, _KYN = np.meshgrid(_AXN, _AXN)
KXN = torch.tensor(_KXN[_MASK], dtype=torch.float32, device=DEVICE)
KYN = torch.tensor(_KYN[_MASK], dtype=torch.float32, device=DEVICE)
MASK_T = torch.tensor(_MASK, device=DEVICE)


# ==========================================================================
#  informativeness features + input maps (identical recipe train & inference)
# ==========================================================================
def beamform_prior_np(P, z, kxb, kyb, om, chunk=512):
    B = len(kxb)
    pw = np.empty(B)
    for i in range(0, B, chunk):
        psi = (kxb[i:i + chunk, None] * P[None, :, 0]
               + kyb[i:i + chunk, None] * P[None, :, 1]
               - om[i:i + chunk, None] * P[None, :, 2])
        pw[i:i + chunk] = (z[None] * np.cos(psi)).sum(1) ** 2 \
            + (z[None] * np.sin(psi)).sum(1) ** 2
    return pw


def info_features(P, z, kp, sigma0):
    """7 cheap informativeness features (numpy). Depends only on the obs and
    the estimated peak wavenumber -- NOT on the ground truth."""
    n = len(z)
    ax = P[:, 0].max() - P[:, 0].min()
    ay = P[:, 1].max() - P[:, 1].min()
    T = P[:, 2].max() - P[:, 2].min()
    wp = float(omega(kp, DEPTH))
    axw = kp * ax / (2 * np.pi)
    ayw = kp * ay / (2 * np.pi)
    kxb = KXN.cpu().numpy() * kp
    kyb = KYN.cpu().numpy() * kp
    om = np.sqrt(G * np.hypot(kxb, kyb) * np.tanh(np.hypot(kxb, kyb) * DEPTH))
    pw = beamform_prior_np(P, z, kxb, kyb, om)
    pw = pw / max(pw.max(), 1e-12)
    conc = pw.max() / max(pw.mean(), 1e-12)
    dens = n / (kp ** 2 * ax * ay / (2 * np.pi) ** 2 + 1e-9)
    f = np.array([
        np.log10(n),
        np.log10(1 + axw),
        np.log10(1 + ayw),
        np.log10(1 + wp * T / (2 * np.pi)),
        np.log10(np.std(z) / (sigma0 + 1e-9) + 1e-9),
        np.log10(conc + 1e-9),
        np.log10(1 + dens),
    ], np.float32)
    return f


def make_X(P, z, kp):
    """6-channel normalized beamform-power maps (matches d3_prior_net_data)."""
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], DEPTH, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], DEPTH, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    return np.stack([p1, p2, pm,
                     np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                     np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)


# ==========================================================================
#  the net: residual backbone + informativeness head
# ==========================================================================
class InfoHead(nn.Module):
    """standardized info features -> (alpha_prior, alpha_bw, log_smult)."""

    def __init__(self, n_feat=N_FEAT, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_feat, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, 3))
        with torch.no_grad():
            self.f[-1].weight.mul_(0.1)
            # bias: alpha ~ sigmoid(-0.8) ~ 0.31 start; log_smult ~ log 3
            self.f[-1].bias[:] = torch.tensor([-0.8, -0.8, float(np.log(3.0))])

    def forward(self, feats):
        o = self.f(feats)
        a_prior = torch.sigmoid(o[..., 0])
        a_bw = torch.sigmoid(o[..., 1])
        log_smult = o[..., 2]
        return a_prior, a_bw, log_smult


class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PriorNet()               # -> residual log10 map
        self.head = InfoHead()
        self.register_buffer('feat_mean', torch.zeros(N_FEAT))
        self.register_buffer('feat_std', torch.ones(N_FEAT))

    def zero_output(self):
        with torch.no_grad():                    # do-no-harm init: r ~ 0
            self.backbone.out.weight.mul_(0.0)
            self.backbone.out.bias.mul_(0.0)

    def standardize(self, feats):
        return (feats - self.feat_mean) / (self.feat_std + 1e-6)

    def forward(self, X, feats):
        r = self.backbone(X)                     # (B,51,51) log10 residual
        a_prior, a_bw, log_smult = self.head(self.standardize(feats))
        return r, a_prior, a_bw, log_smult


# ==========================================================================
#  differentiable MMSE-anchored solve  (torch; used in train and eval)
# ==========================================================================
def anchored_field(r_map, a_prior, a_bw, P, z, Q, kp, sigma0, log_smult,
                   bw_scale=0.15, clampr=4.0):
    """Field at Q via the linear MMSE with prior
        gamma_free = gamma0 * 10^(a_prior * r),
        gamma_bw   = a_bw * bw_scale * gamma_free   at (2k, 2w) off-shell.
    a_prior=a_bw=0  ->  gamma_free=gamma0, gamma_bw=0  =>  classical anchor."""
    kxb = KXN * kp
    kyb = KYN * kp
    k = torch.sqrt(kxb ** 2 + kyb ** 2 + 1e-12)
    om = torch.sqrt(G * k * torch.tanh(k * DEPTH) + 1e-12)
    # classical beamform-power prior on the atoms
    pw = _beamform_prior_t(P, z, kxb, kyb, om)
    pw = pw / pw.max().clamp_min(1e-12)
    r = r_map[MASK_T]
    gain = torch.pow(10.0, torch.clamp(a_prior * r, -clampr, clampr))
    g_free = pw * gain
    # bound-wave arm (rides on the free prior, gated by a_bw)
    kxb2 = torch.cat([kxb, 2.0 * kxb])
    kyb2 = torch.cat([kyb, 2.0 * kyb])
    om2 = torch.cat([om, 2.0 * om])
    g = torch.cat([g_free, a_bw * bw_scale * g_free])
    g = g / g.sum().clamp_min(1e-30)
    gam = torch.cat([g, g]) * torch.var(z)

    def design(Qp):
        psi = kxb2[None] * Qp[:, 0:1] + kyb2[None] * Qp[:, 1:2] \
            - om2[None] * Qp[:, 2:3]
        return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)

    Phi = design(P)
    s2 = (torch.exp(log_smult) * sigma0) ** 2
    n = len(z)
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(n, device=P.device)
    alpha = torch.linalg.solve(K, z)
    pred = (design(Q) * gam[None]) @ (Phi.T @ alpha)
    return pred


def _beamform_prior_t(P, z, kxb, kyb, om, chunk=1024):
    B = len(kxb)
    pw = torch.empty(B, device=P.device, dtype=P.dtype)
    for i in range(0, B, chunk):
        psi = (kxb[i:i + chunk, None] * P[None, :, 0]
               + kyb[i:i + chunk, None] * P[None, :, 1]
               - om[i:i + chunk, None] * P[None, :, 2])
        pw[i:i + chunk] = (z[None] * torch.cos(psi)).sum(1) ** 2 \
            + (z[None] * torch.sin(psi)).sum(1) ** 2
    return pw


def classical_anchor_field(P, z, Q, kp, sigma0, smult):
    """Independent re-implementation of the free-wave beamform-prior MMSE
    (pexp=1, given smult, no EM). Used to VERIFY anchored_field(a=0)."""
    kxb = (KXN * kp).cpu().numpy()
    kyb = (KYN * kp).cpu().numpy()
    om = np.sqrt(G * np.hypot(kxb, kyb) * np.tanh(np.hypot(kxb, kyb) * DEPTH))
    pw = beamform_prior_np(P.cpu().numpy(), z.cpu().numpy(), kxb, kyb, om)
    pw = pw / max(pw.max(), 1e-12)
    g = pw / pw.sum()
    gam = np.concatenate([g, g]) * float(z.var(unbiased=True))

    def design(Qp):
        psi = kxb[None] * Qp[:, 0:1] + kyb[None] * Qp[:, 1:2] \
            - om[None] * Qp[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    Pn, zn, Qn = P.cpu().numpy(), z.cpu().numpy(), Q.cpu().numpy()
    Phi = design(Pn)
    s2 = (smult * sigma0) ** 2
    n = len(zn)
    K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(n)
    w = Phi.T @ np.linalg.solve(K, zn)
    return (design(Qn) * gam[None]) @ w


# ==========================================================================
#  widened data generation (LORO-aware)
# ==========================================================================
# TRAIN withholds: high Hs (>3), the fast track (U>0.43), and the high
# bound-wave dose -- these are the held-out axes. 'spread' seas span the
# eval determinacy range; they are binned by MEASURED oracle_ls2 determinacy
# (not by an a-priori label -- the smoke showed fixed<track determinacy, the
# synthetic-aperture effect, so a-priori 'well'/'under' labels are unsafe).
REGIMES = {
    # name        Hs         U (in c_p)   nfrac        bw
    'train':  dict(hs=(1.0, 3.0), U=(0.0, 0.43), nf=(0.03, 0.08), bw=(0.0, 0.3)),
    'spread': dict(hs=(1.5, 3.0), U=(0.0, 1.5),  nf=(0.02, 0.09), bw=(0.0, 0.2)),
    'extrap': dict(hs=(4.0, 5.0), U=(1.5, 1.5),  nf=(0.04, 0.08), bw=(0.3, 0.5)),
}


def gen_sample(rng, regime='train'):
    """One widened training sample; returns arrays + info features."""
    R = REGIMES[regime]
    Tp = rng.uniform(5.0, 11.0)
    Hs = rng.uniform(*R['hs'])
    gam_j = rng.uniform(1.5, 6.0)
    spread = rng.uniform(15.0, 45.0)
    mdir = rng.uniform(0.0, 360.0)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=gam_j, direction_deg=mdir,
                           direction_spread_deg=spread, seed=seed)
    kxm = np.array([x.kx for x in c])
    kym = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c])
    phm = np.array([x.phi for x in c])
    k = np.hypot(kxm, kym)
    om = np.sqrt(G * k * np.tanh(k * DEPTH) + 1e-12)
    uc = rng.uniform(0.0, 0.4)
    ua = rng.uniform(0, 2 * np.pi)
    om = om + kxm * uc * np.cos(ua) + kym * uc * np.sin(ua)

    # 2nd-order self-super-harmonic bound waves
    bw_lvl = rng.uniform(*R['bw'])
    bw = None
    if bw_lvl > 1e-4:
        K = min(12, len(Am))
        top = np.argsort(Am)[-K:]
        bw = dict(kx=2 * kxm[top], ky=2 * kym[top], om=2 * om[top],
                  ph=2 * phm[top],
                  A=0.5 * np.hypot(kxm[top], kym[top]) * Am[top] ** 2)

    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = 10 * Tp
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Ulo, Uhi = R['U']
    ucp = rng.uniform(Ulo, Uhi)
    mode = 'fixed' if ucp < 1e-6 else 'moving'
    P = sample_obs(mode, ucp * cp, lam, cp, T, gx, gy, r_fp, rng)
    nfrac = rng.uniform(*R['nf'])
    psi = (kxm[None] * P[:, 0:1] + kym[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + phm[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    if bw is not None:
        psib = (bw['kx'][None] * P[:, 0:1] + bw['ky'][None] * P[:, 1:2]
                - bw['om'][None] * P[:, 2:3] + bw['ph'][None])
        zb = (bw['A'][None] * np.cos(psib)).sum(1)
        bw['sc'] = (np.sqrt(bw_lvl) * (Hs / 4.0)) / max(np.std(zb), 1e-9)
        z = z + bw['sc'] * zb

    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)

    X = make_X(P, z, kp)
    feats = info_features(P, z, kp, nfrac * Hs)

    qx = rng.uniform(0, side, 256)
    qy = rng.uniform(0, side, 256)
    tq = np.full(256, T)
    psi_q = (kxm[None] * qx[:, None] + kym[None] * qy[:, None]
             - om[None] * tq[:, None] + phm[None])
    gt_q = (Am[None] * np.cos(psi_q)).sum(1)
    if bw is not None:
        psib_q = (bw['kx'][None] * qx[:, None] + bw['ky'][None] * qy[:, None]
                  - bw['om'][None] * tq[:, None] + bw['ph'][None])
        gt_q = gt_q + bw['sc'] * (bw['A'][None] * np.cos(psib_q)).sum(1)
    Q = np.column_stack([qx, qy, tq]).astype(np.float32)
    return dict(X=X, P=P.astype(np.float32), z=z.astype(np.float32),
                Q=Q, gt_q=gt_q.astype(np.float32), feats=feats,
                kp=float(kp), Hs=float(Hs), sigma0=float(nfrac * Hs))


def gen_batch(n, seed, regime='train'):
    rng = np.random.default_rng(seed)
    return [gen_sample(rng, regime) for _ in range(n)]


# ==========================================================================
#  training
# ==========================================================================
def train(steps, n_train, bs, lr, lam_gate, seed, ckpt):
    torch.manual_seed(seed)
    t0 = time.time()
    print(f"generating {n_train} widened train samples ...", flush=True)
    data = gen_batch(n_train, seed)
    print(f"  gen done ({time.time()-t0:.0f}s)", flush=True)
    feats = np.stack([d['feats'] for d in data])
    gn = GenNet().to(DEVICE)
    try:
        ck = torch.load(INIT, map_location=DEVICE)
        gn.backbone.load_state_dict(ck['model'])
        print(f"  warm-started backbone from {INIT}")
    except Exception as e:
        print(f"  backbone from scratch ({e})")
    gn.zero_output()                              # do-no-harm start
    gn.feat_mean.copy_(torch.tensor(feats.mean(0)))
    gn.feat_std.copy_(torch.tensor(feats.std(0) + 1e-6))

    opt = torch.optim.AdamW(
        [{'params': gn.backbone.parameters(), 'lr': lr},
         {'params': gn.head.parameters(), 'lr': 1e-3}], weight_decay=1e-5)
    n_val = max(1, n_train // 10)
    ti = np.arange(n_val, n_train)
    vi = np.arange(n_val)
    rng = np.random.default_rng(seed)

    def one(d):
        P = torch.tensor(d['P'], device=DEVICE)
        z = torch.tensor(d['z'], device=DEVICE)
        Q = torch.tensor(d['Q'], device=DEVICE)
        gt = torch.tensor(d['gt_q'], device=DEVICE)
        X = torch.tensor(d['X'][None], device=DEVICE)
        ft = torch.tensor(d['feats'][None], device=DEVICE)
        r, ap, ab, ls = gn(X, ft)
        pred = anchored_field(r[0], ap[0], ab[0], P, z, Q, d['kp'],
                              d['sigma0'], ls[0])
        task = ((pred - gt) ** 2).mean() / (d['Hs'] / 4) ** 2
        gate = lam_gate * (ap[0] + ab[0])
        return task, gate, float(ap[0].detach()), float(ab[0].detach())

    def val():
        gn.eval()
        with torch.no_grad():
            ts, aps, abs_ = [], [], []
            for i in vi[:80]:
                t, _, a, b = one(data[int(i)])
                ts.append(float(t)); aps.append(a); abs_.append(b)
        gn.train()
        return np.mean(ts), np.mean(aps), np.std(aps), np.mean(abs_)

    v0, ap0, aps0, ab0 = val()
    print(f"step     0 val {v0:.4f}  a_prior {ap0:.2f}+-{aps0:.2f} "
          f"a_bw {ab0:.2f}", flush=True)
    best = (np.inf, 0)
    for step in range(1, steps + 1):
        ii = rng.choice(ti, bs, replace=False)
        loss = 0.0
        for i in ii:
            t, g, _, _ = one(data[int(i)])
            loss = loss + t + g
        loss = loss / bs
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gn.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            vl, apm, aps, abm = val()
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                # feat_mean/feat_std are registered buffers -> already in
                # state_dict; no redundant numpy entries (keeps the ckpt
                # weights_only-loadable).
                torch.save(dict(model=gn.state_dict(), step=step,
                                val=float(vl)), ckpt)
                tag = ' *saved'
            print(f"step {step:5d} [{time.time()-t0:5.0f}s] "
                  f"train {float(loss.detach()):.4f} val {vl:.4f} "
                  f"a_prior {apm:.2f}+-{aps:.2f} a_bw {abm:.2f}{tag}",
                  flush=True)
    print(f"best val {best[0]:.4f} @ {best[1]} (step-0 {v0:.4f}) -> {ckpt}")
    # val-based model selection: eval the BEST checkpoint, not the drifted
    # final-step net (the gate keeps opening past the val optimum).
    gn.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                  weights_only=False)['model'])
    gn.eval()
    print(f"  reloaded best-val checkpoint (step {best[1]}) for evaluation")
    return gn


def load_gennet(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    gn = GenNet().to(DEVICE)
    gn.load_state_dict(ck['model'])
    gn.eval()
    return gn


def gennet_field(gn, P, z, Q, kp, sigma0):
    """Inference field (numpy) via the trained GenNet + anchored solve."""
    X = torch.tensor(make_X(P, z, kp)[None], device=DEVICE)
    ft = torch.tensor(info_features(P, z, kp, sigma0)[None], device=DEVICE)
    Pt = torch.tensor(P.astype(np.float32), device=DEVICE)
    zt = torch.tensor(z.astype(np.float32), device=DEVICE)
    Qt = torch.tensor(Q.astype(np.float32), device=DEVICE)
    with torch.no_grad():
        r, ap, ab, ls = gn(X, ft)
        f = anchored_field(r[0], ap[0], ab[0], Pt, zt, Qt, kp, sigma0, ls[0])
    return f.cpu().numpy(), float(ap[0]), float(ab[0])


# ==========================================================================
#  LORO evaluation
# ==========================================================================
def build_sea(regime, si):
    """A held-out sea + its obs + scoring geometry, mirroring the harness."""
    R = REGIMES[regime]
    rng = np.random.default_rng(7000 + 31 * si + hashabs(regime))
    Tp = 7.0
    Hs = rng.uniform(*R['hs'])
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=rng.uniform(0, 360),
                           direction_spread_deg=30.0, seed=4000 + si)
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = 10 * Tp
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)),
                                           replace=False)
    Xs, Ys = Xs[take], Ys[take]
    gt_u = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T)
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(*R['U']) * cp
    nfrac = rng.uniform(*R['nf'])
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    # bound-wave dose
    bw_lvl = rng.uniform(*R['bw'])
    if bw_lvl > 1e-4:
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
        bkx, bky = 2 * kx[top], 2 * ky[top]
        bom, bph = 2 * om[top], 2 * ph[top]
        bA = 0.5 * np.hypot(kx[top], ky[top]) * Am[top] ** 2
        psib = (bkx[None] * P[:, 0:1] + bky[None] * P[:, 1:2]
                - bom[None] * P[:, 2:3] + bph[None])
        zb = (bA[None] * np.cos(psib)).sum(1)
        sc = (np.sqrt(bw_lvl) * (Hs / 4)) / max(np.std(zb), 1e-9)
        z = z + sc * zb
        psib_q = (bkx[None] * Xs[:, None] + bky[None] * Ys[:, None]
                  - bom[None] * T + bph[None])
        gt_u = gt_u + sc * (bA[None] * np.cos(psib_q)).sum(1)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    # field-reconstruction DETERMINACY (independent under-determination
    # signal, non-degenerate): oracle_ls2 -- true wavenumbers of the top-32
    # modes given, amplitudes/phases by ridge LS, corr with GT at the scoring
    # pixels. This is the achievability harness's ceiling on how well the
    # field can be reconstructed once identification is solved; it is NOT a
    # function of the net's informativeness features (no circularity).
    order = np.argsort(Am)[::-1][:32]
    obs = np.column_stack([P, z]).astype(np.float32)
    try:
        A_l, ph_l = fit_amplitudes_phases(kx[order].astype(np.float32),
                                          ky[order].astype(np.float32), obs,
                                          DEPTH, lambda_reg=1.0)
        f_or = render_modes(kx[order], ky[order], np.asarray(A_l, float),
                            np.asarray(ph_l, float), DEPTH, Xs, Ys, T)
        det = corr(f_or, gt_u)
    except Exception:
        det = np.nan
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, kp=kp, sigma0=float(nfrac * Hs),
                det=float(det), Hs=float(Hs))


def _score_sea(gn, cur_net, cur_head, S, si):
    P, z, Pq, gt_u = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    f_cl = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=s0, kp_hint=kp, seed=si)
    f_cur = net_gamma_mmse(cur_net, P, z, DEPTH, Pq, s0, seed=si, kp_hint=kp,
                           smult_head=cur_head, bound_wave=True)
    f_gn, ap, ab = gennet_field(gn, P, z, Pq, kp, s0)
    return dict(cl=corr(f_cl, gt_u), cur=corr(f_cur, gt_u),
                gn=corr(f_gn, gt_u), ap=ap, ab=ab, det=S['det'], Hs=S['Hs'])


def loro_eval(gn, cur_net, cur_head, n_seas):
    """Held-out seas binned by MEASURED field-reconstruction determinacy
    (oracle_ls2), plus a separate held-out STEEPNESS+aperture axis (Hs 4-5,
    U=1.5cp, high bound wave) never seen in training."""
    print("\n===== LORO evaluation =====", flush=True)
    pooled = []
    for si in range(n_seas):
        r = _score_sea(gn, cur_net, cur_head, build_sea('spread', si), si)
        pooled.append(r)
        print(f"  spread sea {si}: det {r['det']:+.2f} | cl {r['cl']:+.3f} "
              f"cur {r['cur']:+.3f} gn {r['gn']:+.3f} | ap {r['ap']:.2f} "
              f"ab {r['ab']:.2f}", flush=True)
    n_ex = max(2, n_seas // 2)
    extrap = []
    for si in range(n_ex):
        r = _score_sea(gn, cur_net, cur_head, build_sea('extrap', si), si)
        extrap.append(r)
        print(f"  extrap sea {si}: det {r['det']:+.2f} | cl {r['cl']:+.3f} "
              f"cur {r['cur']:+.3f} gn {r['gn']:+.3f} | ap {r['ap']:.2f} "
              f"ab {r['ab']:.2f}", flush=True)

    # bin pooled by measured determinacy: top third = WELL, bottom third = UNDER
    order = np.argsort([r['det'] for r in pooled])
    k = max(1, len(pooled) // 3)
    under_set = [pooled[i] for i in order[:k]]      # lowest determinacy
    well_set = [pooled[i] for i in order[-k:]]      # highest determinacy

    def M(rows, key):
        return float(np.nanmean([r[key] for r in rows]))
    d_well = M(well_set, 'gn') - M(well_set, 'cl')          # do no harm
    d_under = M(under_set, 'gn') - M(under_set, 'cur')      # beat current net
    wr_under = float(np.mean([r['gn'] >= r['cur'] for r in under_set]))
    d_extrap = M(extrap, 'gn') - M(extrap, 'cl')
    ap_well, ap_under = M(well_set, 'ap'), M(under_set, 'ap')
    ap_all = [r['ap'] for r in pooled]
    und_all = [1 - r['det'] for r in pooled]
    gate_std = float(np.std(ap_all))
    gate_corr = (float(np.corrcoef(ap_all, und_all)[0, 1])
                 if np.std(und_all) > 1e-6 and gate_std > 1e-6 else np.nan)

    passes = dict(
        a_do_no_harm=(d_well >= -0.01),
        b_under_det=(d_under >= 0 and wr_under >= 0.5),
        c_extrapolate=(d_extrap >= -0.01),
        d_gate_sanity=((ap_well < ap_under - 0.03) and gate_std > 0.02),
    )
    print(f"\ndeterminacy bins: well(top {k}) det>={pooled[order[-k]]['det']:.2f}"
          f"  under(bot {k}) det<={pooled[order[k-1]]['det']:.2f}")
    print("----- pre-registered LORO criteria -----")
    print(f"(a) do-no-harm   gn-classical | high-det = {d_well:+.3f} "
          f">= -0.01 ? {passes['a_do_no_harm']}")
    print(f"(b) under-det    gn-current   | low-det  = {d_under:+.3f} "
          f"(win {wr_under:.0%}) >=0 & >=50% ? {passes['b_under_det']}")
    print(f"(c) extrapolate  gn-classical | Hs4-5    = {d_extrap:+.3f} "
          f">= -0.01 ? {passes['c_extrapolate']}")
    print(f"(d) gate sanity  a_prior high-det {ap_well:.2f} < low-det "
          f"{ap_under:.2f}-0.03 & std {gate_std:.3f}>0.02 ? "
          f"{passes['d_gate_sanity']}")
    print(f"    gate<->under-determination corr = {gate_corr:+.2f} "
          f"(design wants > 0)")
    verdict = all(passes.values())
    print(f"\nLORO VERDICT: {'PASS' if verdict else 'FAIL'} "
          f"(smoke n_seas={n_seas}; confirmatory needs n_seas>=18 pooled + "
          f"full-length training)")
    out = dict(pooled=pooled, extrap=extrap, passes=passes,
               verdict=bool(verdict), d_well=d_well, d_under=d_under,
               d_extrap=d_extrap, gate_corr=gate_corr, n_seas=n_seas)
    json.dump(out, open('results/direction3/e28_gennet_loro.json', 'w'),
              indent=1, default=float)
    return out


# ==========================================================================
#  self-test  (the load-bearing correctness gate)
# ==========================================================================
def selftest():
    print("=== e28_gennet self-test ===")
    d = gen_sample(np.random.default_rng(3), 'train')
    P = torch.tensor(d['P'], device=DEVICE)
    z = torch.tensor(d['z'], device=DEVICE)
    Q = torch.tensor(d['Q'], device=DEVICE)
    kp, s0 = d['kp'], d['sigma0']
    gn = GenNet().to(DEVICE)
    gn.zero_output()
    gn.eval()
    r = torch.zeros(NGRID, NGRID, device=DEVICE)
    smult = 3.0
    ls = torch.log(torch.tensor(smult, device=DEVICE))

    # 1) anchor: alpha=0 must equal the independent classical MMSE
    f0 = anchored_field(r, torch.tensor(0.0, device=DEVICE),
                        torch.tensor(0.0, device=DEVICE), P, z, Q, kp, s0,
                        ls).cpu().numpy()
    f_cl = classical_anchor_field(P, z, Q, kp, s0, smult)
    err = float(np.max(np.abs(f0 - f_cl)))
    print(f"[1] anchor |field(a=0) - classical|_max = {err:.2e}  "
          f"(<1e-4 ? {err < 1e-4})")

    # 2) zero-output backbone => r==0 => forward gate produces valid alphas
    X = torch.tensor(d['X'][None], device=DEVICE)
    ft = torch.tensor(d['feats'][None], device=DEVICE)
    with torch.no_grad():
        rr, ap, ab, lsm = gn(X, ft)
    print(f"[2] r max |.| = {float(rr.abs().max()):.2e} (zero-init)  "
          f"a_prior {float(ap):.3f} a_bw {float(ab):.3f} "
          f"in[0,1] ? {0 <= float(ap) <= 1 and 0 <= float(ab) <= 1}")

    # 3) opening the gate MUST change the field (net is actually wired)
    r2 = torch.randn(NGRID, NGRID, device=DEVICE) * 0.5
    f1 = anchored_field(r2, torch.tensor(1.0, device=DEVICE),
                        torch.tensor(1.0, device=DEVICE), P, z, Q, kp, s0,
                        ls).cpu().numpy()
    dchg = float(np.max(np.abs(f1 - f0)))
    print(f"[3] |field(a=1,r!=0) - field(a=0)|_max = {dchg:.3e} "
          f"(>0 ? {dchg > 1e-6})")

    # 4) gradient flows to the residual OUTPUT head AND the info head.
    #    NOTE: with exact zero-init of backbone.out, grad reaches out.weight
    #    but is BLOCKED to upstream backbone layers on step 1 (out.weight==0
    #    multiplies the upstream gradient) -- standard zero-init residual
    #    behavior; those layers unblock after the first optimizer step. We
    #    verify the two trainable heads receive gradient at init.
    gn.zero_grad()
    rr, ap, ab, lsm = gn(X, ft)
    pred = anchored_field(rr[0], ap[0], ab[0], P, z, Q, kp, s0, lsm[0])
    loss = ((pred - torch.tensor(d['gt_q'], device=DEVICE)) ** 2).mean() \
        + 0.02 * (ap[0] + ab[0])
    loss.backward()
    go = gn.backbone.out.weight.grad
    gh = gn.head.f[0].weight.grad
    grad_ok = (go is not None and float(go.abs().sum()) > 0
               and gh is not None and float(gh.abs().sum()) > 0)
    # confirm upstream unblocks after one step: perturb out then re-backward
    with torch.no_grad():
        gn.backbone.out.weight.add_(0.05)
    gn.zero_grad()
    rr, ap, ab, lsm = gn(X, ft)
    pred = anchored_field(rr[0], ap[0], ab[0], P, z, Q, kp, s0, lsm[0])
    (((pred - torch.tensor(d['gt_q'], device=DEVICE)) ** 2).mean()).backward()
    ge = gn.backbone.e1[0].weight.grad
    up_ok = ge is not None and float(ge.abs().sum()) > 0
    print(f"[4] grad out-head {grad_ok} info-head {grad_ok} | upstream "
          f"unblocks after out!=0 ? {up_ok}")

    # 5) feature determinism
    f_a = info_features(d['P'], d['z'], kp, s0)
    f_b = info_features(d['P'], d['z'], kp, s0)
    print(f"[5] info_features deterministic ? {np.allclose(f_a, f_b)} "
          f"shape {f_a.shape}")
    ok = (err < 1e-4 and dchg > 1e-6 and np.allclose(f_a, f_b)
          and grad_ok and up_ok)
    print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--n_train', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lam_gate', type=float, default=0.02)
    ap.add_argument('--eval_seas', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', default=CKPT)
    a = ap.parse_args()
    print(f"device {DEVICE}  bank atoms {len(KXN)}  n_feat {N_FEAT}",
          flush=True)

    if a.selftest:
        return 0 if selftest() else 1

    if a.smoke:
        a.steps = a.steps if a.steps != 8000 else 400
        a.n_train = a.n_train if a.n_train != 4000 else 500
        a.bs = 6
        a.eval_seas = a.eval_seas if a.eval_seas != 10 else 9

    gn = train(a.steps, a.n_train, a.bs, a.lr, a.lam_gate, a.seed, a.ckpt)

    # current-net baseline (l1s2bw + its SmultHead)
    ck = torch.load(INIT, map_location=DEVICE)
    cur = PriorNet().to(DEVICE); cur.load_state_dict(ck['model']); cur.eval()
    cur_head = SmultHead().to(DEVICE)
    cur_head.load_state_dict(ck['head']); cur_head.eval()

    loro_eval(gn, cur, cur_head, a.eval_seas)
    return 0


if __name__ == '__main__':
    sys.exit(main())
