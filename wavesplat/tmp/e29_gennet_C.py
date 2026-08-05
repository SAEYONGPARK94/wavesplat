"""
e29_gennet_C.py — design "C": CLASS-CEILING GATE + RESIDUAL-BOOST ESTIMATOR
============================================================================

VERDICT ON E-28 FIRST (read the code, not the summary): designs A and B were
NOT two orthogonal architectures. Both emit a diagonal gamma(k) over the SAME
fixed normalized 51x51 annulus bank and hand it to the SAME linear MMSE; only
the conditioning scheme differed (scalar gate vs FiLM), and both best-val
checkpoints sat <2 epochs from their warm start. E-28 therefore licenses only
"conditioning schemes on gamma-over-a-fixed-bank, near the incumbent basin,
don't beat the incumbent" — NOT "no architecture beats the incumbent".
Crucially, NEITHER measured the class ceiling: their determinacy proxy
(oracle_ls2, top-32 modes, ridge lam=1) is WEAKER than the incumbent in
several seas (e28_gennet_loro.json sea 3: det 0.036 vs cur 0.413), so it
bounds nothing.

DESIGN C = two parts, in order:

GATE-0 (no training): measure the DIAGONAL-GAMMA CLASS CEILING directly.
  Oracle gamma = true realized mode energies (A_m^2/2, incl. the injected
  bound-wave components) deposited bilinearly onto the exact bank the
  incumbent uses, swept over blur {0, 1.0 cells} x smult {0.5,1,3,10}, best
  per scene, through the exact same solver, on the exact same held-out seas
  as E-28-A (identical RNG; we verify per-sea cl/cur against
  e28_gennet_loro.json). Headroom H = mean(oracle_best - cur).
  H bounds every gamma-emitting design (A, B, and any conditioner).
  PRE-REGISTERED: if H < 0.03 on the spread arena, the class is declared
  saturated at the incumbent — full training is ABORTED and the deliverable
  is the ceiling measurement itself (first direct quantitative test of the
  E-28 "wall" interpretation).

BOOSTNET (leaves the one-shot class): frozen incumbent = stage 1.
  e = z - f1(obs). Stage-2 net sees the original 6ch beamform maps PLUS 6ch
  beamform maps OF THE RESIDUAL (re-normalized => structure that per-map
  max-normalization hid under the dominant peak in pass 1) and emits
  r2(k) (log-modulation of the residual-beamform prior) and a scalar
  beta (zero-init head + 2 residual-magnitude scalars). Final field
  f = f1 + beta * MMSE(gamma2 ; e). At init beta=0 => EXACTLY the incumbent
  (do-no-harm anchored at the thing to beat, not at the classical MMSE).
  MECHANISM: the MMSE noise floor (smult~3) + prior shrinkage leave
  systematic under-fit of weak/mis-prioritized components in the obs
  residual; a second solve on the residual reduces effective shrinkage
  adaptively (L2-boosting: 1-(1-h)^2 > h), and the net's only job is to
  decide WHERE the residual is signal vs noise. Precedent: the one learned
  method in this project that broke through to RELAX level was iterative
  beamform-detect + residual iteration (d3_beamform_net); RELAX/CLEAN are
  residual-iterative; "M8 residual boosting" was on record and never tried
  at this arena. This is a different estimator class (two data-dependent
  solves, nonlinear in z through the residual path), a different input
  signal (solver misfit, unavailable to any one-shot design), and a
  different anchor (incumbent, not classical).

BONUS DIAGNOSTIC ARM (not part of C's claim): "ens" = 3-seed incumbent
  ensemble (mean log-gamma of prior_net_l1s2bw{,_s1,_s2}, base smult head).
  Pure variance reduction; if THIS beats the incumbent, "no design wins"
  claims need a caveat regardless of C.

PRE-REGISTERED SUCCESS CRITERIA (declared before any training; all on
held-out seas, arms = cl(hand mmse_bank_estimate), cur(l1s2bw full path),
boost, oracle, ens):
  GATE-0    H = mean(oracle_best - cur) on >=16 spread seas.
            H < 0.03  => class saturated; training aborted; C reports the
            negative + ceiling numbers. (--force_train overrides, labeled
            exploratory.)
  If trained (H >= 0.03):
  (a) DO-NO-HARM  pooled mean(boost - cur) >= -0.005     (init is exact cur)
  (b) WIN         pooled mean(boost - cur) >= +0.02 AND win-rate >= 60%
                  over >= 16 spread seas.   (b) FAILS => C FAILS, full stop.
  (c) OOD         extrap regime (Hs 4-5, U=1.5cp, bw 0.3-0.5):
                  mean(boost - cur) >= 0 (no EX_Hs-style regression).
  (d) MECHANISM   Spearman(per-sea boost gain, per-sea oracle headroom) > 0
                  (boost should win where gamma headroom exists).

Usage:
  python3 tmp/e29_gennet_C.py --selftest
  python3 tmp/e29_gennet_C.py --smoke                # gate0(6) + tiny train + eval(6+3)
  python3 tmp/e29_gennet_C.py --gate0 --gate0_seas 18   # ceiling measurement only
  python3 tmp/e29_gennet_C.py --steps 4000 --n_train 2500 --eval_seas 18   # full
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

from scipy.ndimage import gaussian_filter

from wavesim.wave_generator import jonswap_components
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
CKPT = 'results/direction3/e29_gennet_C.pt'
INCUMBENT = 'results/direction3/prior_net_l1s2bw.pt'
SEED_CKPTS = ['results/direction3/prior_net_l1s2bw.pt',
              'results/direction3/prior_net_l1s2bw_s1.pt',
              'results/direction3/prior_net_l1s2bw_s2.pt']
E28_JSON = 'results/direction3/e28_gennet_loro.json'
OUT_GATE0 = 'results/direction3/e29_gate0_ceiling.json'
OUT_LORO = 'results/direction3/e29_gennet_C_loro.json'
CACHE = 'tmp/e29_train_cache.npy'
BW_SCALE = 0.15

_MASK = band_mask()
_AXN = np.linspace(-KREL, KREL, NGRID)
_KXN, _KYN = np.meshgrid(_AXN, _AXN)          # 'xy': [row=iy, col=ix]
KXN_NP, KYN_NP = _KXN[_MASK], _KYN[_MASK]     # (NB,) normalized annulus atoms
NB = len(KXN_NP)
KXN = torch.tensor(KXN_NP, dtype=torch.float32, device=DEVICE)
KYN = torch.tensor(KYN_NP, dtype=torch.float32, device=DEVICE)
MASK_T = torch.tensor(_MASK, device=DEVICE)

# same regimes as e28_gennet_A (arena comparability)
REGIMES = {
    'train':  dict(hs=(1.0, 3.0), U=(0.0, 0.43), nf=(0.03, 0.08), bw=(0.0, 0.3)),
    'spread': dict(hs=(1.5, 3.0), U=(0.0, 1.5),  nf=(0.02, 0.09), bw=(0.0, 0.2)),
    'extrap': dict(hs=(4.0, 5.0), U=(1.5, 1.5),  nf=(0.04, 0.08), bw=(0.3, 0.5)),
}


# ==========================================================================
#  shared numerics
# ==========================================================================
def bank_atoms(kp):
    """Free + bound-wave bank columns exactly as net_gamma_mmse builds them."""
    kxb = KXN_NP * kp
    kyb = KYN_NP * kp
    om = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
    kxb2 = np.concatenate([kxb, 2.0 * kxb])
    kyb2 = np.concatenate([kyb, 2.0 * kyb])
    om2 = np.concatenate([om, 2.0 * om])
    return kxb2, kyb2, om2


def bank_power_np(P, r, kp, chunk=1024):
    """Beamform power of series r at ALL bank atoms (free + bw), max-norm."""
    kxb, kyb, om = bank_atoms(kp)
    B = len(kxb)
    pw = np.empty(B)
    for i in range(0, B, chunk):
        psi = (kxb[i:i + chunk, None] * P[None, :, 0]
               + kyb[i:i + chunk, None] * P[None, :, 1]
               - om[i:i + chunk, None] * P[None, :, 2])
        pw[i:i + chunk] = (r[None] * np.cos(psi)).sum(1) ** 2 \
            + (r[None] * np.sin(psi)).sum(1) ** 2
    return (pw / max(pw.max(), 1e-12)).astype(np.float32)


def maps6(P, r, kp):
    """The shipped 6-channel normalized beamform-map recipe for series r."""
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], r[m1], DEPTH, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], r[m2], DEPTH, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    return np.stack([p1, p2, pm,
                     np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                     np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)


def two_stage_kp(P, z):
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=3.5 * kp, n=120)
    return max(np.hypot(bx, by), 1e-3)


def design_t(kxb, kyb, om, Q):
    psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] - om[None] * Q[:, 2:3]
    return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)


# ==========================================================================
#  stage 1 = frozen incumbent (l1s2bw + SmultHead + bound-wave arm)
# ==========================================================================
_CUR = {}


def load_incumbent():
    if 'net' not in _CUR:
        ck = torch.load(INCUMBENT, map_location=DEVICE)
        net = PriorNet().to(DEVICE)
        net.load_state_dict(ck['model'])
        net.eval()
        head = SmultHead().to(DEVICE)
        head.load_state_dict(ck['head'])
        head.eval()
        _CUR['net'], _CUR['head'] = net, head
    return _CUR['net'], _CUR['head']


def incumbent_field(P, z, Pq, kp, sigma0):
    net, head = load_incumbent()
    return net_gamma_mmse(net, P, z, DEPTH, Pq, sigma0, seed=0, kp_hint=kp,
                          smult_head=head, bound_wave=True)


class MeanNet(nn.Module):
    """diagnostic ensemble: mean log-gamma over seed checkpoints."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        return torch.stack([n(x) for n in self.nets]).mean(0)


_ENS = {}


def load_ensemble():
    if 'net' not in _ENS:
        nets = []
        for p in SEED_CKPTS:
            if not os.path.exists(p):
                continue
            ck = torch.load(p, map_location=DEVICE)
            n = PriorNet().to(DEVICE)
            n.load_state_dict(ck['model'])
            n.eval()
            nets.append(n)
        _ENS['net'] = MeanNet(nets).eval() if len(nets) >= 2 else None
        _ENS['k'] = len(nets)
    return _ENS['net']


# ==========================================================================
#  GATE-0: diagonal-gamma class ceiling (oracle deposit through same solver)
# ==========================================================================
def deposit_grid(kxs, kys, en, kp):
    """Bilinear deposit of per-mode energies onto the 51x51 normalized grid."""
    grid = np.zeros((NGRID, NGRID))
    sc = (NGRID - 1) / (2 * KREL)
    for kx, ky, e in zip(kxs, kys, en):
        u = (kx / kp + KREL) * sc                # column (x) coordinate
        v = (ky / kp + KREL) * sc                # row (y) coordinate
        if not (0 <= u <= NGRID - 1 and 0 <= v <= NGRID - 1):
            continue
        i0, j0 = int(v), int(u)
        fv, fu = v - i0, u - j0
        for di, wv in ((0, 1 - fv), (1, fv)):
            for dj, wu in ((0, 1 - fu), (1, fu)):
                i, j = i0 + di, j0 + dj
                if i < NGRID and j < NGRID:
                    grid[i, j] += e * wv * wu
    return grid


def oracle_field(S, blurs=(0.0, 1.0), smults=(0.5, 1.0, 3.0, 10.0)):
    """Best-per-scene oracle-gamma MMSE = generous ceiling of the
    diagonal-gamma-on-this-bank class (true realized energies, GT-selected
    blur/smult). Returns (best corr, chosen blur, chosen smult)."""
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    g_free_raw = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp)
    if S['bw'] is not None:
        b = S['bw']
        # bw atom i sits at (2*kx_i, 2*ky_i): deposit at the PARENT position
        g_bw_raw = deposit_grid(b['kx'] / 2, b['ky'] / 2,
                                (S['bw_sc'] * b['A']) ** 2 / 2, kp)
    else:
        g_bw_raw = np.zeros_like(g_free_raw)
    kxb, kyb, om = bank_atoms(kp)
    kxb_t = torch.tensor(kxb, dtype=torch.float32, device=DEVICE)
    kyb_t = torch.tensor(kyb, dtype=torch.float32, device=DEVICE)
    om_t = torch.tensor(om, dtype=torch.float32, device=DEVICE)
    Pt = torch.tensor(P, dtype=torch.float32, device=DEVICE)
    zt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
    Qt = torch.tensor(Pq, dtype=torch.float32, device=DEVICE)
    Phi = design_t(kxb_t, kyb_t, om_t, Pt)
    Phi_q = design_t(kxb_t, kyb_t, om_t, Qt)
    n = len(z)
    eye = torch.eye(n, device=DEVICE)
    best = (-2.0, None, None)
    for blur in blurs:
        gf = gaussian_filter(g_free_raw, blur) if blur > 0 else g_free_raw
        gb = gaussian_filter(g_bw_raw, blur) if blur > 0 else g_bw_raw
        g = np.concatenate([gf[_MASK], gb[_MASK]])
        if g.sum() <= 0:
            continue
        g = g / g.sum()
        gam = np.concatenate([g, g]) * float(np.var(z))
        gam_t = torch.tensor(gam, dtype=torch.float32, device=DEVICE)
        for smult in smults:
            s2 = (smult * s0) ** 2
            K = (Phi * gam_t[None]) @ Phi.T + s2 * eye
            w = Phi.T @ torch.linalg.solve(K, zt)
            fq = ((Phi_q * gam_t[None]) @ w).cpu().numpy()
            c = corr(fq, gt)
            if c > best[0]:
                best = (c, blur, smult)
    return best


# ==========================================================================
#  held-out sea builder — VERBATIM rng replication of e28_gennet_A.build_sea
#  (same seas as the E-28 A eval), extended to return the true modes; the
#  slow oracle_ls2 'det' is skipped (rng-neutral: it consumed no rng draws).
# ==========================================================================
def build_sea_c(regime, si):
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
    bw_lvl = rng.uniform(*R['bw'])
    bw, bw_sc = None, 0.0
    if bw_lvl > 1e-4:
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
        bkx, bky = 2 * kx[top], 2 * ky[top]
        bom, bph = 2 * om[top], 2 * ph[top]
        bA = 0.5 * np.hypot(kx[top], ky[top]) * Am[top] ** 2
        psib = (bkx[None] * P[:, 0:1] + bky[None] * P[:, 1:2]
                - bom[None] * P[:, 2:3] + bph[None])
        zb = (bA[None] * np.cos(psib)).sum(1)
        bw_sc = (np.sqrt(bw_lvl) * (Hs / 4)) / max(np.std(zb), 1e-9)
        z = z + bw_sc * zb
        psib_q = (bkx[None] * Xs[:, None] + bky[None] * Ys[:, None]
                  - bom[None] * T + bph[None])
        gt_u = gt_u + bw_sc * (bA[None] * np.cos(psib_q)).sum(1)
        bw = dict(kx=bkx, ky=bky, om=bom, ph=bph, A=bA)
    kp = two_stage_kp(P, z)
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, kp=kp, sigma0=float(nfrac * Hs),
                Hs=float(Hs), m_kx=kx, m_ky=ky, m_A=Am, m_ph=ph,
                bw=bw, bw_sc=float(bw_sc))


# ==========================================================================
#  BoostNet (stage 2)
# ==========================================================================
class PriorNet12(PriorNet):
    """PriorNet with a 12-channel stem; first 6 input channels warm-started
    from the incumbent, the residual-map 6 zero-initialized (=> at init the
    backbone response equals the incumbent's response to the original maps)."""

    def __init__(self, ch=48):
        super().__init__(ch=ch)
        old = self.e1[0]
        new = nn.Conv2d(12, ch, 3, padding=1)
        with torch.no_grad():
            new.weight.zero_()
            new.bias.zero_()
        self.e1[0] = new

    def warm_start(self, sd):
        """Copy incumbent weights; stem first-6-channel slice from incumbent."""
        own = self.state_dict()
        for k, v in sd.items():
            if k == 'e1.0.weight':
                own[k][:, :6].copy_(v)
                own[k][:, 6:].zero_()
            else:
                own[k].copy_(v)
        self.load_state_dict(own)


class BetaHead(nn.Module):
    """pooled 12ch stats + 2 residual-magnitude scalars -> beta. Final layer
    zero-init => beta == 0 exactly at init (exact incumbent)."""

    def __init__(self, c=12, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * c + 2, h), nn.SiLU(),
                               nn.Linear(h, 1))
        with torch.no_grad():
            self.f[-1].weight.zero_()
            self.f[-1].bias.zero_()

    def forward(self, X, extras):                # X (B,12,51,51) extras (B,2)
        p = torch.cat([X.mean(dim=(2, 3)), X.std(dim=(2, 3)), extras], 1)
        return self.f(p).squeeze(-1)


class BoostNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PriorNet12()
        self.head = BetaHead()
        self.log_smult2 = nn.Parameter(torch.tensor(float(np.log(3.0))))

    def zero_output(self):
        with torch.no_grad():
            self.backbone.out.weight.zero_()
            self.backbone.out.bias.zero_()

    def forward(self, X12, extras):
        r2 = self.backbone(X12)                  # (B,51,51) log10 modulation
        beta = self.head(X12, extras)            # (B,)
        return r2, beta


def boost_f2(r2_map, ls2, P, e, Q, kp, sigma0, pw_res, clampr=3.0):
    """Stage-2 MMSE on the residual e with prior
    g_free = pw_res_free * 10^r2 ; g_bw = BW_SCALE * pw_res_bw * 10^r2."""
    kxb = KXN * kp
    kyb = KYN * kp
    k = torch.sqrt(kxb ** 2 + kyb ** 2 + 1e-12)
    om = torch.sqrt(G * k * torch.tanh(k * DEPTH) + 1e-12)
    kxb2 = torch.cat([kxb, 2.0 * kxb])
    kyb2 = torch.cat([kyb, 2.0 * kyb])
    om2 = torch.cat([om, 2.0 * om])
    mod = torch.pow(10.0, torch.clamp(r2_map[MASK_T], -clampr, clampr))
    g = torch.cat([pw_res[:NB] * mod, BW_SCALE * pw_res[NB:] * mod])
    g = g / g.sum().clamp_min(1e-30)
    gam = torch.cat([g, g]) * e.var()
    Phi = design_t(kxb2, kyb2, om2, P)
    s2 = (torch.exp(ls2) * sigma0) ** 2
    n = len(e)
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(n, device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, e)
    return (design_t(kxb2, kyb2, om2, Q) * gam[None]) @ w


def make_stage2_inputs(P, z, e, kp, sigma0):
    X12 = np.concatenate([maps6(P, z, kp), maps6(P, e, kp)], 0)
    pw_res = bank_power_np(P, e, kp)
    extras = np.array([np.log10(np.std(e) / (sigma0 + 1e-9) + 1e-9),
                       np.log10(np.std(e) / (np.std(z) + 1e-9) + 1e-9)],
                      np.float32)
    return X12, pw_res, extras


def boost_field_eval(bn, P, z, Q, kp, sigma0):
    """Inference: frozen stage-1 incumbent + learned stage-2 boost."""
    n = len(z)
    f1 = incumbent_field(P, z, np.vstack([P, Q]), kp, sigma0)
    e = (z - f1[:n]).astype(np.float32)
    X12, pw_res, extras = make_stage2_inputs(P, z, e, kp, sigma0)
    with torch.no_grad():
        Xt = torch.tensor(X12[None], device=DEVICE)
        ex = torch.tensor(extras[None], device=DEVICE)
        r2, beta = bn(Xt, ex)
        f2 = boost_f2(r2[0], bn.log_smult2,
                      torch.tensor(P, dtype=torch.float32, device=DEVICE),
                      torch.tensor(e, device=DEVICE),
                      torch.tensor(Q, dtype=torch.float32, device=DEVICE),
                      kp, sigma0, torch.tensor(pw_res, device=DEVICE))
        out = f1[n:] + float(beta[0]) * f2.cpu().numpy()
    return out, float(beta[0])


# ==========================================================================
#  training data (train regime, mirrors e28_gennet_A.gen_sample + stage 1)
# ==========================================================================
def gen_sample(rng, regime='train'):
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
    kxm = np.array([x.kx for x in c]); kym = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); phm = np.array([x.phi for x in c])
    k = np.hypot(kxm, kym)
    om = np.sqrt(G * k * np.tanh(k * DEPTH) + 1e-12)
    uc = rng.uniform(0.0, 0.4)
    ua = rng.uniform(0, 2 * np.pi)
    om = om + kxm * uc * np.cos(ua) + kym * uc * np.sin(ua)
    bw_lvl = rng.uniform(*R['bw'])
    bw = None
    if bw_lvl > 1e-4:
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
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
    ucp = rng.uniform(*R['U'])
    P = sample_obs('fixed' if ucp < 1e-6 else 'moving', ucp * cp, lam, cp, T,
                   gx, gy, r_fp, rng)
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
    kp = two_stage_kp(P, z)
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
    Q = np.column_stack([qx, qy, tq])

    # frozen stage 1 (incumbent) — precomputed once per sample
    sigma0 = float(nfrac * Hs)
    f1 = incumbent_field(P, z, np.vstack([P, Q]), kp, sigma0)
    e = (z - f1[:N_OBS]).astype(np.float32)
    X12, pw_res, extras = make_stage2_inputs(P, z, e, kp, sigma0)
    return dict(X12=X12, P=P.astype(np.float32), e=e,
                Q=Q.astype(np.float32), gt_q=gt_q.astype(np.float32),
                f1_q=f1[N_OBS:].astype(np.float32), pw_res=pw_res,
                extras=extras, kp=float(kp), Hs=float(Hs), sigma0=sigma0)


def gen_batch(n, seed, regime='train', verbose=True):
    rng = np.random.default_rng(seed)
    out = []
    t0 = time.time()
    for i in range(n):
        out.append(gen_sample(rng, regime))
        if verbose and (i + 1) % max(1, n // 8) == 0:
            print(f"  gen {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return out


# ==========================================================================
#  training
# ==========================================================================
def train(steps, n_train, bs, lr, seed, ckpt, cache=CACHE):
    torch.manual_seed(seed)
    t0 = time.time()
    if cache and os.path.exists(cache):
        data = list(np.load(cache, allow_pickle=True))
        print(f"loaded {len(data)} cached train samples from {cache}")
        data = data[:n_train]
        if len(data) < n_train:
            data += gen_batch(n_train - len(data), seed + 999)
            np.save(cache, np.array(data, dtype=object))
    else:
        print(f"generating {n_train} train samples (incl. frozen stage 1)...",
              flush=True)
        data = gen_batch(n_train, seed)
        if cache:
            np.save(cache, np.array(data, dtype=object))
    print(f"  data ready ({time.time()-t0:.0f}s)", flush=True)

    bn = BoostNet().to(DEVICE)
    ck = torch.load(INCUMBENT, map_location=DEVICE)
    bn.backbone.warm_start(ck['model'])
    bn.zero_output()
    opt = torch.optim.AdamW(
        [{'params': bn.backbone.parameters(), 'lr': lr},
         {'params': bn.head.parameters(), 'lr': 1e-3},
         {'params': [bn.log_smult2], 'lr': 3e-3}], weight_decay=1e-5)

    n_val = max(24, n_train // 10)
    vi = np.arange(n_val)
    ti = np.arange(n_val, n_train)
    rng = np.random.default_rng(seed)

    def one(d):
        P = torch.tensor(d['P'], device=DEVICE)
        e = torch.tensor(d['e'], device=DEVICE)
        Q = torch.tensor(d['Q'], device=DEVICE)
        gt = torch.tensor(d['gt_q'], device=DEVICE)
        f1q = torch.tensor(d['f1_q'], device=DEVICE)
        X = torch.tensor(d['X12'][None], device=DEVICE)
        ex = torch.tensor(d['extras'][None], device=DEVICE)
        pw = torch.tensor(d['pw_res'], device=DEVICE)
        r2, beta = bn(X, ex)
        f2 = boost_f2(r2[0], bn.log_smult2, P, e, Q, d['kp'], d['sigma0'], pw)
        pred = f1q + beta[0] * f2
        task = ((pred - gt) ** 2).mean() / (d['Hs'] / 4) ** 2
        return task, float(beta[0].detach())

    def val():
        bn.eval()
        with torch.no_grad():
            ts, bts = [], []
            for i in vi[:96]:
                t, b = one(data[int(i)])
                ts.append(float(t)); bts.append(b)
        bn.train()
        return float(np.mean(ts)), float(np.mean(bts)), float(np.std(bts))

    v0, b0, bs0 = val()   # == pure incumbent val loss (beta==0 at init)
    print(f"step     0 val {v0:.4f} (=incumbent baseline)  "
          f"beta {b0:.3f}+-{bs0:.3f}", flush=True)
    best = (v0, 0)
    torch.save(dict(model=bn.state_dict(), step=0, val=v0, val0=v0), ckpt)
    log_every = max(10, steps // 40)
    for step in range(1, steps + 1):
        ii = rng.choice(ti, bs, replace=False)
        loss = 0.0
        for i in ii:
            t, _ = one(data[int(i)])
            loss = loss + t
        loss = loss / bs
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bn.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            vl, bm, bstd = val()
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                torch.save(dict(model=bn.state_dict(), step=step, val=vl,
                                val0=v0), ckpt)
                tag = ' *saved'
            print(f"step {step:5d} [{time.time()-t0:5.0f}s] "
                  f"train {float(loss.detach()):.4f} val {vl:.4f} "
                  f"beta {bm:+.3f}+-{bstd:.3f} "
                  f"smult2 {float(torch.exp(bn.log_smult2)):.2f}{tag}",
                  flush=True)
    print(f"best val {best[0]:.4f} @ step {best[1]} (incumbent-baseline "
          f"{v0:.4f}; improvement {v0-best[0]:+.4f})")
    bn.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                  weights_only=False)['model'])
    bn.eval()
    return bn, dict(val0=v0, best_val=best[0], best_step=best[1])


# ==========================================================================
#  GATE-0 + LORO evaluation
# ==========================================================================
def score_sea(S, si, bn=None, with_ens=True):
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    row = dict(Hs=S['Hs'])
    f_cl = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=s0, kp_hint=kp, seed=si)
    row['cl'] = corr(f_cl, gt)
    f_cur = incumbent_field(P, z, Pq, kp, s0)
    row['cur'] = corr(f_cur, gt)
    oc, ob, osm = oracle_field(S)
    row['oracle'], row['o_blur'], row['o_smult'] = oc, ob, osm
    row['head'] = oc - row['cur']
    if with_ens:
        ens = load_ensemble()
        if ens is not None:
            _, hd = load_incumbent()
            f_e = net_gamma_mmse(ens, P, z, DEPTH, Pq, s0, seed=0, kp_hint=kp,
                                 smult_head=hd, bound_wave=True)
            row['ens'] = corr(f_e, gt)
    if bn is not None:
        f_b, beta = boost_field_eval(bn, P, z, Pq, kp, s0)
        row['boost'] = corr(f_b, gt)
        row['beta'] = beta
    return row


def gate0(n_seas, with_ens=True):
    print(f"\n===== GATE-0: diagonal-gamma class ceiling ({n_seas} spread "
          f"seas) =====", flush=True)
    e28 = None
    if os.path.exists(E28_JSON):
        e28 = json.load(open(E28_JSON))['pooled']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        r = score_sea(S, si, bn=None, with_ens=with_ens)
        rows.append(r)
        rep = ''
        if e28 is not None and si < len(e28):
            dcl = abs(r['cl'] - e28[si]['cl'])
            dcu = abs(r['cur'] - e28[si]['cur'])
            rep = (f" | e28-repro d(cl) {dcl:.1e} d(cur) {dcu:.1e}"
                   f" {'OK' if max(dcl, dcu) < 1e-6 else '**MISMATCH**'}")
        print(f"  sea {si:2d}: cl {r['cl']:+.3f} cur {r['cur']:+.3f} "
              f"ens {r.get('ens', float('nan')):+.3f} "
              f"oracle {r['oracle']:+.3f} (blur {r['o_blur']} smult "
              f"{r['o_smult']}) head {r['head']:+.3f}{rep}", flush=True)
    H = float(np.mean([r['head'] for r in rows]))
    Hmed = float(np.median([r['head'] for r in rows]))
    d_ens = (float(np.mean([r['ens'] - r['cur'] for r in rows]))
             if 'ens' in rows[0] else float('nan'))
    print(f"\nGATE-0 headroom H = mean(oracle - cur) = {H:+.4f} "
          f"(median {Hmed:+.4f})")
    print(f"        ens - cur = {d_ens:+.4f} (diagnostic arm)")
    verdict = 'PROCEED (H >= 0.03)' if H >= 0.03 else \
        'CLASS SATURATED (H < 0.03) -> training aborted per pre-registration'
    print(f"GATE-0 VERDICT: {verdict}")
    out = dict(rows=rows, H=H, H_median=Hmed, d_ens=d_ens, n_seas=n_seas,
               proceed=bool(H >= 0.03))
    json.dump(out, open(OUT_GATE0, 'w'), indent=1, default=float)
    print(f"saved {OUT_GATE0}")
    return out


def loro_eval(bn, n_spread, n_extrap, gate0_rows=None):
    print(f"\n===== LORO eval: {n_spread} spread + {n_extrap} extrap =====",
          flush=True)
    pooled, extrap = [], []
    for si in range(n_spread):
        S = build_sea_c('spread', si)
        r = score_sea(S, si, bn=bn)
        pooled.append(r)
        print(f"  spread {si:2d}: cl {r['cl']:+.3f} cur {r['cur']:+.3f} "
              f"ens {r.get('ens', float('nan')):+.3f} boost {r['boost']:+.3f} "
              f"oracle {r['oracle']:+.3f} beta {r['beta']:+.3f}", flush=True)
    for si in range(n_extrap):
        S = build_sea_c('extrap', si)
        r = score_sea(S, si, bn=bn)
        extrap.append(r)
        print(f"  extrap {si:2d}: cl {r['cl']:+.3f} cur {r['cur']:+.3f} "
              f"ens {r.get('ens', float('nan')):+.3f} boost {r['boost']:+.3f} "
              f"oracle {r['oracle']:+.3f} beta {r['beta']:+.3f}", flush=True)

    def M(rows, k):
        return float(np.nanmean([r[k] for r in rows]))
    d_pool = M(pooled, 'boost') - M(pooled, 'cur')
    wins = float(np.mean([r['boost'] >= r['cur'] for r in pooled]))
    d_ex = M(extrap, 'boost') - M(extrap, 'cur')
    d_ens = (M(pooled, 'ens') - M(pooled, 'cur')
             if 'ens' in pooled[0] else float('nan'))
    gains = np.array([r['boost'] - r['cur'] for r in pooled])
    heads = np.array([r['head'] for r in pooled])
    if np.std(gains) > 1e-9 and np.std(heads) > 1e-9:
        from scipy.stats import spearmanr
        rho_mech = float(spearmanr(gains, heads).statistic)
    else:
        rho_mech = float('nan')
    passes = dict(
        a_do_no_harm=bool(d_pool >= -0.005),
        b_win=bool(d_pool >= 0.02 and wins >= 0.6),
        c_ood=bool(d_ex >= 0.0),
        d_mechanism=bool(rho_mech > 0),
    )
    print("\n----- pre-registered criteria (design C) -----")
    print(f"(a) do-no-harm  boost-cur pooled = {d_pool:+.4f} >= -0.005 ? "
          f"{passes['a_do_no_harm']}")
    print(f"(b) WIN         {d_pool:+.4f} >= +0.02 & win {wins:.0%} >= 60% ? "
          f"{passes['b_win']}   <-- C stands or falls here")
    print(f"(c) OOD extrap  boost-cur = {d_ex:+.4f} >= 0 ? {passes['c_ood']}")
    print(f"(d) mechanism   Spearman(gain, headroom) = {rho_mech:+.3f} > 0 ? "
          f"{passes['d_mechanism']}")
    print(f"    [diag] ens-cur pooled = {d_ens:+.4f}")
    verdict = passes['b_win'] and passes['a_do_no_harm'] and passes['c_ood']
    print(f"\nC VERDICT: {'PASS' if verdict else 'FAIL'}")
    out = dict(pooled=pooled, extrap=extrap, passes=passes,
               verdict=bool(verdict), d_pool=d_pool, wins=wins, d_extrap=d_ex,
               d_ens=d_ens, rho_mechanism=rho_mech, n_spread=n_spread,
               n_extrap=n_extrap)
    json.dump(out, open(OUT_LORO, 'w'), indent=1, default=float)
    print(f"saved {OUT_LORO}")
    return out


# ==========================================================================
#  selftest
# ==========================================================================
def selftest():
    print("=== e29_gennet_C selftest ===")
    ok = True

    # [1] arena identity: build_sea_c reproduces e28_gennet_A.build_sea
    import e28_gennet_A as A
    Sa = A.build_sea('spread', 0)
    Sc = build_sea_c('spread', 0)
    z_ok = np.allclose(Sa['z'], Sc['z'])
    g_ok = np.allclose(Sa['gt_u'], Sc['gt_u'])
    k_ok = abs(Sa['kp'] - Sc['kp']) < 1e-12
    print(f"[1] arena identity vs e28_A.build_sea: z {z_ok} gt {g_ok} "
          f"kp {k_ok}")
    ok &= z_ok and g_ok and k_ok

    # [2] zero-init exactness: boost == incumbent (beta = 0)
    rng = np.random.default_rng(11)
    d = gen_sample(rng, 'train')
    bn = BoostNet().to(DEVICE)
    ck = torch.load(INCUMBENT, map_location=DEVICE)
    bn.backbone.warm_start(ck['model'])
    bn.zero_output()
    bn.eval()
    with torch.no_grad():
        Xt = torch.tensor(d['X12'][None], device=DEVICE)
        ex = torch.tensor(d['extras'][None], device=DEVICE)
        r2, beta = bn(Xt, ex)
    f2 = boost_f2(r2[0], bn.log_smult2,
                  torch.tensor(d['P'], device=DEVICE),
                  torch.tensor(d['e'], device=DEVICE),
                  torch.tensor(d['Q'], device=DEVICE),
                  d['kp'], d['sigma0'],
                  torch.tensor(d['pw_res'], device=DEVICE))
    pred = torch.tensor(d['f1_q'], device=DEVICE) + beta[0] * f2
    dev = float((pred - torch.tensor(d['f1_q'], device=DEVICE)).abs().max())
    print(f"[2] zero-init |boost - incumbent|_max = {dev:.2e} beta "
          f"{float(beta[0]):.2e} r2max {float(r2.abs().max()):.2e} "
          f"(<1e-6 ? {dev < 1e-6})")
    ok &= dev < 1e-6

    # [3] staged gradient unblocking: beta head first, then r2 after beta!=0
    bn.train()
    bn.zero_grad()
    r2, beta = bn(Xt, ex)
    f2 = boost_f2(r2[0], bn.log_smult2,
                  torch.tensor(d['P'], device=DEVICE),
                  torch.tensor(d['e'], device=DEVICE),
                  torch.tensor(d['Q'], device=DEVICE),
                  d['kp'], d['sigma0'],
                  torch.tensor(d['pw_res'], device=DEVICE))
    gt = torch.tensor(d['gt_q'], device=DEVICE)
    f1q = torch.tensor(d['f1_q'], device=DEVICE)
    ((f1q + beta[0] * f2 - gt) ** 2).mean().backward()
    g_beta = float(bn.head.f[-1].weight.grad.abs().sum())
    print(f"[3a] beta-head grad at init = {g_beta:.3e} (>0 ? {g_beta > 0})")
    ok &= g_beta > 0
    with torch.no_grad():                       # force beta ~ 0.3, re-check r2
        bn.head.f[-1].bias.fill_(0.3)
    bn.zero_grad()
    r2, beta = bn(Xt, ex)
    f2 = boost_f2(r2[0], bn.log_smult2,
                  torch.tensor(d['P'], device=DEVICE),
                  torch.tensor(d['e'], device=DEVICE),
                  torch.tensor(d['Q'], device=DEVICE),
                  d['kp'], d['sigma0'],
                  torch.tensor(d['pw_res'], device=DEVICE))
    ((f1q + beta[0] * f2 - gt) ** 2).mean().backward()
    g_out = float(bn.backbone.out.weight.grad.abs().sum())
    print(f"[3b] r2 out-conv grad once beta!=0 = {g_out:.3e} "
          f"(>0 ? {g_out > 0})")
    ok &= g_out > 0

    # [4] oracle sanity on one sea (should sit at/above the incumbent)
    r = score_sea(Sc, 0, bn=None, with_ens=False)
    print(f"[4] sea0: cl {r['cl']:+.3f} cur {r['cur']:+.3f} oracle "
          f"{r['oracle']:+.3f} (blur {r['o_blur']} smult {r['o_smult']}) "
          f"headroom {r['head']:+.3f}")
    ok &= np.isfinite(r['oracle'])

    # [5] input determinism
    a1 = make_stage2_inputs(d['P'].astype(float), d['e'] + 0.0, d['e'],
                            d['kp'], d['sigma0'])
    a2 = make_stage2_inputs(d['P'].astype(float), d['e'] + 0.0, d['e'],
                            d['kp'], d['sigma0'])
    det_ok = all(np.allclose(x, y) for x, y in zip(a1, a2))
    print(f"[5] make_stage2_inputs deterministic ? {det_ok}")
    ok &= det_ok

    print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gate0', action='store_true',
                    help='run only the class-ceiling measurement')
    ap.add_argument('--gate0_seas', type=int, default=16)
    ap.add_argument('--skip_gate0', action='store_true')
    ap.add_argument('--force_train', action='store_true',
                    help='train even if gate-0 says the class is saturated '
                         '(exploratory, labeled as such)')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--n_train', type=int, default=2500)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--eval_seas', type=int, default=18)
    ap.add_argument('--eval_extrap', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--cache', default=CACHE)
    a = ap.parse_args()
    print(f"device {DEVICE}  bank atoms {NB} free + {NB} bw", flush=True)

    if a.selftest:
        return 0 if selftest() else 1

    if a.smoke:
        a.gate0_seas = min(a.gate0_seas, 6)
        a.steps = a.steps if a.steps != 4000 else 150
        a.n_train = a.n_train if a.n_train != 2500 else 80
        a.bs = 6
        a.eval_seas = min(a.eval_seas, 6)
        a.eval_extrap = min(a.eval_extrap, 3)
        a.cache = 'tmp/e29_smoke_cache.npy'

    g0 = None
    if a.gate0:
        gate0(a.gate0_seas)
        return 0
    if not a.skip_gate0:
        g0 = gate0(a.gate0_seas)
        if not g0['proceed'] and not a.force_train and not a.smoke:
            print("\nPre-registered abort: diagonal-gamma class saturated; "
                  "no training. (--force_train to override.)")
            return 0
        if not g0['proceed'] and a.smoke:
            print("\n[smoke] gate-0 says saturated; continuing anyway for "
                  "mechanism telemetry (smoke is exploratory by definition).")

    bn, tinfo = train(a.steps, a.n_train, a.bs, a.lr, a.seed, a.ckpt, a.cache)
    loro_eval(bn, a.eval_seas, a.eval_extrap,
              gate0_rows=g0['rows'] if g0 else None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
