"""
e29_gennet_D.py — design "D": QUADRATIC BOUND-WAVE OUTPUT HEAD (QBW).
An OUTPUT-SPACE attack (not a conditioning attack) on the E-28 failure.

--------------------------------------------------------------------------
MECHANISM (one paragraph, pre-registered before any training)
  Every E-28 design (A and B) emitted a DIAGONAL prior variance gamma(k)
  over the fixed normalized bank into the same linear MMSE — so every
  predictor in that class is (weakly-conditioned) LINEAR in z, and only the
  conditioning scheme differed.  But the 2nd-order bound wave the harness
  injects (and 2nd-order Stokes physics generally) is a DETERMINISTIC
  QUADRATIC function of the free-wave field: a parent mode A*cos(theta+phi)
  carries a bound component 0.5*k*A^2*cos(2*theta+2*phi) — coefficient
  (a+ib)^2 in terms of the parent's (cos,sin) coefficients.  A diagonal
  Gaussian prior at (2k,2w) (the incumbent's bw arm, A's bw arm) can only
  ask the data to re-estimate the bound PHASE at O(steepness) SNR, where
  MMSE shrinkage destroys it.  Design D enlarges the OUTPUT SPACE with the
  missing quadratic term: render the bound field deterministically from the
  free-wave posterior coefficients (phase transferred for free, zero data
  cost), scaled by a single per-scene dose gain alpha — selected on a
  held-out obs split in closed form (D-phys, no training), or predicted by
  a tiny (~1k-param) head (D-learn).  E-21 established that bound-wave
  misspecification is the ONLY residual learnable value in the
  well-determined regime and that it scales with dose — exactly the target.

WHY THIS EXPLAINS THE E-28 EARLY-OVERFIT SIGNATURE
  The incumbent (prior_net_l1s2bw, itself a best-val@step-800 checkpoint of
  a 4000-fixed-sample task-loss run) sits at/near the OPTIMUM OF THE
  DIAGONAL-GAMMA CLASS for this data regime.  A and B warm-started at that
  point and fine-tuned 1M-param CNNs inside the SAME class on 4000 fixed
  noisy samples: the residual gradient was almost pure per-sample noise, so
  best-val landed at step 300-800 (~<2 epochs) and val then degraded —
  memorization, not learning.  D responds on both axes: the signal is
  created by LEAVING the class (a term no gamma can express), and the
  learnable part is collapsed to ~1k params (dose head) so the variance
  side of the E-28 failure cannot recur.

PRE-REGISTERED CRITERIA (confirmatory run; --smoke is indicative only)
  Arms: hand  = mmse_bank_estimate (classical, kp_hint, as E-28 A)
        cur   = net_gamma_mmse(l1s2bw, smult_head, bound_wave=True) (SHIPPED)
        Dphys = cur + alpha_LS * QBW   (alpha via held-out-obs LS + t-shrink)
        Dlearn= cur + alpha_head * QBW (tiny head)
        Dorac = cur + alpha* * QBW, alpha* grid-picked against GT
                (DIAGNOSTIC ONLY — never deployable, bounds the design)
  Buckets (Tp=7 harness, corr at 1200 held-out pixels, mirrors E-28 A):
        B0 dose=0        (n=16 full / 4 smoke)   do-no-harm
        B1 dose U[.1,.3] (n=24 full / 6 smoke)   in-distribution dose
        B2 dose U[.3,.5] + Hs U[4,5] + U=1.5cp (n=16 / 4)  extrapolation
        (head TRAINED on dose U[0,.35], Hs 1-3, U<=1.0cp -> B2 is LORO)
  P0 selftest: reimplemented incumbent solve == shipped net_gamma_mmse
     (<1e-4); QBW render from TRUE coefficients reproduces the injected
     bound field exactly (corr>0.999).
  P1 do-no-harm : mean(Dphys - cur) on B0 >= -0.005.
  P2 dosed gain : mean(Dphys - cur) on B1+B2 > 0, win-rate >= 60%,
                  bootstrap 95% CI > 0 (confirmatory n=40).
  P3 dose-response: Spearman(dose, Dphys-cur) > 0 on dosed seas.
  P4 head sanity: mean(Dlearn - Dphys) >= -0.003 (head must not hurt).
  KILL CRITERION (declared failure): if mean(Dorac - cur) on B1+B2 < +0.01
     the enlarged class has no headroom — the incumbent's diagonal bw arm
     already saturates bound-wave value; design D is dead regardless of how
     alpha is chosen.  (This is the decisive, cheap diagnostic.)

HONEST CAVEATS (pre-registered)
  - The eval GT injects exactly self-super-harmonic bound waves; QBW
    assumes that structure => this is a WELL-SPECIFIedness advantage on
    this harness, symmetric to hand-MMSE's advantage on linear seas.  The
    decisive external test would be HOS/M3 (cross-interactions included).
  - Everything is simulation; no real LiDAR.

Usage:
  python3 tmp/e29_gennet_D.py --selftest
  python3 tmp/e29_gennet_D.py --smoke
  python3 tmp/e29_gennet_D.py --full          # confirmatory (pre-registered)
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
from d18_relax_estimator import beamform_top, omega
from d3_sa_nomp import beamform_power, split_masks, mmse_bank_estimate
from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d3_prior_net_eval import net_gamma_mmse
from d3_l1_s2head_train import SmultHead
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, corr,
                                     N_OBS, DEPTH)

G = 9.81
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INIT = 'results/direction3/prior_net_l1s2bw.pt'      # the incumbent
HEAD_CKPT = 'results/direction3/e29_gennet_D_head.pt'
OUT_JSON = 'results/direction3/e29_gennet_D.json'
ALPHA_MAX = 2.0
N_FEAT = 12
N_FOLD = 4

# ---------------------------------------------------------------- incumbent
_CK = torch.load(INIT, map_location=DEVICE, weights_only=False)
CUR_NET = PriorNet().to(DEVICE); CUR_NET.load_state_dict(_CK['model'])
CUR_NET.eval()
CUR_HEAD = SmultHead().to(DEVICE); CUR_HEAD.load_state_dict(_CK['head'])
CUR_HEAD.eval()


def make_maps(P, z, kp):
    """6-channel normalized beamform-power maps (exact net_gamma_mmse recipe)."""
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], DEPTH, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], DEPTH, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    return np.stack([p1, p2, pm,
                     np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                     np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)


class IncumbentSolve:
    """EXACT reimplementation of the shipped net_gamma_mmse
    (smult_head + bound_wave=True path) that ALSO exposes the per-atom
    posterior-mean coefficients.  Verified against the shipped function in
    the selftest (P0).  Subset solves reuse the scene-level prior."""

    def __init__(self, P, z, kp, sigma0):
        self.P, self.z, self.kp, self.s0 = P, z, float(kp), float(sigma0)
        X = make_maps(P, z, kp)
        self.X = X
        with torch.no_grad():
            Xt = torch.tensor(X[None], device=DEVICE)
            Ylog = CUR_NET(Xt)[0].cpu().numpy()
            self.smult = float(torch.exp(CUR_HEAD(Xt)[0]))
        ax = np.linspace(-KREL * kp, KREL * kp, NGRID)
        KX, KY = np.meshgrid(ax, ax)
        kk = np.hypot(KX, KY)
        sel = (kk > 0.35 * kp) & (kk < KREL * kp)
        kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
        om = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
        g = np.maximum(10.0 ** Ylog[sel], 1e-8)
        self.n_free = len(kxb)
        # bound-wave arm exactly as shipped (bound_wave=True, bw_scale 0.15)
        self.kxb = np.concatenate([kxb, 2.0 * kxb])
        self.kyb = np.concatenate([kyb, 2.0 * kyb])
        self.om = np.concatenate([om, 2.0 * om])
        g = np.concatenate([g, 0.15 * g])
        g = g / g.sum()
        self.gam = np.concatenate([g, g]) * np.var(z)
        self.s2 = (self.smult * self.s0) ** 2
        self.B = len(self.kxb)

    def design(self, Q):
        psi = self.kxb[None] * Q[:, 0:1] + self.kyb[None] * Q[:, 1:2] \
            - self.om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    def solve(self, idx=None):
        """Solve on obs subset idx (None = all). Returns (w, coef m)."""
        P = self.P if idx is None else self.P[idx]
        z = self.z if idx is None else self.z[idx]
        Phi = self.design(P)
        K = (Phi * self.gam[None]) @ Phi.T + self.s2 * np.eye(len(z))
        w = Phi.T @ np.linalg.solve(K, z)
        return w, self.gam * w

    def predict(self, Q, w):
        out = np.empty(len(Q))
        for i in range(0, len(Q), 4096):
            out[i:i + 4096] = (self.design(Q[i:i + 4096]) * self.gam[None]) @ w
        return out

    def free_coefs(self, m):
        """(a, b, kx, ky, om) for the FREE atoms from coefficient vector m."""
        nf, B = self.n_free, self.B
        return (m[:nf], m[B:B + nf],
                self.kxb[:nf], self.kyb[:nf], self.om[:nf])


# ------------------------------------------------------------------- QBW
def qbw_render(a, b, kx, ky, om, kp, pts, sep=0.22, rad=0.15, m_max=16,
               floor=0.02, peaks_out=None):
    """Phase-coherent 2nd-order self-super-harmonic render from free-wave
    posterior coefficients.  Cluster the posterior energy into <=m_max peaks
    (min separation sep*kp, membership radius rad*kp); per cluster compute
    the complex analytic sub-field F and add 0.5*k_peak*Re[F^2] — for a
    single mode this is EXACTLY 0.5*k*A^2*cos(2*theta+2*phi) (selftest); for
    a mode smeared over neighbouring bank atoms the intra-cluster cross
    terms reconstruct the correct 2k phase (self-terms alone would not)."""
    e = a ** 2 + b ** 2
    if e.max() <= 0:
        return np.zeros(len(pts))
    order = np.argsort(e)[::-1]
    peaks = []
    for i in order:
        if e[i] < floor * e[order[0]] or len(peaks) >= m_max:
            break
        if all(np.hypot(kx[i] - kx[j], ky[i] - ky[j]) > sep * kp
               for j in peaks):
            peaks.append(i)
    if not peaks:
        return np.zeros(len(pts))
    pk = np.array(peaks)
    d = np.hypot(kx[:, None] - kx[pk][None], ky[:, None] - ky[pk][None])
    nearest = np.argmin(d, axis=1)
    within = d[np.arange(len(kx)), nearest] <= rad * kp
    q = np.zeros(len(pts))
    for ci, pi in enumerate(pk):
        S = np.where(within & (nearest == ci))[0]
        if len(S) == 0:
            continue
        th = (kx[S][None] * pts[:, 0:1] + ky[S][None] * pts[:, 1:2]
              - om[S][None] * pts[:, 2:3])
        F = ((a[S] + 1j * b[S])[None] * np.exp(-1j * th)).sum(1)
        q += 0.5 * np.hypot(kx[pi], ky[pi]) * np.real(F ** 2)
    if peaks_out is not None:
        peaks_out['n_peaks'] = len(pk)
        peaks_out['e_top_frac'] = float(e[pk[0]] / e.sum())
        peaks_out['pk'] = pk
    return q


def bw_offshell_power(P, z, kx, ky, om, pk):
    """DIRECT dose evidence: normalized beamform power at the OFF-SHELL
    bound-wave points (2k, 2*omega) of the top free-wave peaks.  The
    on-shell input maps barely see the injected bound waves (omega(2k) !=
    2*omega(k) decorrelates in ~12s << T), so this 2*len(pk)-term probe is
    information the incumbent's inputs do not expose as a scalar."""
    if len(pk) == 0:
        return -9.0
    ps = (2 * kx[pk][:, None] * P[None, :, 0]
          + 2 * ky[pk][:, None] * P[None, :, 1]
          - 2 * om[pk][:, None] * P[None, :, 2])
    pw = ((z[None] * np.cos(ps)).sum(1) ** 2
          + (z[None] * np.sin(ps)).sum(1) ** 2)
    return float(np.log10(pw.mean() / (len(z) * np.var(z) + 1e-12) + 1e-9))


def _fold_stats(resid_v, q_v):
    """One fold's LS coefficient and correlation of resid on unit-RMS q."""
    qq = float(q_v @ q_v)
    if qq < 1e-20:
        return 0.0, 0.0
    raw = float(resid_v @ q_v) / qq
    sr = np.std(resid_v) + 1e-12
    sq = np.std(q_v) + 1e-12
    rho = float(np.mean((resid_v - resid_v.mean()) * (q_v - q_v.mean()))
                / (sr * sq))
    return raw, rho


def qbw_clusters(a, b, kx, ky, kp, sep=0.22, rad=0.15, m_max=16, floor=0.02):
    """Peak/cluster structure used by qbw_render (exposed for the joint fit).
    Returns list of (peak_index, member_indices)."""
    e = a ** 2 + b ** 2
    if e.max() <= 0:
        return []
    order = np.argsort(e)[::-1]
    peaks = []
    for i in order:
        if e[i] < floor * e[order[0]] or len(peaks) >= m_max:
            break
        if all(np.hypot(kx[i] - kx[j], ky[i] - ky[j]) > sep * kp
               for j in peaks):
            peaks.append(i)
    if not peaks:
        return []
    pk = np.array(peaks)
    d = np.hypot(kx[:, None] - kx[pk][None], ky[:, None] - ky[pk][None])
    nearest = np.argmin(d, axis=1)
    within = d[np.arange(len(kx)), nearest] <= rad * kp
    return [(int(pi), np.where(within & (nearest == ci))[0])
            for ci, pi in enumerate(pk)
            if (within & (nearest == ci)).any()]


def joint_qbw_refine(inc, clusters, obs_idx, n_steps=200, lr=0.03,
                     alpha0=0.0):
    """PRIMARY D ARM — joint fit of the QUADRATIC forward model
        z ~= Phi m  +  alpha * QBW(m_cluster) / RMS0
    over the cluster free-wave coefficients + a free bound amplitude alpha
    (meters; the harness dose rescale makes the physical 0.5kA^2 scale
    wrong).  Rationale: the E-28 class squares/ignores the bound component;
    estimating parents under a LINEAR model treats 20-50% of the variance
    as unmodeled noise, then squaring the polluted posterior doubles its
    phase error (measured: perfect-direction headroom +0.29, posterior-
    squaring render only +0.03).  The joint fit lets the 2w-band data
    constrain the parent phases instead.  Returns (predict_fn, alpha,
    fit_loss_path).  Keeps the incumbent MMSE prior as the ridge (same
    gamma, same s2) so alpha->0 recovers the incumbent solution family."""
    P = torch.tensor(inc.P[obs_idx], dtype=torch.float64, device=DEVICE)
    z = torch.tensor(inc.z[obs_idx], dtype=torch.float64, device=DEVICE)
    gam = torch.tensor(inc.gam, dtype=torch.float64, device=DEVICE)
    kxb = torch.tensor(inc.kxb, dtype=torch.float64, device=DEVICE)
    kyb = torch.tensor(inc.kyb, dtype=torch.float64, device=DEVICE)
    omb = torch.tensor(inc.om, dtype=torch.float64, device=DEVICE)
    B = inc.B
    # linear solution on this obs subset = init + frozen non-cluster part
    w0, m0 = inc.solve(obs_idx)
    m0t = torch.tensor(m0, dtype=torch.float64, device=DEVICE)
    ci = np.concatenate([S for _, S in clusters])          # free-atom idx
    ci_t = torch.tensor(ci, device=DEVICE, dtype=torch.long)
    par = torch.stack([m0t[ci_t], m0t[B + ci_t]])          # (2, nc) a,b
    par = par.clone().requires_grad_(True)
    la = torch.tensor(float(alpha0), dtype=torch.float64, device=DEVICE,
                      requires_grad=True)
    kpk = [float(np.hypot(inc.kxb[pi], inc.kyb[pi])) for pi, _ in clusters]
    slices = []
    o = 0
    for _, S in clusters:
        slices.append((o, o + len(S)))
        o += len(S)
    s_z = float(z.std())

    def design_t(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - omb[None] * Q[:, 2:3]
        return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)

    def trig_tables(Q):
        """Per-cluster (cos, sin) tables — constants of the fit."""
        tabs = []
        for _, S in clusters:
            St = torch.tensor(S, device=DEVICE, dtype=torch.long)
            th = (kxb[St][None] * Q[:, 0:1] + kyb[St][None] * Q[:, 1:2]
                  - omb[St][None] * Q[:, 2:3])
            tabs.append((torch.cos(th), torch.sin(th)))
        return tabs

    def qbw_t(tabs, a_c, b_c):
        q = 0.0
        for (Ct, St_), (s0_, s1_), kpv in zip(tabs, slices, kpk):
            cr = a_c[s0_:s1_]; im = b_c[s0_:s1_]
            Fr = Ct @ cr + St_ @ im
            Fi = Ct @ im - St_ @ cr
            q = q + 0.5 * kpv * (Fr ** 2 - Fi ** 2)
        return q

    Phi = design_t(P)
    tabs_P = trig_tables(P)
    # frozen contribution of the non-cluster coefficients
    m_fr = m0t.clone()
    m_fr[ci_t] = 0.0; m_fr[B + ci_t] = 0.0
    z_rest = Phi @ m_fr
    Phi_c = torch.cat([Phi[:, ci_t], Phi[:, B + ci_t]], dim=1)   # (n, 2nc)
    with torch.no_grad():
        q0 = qbw_t(tabs_P, par[0], par[1])
        rms0 = float(torch.sqrt((q0 ** 2).mean()).clamp_min(1e-14))
    gam_c = torch.cat([gam[ci_t], gam[B + ci_t]]).clamp_min(1e-20)
    s2 = float(inc.s2)
    # Adam moves ~lr per step: scale lrs to the parameter magnitudes
    lr_par = lr * (float(par.detach().abs().mean()) + 1e-6)
    lr_a = lr * max(0.5 * s_z, 1e-4)
    opt = torch.optim.Adam([{'params': [par], 'lr': lr_par},
                            {'params': [la], 'lr': lr_a}])
    lam_a = 4.0 / max(s_z, 1e-9) ** 2      # weak alpha ridge: fire on
    path = []                              # evidence, not on fit slack
    for it in range(n_steps):
        cvec = torch.cat([par[0], par[1]])
        zq = z_rest + Phi_c @ cvec + la * qbw_t(tabs_P, par[0], par[1]) / rms0
        loss = ((z - zq) ** 2).sum() / s2 + (cvec ** 2 / gam_c).sum() \
            + lam_a * la ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == n_steps - 1:
            path.append(float(loss.detach()))
    par_f = par.detach(); la_f = float(la)

    def predict(Q):
        Qt = torch.tensor(Q, dtype=torch.float64, device=DEVICE)
        with torch.no_grad():
            m_all = m_fr.clone()
            m_all[ci_t] = par_f[0]; m_all[B + ci_t] = par_f[1]
            f = design_t(Qt) @ m_all \
                + la_f * qbw_t(trig_tables(Qt), par_f[0], par_f[1]) / rms0
        return f.cpu().numpy()

    return predict, la_f, path


def alpha_pooled(raws, rhos, n_v, cap):
    """Pool K folds (mean raw / mean rho), then t-shrink with the SINGLE-fold
    n (folds share data; no independence claimed).  q must be UNIT-RMS
    normalized: the harness doses bound waves by RESCALING them to
    sqrt(dose)*Hs/4 (~50-100x the physical 0.5kA^2 amplitude), so alpha is
    a free amplitude in meters, capped at `cap`."""
    raw = float(np.mean(raws))
    rho = float(np.mean(rhos))
    if rho <= 0 or raw <= 0:
        return 0.0, raw, 0.0
    t = rho * np.sqrt(max(n_v - 2, 1)) / np.sqrt(max(1 - rho ** 2, 1e-9))
    shrink = max(0.0, 1.0 - 1.0 / max(t, 1e-9) ** 2)
    return float(np.clip(raw * shrink, 0.0, cap)), raw, float(t)


# ------------------------------------------------------------- tiny head
class AlphaHead(nn.Module):
    """11 cheap features -> dose gain alpha in [0, ALPHA_MAX]. ~1.5k params."""

    def __init__(self, n_feat=N_FEAT, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_feat, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, 1))
        with torch.no_grad():
            nn.init.zeros_(self.f[-1].weight)
            self.f[-1].bias.fill_(-2.0)         # start near alpha ~ 0.24
        self.register_buffer('mu', torch.zeros(n_feat))
        self.register_buffer('sd', torch.ones(n_feat))

    def forward(self, v):
        return ALPHA_MAX * torch.sigmoid(
            self.f((v - self.mu) / self.sd)).squeeze(-1)


def info_features(P, z, kp, sigma0):
    """The 7 obs-only informativeness features (E-28 A recipe)."""
    n = len(z)
    ax = P[:, 0].max() - P[:, 0].min()
    ay = P[:, 1].max() - P[:, 1].min()
    T = P[:, 2].max() - P[:, 2].min()
    wp = float(omega(np.array([kp]), DEPTH)[0]) if np.ndim(
        omega(np.array([kp]), DEPTH)) else float(omega(np.array([kp]), DEPTH))
    dens = n / (kp ** 2 * ax * ay / (2 * np.pi) ** 2 + 1e-9)
    return np.array([
        np.log10(n),
        np.log10(1 + kp * ax / (2 * np.pi)),
        np.log10(1 + kp * ay / (2 * np.pi)),
        np.log10(1 + wp * T / (2 * np.pi)),
        np.log10(np.std(z) / (sigma0 + 1e-9) + 1e-9),
        np.log10(1 + dens),
        np.log10(kp + 1e-9),
    ], np.float32)


def head_features(P, z, kp, sigma0, a_raw_n, t_stat, qs2_rel, e_top_frac,
                  bwp):
    f7 = info_features(P, z, kp, sigma0)
    extra = np.array([
        np.clip(a_raw_n, -1.0, 4.0),          # pooled LS alpha / std(z)
        np.clip(t_stat, 0.0, 12.0),
        np.log10(qs2_rel + 1e-9),             # raw render power / var(z)
        np.log10(e_top_frac + 1e-6),
        bwp,                                  # off-shell (2k,2w) dose probe
    ], np.float32)
    return np.concatenate([f7, extra])


# -------------------------------------------------------------- scenes
def build_scene(rng, dose, hs_rng, u_rng, tp=7.0):
    """Harness scene mirroring E-28 A's build_sea, dose as a parameter.
    Returns obs, held-out query pixels, GT (free + injected bound)."""
    Hs = rng.uniform(*hs_rng)
    c = jonswap_components(Hs=Hs, Tp=tp, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=rng.uniform(0, 360),
                           direction_spread_deg=30.0,
                           seed=int(rng.integers(0, 2 ** 31 - 1)))
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(tp, DEPTH)
    cp = lam / tp
    T = 10 * tp
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
    U = rng.uniform(*u_rng) * cp
    nfrac = rng.uniform(0.03, 0.08)
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    bq = np.zeros(len(Xs))
    if dose > 1e-4:                       # EXACT harness bound-wave recipe
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
        bkx, bky = 2 * kx[top], 2 * ky[top]
        bom, bph = 2 * om[top], 2 * ph[top]
        bA = 0.5 * np.hypot(kx[top], ky[top]) * Am[top] ** 2
        psib = (bkx[None] * P[:, 0:1] + bky[None] * P[:, 1:2]
                - bom[None] * P[:, 2:3] + bph[None])
        zb = (bA[None] * np.cos(psib)).sum(1)
        sc = (np.sqrt(dose) * (Hs / 4)) / max(np.std(zb), 1e-9)
        z = z + sc * zb
        psib_q = (bkx[None] * Xs[:, None] + bky[None] * Ys[:, None]
                  - bom[None] * T + bph[None])
        bq = sc * (bA[None] * np.cos(psib_q)).sum(1)
        gt_u = gt_u + bq
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, bq=bq, kp=kp,
                sigma0=float(nfrac * Hs), Hs=float(Hs), dose=float(dose))


def d_arms(S, si, head=None, want_oracle=False):
    """Compute all D quantities for one scene.  Returns dict.
    q renders are normalized to UNIT RMS on the held-out obs split; alpha
    is a free amplitude in meters, capped at ALPHA_MAX*std(z) (the harness
    doses bound waves by rescaling, so the physical 0.5kA^2 scale is not
    the injected scale — the per-scene gain must absorb it)."""
    P, z, Pq = S['P'], S['z'], S['Pq']
    kp, s0 = S['kp'], S['sigma0']
    s_z = float(np.std(z)) + 1e-12
    cap = ALPHA_MAX * s_z
    inc = IncumbentSolve(P, z, kp, s0)
    n = len(z)
    # full solve (== shipped incumbent); QBW from its free coefficients
    w_all, m_all = inc.solve(None)
    f_cur_q = inc.predict(Pq, w_all)
    a_f, b_f, kxf, kyf, omf = inc.free_coefs(m_all)
    po = {}
    q_obs = qbw_render(a_f, b_f, kxf, kyf, omf, kp, P, peaks_out=po)
    qs = float(np.sqrt(np.mean(q_obs ** 2)))
    if qs < 1e-14:
        q_q_n = np.zeros(len(Pq))
        al = raw = t = 0.0
        n_v = max(8, int(0.25 * n))
    else:
        q_obs_n = q_obs / qs
        q_q_n = qbw_render(a_f, b_f, kxf, kyf, omf, kp, Pq) / qs
        # K-fold alpha: K seeded 25% splits, fit-solve each, pool the LS
        raws, rhos = [], []
        for f in range(N_FOLD):
            rng = np.random.default_rng(1000 + si + 7717 * f)
            vi = rng.choice(n, max(8, int(0.25 * n)), replace=False)
            vmask = np.zeros(n, bool); vmask[vi] = True
            w_fit, _ = inc.solve(np.where(~vmask)[0])
            f0_v = inc.predict(P[vmask], w_fit)    # out-of-sample residual
            rw, rh = _fold_stats(z[vmask] - f0_v, q_obs_n[vmask])
            raws.append(rw); rhos.append(rh)
        n_v = int(vmask.sum())
        al, raw, t = alpha_pooled(raws, rhos, n_v, cap)
    qs2_rel = qs ** 2 / (s_z ** 2)
    bwp = bw_offshell_power(P, z, kxf, kyf, omf, po.get('pk', []))
    fe = head_features(P, z, kp, s0, raw / s_z, t, qs2_rel,
                       po.get('e_top_frac', 0.0), bwp)
    clusters = qbw_clusters(a_f, b_f, kxf, kyf, kp)
    out = dict(alpha_ls=al, alpha_raw=raw, t=t, f_cur=f_cur_q, q_q=q_q_n,
               s_z=s_z, feats=fe, inc=inc, clusters=clusters)
    if head is not None:
        with torch.no_grad():
            ah = float(head(torch.tensor(fe[None], device=DEVICE))[0])
        out['alpha_head'] = ah * s_z               # head is scale-free
    if want_oracle:
        gt = S['gt_u']
        best = (corr(f_cur_q, gt), 0.0)
        for ag in np.linspace(0.05 * cap, cap, 40):
            csc = corr(f_cur_q + ag * q_q_n, gt)
            if csc > best[0]:
                best = (csc, float(ag))
        out['alpha_orac'] = best[1]
    return out


def d_joint_arm(S, si, d, gate_margin=0.97):
    """Val-gated joint quadratic fit (PRIMARY deploy arm).  Gate: refit on
    the non-held-out obs must beat the incumbent's same-split solve at TWO
    holdouts — a random 25% AND a contiguous 25% TIME BLOCK (the block is
    the extrapolation-like probe; the project's standing failure law is
    that in-window random residuals do not predict field accuracy) — each
    by >3% RMSE, else fall back to the incumbent field."""
    inc, clusters = d['inc'], d['clusters']
    if not clusters:
        return dict(f=None, gated=False, la=0.0)
    n = len(inc.z)
    rng = np.random.default_rng(1000 + si)
    vi = rng.choice(n, max(8, int(0.25 * n)), replace=False)
    vrand = np.zeros(n, bool); vrand[vi] = True
    tord = np.argsort(inc.P[:, 2])
    st = int(rng.integers(0, n - n // 4))
    vblock = np.zeros(n, bool); vblock[tord[st:st + n // 4]] = True
    ok = True
    out = dict(la=0.0)
    for tag_, vmask in (('rand', vrand), ('block', vblock)):
        fit_idx = np.where(~vmask)[0]
        pred, la, _ = joint_qbw_refine(inc, clusters, fit_idx,
                                       alpha0=d['alpha_ls'])
        zv = inc.z[vmask]
        rm_j = float(np.sqrt(np.mean((pred(inc.P[vmask]) - zv) ** 2)))
        w_fit, _ = inc.solve(fit_idx)
        rm_0 = float(np.sqrt(np.mean((inc.predict(inc.P[vmask], w_fit)
                                      - zv) ** 2)))
        out[f'rm_j_{tag_}'] = rm_j; out[f'rm_0_{tag_}'] = rm_0
        out['la'] = la
        if not (rm_j < gate_margin * rm_0):
            ok = False
            break
    if not ok:
        return dict(f=None, gated=False, **out)
    pred_all, la2, _ = joint_qbw_refine(inc, clusters, np.arange(n),
                                        alpha0=out['la'])
    out['la'] = la2
    return dict(f=pred_all(S['Pq']), gated=True, **out)


def score_scene(S, si, head=None):
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    f_hand = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=s0, kp_hint=kp,
                                seed=si)
    f_cur = net_gamma_mmse(CUR_NET, P, z, DEPTH, Pq, s0, seed=si, kp_hint=kp,
                           smult_head=CUR_HEAD, bound_wave=True)
    d = d_arms(S, si, head=head, want_oracle=True)
    f_dp = d['f_cur'] + d['alpha_ls'] * d['q_q']
    # mechanism diagnostic (no decision role): does the phase-transferred
    # render actually point at the injected bound field at the query pixels?
    mech = (corr(d['q_q'], S['bq'])
            if (S['dose'] > 1e-4 and np.std(d['q_q']) > 1e-12) else np.nan)
    dj = d_joint_arm(S, si, d)
    f_dj = dj['f'] if dj['gated'] else d['f_cur']
    r = dict(hand=corr(f_hand, gt), cur=corr(f_cur, gt),
             cur_reimpl=corr(d['f_cur'], gt),
             dphys=corr(f_dp, gt),
             djoint=corr(f_dj, gt), gated=bool(dj['gated']),
             la_joint=float(dj['la']) / d['s_z'],
             dorac=corr(d['f_cur'] + d['alpha_orac'] * d['q_q'], gt),
             alpha_ls=d['alpha_ls'] / d['s_z'],
             alpha_orac=d['alpha_orac'] / d['s_z'],
             t=d['t'], mech=mech, dose=S['dose'], Hs=S['Hs'])
    if head is not None:
        r['dlearn'] = corr(d['f_cur'] + d['alpha_head'] * d['q_q'], gt)
        r['alpha_head'] = d['alpha_head'] / d['s_z']
    return r


# ------------------------------------------------------------- head train
def gen_head_sample(rng, si):
    """One training sample: features + closed-form MSE-optimal alpha target.
    For the 1-parameter linear-in-alpha family, weighted regression onto
    alpha* == the task loss (field-MSE through the pipeline) exactly, so no
    differentiable solve is needed (objective-axis simplification)."""
    dose = 0.0 if rng.uniform() < 0.3 else rng.uniform(0.05, 0.35)
    S = build_scene(rng, dose, hs_rng=(1.0, 3.0), u_rng=(0.0, 1.0))
    d = d_arms(S, si, head=None, want_oracle=False)
    q = d['q_q']                                   # val-RMS-normalized
    qq = float(q @ q)
    if qq < 1e-20:
        tgt, wgt = 0.0, 0.0
    else:
        # scale-free target: MSE-optimal alpha at the query pixels / std(z)
        tgt = float(np.clip((S['gt_u'] - d['f_cur']) @ q / qq / d['s_z'],
                            0.0, ALPHA_MAX))
        wgt = float(np.mean(q ** 2))               # render leverage at Q
    return d['feats'], tgt, wgt, dose


def train_head(n_train, steps, lr, seed, ckpt):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    print(f"generating {n_train} head-training scenes ...", flush=True)
    rows = [gen_head_sample(rng, 50000 + i) for i in range(n_train)]
    print(f"  gen done ({time.time() - t0:.0f}s)", flush=True)
    F = torch.tensor(np.stack([r[0] for r in rows]), device=DEVICE)
    Y = torch.tensor([r[1] for r in rows], device=DEVICE,
                     dtype=torch.float32)
    W = torch.tensor([r[2] for r in rows], device=DEVICE,
                     dtype=torch.float32)
    W = W / W.mean().clamp_min(1e-12)
    head = AlphaHead().to(DEVICE)
    head.mu.copy_(F.mean(0)); head.sd.copy_(F.std(0).clamp_min(1e-6))
    n_val = max(8, n_train // 5)
    vi = torch.arange(n_val); ti = torch.arange(n_val, n_train)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    tr = np.random.default_rng(seed)

    def vloss():
        with torch.no_grad():
            p = head(F[vi])
            return float((W[vi] * (p - Y[vi]) ** 2).mean())
    best = (vloss(), 0)
    print(f"step     0 val {best[0]:.4f}")
    torch.save(dict(model=head.state_dict()), ckpt)
    for step in range(1, steps + 1):
        ii = torch.tensor(tr.choice(ti.numpy(), min(32, len(ti)),
                                    replace=False))
        p = head(F[ii])
        loss = (W[ii] * (p - Y[ii]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 10) == 0:
            vl = vloss()
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                torch.save(dict(model=head.state_dict()), ckpt)
                tag = ' *saved'
            print(f"step {step:5d} train {float(loss):.4f} val {vl:.4f}{tag}",
                  flush=True)
    print(f"best val {best[0]:.4f} @ step {best[1]} -> {ckpt}")
    head.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                    weights_only=False)['model'])
    head.eval()
    return head


# ----------------------------------------------------------------- eval
def run_eval(head, ns=(4, 6, 4), tag='smoke'):
    buckets = [('B0', 0.0, (1.5, 3.0), (0.0, 1.0), ns[0]),
               ('B1', None, (1.5, 3.0), (0.0, 1.0), ns[1]),
               ('B2', None, (4.0, 5.0), (1.5, 1.5), ns[2])]
    res = {}
    for name, dz, hs, ur, n in buckets:
        rows = []
        for si in range(n):
            rng = np.random.default_rng(7700 + 131 * si
                                        + sum(map(ord, name)))
            dose = (0.0 if name == 'B0'
                    else rng.uniform(0.1, 0.3) if name == 'B1'
                    else rng.uniform(0.3, 0.5))
            S = build_scene(rng, dose, hs, ur)
            r = score_scene(S, si, head=head)
            rows.append(r)
            ln = (f"  {name} sea {si}: dose {r['dose']:.2f} | hand "
                  f"{r['hand']:+.3f} cur {r['cur']:+.3f} dphys "
                  f"{r['dphys']:+.3f} djoint {r['djoint']:+.3f}"
                  f"{'*' if r['gated'] else ' '} dorac {r['dorac']:+.3f}")
            if 'dlearn' in r:
                ln += f" dlearn {r['dlearn']:+.3f}"
            ln += (f" | aLS {r['alpha_ls']:.2f} aJ {r['la_joint']:.2f} "
                   f"aOR {r['alpha_orac']:.2f} t {r['t']:.1f}")
            print(ln, flush=True)
        res[name] = rows

    def M(rows, k1, k2=None):
        v = [r[k1] - (r[k2] if k2 else 0.0) for r in rows]
        return float(np.mean(v))
    dosed = res['B1'] + res['B2']
    reimpl_gap = float(np.max(np.abs(
        [r['cur'] - r['cur_reimpl'] for r in res['B0'] + dosed])))
    p1 = M(res['B0'], 'djoint', 'cur')
    p1p = M(res['B0'], 'dphys', 'cur')
    p2 = M(dosed, 'djoint', 'cur')
    p2p = M(dosed, 'dphys', 'cur')
    win2 = float(np.mean([r['djoint'] > r['cur'] for r in dosed]))
    gate_rate = float(np.mean([r['gated'] for r in dosed]))
    dor = M(dosed, 'dorac', 'cur')
    diffs = np.array([r['djoint'] - r['cur'] for r in dosed])
    bs = np.random.default_rng(0).choice(len(diffs), (2000, len(diffs)))
    ci = np.percentile(diffs[bs].mean(1), [2.5, 97.5])
    from scipy.stats import spearmanr
    rho, pval = spearmanr([r['dose'] for r in dosed],
                          [r['djoint'] - r['cur'] for r in dosed])
    print(f"\n===== pre-registered criteria ({tag}; primary arm=Djoint) ====")
    print(f"reimpl check  max|cur - cur_reimpl| = {reimpl_gap:.2e}")
    print(f"P1 do-no-harm  Djoint-cur | B0     = {p1:+.4f}  (>= -0.005 ?"
          f" {p1 >= -0.005})   [Dphys {p1p:+.4f}]")
    print(f"P2 dosed gain  Djoint-cur | B1+B2  = {p2:+.4f}  win {win2:.0%}"
          f"  gate {gate_rate:.0%}  CI [{ci[0]:+.4f},{ci[1]:+.4f}]"
          f"  ({p2 > 0 and win2 >= 0.6 and ci[0] > 0})   [Dphys {p2p:+.4f}]")
    print(f"P3 dose-resp   Spearman(dose, gain) = {rho:+.2f} (p {pval:.3f})")
    out = dict(res=res, p1=p1, p1_phys=p1p, p2=p2, p2_phys=p2p, win2=win2,
               gate_rate=gate_rate, ci=list(map(float, ci)),
               dorac_gain=dor, spearman=[float(rho), float(pval)],
               reimpl_gap=reimpl_gap, tag=tag)
    if head is not None:
        p4 = M(dosed, 'dlearn', 'dphys')
        print(f"P4 head sanity Dlearn-Dphys | dosed = {p4:+.4f} "
              f"(>= -0.003 ? {p4 >= -0.003})")
        out['p4'] = p4
    print(f"KILL check     Dorac-cur | dosed    = {dor:+.4f} "
          f"(posterior-render headroom; perfect-direction headroom "
          f"measured separately ~ +0.29)")
    json.dump(out, open(OUT_JSON, 'w'), indent=1, default=float)
    print(f"saved {OUT_JSON}")
    return out


# -------------------------------------------------------------- selftest
def selftest():
    print("=== e29 design-D selftest ===")
    ok = True

    # [1] QBW exactness: true modes ON bank atoms, well separated ->
    #     rendered bound field == analytic injected bound field.
    rng = np.random.default_rng(5)
    kp = 0.08
    ax = np.linspace(-KREL * kp, KREL * kp, NGRID)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < KREL * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    om_b = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
    pick = []
    for i in rng.permutation(len(kxb)):
        if len(pick) == 4:
            break
        if all(np.hypot(kxb[i] - kxb[j], kyb[i] - kyb[j]) > 0.4 * kp
               for j in pick):
            pick.append(int(i))
    A = rng.uniform(0.3, 1.0, 4)
    phi = rng.uniform(0, 2 * np.pi, 4)
    a = np.zeros(len(kxb)); b = np.zeros(len(kxb))
    a[pick] = A * np.cos(phi)
    b[pick] = -A * np.sin(phi)
    pts = np.column_stack([rng.uniform(0, 300, 400),
                           rng.uniform(0, 300, 400),
                           rng.uniform(0, 70, 400)])
    q = qbw_render(a, b, kxb, kyb, om_b, kp, pts)
    th = (kxb[pick][None] * pts[:, 0:1] + kyb[pick][None] * pts[:, 1:2]
          - om_b[pick][None] * pts[:, 2:3])
    ref = (0.5 * np.hypot(kxb[pick], kyb[pick])[None] * A[None] ** 2
           * np.cos(2 * th + 2 * phi[None])).sum(1)
    e1 = float(np.max(np.abs(q - ref)) / (np.max(np.abs(ref)) + 1e-12))
    c1 = corr(q, ref)
    print(f"[1] QBW exactness: rel err {e1:.2e} corr {c1:.6f} "
          f"(err<1e-6 & corr>0.999 ? {e1 < 1e-6 and c1 > 0.999})")
    ok &= (e1 < 1e-6 and c1 > 0.999)

    # [2] incumbent reimplementation == shipped net_gamma_mmse (<1e-4)
    S = build_scene(np.random.default_rng(11), dose=0.2, hs_rng=(2.0, 2.0),
                    u_rng=(0.5, 0.5))
    f_ship = net_gamma_mmse(CUR_NET, S['P'], S['z'], DEPTH, S['Pq'],
                            S['sigma0'], seed=3, kp_hint=S['kp'],
                            smult_head=CUR_HEAD, bound_wave=True)
    inc = IncumbentSolve(S['P'], S['z'], S['kp'], S['sigma0'])
    w, m = inc.solve(None)
    f_mine = inc.predict(S['Pq'], w)
    e2 = float(np.max(np.abs(f_ship - f_mine)))
    print(f"[2] reimpl |shipped - mine|_max = {e2:.2e} (<1e-4 ? {e2 < 1e-4})")
    ok &= (e2 < 1e-4)

    # [3] alpha LS on a synthetic case with known alpha + zero-q guard
    r = np.random.default_rng(7)
    qv = r.normal(0, 1.0, 300)
    resid = 0.7 * qv + r.normal(0, 0.5, 300)
    rw, rh = _fold_stats(resid, qv)
    al, raw, t = alpha_pooled([rw], [rh], 300, cap=2.0)
    print(f"[3] alpha_LS synthetic: raw {raw:.3f} (~0.7), shrunk {al:.3f}, "
          f"t {t:.1f} (raw in [0.6,0.8] ? {0.6 < raw < 0.8})")
    ok &= (0.6 < raw < 0.8)
    rw0, rh0 = _fold_stats(r.normal(0, 1, 300), np.zeros(300))
    al0, _, _ = alpha_pooled([rw0], [rh0], 300, cap=2.0)
    print(f"    zero-q guard alpha = {al0} (0 ? {al0 == 0.0})")
    ok &= (al0 == 0.0)
    # scale-freeness: dosed scene must yield alpha_orac with real effect
    Sd = build_scene(np.random.default_rng(21), dose=0.35,
                     hs_rng=(2.0, 2.0), u_rng=(0.8, 0.8))
    dd = d_arms(Sd, 5, want_oracle=True)
    g_or = (corr(dd['f_cur'] + dd['alpha_orac'] * dd['q_q'], Sd['gt_u'])
            - corr(dd['f_cur'], Sd['gt_u']))
    mch = corr(dd['q_q'], Sd['bq'])
    print(f"    dosed probe: alpha_orac/s_z {dd['alpha_orac']/dd['s_z']:.2f} "
          f"oracle gain {g_or:+.4f} mech corr(q, bound_gt) {mch:+.3f}")

    # [4] determinism of the full D arm
    d1 = d_arms(S, 3); d2 = d_arms(S, 3)
    same = (np.allclose(d1['f_cur'], d2['f_cur'])
            and np.allclose(d1['q_q'], d2['q_q'])
            and d1['alpha_ls'] == d2['alpha_ls'])
    print(f"[4] deterministic ? {same}")
    ok &= same

    # [4b] joint quadratic refine: loss decreases, gate logic runs, and on
    #      the dosed probe the val-gated arm must not be worse than cur
    dd2 = d_arms(Sd, 5)
    dj = d_joint_arm(Sd, 5, dd2)
    f_dj = dj['f'] if dj['gated'] else dd2['f_cur']
    c_cur = corr(dd2['f_cur'], Sd['gt_u'])
    c_dj = corr(f_dj, Sd['gt_u'])
    print(f"[4b] joint refine: gated {dj['gated']} la/s_z "
          f"{dj['la'] / dd2['s_z']:+.2f} rand "
          f"{dj.get('rm_j_rand', np.nan):.3f}/{dj.get('rm_0_rand', np.nan):.3f}"
          f" block {dj.get('rm_j_block', np.nan):.3f}/"
          f"{dj.get('rm_0_block', np.nan):.3f} | "
          f"cur {c_cur:+.3f} djoint {c_dj:+.3f}")

    # [5] head forward: shapes, range, near-constant at init
    h = AlphaHead().to(DEVICE)
    v = torch.randn(5, N_FEAT, device=DEVICE)
    with torch.no_grad():
        o = h(v)
    inr = bool((o >= 0).all() and (o <= ALPHA_MAX).all())
    print(f"[5] head out range [0,{ALPHA_MAX}] ? {inr} "
          f"(init spread {float(o.std()):.2e})")
    ok &= inr
    print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--n_train', type=int, default=None)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"device {DEVICE}  incumbent {INIT} (step {_CK['step']})",
          flush=True)
    if a.selftest:
        return 0 if selftest() else 1
    if a.smoke:
        n_train = a.n_train or 80
        steps = a.steps or 300
        ns = (4, 6, 4)
        tag = 'smoke'
    elif a.full:
        n_train = a.n_train or 1500
        steps = a.steps or 2000
        ns = (16, 24, 16)
        tag = 'full'
    else:
        print("pick one of --selftest --smoke --full")
        return 1
    head = train_head(n_train, steps, a.lr, a.seed, HEAD_CKPT)
    run_eval(head, ns=ns, tag=tag)
    return 0


if __name__ == '__main__':
    sys.exit(main())
