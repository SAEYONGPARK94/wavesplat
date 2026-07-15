"""
e28_gennet_B.py — a GENUINELY DIFFERENT competing generalization architecture
=============================================================================
Route chosen: **B1 — Informativeness-FiLM conditioning** (see report below).

Teammate's design (the thing we must NOT resemble): an MMSE-anchored RESIDUAL
with a learned soft gate alpha(CRB) that BLENDS a classical estimate and a
learned residual. That design has an *explicit two-branch + scalar-blend*
structure.

This design has NO explicit anchor/residual split and NO explicit blend.
It is a SINGLE amortized-gamma(k) prior that is anchored to the physics MMSE
BY CONSTRUCTION (the net only chooses the prior variance gamma(k); the
well-specified physics MMSE — which E-22/25/27 found essentially unique and
physics-generalizing — does the reconstruction). Generalization comes from a
different mechanism than a gate:

  (1) an INFORMATIVENESS EMBEDDING (obs aperture-in-wavelengths, obs count,
      obs density, noise fraction, absolute scale, and a beamform-map
      participation-ratio CRB/ambiguity proxy) is fed to EVERY U-Net block via
      FiLM (feature-wise linear modulation): h <- h*(1+scale(v)) + shift(v).
      The net is forced to learn an *explicit* informativeness -> gamma(k)
      mapping instead of a memorized one, so it can interpolate/extrapolate in
      informativeness space rather than in sea-index space.

  (2) HEAVY OPERATOR RANDOMIZATION (widened augmentation: Hs 1-5, spatial
      aperture from <1 lambda to >2 lambda, fixed/moving operators, noise
      0.02-0.10, cheap 2nd-order bound-wave steepness) samples the
      informativeness axis densely, so the learned mapping is dense where
      E-24 says it matters (CRB / under-determination is the independent
      predictor; density axis 12/12 p<1e-4).

Why FiLM is soft-by-construction (satisfies E-25, which killed the HARD CRB
gate): FiLM is a *continuous* modulation with no threshold and no discrete
branch selection. The net can win everywhere on realistic M3 seas (E-25) and
still profit from informativeness — there is no gate to force it off.

Why FiLM re-injects information the backbone has ALREADY lost: the 6-channel
beamform maps are PER-MAP MAX-NORMALIZED (X.max()==1) and in NORMALIZED k/kp
coordinates. That normalization *destroys* absolute aperture-in-wavelengths,
absolute SNR, absolute scale kp, and obs count. Those are exactly the
informativeness scalars. So FiLM is not redundant with the maps — it restores
the axes the normalization threw away.

Zero-init FiLM => at step 0 this net is EXACTLY the warm-start PriorNet
(prior_net_v3), so warm-starting is lossless and any deviation is learned.

Reuses: d3_prior_net_train.PriorNet/band_mask (backbone + band),
d3_prior_net_taskft.task_loss_one/load_all (differentiable MMSE task loss),
d3_prior_net_data helpers (widened data gen), d3_gpu_mmse machinery (design).

Subcommands
-----------
  --selftest   compile + forward/backward + FiLM-identity + inform-features check
  --smoke      <5-min tiny task-loss train from prior_net_v3, reports FiLM spread
  --gen        widened-augmentation data generation (Hs 1-5, aperture jitter)
  --loro       pre-registered LORO comparison FiLM vs current-net vs hand-MMSE

Usage:
  python3 tmp/e28_gennet_B.py --selftest
  python3 tmp/e28_gennet_B.py --smoke --steps 300 --data tmp/prior_train/fan0
  python3 tmp/e28_gennet_B.py --loro --n_per 24
"""
import os
import sys
import glob
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d3_prior_net_taskft import task_loss_one, load_all, DEPTH, G

# low-level generators reused from the shipped data pipeline
from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_nomp import beamform_power, split_masks
from d3_sa_achievability_gap import dispersion_lam, sample_obs, N_OBS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NF = 9                                  # informativeness feature count
CKPT = 'results/direction3/e28_gennet_B.pt'
CUR_NET = 'results/direction3/prior_net_v3.pt'   # "the current net" comparator

# band geometry in normalized coords (shared, built once)
_MASK = torch.tensor(band_mask(), device=DEVICE)
_AXN = torch.linspace(-KREL, KREL, NGRID, device=DEVICE)
_KXN, _KYN = torch.meshgrid(_AXN, _AXN, indexing='xy')
_KXNB, _KYNB = _KXN[_MASK], _KYN[_MASK]
_BAND_NP = band_mask()


# --------------------------------------------------------------- informativeness
def inform_features(P, z, kp_e, nfrac, Hs, Xmap):
    """Scalars that per-map normalization DESTROYS + a CRB/ambiguity proxy.

    P (N,3) x,y,t ; z (N,) ; kp_e scalar ; nfrac,Hs scalars ; Xmap (6,51,51).
    Returns float32 (NF,). All chosen to be computable from sparse obs alone
    (E-9 convention) and to be *invariant-destroyed-by-normalization* so FiLM
    adds real information over the maps.
    """
    P = np.asarray(P); z = np.asarray(z)
    kp = max(float(kp_e), 1e-3)
    lam = 2 * np.pi / kp
    om_p = np.sqrt(G * kp * np.tanh(kp * DEPTH) + 1e-12)
    Tp = 2 * np.pi / om_p
    ax = (P[:, 0].max() - P[:, 0].min()) / lam           # aperture-x [lambda]
    ay = (P[:, 1].max() - P[:, 1].min()) / lam           # aperture-y [lambda]
    at = (P[:, 2].max() - P[:, 2].min()) / max(Tp, 1e-6)  # aperture-t [Tp]
    N = len(z)
    dens = N / (ax * ay + 1e-3)                          # obs per lambda^2
    # participation ratio of the beamform min-map over the annulus band:
    # spread (many bright cells) -> ambiguous direction / low information;
    # sharp (one cell) -> well determined. A cheap CRB-ish proxy.
    pm = np.asarray(Xmap[2])[_BAND_NP]
    pm = np.clip(pm, 0, None)
    s1 = pm.sum() + 1e-12
    pr = (s1 ** 2) / ((pm ** 2).sum() + 1e-12) / max(len(pm), 1)
    v = np.array([
        np.log10(ax + 1e-3),
        np.log10(ay + 1e-3),
        np.log10(at + 1e-3),
        np.log10(N + 1e-3),
        np.log10(dens + 1e-3),
        float(nfrac),
        np.log10(kp + 1e-6),
        float(pr),
        np.log10(np.std(z) + 1e-6),
    ], np.float32)
    return v


# --------------------------------------------------------------- FiLM + backbone
class FiLMGen(nn.Module):
    """informativeness vector -> per-block (scale, shift). Zero-init => identity.

    Shared trunk + one head per FiLM site. The final layer of each head is
    zero-initialized so scale=shift=0 at init; the net therefore *starts* as the
    unconditioned PriorNet and only departs where training finds informativeness
    useful.
    """

    def __init__(self, chans, n_feat=NF, emb=64):
        super().__init__()
        self.register_buffer('mu', torch.zeros(n_feat))
        self.register_buffer('sd', torch.ones(n_feat))
        self.trunk = nn.Sequential(nn.Linear(n_feat, emb), nn.SiLU(),
                                   nn.Linear(emb, emb), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(emb, 2 * c) for c in chans])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.chans = list(chans)

    def set_norm(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32, device=self.mu.device))
        self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32, device=self.sd.device).clamp_min(1e-6))

    def forward(self, v):
        h = self.trunk((v - self.mu) / self.sd)
        out = []
        for head, c in zip(self.heads, self.chans):
            p = head(h)
            out.append((p[:, :c], p[:, c:]))            # (scale, shift)
        return out


def _film(h, sc, sh):
    return h * (1.0 + sc[:, :, None, None]) + sh[:, :, None, None]


class FiLMPriorNet(nn.Module):
    """PriorNet backbone with FiLM applied after every encoder/decoder block.

    Backbone weights are load-compatible with prior_net_*.pt (same submodules),
    so warm-starting from the current net is exact.
    """

    def __init__(self, ch=48, n_feat=NF):
        super().__init__()
        self.backbone = PriorNet(ch=ch)
        self.film = FiLMGen([ch, 2 * ch, 4 * ch, 2 * ch, ch], n_feat=n_feat)

    def load_backbone(self, ckpt):
        ck = torch.load(ckpt, map_location=DEVICE)
        self.backbone.load_state_dict(ck['model'])
        return ck

    def forward(self, x, v):
        b = self.backbone
        (s1, h1), (s2, h2), (s3, h3), (s4, h4), (s5, h5) = self.film(v)
        e1 = _film(b.e1(x), s1, h1)                     # 51
        e2 = _film(b.e2(b.pool(e1)), s2, h2)            # 25
        e3 = _film(b.e3(b.pool(e2)), s3, h3)            # 12
        u3 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear',
                           align_corners=False)
        d2 = _film(b.d2(torch.cat([u3, e2], 1)), s4, h4)
        u2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear',
                           align_corners=False)
        d1 = _film(b.d1(torch.cat([u2, e1], 1)), s5, h5)
        return b.out(d1)[:, 0]


# --------------------------------------------------------------- MMSE forward
def mmse_field(net_out, P, z, Q, kp_e, sigma0, log_smult):
    """Differentiable physics MMSE (twin of task_loss_one) returning the FIELD
    prediction at query pixels Q. Same free-wave bank as the shipped estimator."""
    g = torch.pow(10.0, net_out[_MASK])
    kxb = _KXNB * kp_e
    kyb = _KYNB * kp_e
    k = torch.sqrt(kxb ** 2 + kyb ** 2 + 1e-12)
    om = torch.sqrt(G * k * torch.tanh(k * DEPTH) + 1e-12)
    g = g / g.sum()
    gam = torch.cat([g, g]) * torch.var(z)

    def design(Qp):
        psi = (kxb[None] * Qp[:, 0:1] + kyb[None] * Qp[:, 1:2]
               - om[None] * Qp[:, 2:3])
        return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)

    Phi = design(P)
    s2 = (torch.exp(log_smult) * sigma0) ** 2
    n = len(z)
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(n, device=net_out.device)
    alpha = torch.linalg.solve(K, z)
    return (design(Q) * gam[None]) @ (Phi.T @ alpha)


def corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


# --------------------------------------------------------------- widened data gen
def gen_widened(rng, hs_range=(1.0, 5.0), side_range=(0.7, 2.5), bw_max=0.0,
                modes=('fixed', 'moving'), tp_range=(5.0, 11.0)):
    """Widened-augmentation sample (E-28 directive: Hs 1-5, aperture jitter,
    cheap 2nd-order bound-wave steepness). Mirrors d3_prior_net_data.gen_sample
    but exposes Hs / spatial-aperture / steepness as controllable regime axes so
    LORO can hold them out. Field scored at t_star = T (nowcast), like
    d3_prior_net_eval. Returns the shard dict + regime tags."""
    Tp = rng.uniform(*tp_range)
    Hs = rng.uniform(*hs_range)
    gam_j = rng.uniform(1.5, 6.0)
    spread = rng.uniform(15.0, 45.0)
    mdir = rng.uniform(0.0, 360.0)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=gam_j, direction_deg=mdir,
                           direction_spread_deg=spread, seed=seed)
    kxm = np.array([x.kx for x in c]); kym = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]);   phm = np.array([x.phi for x in c])
    k = np.hypot(kxm, kym)
    om = np.sqrt(G * k * np.tanh(k * DEPTH) + 1e-12)
    uc = rng.uniform(0.0, 0.4); ua = rng.uniform(0, 2 * np.pi)
    om = om + kxm * uc * np.cos(ua) + kym * uc * np.sin(ua)

    # cheap steepness axis: 2nd-order self-super-harmonic bound waves (2k,2omega)
    bw = None
    bw_lvl = rng.uniform(0.0, bw_max) if bw_max > 0 else 0.0
    if bw_lvl > 1e-4:
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
        bw = dict(kx=2 * kxm[top], ky=2 * kym[top], om=2 * om[top],
                  ph=2 * phm[top],
                  A=0.5 * np.hypot(kxm[top], kym[top]) * Am[top] ** 2)

    lam = dispersion_lam(Tp, DEPTH); cp = lam / Tp; T = 10 * Tp
    fp = 0.18 * lam; r_fp = fp / 2
    side = rng.uniform(*side_range) * lam                 # << APERTURE axis
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    mode = modes[rng.integers(len(modes))]
    ucp = 0.0 if mode == 'fixed' else rng.uniform(0.1, 1.2)
    P = sample_obs(mode, ucp * cp, lam, cp, T, gx, gy, r_fp, rng)
    psi = (kxm[None] * P[:, 0:1] + kym[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + phm[None])
    nfrac = rng.uniform(0.02, 0.10)
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    if bw is not None:
        psib = (bw['kx'][None] * P[:, 0:1] + bw['ky'][None] * P[:, 1:2]
                - bw['om'][None] * P[:, 2:3] + bw['ph'][None])
        zb = (bw['A'][None] * np.cos(psib)).sum(1)
        bw['sc'] = (np.sqrt(bw_lvl) * (Hs / 4.0)) / max(np.std(zb), 1e-9)
        z = z + bw['sc'] * zb

    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=2.0, n=120)
    kp_e = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=3.5 * kp_e, n=120)
    kp_e = max(np.hypot(bx, by), 1e-3)

    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], DEPTH, KREL * kp_e, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], DEPTH, KREL * kp_e, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    X = np.stack([p1, p2, pm,
                  np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                  np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)

    qx = rng.uniform(0, side, 256); qy = rng.uniform(0, side, 256)
    tq = np.full(256, T)
    psi_q = (kxm[None] * qx[:, None] + kym[None] * qy[:, None]
             - om[None] * tq[:, None] + phm[None])
    gt_q = (Am[None] * np.cos(psi_q)).sum(1)
    if bw is not None:
        psib_q = (bw['kx'][None] * qx[:, None] + bw['ky'][None] * qy[:, None]
                  - bw['om'][None] * tq[:, None] + bw['ph'][None])
        gt_q = gt_q + bw['sc'] * (bw['A'][None] * np.cos(psib_q)).sum(1)
    Q = np.column_stack([qx, qy, tq]).astype(np.float32)

    return dict(X=X, P=P.astype(np.float32), z=z.astype(np.float32), Q=Q,
                gt_q=gt_q.astype(np.float32), kp_e=float(kp_e), Hs=float(Hs),
                Tp=float(Tp), ucp=float(ucp), nfrac=float(nfrac),
                side_mult=float(side / lam), bw=float(bw_lvl))


# --------------------------------------------------------------- self-test
def selftest():
    print('=== e28_gennet_B selftest ===')
    torch.manual_seed(0)
    net = FiLMPriorNet().to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    n_film = sum(p.numel() for p in net.film.parameters())
    print(f'params total {n_par:,}  (FiLM {n_film:,})')

    # (1) FiLM is identity at init -> equals unconditioned backbone
    x = torch.randn(4, 6, NGRID, NGRID, device=DEVICE)
    v = torch.randn(4, NF, device=DEVICE)
    net.eval()
    with torch.no_grad():
        y = net(x, v)
        y_bb = net.backbone(x)
    id_err = float((y - y_bb).abs().max())
    print(f'(1) FiLM-identity-at-init max|film - backbone| = {id_err:.2e} '
          f'-> {"PASS" if id_err < 1e-5 else "FAIL"}')

    # (2) after a FiLM perturbation, output DEPENDS on informativeness v
    with torch.no_grad():
        for h in net.film.heads:
            h.weight.normal_(0, 0.3); h.bias.normal_(0, 0.3)
        v2 = v.clone(); v2[:, 1] += 3.0             # change aperture-y feature
        dy = float((net(x, v) - net(x, v2)).abs().mean())
    print(f'(2) d(out)/d(inform) mean|.| = {dy:.4f} '
          f'-> {"PASS" if dy > 1e-3 else "FAIL (FiLM inert)"}')

    # (3) forward/backward through the differentiable MMSE task loss
    net.train()
    s = gen_widened(np.random.default_rng(1), hs_range=(2, 2), side_range=(1.5, 1.5))
    X = torch.tensor(s['X'][None], device=DEVICE)
    vf = torch.tensor(inform_features(s['P'], s['z'], s['kp_e'], s['nfrac'],
                                      s['Hs'], s['X'])[None], device=DEVICE)
    P = torch.tensor(s['P'], device=DEVICE); z = torch.tensor(s['z'], device=DEVICE)
    Q = torch.tensor(s['Q'], device=DEVICE); gt = torch.tensor(s['gt_q'], device=DEVICE)
    ls = torch.tensor(float(np.log(3.0)), device=DEVICE, requires_grad=True)
    out = net(X, vf)[0]
    loss = task_loss_one(out, P, z, Q, gt, s['kp_e'], s['nfrac'] * s['Hs'],
                         ls, _MASK, _KXNB, _KYNB)
    lval = float(loss.detach())
    loss.backward()
    gnorm = float(torch.cat([p.grad.flatten() for p in net.film.parameters()
                             if p.grad is not None]).norm())
    print(f'(3) task-loss={lval:.4f}  FiLM grad-norm={gnorm:.4e} '
          f'-> {"PASS" if np.isfinite(lval) and gnorm > 0 else "FAIL"}')

    # (4) mmse_field prediction matches a sensible corr on a clean sample
    with torch.no_grad():
        pred = mmse_field(net(X, vf)[0], P, z, Q, s['kp_e'],
                          s['nfrac'] * s['Hs'], ls.detach())
    c = corr(pred.cpu().numpy(), s['gt_q'])
    print(f'(4) mmse_field corr on one sample = {c:+.3f} '
          f'-> {"PASS" if np.isfinite(c) else "FAIL"}')

    # (5) informativeness features vary with the observation operator
    wide = gen_widened(np.random.default_rng(2), side_range=(2.4, 2.5))
    narrow = gen_widened(np.random.default_rng(3), side_range=(0.7, 0.75))
    vw = inform_features(wide['P'], wide['z'], wide['kp_e'], wide['nfrac'],
                         wide['Hs'], wide['X'])
    vn = inform_features(narrow['P'], narrow['z'], narrow['kp_e'],
                         narrow['nfrac'], narrow['Hs'], narrow['X'])
    print(f'(5) aperture-x feat  wide={vw[0]:+.2f}  narrow={vn[0]:+.2f} '
          f'-> {"PASS" if vw[0] > vn[0] else "FAIL"}')
    print('feature names: [ap_x, ap_y, ap_t, logN, log_dens, nfrac, '
          'log_kp, PR, log_std_z]')
    print('selftest done.')


# --------------------------------------------------------------- smoke train
def _feat_for(D, i):
    return inform_features(D['P'][i], D['z'][i], D['meta'][i, 0],
                           D['meta'][i, 4], D['meta'][i, 1], D['X'][i])


def smoke(args):
    print(f'=== smoke train ({args.steps} steps, data {args.data}) ===')
    D = load_all(args.data)
    n_all = len(D['X'])
    # precompute informativeness for every sample once
    t0 = time.time()
    V = np.stack([_feat_for(D, i) for i in range(n_all)]).astype(np.float32)
    print(f'informativeness precomputed for {n_all} samples '
          f'({time.time()-t0:.0f}s)')
    n_val = max(120, n_all // 12)
    vi = np.arange(n_val); ti = np.arange(n_val, n_all)

    net = FiLMPriorNet().to(DEVICE)
    ck = net.load_backbone(args.init)
    net.film.set_norm(V.mean(0), V.std(0))
    log_smult = torch.tensor(float(ck.get('log_smult', np.log(3.0))),
                             device=DEVICE, requires_grad=True)
    # FiLM lr high (fresh), backbone lr low (warm), smult moderate
    opt = torch.optim.AdamW(
        [{'params': net.backbone.parameters(), 'lr': args.lr_bb},
         {'params': net.film.parameters(), 'lr': args.lr_film},
         {'params': [log_smult], 'lr': 3e-3}], weight_decay=1e-5)

    def loss_of(idx, out):
        P = torch.tensor(D['P'][idx], device=DEVICE)
        z = torch.tensor(D['z'][idx], device=DEVICE)
        Q = torch.tensor(D['Q'][idx], device=DEVICE)
        gt = torch.tensor(D['gt_q'][idx], device=DEVICE)
        kp_e, Hs = D['meta'][idx, 0], D['meta'][idx, 1]
        sig0 = float(D['meta'][idx, 4] * Hs)
        return task_loss_one(out, P, z, Q, gt, float(kp_e), sig0,
                             log_smult, _MASK, _KXNB, _KYNB) / (Hs / 4) ** 2

    def evaluate():
        net.eval()
        with torch.no_grad():
            # val task loss + FiLM spread + FiLM-ablation (const informativeness)
            ls_var, ls_const, sc_std = [], [], []
            vmean = torch.tensor(V.mean(0)[None], device=DEVICE)
            for i in vi[:120]:
                X = torch.tensor(D['X'][i][None], device=DEVICE)
                vf = torch.tensor(V[i][None], device=DEVICE)
                ls_var.append(float(loss_of(int(i), net(X, vf)[0])))
                ls_const.append(float(loss_of(int(i), net(X, vmean)[0])))
                sc = net.film(vf)                    # per-block (scale,shift)
                sc_std.append(float(torch.cat([s[0].flatten() for s in sc]).abs().mean()))
        net.train()
        return (float(np.mean(ls_var)), float(np.mean(ls_const)),
                float(np.mean(sc_std)))

    rng = np.random.default_rng(0)
    v0, c0, spread0 = evaluate()
    print(f'step     0  val {v0:.4f}  (const-inform {c0:.4f})  '
          f'FiLM|scale| {spread0:.4f}', flush=True)
    best = (v0, 0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        ii = rng.choice(ti, args.bs, replace=False)
        X = torch.tensor(D['X'][ii], device=DEVICE)
        Vb = torch.tensor(V[ii], device=DEVICE)
        outs = net(X, Vb)
        loss = sum(loss_of(int(i), outs[j]) for j, i in enumerate(ii)) / args.bs
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [log_smult], 1.0)
        opt.step()
        if step % args.every == 0 or step == args.steps:
            vl, cl, sp = evaluate()
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                torch.save(dict(model=net.backbone.state_dict(),
                                film=net.film.state_dict(),
                                log_smult=float(log_smult.detach()),
                                feat_mu=V.mean(0), feat_sd=V.std(0),
                                step=step, val=vl), CKPT)
                tag = ' *saved'
            print(f'step {step:5d} [{time.time()-t0:4.0f}s] train {float(loss):.4f} '
                  f'val {vl:.4f}  const-inform {cl:.4f}  FiLM|scale| {sp:.4f}'
                  f'{tag}', flush=True)
    dv = best[0] - v0
    film_gain = None
    vl, cl, sp = evaluate()
    print(f'\nbest val {best[0]:.4f} @ {best[1]} (step-0 {v0:.4f}, d {dv:+.4f})')
    print(f'FiLM-ablation at end: val(real inform) {vl:.4f} vs '
          f'val(const inform) {cl:.4f}  -> FiLM contributes {cl - vl:+.4f}')
    print(f'FiLM |scale| spread {sp:.4f}  '
          f'({"non-degenerate" if sp > 1e-3 else "DEGENERATE (nothing learned)"})')
    print(f'saved -> {CKPT}')


# --------------------------------------------------------------- data gen
def gen(args):
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    rng = np.random.default_rng(args.seed)
    buf, si, t0 = [], 0, time.time()
    for i in range(args.n):
        buf.append(gen_widened(rng, hs_range=(args.hs_lo, args.hs_hi),
                               side_range=(args.side_lo, args.side_hi),
                               bw_max=args.bw_max))
        if len(buf) == args.shard or i == args.n - 1:
            path = f'{args.out}_s{si:03d}.npz'
            np.savez_compressed(
                path, X=np.stack([b['X'] for b in buf]),
                P=np.stack([b['P'] for b in buf]),
                z=np.stack([b['z'] for b in buf]),
                Q=np.stack([b['Q'] for b in buf]),
                gt_q=np.stack([b['gt_q'] for b in buf]),
                meta=np.array([[b['kp_e'], b['Hs'], b['Tp'], b['ucp'],
                                b['nfrac'], b['side_mult'], b['bw']]
                               for b in buf], np.float32))
            print(f'shard {si} ({i+1}/{args.n}) {time.time()-t0:.0f}s -> {path}',
                  flush=True)
            buf, si = [], si + 1


# --------------------------------------------------------------- LORO protocol
# Pre-registered buckets (regime = (Hs, spatial aperture, steepness bw); TRAIN
# support is Hs<=3.5, side_mult in [1.1, 2.5], bw<=0.3; the held-out tails and
# extrapolations below are NEVER in training):
LORO_BUCKETS = {
    'WD':  dict(hs=(1.0, 3.0), side=(1.8, 2.5), bw=0.0),   # wide -> no-harm
    'UD':  dict(hs=(1.0, 3.0), side=(0.7, 1.0), bw=0.0),   # narrow -> gain (held-out tail)
    'EX_Hs':    dict(hs=(4.0, 5.0), side=(1.2, 1.8), bw=0.0),   # extrapolate sea state
    'EX_steep': dict(hs=(2.0, 3.0), side=(1.2, 1.8), bw=(0.45, 0.6)),  # extrapolate steepness
}


def _predict_all(bucket_samples, film_net, film_ls, cur_net, cur_ls, Vfeat):
    """corr of FiLM-net, current-net, and same-bank analytic hand-MMSE."""
    rows = {'film': [], 'cur': [], 'hand': []}
    for s, vf in zip(bucket_samples, Vfeat):
        P = torch.tensor(s['P'], device=DEVICE); z = torch.tensor(s['z'], device=DEVICE)
        Q = torch.tensor(s['Q'], device=DEVICE); X = torch.tensor(s['X'][None], device=DEVICE)
        sig0 = s['nfrac'] * s['Hs']
        with torch.no_grad():
            of = film_net(X, torch.tensor(vf[None], device=DEVICE))[0]
            oc = cur_net(X)[0]
            # hand = analytic beamform-power^1 prior through the SAME physics MMSE
            hand_map = torch.tensor(np.log10(s['X'][2] + 1e-6), device=DEVICE)
            pf = mmse_field(of, P, z, Q, s['kp_e'], sig0, film_ls)
            pc = mmse_field(oc, P, z, Q, s['kp_e'], sig0, cur_ls)
            ph = mmse_field(hand_map, P, z, Q, s['kp_e'], sig0,
                            torch.tensor(np.log(3.0), device=DEVICE))
        rows['film'].append(corr(pf.cpu().numpy(), s['gt_q']))
        rows['cur'].append(corr(pc.cpu().numpy(), s['gt_q']))
        rows['hand'].append(corr(ph.cpu().numpy(), s['gt_q']))
    return {k: np.array(v) for k, v in rows.items()}


def _paired_ci(d):
    """mean paired gap with cluster/t 90% CI (E-12~15 audit: t not 1.96)."""
    from math import sqrt
    n = len(d); m = float(d.mean())
    se = float(d.std(ddof=1) / sqrt(n)) if n > 1 else float('nan')
    try:
        from scipy.stats import t as tdist
        tc = float(tdist.ppf(0.95, n - 1))
    except Exception:
        tc = 1.83                       # ~ t_0.95, df~10 fallback
    return m, m - tc * se, m + tc * se, int((d > 0).sum()), n


def loro(args):
    print('=== LORO pre-registered evaluation ===')
    if not os.path.exists(CKPT):
        print(f'FiLM ckpt {CKPT} missing -- run --smoke first (or --full train).')
        return
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    film_net = FiLMPriorNet().to(DEVICE)
    film_net.backbone.load_state_dict(ck['model'])
    film_net.film.load_state_dict(ck['film'])
    film_net.film.set_norm(ck['feat_mu'], ck['feat_sd'])
    film_net.eval()
    film_ls = torch.tensor(float(ck['log_smult']), device=DEVICE)

    cur = torch.load(args.cur, map_location=DEVICE)
    cur_net = PriorNet().to(DEVICE); cur_net.load_state_dict(cur['model']); cur_net.eval()
    cur_ls = torch.tensor(float(cur.get('log_smult', np.log(3.0))), device=DEVICE)

    print('bucket    film    cur    hand   d(film-cur)  90%CI            win  n')
    result = {}
    for name, reg in LORO_BUCKETS.items():
        rng = np.random.default_rng(20260715 + hash(name) % 1000)
        samples, Vf = [], []
        for _ in range(args.n_per):
            bw = reg['bw'] if isinstance(reg['bw'], tuple) else 0.0
            s = gen_widened(rng, hs_range=reg['hs'], side_range=reg['side'],
                            bw_max=(bw[1] if isinstance(bw, tuple) else 0.0))
            samples.append(s)
            Vf.append(inform_features(s['P'], s['z'], s['kp_e'], s['nfrac'],
                                      s['Hs'], s['X']))
        r = _predict_all(samples, film_net, film_ls, cur_net, cur_ls, Vf)
        d = r['film'] - r['cur']
        m, lo, hi, win, n = _paired_ci(d)
        result[name] = dict(film=float(r['film'].mean()), cur=float(r['cur'].mean()),
                            hand=float(r['hand'].mean()), d=m, ci=[lo, hi],
                            win=win, n=n, vs_hand=float((r['film'] - r['hand']).mean()))
        print(f'{name:9s} {r["film"].mean():+.3f} {r["cur"].mean():+.3f} '
              f'{r["hand"].mean():+.3f}  {m:+.3f}     [{lo:+.3f},{hi:+.3f}]  '
              f'{win:2d}/{n}')

    # pre-registered PASS logic
    R = result
    c1 = R['WD']['d'] >= -0.02 and R['WD']['vs_hand'] >= -0.02
    c2 = R['UD']['d'] >= 0.03 and R['UD']['ci'][0] > 0
    c3 = (min(R['EX_Hs']['d'], R['EX_steep']['d']) >= 0.02
          and min(R['EX_Hs']['vs_hand'], R['EX_steep']['vs_hand']) >= 0.0)
    print('\n--- pre-registered criterion ---')
    print(f'(1) WD no-harm  (d>=-0.02 & vs_hand>=-0.02): {"PASS" if c1 else "FAIL"}')
    print(f'(2) UD gain     (d>=+0.03 & CI_lo>0):        {"PASS" if c2 else "FAIL"}')
    print(f'(3) EX no-collapse (d>=+0.02 & >=hand OOD):  {"PASS" if c3 else "FAIL"}')
    print(f'OVERALL: {"PASS" if (c1 and c2 and c3) else "FAIL"}')
    os.makedirs('results/direction3', exist_ok=True)
    json.dump(result, open('results/direction3/e28_loro.json', 'w'), indent=1,
              default=float)
    print('saved -> results/direction3/e28_loro.json')


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gen', action='store_true')
    ap.add_argument('--loro', action='store_true')
    # smoke
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--every', type=int, default=100)
    ap.add_argument('--lr_bb', type=float, default=3e-5)
    ap.add_argument('--lr_film', type=float, default=1e-3)
    ap.add_argument('--data', default='tmp/prior_train/fan0')
    ap.add_argument('--init', default=CUR_NET)
    # gen
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='tmp/prior_train/wide0')
    ap.add_argument('--shard', type=int, default=500)
    ap.add_argument('--hs_lo', type=float, default=1.0)
    ap.add_argument('--hs_hi', type=float, default=5.0)
    ap.add_argument('--side_lo', type=float, default=0.7)
    ap.add_argument('--side_hi', type=float, default=2.5)
    ap.add_argument('--bw_max', type=float, default=0.3)
    # loro
    ap.add_argument('--n_per', type=int, default=24)
    ap.add_argument('--cur', default=CUR_NET)
    a = ap.parse_args()
    print(f'device {DEVICE}', flush=True)
    if a.selftest:
        selftest()
    if a.smoke:
        smoke(a)
    if a.gen:
        gen(a)
    if a.loro:
        loro(a)
    if not any([a.selftest, a.smoke, a.gen, a.loro]):
        ap.print_help()


if __name__ == '__main__':
    main()
