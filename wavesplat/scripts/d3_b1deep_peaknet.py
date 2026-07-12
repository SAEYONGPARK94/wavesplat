"""B-1 DEEP -- learned peak identifier (PSF-inversion hypothesis).
Frozen prereg: docs/B1B2_DEEP_prereg_2026-07-12.md ("B-1 심화" section).

New hypothesis (opened by the v2 refine-SELECT FAIL): the holdout residual is a
per-candidate local statistic whose margin collapses at low SNR, but the
SIDELOBE GEOMETRY is a deterministic function of the sampling pattern -- a
learner that sees the WHOLE beamform power map can invert the PSF and pick the
true peak. Distinct from the closed cells: the negative arm-selection learning
FAILED on real palm-picks; this is supervised learning on EXACT synthetic
labels (analytic single-component scenes) -- an unjudged cell.

Design (FROZEN):
  train data : analytic 1-comp scenes, unlimited on-the-fly.
               k ~ log-uniform(0.05, 1.2), direction uniform, amplitude from
               Hs 0.2-1.0 (Hs/2) with a log-uniform low-SNR emphasis, phase
               uniform, observations = a7v2 stratified geometry (reused from a
               NON-BF2 pool) + 2 cm noise.  label = true (kx, ky).
               >=30,000 scenes, >=20,000 steps, convergence confirmed.
  model      : kp-normalised beamform power map (51x51, existing recipe) ->
               small CNN -> true-peak (kx/kp, ky/kp) soft-argmax regression
               + confidence head (is the true peak the beamform top-1?).
               <=1M params.  NO BF2-specific info (k distribution is broadband
               log-uniform, geometry pool excludes BF2).
  judge path : learned k_hat -> a7v2 continuous-k refine + fine atom bank
               (existing plumbing) -> recovery scoring.  fire = confidence gate
               calibrated on 1comp-vs-5comp (NON-BF2), like a7v2.

Gates (FROZEN):
  G-B1D-1 identification : BF2 5 seeds |dk|/k_true <= 0.05 in >=3/5
                           (residual-select v2 was 2/5).
  G-B1D-2 recovery       : identified-seed field corr(lead0) & encounter rho
                           >= 0.5, in >=3/5 (via the mandated fine-bank path).
  G-B1D-3 harmless       : confidence gate -> 6 synth dfield >= -0.01 AND YS17
                           12 windows dencounter >= -0.02.
  FAIL branch            : "global PSF information also cannot identify at low
                           SNR = BF2 floor is an information limit" (triple lock).

python3 scripts/d3_b1deep_peaknet.py               # train (if no ckpt) + eval
python3 scripts/d3_b1deep_peaknet.py --smoke       # tiny pool/steps, skip real
python3 scripts/d3_b1deep_peaknet.py --eval_only   # load ckpt, run gates
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, DEV, build_bank, beamform_prior, design, G
from d3_a7_finebank import (load_synth, build_queries, field_track_corr,
                            solve_mmse_bank, hand_predict, APP_EPISODES,
                            U_SHIP, LEADS)
from d3_a7v2_refine import (augment_finebank_v2, _single_fit, _eval_resid,
                            _refine_single)

OUT = 'results/direction3/b1deep_eval.json'
CKPT = 'results/direction3/b1deep_peaknet.pt'
DATE = '2026-07-12'
NGRID = 51                     # beamform map resolution (frozen: 51x51)
B_HI = 2.5                     # map spans [-2.5 kp, 2.5 kp]^2 (a7v2 recipe)
NOISE = 0.02                   # 2 cm LiDAR noise (frozen)
K_LO, K_HI = 0.05, 1.2         # log-uniform true |k| (frozen)
CONF_TOL = 0.20                # true-peak-is-top1 label tolerance (norm units)
AMP_LO, AMP_HI = 0.03, 1.0     # log-uniform down-scale of Hs/2 (low-SNR emphasis)
N_TRAIN = 40000                # >= 30k scenes (frozen minimum)
N_VAL = 4000
STEPS = 20000                  # >= 20k (frozen); extended if not converged
BATCH = 128
LR = 1.5e-3

# geometry pool: NON-BF2 fixedT episodes only (no BF2-specific info)
POOL_GLOBS = [
    'data/episode_BF3_1comp_seed*_fixedT.h5',
    'data/episode_BF3_2comp_seed*_fixedT.h5',
    'data/episode_BF3_5comp_seed[0-4]_fixedT.h5',
    'data/episode_BF4_1comp_seed*_fixedT.h5',
    'data/episode_BF4_5comp_seed[0-4]_fixedT.h5',
    'data/episode_BF4_15comp_seed[0-4]_fixedT.h5',
]
CAL_POS = ['data/episode_BF3_1comp_seed%d_fixedT.h5',
           'data/episode_BF4_1comp_seed%d_fixedT.h5']
CAL_NEG = ['data/episode_BF3_5comp_seed%d_fixedT.h5',
           'data/episode_BF4_5comp_seed%d_fixedT.h5']

# normalised-k coordinate grids (constant; indexing='xy' matches beamform_grid)
_axn = torch.linspace(-B_HI, B_HI, NGRID, device=DEV)
_KXn, _KYn = torch.meshgrid(_axn, _axn, indexing='xy')
_KN = torch.hypot(_KXn, _KYn)
_INBAND = (_KN > 0.35) & (_KN < 2.5)          # (NGRID,NGRID) bool


# =============================================================== beamform map
def power_map(Pt, zt, kp, depth, n=NGRID, b_hi=B_HI):
    """kp-normalised beamform power map, out-of-band (0.35-2.5 kp) zeroed.
    Byte-identical math to a7v2.beamform_grid; returns an (n,n) GPU tensor."""
    ax = torch.linspace(-b_hi * kp, b_hi * kp, n, device=DEV)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy')
    kx, ky = KX.reshape(-1), KY.reshape(-1)
    k = torch.hypot(kx, ky)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-12)
    x, y, t = Pt[:, 0], Pt[:, 1], Pt[:, 2]
    pw = torch.empty_like(kx)
    B = 4096
    for i in range(0, len(kx), B):
        ps = (kx[i:i + B, None] * x[None] + ky[i:i + B, None] * y[None]
              - om[i:i + B, None] * t[None])
        pw[i:i + B] = ((zt[None] * torch.cos(ps)).sum(1) ** 2
                       + (zt[None] * torch.sin(ps)).sum(1) ** 2)
    inband = (k > 0.35 * kp) & (k < 2.5 * kp)
    pw = torch.where(inband, pw, torch.zeros_like(pw))
    return pw.reshape(n, n)


def to_input(pw):
    """(n,n) raw power -> (2,n,n) CNN input: max-normalised power + its sqrt
    (dynamic-range compression that emphasises sidelobes)."""
    m = pw.max().clamp_min(1e-12)
    p = pw / m
    return torch.stack([p, torch.sqrt(p)], 0)


def argmax_nk(pw):
    """normalised (kx/kp, ky/kp) of the in-band beamform top-1 cell."""
    j = int(torch.argmax(pw.reshape(-1)))
    return float(_KXn.reshape(-1)[j]), float(_KYn.reshape(-1)[j])


# =========================================================== analytic scenes
def load_geometry_pool(globs=POOL_GLOBS, cap=None):
    """(x,y,t) stratified LiDAR geometry from NON-BF2 fixedT episodes; wave is
    synthesised analytically on top, so only the sampling geometry is reused."""
    paths = sorted({p for g in globs for p in glob.glob(g)})
    if cap:
        paths = paths[:cap]
    pool = []
    for p in paths:
        ep = load_synth(p, np.random.default_rng(11))
        pool.append(dict(
            P=torch.tensor(ep['P'], device=DEV, dtype=torch.float32),
            depth=float(ep['depth']), n=len(ep['z']), path=os.path.basename(p)))
    return pool


def gen_scene(pool, rng):
    """one analytic 1-comp scene: eta = A cos(k.x - w(k)t + phi) + noise 2cm."""
    geo = pool[rng.integers(len(pool))]
    P, depth = geo['P'], geo['depth']
    kmag = float(math.exp(rng.uniform(math.log(K_LO), math.log(K_HI))))
    ang = float(rng.uniform(0, 2 * math.pi))
    kx, ky = kmag * math.cos(ang), kmag * math.sin(ang)
    Hs = float(rng.uniform(0.2, 1.0))
    # low-SNR emphasis: log-uniform down-scale of the nominal Hs/2 amplitude
    # (frozen "저SNR 강조": tail reaches amplitude ~ noise so the main lobe is
    #  driven sub-dominant to sidelobes, matching the BF2 buried-peak regime)
    A = (Hs / 2.0) * float(math.exp(rng.uniform(math.log(AMP_LO),
                                                math.log(AMP_HI))))
    phi = float(rng.uniform(0, 2 * math.pi))
    om = math.sqrt(G * kmag * math.tanh(kmag * depth) + 1e-12)
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    z = A * torch.cos(kx * x + ky * y - om * t + phi)
    z = z + NOISE * torch.randn(len(z), device=DEV)
    pw = power_map(P, z, kmag, depth)
    inp = to_input(pw)
    tnx, tny = math.cos(ang), math.sin(ang)          # true nk (radius 1)
    anx, any_ = argmax_nk(pw)
    conf = 1.0 if math.hypot(anx - tnx, any_ - tny) < CONF_TOL else 0.0
    return inp, (tnx, tny), conf


def build_dataset(pool, n, seed, tag):
    """precompute n scenes -> CPU tensors (inputs fp16, targets fp32)."""
    rng = np.random.default_rng(seed)
    X = torch.empty((n, 2, NGRID, NGRID), dtype=torch.float16)
    Y = torch.empty((n, 2), dtype=torch.float32)
    C = torch.empty((n,), dtype=torch.float32)
    t0 = time.time()
    for i in range(n):
        inp, tnk, conf = gen_scene(pool, rng)
        X[i] = inp.half().cpu()
        Y[i] = torch.tensor(tnk)
        C[i] = conf
        if (i + 1) % 10000 == 0:
            print(f"  [{tag}] {i+1}/{n}  ({(time.time()-t0):.0f}s, "
                  f"conf+ {float(C[:i+1].mean()):.2f})", flush=True)
    return X, Y, C


# ================================================================== the net
class PeakNet(nn.Module):
    """dilated conv stack (medium-range PSF context) + global-context fusion ->
    masked spatial soft-argmax head (peak position) + confidence head."""
    def __init__(self, ch=48):
        super().__init__()
        self.c0 = nn.Conv2d(2, ch, 3, padding=1)
        self.b0 = nn.BatchNorm2d(ch)
        dil = [1, 2, 4, 8]
        self.convs = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, padding=d, dilation=d) for d in dil])
        self.bns = nn.ModuleList([nn.BatchNorm2d(ch) for _ in dil])
        self.gfc = nn.Linear(ch, ch)                 # global context
        self.head = nn.Conv2d(ch, 1, 1)              # heatmap logits
        self.conf = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(),
                                  nn.Linear(ch, 1))   # confidence
        self.register_buffer('KXn', _KXn.clone())
        self.register_buffer('KYn', _KYn.clone())
        self.register_buffer('mask', _INBAND.clone())

    def forward(self, x):
        h = F.relu(self.b0(self.c0(x)))
        for c, b in zip(self.convs, self.bns):
            h = F.relu(b(c(h)) + h)                  # residual dilated blocks
        g = h.mean(dim=(2, 3))                        # (B,ch) global pool
        h = h + torch.tanh(self.gfc(g))[:, :, None, None]
        logit = self.head(h).squeeze(1)              # (B,n,n)
        logit = logit.masked_fill(~self.mask[None], -1e9)
        Bn = logit.shape[0]
        w = F.softmax(logit.reshape(Bn, -1), dim=1).reshape(Bn, NGRID, NGRID)
        pnx = (w * self.KXn[None]).sum(dim=(1, 2))
        pny = (w * self.KYn[None]).sum(dim=(1, 2))
        conf = self.conf(g).squeeze(1)
        return torch.stack([pnx, pny], 1), logit, conf


def true_cell_index(Y):
    """nearest in-band grid cell index (flattened) to each true nk in Y."""
    kxn = _KXn.reshape(-1)[None]                      # (1, n*n)
    kyn = _KYn.reshape(-1)[None]
    d2 = (kxn - Y[:, 0:1]) ** 2 + (kyn - Y[:, 1:2]) ** 2
    d2 = d2.masked_fill(~_INBAND.reshape(-1)[None], 1e9)
    return d2.argmin(1)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


# ==================================================================== train
def train(pool, steps=STEPS, log=print):
    log(f"building train pool ({N_TRAIN}) + val ({N_VAL}) ...")
    Xtr, Ytr, Ctr = build_dataset(pool, N_TRAIN, 100, 'train')
    Xva, Yva, Cva = build_dataset(pool, N_VAL, 999, 'val')
    Yva_d = Yva.to(DEV)
    net = PeakNet().to(DEV)
    log(f"PeakNet params = {n_params(net):,} (<=1M gate {'OK' if n_params(net)<=1e6 else 'FAIL'})")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(7)

    def evaluate():
        net.eval()
        errs, confok = [], []
        with torch.no_grad():
            for i in range(0, N_VAL, 512):
                xb = Xva[i:i + 512].float().to(DEV)
                pnk, _, conf = net(xb)
                e = torch.hypot(pnk[:, 0] - Yva_d[i:i + 512, 0],
                                pnk[:, 1] - Yva_d[i:i + 512, 1])
                errs.append(e.cpu())
                confok.append(((conf > 0) == (Cva[i:i + 512].to(DEV) > 0.5)).cpu())
        net.train()
        errs = torch.cat(errs)
        return dict(med_dnk=float(errs.median()),
                    id05=float((errs <= 0.05).float().mean()),
                    id10=float((errs <= 0.10).float().mean()),
                    conf_acc=float(torch.cat(confok).float().mean()))

    curve = []
    best = None
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = rng.integers(0, N_TRAIN, BATCH)
        xb = Xtr[idx].float().to(DEV)
        yb = Ytr[idx].to(DEV)
        cb = Ctr[idx].to(DEV)
        pnk, logit, conf = net(xb)
        tcell = true_cell_index(yb)
        l_reg = F.mse_loss(pnk, yb)
        l_heat = F.cross_entropy(logit.reshape(BATCH, -1), tcell)
        l_conf = F.binary_cross_entropy_with_logits(conf, cb)
        loss = l_reg + 0.2 * l_heat + 0.3 * l_conf
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 500 == 0 or s == 1:
            m = evaluate()
            curve.append(dict(step=s, id05=m['id05'], id10=m['id10'],
                              med_dnk=m['med_dnk'], conf_acc=m['conf_acc'],
                              loss=float(loss)))
            if best is None or m['id05'] > best[0]:
                best = (m['id05'], {k: v.detach().cpu().clone()
                                    for k, v in net.state_dict().items()})
            log(f"  step {s:5d}  loss {float(loss):.4f}  val id@.05 "
                f"{m['id05']:.3f} id@.10 {m['id10']:.3f} med|dnk| "
                f"{m['med_dnk']:.3f} confacc {m['conf_acc']:.3f} "
                f"({(time.time()-t0)/60:.1f}m)", flush=True)
    # restore best-id05 weights
    net.load_state_dict(best[1])
    return net, curve, dict(best_id05=best[0])


def converged(curve, tail=6):
    """last `tail` (=3k steps @500) val id@.05 improved < 2% of its level."""
    if len(curve) < tail + 1:
        return False
    a = curve[-tail - 1]['id05']
    b = curve[-1]['id05']
    return (b - a) <= 0.02 * max(a, 1e-6)


# ============================================================== identify path
def peaknet_identify(net, P, z, kp, depth):
    """learned k_hat (denormalised) + confidence from the beamform map."""
    Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    pw = power_map(Pt, zt, kp, depth)
    inp = to_input(pw)[None]
    net.eval()
    with torch.no_grad():
        pnk, _, conf = net(inp)
    nkx, nky = float(pnk[0, 0]), float(pnk[0, 1])
    prob = float(torch.sigmoid(conf)[0])
    anx, any_ = argmax_nk(pw)
    return dict(khat=(nkx * kp, nky * kp), nk=(nkx, nky), prob=prob,
                argmax_nk=(anx, any_), argmax_kabs=float(math.hypot(anx, any_) * kp))


def peaknet_predict(net, P, z, depth, Q, sigma0, kp, theta, force=False):
    """identify -> refine -> (fire) fine-bank solve / (else) hand path."""
    idd = peaknet_identify(net, P, z, kp, depth)
    fired = force or (idd['prob'] >= theta)
    kx0, ky0 = idd['khat']
    rx, ry = _refine_single(kx0, ky0, P, z, depth)     # a7v2 continuous refine
    idd['refined'] = (rx, ry)
    Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
    if not fired:
        pred = mmse_gpu(Pt, zt, depth, Qt, sigma0, kp_hint=kp).cpu().numpy()
        return pred, idd, False, dict(n_fine=0)
    kxb2, kyb2, om2, pw2, info = augment_finebank_v2(Pt, zt, depth, kp, rx, ry)
    pred = solve_mmse_bank(Pt, zt, depth, Qt, sigma0, kxb2, kyb2, om2,
                           pw2).cpu().numpy()
    return pred, idd, True, info


def single_forecast(kx, ky, P, z, depth, Q):
    """diagnostic recovery: pure single dispersion-constrained plane wave from
    the identified (refined) k, forecast on Q (NOT the mandated fine-bank)."""
    coef, _ = _single_fit(kx, ky, P, z, depth)
    k = math.hypot(kx, ky)
    om = math.sqrt(G * k * math.tanh(k * depth) + 1e-12)
    psi = kx * Q[:, 0] + ky * Q[:, 1] - om * Q[:, 2]
    return coef[0] * np.cos(psi) + coef[1] * np.sin(psi)


# ============================================================= calibrate fire
def calibrate_conf(net):
    """confidence threshold on NON-BF2 1comp (fire) vs 5comp (no-fire), Youden;
    ties -> largest theta (harmless-favouring), exactly like a7v2."""
    def prob_of(path):
        ep = load_synth(path, np.random.default_rng(11))
        return peaknet_identify(net, ep['P'], ep['z'], ep['kp'], ep['depth'])['prob']
    pos = np.array([prob_of(p % s) for p in CAL_POS for s in range(5)])
    neg = np.array([prob_of(p % s) for p in CAL_NEG for s in range(5)])
    cands = np.unique(np.concatenate([pos, neg]))
    mids = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
    best = None
    for th in mids:
        tpr = float((pos >= th).mean()); fpr = float((neg >= th).mean())
        j = tpr - fpr
        if best is None or j > best[0] + 1e-12 or (abs(j - best[0]) < 1e-12
                                                   and th > best[1]):
            best = (j, float(th), tpr, fpr)
    j, theta, tpr, fpr = best
    return dict(theta=theta, youden_j=j, tpr_1comp=tpr, fpr_5comp=fpr,
                pos=pos.tolist(), neg=neg.tolist())


# =================================================================== gate 1/2
def gate_bf2(net, theta):
    rows = []
    for seed in range(5):
        ep = load_synth('data/episode_BF2_1comp_seed%d_fixedT.h5' % seed,
                        np.random.default_rng(11))
        tk = ep['kp'] * ep['khat']
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        pn, idd, fired, info = peaknet_predict(net, ep['P'], ep['z'],
                                               ep['depth'], Q, sig0, ep['kp'],
                                               theta)
        pf, _, _, finfo = peaknet_predict(net, ep['P'], ep['z'], ep['depth'],
                                          Q, sig0, ep['kp'], theta, force=True)
        rx, ry = idd['refined']
        nkx, nky = idd['nk']
        # identification error: refined k (fed to fine bank) & raw CNN k
        dk_refined = float(math.hypot(rx - tk[0], ry - tk[1]) / ep['kp'])
        dk_cnn = float(math.hypot(nkx * ep['kp'] - tk[0],
                                  nky * ep['kp'] - tk[1]) / ep['kp'])
        dk_argmax = float(math.hypot(idd['argmax_nk'][0] * ep['kp'] - tk[0],
                                     idd['argmax_nk'][1] * ep['kp'] - tk[1])
                          / ep['kp'])
        ch, rh, _ = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, rn, _ = field_track_corr(pn, ep, npix, truth_tr, nf)
        cf, rf, _ = field_track_corr(pf, ep, npix, truth_tr, nf)
        # diagnostic recovery: single plane wave from the identified refined k
        ps = single_forecast(rx, ry, ep['P'], ep['z'], ep['depth'], Q)
        cs, rs, _ = field_track_corr(ps, ep, npix, truth_tr, nf)
        rows.append(dict(
            seed=seed, true_kabs=float(ep['kp']), true_nk=ep['khat'].tolist(),
            prob=idd['prob'], fired=fired,
            argmax_kabs=idd['argmax_kabs'], dk_argmax=dk_argmax,
            cnn_nk=[nkx, nky], dk_cnn=dk_cnn,
            refined=[rx, ry], dk_refined=dk_refined,
            identified=bool(dk_refined <= 0.05),
            hand_f0=ch[0], hand_rho=rh,
            fire_f0=cn[0], fire_rho=rn,
            forced_f0=cf[0], forced_rho=rf,
            single_f0=cs[0], single_rho=rs,
            hand_field=ch, forced_field=cf, single_field=cs))
        print(f"[BF2 s{seed}] prob {idd['prob']:.3f} fire {fired}  argmax|k| "
              f"{idd['argmax_kabs']:.2f} -> CNN dk {dk_cnn:.3f} -> refined dk "
              f"{dk_refined:.3f} {'ID' if dk_refined<=0.05 else '  '}  "
              f"(true {ep['kp']:.2f})  fine f0/rho {cn[0]:+.2f}/{rn:+.2f}  "
              f"single f0/rho {cs[0]:+.2f}/{rs:+.2f}", flush=True)
    n_id = int(sum(r['identified'] for r in rows))
    fire = np.array([(r['fire_f0'] >= 0.5) and (r['fire_rho'] >= 0.5)
                     for r in rows])
    forced = np.array([(r['forced_f0'] >= 0.5) and (r['forced_rho'] >= 0.5)
                       for r in rows])
    single = np.array([(r['single_f0'] >= 0.5) and (r['single_rho'] >= 0.5)
                       for r in rows])
    return rows, dict(
        n_identified=n_id, id_pass=n_id >= 3,
        n_recover_finebank=int(fire.sum()),
        n_recover_forced=int(forced.sum()),
        n_recover_single=int(single.sum()),
        recover_pass=int(fire.sum()) >= 3)


def gate_harmless_synth(net, theta):
    rows = []
    for fname in APP_EPISODES:
        ep = load_synth('data/' + fname, np.random.default_rng(11))
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        pn, idd, fired, info = peaknet_predict(net, ep['P'], ep['z'],
                                               ep['depth'], Q, sig0, ep['kp'],
                                               theta)
        ch, _, _ = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, _, _ = field_track_corr(pn, ep, npix, truth_tr, nf)
        dfield = [cn[k] - ch[k] for k in range(len(LEADS))]
        rows.append(dict(episode=fname, sea=ep['sea'], ncomp=ep['ncomp'],
                         prob=idd['prob'], fired=fired, dfield=dfield,
                         dfield_min=float(min(dfield))))
        print(f"[harm synth {fname}] prob {idd['prob']:.3f} fire {fired}  "
              f"dfield {[f'{d:+.3f}' for d in dfield]}", flush=True)
    return rows


def gate_harmless_real(net, theta, smoke=False):
    if smoke:
        return [], dict(skipped=True)
    from d3_sa_multirecord import MRecord
    from d3_gate_promotion_eval import window_data
    from d3_falcon_real_operator import RECORDS
    name = 'YS17'
    cfg = RECORDS[name]
    print(f"[harm real] loading {name} ...", flush=True)
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
        pn, idd, fired, info = peaknet_predict(net, P, z, cfg['depth'], Pq,
                                               sig0, kp, theta)
        ch = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
        cn = float(np.corrcoef(pn[v], gt_q[v])[0, 1])
        rows.append(dict(record=name, wi=wi, prob=idd['prob'], fired=fired,
                         hand_enc=ch, fire_enc=cn, denc=cn - ch))
        print(f"[harm real {name} w{wi}] prob {idd['prob']:.3f} fire {fired}  "
              f"hand {ch:+.2f} fire {cn:+.2f}  d {cn-ch:+.3f}", flush=True)
    d = np.array([r['denc'] for r in rows])
    return rows, dict(n=len(rows), min_denc=float(d.min()) if len(d) else None,
                      mean_denc=float(d.mean()) if len(d) else None)


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--skip_real', action='store_true')
    a = ap.parse_args()
    global N_TRAIN, N_VAL, STEPS, POOL_GLOBS
    if a.smoke:
        N_TRAIN, N_VAL, STEPS = 3000, 500, 800
        POOL_GLOBS = ['data/episode_BF3_1comp_seed*_fixedT.h5',
                      'data/episode_BF4_5comp_seed[0-1]_fixedT.h5']
    t0 = time.time()
    print(f"device {DEV}  smoke={a.smoke}", flush=True)

    if a.eval_only and os.path.exists(CKPT):
        net = PeakNet().to(DEV)
        net.load_state_dict(torch.load(CKPT, map_location=DEV))
        curve, tinfo, converged_flag = [], {'loaded': True}, True
        print(f"loaded ckpt {CKPT}", flush=True)
    else:
        pool = load_geometry_pool()
        print(f"geometry pool: {len(pool)} episodes, nobs "
              f"{min(g['n'] for g in pool)}-{max(g['n'] for g in pool)}",
              flush=True)
        net, curve, tinfo = train(pool, STEPS)
        converged_flag = converged(curve)
        print(f"convergence (last 3k val id@.05 gain <2%): {converged_flag}",
              flush=True)
        if not converged_flag and not a.smoke:
            print("--- not converged: extending +10k steps (prereg) ---",
                  flush=True)
            net2, curve2, tinfo2 = train(pool, 10000)
            # keep the better final model
            if tinfo2['best_id05'] >= tinfo['best_id05']:
                net, curve_ext = net2, curve2
            else:
                curve_ext = curve2
            curve = curve + [{**c, 'step': c['step'] + STEPS} for c in curve_ext]
            converged_flag = converged(curve)
            tinfo['best_id05'] = max(tinfo['best_id05'], tinfo2['best_id05'])
        torch.save(net.state_dict(), CKPT)
        print(f"saved ckpt {CKPT}", flush=True)

    print("\n--- confidence-gate calibration (1comp vs 5comp, non-BF2) ---",
          flush=True)
    cal = calibrate_conf(net)
    theta = cal['theta']
    print(f"pos(1comp) prob {['%.2f' % v for v in cal['pos']]}")
    print(f"neg(5comp) prob {['%.2f' % v for v in cal['neg']]}")
    print(f"theta {theta:.4f}  Youden {cal['youden_j']:.2f}  TPR "
          f"{cal['tpr_1comp']:.2f} FPR {cal['fpr_5comp']:.2f}", flush=True)

    print("\n--- G-B1D-1 identification / G-B1D-2 recovery (BF2 x5) ---",
          flush=True)
    g1_rows, g1 = gate_bf2(net, theta)
    fire_rate = float(np.mean([r['fired'] for r in g1_rows]))
    print(f"G-B1D-1 identification: {g1['n_identified']}/5 |dk|/k<=0.05  "
          f"({'PASS' if g1['id_pass'] else 'FAIL'})", flush=True)
    print(f"G-B1D-2 recovery(finebank): {g1['n_recover_finebank']}/5  "
          f"(forced {g1['n_recover_forced']}/5, single-comp diag "
          f"{g1['n_recover_single']}/5)  "
          f"{'PASS' if g1['recover_pass'] else 'FAIL'}", flush=True)

    print("\n--- G-B1D-3 harmless: synth 6 ---", flush=True)
    g3s = gate_harmless_synth(net, theta)
    g3s_min = float(min(d for r in g3s for d in r['dfield']))
    print(f"synth min dfield {g3s_min:+.4f}  "
          f"({'PASS' if g3s_min >= -0.01 else 'FAIL'} @ -0.01)", flush=True)

    print("\n--- G-B1D-3 harmless: real YS17 12 windows ---", flush=True)
    smoke_real = a.smoke or a.skip_real
    g3r, g3r_sum = gate_harmless_real(net, theta, smoke=smoke_real)
    if not smoke_real:
        real_min = g3r_sum['min_denc']
        g3_pass = (g3s_min >= -0.01) and (real_min is not None
                                          and real_min >= -0.02)
        print(f"real min denc {real_min}  "
              f"({'PASS' if real_min is not None and real_min >= -0.02 else 'FAIL'}"
              f" @ -0.02)", flush=True)
    else:
        real_min = None
        g3_pass = g3s_min >= -0.01

    out = dict(
        date=DATE, prereg='docs/B1B2_DEEP_prereg_2026-07-12.md (B-1 심화)',
        method='learned peak identifier (PSF inversion)',
        design=dict(ngrid=NGRID, b_hi=B_HI, noise=NOISE, k_band=[K_LO, K_HI],
                    amp_downscale=[AMP_LO, AMP_HI],
                    n_train=N_TRAIN, steps_planned=STEPS, batch=BATCH,
                    conf_tol=CONF_TOL, n_params=n_params(net),
                    geometry_pool='NON-BF2 fixedT (BF3/BF4)',
                    note='k_hat -> a7v2 refine + fine bank (mandated recovery); '
                         'single-comp forecast reported as recovery diagnostic'),
        training=dict(best_id05=tinfo.get('best_id05'),
                      converged=converged_flag, curve=curve),
        conf_calibration=cal,
        gate1_bf2=dict(summary=g1, fire_rate=fire_rate, seeds=g1_rows),
        gate3_harmless=dict(synth=g3s, synth_min_dfield=g3s_min,
                            real=g3r, real_summary=g3r_sum, passed=g3_pass),
        gates=dict(
            G_B1D_1_identification=dict(passed=g1['id_pass'],
                                        n_identified=g1['n_identified'],
                                        criterion='|dk|/k_true<=0.05 in >=3/5'),
            G_B1D_2_recovery=dict(passed=g1['recover_pass'],
                                  n_recover_finebank=g1['n_recover_finebank'],
                                  n_recover_forced=g1['n_recover_forced'],
                                  n_recover_single_diag=g1['n_recover_single'],
                                  criterion='field0>=0.5 AND rho>=0.5 in >=3/5 '
                                            '(mandated fine-bank path)'),
            G_B1D_3_harmless=dict(passed=g3_pass, synth_min_dfield=g3s_min,
                                  real_min_denc=real_min)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n================= GATES =================", flush=True)
    print(f"G-B1D-1 identification : {'PASS' if g1['id_pass'] else 'FAIL'}  "
          f"({g1['n_identified']}/5 |dk|/k<=0.05; v2 was 2/5)")
    print(f"G-B1D-2 recovery       : {'PASS' if g1['recover_pass'] else 'FAIL'}"
          f"  ({g1['n_recover_finebank']}/5 fine-bank; single-comp diag "
          f"{g1['n_recover_single']}/5; oracle-k fine-bank ceiling 1/5)")
    print(f"G-B1D-3 harmless       : {'PASS' if g3_pass else 'FAIL'}  "
          f"(synth min dfield {g3s_min:+.4f}"
          f"{'' if smoke_real else f', real min denc {real_min}'})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
