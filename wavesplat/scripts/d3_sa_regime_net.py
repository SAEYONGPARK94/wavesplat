"""
d3_sa_regime_net.py — Regime-targeted track-aware detection net (10 Tp, 2 lam
patch, continuous spectra) evaluated against the calibrated ceiling.

Context (2026-07-04 continuous-spectrum calibration): on continuous JONSWAP
seas the right reference is the bin-class linear MMSE ~ oracle_ls2 (true top-32
k + ridge LS, lambda 1.0): fixed 0.28 / track043 0.41 / track100 0.46, while
RELAX sits flat at ~0.20 — an identification/estimation gap that GROWS with
platform speed (+0.08/+0.18/+0.28). This net attacks exactly that gap: learn
to locate the energetic spectral bins from the beamform map of THIS regime's
sampling operators (fixed disk + ping-pong tracks, 10 Tp windows), then ridge
LS on the detected bins.

Differences from d3_beamform_net: KLIM 0.25 (kp ~ 0.083 — 3x finer detection
grid around the peak), training operators = this regime's (not scatter), seas
= continuous nc=200 (not 5/15/30 discrete), T = 10 Tp.

Usage:
  python3 scripts/d3_sa_regime_net.py --steps 4000        # train + eval
  python3 scripts/d3_sa_regime_net.py --eval_only         # eval saved ckpt
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, true_field, corr,
                                     hashabs, TP, HS, DEPTH, NOISE_FRAC, N_OBS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
KLIM, NK = 0.25, 64
CKPT = 'results/direction3/regime_net.pt'
OUT = 'results/direction3/sa_regime_net_eval.json'


def kgrid():
    kx = torch.linspace(-KLIM, KLIM, NK, device=DEVICE)
    KX, KY = torch.meshgrid(kx, kx, indexing='xy')
    k = torch.sqrt(KX ** 2 + KY ** 2) + 1e-9
    om = torch.sqrt(G * k * torch.tanh(k * DEPTH))
    return KX, KY, om


KX_G, KY_G, OM_G = kgrid()


def beamform(P, z):
    kxf, kyf, omf = KX_G.reshape(-1), KY_G.reshape(-1), OM_G.reshape(-1)
    phi = (kxf[None, :, None] * P[:, None, :, 0]
           + kyf[None, :, None] * P[:, None, :, 1]
           - omf[None, :, None] * P[:, None, :, 2])
    resp = (z[:, None, :] * torch.exp(-1j * phi)).sum(-1)
    B = (resp.abs() ** 2).reshape(P.shape[0], NK, NK) / (P.shape[1] ** 2)
    return torch.log1p(B / (B.amax(dim=(1, 2), keepdim=True) + 1e-9) * 50.0)


def target_heatmap(kx, ky, w):
    """Vectorized over components (the per-component python loop was the
    training bottleneck: 1.85 s/step -> ~0.05 s/step)."""
    ii = ((torch.as_tensor(ky, device=DEVICE, dtype=torch.float32) + KLIM)
          / (2 * KLIM) * (NK - 1))
    jj = ((torch.as_tensor(kx, device=DEVICE, dtype=torch.float32) + KLIM)
          / (2 * KLIM) * (NK - 1))
    a = torch.as_tensor(w, device=DEVICE, dtype=torch.float32)
    ok = (ii >= 0) & (ii < NK) & (jj >= 0) & (jj < NK)
    if not ok.any():
        return torch.zeros(NK, NK, device=DEVICE)
    ii, jj, a = ii[ok], jj[ok], a[ok]
    yy = torch.arange(NK, device=DEVICE, dtype=torch.float32)
    gy = torch.exp(-0.5 * ((yy[None] - ii[:, None]) / 1.5) ** 2)   # (n, NK)
    gxx = torch.exp(-0.5 * ((yy[None] - jj[:, None]) / 1.5) ** 2)  # (n, NK)
    H = (a[:, None, None] * gy[:, :, None] * gxx[:, None, :]).amax(0)
    return H / (H.max() + 1e-9)


class DetectCNN(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        L = []
        c_in = 1
        for _ in range(6):
            L += [nn.Conv2d(c_in, ch, 3, padding=1), nn.GELU(),
                  nn.GroupNorm(8, ch)]
            c_in = ch
        L += [nn.Conv2d(ch, 1, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return torch.sigmoid(self.net(x[:, None])).squeeze(1)


def regime_geometry():
    lam = dispersion_lam(TP, DEPTH)
    cp = lam / TP
    T = 10 * TP
    fp = 0.18 * lam
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    return lam, cp, T, fp, side, gx, gy


def make_batch(Bsz):
    lam, cp, T, fp, side, gx, gy = regime_geometry()
    P = np.zeros((Bsz, N_OBS, 3), np.float32)
    Z = np.zeros((Bsz, N_OBS), np.float32)
    hm = []
    for b in range(Bsz):
        Tp = np.random.uniform(5.5, 8.5)
        Hs = np.random.uniform(1.0, 3.0)
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                               seed=int(np.random.randint(1e9)))
        kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
        A = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
        om = omega_of(kx, ky, DEPTH)
        rng = np.random.default_rng(np.random.randint(1e9))
        u = b % 3
        if u == 0:
            Pb = sample_obs('fixed', 0.0, lam, cp, T, gx, gy, fp / 2, rng)
        else:
            ur = float(np.random.uniform(0.15, 1.2))
            Pb = sample_obs('moving', ur * cp, lam, cp, T, gx, gy, fp / 2, rng)
            if u == 2:                                  # random heading
                cx = cy = side / 2
                th = np.random.uniform(0, np.pi)
                R = np.array([[np.cos(th), -np.sin(th)],
                              [np.sin(th), np.cos(th)]])
                Pb[:, :2] = (Pb[:, :2] - [cx, cy]) @ R.T + [cx, cy]
        psi = (kx[None] * Pb[:, 0:1] + ky[None] * Pb[:, 1:2]
               - om[None] * Pb[:, 2:3] + ph[None])
        Z[b] = ((A[None] * np.cos(psi)).sum(1)
                + np.random.normal(0, NOISE_FRAC * Hs, N_OBS))
        P[b] = Pb
        hm.append(target_heatmap(kx, ky, A ** 2))
    Bmap = beamform(torch.tensor(P, device=DEVICE),
                    torch.tensor(Z, device=DEVICE))
    return Bmap, torch.stack(hm)


def train(steps, Bsz=24, lr=3e-4):
    model = DetectCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        Bmap, H = make_batch(Bsz)
        Hp = model(Bmap)
        loss = (((Hp - H) ** 2) * (1 + 20 * H)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0 or s == steps - 1:
            print(f"  step {s:4d} loss {loss.item():.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model


def extract_peaks(Hp, n_max=32, thresh=0.1):
    from scipy.ndimage import maximum_filter
    H = Hp.detach().cpu().numpy()
    mx = maximum_filter(H, size=3)
    pk = np.argwhere((H == mx) & (H > thresh * H.max()))
    vals = H[pk[:, 0], pk[:, 1]]
    order = np.argsort(vals)[::-1][:n_max]
    pk = pk[order]
    kx = -KLIM + pk[:, 1] / (NK - 1) * 2 * KLIM
    ky = -KLIM + pk[:, 0] / (NK - 1) * 2 * KLIM
    return kx.astype(np.float32), ky.astype(np.float32)


@torch.no_grad()
def evaluate(model, n_seas=10):
    """Same seas/conditions/pixels as the continuous achievability run."""
    lam, cp, T, fp, side, gx, gy = regime_geometry()
    r_fp = fp / 2
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    if len(Xs) > 1500:
        take = np.random.default_rng(0).choice(len(Xs), 1500, replace=False)
        Xs, Ys = Xs[take], Ys[take]
    t_star = T
    conds = [('fixed', None), ('track043', 0.43), ('track100', 1.0)]
    res = {c: [] for c, _ in conds}
    for si in range(n_seas):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               seed=4000 + si)
        th_full = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
                   np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om = omega_of(th_full[0], th_full[1], DEPTH)
        gt_u = render_modes(*th_full, DEPTH, Xs, Ys, t_star)
        sigma = NOISE_FRAC * HS
        for cname, ur in conds:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (ur or 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                           cp, T, gx, gy, r_fp, rng)
            psi = (th_full[0][None] * P[:, 0:1] + th_full[1][None] * P[:, 1:2]
                   - om[None] * P[:, 2:3] + th_full[3][None])
            z = (th_full[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            Bmap = beamform(torch.tensor(P[None], device=DEVICE,
                                         dtype=torch.float32),
                            torch.tensor(z[None], device=DEVICE,
                                         dtype=torch.float32))
            kx, ky = extract_peaks(model(Bmap)[0])
            if len(kx) == 0:
                res[cname].append(0.0)
                continue
            obs = np.column_stack([P, z]).astype(np.float32)
            A_f, ph_f = fit_amplitudes_phases(kx, ky, obs, DEPTH,
                                              lambda_reg=1.0)
            f = render_modes(kx.astype(float), ky.astype(float),
                             np.asarray(A_f, float), np.asarray(ph_f, float),
                             DEPTH, Xs, Ys, t_star)
            res[cname].append(corr(f, gt_u))
    print("\n===== regime net vs calibrated references (continuous seas) =====")
    ref = json.load(open('results/direction3/sa_achievability_gap_cont.json'))
    print(f"{'cond':10s} {'net2':>8s} {'relax':>8s} {'oracle_ls2':>11s}")
    summary = {}
    for cname, _ in conds:
        v = float(np.mean(res[cname]))
        summary[cname] = dict(net2=v,
                              relax=ref['summary'][cname]['relax'],
                              oracle_ls2=ref['summary'][cname]['oracle_ls2'])
        print(f"{cname:10s} {v:>8.3f} {ref['summary'][cname]['relax']:>8.3f} "
              f"{ref['summary'][cname]['oracle_ls2']:>11.3f}")
    json.dump(dict(summary=summary, rows=res), open(OUT, 'w'), indent=1,
              default=float)
    print(f"saved {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--eval_only', action='store_true')
    args = ap.parse_args()
    if args.eval_only:
        model = DetectCNN().to(DEVICE)
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        model.eval()
    else:
        model = train(args.steps)
        torch.save(model.state_dict(), CKPT)
        print(f"saved {CKPT}")
        model.eval()
    evaluate(model)


if __name__ == '__main__':
    main()
