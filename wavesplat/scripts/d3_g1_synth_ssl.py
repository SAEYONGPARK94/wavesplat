"""G1: self-supervised pretraining on thousands of synthetic seas.

W7 pretrained on the 42 GB stereo archive and a linear probe read the RECORD's
excess kurtosis at R^2 = 0.76. Leave-one-record-out destroyed it (R^2 = -0.60):
the encoder had learned to recognise which of 13 records a patch came from and
look up its kurtosis. Thirteen seas is the sample size, and no number of
gradient steps manufactures a fourteenth.

Waves, unlike stereo archives, can be made to order. Second-order Stokes gives
an exact bound-wave field and therefore a sea whose non-Gaussianity is known by
construction and dialled by steepness. Thousands of distinct seas make
leave-sea-out a real test, and fingerprinting impossible.

THE TASK. Label = the sea's excess kurtosis, estimated from a 32-frame temporal
ensemble it never sees. Input = ONE standardized 64x64 frame. The patch spans
only 3-4 wavelengths, so a single frame's spatial kurtosis is a noisy estimate
of the sea's -- there is real room to do better by reading crest sharpening
rather than counting a fourth moment.

THE BASELINES, because eight of yesterday's retractions were starved baselines:
  plug-in       the input frame's own sample excess kurtosis, no parameters
  random-init   the same architecture, untrained (random-feature basis)
Both are fit and scored exactly like the SSL probe.

PRE-REGISTERED GATE: on held-out seas, SSL features must beat BOTH baselines by
dR^2 >= 0.05. Anything less means the representation added nothing over simply
computing the kurtosis of what you already measured, and CL1 needs no encoder.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, '..')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.stokes2 import Stokes2Transfer
from d3_sa_fim_identifiability import omega_of
from d3_ssl_pretrain import MAE, excess_kurtosis

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/g1_synth_ssl.json'
CKPT = 'results/direction3/g1_synth_ssl.pt'
CACHE = 'tmp/g1_seas.npz'

G = 9.81
DEPTH = 45.0
NG = 64                 # 64 x 64 field
L = 200.0               # m, so 3-4 wavelengths across the patch
NC = 48
N_IN = 8                # input frames per sea
N_LAB = 32              # label frames per sea, disjoint in time from the inputs
PATCH = 64


EPS_LO, EPS_HI = 0.02, 0.22     # k_p Hs / 2; second order is a fiction above


def make_sea(rng):
    """One sea: free JONSWAP surface + exact second-order bound waves.

    Hs, Tp and the steepness dial are drawn jointly under a physical steepness
    constraint. Without it, Hs = 8 m at Tp = 5 s gives eps = 0.64 -- past
    breaking, where the second-order expansion is not an approximation of
    anything and dkurt runs to +2.7.
    """
    for _ in range(200):
        Hs = rng.uniform(1.0, 8.0)
        Tp = rng.uniform(5.0, 13.0)
        s = rng.uniform(0.7, 1.8)                 # extra steepness dial
        eps = ((2 * np.pi / Tp) ** 2 / G) * (s * Hs) / 2
        if EPS_LO <= eps <= EPS_HI:
            break
    else:
        raise RuntimeError('steepness rejection sampler failed')
    comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=NC, depth=DEPTH,
                               direction_deg=rng.uniform(0, 360),
                               direction_spread_deg=rng.uniform(10, 50),
                               seed=int(rng.integers(1 << 30)))
    kx = np.array([c.kx for c in comps])
    ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps]) * s
    ph = np.array([c.phi for c in comps])
    om = omega_of(kx, ky, DEPTH)
    S2 = Stokes2Transfer(kx, ky, DEPTH)

    ax = np.linspace(0, L, NG)
    X, Y = np.meshgrid(ax, ax)
    xr, yr = X.ravel(), Y.ravel()

    def field(t, with_bound=True):
        psi = kx[None] * xr[:, None] + ky[None] * yr[:, None] - om[None] * t \
            + ph[None]
        free = (A[None] * np.cos(psi)).sum(1)
        if not with_bound:
            return free.reshape(NG, NG)
        bound = S2.correction_np(xr, yr, float(t), A, kx, ky, ph, om)
        return (free + bound).reshape(NG, NG)

    def ekurt(a):
        v = a.ravel() - a.mean()
        return float((v ** 4).mean() / v.std() ** 4 - 3.0)

    dt = Tp / 3.7                                  # decorrelate frames
    fin = np.stack([field(i * dt) for i in range(N_IN)])
    ts = [(N_IN + i) * dt for i in range(N_LAB)]
    lab = np.stack([field(t) for t in ts])
    lab_free = np.stack([field(t, with_bound=False) for t in ts])

    # Total kurtosis of a wave patch mixes TWO things: bound-wave nonlinearity
    # (positive) and narrowbandedness (an almost monochromatic patch tends to
    # the arcsine law, kurt -> -1.5). Only the first is what CL1 must detect.
    # Synthesis lets us subtract the second exactly: the same sea without its
    # bound waves.
    k_tot = ekurt(lab)
    k_free = ekurt(lab_free)
    return (fin.astype(np.float32), k_tot - k_free, k_tot, k_free,
            Hs, Tp, float(s))


def build(n_sea, seed):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if len(d['dkurt']) >= n_sea:
            print(f"cache {CACHE}: {len(d['dkurt'])} seas", flush=True)
            return (d['X'][:n_sea], d['dkurt'][:n_sea], d['ktot'][:n_sea],
                    d['meta'][:n_sea])
    rng = np.random.default_rng(seed)
    X, DK, KT, M = [], [], [], []
    t0 = time.time()
    for i in range(n_sea):
        f, dk, kt, kf, hs, tp, s = make_sea(rng)
        X.append(f)
        DK.append(dk)
        KT.append(kt)
        M.append([hs, tp, s, kf])
        if (i + 1) % 200 == 0:
            print(f"  sea {i+1}/{n_sea}  dkurt {dk:+.3f}  ktot {kt:+.3f}  "
                  f"({time.time()-t0:.0f} s)", flush=True)
    X = np.stack(X)                                # (S, N_IN, NG, NG)
    DK, KT, M = np.array(DK), np.array(KT), np.array(M)
    os.makedirs('tmp', exist_ok=True)
    np.savez_compressed(CACHE, X=X, dkurt=DK, ktot=KT, meta=M)
    return X, DK, KT, M


def standardize(a):
    a = a - a.mean((-2, -1), keepdims=True)
    return a / (a.std((-2, -1), keepdims=True) + 1e-8)


def mask_of(bs, ratio, dev, rng):
    m = torch.rand(bs, 8, 8, device=dev, generator=rng) < ratio
    return m.repeat_interleave(8, 1).repeat_interleave(8, 2)


def features(model, Z, bs=256):
    F = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Z), bs):
            x = Z[i:i + bs].to(DEV)
            inp = torch.stack([x, torch.zeros_like(x)], 1)
            F.append(model.features(inp).cpu().double())
    return torch.cat(F).numpy()


def probe(Ftr, ytr, Fte, yte, ridge=1e-2):
    A = np.concatenate([Ftr, np.ones((len(Ftr), 1))], 1)
    W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ ytr)
    B = np.concatenate([Fte, np.ones((len(Fte), 1))], 1)
    p = B @ W
    return float(1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=2000)
    ap.add_argument('--held', type=int, default=400)
    ap.add_argument('--train_seas', type=int, default=0,
                    help='cap the number of DISTINCT training seas (0 = all). '
                         'The held-out set is unchanged, so runs differ only '
                         'in sea diversity -- the axis W7 could not move.')
    ap.add_argument('--tag', type=str, default='')
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    X, K, KT, M = build(a.seas, a.seed)
    print(f"\n{len(K)} seas", flush=True)
    print(f"  label  dkurt (bound-wave only)  {K.min():+.3f} .. {K.max():+.3f}"
          f"   std {K.std():.3f}", flush=True)
    print(f"  total  kurt                     {KT.min():+.3f} .. {KT.max():+.3f}"
          f"   std {KT.std():.3f}", flush=True)
    print(f"  corr(dkurt, total kurt) {np.corrcoef(K, KT)[0,1]:+.3f}  "
          f"<- if high, total kurt would have been an adequate label anyway",
          flush=True)
    if K.std() < 0.02:
        print("[!] dkurt spread is degenerate; the task is not measurable.")
        return 1

    n_pool = len(K) - a.held
    n_tr = n_pool if a.train_seas <= 0 else min(a.train_seas, n_pool)
    Ztr = torch.tensor(standardize(X[:n_tr]).reshape(-1, PATCH, PATCH))
    Zte = torch.tensor(standardize(X[n_pool:]).reshape(-1, PATCH, PATCH))
    ytr = np.repeat(K[:n_tr], N_IN)
    yte = np.repeat(K[n_pool:], N_IN)
    print(f"train {n_tr} seas / {len(Ztr)} frames    "
          f"held-out {a.held} seas / {len(Zte)} frames", flush=True)

    # ---- baseline 0: the frame's own sample kurtosis, zero parameters -------
    ptr = excess_kurtosis(Ztr).double().numpy()[:, None]
    pte = excess_kurtosis(Zte).double().numpy()[:, None]
    r2_plug = probe(ptr, ytr, pte, yte)

    # ---- baseline 1: random-init features -----------------------------------
    torch.manual_seed(a.seed)
    rand = MAE(dim=a.dim).to(DEV)
    r2_rand = probe(features(rand, Ztr), ytr, features(rand, Zte), yte)
    print(f"\nbaselines on held-out seas:   plug-in R2 {r2_plug:+.3f}   "
          f"random-init R2 {r2_rand:+.3f}", flush=True)

    # ---- SSL pretraining ----------------------------------------------------
    model = MAE(dim=a.dim).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    hist, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        model.train()
        ii = rng.integers(0, len(Ztr), a.bs)
        x = Ztr[ii].to(DEV)
        m = mask_of(a.bs, 0.75, DEV, g)
        inp = torch.stack([torch.where(m, torch.zeros_like(x), x),
                           m.float()], 1)
        rec, _ = model(inp)
        loss = ((rec - x) ** 2)[m].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % 10000 == 0 or step == a.steps:
            r2 = probe(features(model, Ztr), ytr, features(model, Zte), yte)
            hist.append({'step': step, 'loss': float(loss), 'r2_heldout': r2})
            print(f"  step {step:7d}  masked MSE {float(loss):.4f}   "
                  f"held-out sea-kurt R2 {r2:+.3f}   "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    r2_ssl = hist[-1]['r2_heldout']
    d_plug, d_rand = r2_ssl - r2_plug, r2_ssl - r2_rand
    passed = bool(d_plug >= 0.05 and d_rand >= 0.05)

    print(f"\n=== G1 gate (pre-registered: beat BOTH by dR2 >= 0.05) ===")
    print(f"  plug-in     {r2_plug:+.3f}    SSL - plug-in  {d_plug:+.3f}")
    print(f"  random-init {r2_rand:+.3f}    SSL - random   {d_rand:+.3f}")
    print(f"  SSL         {r2_ssl:+.3f}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if not passed and d_plug < 0.05:
        print("\n  If the plug-in estimator cannot be beaten, CL1 should read "
              "the kurtosis\n  of its own observations and skip the encoder "
              "entirely.")

    suff = a.tag or ''
    torch.save({'model': model.state_dict(), 'dim': a.dim},
               CKPT.replace('.pt', f'{suff}.pt'))
    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT.replace('.json', f'{suff}.json'), 'w') as f:
        json.dump({'seas': len(K), 'train_seas': n_tr, 'dim': a.dim,
                   'held': a.held, 'steps': a.steps,
                   'kurt_range': [float(K.min()), float(K.max())],
                   'r2_plugin': r2_plug, 'r2_random': r2_rand,
                   'r2_ssl': r2_ssl, 'passed': passed, 'history': hist}, f,
                  indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
