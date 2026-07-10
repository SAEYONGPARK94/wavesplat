"""T3: amortize the INFORMATION MAP itself. Prize = LATENCY, nothing else.

S1 proved the posterior contraction c_i = 1 - Var_post/Var_prior IS the
achievable information floor for the linear-Gaussian model -- a function of the
observation geometry and the prior alone, no ground truth. It is exact and it
is the reduction. But it costs an n x n solve with a ~2000-atom design every
time the geometry changes (a new range gate, a shorter window, a thinner scan).

This script learns a small net that maps an observation-geometry SUMMARY to the
6-band contraction vector, so the map can be previewed in microseconds when a
controller is choosing where to look. The exact formula stays available offline;
the net never replaces it, it only amortizes it. NO accuracy claim is made and
none is possible -- the teacher is a computation, so the net can at best match
it, and the only thing bought is wall-clock.

  input   : per-scan range histogram, 12 range bins x 12 scan-time bins (log1p
            counts) + 3 scalars (N, max range, window span). All read off the
            observations -- deployable.
  output  : 6-band posterior contraction, the BINS of d3_s1_posterior_contraction
            (0.10-0.20, 0.20-0.35, 0.35-0.60, 0.60-1.00, 1.00-1.60, 1.60-2.50).
  teacher : the EXACT S1 contraction, ported here (float64), on ~2000 random
            geometries with Rmax in [80,400] m, T in [2.5,40] s, N in [200,1200].

  GATE    : median relative error <= 10% per band on a held-out split (bands
            whose true contraction is numerically negligible are reported with
            absolute error too, since relative error on ~0 is meaningless).
  LATENCY : exact contraction vs net forward on the same GPU, median over the
            held-out geometries. This is the entire deliverable.
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

from d3_gpu_mmse import build_bank, design
from d3_g3_adversarial_sea import DEV, DEPTH, KP, SIGMA
import d3_static_field as SF

OUT = 'results/direction3/t3_infomap_net.json'
CKPT = 'results/direction3/t3_infomap_net.pt'
CACHE = 'results/direction3/_t3_infomap_data.npz'

# S1 conventions, replicated exactly ----------------------------------------
S1_BAND = (0.10, 2.5)
SIG = 0.05 * 4 * SIGMA
BINS = [(0.10, 0.20), (0.20, 0.35), (0.35, 0.60), (0.60, 1.00),
        (1.00, 1.60), (1.60, 2.50)]
NB = len(BINS)

# geometry ranges
RMAX_RANGE = (80.0, 400.0)
T_RANGE = (2.5, 40.0)
N_RANGE = (200, 1200)
RMIN = 15.0
SCAN_DT = 0.5

# feature histogram grid (absolute, so it encodes Rmax / T / N)
TBINS = np.linspace(-40.0, 0.0, 13)
RBINS = np.linspace(RMIN, 400.0, 13)


# --------------------------------------------------------- exact teacher (S1)
def contraction(P):
    """c per 6 k/kp band, EXACTLY as d3_s1_posterior_contraction.contraction."""
    kxb, kyb, om = build_bank(float(KP), S1_BAND, float(KP) / 10.0, DEPTH, DEV,
                              torch.float64)
    Phi = design(P.double(), kxb, kyb, om)
    B = len(kxb)
    gm = torch.ones(B, device=DEV, dtype=torch.float64) / B
    gam = torch.cat([gm, gm]) * (SIGMA ** 2)
    n = len(P)
    K = (Phi * gam[None]) @ Phi.T + (SIG ** 2) * torch.eye(
        n, device=DEV, dtype=torch.float64)
    M = torch.linalg.solve(K, Phi)
    c = gam * (Phi * M).sum(0)
    c = c.reshape(2, B).mean(0).clamp(0.0, 1.0)
    kmag = (torch.hypot(kxb, kyb) / KP).cpu().numpy()
    cc = c.cpu().numpy()
    out = np.array([cc[(kmag >= a) & (kmag < b)].mean() for a, b in BINS])
    del K, M, Phi
    return out


def fan(rmax, T, n, rng):
    n_scan = max(2, int(round(T / SCAN_DT)) + 1)
    scan_t = np.linspace(-T, 0.0, n_scan)
    ti = rng.integers(0, n_scan, n)
    t = scan_t[ti]
    rr = rng.uniform(RMIN, rmax, n)
    aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, n)
    return np.column_stack([rr * np.cos(aa), rr * np.sin(aa), t])


def summary(P):
    """Deployable geometry summary: 12x12 log1p histogram + 3 scalars."""
    r = np.hypot(P[:, 0], P[:, 1])
    H, _, _ = np.histogram2d(P[:, 2], r, bins=[TBINS, RBINS])
    feat = np.log1p(H.ravel())
    scal = np.array([len(P) / 1000.0, r.max() / 400.0, (-P[:, 2].min()) / 40.0])
    return np.concatenate([feat, scal]).astype(np.float32)


def build_dataset(n_geo, seed=0):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if len(d['X']) >= n_geo:
            print(f"loaded cached dataset {CACHE}  ({len(d['X'])} geometries)")
            return d['X'][:n_geo], d['Y'][:n_geo]
    rng = np.random.default_rng(seed)
    X, Y = [], []
    t0 = time.time()
    for i in range(n_geo):
        rmax = rng.uniform(*RMAX_RANGE)
        T = rng.uniform(*T_RANGE)
        n = int(rng.integers(*N_RANGE))
        P = fan(rmax, T, n, rng)
        with torch.no_grad():
            y = contraction(torch.tensor(P, device=DEV, dtype=torch.float32))
        X.append(summary(P))
        Y.append(y)
        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()
            print(f"  teacher {i+1}/{n_geo}  ({time.time()-t0:.0f}s)", flush=True)
    X, Y = np.array(X, np.float32), np.array(Y, np.float32)
    os.makedirs('results/direction3', exist_ok=True)
    np.savez(CACHE, X=X, Y=Y)
    return X, Y


class InfoNet(nn.Module):
    def __init__(self, din, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, NB))

    def forward(self, x):
        return torch.sigmoid(self.f(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geo', type=int, default=2000)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print(f"device {DEV}   generating/loading {a.geo} geometries...", flush=True)
    X, Y = build_dataset(a.geo, seed=a.seed)
    print(f"features {X.shape}  labels {Y.shape}")
    print(f"per-band mean contraction (teacher): "
          f"{np.array2string(Y.mean(0), precision=3)}")

    # split
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X))
    ntr = int(0.8 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    t = lambda z: torch.tensor(z, device=DEV, dtype=torch.float32)
    Xtr, Ytr = t(Xn[tr]), t(Y[tr])
    Xte, Yte = t(Xn[te]), t(Y[te])

    net = InfoNet(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    bs = 256
    curve = []
    for st in range(1, a.steps + 1):
        bi = torch.randint(0, len(Xtr), (bs,), device=DEV)
        pred = net(Xtr[bi])
        loss = ((pred - Ytr[bi]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if st % 500 == 0:
            with torch.no_grad():
                vl = float(((net(Xte) - Yte) ** 2).mean())
            curve.append((st, float(loss.detach()), vl))
            print(f"  step {st:5d}  train {float(loss):.5f}  val {vl:.5f}",
                  flush=True)

    with torch.no_grad():
        P_te = net(Xte).cpu().numpy()
    Yte_np = Y[te]
    eps = 1e-3
    print(f"\n{'band':<14}{'mean true':>10}{'med |abs err|':>14}"
          f"{'med rel err':>13}{'gate':>7}")
    per_band = {}
    all_pass = True
    for j, (a0, b0) in enumerate(BINS):
        tvals = Yte_np[:, j]
        pvals = P_te[:, j]
        abserr = np.abs(pvals - tvals)
        relerr = abserr / np.maximum(tvals, eps)
        med_rel = float(np.median(relerr))
        med_abs = float(np.median(abserr))
        mt = float(tvals.mean())
        # a band with negligible true contraction cannot be judged on rel err
        judged = mt >= 0.02
        ok = (med_rel <= 0.10) if judged else True
        all_pass &= ok
        tag = ('PASS' if ok else 'FAIL') if judged else 'n/a*'
        print(f"{f'{a0}-{b0}':<14}{mt:>10.3f}{med_abs:>14.4f}"
              f"{med_rel:>13.3f}{tag:>7}")
        per_band[f'{a0}-{b0}'] = dict(mean_true=mt, med_abs=med_abs,
                                      med_rel=med_rel, judged=bool(judged),
                                      pass_=bool(ok))
    print("  * bands with mean true contraction < 0.02 are numerically "
          "negligible;\n    relative error on ~0 is meaningless, absolute "
          "error reported instead.")

    # ---------------------------------------------------------- latency
    torch.cuda.synchronize()
    n_lat = min(60, len(te))
    lat_exact, lat_net = [], []
    with torch.no_grad():
        for k in range(n_lat):
            rmax = float(80 + (400 - 80) * (k / n_lat))
            T = float(np.random.default_rng(k).uniform(*T_RANGE))
            n = int(np.random.default_rng(100 + k).integers(*N_RANGE))
            P = fan(rmax, T, n, np.random.default_rng(200 + k))
            Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
            xn = torch.tensor(((summary(P) - mu) / sd)[None], device=DEV,
                              dtype=torch.float32)
            torch.cuda.synchronize(); t0 = time.time()
            _ = contraction(Pt)
            torch.cuda.synchronize(); lat_exact.append((time.time() - t0) * 1e3)
            torch.cuda.synchronize(); t0 = time.time()
            _ = net(xn)
            torch.cuda.synchronize(); lat_net.append((time.time() - t0) * 1e3)
    me, mn = float(np.median(lat_exact)), float(np.median(lat_net))
    print(f"\nlatency (median over {n_lat} geometries, same GPU):")
    print(f"  exact contraction : {me:8.3f} ms")
    print(f"  net forward       : {mn:8.4f} ms")
    print(f"  speedup           : {me/max(mn,1e-9):8.0f}x")

    print(f"\n=== T3 gate: median rel err <= 10% per (judged) band -> "
          f"{'PASS' if all_pass else 'FAIL'} ===")
    print("  prize is LATENCY ONLY; the exact formula is the reduction and "
          "stays available offline.")

    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd), CKPT)
    with open(OUT, 'w') as f:
        json.dump(dict(per_band=per_band, all_pass=bool(all_pass),
                       latency_exact_ms=me, latency_net_ms=mn,
                       speedup=me / max(mn, 1e-9),
                       teacher_mean=Y.mean(0).tolist(),
                       n_geo=len(X), curve=curve), f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
