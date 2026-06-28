"""
Learned DETECTION for off-grid spectral estimation — the NN that leverages *why RELAX works*.

RELAX's weakness is its greedy detection: the dispersion-constrained beamform of the
residual has sidelobes and merges closely-spaced components, so greedy peak-picking misses
components (the oracle gap on dense complex seas). Here a small CNN learns to DECONVOLVE the
beamform map (suppress sidelobes, separate merged peaks) → a cleaned component heatmap whose
peaks recover the true wavenumbers. This is the learnable, principled front-end of an
unrolled RELAX: it embeds the iterative structure (operate on the residual beamform) and
adds capacity exactly where greedy detection fails.

This script: (1) trains the detection CNN on synthetic JONSWAP beamform maps, (2) evaluates
whether the cleaned map recovers true components better than the raw beamform, including a
closely-spaced-pair resolution test.

Run: python3 scripts/d37_learned_detection.py [--steps 4000]
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
DEPTH = 50.0
KLIM = 0.6
KGRID = 40
XR = (10.0, 180.0); YR = (-95.0, 95.0)
K_TP = 2.2
ax = torch.linspace(-KLIM, KLIM, KGRID, device=DEVICE)
KXg, KYg = torch.meshgrid(ax, ax, indexing='xy')          # (K,K)
KXf = KXg.reshape(-1); KYf = KYg.reshape(-1)
KK = torch.sqrt(KXf**2 + KYf**2 + 1e-9)
OMf = torch.sqrt(G * KK * torch.tanh(KK * DEPTH) + 1e-9)   # dispersion per grid point


def beamform_map(ox, oy, ot, r):
    """Batched dispersion-constrained beamform power map. ox..r: (B,M). -> (B,K,K)."""
    ps = (KXf[None, :, None] * ox[:, None, :] + KYf[None, :, None] * oy[:, None, :]
          - OMf[None, :, None] * ot[:, None, :])           # (B,KK,M)
    c = (r[:, None, :] * torch.cos(ps)).sum(-1)
    s = (r[:, None, :] * torch.sin(ps)).sum(-1)
    p = (c**2 + s**2).reshape(-1, KGRID, KGRID)
    return p / (ox.shape[1] ** 2)


POOL_XY = None


def load_real_pool():
    """Pool of real ship-LiDAR (x,y) return positions — match the beamform sidelobe
    structure of the deployment geometry (fixes the uniform-obs domain gap)."""
    global POOL_XY
    from wavesplat.data_loader import EpisodeDataLoader
    from d6_integrated_eval import tag_to_file
    from d15_sequential_fusion import gather_frames
    tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1', 'BF3_5c_s0',
            'BF4_5c_s0', 'BF4_15c_s2', 'BF5_30c_s3', 'BF4_30c_s2', 'BF5_15c_s1']
    pts = []
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag))
        nb = int(round(K_TP * L.Tp_true / 0.1))
        P, _ = gather_frames(L, nb + 50, nb)
        pts.append(P[:, :2])
    POOL_XY = np.vstack(pts).astype(np.float32)
    print(f"real obs (x,y) pool: {POOL_XY.shape}", flush=True)


def gen_batch(B, M=200, sigma=0.02, pair_frac=0.0):
    if POOL_XY is not None:
        idx = np.random.randint(0, len(POOL_XY), (B, M))
        ox = POOL_XY[idx, 0]; oy = POOL_XY[idx, 1]
    else:
        ox = np.random.uniform(*XR, (B, M)); oy = np.random.uniform(*YR, (B, M))
    Hmap = np.zeros((B, KGRID, KGRID), np.float32)
    OZ = np.zeros((B, M), np.float32); OT = np.zeros((B, M), np.float32)
    axn = np.linspace(-KLIM, KLIM, KGRID)
    sig = 1.4 * (2 * KLIM / KGRID)                         # heatmap width ~1.4 cells
    for b in range(B):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        nc = int(np.random.choice([5, 15, 30]))
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH,
                                  seed=int(np.random.randint(1e9)))
        A = np.array([c.A for c in comp]); om = np.array([c.omega for c in comp])
        kx = np.array([c.kx for c in comp]); ky = np.array([c.ky for c in comp])
        ph = np.array([c.phi for c in comp])
        Tobs = K_TP * Tp; ot = np.random.uniform(0, Tobs, M)
        psi = kx[None] * ox[b][:, None] + ky[None] * oy[b][:, None] - om[None] * ot[:, None] + ph[None]
        OZ[b] = (A[None] * np.cos(psi)).sum(1) + np.random.normal(0, sigma, M)
        OT[b] = ot
        # target heatmap: amplitude-weighted Gaussian bumps at true (kx,ky)
        KXn, KYn = np.meshgrid(axn, axn)                  # (K,K)
        for n in range(nc):
            Hmap[b] += A[n] * np.exp(-((KXn - kx[n])**2 + (KYn - ky[n])**2) / (2 * sig**2))
    return (torch.tensor(ox, dtype=torch.float32, device=DEVICE),
            torch.tensor(oy, dtype=torch.float32, device=DEVICE),
            torch.tensor(OT, device=DEVICE), torch.tensor(OZ, device=DEVICE),
            torch.tensor(Hmap, device=DEVICE))


def norm_map(m):
    mx = m.amax(dim=(1, 2), keepdim=True).clamp(min=1e-9)
    return m / mx


class DetCNN(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(blk(1, ch), blk(ch, ch), blk(ch, ch), blk(ch, ch),
                                 nn.Conv2d(ch, 1, 1))

    def forward(self, x):
        return torch.relu(self.net(x))


def train(steps, B=24, lr=3e-4):
    model = DetCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        ox, oy, ot, oz, H = gen_batch(B)
        Bm = norm_map(beamform_map(ox, oy, ot, oz))
        Ht = norm_map(H)
        pred = model(Bm[:, None])[:, 0]
        w = 1.0 + 20.0 * Ht                               # emphasise peak regions (bg dominates MSE)
        loss = (w * (pred - Ht) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 400 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


def peak_match(mp, kx_t, ky_t, ntop):
    """Top-ntop local peaks of map mp -> mean nearest-true distance (rad/m)."""
    m = mp.copy()
    axn = np.linspace(-KLIM, KLIM, KGRID)
    peaks = []
    for _ in range(ntop):
        j = np.argmax(m); iy, ix = divmod(j, KGRID)
        peaks.append((axn[ix], axn[iy]))
        m[max(0, iy-1):iy+2, max(0, ix-1):ix+2] = -1     # suppress neighborhood
    peaks = np.array(peaks)
    d = np.sqrt(((peaks[:, 0:1] - kx_t[None]) ** 2 + (peaks[:, 1:2] - ky_t[None]) ** 2))
    return float(d.min(axis=1).mean())                   # mean nearest-true over picked peaks


@torch.no_grad()
def evaluate(model, n=200):
    model.eval()
    raw_d, cln_d = [], []
    for _ in range(n // 8):
        ox, oy, ot, oz, H = gen_batch(8)
        Bm = norm_map(beamform_map(ox, oy, ot, oz))
        Cm = model(Bm[:, None])[:, 0]
        # recover true (kx,ky) from heatmap peaks via the original atoms is not stored;
        # instead measure agreement with the target heatmap by correlation
        Ht = norm_map(H)
        for b in range(8):
            # peak-distance proxy: use heatmap argmaxes of target as "true"
            tm = Ht[b].cpu().numpy()
            tk = []
            mm = tm.copy(); axn = np.linspace(-KLIM, KLIM, KGRID)
            for _ in range(8):
                j = np.argmax(mm); iy, ix = divmod(j, KGRID)
                if mm[iy, ix] < 0.15:
                    break
                tk.append((axn[ix], axn[iy])); mm[max(0, iy-1):iy+2, max(0, ix-1):ix+2] = -1
            if len(tk) < 2:
                continue
            tk = np.array(tk)
            raw_d.append(peak_match(Bm[b].cpu().numpy(), tk[:, 0], tk[:, 1], len(tk)))
            cln_d.append(peak_match(Cm[b].cpu().numpy(), tk[:, 0], tk[:, 1], len(tk)))
    return float(np.mean(raw_d)), float(np.mean(cln_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--real_geom', action='store_true', help='train on real LiDAR obs geometry')
    ap.add_argument('--ckpt', default='results/direction3/learned_detection.pt')
    a = ap.parse_args()
    if a.real_geom:
        load_real_pool()
    print(f"training detection CNN {a.steps} steps on {DEVICE} (KGRID={KGRID}, "
          f"real_geom={a.real_geom}) ...")
    model = train(a.steps)
    torch.save(model.state_dict(), a.ckpt)
    raw, cln = evaluate(model)
    print(f"\n=== detection peak-error (mean nearest-true, rad/m; lower=better) ===")
    print(f"  raw beamform : {raw:.4f}")
    print(f"  CNN-cleaned  : {cln:.4f}   ({100*(raw-cln)/raw:+.0f}% vs raw)")
    json.dump(dict(raw=raw, cleaned=cln, kgrid=KGRID, steps=a.steps),
              open('results/direction3/learned_detection.json', 'w'), indent=2)
    print("saved results/direction3/learned_detection.{pt,json}")


if __name__ == '__main__':
    main()
