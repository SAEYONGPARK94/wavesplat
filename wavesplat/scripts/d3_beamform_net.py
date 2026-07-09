"""Beamform-spectrum DETECTION net — new input format that beats mean-collapse.

Every prior learned estimator regressed from raw obs points → posterior-mean collapse on the
underdetermined field. RELAX works because it operates in the BEAMFORM (k-space matched-filter)
domain where components are localized PEAKS. Here we adopt that domain as the INPUT: compute
the dispersion-focused beamform power map B(kx,ky) from the obs, and train a CNN to output a
component-location HEATMAP (a DETECTION task — well-posed, no mean-collapse; the CNN learns to
deconvolve the beamformer PSF / sidelobes = learned CLEAN). Peaks → off-grid components →
field. UQ = the detection heatmap. Beamforming is a fixed linear front-end (not a RELAX hybrid).

Run: python3 scripts/d3_beamform_net.py [--steps 4000] [--eval_only ckpt.pt]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d40_learned_acquisition import pool_xy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81; DEPTH = 50.0
KLIM = 0.6; NK = 64
M_OBS = 400


def kgrid():
    kx = torch.linspace(-KLIM, KLIM, NK, device=DEVICE)
    KX, KY = torch.meshgrid(kx, kx, indexing='xy')
    k = torch.sqrt(KX**2 + KY**2) + 1e-9
    om = torch.sqrt(G * k * torch.tanh(k * DEPTH))
    return KX, KY, om            # (NK,NK) each


KX_G, KY_G, OM_G = kgrid()


def beamform(P, z):
    """obs (B,M,3)=(x,y,t), z (B,M) → beamform power map (B,NK,NK), log-normalised."""
    B = P.shape[0]
    kxf = KX_G.reshape(-1); kyf = KY_G.reshape(-1); omf = OM_G.reshape(-1)     # (NK*NK,)
    phi = (kxf[None, :, None] * P[:, None, :, 0] + kyf[None, :, None] * P[:, None, :, 1]
           - omf[None, :, None] * P[:, None, :, 2])                            # (B,G,M)
    resp = (z[:, None, :] * torch.exp(-1j * phi)).sum(-1)                      # (B,G) complex
    Bmap = (resp.abs() ** 2).reshape(B, NK, NK) / (P.shape[1] ** 2)
    Bmap = torch.log1p(Bmap / (Bmap.amax(dim=(1, 2), keepdim=True) + 1e-9) * 50.0)
    return Bmap


def target_heatmap(kx, ky, w):
    """Gaussian bumps (σ~1.5 cells) at true (kx,ky), weighted by w. kx,ky,w: lists per sample."""
    ii = ((kx + KLIM) / (2 * KLIM) * (NK - 1))
    jj = ((ky + KLIM) / (2 * KLIM) * (NK - 1))
    yy, xx = torch.meshgrid(torch.arange(NK, device=DEVICE).float(),
                            torch.arange(NK, device=DEVICE).float(), indexing='ij')
    H = torch.zeros(NK, NK, device=DEVICE)
    for a, i, j in zip(w, ii, jj):
        H = H + a * torch.exp(-((xx - i) ** 2 + (yy - j) ** 2) / (2 * 1.5 ** 2))
    return H / (H.max() + 1e-9)


class DetectCNN(nn.Module):
    def __init__(self, c=48):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(blk(1, c), blk(c, c), blk(c, c), blk(c, c), blk(c, c),
                                 nn.Conv2d(c, 1, 1))

    def forward(self, B):
        return torch.sigmoid(self.net(B[:, None]))[:, 0]      # (B,NK,NK) in [0,1]


def make_batch(Bsz, pool):
    x0, x1 = pool[:, 0].min(), pool[:, 0].max(); y0, y1 = pool[:, 1].min(), pool[:, 1].max()
    P = np.zeros((Bsz, M_OBS, 3), np.float32); Z = np.zeros((Bsz, M_OBS), np.float32)
    comps = []
    for b in range(Bsz):
        Hs = np.random.uniform(0.5, 3.5); Tp = np.random.uniform(5.0, 12.0)
        nc = int(np.random.choice([5, 15, 30]))
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
        kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c]); ph = np.array([x.phi for x in c])
        Tobs = K_TP * Tp
        idx = np.random.choice(len(pool), M_OBS, replace=len(pool) < M_OBS)
        ox, oy = pool[idx, 0], pool[idx, 1]; ot = np.random.uniform(0, Tobs, M_OBS)
        psi = kx[None] * ox[:, None] + ky[None] * oy[:, None] - om[None] * ot[:, None] + ph[None]
        Z[b] = (A[None] * np.cos(psi)).sum(1) + np.random.normal(0, 0.02, M_OBS)
        P[b] = np.column_stack([ox, oy, ot])
        comps.append((kx, ky, A ** 2))
    Pt = torch.tensor(P, device=DEVICE); Zt = torch.tensor(Z, device=DEVICE)
    Bmap = beamform(Pt, Zt)
    H = torch.stack([target_heatmap(torch.tensor(kx, device=DEVICE), torch.tensor(ky, device=DEVICE),
                                    torch.tensor(w, device=DEVICE)) for kx, ky, w in comps])
    return Bmap, H


def train(steps, Bsz=32, lr=3e-4):
    pool = pool_xy().astype(np.float32)
    model = DetectCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        Bmap, H = make_batch(Bsz, pool)
        Hp = model(Bmap)
        # weighted MSE (emphasise the peaks — most cells are ~0)
        loss = (((Hp - H) ** 2) * (1 + 20 * H)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 200 == 0 or s == steps - 1:
            with torch.no_grad():
                peak_hit = (Hp * (H > 0.3)).sum() / ((H > 0.3).sum() + 1e-6)
            print(f"  step {s:4d}  loss {loss.item():.5f}  peak-resp {peak_hit.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


def extract(Hp, thresh=0.15):
    """local maxima of the heatmap → (kx,ky) list."""
    from scipy.ndimage import maximum_filter
    H = Hp.cpu().numpy()
    mx = maximum_filter(H, size=5)
    pk = np.argwhere((H == mx) & (H > thresh * H.max()))
    kx = -KLIM + pk[:, 1] / (NK - 1) * 2 * KLIM
    ky = -KLIM + pk[:, 0] / (NK - 1) * 2 * KLIM
    return kx.astype(np.float32), ky.astype(np.float32)


@torch.no_grad()
def evaluate(model, episodes):
    model.eval(); out = {}
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
        i0 = nb + 3; i1 = len(L) - 2; sel = np.linspace(i0, i1, 3).astype(int); nr = []
        for i_star in sel:
            fstar = L[int(i_star)]; P, z = gather_frames(L, int(i_star), nb)
            if len(z) > M_OBS:
                idx = np.random.default_rng(0).choice(len(z), M_OBS, replace=False); P, z = P[idx], z[idx]
            t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
            Bmap = beamform(torch.tensor(Ps[None], device=DEVICE, dtype=torch.float32),
                            torch.tensor(z[None], device=DEVICE, dtype=torch.float32))
            Hp = model(Bmap)[0]
            kx, ky = extract(Hp)
            if len(kx) == 0:
                nr.append(1.0); continue
            obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], z]).astype(np.float32)
            A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
            fld = render_wave_field(kx, ky, A, phi, depth, gx, gy, float(fstar.t) - t0)
            nr.append(panel(fld, fstar.gt_field.numpy(), Hs)['nrmse'])
        out[tag] = float(np.mean(nr)); print(f"  {tag:12s} beamform-detect nRMSE {out[tag]:.3f}", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--ckpt', default='results/direction3/beamform_net.pt')
    ap.add_argument('--eval_only', default=None)
    a = ap.parse_args()
    eps = ['BF3_5c_s0', 'BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1']
    if a.eval_only:
        model = DetectCNN().to(DEVICE); model.load_state_dict(torch.load(a.eval_only, map_location=DEVICE))
    else:
        print(f"training {a.steps} steps on {DEVICE} (beamform-spectrum detection CNN)...")
        model = train(a.steps); torch.save(model.state_dict(), a.ckpt); print(f"saved {a.ckpt}")
    print("\n=== eval (beamform-detect vs DPS simple~0.19, RELAX simple 0.05-0.09) ===")
    res = evaluate(model, eps); print(f"mean nRMSE: {np.mean(list(res.values())):.3f}")
    json.dump(res, open('results/direction3/beamform_net_eval.json', 'w'), indent=2)
