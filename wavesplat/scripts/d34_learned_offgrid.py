"""
Prototype — learned OFF-GRID component estimator ("amortized RELAX").

Outputs a SPARSE set of off-grid atoms {(kx,ky,A,phi)} from the observation window and
renders the field as a sum of cosines (sharp by construction — no smoothing kernel).
Set-transformer encoder over obs + DETR-style atom-query decoder + dispersion hard
constraint. Trained on synthetic JONSWAP, evaluated on the benchmark complex episodes vs
DPS (~0.20 plateau) and RELAX (0.10-0.16).

Design: docs/learned_offgrid_design.md
Run: python3 scripts/d34_learned_offgrid.py [--steps 2000] [--eval_only ckpt.pt]
"""
import os, sys, math, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
DEPTH = 50.0
KLIM = 0.8
NQ = 48                       # atom queries (max model order; >30 for headroom + warm-start candidates)
XS, YS, TS = 120.0, 100.0, 10.0   # coordinate normalisation scales
XR = (10.0, 180.0); YR = (-95.0, 95.0)   # training footprint (matches real obs)
GW, GH = 28, 20               # training GT grid


def omega_of(kx, ky):
    k = torch.sqrt(kx**2 + ky**2 + 1e-9)
    return torch.sqrt(G * k * torch.tanh(k * DEPTH) + 1e-12)


# ───────────────────────── synthetic data ─────────────────────────
def make_batch(B, M_obs=400, sigma=0.02):
    gx = np.linspace(*XR, GW); gy = np.linspace(*YR, GH)
    Xg, Yg = np.meshgrid(gx, gy)                       # (GH, GW)
    obs = np.zeros((B, M_obs, 4), np.float32)
    gtf = np.zeros((B, GH, GW), np.float32)
    atoms = np.zeros((B, NQ, 3), np.float32)           # (kx,ky,A) padded
    amask = np.zeros((B, NQ), np.float32)
    Hs_b = np.zeros(B, np.float32); tstar_b = np.zeros(B, np.float32)
    for b in range(B):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        nc = int(np.random.choice([5, 15, 30]))
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH,
                                  seed=int(np.random.randint(1e9)))
        A = np.array([c.A for c in comp]); om = np.array([c.omega for c in comp])
        kx = np.array([c.kx for c in comp]); ky = np.array([c.ky for c in comp])
        ph = np.array([c.phi for c in comp])
        Tobs = K_TP * Tp; tstar = Tobs
        ox = np.random.uniform(*XR, M_obs); oy = np.random.uniform(*YR, M_obs)
        ot = np.random.uniform(0.0, Tobs, M_obs)
        psi = (kx[None] * ox[:, None] + ky[None] * oy[:, None]
               - om[None] * ot[:, None] + ph[None])     # (M, nc)
        oz = (A[None] * np.cos(psi)).sum(1) + np.random.normal(0, sigma, M_obs)
        obs[b] = np.column_stack([ox, oy, ot, oz])
        psg = (kx[None, None] * Xg[..., None] + ky[None, None] * Yg[..., None]
               - om[None, None] * tstar + ph[None, None])
        gtf[b] = (A[None, None] * np.cos(psg)).sum(-1)
        na = min(len(A), NQ)
        atoms[b, :na, 0] = kx[:na]; atoms[b, :na, 1] = ky[:na]; atoms[b, :na, 2] = A[:na]
        amask[b, :na] = 1.0
        Hs_b[b] = Hs; tstar_b[b] = tstar
    t = lambda a: torch.tensor(a, device=DEVICE)
    return (t(obs), t(gtf), torch.tensor(gx, dtype=torch.float32, device=DEVICE),
            torch.tensor(gy, dtype=torch.float32, device=DEVICE),
            t(Hs_b), t(tstar_b), t(atoms), t(amask))


# ───────────────────────── model ─────────────────────────
class Model(nn.Module):
    def __init__(self, d=160, heads=4, enc=4, dec=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(el, enc)
        self.q = nn.Parameter(torch.randn(NQ, d) * 0.02)
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, dec)
        self.head = nn.Linear(d, 6)   # kx_raw, ky_raw, logA, cos, sin, presence

    def forward(self, obs):
        B = obs.shape[0]
        x = obs.clone()
        x[..., 0] /= XS; x[..., 1] /= YS; x[..., 2] /= TS   # normalise coords
        h = self.enc(self.inp(x))                            # (B,M,d)
        q = self.q[None].expand(B, -1, -1)
        o = self.dec(q, h)                                   # (B,NQ,d)
        p = self.head(o)
        kx = torch.tanh(p[..., 0]) * KLIM
        ky = torch.tanh(p[..., 1]) * KLIM
        A = torch.nn.functional.softplus(p[..., 2]) * 0.1
        cs = torch.tanh(p[..., 3]); sn = torch.tanh(p[..., 4])
        nrm = torch.sqrt(cs**2 + sn**2 + 1e-9); cs = cs / nrm; sn = sn / nrm
        pres = torch.sigmoid(p[..., 5])
        return kx, ky, A, cs, sn, pres


def render(kx, ky, A, cs, sn, pres, gx, gy, tstar):
    """field (B,GH,GW) = sum_q pres*A*cos(psi+phi), cos(psi+phi)=cs*cos(psi)-sn*sin(psi)."""
    Xg, Yg = torch.meshgrid(gx, gy, indexing='xy')        # (GW?,) -> use 'xy': (GH,GW)
    Xg = Xg.reshape(-1); Yg = Yg.reshape(-1)               # (P,)
    om = omega_of(kx, ky)                                  # (B,NQ)
    psi = (kx[:, :, None] * Xg[None, None] + ky[:, :, None] * Yg[None, None]
           - om[:, :, None] * tstar)                       # (B,NQ,P)
    w = (pres * A)[:, :, None]
    field = (w * (cs[:, :, None] * torch.cos(psi) - sn[:, :, None] * torch.sin(psi))).sum(1)
    return field.reshape(kx.shape[0], len(gy), len(gx))


def render_obs(kx, ky, A, cs, sn, pres, ox, oy, ot):
    om = omega_of(kx, ky)
    psi = (kx[:, :, None] * ox[:, None] + ky[:, :, None] * oy[:, None]
           - om[:, :, None] * ot[:, None])
    w = (pres * A)[:, :, None]
    return (w * (cs[:, :, None] * torch.cos(psi) - sn[:, :, None] * torch.sin(psi))).sum(1)


def match_loss(pk, pky, pw, tk, tky, tw, tmask):
    """Weighted bidirectional Chamfer over (kx,ky), normalised by KLIM.
    Direct supervision on wavenumber placement → bypasses the non-convex field loss."""
    dk = (pk[:, :, None] - tk[:, None, :]) / KLIM        # (B,NQ,NMAX)
    dy = (pky[:, :, None] - tky[:, None, :]) / KLIM
    D = dk**2 + dy**2
    # each TRUE atom → nearest predicted (weighted by true amplitude)
    near_t = D.min(dim=1).values                          # (B,NMAX)
    lt = (near_t * tw * tmask).sum(1) / (tw * tmask).sum(1).clamp(min=1e-6)
    # each PRED atom → nearest *real* true (weighted by predicted weight)
    Dm = D + (1 - tmask[:, None, :]) * 1e6
    near_p = Dm.min(dim=2).values                         # (B,NQ)
    lp = (near_p * pw).sum(1) / pw.sum(1).clamp(min=1e-6)
    return (lt + lp).mean()


def hungarian_reg(pk, pky, pw, tk, tky, tw, tmask):
    """One-to-one Hungarian assignment (true -> pred) then differentiable regression of
    assigned predictions onto true (kx,ky). Pushes precision on the dominant components,
    which Chamfer (many-to-one) cannot."""
    B, NQ = pk.shape
    with torch.no_grad():
        DK = (pk[:, :, None] - tk[:, None, :]) / KLIM
        DY = (pky[:, :, None] - tky[:, None, :]) / KLIM
        cost = (DK**2 + DY**2).cpu().numpy()              # (B,NQ,NMAX)
    tot = pk.new_zeros(())
    cnt = 0
    for b in range(B):
        nt = int(tmask[b].sum().item())
        if nt == 0:
            continue
        rows, cols = linear_sum_assignment(cost[b][:, :nt])   # rows->pred, cols->true
        rows = torch.as_tensor(rows, device=pk.device); cols = torch.as_tensor(cols, device=pk.device)
        w = tw[b, cols] / (tw[b, cols].sum() + 1e-6)
        d = ((pk[b, rows] - tk[b, cols]) / KLIM) ** 2 + ((pky[b, rows] - tky[b, cols]) / KLIM) ** 2
        tot = tot + (w * d).sum()
        cnt += 1
    return tot / max(cnt, 1)


# ───────────────────────── train ─────────────────────────
def train(steps, B=24, lr=3e-4):
    model = Model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        obs, gtf, gx, gy, Hs, tstar, atoms, amask = make_batch(B)
        kx, ky, A, cs, sn, pres = model(obs)
        fld = render(kx, ky, A, cs, sn, pres, gx, gy, tstar.view(B, 1, 1))
        # field loss (Hs-normalised, ~nRMSE^2)
        fl = (((fld - gtf) ** 2).mean(dim=(1, 2)) / (Hs ** 2 + 1e-6)).mean()
        # data-consistency on obs
        oz_hat = render_obs(kx, ky, A, cs, sn, pres, obs[..., 0], obs[..., 1], obs[..., 2])
        ol = (((oz_hat - obs[..., 3]) ** 2).mean(dim=1) / (Hs ** 2 + 1e-6)).mean()
        # wavenumber supervision: Chamfer coverage + Hungarian precision regression
        ml = match_loss(kx, ky, pres * A, atoms[..., 0], atoms[..., 1], atoms[..., 2], amask)
        hr = hungarian_reg(kx, ky, pres * A, atoms[..., 0], atoms[..., 1], atoms[..., 2], amask)
        loss = fl + 0.3 * ol + 1.0 * ml + 2.0 * hr
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 200 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  field {fl.item():.4f}  obs {ol.item():.4f}  "
                  f"match {ml.item():.4f}  hung {hr.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


# ───────────────────────── eval on benchmark ─────────────────────────
@torch.no_grad()
def evaluate(model, episodes, frames_n=3, m_obs=400):
    model.eval()
    out = {}
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        sel = np.linspace(i0, i1, frames_n).astype(int)
        nr = []
        gxt = torch.tensor(gx, dtype=torch.float32, device=DEVICE)
        gyt = torch.tensor(gy, dtype=torch.float32, device=DEVICE)
        for i_star in sel:
            fstar = L[int(i_star)]
            P, z = gather_frames(L, int(i_star), n_back)
            if len(z) > m_obs:
                idx = np.random.default_rng(0).choice(len(z), m_obs, replace=False); P, z = P[idx], z[idx]
            # shift time so window starts near 0 (match training)
            t0 = P[:, 2].min()
            obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2] - t0, z]).astype(np.float32)
            ob = torch.tensor(obs[None], device=DEVICE)
            kx, ky, A, cs, sn, pres = model(ob)
            tstar = torch.tensor([float(fstar.t) - t0], device=DEVICE)
            fld = render(kx, ky, A, cs, sn, pres, gxt, gyt, tstar.view(1, 1, 1))[0].cpu().numpy()
            nr.append(panel(fld, fstar.gt_field.numpy(), Hs)['nrmse'])
        out[tag] = float(np.mean(nr))
        print(f"  {tag:12s} learned-offgrid nRMSE {out[tag]:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--ckpt', default='results/direction3/learned_offgrid.pt')
    ap.add_argument('--eval_only', default=None)
    a = ap.parse_args()
    eps = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1', 'BF3_5c_s0', 'BF4_5c_s0']
    if a.eval_only:
        ck = torch.load(a.eval_only, map_location=DEVICE)
        model = Model().to(DEVICE); model.load_state_dict(ck)
    else:
        print(f"training {a.steps} steps on {DEVICE} ...")
        model = train(a.steps)
        torch.save(model.state_dict(), a.ckpt); print(f"saved {a.ckpt}")
    print("\n=== eval on benchmark (learned off-grid vs DPS~0.20, RELAX 0.10-0.16) ===")
    res = evaluate(model, eps)
    print(f"\nmean nRMSE: {np.mean(list(res.values())):.3f}")
    json.dump(res, open('results/direction3/learned_offgrid_eval.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
