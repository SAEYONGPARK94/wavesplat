"""
F4 — neural operator (learned obs→field), the test of whether a LEARNED method beats the
~0.233 continuous-spectrum plateau that F1 (non-learned dense+smooth) and DPS both hit.

Unlike d34 (which output discrete atoms → needed wavenumber precision → collapsed), this
outputs the FIELD directly via a conditional neural field: encode the obs set, then for each
query (x,y,t) cross-attend to the obs context and predict elevation. On continuous (smooth)
targets there is no wavenumber-precision barrier. Trained on synthetic continuous-spectrum
fields; evaluated on the cached HOS windows.

Run: python3 scripts/d52_F4_neural_operator.py [--steps 4000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from d23_main_results import K_TP, panel
from d26_relax_iterative import iterative_relax
from d40_learned_acquisition import pool_xy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPTH = 50.0; XS, YS, TS, ZS = 120.0, 100.0, 10.0, 1.0
CACHE = 'results/direction3/_hos_windows.npz'


def gen_batch(B, M_obs=200, Q=400):
    """Continuous-spectrum synthetic: render from many (64) JONSWAP components."""
    pool = pool_xy()
    obs = np.zeros((B, M_obs, 4), np.float32); qpt = np.zeros((B, Q, 3), np.float32)
    qz = np.zeros((B, Q), np.float32); Hsb = np.zeros(B, np.float32)
    for b in range(B):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=64, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A = np.array([c.A for c in comp]); om = np.array([c.omega for c in comp])
        kx = np.array([c.kx for c in comp]); ky = np.array([c.ky for c in comp]); ph = np.array([c.phi for c in comp])
        Tobs = K_TP * Tp; tstar = Tobs
        oi = np.random.randint(0, len(pool), M_obs)
        ox, oy = pool[oi, 0], pool[oi, 1]; ot = np.random.uniform(0, Tobs, M_obs)
        oz = (A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)
        oz += np.random.normal(0, 0.02, M_obs)
        obs[b] = np.column_stack([ox, oy, ot, oz])
        qx = np.random.uniform(10, 250, Q); qy = np.random.uniform(-100, 100, Q)
        qz[b] = (A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*tstar+ph[None])).sum(1)
        qpt[b] = np.column_stack([qx, qy, np.full(Q, tstar)]); Hsb[b] = Hs
    t = lambda a: torch.tensor(a, device=DEVICE)
    return t(obs), t(qpt), t(qz), t(Hsb)


class NeuralOperator(nn.Module):
    def __init__(self, d=160, heads=4, enc=4, dec=3):
        super().__init__()
        self.oenc = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(el, enc)
        self.qenc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        dl = nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, dec)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, obs, q):
        o = obs.clone(); o[..., 0] /= XS; o[..., 1] /= YS; o[..., 2] /= TS; o[..., 3] /= ZS
        ctx = self.enc(self.oenc(o))
        qn = q.clone(); qn[..., 0] /= XS; qn[..., 1] /= YS; qn[..., 2] /= TS
        h = self.dec(self.qenc(qn), ctx)
        return self.head(h)[..., 0]


def train(steps, B=24, lr=3e-4):
    model = NeuralOperator().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0 = time.time()
    for s in range(steps):
        obs, q, qz, Hs = gen_batch(B)
        pred = model(obs, q)
        loss = (((pred - qz)**2).mean(1) / (Hs**2 + 1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 400 == 0 or s == steps-1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


def evaluate(model):
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    f4, rel = [], []
    for w in wins:
        target = {'gx': gx, 'gy': gy, 't': float(w['t']), 'gt': w['gt'], 'Hs': float(w['Hs'])}
        P, z = w['P'], w['z']
        if len(z) > 200:
            i = np.random.default_rng(0).choice(len(z), 200, replace=False); P, z = P[i], z[i]
        ob = torch.tensor(np.column_stack([P[:,0], P[:,1], P[:,2]-P[:,2].min(), z])[None], dtype=torch.float32, device=DEVICE)
        q = torch.tensor(np.column_stack([grid, np.full(len(grid), float(w['t'])-P[:,2].min())])[None], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            fld = model(ob, q)[0].cpu().numpy().reshape(len(gy), len(gx))
        f4.append(panel(fld, w['gt'], w['Hs'])['nrmse'])
        rel.append(panel(iterative_relax(w['P'], w['z'], DEPTH, target, 3)[0], w['gt'], w['Hs'])['nrmse'])
        print(f"[{w['tag']}] F4(neural-op) {f4[-1]:.3f} | RELAX {rel[-1]:.3f}", flush=True)
    mf, mr = float(np.mean(f4)), float(np.mean(rel))
    print(f"\nOVERALL (HOS continuous)  F4 {mf:.3f} | RELAX {mr:.3f} | DPS-32 0.234 | F1 0.233")
    print(f"  → F4 vs 0.233 plateau: {mf-0.233:+.3f}  ({'F4 BREAKS the plateau' if mf<0.228 else 'F4 ~plateau (info-limit confirmed)'})")
    json.dump(dict(f4=mf, relax=mr), open('results/direction3/F4_operator.json', 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=4000); a = ap.parse_args()
    print(f"training neural operator {a.steps} steps...", flush=True)
    model = train(a.steps)
    print("\n=== eval F4 on continuous HOS ===", flush=True)
    evaluate(model)


if __name__ == '__main__':
    main()
