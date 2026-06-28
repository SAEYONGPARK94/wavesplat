"""
#2 — forecast integration: the SAME amortized posterior forecasts the future field with
calibrated, lead-time-growing UQ (a capability RELAX/F1 point estimators lack).

Train the conditional neural field with query points at FUTURE times (t* + lead); target =
the rendered future field. At inference, querying future times gives a phase-resolved
forecast + per-pixel sigma. Eval: forecast skill (nRMSE/Pearson) vs lead time, whether the
predicted UQ grows with lead and is calibrated, vs a persistence baseline.

Run: python3 scripts/d58_forecast_posterior.py [--steps 9000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from d23_main_results import K_TP
from d40_learned_acquisition import pool_xy
from d56_amortized_posterior import PosteriorNet
from d52_F4_neural_operator import XS, YS, TS, ZS, DEVICE
from wavesplat.evaluation.metrics import pearson_correlation

MAXLEAD = 5.0   # seconds of forecast horizon


def render(comp, x, y, t):
    A = np.array([c.A for c in comp]); om = np.array([c.omega for c in comp])
    kx = np.array([c.kx for c in comp]); ky = np.array([c.ky for c in comp]); ph = np.array([c.phi for c in comp])
    return (A[None] * np.cos(kx[None]*x[:,None] + ky[None]*y[:,None] - om[None]*t[:,None] + ph[None])).sum(1)


def gen_batch_fc(B, M_obs=200, Q=400):
    pool = pool_xy()
    obs = np.zeros((B, M_obs, 4), np.float32); qpt = np.zeros((B, Q, 3), np.float32)
    qz = np.zeros((B, Q), np.float32); Hsb = np.zeros(B, np.float32)
    for b in range(B):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=64, depth=50.0, seed=int(np.random.randint(1e9)))
        Tobs = K_TP * Tp; tstar = Tobs
        oi = np.random.randint(0, len(pool), M_obs); ox, oy = pool[oi, 0], pool[oi, 1]
        ot = np.random.uniform(0, Tobs, M_obs)
        obs[b] = np.column_stack([ox, oy, ot, render(comp, ox, oy, ot) + np.random.normal(0, 0.02, M_obs)])
        qx = np.random.uniform(10, 250, Q); qy = np.random.uniform(-100, 100, Q)
        # query times: mix nowcast (t*) and forecast (t* + lead)
        lead = np.where(np.random.rand(Q) < 0.4, 0.0, np.random.uniform(0, MAXLEAD, Q))
        qt = tstar + lead
        qz[b] = render(comp, qx, qy, qt); qpt[b] = np.column_stack([qx, qy, qt]); Hsb[b] = Hs
    t = lambda a: torch.tensor(a, device=DEVICE)
    return t(obs), t(qpt), t(qz), t(Hsb)


def train(steps, B=24, lr=3e-4):
    model = PosteriorNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0 = time.time()
    for s in range(steps):
        obs, q, qz, Hs = gen_batch_fc(B)
        mean, logv = model(obs, q); logv = logv.clamp(-8, 4)
        nll = 0.5 * ((mean - qz)**2 / logv.exp() + logv)
        loss = (nll.mean(1) / (Hs**2 + 1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 900 == 0 or s == steps-1:
            print(f"  step {s:4d}  nll {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def evaluate(model, n_ep=24):
    pool = pool_xy(); gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    leads = [0.0, 1.0, 2.0, 3.0, 5.0]
    agg = {L: dict(nr=[], rho=[], sig=[], cov=[], pers=[]) for L in leads}
    for _ in range(n_ep):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=64, depth=50.0, seed=int(np.random.randint(1e9)))
        Tobs = K_TP * Tp; oi = np.random.randint(0, len(pool), 200); ox, oy = pool[oi,0], pool[oi,1]
        ot = np.random.uniform(0, Tobs, 200)
        oz = render(comp, ox, oy, ot) + np.random.normal(0, 0.02, 200)
        ob = torch.tensor(np.column_stack([ox, oy, ot, oz])[None], dtype=torch.float32, device=DEVICE)
        gt_now = render(comp, grid[:,0], grid[:,1], np.full(len(grid), Tobs))
        for L in leads:
            tq = Tobs + L
            q = torch.tensor(np.column_stack([grid, np.full(len(grid), tq)])[None], dtype=torch.float32, device=DEVICE)
            mean, logv = model(ob, q)
            m = mean[0].cpu().numpy(); sig = (0.5*logv[0]).exp().cpu().numpy()
            gt = render(comp, grid[:,0], grid[:,1], np.full(len(grid), tq))
            agg[L]['nr'].append(np.sqrt(np.mean((m-gt)**2))/max(Hs,1e-6))
            agg[L]['rho'].append(float(np.corrcoef(m, gt)[0,1]))
            agg[L]['sig'].append(float(sig.mean()))
            agg[L]['cov'].append(float((np.abs(m-gt) <= sig).mean()))
            agg[L]['pers'].append(np.sqrt(np.mean((gt_now-gt)**2))/max(Hs,1e-6))  # persistence
    print("\n=== forecast skill & UQ vs lead time (amortised posterior) ===")
    print(f"{'lead(s)':>7} {'fc-nRMSE':>9} {'Pearson':>8} {'persist':>8} {'mean σ':>7} {'±1σ cov':>8}")
    out = {}
    for L in leads:
        a = agg[L]; r = dict(nrmse=float(np.mean(a['nr'])), rho=float(np.mean(a['rho'])),
                             persist=float(np.mean(a['pers'])), sigma=float(np.mean(a['sig'])), cov1=float(np.mean(a['cov'])))
        out[f'{L:.0f}s'] = r
        print(f"{L:7.0f} {r['nrmse']:9.3f} {r['rho']:8.2f} {r['persist']:8.3f} {r['sigma']:7.3f} {r['cov1']:8.2f}")
    json.dump(out, open('results/direction3/forecast_posterior.json', 'w'), indent=2)
    print("\n  → UQ가 lead와 함께 커지고(σ↑) 보정되며(±1σ~0.68), persistence를 이기면 = 예보+UQ 통합 성공.")
    print("saved results/direction3/forecast_posterior.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=9000); a = ap.parse_args()
    print(f"training forecast posterior {a.steps} steps...", flush=True)
    model = train(a.steps); model.eval()
    evaluate(model)


if __name__ == '__main__':
    main()
