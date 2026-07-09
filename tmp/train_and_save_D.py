"""Train the amortized posterior D (forecast-aware: queries span window + future) and save checkpoint."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from d56_amortized_posterior import PosteriorNet
from d52_F4_neural_operator import XS, YS, TS, ZS, DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
K_TP = 2.2; DEPTH = 50.0

def gen_batch_fc(B, M_obs=200, Q=400, fc_max=6.0):
    pool = pool_xy()
    obs = np.zeros((B, M_obs, 4), np.float32); qpt = np.zeros((B, Q, 3), np.float32)
    qz = np.zeros((B, Q), np.float32); Hsb = np.zeros(B, np.float32)
    for b in range(B):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=64, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A=np.array([c.A for c in comp]); om=np.array([c.omega for c in comp])
        kx=np.array([c.kx for c in comp]); ky=np.array([c.ky for c in comp]); ph=np.array([c.phi for c in comp])
        Tobs = K_TP*Tp
        oi = np.random.randint(0, len(pool), M_obs); ox, oy = pool[oi,0], pool[oi,1]
        ot = np.random.uniform(0, Tobs, M_obs)
        oz = (A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1) + np.random.normal(0,0.02,M_obs)
        obs[b] = np.column_stack([ox, oy, ot, oz])
        qx = np.random.uniform(10,250,Q); qy = np.random.uniform(-100,100,Q)
        qt = np.random.uniform(0.5*Tobs, Tobs+fc_max, Q)          # spans late-window + FUTURE (forecast)
        qz[b] = (A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1)
        qpt[b] = np.column_stack([qx, qy, qt]); Hsb[b] = Hs
    t=lambda a: torch.tensor(a, device=DEVICE)
    return t(obs), t(qpt), t(qz), t(Hsb)

def main(steps=6000, B=24, lr=3e-4):
    model = PosteriorNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0=time.time()
    for s in range(steps):
        obs,q,qz,Hs = gen_batch_fc(B)
        mean,logv = model(obs,q); logv=logv.clamp(-8,4)
        nll = 0.5*((mean-qz)**2/logv.exp()+logv)
        loss = (nll.mean(1)/(Hs**2+1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s%800==0 or s==steps-1:
            with torch.no_grad(): rmse=((mean-qz)**2).mean().sqrt().item()
            print(f"  step {s:4d} loss {loss.item():.4f} rmse {rmse:.3f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), 'results/direction3/amortized_posterior_fc.pt')
    print("saved results/direction3/amortized_posterior_fc.pt")

if __name__=='__main__':
    main()
