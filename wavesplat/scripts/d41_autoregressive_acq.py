"""
(a) Autoregressive acquisition policy — close the one-shot −16% to greedy's −25%.

The one-shot AcqNet (d40) cannot WHITEN: it scores points independently, so it picks
redundant high-leverage points. Greedy D-optimal whitens (F^{-1} update) to avoid
redundancy. Here the policy CONDITIONS on the already-selected points (a per-point
'selected' feature) and is trained to predict the NEXT greedy picks given a prefix —
learning the redundancy-aware selection. At inference it is applied iteratively (select a
block, mark, re-score), approximating sequential greedy at NN speed.

Run: python3 scripts/d41_autoregressive_acq.py [--gen N] [--steps 6000] [--eval]
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import d40_learned_acquisition as D40
from d40_learned_acquisition import (pool_xy, coarse_theta, relax_nr, greedy_dopt,
                                      theta_of, NMAX, KLIM, XS, YS, TS, M_CAND, K_SEL, DEPTH, EPISODES)
from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP
import d1_crb as C1

DEVICE = D40.DEVICE
DATA = 'results/direction3/acq_dataset_ar.npz'
CKPT = 'results/direction3/acq_net_ar.pt'


def gen_dataset(N):
    pool = pool_xy()
    COMP = np.zeros((N, NMAX, 4), np.float32); CM = np.zeros((N, NMAX), np.float32)
    PTS = np.zeros((N, M_CAND, 3), np.float32); ORD = np.zeros((N, K_SEL), np.int64)
    t0 = time.time()
    for i in range(N):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        nc = int(np.random.choice([15, 30]))
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A = np.array([c.A for c in comp]); kx = np.array([c.kx for c in comp])
        ky = np.array([c.ky for c in comp]); ph = np.array([c.phi for c in comp])
        idx = np.random.randint(0, len(pool), M_CAND)
        P = np.column_stack([pool[idx, 0], pool[idx, 1], np.random.uniform(0, K_TP * Tp, M_CAND)]).astype(np.float32)
        th = torch.tensor(np.stack([A, kx, ky, ph], 1), dtype=C1.DT, device=C1.DEVICE)
        sel = greedy_dopt(th, P, K_SEL, DEPTH)               # ordered
        na = min(nc, NMAX)
        COMP[i, :na] = np.stack([kx, ky, A, ph], 1)[:na]; CM[i, :na] = 1.0
        PTS[i] = P; ORD[i] = sel
        if (i + 1) % 100 == 0:
            print(f"  gen {i+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(DATA, COMP=COMP, CM=CM, PTS=PTS, ORD=ORD)
    print(f"saved {DATA}")


class AcqNetAR(nn.Module):
    def __init__(self, d=128, heads=4, layers=3):
        super().__init__()
        self.cenc = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.penc = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))   # +selected flag
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, comp, cmask, pts, selflag):
        ck = torch.stack([comp[..., 0] / KLIM, comp[..., 1] / KLIM, comp[..., 2],
                          torch.cos(comp[..., 3]), torch.sin(comp[..., 3])], -1)
        cz = self.cenc(ck)
        pz = self.penc(torch.stack([pts[..., 0] / XS, pts[..., 1] / YS, pts[..., 2] / TS, selflag], -1))
        h = self.dec(pz, cz, memory_key_padding_mask=(cmask < 0.5))
        return self.head(h)[..., 0]


def train(steps, B=32, lr=3e-4):
    d = np.load(DATA)
    COMP = torch.tensor(d['COMP'], device=DEVICE); CM = torch.tensor(d['CM'], device=DEVICE)
    PTS = torch.tensor(d['PTS'], device=DEVICE); ORD = torch.tensor(d['ORD'], device=DEVICE)
    N = COMP.shape[0]; ntr = int(N * 0.95); M = PTS.shape[1]
    model = AcqNetAR().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(M - K_SEL) / K_SEL], device=DEVICE))
    t0 = time.time()
    for s in range(steps):
        bi = torch.randint(0, ntr, (B,), device=DEVICE)
        selflag = torch.zeros(B, M, device=DEVICE); tgt = torch.zeros(B, M, device=DEVICE)
        for b in range(B):
            j = int(torch.randint(0, K_SEL, (1,)))            # prefix length (already selected)
            pre = ORD[bi[b], :j]; rest = ORD[bi[b], j:]
            selflag[b, pre] = 1.0; tgt[b, rest] = 1.0         # predict the remaining greedy picks
        logit = model(COMP[bi], CM[bi], PTS[bi], selflag)
        loss = lossf(logit * (1 - selflag) - 1e4 * selflag, tgt)   # ignore already-selected
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 500 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), CKPT); print(f"saved {CKPT}")
    return model


@torch.no_grad()
def ar_select(model, comp_np, P, K, blocks=8):
    na = min(len(comp_np), NMAX)
    comp = np.zeros((1, NMAX, 4), np.float32); cm = np.zeros((1, NMAX), np.float32)
    comp[0, :na] = comp_np[:na]; cm[0, :na] = 1.0
    ct = torch.tensor(comp, device=DEVICE); cmt = torch.tensor(cm, device=DEVICE)
    pt = torch.tensor(P[None], dtype=torch.float32, device=DEVICE)
    M = len(P); selflag = torch.zeros(1, M, device=DEVICE)
    chosen = []
    per = max(1, K // blocks)
    while len(chosen) < K:
        logit = model(ct, cmt, pt, selflag)[0]
        logit[selflag[0] > 0.5] = -1e9
        take = min(per, K - len(chosen))
        nxt = logit.topk(take).indices
        for ix in nxt.tolist():
            chosen.append(ix); selflag[0, ix] = 1.0
    return np.array(chosen)


def evaluate(model, episodes, K=120, seedK=80, pool=900, frames=2):
    rng = np.random.default_rng(0); out = {}; tlearn = 0.0; nfr = 0
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); th_true = theta_of(L)
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfrm = len(L)
        i0 = n_back + 3; i1 = nfrm - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfrm - 60)
        sels = np.linspace(i0, i1, frames).astype(int)
        rec = {k: [] for k in ['rand', 'ar', 'coarse_greedy', 'oracle']}
        for i_star in sels:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            if len(z) > pool:
                ii = rng.choice(len(z), pool, replace=False); P, z = P[ii], z[ii]
            seed = rng.choice(len(z), min(seedK, len(z)), replace=False)
            cth = coarse_theta(P, z, depth, seed)
            ridx = rng.choice(len(z), K, replace=False)
            rec['rand'].append(relax_nr(P[ridx], z[ridx], depth, target))
            t0 = time.time(); lidx = ar_select(model, cth, P, K); tlearn += time.time() - t0; nfr += 1
            rec['ar'].append(relax_nr(P[lidx], z[lidx], depth, target))
            cth_t = torch.tensor(cth[:, [2, 0, 1, 3]], dtype=C1.DT, device=C1.DEVICE)
            gidx = greedy_dopt(cth_t, P, K, depth)
            rec['coarse_greedy'].append(relax_nr(P[gidx], z[gidx], depth, target))
            oidx = greedy_dopt(th_true, P, K, depth)
            rec['oracle'].append(relax_nr(P[oidx], z[oidx], depth, target))
        out[tag] = {k: float(np.mean(v)) for k, v in rec.items()}; o = out[tag]
        print(f"[{tag:12s}] rand {o['rand']:.3f} | AR-learned {o['ar']:.3f} | "
              f"coarse-greedy {o['coarse_greedy']:.3f} | oracle {o['oracle']:.3f}", flush=True)
    m = lambda k: float(np.mean([out[t][k] for t in out]))
    print(f"\nOVERALL  rand {m('rand'):.3f} | AR-LEARNED {m('ar'):.3f} "
          f"(−{100*(m('rand')-m('ar'))/m('rand'):.0f}%) | coarse-greedy {m('coarse_greedy'):.3f} | oracle {m('oracle'):.3f}")
    print(f"AR timing/window: {1e3*tlearn/max(nfr,1):.1f} ms")
    json.dump(out, open('results/direction3/acq_ar_eval.json', 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', type=int, default=0); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--eval', action='store_true')
    a = ap.parse_args()
    if a.gen:
        gen_dataset(a.gen)
    if a.eval:
        model = AcqNetAR().to(DEVICE); model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    else:
        model = train(a.steps)
    print("\n=== eval (autoregressive acquisition) ===")
    evaluate(model, EPISODES)


if __name__ == '__main__':
    main()
