"""
Learned active-sensing acquisition policy — the learning-central contribution.

Estimation is information-limited (CRB), so naive learning cannot beat classical RELAX
(shown exhaustively). But WHERE one samples moves the CRB wall: a coarse-estimate-driven
D-optimal selection lowers achieved error by ~25% on complex seas (d39). Greedy D-optimal
is O(K·d^3) per window — too slow for real-time scan control. Here a neural policy amortises
it: cross-attention from candidate returns to the (estimated) spectrum predicts a per-return
acquisition score in ONE forward pass; the top-K are kept. Trained by imitating greedy
D-optimal on synthetic windows; deployed with a coarse RELAX estimate (no oracle).

This script: (1) builds a cached dataset of (spectrum, candidate points, greedy-D-optimal
labels), (2) trains the AcqNet, (3) evaluates on benchmark complex seas — learned selection
-> RELAX nRMSE vs random / coarse-greedy-D-opt / oracle-D-opt, and timing.

Run: python3 scripts/d40_learned_acquisition.py [--gen N] [--steps 4000] [--eval]
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d18_relax_estimator import relax_step
from d26_relax_iterative import iterative_relax
from d11_active_sensing import theta_of, greedy_dopt
import d1_crb as C1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPTH = 50.0; KLIM = 0.7; NMAX = 32
XS, YS, TS = 120.0, 100.0, 10.0
M_CAND = 400; K_SEL = 80
DATA = 'results/direction3/acq_dataset.npz'
CKPT = 'results/direction3/acq_net.pt'
_POOL = None


def pool_xy():
    global _POOL
    if _POOL is None:
        tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1', 'BF4_5c_s0', 'BF3_5c_s0']
        pts = []
        for tg in tags:
            L = EpisodeDataLoader('data/' + tag_to_file(tg)); nb = int(round(K_TP * L.Tp_true / 0.1))
            P, _ = gather_frames(L, nb + 50, nb); pts.append(P[:, :2])
        _POOL = np.vstack(pts).astype(np.float32)
    return _POOL


def gen_dataset(N):
    pool = pool_xy()
    COMP = np.zeros((N, NMAX, 4), np.float32); CM = np.zeros((N, NMAX), np.float32)
    PTS = np.zeros((N, M_CAND, 3), np.float32); LAB = np.zeros((N, M_CAND), np.float32)
    t0 = time.time()
    for i in range(N):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        nc = int(np.random.choice([15, 30]))                 # complex (where sensing matters)
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH,
                                  seed=int(np.random.randint(1e9)))
        A = np.array([c.A for c in comp]); kx = np.array([c.kx for c in comp])
        ky = np.array([c.ky for c in comp]); ph = np.array([c.phi for c in comp])
        Tobs = K_TP * Tp
        idx = np.random.randint(0, len(pool), M_CAND)
        P = np.column_stack([pool[idx, 0], pool[idx, 1], np.random.uniform(0, Tobs, M_CAND)]).astype(np.float32)
        th = torch.tensor(np.stack([A, kx, ky, ph], 1), dtype=C1.DT, device=C1.DEVICE)
        sel = greedy_dopt(th, P, K_SEL, DEPTH)
        na = min(nc, NMAX)
        COMP[i, :na] = np.stack([kx, ky, A, ph], 1)[:na]; CM[i, :na] = 1.0
        PTS[i] = P; LAB[i, sel] = 1.0
        if (i + 1) % 100 == 0:
            print(f"  gen {i+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(DATA, COMP=COMP, CM=CM, PTS=PTS, LAB=LAB)
    print(f"saved {DATA}  {COMP.shape}")


class AcqNet(nn.Module):
    def __init__(self, d=128, heads=4, layers=3):
        super().__init__()
        self.cenc = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.penc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, layers)        # points attend to comps + self
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, comp, cmask, pts):
        ck = torch.stack([comp[..., 0] / KLIM, comp[..., 1] / KLIM, comp[..., 2],
                          torch.cos(comp[..., 3]), torch.sin(comp[..., 3])], -1)
        cz = self.cenc(ck)
        pz = self.penc(torch.stack([pts[..., 0] / XS, pts[..., 1] / YS, pts[..., 2] / TS], -1))
        kpm = (cmask < 0.5)                                  # (B,NMAX) True=pad to mask
        h = self.dec(pz, cz, memory_key_padding_mask=kpm)
        return self.head(h)[..., 0]                          # (B,M) logits


def train(steps, B=32, lr=3e-4):
    d = np.load(DATA)
    COMP = torch.tensor(d['COMP'], device=DEVICE); CM = torch.tensor(d['CM'], device=DEVICE)
    PTS = torch.tensor(d['PTS'], device=DEVICE); LAB = torch.tensor(d['LAB'], device=DEVICE)
    N = COMP.shape[0]; ntr = int(N * 0.95)
    model = AcqNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    pos = LAB[:ntr].mean(); pw = torch.tensor([(1 - pos) / pos], device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    t0 = time.time()
    for s in range(steps):
        bi = torch.randint(0, ntr, (B,), device=DEVICE)
        logit = model(COMP[bi], CM[bi], PTS[bi])
        loss = lossf(logit, LAB[bi])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 400 == 0 or s == steps - 1:
            with torch.no_grad():
                vi = torch.arange(ntr, N, device=DEVICE)
                vl = model(COMP[vi], CM[vi], PTS[vi])
                # top-K overlap with greedy label
                topk = vl.topk(K_SEL, dim=1).indices
                ov = LAB[vi].gather(1, topk).mean().item()
            print(f"  step {s:4d}  loss {loss.item():.4f}  val top{K_SEL}-overlap {ov:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), CKPT); print(f"saved {CKPT}")
    return model


def coarse_theta(P, z, depth, seed_idx):
    Ps, zs = P[seed_idx], z[seed_idx]
    kx, ky = relax_step(np.zeros(0), np.zeros(0), Ps, zs, depth)
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    return np.stack([kx, ky, A, phi], 1).astype(np.float32)


@torch.no_grad()
def acq_select(model, comp_np, P, K):
    na = min(len(comp_np), NMAX)
    comp = np.zeros((1, NMAX, 4), np.float32); cm = np.zeros((1, NMAX), np.float32)
    comp[0, :na] = comp_np[:na]; cm[0, :na] = 1.0
    logit = model(torch.tensor(comp, device=DEVICE), torch.tensor(cm, device=DEVICE),
                  torch.tensor(P[None], dtype=torch.float32, device=DEVICE))[0]
    return logit.topk(min(K, len(P))).indices.cpu().numpy()


def evaluate(model, episodes, K=120, seedK=80, pool=900, frames=2):
    rng = np.random.default_rng(0)
    out = {}; tg = {'greedy': 0.0, 'learned': 0.0}
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        th_true = theta_of(L)
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        sel = np.linspace(i0, i1, frames).astype(int)
        rec = {k: [] for k in ['rand', 'learned', 'coarse_greedy', 'oracle']}
        for i_star in sel:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            if len(z) > pool:
                ii = rng.choice(len(z), pool, replace=False); P, z = P[ii], z[ii]
            seed = rng.choice(len(z), min(seedK, len(z)), replace=False)
            cth = coarse_theta(P, z, depth, seed)
            ridx = rng.choice(len(z), K, replace=False)
            rec['rand'].append(relax_nr(P[ridx], z[ridx], depth, target))
            # learned (amortised) — timed
            t0 = time.time(); lidx = acq_select(model, cth, P, K); tg['learned'] += time.time() - t0
            rec['learned'].append(relax_nr(P[lidx], z[lidx], depth, target))
            # coarse greedy D-opt — timed
            cth_t = torch.tensor(cth[:, [2, 0, 1, 3]], dtype=C1.DT, device=C1.DEVICE)  # [A,kx,ky,phi]
            t0 = time.time(); gidx = greedy_dopt(cth_t, P, K, depth); tg['greedy'] += time.time() - t0
            rec['coarse_greedy'].append(relax_nr(P[gidx], z[gidx], depth, target))
            oidx = greedy_dopt(th_true, P, K, depth)
            rec['oracle'].append(relax_nr(P[oidx], z[oidx], depth, target))
        out[tag] = {k: float(np.mean(v)) for k, v in rec.items()}
        o = out[tag]
        print(f"[{tag:12s}] rand {o['rand']:.3f} | learned {o['learned']:.3f} | "
              f"coarse-greedy {o['coarse_greedy']:.3f} | oracle {o['oracle']:.3f}", flush=True)
    m = lambda k: float(np.mean([out[t][k] for t in out]))
    nfr = len(episodes) * frames
    print(f"\nOVERALL  rand {m('rand'):.3f} | LEARNED {m('learned'):.3f} "
          f"(−{100*(m('rand')-m('learned'))/m('rand'):.0f}%) | coarse-greedy {m('coarse_greedy'):.3f} | "
          f"oracle {m('oracle'):.3f}")
    print(f"timing/window:  learned {1e3*tg['learned']/nfr:.1f} ms  |  greedy {1e3*tg['greedy']/nfr:.1f} ms")
    json.dump(out, open('results/direction3/acq_eval.json', 'w'), indent=2)
    print("saved results/direction3/acq_eval.json")


def relax_nr(P, z, depth, target):
    fld, _ = iterative_relax(P, z, depth, target, 3)
    return panel(fld, target['gt'], target['Hs'])['nrmse']


EPISODES = ['BF4_15c_s0', 'BF4_15c_s1', 'BF4_30c_s0', 'BF4_30c_s1', 'BF5_15c_s0', 'BF5_30c_s1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', type=int, default=0)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--episodes', nargs='+', default=EPISODES)
    ap.add_argument('--frames', type=int, default=2)
    ap.add_argument('--slope', action='store_true', help='evaluate on slope-raydrop episodes')
    a = ap.parse_args()
    if a.gen:
        gen_dataset(a.gen)
    if a.eval:
        model = AcqNet().to(DEVICE); model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    else:
        model = train(a.steps)
    if a.slope:
        global tag_to_file
        import d6_integrated_eval as _d6
        _orig = _d6.tag_to_file
        tag_to_file = lambda t: _orig(t).replace('_uniform.h5', '_slope.h5')
        globals()['tag_to_file'] = tag_to_file
    print(f"\n=== eval on {len(a.episodes)} episodes (slope={a.slope}) ===")
    evaluate(model, a.episodes, frames=a.frames)


if __name__ == '__main__':
    main()
