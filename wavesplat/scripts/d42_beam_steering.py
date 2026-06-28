"""
#1 — TRUE closed-loop beam-steering active sensing (not return subsampling).

Previous active-sensing scripts selected from already-measured returns (informative
subsampling). Real active sensing chooses WHERE to measure before measuring. Here the
candidate set is a dense pool of possible aim points over the LiDAR footprint × window; the
policy chooses K aim points; we MEASURE there (render the true surface + noise); estimate;
and ITERATE (seed → estimate → select → measure → re-estimate) — a closed sensing loop.

Compares, at matched measurement budget: random aim | learned-policy aim | greedy-D-opt aim
(coarse) | full dense pool (upper bound). Reports the budget→nRMSE curve (iterative loop).

Run: python3 scripts/d42_beam_steering.py [--frames 2]
"""
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d26_relax_iterative import iterative_relax
from d40_learned_acquisition import AcqNet, coarse_theta, greedy_dopt, theta_of, CKPT, DEVICE, NMAX
import d1_crb as C1

XR = (10.0, 178.0); YR = (-95.0, 95.0); G = 9.81
EPISODES = ['BF4_15c_s0', 'BF4_15c_s1', 'BF4_30c_s0', 'BF4_30c_s1', 'BF5_15c_s0', 'BF5_30c_s1']


def measure(comp_full, P, sigma=0.02, rng=None):
    """Render the TRUE surface at aim points P=(M,3)[x,y,t] + sensor noise (the measurement)."""
    A = comp_full[:, 0]; om = comp_full[:, 1]; kx = comp_full[:, 2]; ky = comp_full[:, 3]; ph = comp_full[:, 4]
    psi = kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2] - om[None] * P[:, 2:3] + ph[None]
    z = (A[None] * np.cos(psi)).sum(1)
    if rng is not None:
        z = z + rng.normal(0, sigma, len(z))
    return z.astype(np.float32)


def aim_pool(L, i_star, n_back, npool, rng):
    """Dense pool of POSSIBLE aim points over footprint × the window ending at i_star."""
    thi = float(L[int(i_star)].t); tlo = float(L[int(max(0, i_star - n_back))].t)
    x = rng.uniform(*XR, npool); y = rng.uniform(*YR, npool); t = rng.uniform(tlo, thi, npool)
    return np.column_stack([x, y, t]).astype(np.float32)


@torch.no_grad()
def policy_score(model, comp_np, P):
    na = min(len(comp_np), NMAX)
    comp = np.zeros((1, NMAX, 4), np.float32); cm = np.zeros((1, NMAX), np.float32)
    comp[0, :na] = comp_np[:na]; cm[0, :na] = 1.0
    return model(torch.tensor(comp, device=DEVICE), torch.tensor(cm, device=DEVICE),
                 torch.tensor(P[None], dtype=torch.float32, device=DEVICE))[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=2)
    ap.add_argument('--npool', type=int, default=1500)
    ap.add_argument('--seedK', type=int, default=80)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--perround', type=int, default=70)
    ap.add_argument('--episodes', nargs='+', default=EPISODES)
    a = ap.parse_args()
    model = AcqNet().to(DEVICE); model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    rng = np.random.default_rng(0)
    budgets = [a.seedK + a.perround * r for r in range(a.rounds + 1)]
    out = {}
    for tag in a.episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        comp_full = np.array(L.wave_components)
        th_true = theta_of(L)
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        sel = np.linspace(i0, i1, a.frames).astype(int)
        # per-method nRMSE at the final budget + the learned budget curve
        rec = {k: [] for k in ['rand', 'learned', 'greedy', 'full']}
        curve = []
        for i_star in sel:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            pool = aim_pool(L, i_star, n_back, a.npool, rng)
            zpool = measure(comp_full, pool, rng=rng)         # latent (only revealed where we aim)
            Kf = budgets[-1]
            # full dense (upper bound): measure all pool
            rec['full'].append(relax_nr(pool, zpool, depth, target))
            # random aim Kf
            ri = rng.choice(a.npool, Kf, replace=False)
            rec['rand'].append(relax_nr(pool[ri], zpool[ri], depth, target))
            # CLOSED-LOOP learned: seed -> estimate -> select more -> measure -> re-estimate.
            # Track learned/random/greedy at EACH budget to see WHERE active sensing helps.
            chosen = list(rng.choice(a.npool, a.seedK, replace=False))
            lc, rc, gc = [], [], []
            for r in range(a.rounds + 1):
                idx = np.array(chosen); B = len(idx)
                lc.append(relax_nr(pool[idx], zpool[idx], depth, target))
                rj = rng.choice(a.npool, B, replace=False)
                rc.append(relax_nr(pool[rj], zpool[rj], depth, target))
                seedb = rng.choice(a.npool, min(a.seedK, B), replace=False)
                cthb = coarse_theta(pool, zpool, depth, seedb)
                gtb = torch.tensor(cthb[:, [2, 0, 1, 3]], dtype=C1.DT, device=C1.DEVICE)
                gi = greedy_dopt(gtb, pool, B, depth)
                gc.append(relax_nr(pool[gi], zpool[gi], depth, target))
                if r < a.rounds:
                    cth = coarse_theta(pool, zpool, depth, idx)
                    sc = policy_score(model, cth, pool); sc[idx] = -1e9
                    chosen += list(np.argsort(-sc)[:a.perround])
            rec['learned'].append(lc[-1]); curve.append(dict(learned=lc, rand=rc, greedy=gc))
            # greedy-D-opt aim (coarse), same final budget (one shot)
            seed = rng.choice(a.npool, a.seedK, replace=False)
            cth = coarse_theta(pool, zpool, depth, seed)
            cth_t = torch.tensor(cth[:, [2, 0, 1, 3]], dtype=C1.DT, device=C1.DEVICE)
            gidx = greedy_dopt(cth_t, pool, Kf, depth)
            rec['greedy'].append(relax_nr(pool[gidx], zpool[gidx], depth, target))
        agg = {k: float(np.mean(v)) for k, v in rec.items() if k != 'full' or True}
        cl = {m: np.mean([c[m] for c in curve], axis=0).tolist() for m in ['learned', 'rand', 'greedy']}
        out[tag] = dict(full=agg['full'], budgets=budgets, **cl)
        print(f"[{tag:12s}] full-pool {agg['full']:.3f} | budget-curve @{budgets}", flush=True)
        for mth in ['rand', 'learned', 'greedy']:
            print(f"    {mth:8s} {['%.3f'%c for c in cl[mth]]}", flush=True)
    json.dump(out, open('results/direction3/beam_steering.json', 'w'), indent=2)
    print(f"\n=== OVERALL budget→nRMSE (free-aim beam steering) @ {budgets} ===")
    for mth in ['rand', 'learned', 'greedy']:
        cur = np.mean([out[t][mth] for t in out], axis=0)
        print(f"  {mth:8s} {['%.3f'%c for c in cur]}")
    print(f"  full-pool {np.mean([out[t]['full'] for t in out]):.3f}")
    print("saved results/direction3/beam_steering.json")


def relax_nr(P, z, depth, target):
    fld, _ = iterative_relax(P, z, depth, target, 3)
    return panel(fld, target['gt'], target['Hs'])['nrmse']


if __name__ == '__main__':
    main()
