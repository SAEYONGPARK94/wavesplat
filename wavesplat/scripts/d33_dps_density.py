"""
Probe — is the learned DPS prior's plateau caused by its observation cap (scalability)
or by posterior-mean over-smoothing?

For a few complex episodes (same period window 2.2*Tp), sweep the observation budget and
reconstruct with DPS (learned) and RELAX (off-grid). If DPS keeps improving with m, the
limit is the O(m^3) likelihood cap (a scalable likelihood would help); if DPS plateaus
while RELAX keeps dropping, the limit is over-smoothing (learning must output sharp
off-grid components, not a hedged mean).

Run: python3 scripts/d33_dps_density.py [--caps 200 400 700] [--frames 2] [--nsamp 5]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d26_relax_iterative import iterative_relax
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser

CKPT = 'results/direction3/setdiff_prior_v2.pt'
GUID = 0.1
EPISODES = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1']
RELAX_CAPS = [50, 100, 150, 200, 400, 700, 1500, 3000]


def subsample(P, z, m, seed=0):
    if len(z) <= m:
        return P, z
    i = np.random.default_rng(seed).choice(len(z), m, replace=False)
    return P[i], z[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caps', type=int, nargs='+', default=[200, 400, 700])  # DPS caps
    ap.add_argument('--frames', type=int, default=2)
    ap.add_argument('--nsamp', type=int, default=5)
    ap.add_argument('--episodes', nargs='+', default=EPISODES)
    ap.add_argument('--out', default='results/direction3/dps_density.json')
    a = ap.parse_args()

    ck = torch.load(CKPT, map_location=DP.DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    print(f"device={DP.DEVICE} DPS caps={a.caps} RELAX caps={RELAX_CAPS} "
          f"frames={a.frames} nsamp={a.nsamp}")

    out = {}
    t0 = time.time()
    for tag in a.episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        frames = np.linspace(i0, i1, a.frames).astype(int)
        dps_by_cap = {m: [] for m in a.caps}
        rel_by_cap = {m: [] for m in RELAX_CAPS}
        for i_star in frames:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            # DPS at each cap
            for m in a.caps:
                Ps, zs = subsample(P, z, m, seed=0)
                Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, depth,
                                   guidance=GUID, n=a.nsamp, seed=0)
                flds = [DP.atoms_to_field(Xd[b], stats, Ps, zs, depth, target)[0] for b in range(a.nsamp)]
                flds = [f for f in flds if f is not None]
                bma = np.mean(flds, axis=0) if flds else np.zeros_like(target['gt'])
                dps_by_cap[m].append(panel(bma, target['gt'], Hs)['nrmse'])
            # RELAX at each cap
            for m in RELAX_CAPS:
                Ps, zs = subsample(P, z, m, seed=0)
                fld, _ = iterative_relax(Ps, zs, depth, target, 3)
                rel_by_cap[m].append(panel(fld, target['gt'], Hs)['nrmse'])
        out[tag] = dict(dps={str(m): float(np.mean(v)) for m, v in dps_by_cap.items()},
                        relax={str(m): float(np.mean(v)) for m, v in rel_by_cap.items()})
        ds = "  ".join(f"m{m}:{out[tag]['dps'][str(m)]:.3f}" for m in a.caps)
        rs = "  ".join(f"m{m}:{out[tag]['relax'][str(m)]:.3f}" for m in RELAX_CAPS)
        print(f"[{tag:12s}] DPS  {ds}   | RELAX  {rs}   ({time.time()-t0:.0f}s)", flush=True)

    json.dump(out, open(a.out, 'w'), indent=2)
    # aggregate
    print("\n=== mean over complex episodes: nRMSE vs observation budget ===")
    print(f"{'cap':>6}  {'DPS(learned)':>13}  {'RELAX(off-grid)':>16}")
    allcaps = sorted(set(a.caps) | set(RELAX_CAPS))
    for m in allcaps:
        dv = [out[t]['dps'][str(m)] for t in out if str(m) in out[t]['dps']]
        rv = [out[t]['relax'][str(m)] for t in out if str(m) in out[t]['relax']]
        ds = f"{np.mean(dv):.3f}" if dv else "  -  "
        rs = f"{np.mean(rv):.3f}" if rv else "  -  "
        print(f"{m:>6}  {ds:>13}  {rs:>16}")
    print(f"saved {a.out}")


if __name__ == '__main__':
    main()
