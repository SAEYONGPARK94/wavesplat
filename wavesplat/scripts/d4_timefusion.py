"""
Direction I×III pilot (a) — does whole-window TIME FUSION lower the achieved error?

Direction I proved the residual gap is the information wall and the only lever to
lower it is synthetic aperture = more observation time (D_eff = L + c_g·T_obs).
Here we widen the grid-free DPS observation window W (frames) at a FIXED target
frame and measure whether the BMA field nRMSE drops — i.e. whether the learned-
prior posterior actually exploits the extra information that time provides.

Run: python3 scripts/d4_timefusion.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms

DEVICE = DP.DEVICE
TI = 80                          # fixed target frame (tracking index) — fits W up to ~70


def gather_window_W(L, ti, W, m_cap=200):
    frames = L.get_tracking_frames(2.0)
    P, z = [], []
    for f in frames[ti-W+1:ti+1]:
        x = f.x_obs.numpy(); y = f.y_obs.numpy(); zz = f.z_obs.numpy(); t = float(f.t)
        P.append(np.column_stack([x, y, np.full_like(x, t)])); z.append(zz)
    P = np.vstack(P); z = np.concatenate(z)
    if len(z) > m_cap:
        idx = np.random.default_rng(0).choice(len(z), m_cap, replace=False)
        P, z = P[idx], z[idx]
    return P, z, frames


def eval_W(model, stats, abar, betas, L, W, depth, target, guidance=0.1, nsamp=6):
    P, z, _ = gather_window_W(L, TI, W)
    Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=guidance, n=nsamp, seed=0)
    fields = []
    for b in range(nsamp):
        f, _ = DP.atoms_to_field(Xd[b], stats, P, z, depth, target)
        if f is not None:
            fields.append(f)
    bma = np.mean(fields, axis=0)
    nr = float(np.sqrt(np.mean((bma - target['gt'])**2)) / max(target['Hs'], 1e-6))
    Tspan = float(P[:, 2].max() - P[:, 2].min())
    return nr, len(z), Tspan


def main():
    ck = torch.load(DP.CKPT, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser().to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    episodes = ['episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_uniform.h5']
    Ws = [10, 30, 60]
    out = {'TI': TI, 'guidance': 0.1, 'nsamp': 6, 'episodes': {}}
    print(f"{'episode':30s} " + " ".join(f"W={w}({w/10:.0f}s)" for w in Ws))
    for ep in episodes:
        L = EpisodeDataLoader('data/' + ep)
        depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        frames = L.get_tracking_frames(2.0); tgt = frames[TI + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}
        res = {}
        line = []
        for W in Ws:
            nr, M, Tspan = eval_W(model, stats, abar, betas, L, W, depth, target)
            res[W] = dict(nrmse=nr, M=M, Tspan=Tspan)
            line.append(f"{nr:.4f}")
        out['episodes'][ep] = res
        print(f"{ep:30s} " + "   ".join(line))
    json.dump(out, open('results/direction3/timefusion.json', 'w'), indent=2)

    # figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    cols = {'episode_BF4_5comp_seed0_uniform.h5': 'tab:green',
            'episode_BF4_15comp_seed0_uniform.h5': 'tab:orange'}
    lab = {'episode_BF4_5comp_seed0_uniform.h5': 'BF4 5c',
           'episode_BF4_15comp_seed0_uniform.h5': 'BF4 15c'}
    for ep, res in out['episodes'].items():
        ts = [res[W]['Tspan'] for W in Ws]
        nr = [res[W]['nrmse'] for W in Ws]
        ax.plot(ts, nr, 'o-', color=cols[ep], label=lab[ep], lw=2)
    ax.axhline(0.205, ls='--', c='tab:blue', label='WaveSplat fused (0.205)')
    ax.axhline(0.06, ls=':', c='k', label='exact (0.06)')
    ax.set_xlabel('observation window span [s]'); ax.set_ylabel('grid-free DPS (BMA) field nRMSE')
    ax.set_title('Pilot (a) — time fusion lowers the achieved error\n'
                 '(widening the DPS window = synthetic aperture, the Dir-I lever)')
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig('results/direction3/fig_timefusion.png', dpi=140)
    print('saved results/direction3/timefusion.json + fig_timefusion.png')


if __name__ == '__main__':
    main()
