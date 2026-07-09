"""Generative-UQ visual: on one complex episode, show that the DPS set-diffusion
posterior gives a calibrated per-pixel uncertainty (spread predicts error) — the
thing the field-prior lacked. Panels: GT | posterior mean (BMA) | per-pixel std |
|err| vs std calibration scatter + binned reliability.
Run: python3 tmp/uq_figure.py [--tag BF4_15c_s0] [--guid 0.05] [--nsamp 16]
"""
import os, sys, json, argparse, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
from d23_main_results import K_TP, panel
from d15_sequential_fusion import gather_frames
from d6_integrated_eval import tag_to_file
from wavesplat.data_loader import EpisodeDataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='BF4_15c_s0')
ap.add_argument('--guid', type=float, default=0.05)
ap.add_argument('--nsamp', type=int, default=16)
ap.add_argument('--mcap', type=int, default=200)
ap.add_argument('--ckpt', default='results/direction3/setdiff_prior_long.pt')
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location=DP.DEVICE); stats = ck['stats']
model = SetDenoiser(**ck['arch']).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)

L = EpisodeDataLoader('data/' + tag_to_file(a.tag)); depth = L.depth; Hs = L.Hs_true
gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
n_back = int(round(K_TP * L.Tp_true / 0.1)); i_star = n_back + 3
fstar = L[int(i_star)]
target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
P, z = gather_frames(L, int(i_star), n_back)
idx = np.random.default_rng(0).choice(len(z), min(a.mcap, len(z)), replace=False)
Ps, zs = P[idx], z[idx]

Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, depth, guidance=a.guid, n=a.nsamp, seed=0)
flds = [f for b in range(a.nsamp)
        for f, _ in [DP.atoms_to_field(Xd[b], stats, Ps, zs, depth, target)] if f is not None]
arr = np.stack(flds); bma = arr.mean(0); std = arr.std(0)
gt = target['gt']; err = np.abs(bma - gt)
nr = panel(bma, gt, Hs)['nrmse']
s, e = std.ravel(), err.ravel()
pc = float(np.corrcoef(s, e)[0, 1])
# binned reliability: mean |err| per std-decile
order = np.argsort(s); nb = 10; edges = np.linspace(0, len(s), nb + 1).astype(int)
bx = [s[order[edges[i]:edges[i+1]]].mean() for i in range(nb)]
by = [e[order[edges[i]:edges[i+1]]].mean() for i in range(nb)]
print(f"{a.tag}  nRMSE(BMA)={nr:.3f}  UQ per-pixel corr(|err|,std)={pc:.3f}  nsamp={len(flds)}")

fig, ax = plt.subplots(1, 5, figsize=(24, 4.6))
vmax = np.abs(gt).max()
for k, (img, ttl, cmap, vlim) in enumerate([
        (gt, 'GT field [m]', 'RdBu_r', vmax), (bma, f'DPS posterior mean\nnRMSE={nr:.3f}', 'RdBu_r', vmax),
        (err, '|error| [m]', 'magma', None), (std, 'posterior std (UQ)', 'magma', None)]):
    im = ax[k].imshow(img, cmap=cmap, origin='lower', vmin=(-vlim if vlim else None), vmax=(vlim if vlim else None))
    ax[k].set_title(ttl); ax[k].set_xticks([]); ax[k].set_yticks([]); fig.colorbar(im, ax=ax[k], fraction=0.046)
ax[4].scatter(s, e, s=3, alpha=0.15, c='tab:gray')
ax[4].plot(bx, by, '-o', c='tab:red', lw=2, label='decile mean')
ax[4].set_xlabel('posterior std'); ax[4].set_ylabel('|error|')
ax[4].set_title(f'UQ calibration  corr={pc:.2f}'); ax[4].legend(); ax[4].grid(alpha=0.3)
fig.suptitle(f'Set-diffusion component prior — generative UQ on {a.tag} '
             f'(guid={a.guid}, {len(flds)} posterior samples)', fontsize=13)
fig.tight_layout()
out = f'results/direction3/uq_calibration_{a.tag}.png'
fig.savefig(out, dpi=120); print(f'saved {out}')
json.dump(dict(tag=a.tag, nrmse=nr, pix_corr=pc, reliability=dict(std_bin=[float(x) for x in bx],
          err_bin=[float(y) for y in by])), open(f'results/direction3/uq_calibration_{a.tag}.json', 'w'), indent=2)
