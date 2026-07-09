"""15-window ceiling visualization + wave-characteristic metrics (beyond nRMSE).

3-panel GIF per window: GT (HOS) | B2 diffusion+polish (ceiling method) | Oracle floor (true 32
free modes, linear render). Plus per-window metrics comparing GT vs B2 vs Oracle:
  - nRMSE (amplitude error)
  - spatial correlation ρ (SHAPE similarity — high even if amplitude off)
  - Hs ratio (recovered / true significant height)
  - skewness λ3 (crest-trough asymmetry = the leading NONLINEARITY signature; HOS GT > 0,
    a linear reconstruction ≈ 0 → the gap IS the representation wall, quantified)
  - excess kurtosis λ4 (peakedness / extreme-crest tendency)

Run: python3 scripts/d3_ceiling_gif.py
"""
import os, sys, time, json
import numpy as np
import torch
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
import d44_hos_nonlinear as H
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms
from d3_setdiff_posterior_b2 import polish_atoms
from d3_r5_oracle import init_modes

DEVICE = DP.DEVICE
GX = np.linspace(10, 250, 80); GY = np.linspace(-100, 100, 60)
OUT = 'results/gifs'; PRIOR = 'results/direction3/setdiff_prior.pt'
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]


def wave_metrics(field, gt, Hs):
    fr_ = field.ravel(); gtr = gt.ravel()
    return dict(nrmse=float(np.sqrt(np.mean((field - gt) ** 2)) / max(Hs, 1e-6)),
                corr=float(np.corrcoef(fr_, gtr)[0, 1]),
                hs_ratio=float(4 * field.std() / max(4 * gt.std(), 1e-6)),
                skew=float(skew(fr_)), kurt=float(kurtosis(fr_)))


def b2_components(prior, stats, abar, betas, Ps, zs, depth, nsamp=2):
    Xd = DP.dps_sample(prior, stats, abar, betas, Ps, zs, depth, guidance=0.1, n=nsamp, seed=0)
    Pt = torch.tensor(Ps, device=DEVICE); zt = torch.tensor(zs, device=DEVICE)
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    comps = []
    for b in range(nsamp):
        xb = polish_atoms(Xd[b], stats, Pt, zt, depth, iters=80)
        kx, ky, a = denorm_atoms(xb, stats)
        kx, ky, a = kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()
        keep = a > 0.03 * max(a.max(), 1e-9); kx, ky = kx[keep], ky[keep]
        if len(kx) == 0:
            continue
        if len(kx) > 30:
            o = np.argsort(-a[keep])[:30]; kx, ky = kx[o], ky[o]
        A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
        comps.append((kx.astype(np.float32), ky.astype(np.float32), A, phi))
    return comps


def render_bma(comps, depth, t):
    return np.mean([render_wave_field(kx, ky, A, phi, depth, GX, GY, t) for kx, ky, A, phi in comps], axis=0)


def make(tag, Hs, Tp, seed, prior, stats, abar, betas):
    depth = H.DEPTH; n_back = int(round(2.2 * Tp / 0.1)); t0 = time.time()
    fr = H.build_window(Hs, Tp, seed, n_back, GX, GY, M=3)
    i_star = len(fr) - 2
    P = np.vstack([f['P'] for f in fr[:i_star + 1]]); z = np.concatenate([f['z'] for f in fr[:i_star + 1]])
    idx = np.random.default_rng(seed).choice(len(z), min(400, len(z)), replace=False)
    Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)
    comps = b2_components(prior, stats, abar, betas, Ps, zs, depth)
    okx, oky, oA, ophi = init_modes(Hs, Tp, seed); ocomp = (okx[:32], oky[:32], oA[:32], ophi[:32])

    # nowcast metrics (GT vs B2 vs Oracle)
    gtn = fr[i_star]['gt']; tn = fr[i_star]['t']; Hs_t = float(gtn.std() * 4)
    b2n = render_bma(comps, depth, tn); orn = render_wave_field(*ocomp, depth, GX, GY, tn)
    m = dict(gt=dict(skew=float(skew(gtn.ravel())), kurt=float(kurtosis(gtn.ravel())), Hs=Hs_t),
             b2=wave_metrics(b2n, gtn, Hs_t), oracle=wave_metrics(orn, gtn, Hs_t))

    # GIF
    fidx = np.linspace(n_back // 2, i_star, 24).astype(int)
    gts = [fr[i]['gt'] for i in fidx]; ts = [fr[i]['t'] for i in fidx]
    vmax = max(np.abs(g).max() for g in gts); ext = [GX.min(), GX.max(), GY.min(), GY.max()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    titles = ['GT (HOS nonlinear)', 'B2 diffusion+polish\n(ceiling method)', 'Oracle floor\n(true 32 modes, linear)']
    ims = []
    for ax, ti in zip(axes, titles):
        im = ax.imshow(np.zeros((len(GY), len(GX))), origin='lower', extent=ext, aspect='auto',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(ti, fontsize=11); ax.set_xticks([]); ax.set_yticks([]); ims.append(im)
    fig.colorbar(ims[0], ax=axes, fraction=0.015, pad=0.01, label='η [m]')
    sup = fig.suptitle('', fontsize=11)

    def upd(f):
        gt = gts[f]; t = ts[f]
        b2f = render_bma(comps, depth, t); orf = render_wave_field(*ocomp, depth, GX, GY, t)
        ims[0].set_data(gt); ims[1].set_data(b2f); ims[2].set_data(orf)
        mb = wave_metrics(b2f, gt, Hs_t); mo = wave_metrics(orf, gt, Hs_t)
        axes[0].set_xlabel(f'skew {skew(gt.ravel()):+.2f}', fontsize=10)
        axes[1].set_xlabel(f'nRMSE {mb["nrmse"]:.2f}  ρ {mb["corr"]:.2f}  skew {mb["skew"]:+.2f}', fontsize=10)
        axes[2].set_xlabel(f'nRMSE {mo["nrmse"]:.2f}  ρ {mo["corr"]:.2f}  skew {mo["skew"]:+.2f}', fontsize=10)
        sup.set_text(f'{tag} s{seed}  Hs={Hs}m Tp={Tp}s  —  GT skew>0 (sharp crests); '
                     f'B2 & Oracle skew≈0 (linear) → representation wall')
        return ims

    anim = FuncAnimation(fig, upd, frames=len(fidx), interval=140, blit=False)
    path = f'{OUT}/ceiling_{tag}_s{seed}.gif'
    anim.save(path, writer=PillowWriter(fps=7)); plt.close(fig)
    print(f"  [{tag} s{seed}] gt_skew {m['gt']['skew']:+.2f}  b2(nr {m['b2']['nrmse']:.2f} ρ {m['b2']['corr']:.2f} "
          f"skew {m['b2']['skew']:+.2f})  or(nr {m['oracle']['nrmse']:.2f} ρ {m['oracle']['corr']:.2f})  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(PRIOR, map_location=DEVICE); stats = ck['stats']
    prior = SetDenoiser(**ck.get('arch', {})).to(DEVICE); prior.load_state_dict(ck['model']); prior.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)
    allm = {}
    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            allm[f'{tag}_s{seed}'] = make(tag, Hs, Tp, seed, prior, stats, abar, betas)
    json.dump(allm, open('results/direction3/ceiling_metrics.json', 'w'), indent=2)
    # summary
    def mean(path):
        return float(np.mean([m[path[0]][path[1]] for m in allm.values()]))
    print("\n=== 15-window means (GT vs B2 vs Oracle) ===", flush=True)
    print(f"  GT     skew {mean(('gt','skew')):+.3f}  kurt {mean(('gt','kurt')):+.3f}", flush=True)
    print(f"  B2     nRMSE {mean(('b2','nrmse')):.3f}  corr {mean(('b2','corr')):.3f}  "
          f"Hs_ratio {mean(('b2','hs_ratio')):.3f}  skew {mean(('b2','skew')):+.3f}", flush=True)
    print(f"  Oracle nRMSE {mean(('oracle','nrmse')):.3f}  corr {mean(('oracle','corr')):.3f}  "
          f"Hs_ratio {mean(('oracle','hs_ratio')):.3f}  skew {mean(('oracle','skew')):+.3f}", flush=True)
    print("saved results/direction3/ceiling_metrics.json", flush=True)


if __name__ == '__main__':
    main()
