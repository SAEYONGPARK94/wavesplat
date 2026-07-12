"""One-page gallery of DIFFERENT WAVES: calm single wave (mesh-fixed,
recovered), wind seas, storm head-sea, and two real oceans — truth vs the
deployed estimate for each."""
import os
import sys

import h5py
import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_l1_s2head_train import SmultHead
from d3_a7v2_refine import a7v2_predict
from d3_falcon_real_operator import RECORDS
from d3_sa_multirecord import MRecord
from d3_external_figs import Window

C_INK = '#0b0b0b'; C_INK2 = '#52514e'; C_MUT = '#898781'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                     'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb',
                     'savefig.facecolor': '#fcfcfb'})
OUT = 'results/direction3/fig_wave_gallery_2026-07-13.png'


def load_ep(path, rng):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        Hs = float(f['config'].attrs['Hs'])
        depth = float(f['config'].attrs['depth'])
        ts = f['frames/timestamps'][:]
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])
        obs_i = np.linspace(20, 220, 12).astype(int)
        t_end = float(ts[obs_i[-1]])
        P, zv = [], []
        for i in obs_i:
            m = f['frames/lidar_valid_mask'][i]
            p = f['frames/lidar_points'][i][m]
            P.append(np.column_stack([p[:, 0], p[:, 1],
                                      np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        rr = np.hypot(P[:, 0], P[:, 1])
        edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        take = [rng.choice(np.where(b == j)[0],
                           min((b == j).sum(), 167), replace=False)
                for j in range(12) if (b == j).sum()]
        sel = np.concatenate(take)
        Zt = f['frames/gt_wave_field'][obs_i[-1]]
    return dict(P=P[sel], z=zv[sel], gx=gx, gy=gy, Zt=Zt, kp=kp, Hs=Hs,
                depth=depth, t_end=t_end)


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(
        'results/direction3/prior_net_l1s2bw.pt', map_location=DEVICE)['head'])
    head.eval()
    rng = np.random.default_rng(11)

    panels = []   # (title, gx, gy, Ztruth, Zest, corr)

    synth = [
        ('tmp/finemesh/episode_BF2_1comp_seed0_fixedT.h5',
         'calm sea — single 8.2 m wave (BF2, mesh-fixed)\nregime-hybrid '
         'estimate', 'hybrid'),
        ('data/episode_BF3_5comp_seed0_fixedT.h5',
         'light wind sea — 5 components (BF3)\nlearned estimate', 'net'),
        ('data/episode_BF4_15comp_seed0_fixedT.h5',
         'moderate wind sea — 15 components (BF4)\nlearned estimate',
         'net'),
        ('data/episode_BF5_15comp_seed5_fixedT_dir180.h5',
         'rough head sea — 15 comp., dir 180$°$ (BF5)\nlearned estimate',
         'net'),
    ]
    for path, ttl, arm in synth:
        ep = load_ep(path, rng)
        GX, GY = np.meshgrid(ep['gx'], ep['gy'])
        Q = np.column_stack([GX.ravel(), GY.ravel(),
                             np.full(GX.size, ep['t_end'])])
        sig0 = 0.05 * ep['Hs']
        if arm == 'hybrid':
            pred, _, _, _ = a7v2_predict(ep['P'], ep['z'], ep['depth'], Q,
                                         sig0, ep['kp'], 0.0, 'ev1',
                                         force=True)
        else:
            pred = np.asarray(net_gamma_mmse(net, ep['P'], ep['z'],
                                             ep['depth'], Q, sig0, seed=0,
                                             kp_hint=ep['kp'],
                                             use_em=False, smult_head=head))
        Ze = pred.reshape(GX.shape)
        c = float(np.corrcoef(Ze.ravel(), ep['Zt'].ravel())[0, 1])
        panels.append((ttl, ep['gx'], ep['gy'], ep['Zt'], Ze, c))
        print(f"{ttl.splitlines()[0]}: corr {c:+.2f}", flush=True)

    for name, wi, ttl in (('YS17', 2, 'real ocean — Yellow Sea wind sea\n'
                           'learned estimate (stereo truth)'),
                          ('LJ18g', 8, 'real ocean — Atlantic storm '
                           '(Hs 7.3 m)\nlearned estimate (stereo truth)')):
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        w = Window(name, wi, rec=rec, net=net, head=head)
        gx, gy, tr, fh, fn = w.fields([0.0], stride=2)
        Zt, Ze = tr[0], fn[0]
        m = np.isfinite(Zt)
        c = float(np.corrcoef(Ze[m], Zt[m])[0, 1])
        panels.append((ttl, gx, gy, Zt, np.where(m, Ze, np.nan), c))
        print(f"{name} w{wi}: corr {c:+.2f}", flush=True)

    fig, axs = plt.subplots(3, 4, figsize=(13.6, 9.2), dpi=200)
    for i, (ttl, gx, gy, Zt, Ze, c) in enumerate(panels):
        r, c2 = divmod(i, 2)
        axT, axE = axs[r, 2 * c2], axs[r, 2 * c2 + 1]
        vmax = np.nanpercentile(np.abs(Zt), 97)
        for ax, Z, lab in ((axT, Zt, 'truth'),
                           (axE, Ze, f'estimate  corr {c:+.2f}')):
            ax.pcolormesh(gx, gy, Z, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          shading='auto')
            ax.set_title(lab, fontsize=8.5, color=C_INK2)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
        axT.text(0.02, 1.22, ttl, transform=axT.transAxes, fontsize=9.5,
                 color=C_INK, va='bottom')
    fig.suptitle('Many kinds of waves, one estimator family — truth vs '
                 'estimate across sea states, directions and real oceans',
                 fontsize=12.5, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.2)
    fig.savefig(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
