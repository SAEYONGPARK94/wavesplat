"""E-16 decisive experiment, visualized: why the net wins on nonlinear (M3)
physics but not on correct linear physics, at a short wavelength.

For one short-lam sea (Tp7 Hs3), reconstruct the wave field on a dense fan grid
from the SAME 600 sparse observations, with the hand MMSE and the learned net,
for both the analytic-linear field (hand's model is correct) and the HOS-M3
field (bound waves the hand misses). Animate over the forecast leads.

  LINEAR row: hand ~ net ~ truth   -> net adds nothing (Delta < 0)
  M3 row    : hand misses fine crests, net recovers them -> net wins (Delta > 0)

  python3 scripts/d3_decisive_gif.py
"""
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')

import torch                                                     # noqa: E402
from e16_physics_operator import build_sea_linear, build_sea_M  # noqa: E402
from e15_hos_multidraw import CKPT, N_OBS, LEADS                 # noqa: E402
from d3_h7_hos_arena import z_at                                 # noqa: E402
from d3_door1b_field_bias import R_MIN, R_MAX, FOV_H             # noqa: E402
from wavesim.raydrop_threshold import T_MANUAL_50               # noqa: E402
from d3_gpu_mmse import mmse_gpu                                 # noqa: E402
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE    # noqa: E402
from d3_l1_s2head_train import SmultHead                         # noqa: E402

OUT = 'results/direction3/figures/decisive_reconstruction.gif'
NOISE = 0.05
DEPTH = 50.0
TP, HS, SEED = 7.0, 3.0, 0

# fan reconstruction grid (sensor frame: x downrange, y cross-range, +-60 deg)
gx = np.linspace(R_MIN, 170.0, 130)
gy = np.linspace(-95.0, 95.0, 130)
GX, GY = np.meshgrid(gx, gy)
r = np.hypot(GX, GY)
az = np.degrees(np.arctan2(GY, GX))
FAN = (r >= R_MIN) & (r <= R_MAX) & (np.abs(az) <= FOV_H)


def draw_obs(sea, rng):
    """one 600-pt observation draw (byte-for-byte the recipe in one_draw)."""
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep],
                                  np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep])
        rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-N_OBS // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    return P[sel], zv[sel]


def recon(Ps, zs, kp, net, head, lead):
    """hand + net reconstruction on the fan grid at forecast time `lead`."""
    Q = np.column_stack([GX[FAN], GY[FAN], np.full(FAN.sum(), lead)])
    with torch.no_grad():
        ph = mmse_gpu(torch.tensor(Ps, device=DEVICE, dtype=torch.float32),
                      torch.tensor(zs, device=DEVICE, dtype=torch.float32),
                      DEPTH, torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                      NOISE, kp_hint=kp).cpu().numpy()
    pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head))
    return _grid(ph), _grid(pn)


def _grid(vals):
    g = np.full(GX.shape, np.nan)
    g[FAN] = vals
    return g


def truth_grid(snap):
    v = z_at(snap, GX[FAN], GY[FAN])
    return _grid(v)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    print('building linear + M3 short-lam sea (Tp7 Hs3)...', flush=True)
    sea_l = build_sea_linear(TP, HS, SEED)
    sea_3 = build_sea_M(TP, HS, SEED, 3)
    rng = np.random.default_rng(500000)
    Pl, zl = draw_obs(sea_l, np.random.default_rng(500000))
    P3, z3 = draw_obs(sea_3, np.random.default_rng(500000))
    kp = sea_l['kp']

    frames = []
    vmax = 1.6 * HS
    for li, lead in enumerate(LEADS):
        tl = truth_grid(sea_l['leads'][li]); hl, nl = recon(Pl, zl, kp, net, head, lead)
        t3 = truth_grid(sea_3['leads'][li]); h3, n3 = recon(P3, z3, kp, net, head, lead)
        fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.8))
        rows = [('linear physics\n(hand model CORRECT)', tl, hl, nl,
                 corr(hl, tl), corr(nl, tl)),
                ('HOS M3 nonlinear\n(bound waves hand MISSES)', t3, h3, n3,
                 corr(h3, t3), corr(n3, t3))]
        cols = ['truth $\\eta$', 'hand MMSE', 'learned net']
        for ri, (rlab, tt, hh, nn, ch, cn) in enumerate(rows):
            for ci, (field, lab) in enumerate([(tt, cols[0]), (hh, cols[1]),
                                               (nn, cols[2])]):
                ax = axes[ri, ci]
                ax.imshow(field, extent=[gx[0], gx[-1], gy[0], gy[-1]],
                          origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          aspect='equal', interpolation='bilinear')
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(lab, fontsize=11)
                if ci == 0:
                    ax.set_ylabel(rlab, fontsize=9.2)
                if ci > 0:
                    c = ch if ci == 1 else cn
                    ax.text(0.5, -0.06, f'corr {c:+.2f}', transform=ax.transAxes,
                            ha='center', va='top', fontsize=10,
                            color=('#009E73' if c > 0.5 else '#D55E00'))
        d_lin = corr(nl, tl) - corr(hl, tl)
        d_m3 = corr(n3, t3) - corr(h3, t3)
        fig.suptitle(f'E-16 short-$\\lambda$ (Tp7 Hs3, $\\lambda$76 m) — forecast '
                     f'lead {lead:.0f} s\nnet$-$hand:  linear {d_lin:+.2f} '
                     f'(no help)    |    M3 {d_m3:+.2f} (net wins)',
                     fontsize=12, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=95)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
        print(f'  lead {lead}s: linear net-hand {d_lin:+.3f} | M3 {d_m3:+.3f}',
              flush=True)

    frames = frames + frames[::-1]           # ping-pong loop
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=900,
                   loop=0, optimize=True)
    print(f'-> {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
