"""Probe: is the 1-2-mode detection a THRESHOLD artifact or a heatmap property?
Vary extract thresh + maximum_filter size on the 3 smoke windows; report n_modes,
chamfer/cov, LS-field nRMSE, modes_GT. Also dump heatmap stats."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import d44_hos_nonlinear as H
from d3_r5_oracle import init_modes
from d3_e1_oracle_fix import ls_refit_floor, nrmse
from d3_e3_shape_diagnosis import window_data, peak_wavenumber
from wavesplat.model.tnse import fit_amplitudes_phases
import d3_r0_physics_polish as R0
import d3_beamform_net as BN
from d3_beamform_net import beamform, DetectCNN, KLIM, NK
from d3_beamform_offset import DetectOffsetCNN
from scipy.ndimage import maximum_filter

DEPTH = H.DEPTH
net_off = DetectOffsetCNN().to(BN.DEVICE)
net_off.load_state_dict(torch.load('results/direction3/beamform_offset_ds.pt', map_location=BN.DEVICE))
net_off.eval()
net_pl = DetectCNN().to(BN.DEVICE)
net_pl.load_state_dict(torch.load('results/direction3/beamform_net.pt', map_location=BN.DEVICE))
net_pl.eval()


def extract_v(Hn, ox, oy, thresh, size, refine):
    mx = maximum_filter(Hn, size=size)
    pk = np.argwhere((Hn == mx) & (Hn > thresh * Hn.max()))
    if len(pk) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    rows, cols = pk[:, 0], pk[:, 1]
    ci = cols.astype(np.float32); cj = rows.astype(np.float32)
    if refine and ox is not None:
        ci = ci + ox[rows, cols]; cj = cj + oy[rows, cols]
    kx = -KLIM + ci / (NK - 1) * 2 * KLIM
    ky = -KLIM + cj / (NK - 1) * 2 * KLIM
    return kx.astype(np.float32), ky.astype(np.float32)


for tag, Hs, Tp in [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]:
    seed = 0
    kp = peak_wavenumber(Tp)
    gt, t, P, z = window_data(tag, seed)
    Hs_t = float(gt.std() * 4)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), min(400, len(z)), replace=False)
    Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)
    kxT, kyT, AT, _ = init_modes(Hs, Tp, seed)
    kxT32, kyT32 = kxT[:32], kyT[:32]

    Psh = Ps.copy(); Psh[:, 2] -= Psh[:, 2].min()
    Bmap = beamform(torch.tensor(Psh[None], device=BN.DEVICE, dtype=torch.float32),
                    torch.tensor(zs[None], device=BN.DEVICE, dtype=torch.float32))
    with torch.no_grad():
        Hm, oxm, oym = net_off(Bmap)
        Hp = net_pl(Bmap)[0]
    Hn_off = Hm[0].cpu().numpy(); oxn = oxm[0].cpu().numpy(); oyn = oym[0].cpu().numpy()
    Hn_net = Hp.cpu().numpy()

    # heatmap structure: how many cells above frac*max?
    print(f'\n== {tag}_s0  k_p={kp:.4f}  DK/kp={2*KLIM/(NK-1)/kp:.2f}')
    for nm, Hn in (('off', Hn_off), ('net', Hn_net)):
        fr = [float((Hn > f * Hn.max()).mean()) * NK * NK for f in (0.5, 0.15, 0.05, 0.02)]
        print(f'  {nm}: max {Hn.max():.3f}  cells>50/15/5/2%max = '
              + '/'.join(f'{int(v)}' for v in fr))
    # where is the TRUE energy on the grid? (cells within the true top-32 bbox)
    print(f'  true k range: kx [{kxT32.min():+.3f},{kxT32.max():+.3f}] '
          f'ky [{kyT32.min():+.3f},{kyT32.max():+.3f}]  (KLIM={KLIM})')

    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    for nm, Hn, ox, oy in (('off', Hn_off, oxn, oyn), ('net', Hn_net, None, None)):
        for size in (5, 3):
            for th in (0.15, 0.05, 0.02, 0.005):
                kx, ky = extract_v(Hn, ox, oy, th, size, refine=True)
                if len(kx) == 0:
                    print(f'  {nm} sz{size} th{th}: 0 modes'); continue
                A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
                comp = (kx, ky, A, phi)
                from d3_ceiling_gif import render_bma
                f = render_bma([comp], DEPTH, t)
                nr = nrmse(f, gt, Hs_t)
                mgt = R0.modes_gt_refit([comp], gt, t, Hs_t)
                d = np.sqrt((kx[:, None] - kxT32[None, :]) ** 2 +
                            (ky[:, None] - kyT32[None, :]) ** 2).min(1)
                cov = np.sqrt((kxT32[:, None] - kx[None, :]) ** 2 +
                              (kyT32[:, None] - ky[None, :]) ** 2).min(1)
                print(f'  {nm} sz{size} th{th:5.3f}: n={len(kx):3d}  nr={nr:.3f}  '
                      f'mGT={mgt:.3f}  ch/kp={d.mean()/kp:.2f}  cov/kp={cov.mean()/kp:.2f}')
