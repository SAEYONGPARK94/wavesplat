"""Critic-selector label factory v1 (CRITIC_SELECTOR_DESIGN step 1).

For every fixedT v2 episode: build the recipe observation window (12 frames /
20 s, stratified 2000), run the 12 CERTIFIED candidates
    band {(0.35,2.5), (0.15,2.5)} x smult {1,3,10} x gamma {hand, net},
and record, per candidate: the TASK label (corr at forward-fan forecast
queries, t_end + 1..8 s, r 20-120 m -- the dir0-causal forecast region;
episode waves travel +x so the DOWNWAVE forecast is the causal task here;
switches to the encounter track when head-sea v2 episodes exist) and the
deployment-observable FEATURES (val residual as a demoted feature, amplitude
ratio, in-band energy fraction, obs stats, split-half beamform divergence).

Output: results/direction3/critic_labels_v1.json -- one row per
(episode, candidate). The critic trains on these; judgment stays the real
60-window set (zero-shot), per the frozen gates.
"""
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, build_bank, design, beamform_prior
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_g0b_selection_aware import DEV, DT

OUT = 'results/direction3/critic_labels_v1.json'
BANDS = [(0.35, 2.5), (0.15, 2.5)]
SMULTS = [1.0, 3.0, 10.0]


def episode_window(path, rng):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        obs_i = list(range(50, 249, 18))
        t_end = ts[obs_i[-1]]
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1],
                                      np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])
        # forecast queries: downwave (+x) fan, leads 1..8 s -- causal for dir0
        nq = 160
        tq = t_end + rng.uniform(1.0, 8.0, nq)
        rq = rng.uniform(20.0, 120.0, nq)
        aq = rng.uniform(-np.pi / 3, np.pi / 3, nq)
        Q = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), tq])
        A, om, kx, ky, ph = (comp[:, 0], comp[:, 1], comp[:, 2],
                             comp[:, 3], comp[:, 4])
        psi = (kx[None] * Q[:, 0:1] + ky[None] * Q[:, 1:2]
               - om[None] * Q[:, 2:3] + ph[None])
        gt = (A[None] * np.cos(psi)).sum(1)
    rr = np.hypot(P[:, 0], P[:, 1])
    edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), 167)
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    return P[sel], zv[sel], Q, gt, kp


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    rng = np.random.default_rng(2027)
    rows = []
    files = sorted(glob.glob('data/episode_*_fixedT.h5'))
    print(f"episodes: {len(files)}, candidates: "
          f"{len(BANDS)*len(SMULTS)*2}")
    for fi, path in enumerate(files):
        t0 = time.time()
        P, z, Q, gt, kp = episode_window(path, rng)
        zsd = float(np.std(z))
        # split-half beamform divergence (shared feature)
        half = np.arange(len(P)) % 2 == 0
        kxb, kyb, om = build_bank(kp, (0.35, 2.5), kp / 10.0, 50.0, DEV,
                                  torch.float32)
        pw1 = beamform_prior(torch.tensor(P[half], device=DEV,
                                          dtype=torch.float32),
                             torch.tensor(z[half], device=DEV,
                                          dtype=torch.float32),
                             kxb, kyb, om).cpu().numpy()
        pw2 = beamform_prior(torch.tensor(P[~half], device=DEV,
                                          dtype=torch.float32),
                             torch.tensor(z[~half], device=DEV,
                                          dtype=torch.float32),
                             kxb, kyb, om).cpu().numpy()
        p1 = pw1 / max(pw1.max(), 1e-12)
        p2 = pw2 / max(pw2.max(), 1e-12)
        bf_div = float(np.abs(p1 - p2).mean() / (p1.mean() + 1e-12))

        t32 = torch.tensor(P, device=DEV, dtype=torch.float32)
        z32 = torch.tensor(z, device=DEV, dtype=torch.float32)
        Q32 = torch.tensor(Q, device=DEV, dtype=torch.float32)
        for band in BANDS:
            for sm in SMULTS:
                for gname in ('hand', 'net'):
                    if gname == 'hand':
                        with torch.no_grad():
                            pr = mmse_gpu(t32, z32, 50.0, Q32, 0.05,
                                          kp_hint=kp, band=band, no_em=True,
                                          pexp=1.0, smult=sm).cpu().numpy()
                    else:
                        pr = np.asarray(net_gamma_mmse(
                            net, P, z, 50.0, Q, 0.05, kp_hint=kp,
                            use_em=False, smult_fixed=sm))
                    # held-out obs residual (demoted to a feature)
                    vi = np.arange(len(z)) % 4 == 0
                    with torch.no_grad():
                        pv = mmse_gpu(t32[~torch.tensor(vi, device=DEV)],
                                      z32[~torch.tensor(vi, device=DEV)],
                                      50.0, t32[torch.tensor(vi, device=DEV)],
                                      0.05, kp_hint=kp, band=band,
                                      no_em=True, pexp=1.0,
                                      smult=sm).cpu().numpy() \
                            if gname == 'hand' else None
                    vr = float(np.sqrt(np.mean(
                        (pv - z[vi]) ** 2))) / (zsd + 1e-9) \
                        if pv is not None else float('nan')
                    lab = float(np.corrcoef(pr, gt)[0, 1]) \
                        if np.std(pr) > 1e-12 else 0.0
                    rows.append({
                        'file': os.path.basename(path),
                        'band': list(band), 'smult': sm, 'gamma': gname,
                        'label_corr': lab,
                        'feat': {
                            'amp_ratio': float(np.std(pr) / (zsd + 1e-9)),
                            'val_resid': vr,
                            'bf_div': bf_div,
                            'kp': kp, 'zsd': zsd, 'n_obs': int(len(z)),
                        }})
        best = max(r['label_corr'] for r in rows[-12:])
        worst = min(r['label_corr'] for r in rows[-12:])
        print(f"[{fi+1}/{len(files)}] {os.path.basename(path):44s} "
              f"cand corr [{worst:+.2f},{best:+.2f}] "
              f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
