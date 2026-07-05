"""
d3_falcon_native_bench.py — GIVEN-SENSOR validation, step 1 (2026-07-05
사용자 지시: 센서 사양은 주어진 것; 그 사양으로 모사된 데이터에서 검증).

The repo's episodes ARE Falcon-emulated observations (wavesim lidar sim:
FOV 120x25, 10 FPS, raydrop, config comment 'Falcon K1'). Benchmark the
2026-07-05 estimators on the NATIVE per-frame LiDAR points:

  window = W_MEM seconds of frames ending at t0, obs = stacked native
  points (x, y, t, z); estimate; score nRMSE + corr on the GT grid at t0
  (nowcast) and t0 + {2,4} s (short forecast, control-relevant).

Estimators: hand-MMSE, net-MMSE (v3, zero-shot — OOD operator: forward
cone pool, not the SA track). Reference columns: stored WaveSplat P1
numbers are appended for context but protocols differ (P1 = full-episode
tracking) — direct comparison deferred.

Usage: python3 scripts/d3_falcon_native_bench.py [--episodes ...]
       [--w_mem 20] [--n_win 5] [--m_cap 600]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesplat.data_loader import EpisodeDataLoader
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse

G = 9.81
OUT = 'results/direction3/falcon_native_bench.json'

EPISODES_DEFAULT = [
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', default=EPISODES_DEFAULT)
    ap.add_argument('--w_mem', type=float, default=12.0,
                    help='trailing memory [s] (control-compatible; '
                         'episodes are only 30 s long)')
    ap.add_argument('--n_win', type=int, default=5)
    ap.add_argument('--m_cap', type=int, default=600,
                    help='obs cap per window (MMSE O(n^3))')
    a = ap.parse_args()
    net = get_net()
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    for ep in a.episodes:
        if len(out.get(ep, [])) >= a.n_win:
            print(f"[{ep}] already done, skipping", flush=True)
            continue
        L = EpisodeDataLoader('data/' + ep)
        depth, Hs = L.depth, L.Hs_true
        gx = L.gt_grid_x.numpy()
        gy = L.gt_grid_y.numpy()
        Xg, Yg = np.meshgrid(gx, gy)
        lam_p = G * L.Tp_true ** 2 / (2 * np.pi)
        kp = 2 * np.pi / lam_p
        frames = L.get_tracking_frames(0.2)
        dtf = float(frames[1].t - frames[0].t)
        nfw = max(2, int(round(a.w_mem / dtf)))
        n_fc = int(round(4.0 / dtf))            # farthest forecast offset
        lo = nfw
        hi = len(frames) - n_fc - 2
        assert hi > lo, f"episode too short: {len(frames)} frames"
        i0s = np.linspace(lo, hi, a.n_win).astype(int)
        out[ep] = []
        print(f"[{ep}] lam_p={lam_p:.0f} m dtf={dtf:.2f}s "
              f"win={nfw} frames x {a.n_win}", flush=True)
        for i0 in i0s:
            sel = frames[i0 - nfw + 1:i0 + 1]
            P, z = [], []
            for f in sel:
                P.append(np.column_stack([
                    f.x_obs.numpy(), f.y_obs.numpy(),
                    np.full(len(f.x_obs), float(f.t))]))
                z.append(f.z_obs.numpy())
            P = np.vstack(P)
            z = np.concatenate(z)
            if len(z) > a.m_cap:
                ii = np.random.default_rng(0).choice(len(z), a.m_cap,
                                                     replace=False)
                P, z = P[ii], z[ii]
            row = dict(i0=int(i0), n_obs=int(len(z)))
            for dt_f, tag in ((0.0, 'now'), (2.0, 'f2'), (4.0, 'f4')):
                it = i0 + int(round(dt_f / dtf))
                tgt = frames[it]
                gt = tgt.gt_field.numpy()
                Pq = np.column_stack([Xg.ravel(), Yg.ravel(),
                                      np.full(Xg.size, float(tgt.t))])
                sig0 = 0.05 * Hs
                fh = mmse_bank_estimate(P, z, depth, Pq, sigma0=sig0,
                                        kp_hint=kp,
                                        seed=int(i0)).reshape(Xg.shape)
                fn = net_gamma_mmse(net, P, z, depth, Pq, sig0,
                                    seed=int(i0), use_em=False,
                                    kp_hint=kp).reshape(Xg.shape)
                for nm, f_ in (('hand', fh), ('net', fn)):
                    nr = float(np.sqrt(np.mean((f_ - gt) ** 2)) / (Hs / 4))
                    cc = float(np.corrcoef(f_.ravel(), gt.ravel())[0, 1])
                    row[f'{nm}_{tag}_nrmse'] = nr
                    row[f'{nm}_{tag}_corr'] = cc
            out[ep].append(row)
            print("  " + " ".join(
                f"{k}={v:.3f}" for k, v in row.items()
                if k.endswith('corr')), flush=True)
        json.dump(out, open(OUT, 'w'), indent=1, default=float)

    print("\n===== summary (mean over windows) =====")
    for ep in out:
        r = out[ep]
        line = ep.replace('episode_', '').replace('.h5', '') + ': '
        for nm in ('hand', 'net'):
            line += (f"{nm} now {np.mean([x[f'{nm}_now_corr'] for x in r]):+.2f}"
                     f"/f2 {np.mean([x[f'{nm}_f2_corr'] for x in r]):+.2f}"
                     f"/f4 {np.mean([x[f'{nm}_f4_corr'] for x in r]):+.2f}  ")
        print(line)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
