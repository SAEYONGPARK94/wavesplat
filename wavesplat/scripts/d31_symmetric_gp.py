"""
M3 — give ST-DF-GP the SAME time/observation budget as the headline RELAX.

The published ST-DF-GP (0.205) fuses only a W=10 frame (1 s) window; the headline
RELAX uses a period-scaled window (2.2*Tp = 12-20 s) and up to ~3000 observations.
The -42% headline was never tested against a GP given the same budget. This does that.

For each of the d23 frames (same episodes, same 5 frames, same period-scaled window),
we run the GP reconstruction (its own WaveSplat-estimated dispersion components — the
real baseline, NOT oracle) at three budgets:
  - GP @ 1 s window (W=10)               — reproduces the published baseline anchor
  - GP @ period window, m<=200            — apples-to-apples with d30 DPS/RELAX(shared)
  - GP @ period window, m<=3000           — apples-to-apples with the headline RELAX_full
vs RELAX3_full (0.119) and DPS (0.206, from d30) and oracle.

Run: python3 scripts/d31_symmetric_gp.py [--smoke N] [--frames F] [--mcap M]
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, N_FRAMES, panel
from scripts.spacetime_gp_30ep import (comps_to_arrays, build_K_spacetime, gp_mean,
                                       SIGMA_OBS)
from scripts.gp_disp_pilot import estimate_components_wavesplat_opt

W_SHORT = 10           # published GP fusion window (1 s)


def subsample(P, z, m_cap, seed=0):
    if len(z) <= m_cap:
        return P, z
    idx = np.random.default_rng(seed).choice(len(z), m_cap, replace=False)
    return P[idx], z[idx]


def gp_reconstruct(P, z, A, om, kx, ky, target):
    """ST-DF-GP field reconstruction at target from window obs (P,z)."""
    gx, gy = target['gx'], target['gy']
    Xg, Yg = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])
    grid_pts = np.column_stack([grid_xy, np.full(len(grid_xy), target['t'])])
    K_oo = build_K_spacetime(P, P, A, om, kx, ky)
    K_po = build_K_spacetime(grid_pts, P, A, om, kx, ky)
    pred = gp_mean(K_oo, K_po, z).reshape(target['gt'].shape)
    return pred


def window_obs_short(L, i_star, w):
    """W-frame window ending at i_star (the published GP fusion window)."""
    P, z = [], []
    for i in range(max(0, i_star - w + 1), i_star + 1):
        f = L[i]
        P.append(np.column_stack([f.x_obs.numpy(), f.y_obs.numpy(),
                                  np.full(len(f.x_obs), float(f.t))]))
        z.append(f.z_obs.numpy())
    return np.vstack(P), np.concatenate(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--frames', type=int, default=N_FRAMES)
    ap.add_argument('--mcap', type=int, default=200)
    ap.add_argument('--mcap_full', type=int, default=3000)
    ap.add_argument('--out', default='results/direction3/symmetric_gp.json')
    a = ap.parse_args()

    gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
    relax_iter = json.load(open('results/direction3/main_results_iter.json'))
    relax_full = json.load(open('results/direction3/main_results.json'))
    dps = json.load(open('results/direction3/symmetric_dps.json'))
    tags = list(gp.keys())
    if a.smoke:
        tags = tags[:a.smoke]

    out = {}
    t0 = time.time()
    for ti, tag in enumerate(tags):
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        # GP's own dispersion components (WaveSplat-estimated from init window) — the baseline.
        # Cache per episode (estimation is ~30 s/episode and window-independent).
        cache = f'results/direction3/_gpcomps_{tag}.npy'
        if os.path.exists(cache):
            comps = np.load(cache)
        else:
            comps = np.asarray(estimate_components_wavesplat_opt(
                L, apply_reg_amp=True, n_starts=1, lambda_reg=0.5))
            np.save(cache, comps)
        A, om, kx, ky = comps_to_arrays(comps, depth)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        frames = np.linspace(i0, i1, a.frames).astype(int)
        p_short, p200, p3000 = [], [], []
        for i_star in frames:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            # GP @ 1s window (published anchor)
            Psh, zsh = window_obs_short(L, int(i_star), W_SHORT)
            Psh, zsh = subsample(Psh, zsh, a.mcap_full, seed=0)
            p_short.append(panel(gp_reconstruct(Psh, zsh, A, om, kx, ky, target), target['gt'], Hs))
            # GP @ period window
            P, z = gather_frames(L, int(i_star), n_back)
            Ps, zs = subsample(P, z, a.mcap, seed=0)
            p200.append(panel(gp_reconstruct(Ps, zs, A, om, kx, ky, target), target['gt'], Hs))
            Pf, zf = subsample(P, z, a.mcap_full, seed=0)
            p3000.append(panel(gp_reconstruct(Pf, zf, A, om, kx, ky, target), target['gt'], Hs))
        agg = lambda L_, k: (float(np.mean([d[k] for d in L_])), float(np.std([d[k] for d in L_])))
        out[tag] = dict(
            gp_short={k: agg(p_short, k) for k in ['nrmse', 'P', 'VR', 'skill']},
            gp_win200={k: agg(p200, k) for k in ['nrmse', 'P', 'VR', 'skill']},
            gp_win3000={k: agg(p3000, k) for k in ['nrmse', 'P', 'VR', 'skill']},
            relax3_full=relax_iter[tag]['deployable']['nrmse'][0],
            dps200=dps[tag]['dps']['nrmse'][0],
            stdfgp_pub=gp[tag]['fused']['nRMSE'],
            oracle=relax_full[tag]['oracle']['nrmse'][0])
        d = out[tag]
        print(f"[{ti+1:2d}/{len(tags)}] {tag:12s} Tobs={Tobs:4.0f}s  "
              f"GP1s {d['gp_short']['nrmse'][0]:.3f} | GPwin200 {d['gp_win200']['nrmse'][0]:.3f} "
              f"GPwin3000 {d['gp_win3000']['nrmse'][0]:.3f} || RELAX3_full {d['relax3_full']:.3f} "
              f"DPS200 {d['dps200']:.3f} | pubGP {d['stdfgp_pub']:.3f} ora {d['oracle']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(a.out, 'w'), indent=2)

    seas = {}
    for tag, r in out.items():
        s = f"{tag.split('_')[0]}_{tag.split('_')[1]}"; seas.setdefault(s, []).append(r)
    print(f"\n=== per sea-state (GP at matched budgets, window {K_TP}*Tp, {a.frames} frames) ===")
    print(f"{'sea':9s} {'GP_1s':>7} {'GPwin200':>9} {'GPwin3000':>10} | {'RELAX3f':>8} {'DPS200':>7} {'pubGP':>7} {'oracle':>7}")
    A_ = {k: [] for k in ['s', 'g2', 'g3', 'r', 'd', 'p', 'o']}
    for s in sorted(seas):
        rs = seas[s]; m = lambda f: float(np.mean([f(r) for r in rs]))
        gs = m(lambda r: r['gp_short']['nrmse'][0]); g2 = m(lambda r: r['gp_win200']['nrmse'][0])
        g3 = m(lambda r: r['gp_win3000']['nrmse'][0]); rl = m(lambda r: r['relax3_full'])
        dp = m(lambda r: r['dps200']); pg = m(lambda r: r['stdfgp_pub']); o = m(lambda r: r['oracle'])
        print(f"{s:9s} {gs:7.3f} {g2:9.3f} {g3:10.3f} | {rl:8.3f} {dp:7.3f} {pg:7.3f} {o:7.3f}")
        A_['s'].append(gs); A_['g2'].append(g2); A_['g3'].append(g3); A_['r'].append(rl)
        A_['d'].append(dp); A_['p'].append(pg); A_['o'].append(o)
    mm = lambda k: np.mean(A_[k])
    print(f"\nOVERALL  GP_1s {mm('s'):.3f} | GP_win200 {mm('g2'):.3f} | GP_win3000 {mm('g3'):.3f} "
          f"|| RELAX3_full {mm('r'):.3f} | DPS200 {mm('d'):.3f} | pubGP {mm('p'):.3f} | oracle {mm('o'):.3f}")
    print(f"saved {a.out}")


if __name__ == '__main__':
    main()
