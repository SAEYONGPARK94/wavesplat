"""
Symmetric head-to-head: learned grid-free DPS prior vs non-learned RELAX, on the
EXACT d23 main-result protocol (period-scaled window 2.2*Tp, 5 frames, identical
gathered observations, identical metric panel).

Motivation: the headline number (RELAX 0.119, -42%) uses NO learning, while the
Direction-III thesis positioned the learned set-diffusion prior as "mandatory". The
two were never compared under the same time budget (RELAX got 2.2*Tp + multiframe;
DPS got W=3s + 1 frame). This closes that gap.

Fairness: same episodes, same frames, same window span (2.2*Tp), and the SAME
subsampled observations (m_cap) are fed to BOTH estimators. Only the estimator
differs (learned DPS posterior-mean field vs non-learned RELAX). RELAX is also run at
full resolution (from main_results*.json) for reference, and we report ST-DF-GP +
oracle. Note: DPS's GP-marginal likelihood is O(m^3) per diffusion step, so the
shared m_cap is small (~200) — itself a finding about the learned approach.

Run: python3 scripts/d30_symmetric_dps.py [--smoke N] [--frames F] [--nsamp K] [--mcap M]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d18_relax_estimator import relax_step, MCAP
from d17_recursive_estimator import refine_params
from d23_main_results import K_TP, N_FRAMES, panel
from d26_relax_iterative import iterative_relax
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser

CKPT = 'results/direction3/setdiff_prior_v2.pt'
GUID = 0.1


def subsample(P, z, m_cap, seed=0):
    if len(z) <= m_cap:
        return P, z
    idx = np.random.default_rng(seed).choice(len(z), m_cap, replace=False)
    return P[idx], z[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='limit to first N episodes')
    ap.add_argument('--frames', type=int, default=N_FRAMES)
    ap.add_argument('--nsamp', type=int, default=8)
    ap.add_argument('--mcap', type=int, default=200)
    ap.add_argument('--out', default='results/direction3/symmetric_dps.json')
    a = ap.parse_args()

    ck = torch.load(CKPT, map_location=DP.DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    print(f"device={DP.DEVICE}  mcap={a.mcap}  nsamp={a.nsamp}  frames={a.frames}")

    gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
    relax_full = json.load(open('results/direction3/main_results.json'))      # RELAX 1p @ MCAP=3000
    relax_iter = json.load(open('results/direction3/main_results_iter.json'))  # RELAX 3p @ MCAP=3000
    tags = list(gp.keys())
    if a.smoke:
        tags = tags[:a.smoke]

    out = {}
    t0 = time.time()
    for ti, tag in enumerate(tags):
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        frames = np.linspace(i0, i1, a.frames).astype(int)
        dps_p, rel1_p, rel3_p = [], [], []
        for i_star in frames:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            # SHARED subsampled observations fed to both estimators (fair head-to-head)
            Ps, zs = subsample(P, z, a.mcap, seed=0)
            # --- learned DPS posterior-mean field ---
            Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, depth,
                               guidance=GUID, n=a.nsamp, seed=0)
            flds = []
            for b in range(a.nsamp):
                f, _ = DP.atoms_to_field(Xd[b], stats, Ps, zs, depth, target)
                if f is not None:
                    flds.append(f)
            bma = np.mean(flds, axis=0) if flds else np.zeros_like(target['gt'])
            dps_p.append(panel(bma, target['gt'], Hs))
            # --- non-learned RELAX on the SAME (Ps, zs) ---
            kx, ky = relax_step(np.zeros(0), np.zeros(0), Ps, zs, depth)
            _, _, fld1 = refine_params(kx, ky, Ps, zs, depth, target)
            rel1_p.append(panel(fld1, target['gt'], Hs))
            fld3, _ = iterative_relax(Ps, zs, depth, target, 3)
            rel3_p.append(panel(fld3, target['gt'], Hs))
        agg = lambda L_, k: (float(np.mean([d[k] for d in L_])), float(np.std([d[k] for d in L_])))
        out[tag] = dict(
            Tobs=Tobs,
            dps={k: agg(dps_p, k) for k in ['nrmse', 'P', 'VR', 'skill']},
            relax1_shared={k: agg(rel1_p, k) for k in ['nrmse', 'P', 'VR', 'skill']},
            relax3_shared={k: agg(rel3_p, k) for k in ['nrmse', 'P', 'VR', 'skill']},
            relax1_full=relax_full[tag]['deployable']['nrmse'][0],
            relax3_full=relax_iter[tag]['deployable']['nrmse'][0],
            stdfgp=gp[tag]['fused']['nRMSE'],
            oracle=relax_full[tag]['oracle']['nrmse'][0])
        d = out[tag]
        print(f"[{ti+1:2d}/{len(tags)}] {tag:12s} Tobs={Tobs:4.0f}s  "
              f"DPS {d['dps']['nrmse'][0]:.3f}(P{d['dps']['P'][0]:.2f}) | "
              f"RELAX1 {d['relax1_shared']['nrmse'][0]:.3f} RELAX3 {d['relax3_shared']['nrmse'][0]:.3f} "
              f"(shared m≤{a.mcap}) | RELAX3_full {d['relax3_full']:.3f} | "
              f"GP {d['stdfgp']:.3f} | ora {d['oracle']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    json.dump(out, open(a.out, 'w'), indent=2)

    # aggregate per sea-state
    seas = {}
    for tag, r in out.items():
        s = f"{tag.split('_')[0]}_{tag.split('_')[1]}"; seas.setdefault(s, []).append(r)
    print(f"\n=== per sea-state (shared m≤{a.mcap}, window {K_TP}*Tp, {a.frames} frames) ===")
    print(f"{'sea':9s} {'DPS':>7} {'RELAX1':>7} {'RELAX3':>7} | {'RELAX3_full':>11} {'ST-DF-GP':>9} {'oracle':>7}")
    A = {k: [] for k in ['dps', 'r1', 'r3', 'r3f', 'gp', 'ora']}
    for s in sorted(seas):
        rs = seas[s]
        m = lambda f: float(np.mean([f(r) for r in rs]))
        dps = m(lambda r: r['dps']['nrmse'][0]); r1 = m(lambda r: r['relax1_shared']['nrmse'][0])
        r3 = m(lambda r: r['relax3_shared']['nrmse'][0]); r3f = m(lambda r: r['relax3_full'])
        g = m(lambda r: r['stdfgp']); o = m(lambda r: r['oracle'])
        print(f"{s:9s} {dps:7.3f} {r1:7.3f} {r3:7.3f} | {r3f:11.3f} {g:9.3f} {o:7.3f}")
        A['dps'].append(dps); A['r1'].append(r1); A['r3'].append(r3)
        A['r3f'].append(r3f); A['gp'].append(g); A['ora'].append(o)
    mm = lambda k: np.mean(A[k])
    print(f"\nOVERALL  DPS {mm('dps'):.3f} | RELAX1 {mm('r1'):.3f} | RELAX3 {mm('r3'):.3f} (shared) "
          f"|| RELAX3_full {mm('r3f'):.3f} | ST-DF-GP {mm('gp'):.3f} | oracle {mm('ora'):.3f}")
    print(f"saved {a.out}")


if __name__ == '__main__':
    main()
