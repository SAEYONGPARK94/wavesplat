"""
Phase 2 — forecast skill (the applied payoff).

From the deployable estimate (components fitted on [t*-T, t*]), the field is forecast
forward by dispersion: ẑ(x,y,t*+Δ). We measure skill vs lead time Δ and vs distance
downstream (the predictable zone), against a persistence baseline (freeze the field
at t*). Ties to the Direction-I predictable-zone.

Run: python3 scripts/d24_forecast.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesplat.evaluation.metrics import pearson_correlation, skill_score
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d17_recursive_estimator import refine_params
from d18_relax_estimator import relax_step, MCAP

K_TP = 2.2


def estimate(L, i_star, depth):
    Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1))
    fstar = L[i_star]
    target = {'gx': L.gt_grid_x.numpy(), 'gy': L.gt_grid_y.numpy(),
              't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': L.Hs_true}
    P, z = gather_frames(L, i_star, n_back)
    kx, ky = relax_step(np.zeros(0), np.zeros(0), P, z, depth)
    kx, ky, _ = refine_params(kx, ky, P, z, depth, target)
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    if len(obs) > MCAP:
        obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    return kx, ky, A, phi


def main():
    tags = ['BF4_5c_s0', 'BF4_15c_s0', 'BF3_5c_s0']
    i_star = 230
    leads = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    out = {}
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        kx, ky, A, phi = estimate(L, i_star, depth)
        z_now = L[i_star].gt_field.numpy()              # persistence reference
        fc, per = [], []
        for dl in leads:
            j = i_star + int(round(dl / 0.1))
            if j >= len(L):
                break
            gt = L[j].gt_field.numpy(); t = float(L[j].t)
            pred = render_wave_field(kx, ky, A, phi, depth, gx, gy, t)
            fc.append(dict(lead=dl, nrmse=float(np.sqrt(np.mean((pred-gt)**2))/Hs),
                           P=float(pearson_correlation(pred, gt)),
                           skill=float(skill_score(pred, gt))))
            per.append(dict(lead=dl, nrmse=float(np.sqrt(np.mean((z_now-gt)**2))/Hs),
                            P=float(pearson_correlation(z_now, gt))))
        # skill vs distance at lead=2s (predictable zone)
        j = i_star + 20; dist_prof = None
        if j < len(L):
            gt = L[j].gt_field.numpy(); pred = render_wave_field(kx, ky, A, phi, depth, gx, gy, float(L[j].t))
            err_col = np.sqrt(np.mean((pred-gt)**2, axis=0)) / Hs    # per-x RMSE
            dist_prof = dict(x=gx.tolist(), err=err_col.tolist())
        out[tag] = dict(Tp=L.Tp_true, forecast=fc, persistence=per, dist_at2s=dist_prof)
        print(f"{tag}:")
        for f, p in zip(fc, per):
            print(f"   +{f['lead']:.1f}s  forecast nRMSE {f['nrmse']:.3f} P{f['P']:.2f} skill{f['skill']:.2f} "
                  f"| persistence nRMSE {p['nrmse']:.3f} P{p['P']:.2f}", flush=True)
    json.dump(out, open('results/direction3/forecast.json', 'w'), indent=2)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    cols = {'BF4_5c_s0': 'tab:green', 'BF4_15c_s0': 'tab:orange', 'BF3_5c_s0': 'tab:blue'}
    for tag in tags:
        r = out[tag]
        ld = [f['lead'] for f in r['forecast']]
        a1.plot(ld, [f['P'] for f in r['forecast']], 'o-', color=cols[tag], label=f'{tag} forecast')
        a1.plot(ld, [p['P'] for p in r['persistence']], '--', color=cols[tag], alpha=0.5)
        if r['dist_at2s']:
            a2.plot(r['dist_at2s']['x'], r['dist_at2s']['err'], '-', color=cols[tag], label=tag)
    a1.set_xlabel('forecast lead time [s]'); a1.set_ylabel('Pearson (forecast vs GT)')
    a1.set_title('Forecast skill vs lead (solid) vs persistence (dashed)'); a1.grid(alpha=0.25); a1.legend(fontsize=8)
    a2.set_xlabel('x distance [m]'); a2.set_ylabel('forecast RMSE/Hs (per column)')
    a2.set_title('Predictable zone: forecast error vs distance (+2s lead)'); a2.grid(alpha=0.25); a2.legend(fontsize=8)
    fig.suptitle('Phase 2 — deployable forecast skill (estimated components propagated by dispersion)', fontsize=12)
    fig.tight_layout(); fig.savefig('results/direction3/fig_forecast.png', dpi=140)
    print('saved forecast.json + fig_forecast.png')


if __name__ == '__main__':
    main()
