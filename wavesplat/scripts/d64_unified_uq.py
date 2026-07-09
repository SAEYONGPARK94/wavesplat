"""
d64 — Unified physically-grounded predictive UQ for phase-resolved forecasting (#2).

Emerges from the predictable-zone test (predictable_zone_test_VERDICT.md): the FIM/CRB field-σ of
the ESTIMATED components is OVER-CONFIDENT downstream, because it ignores the unidentified energy
that propagates into the forecast point at the group velocity. The honest fix is a UQ that unifies
the two views:

    σ_total²(x,y,t*) = c·σ_FIM²  +  (1−c)·σ_sea²,      c(x,y,t*) = exp(−d_geo / ℓ)

where
  - σ_FIM = CRB field-std from the estimated components' Fisher information (near-field, parameter
    uncertainty; a real lower bound where the spectrum is identified),
  - σ_sea = Hs/4 = the zero-predictor std (far-field ceiling; what you get with no information),
  - d_geo = distance to the observed energy advected forward by the components' group velocities,
  - c ∈ (0,1] = group-velocity COVERAGE (1 = energy observed & reached → trust FIM; 0 = unreached →
    fall back to the sea-state variance).
One length scale ℓ (+ an optional global temperature s) is calibrated on a held-OUT split.

This gives the classical DEPLOYABLE estimator (RELAX) a calibrated predictive uncertainty it
previously lacked (RELAX/F1 provide NO UQ), grounded in physics, validated on NONLINEAR HOS
forecasts. Baseline = FIM-only UQ (global-temperature-scaled), to show it cannot be made calibrated
across forecast lead by a single temperature (the miscalibration is spatial/lead-structured).

Run: python3 scripts/d64_unified_uq.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d62_predictable_zone_test as Z
from wavesplat.model.tnse import render_wave_field

SEAS = [('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3]
CAL_SEEDS, TEST_SEEDS = [0, 1], [2, 3]
HORIZONS = [0.0, 2.0, 4.0, 6.0]


def collect():
    """Run the deployable estimator on HOS windows; collect per-cell (err, sigma_FIM, d_geo, sea_std,
    sea, seed, horizon) over forecast horizons."""
    Z.HORIZONS = HORIZONS
    gx = np.linspace(0.0, 320.0, 90); gy = np.linspace(-110.0, 110.0, 64)
    recs = []
    t0 = time.time()
    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            w = Z.build_window(Hs, Tp, seed, gx, gy)
            kx, ky, A, phi = Z.estimate(w['P'], w['z'], w['t_now'])
            theta = np.column_stack([A, kx, ky, phi])
            sea_std = float(w['gt_at'][0][1].std())            # σ_sea (m) = surface std (= Hs_true/4)
            for h in HORIZONS:
                step = int(round(h / Z.DT_FRAME))
                if step not in w['gt_at']:
                    continue
                t_star, GT = w['gt_at'][step]
                fcast = render_wave_field(kx, ky, A, phi, Z.DEPTH, gx, gy, t_star)
                err = np.abs(fcast - GT)
                sig = Z.fim_sigma_map(theta, w['P'][:, 0], w['P'][:, 1], w['P'][:, 2], gx, gy, t_star)
                dgeo = Z.geo_distance_map(kx, ky, w['P'][:, 0], w['P'][:, 1], w['P'][:, 2], gx, gy, t_star)
                recs.append(dict(tag=tag, seed=seed, h=h, err=err.ravel(), sigFIM=sig.ravel(),
                                 dgeo=dgeo.ravel(), sea_std=sea_std))
            print(f"  [{tag} s{seed}] collected ({time.time()-t0:.0f}s)", flush=True)
    return recs


def cov(err, sigma):
    return float(np.mean(err < sigma)), float(np.mean(err < 2 * sigma))


def fit_temperature(err, sigma, target=0.6827):
    """global multiplier s so that mean(err < s*sigma) == target (bisection)."""
    lo, hi = 1e-3, 1e3
    for _ in range(60):
        s = np.sqrt(lo * hi)
        if np.mean(err < s * sigma) < target:
            lo = s
        else:
            hi = s
    return np.sqrt(lo * hi)


def unified_sigma(sigFIM, dgeo, sea_std, ell):
    c = np.exp(-dgeo / ell)
    return np.sqrt(c * sigFIM ** 2 + (1 - c) * sea_std ** 2)


def main():
    out = 'results/direction3'
    recs = collect()
    for r in recs:
        r['sea_arr'] = np.full_like(r['err'], r['sea_std'])

    def pool(split, keys):
        sub = [r for r in recs if r['seed'] in split]
        return {k: np.concatenate([r[k if k != 'sea' else 'sea_arr'] for r in sub]) for k in keys}

    cal = pool(CAL_SEEDS, ['err', 'sigFIM', 'dgeo', 'sea'])
    # ---- calibrate FIM-only: just a global temperature to hit 0.68 ----
    s_fim = fit_temperature(cal['err'], cal['sigFIM'])
    # ---- calibrate unified: grid ℓ, for each bisect global s to hit 0.68, pick ℓ best on ±2σ ----
    wl_dom = 2 * np.pi / 0.06                                  # ~dominant wavelength scale (m), k~0.06
    ell_grid = [0.25 * wl_dom, 0.5 * wl_dom, wl_dom, 2 * wl_dom, 30.0, 60.0, 120.0]
    best = None
    for ell in ell_grid:
        su = unified_sigma(cal['sigFIM'], cal['dgeo'], cal['sea'], ell)
        s = fit_temperature(cal['err'], su)
        c1, c2 = cov(cal['err'], s * su)
        score = abs(c2 - 0.9545)
        if best is None or score < best[0]:
            best = (score, ell, s, c1, c2)
    _, ell, s_uni, c1_cal, c2_cal = best

    # ---- evaluate on TEST split: overall + per-horizon coverage, spatial corr ----
    from scipy.stats import spearmanr
    te = [r for r in recs if r['seed'] in TEST_SEEDS]
    def ev(rsub):
        e = np.concatenate([r['err'] for r in rsub])
        sf = np.concatenate([r['sigFIM'] for r in rsub])
        dg = np.concatenate([r['dgeo'] for r in rsub])
        ss = np.concatenate([r['sea_arr'] for r in rsub])
        sig_fim = s_fim * sf
        sig_uni = s_uni * unified_sigma(sf, dg, ss, ell)
        c1f, c2f = cov(e, sig_fim); c1u, c2u = cov(e, sig_uni)
        return dict(n=int(e.size),
                    fim=dict(cov1=c1f, cov2=c2f, corr=float(spearmanr(sig_fim, e).correlation)),
                    uni=dict(cov1=c1u, cov2=c2u, corr=float(spearmanr(sig_uni, e).correlation)))
    overall = ev(te)
    per_h = {}
    for h in HORIZONS:
        rs = [r for r in te if r['h'] == h]
        if rs:
            per_h[str(h)] = ev(rs)

    res = dict(calibration=dict(ell=ell, s_fim=s_fim, s_uni=s_uni, cov_cal=[c1_cal, c2_cal],
                                cal_seeds=CAL_SEEDS, test_seeds=TEST_SEEDS, seas=SEAS),
               overall=overall, per_horizon=per_h)
    json.dump(res, open(f'{out}/unified_uq.json', 'w'), indent=2)

    print("\n================ UNIFIED PHYSICS UQ vs FIM-only (HOS forecast, TEST split) ================")
    print(f"  calibrated on seeds {CAL_SEEDS}, tested on {TEST_SEEDS}.  ℓ={ell:.0f} m  "
          f"(both globally temperature-matched to ±1σ=0.68 on calibration)")
    print(f"\n  OVERALL ({overall['n']} cells):   (ideal ±1σ 0.683, ±2σ 0.954)")
    print(f"    FIM-only :  ±1σ {overall['fim']['cov1']:.3f}  ±2σ {overall['fim']['cov2']:.3f}  "
          f"corr(σ,|e|) {overall['fim']['corr']:+.3f}")
    print(f"    UNIFIED  :  ±1σ {overall['uni']['cov1']:.3f}  ±2σ {overall['uni']['cov2']:.3f}  "
          f"corr(σ,|e|) {overall['uni']['corr']:+.3f}")
    print("\n  coverage vs forecast lead (the structured miscalibration):")
    print("    lead     FIM ±1σ / ±2σ        UNIFIED ±1σ / ±2σ")
    for h in HORIZONS:
        if str(h) in per_h:
            f = per_h[str(h)]['fim']; u = per_h[str(h)]['uni']
            print(f"    {h:.0f}s      {f['cov1']:.2f} / {f['cov2']:.2f}          {u['cov1']:.2f} / {u['cov2']:.2f}")
    # honest verdict line
    dev_fim = np.mean([abs(per_h[str(h)]['fim']['cov1'] - 0.683) for h in HORIZONS if str(h) in per_h])
    dev_uni = np.mean([abs(per_h[str(h)]['uni']['cov1'] - 0.683) for h in HORIZONS if str(h) in per_h])
    print(f"\n  mean |±1σ coverage − 0.68| across lead:  FIM {dev_fim:.3f}   UNIFIED {dev_uni:.3f}  "
          f"(lower=better calibrated across lead)")
    make_fig(per_h, out)
    print(f"\nsaved {out}/unified_uq.json")


def make_fig(per_h, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hs = [float(h) for h in sorted(per_h, key=float)]
    f1 = [per_h[str(h)]['fim']['cov1'] for h in hs]
    u1 = [per_h[str(h)]['uni']['cov1'] for h in hs]
    f2 = [per_h[str(h)]['fim']['cov2'] for h in hs]
    u2 = [per_h[str(h)]['uni']['cov2'] for h in hs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].axhline(0.683, color='k', ls='--', lw=1, label='ideal 0.68')
    ax[0].plot(hs, f1, 'o-', color='tab:red', label='FIM-only')
    ax[0].plot(hs, u1, 's-', color='tab:green', label='unified FIM⊕geo')
    ax[0].set_title('±1σ coverage vs forecast lead'); ax[0].set_xlabel('lead [s]'); ax[0].set_ylabel('coverage')
    ax[0].set_ylim(0.4, 1.0); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].axhline(0.954, color='k', ls='--', lw=1, label='ideal 0.95')
    ax[1].plot(hs, f2, 'o-', color='tab:red', label='FIM-only')
    ax[1].plot(hs, u2, 's-', color='tab:green', label='unified FIM⊕geo')
    ax[1].set_title('±2σ coverage vs forecast lead'); ax[1].set_xlabel('lead [s]'); ax[1].set_ylabel('coverage')
    ax[1].set_ylim(0.6, 1.0); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle('Unified physics UQ stays calibrated across forecast lead; FIM-only drifts '
                 '(both temperature-matched on calibration)', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{out}/fig_unified_uq.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}/fig_unified_uq.png")


if __name__ == '__main__':
    main()
