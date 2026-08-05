"""LOCAL (near-observation) rescoring of the rolling-refit streaming validation.

Motivated by the user's (correct) reading that global nRMSE ~ 1 means no skill:
the global grid contains many pixels far from the LiDAR fan, and the 2026-07-27
range-wall finding showed global averages dilute near-sensor skill.  This rerun
scores the SAME arms on distance masks around the observations actually
available at each evaluation time (arm-independent masks), alongside the global
score and -- crucially -- the ZERO-PREDICTOR line on the same mask, so "skill
vs no-skill" is explicit in every regime:

    nRMSE(arm, mask) vs nRMSE(zero, mask) = rms(GT[mask]) / sg

Both normalized by the same global dense-GT std (sg) as every earlier number,
so all columns are directly comparable.  Masks: <=25 m and <=50 m from any
observation in the trailing 40 s of the stream at that evaluation time.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_rolling_refit as rr   # sets up ROOT/scripts paths and chdir

OUT = 'tmp/bankmmse_port_2026-07-29/rolling_refit_local.json'
RADII = [25.0, 50.0]
W_MAX = rr.W_MAX
CADENCES = rr.CADENCES


def obs_distance_field(st, tau, shape_grid):
    """Min distance from each grid pixel to any obs in the trailing 40 s
    (uncapped stream obs -- the informed footprint, arm-independent)."""
    from scipy.spatial import cKDTree
    t_abs = st['t0'] + tau
    m = (st['P_all'][:, 2] >= t_abs - W_MAX) & (st['P_all'][:, 2] <= t_abs + 1e-9)
    O = st['P_all'][m, :2]
    Xq, Yq = shape_grid
    px = np.column_stack([Xq.ravel(), Yq.ravel()])
    d, _ = cKDTree(O).query(px, k=1)
    return d.reshape(Xq.shape)


def main():
    results = {}
    for record, win in rr.WINDOWS:
        key = f'{record}_w{win}'
        st = rr.build_stream(record, win)
        rr.assert_replication(st, record, win)
        taus = [float(v) for v in np.arange(0.0, st['lead_max'] + 1e-9, 1.0)]
        Xq, Yq = np.meshgrid(st['gxq'], st['gyq'])
        dists = {tau: obs_distance_field(st, tau, (Xq, Yq)) for tau in taus}

        fit_cache, render_cache, timing = {}, {}, []
        entry = {'lead_max': st['lead_max'], 'arms': {}, 'zero': {}}

        # zero-predictor baselines per mask (arm-independent)
        for D in [None] + RADII:
            tag = 'global' if D is None else f'le{int(D)}m'
            zc, npx_used = [], []
            for tau in taus:
                j = int(round(tau / st['dt']))
                g = st['Zq'][j]
                m = np.isfinite(g)
                if D is not None:
                    m &= dists[tau] <= D
                npx_used.append(int(m.sum()))
                zc.append(float(np.sqrt(np.mean(g[m] ** 2)) / st['sg'])
                          if m.sum() >= 30 else float('nan'))
            entry['zero'][tag] = dict(curve=zc,
                                      sustained_mean=float(np.nanmean(zc)),
                                      mask_px=npx_used)

        for arm in ['frozen'] + [f'refit{int(R)}' for R in CADENCES]:
            R = None if arm == 'frozen' else float(arm[5:])
            curves = {('global' if D is None else f'le{int(D)}m'): []
                      for D in [None] + RADII}
            for tau in taus:
                t_f = st['t0'] if R is None else \
                    st['t0'] + R * int(np.floor(tau / R + 1e-9))
                fits = rr.fit_at(st, t_f, fit_cache, timing)
                pred = rr.render_med(st, fits, tau, render_cache)
                j = int(round(tau / st['dt']))
                g = st['Zq'][j]
                mfin = np.isfinite(g)
                for D in [None] + RADII:
                    tag = 'global' if D is None else f'le{int(D)}m'
                    m = mfin if D is None else mfin & (dists[tau] <= D)
                    curves[tag].append(
                        float(np.sqrt(np.mean((pred[m] - g[m]) ** 2)) / st['sg'])
                        if m.sum() >= 30 else float('nan'))
            entry['arms'][arm] = {tag: dict(curve=c,
                                            sustained_mean=float(np.nanmean(c)))
                                  for tag, c in curves.items()}

        results[key] = entry
        f, r2 = entry['arms']['frozen'], entry['arms']['refit2']
        z = entry['zero']
        print(f"{key}: le25m frozen {f['le25m']['sustained_mean']:.3f} / "
              f"refit2 {r2['le25m']['sustained_mean']:.3f} / zero "
              f"{z['le25m']['sustained_mean']:.3f} | global frozen "
              f"{f['global']['sustained_mean']:.3f} / refit2 "
              f"{r2['global']['sustained_mean']:.3f} / zero "
              f"{z['global']['sustained_mean']:.3f}", flush=True)
        json.dump(results, open(OUT, 'w'), indent=1)

    print('\nSUMMARY sustained mean (record avg):', flush=True)
    summary = {}
    for record in ('AA15a', 'AA15b', 'YS17'):
        ks = [k for k in results if k.startswith(record)]
        row = {}
        for tag in ['global'] + [f'le{int(D)}m' for D in RADII]:
            row[tag] = {a: float(np.mean(
                [results[k]['arms'][a][tag]['sustained_mean'] for k in ks]))
                for a in results[ks[0]]['arms']}
            row[tag]['zero'] = float(np.mean(
                [results[k]['zero'][tag]['sustained_mean'] for k in ks]))
        summary[record] = row
        for tag, v in row.items():
            print(f"  {record} {tag:>7}: " + ' '.join(
                f"{a}={x:.3f}" for a, x in v.items()), flush=True)
    results['summary_record_mean'] = summary
    json.dump(results, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
