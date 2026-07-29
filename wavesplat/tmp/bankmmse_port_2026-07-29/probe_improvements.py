"""EXPLORATORY probe battery: which known bank-MMSE levers improve the real-sea
windows beyond the just-deployed blind median-of-3 baseline?

All variants run under the DEPLOYABLE constraint (observation-only parameters,
sigma0 = 0.2*std(z), kp from obs unless the variant is explicitly a GT
diagnostic), seeds {42, 1042, 2042}, pointwise-median FIELD -- i.e. exactly the
deployed combination rule, so deltas are attributable to the lever alone.

EXPLORATORY: same 3 ledger windows the port was verified on; nothing here is
pre-registered.  Winners require fresh-window confirmation before deployment.

Baseline (m3_field.json): AA15a_w5 0.4982/0.6206, AA15b_w3 0.6657/0.7025,
YS17_w0 0.7180/0.8705 (nowcast/lead-mean protocol nRMSE).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

sys.path.insert(0, 'tmp/bankmmse_port_2026-07-29')
from measure_blind_vs_frozen import dense_gt_std, score, kp_two_stage, WINDOWS

OUT = 'tmp/bankmmse_port_2026-07-29/probe_improvements.json'
SEEDS = [42, 1042, 2042]


def kp_three_stage(P, z, depth):
    """Two-stage zoom + one more refinement stage (finer grid around kp2)."""
    from d18_relax_estimator import beamform_top
    kp = kp_two_stage(P, z, depth)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=1.5 * kp, n=160)
    return max(np.hypot(bx, by), 1e-3)


def run_variant(sc, sg, call_kw, seeds=SEEDS, select='median'):
    """Median (or min-vr-selected) FIELD over seeds; returns scores + diag."""
    from d3_sa_nomp import mmse_bank_estimate
    npx, nl = sc['npx'], len(sc['leads'])
    s0 = 0.2 * float(np.std(sc['z']))
    fields, vrs = [], []
    t0 = time.time()
    for s in seeds:
        diag = {}
        fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                sigma0=s0, seed=s, diag_out=diag, **call_kw)
        fields.append(fq[:npx * nl].reshape(nl, *sc['Xq'].shape))
        vrs.append(diag.get('vr'))
    if select == 'median':
        fld = np.median(np.stack(fields, axis=0), axis=0)
    else:                                   # 'min_vr' -- GT-free selection
        fld = fields[int(np.argmin(vrs))]
    r = score(fld, sc, sg)
    r['vr'] = [round(float(v), 4) for v in vrs]
    r['seconds'] = round(time.time() - t0, 1)
    return r


def main():
    import d3_compare_gifs as cg

    base = json.load(open('tmp/bankmmse_port_2026-07-29/m3_field.json'))
    results = {}
    for rec, win in WINDOWS:
        key = f'{rec}_w{win}'
        print(f'== {key}', flush=True)
        sc = cg.scene_real(rec, win)
        sg = dense_gt_std(rec, win)
        kp_est = kp_two_stage(sc['P'], sc['z'], sc['depth'])
        kp3 = kp_three_stage(sc['P'], sc['z'], sc['depth'])
        entry = {'baseline_m3': dict(
            nowcast_nrmse=base[key]['m3_field']['nowcast_nrmse'],
            mean_nrmse=base[key]['m3_field']['mean_nrmse'])}
        print(f"  baseline      now {entry['baseline_m3']['nowcast_nrmse']:.4f} "
              f"mean {entry['baseline_m3']['mean_nrmse']:.4f}", flush=True)

        K5 = [42, 1042, 2042, 3042, 4042]
        variants = [
            ('dual_band',  dict(kp_hint=None, dual_band=True),      SEEDS, 'median'),
            ('bound_wave', dict(kp_hint=None, bound_wave=True),     SEEDS, 'median'),
            ('bw_cross',   dict(kp_hint=None, bound_wave=True,
                                bw_cross=True),                     SEEDS, 'median'),
            ('k5',         dict(kp_hint=None),                      K5,    'median'),
            ('bestof3_vr', dict(kp_hint=None),                      SEEDS, 'min_vr'),
            ('dk15',       dict(kp_hint=None, dk_c=kp_est / 15.0),  SEEDS, 'median'),
            ('band_wide',  dict(kp_hint=None, band=(0.25, 3.0)),    SEEDS, 'median'),
            ('kp3stage',   dict(kp_hint=kp3),                       SEEDS, 'median'),
            ('kp_gt_DIAG', dict(kp_hint=float(sc['kp'])),           SEEDS, 'median'),
        ]

        for name, kw, seeds, sel in variants:
            try:
                r = run_variant(sc, sg, kw, seeds=seeds, select=sel)
            except Exception as e:
                print(f'  {name:<12} FAILED: {e}', flush=True)
                entry[name] = dict(error=str(e))
                continue
            entry[name] = r
            d_now = r['nowcast_nrmse'] - entry['baseline_m3']['nowcast_nrmse']
            d_mean = r['mean_nrmse'] - entry['baseline_m3']['mean_nrmse']
            print(f"  {name:<12} now {r['nowcast_nrmse']:.4f} ({d_now:+.4f}) "
                  f"mean {r['mean_nrmse']:.4f} ({d_mean:+.4f}) "
                  f"[{r['seconds']}s]", flush=True)

        # more observations: same window, cap 1200 instead of the protocol 600
        try:
            sc12 = cg.scene_real(rec, win, m_cap=1200)
            r = run_variant(sc12, sg, dict(kp_hint=None))
            entry['mcap1200'] = r
            d_now = r['nowcast_nrmse'] - entry['baseline_m3']['nowcast_nrmse']
            d_mean = r['mean_nrmse'] - entry['baseline_m3']['mean_nrmse']
            print(f"  {'mcap1200':<12} now {r['nowcast_nrmse']:.4f} ({d_now:+.4f}) "
                  f"mean {r['mean_nrmse']:.4f} ({d_mean:+.4f}) "
                  f"[{r['seconds']}s, n={len(sc12['z'])}]", flush=True)
        except Exception as e:
            print(f'  mcap1200 FAILED: {e}', flush=True)
            entry['mcap1200'] = dict(error=str(e))

        entry['kp_est_two_stage'] = float(kp_est)
        entry['kp_est_three_stage'] = float(kp3)
        entry['kp_gt'] = float(sc['kp'])
        results[key] = entry
        json.dump(results, open(OUT, 'w'), indent=1)

    print('wrote', OUT)


if __name__ == '__main__':
    main()
