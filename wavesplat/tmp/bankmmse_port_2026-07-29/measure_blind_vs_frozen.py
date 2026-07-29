"""Port-decision measurement: can the bank-MMSE research arm keep its real-sea
skill when every GT-derived input is replaced by an observation-only one?

Arms per window (same frozen scene_real windows as the 2026-07-27 GIF ledger):
  frozen : mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05*Hs_GT, kp_hint=kp_GT,
           seed=i0)  -- the exact published research call (sanity vs ledger).
  blind  : sigma0 = 0.2*std(z)  (= 0.05 * Hs_obs with Hs_obs = 4*std(z)),
           kp_hint = None (built-in two-stage beamform zoom), seed in {42,1042,2042}
           -- the only parametrization an operational GT-free path may use.

Scores use the identical protocol-lead grid + normalization of
tmp/real_gif_2026-07-27/make_real_estimation_gif.py (sg = std of the dense GT
over the full lead span), so numbers are directly comparable to the ledger
(tmp/confirm2_2026-07-27/real_gif_info.json).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

OUT = 'tmp/bankmmse_port_2026-07-29/blind_vs_frozen.json'
WINDOWS = [('AA15a', 5), ('AA15b', 3), ('YS17', 0)]
BLIND_SEEDS = [42, 1042, 2042]


def dense_gt_std(record, win):
    """sg exactly as the GIF script computes it (dense GT std over lead span)."""
    from d3_falcon_real_operator import RECORDS
    from d3_sa_multirecord import MRecord
    cfg = RECORDS[record]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])
    Zfut = rec.window(i0, n_lead + 1)
    sq = max(1, int(round(4.0 / rec.dx)))
    Z = Zfut[:, ::sq, ::sq]
    return float(np.std(Z[np.isfinite(Z)]))


def score(maps, sc, sg):
    nr, cc = [], []
    for li in range(len(sc['leads'])):
        p = maps[li]; g = sc['maps_gt'][li]; m = np.isfinite(g)
        nr.append(float(np.sqrt(np.mean((p[m] - g[m]) ** 2)) / sg))
        cc.append(float(np.corrcoef(p[m], g[m])[0, 1]))
    return dict(nowcast_nrmse=nr[0], nowcast_corr=cc[0],
                mean_nrmse=float(np.mean(nr)), mean_corr=float(np.mean(cc)),
                nrmse=nr)


def kp_two_stage(P, z, depth):
    """Replicates mmse_bank_estimate's internal kp estimate, for logging only."""
    from d18_relax_estimator import beamform_top
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=3.5 * kp, n=120)
    return max(np.hypot(bx, by), 1e-3)


def main():
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate

    results = {}
    for rec, win in WINDOWS:
        key = f'{rec}_w{win}'
        print(f'== {key}', flush=True)
        sc = cg.scene_real(rec, win)
        sg = dense_gt_std(rec, win)
        npx, nl = sc['npx'], len(sc['leads'])
        Hs_obs = 4.0 * float(np.std(sc['z']))
        kp_est = kp_two_stage(sc['P'], sc['z'], sc['depth'])
        entry = dict(Hs_gt=float(sc['Hs']), Hs_obs=Hs_obs,
                     kp_gt=float(sc['kp']), kp_est_two_stage=float(kp_est),
                     gt_std_dense=sg, n_obs=int(len(sc['z'])))

        t = time.time()
        fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                sigma0=0.05 * sc['Hs'], kp_hint=sc['kp'],
                                seed=sc['seed'])
        maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
        entry['frozen'] = score(maps, sc, sg)
        entry['frozen']['seconds'] = round(time.time() - t, 1)
        print(f"  frozen        now {entry['frozen']['nowcast_nrmse']:.4f} "
              f"mean {entry['frozen']['mean_nrmse']:.4f} "
              f"({entry['frozen']['seconds']}s)", flush=True)

        entry['blind'] = {}
        for s in BLIND_SEEDS:
            t = time.time()
            fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                    sigma0=0.05 * Hs_obs, kp_hint=None, seed=s)
            maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
            r = score(maps, sc, sg)
            r['seconds'] = round(time.time() - t, 1)
            entry['blind'][str(s)] = r
            print(f"  blind s={s:<5} now {r['nowcast_nrmse']:.4f} "
                  f"mean {r['mean_nrmse']:.4f} ({r['seconds']}s)", flush=True)
        nows = [entry['blind'][str(s)]['nowcast_nrmse'] for s in BLIND_SEEDS]
        means = [entry['blind'][str(s)]['mean_nrmse'] for s in BLIND_SEEDS]
        entry['blind_median'] = dict(nowcast_nrmse=float(np.median(nows)),
                                     mean_nrmse=float(np.median(means)),
                                     nowcast_spread=[min(nows), max(nows)])
        print(f"  blind median  now {entry['blind_median']['nowcast_nrmse']:.4f} "
              f"mean {entry['blind_median']['mean_nrmse']:.4f} "
              f"spread {entry['blind_median']['nowcast_spread']}", flush=True)
        results[key] = entry

    json.dump(results, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
