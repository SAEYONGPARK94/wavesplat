"""Does the pointwise-MEDIAN FIELD across 3 blind bank-MMSE fits (seeds
s, s+1000, s+2000 -- the deployed MedianEnsembleUpdater offsets) beat or match
the single-seed runs on the same 3 real windows?  This is the direct evidence
for choosing the ensemble default of the operational bank-MMSE port."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

sys.path.insert(0, 'tmp/bankmmse_port_2026-07-29')
from measure_blind_vs_frozen import dense_gt_std, score, WINDOWS

OUT = 'tmp/bankmmse_port_2026-07-29/m3_field.json'
BASE_SEED = 42
OFFSETS = (0, 1000, 2000)


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
        t = time.time()
        fields = []
        singles = {}
        for off in OFFSETS:
            s = BASE_SEED + off
            fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                    sigma0=0.05 * Hs_obs, kp_hint=None, seed=s)
            maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
            fields.append(maps)
            singles[str(s)] = score(maps, sc, sg)
            print(f"  member s={s:<5} now {singles[str(s)]['nowcast_nrmse']:.4f} "
                  f"mean {singles[str(s)]['mean_nrmse']:.4f}", flush=True)
        med = np.median(np.stack(fields, axis=0), axis=0)
        r = score(med, sc, sg)
        r['seconds_total'] = round(time.time() - t, 1)
        print(f"  m3 FIELD      now {r['nowcast_nrmse']:.4f} "
              f"mean {r['mean_nrmse']:.4f} ({r['seconds_total']}s)", flush=True)
        results[key] = dict(members=singles, m3_field=r)

    json.dump(results, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
