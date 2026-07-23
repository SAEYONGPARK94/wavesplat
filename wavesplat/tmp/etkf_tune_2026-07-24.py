"""
etkf_tune_2026-07-24.py — honest held-out ETKF retune on fine-mesh data.

Protocol (pre-registered, tmp/ETKF_FINEMESH_PREREG_2026-07-24.md R5/R6):
  TUNING set  = the 12 paper scenarios at **seed 1**  (never reported)
  REPORT set  = the 12 paper scenarios at **seed 0**  (never used for selection)
  selection metric fixed in advance = mean nRMSE over the 12 tuning episodes,
  each episode's value being the median over filter seeds.
  filter seeds: 3 during selection (A/B), 7 for every reported number (C).

Stages
  A1  analysis grid x additive inflation            (Ne=50, obs_noise=.05, mi=1.01)
  A2  obs_noise x n_ensemble x mult_inflation       (at the A1 winner)
  A3  max_obs                                       (at the A2 winner)
  C   report — retuned + as-published, REPORT set, both meshes, 7 seeds

Every run also records rho(pred,GT) and the variance ratio so a "winner" that is
merely a better zero-predictor can be identified as such.
CPU/numpy only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import json
import argparse
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SCEN = [
    ('BF2_1comp', 'uniform'), ('BF3_1comp', 'uniform'), ('BF3_2comp', 'uniform'),
    ('BF3_2comp', 'slope'), ('BF3_5comp', 'uniform'), ('BF3_5comp', 'slope'),
    ('BF4_1comp', 'uniform'), ('BF4_5comp', 'uniform'), ('BF4_15comp', 'uniform'),
    ('BF4_30comp', 'uniform'), ('BF5_15comp', 'uniform'), ('BF5_30comp', 'uniform'),
]
TUNE = [f'episode_{s}_seed1_{r}.h5' for s, r in SCEN]
REPORT = [f'episode_{s}_seed0_{r}.h5' for s, r in SCEN]
NEW = 'data_finemesh_2026-07-23'
OLD = 'data'
# config actually used by scripts/run_p1_comparison.py::run_etkf
SHIPPED = dict(nx=40, ny=30, n_ensemble=50, mult_inflation=1.01,
               add_inflation=0.005, obs_noise=0.05, max_obs=40)
# config the paper calls "ETKF (tuned)" (tab:etkf_sweep winning row)
PAPER_TUNED = dict(nx=40, ny=30, n_ensemble=100, mult_inflation=1.00,
                   add_inflation=0.005, obs_noise=0.05, max_obs=40)
NPROC = 10


def _job(a):
    from etkf_lib import run_etkf
    ds, ep, cfg, seed = a
    r = run_etkf(os.path.join(ROOT, ds, ep), seed=seed, diagnostics=True, **cfg)
    return dict(ds=ds, ep=ep, cfg=cfg, seed=seed, nRMSE=r['avg_nRMSE'],
                rho=r['avg_rho'], alpha=r['avg_alpha'],
                nRMSE_alpha=r['avg_nRMSE_alpha'], ms=r['avg_ms'])


def evaluate(cfgs, episodes, ds=NEW, seeds=(0, 1, 2), tag=''):
    jobs = [(ds, ep, c, s) for c in cfgs for ep in episodes for s in seeds]
    print(f'  [{tag}] {len(cfgs)} cfgs x {len(episodes)} eps x {len(seeds)} seeds '
          f'= {len(jobs)} runs', flush=True)
    with Pool(NPROC) as p:
        return p.map(_job, jobs, chunksize=1)


def summarise(res):
    by = {}
    for r in res:
        by.setdefault(json.dumps(r['cfg'], sort_keys=True), {}) \
          .setdefault(r['ep'], []).append(r)
    out = {}
    for key, eps in by.items():
        med = {e: float(np.median([x['nRMSE'] for x in v])) for e, v in eps.items()}
        rho = {e: float(np.median([x['rho'] for x in v])) for e, v in eps.items()}
        out[key] = dict(
            per_episode=med, per_episode_rho=rho,
            spread={e: [float(min(x['nRMSE'] for x in v)),
                        float(max(x['nRMSE'] for x in v))] for e, v in eps.items()},
            mean=float(np.mean(list(med.values()))),
            mean_rho=float(np.mean(list(rho.values()))),
            ms=float(np.median([x['ms'] for v in eps.values() for x in v])))
    return out


def winner(summary):
    k = min(summary, key=lambda k: summary[k]['mean'])
    return json.loads(k), summary[k]


def load(stage):
    return json.load(open(os.path.join(ROOT, 'tmp',
                                       f'etkf_tune_stage{stage}_2026-07-24.json')))


def save(stage, obj):
    p = os.path.join(ROOT, 'tmp', f'etkf_tune_stage{stage}_2026-07-24.json')
    json.dump(obj, open(p, 'w'), indent=1)
    print('written', p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    args = ap.parse_args()

    if args.stage == 'A1':
        cfgs = [dict(SHIPPED, nx=g[0], ny=g[1], add_inflation=ai)
                for g in [(40, 30), (60, 45), (80, 60), (120, 90), (160, 120)]
                for ai in (0.0, 0.005, 0.02)]
        res = evaluate(cfgs, TUNE, tag='A1')
        save('A1', dict(raw=res, summary=summarise(res)))

    elif args.stage == 'A2':
        b, s = winner(load('A1')['summary'])
        print('A1 winner', b, round(s['mean'], 4), 'rho', round(s['mean_rho'], 3), flush=True)
        cfgs = [dict(b, obs_noise=o, n_ensemble=n, mult_inflation=mi)
                for o in (0.02, 0.05, 0.10, 0.20) for n in (50, 100)
                for mi in (1.0, 1.01)]
        res = evaluate(cfgs, TUNE, tag='A2')
        save('A2', dict(a1_winner=b, raw=res, summary=summarise(res)))

    elif args.stage == 'A3':
        b, s = winner(load('A2')['summary'])
        print('A2 winner', b, round(s['mean'], 4), 'rho', round(s['mean_rho'], 3), flush=True)
        cfgs = [dict(b, max_obs=m) for m in (20, 40, 100)]
        res = evaluate(cfgs, TUNE, tag='A3')
        save('A3', dict(a2_winner=b, raw=res, summary=summarise(res)))

    elif args.stage == 'C':
        b, s = winner(load('A3')['summary'])
        print('FINAL cfg', b, round(s['mean'], 4), flush=True)
        out = {'final_cfg': b, 'shipped_cfg': SHIPPED, 'paper_tuned_cfg': PAPER_TUNED,
               'tuning_set': TUNE, 'report_set': REPORT,
               'final_cfg_on_tuning_set': s}
        for ds in (NEW, OLD):
            res = evaluate([b, SHIPPED, PAPER_TUNED], REPORT, ds=ds,
                           seeds=(0, 1, 2, 3, 4, 5, 6), tag=f'C-{ds}')
            out[ds] = summarise(res)
        save('C', out)


if __name__ == '__main__':
    main()
