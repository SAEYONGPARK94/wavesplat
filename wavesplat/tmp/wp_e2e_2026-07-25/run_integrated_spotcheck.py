"""Integrated-arm spot check (PREREG §5a): ft-ckpt learned-init under the phase-fixed tracker.

Reuses tmp/wp_integration_2026-07-25/run_arms._worker VERBATIM (byte-identical tracker treatment
to Stage-1) by monkeypatching its module-global CACHE to the FINE-TUNED atoms. 10 crowded episodes
x 5 paired lanes, learned arm only. Paired vs the Stage-1 OLD-ckpt learned results in analyze.py.
4 workers, nice -10 (set by caller), threads=1.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/wp_integration_2026-07-25')
import run_arms as ra

FT_CACHE = 'tmp/wp_e2e_2026-07-25/ft_atoms_cache.jsonl'
OUT = 'tmp/wp_e2e_2026-07-25/ft_arm_runs.jsonl'
LANES = ra.LANES
SPOT10 = [
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_slope.h5',
    'episode_BF4_15comp_seed1_uniform.h5', 'episode_BF4_15comp_seed1_slope.h5',
    'episode_BF4_30comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_slope.h5',
    'episode_BF4_30comp_seed1_uniform.h5', 'episode_BF4_30comp_seed1_slope.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_slope.h5',
]


def main():
    # load FT atoms into run_arms' module-global CACHE (workers inherit via fork)
    for line in open(FT_CACHE):
        d = json.loads(line)
        if d.get('err') is None and d.get('kx') is not None:
            ra.CACHE[(d['episode'], d['diff_seed'])] = dict(
                kx=d['kx'], ky=d['ky'], energy=d['energy'])
    print(f"loaded {len(ra.CACHE)} ft atom sets", flush=True)

    jobs = [(ep, es, ds, 'learned') for ep in SPOT10 for (es, ds) in LANES]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['episode'], d['est_seed'], d['diff_seed'], d['arm']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"{len(SPOT10)} eps x {len(LANES)} lanes learned; {len(jobs)} runs ({len(done)} done)",
          flush=True)
    t0 = time.time(); n = 0
    with open(OUT, 'a', buffering=1) as f, Pool(4) as pool:
        for res in pool.imap_unordered(ra._worker, jobs):
            f.write(json.dumps(res) + '\n'); n += 1
            tag = ('ERR ' + res['err'][:40]) if res['err'] else \
                (f"init {res['init_nrmse']:.3f} avg {res['avg_nRMSE']:.3f} N{res['final_n']}")
            print(f"[{n:3d}/{len(jobs)}] {res['episode']:38s} "
                  f"e{res['est_seed']}d{res['diff_seed']} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"done: {n} runs, {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
