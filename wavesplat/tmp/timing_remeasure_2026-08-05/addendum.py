"""ADDENDUM to the 2026-08-05 timing re-measurement. Run AFTER harness.py finishes
(serial, never concurrently -- CPU contention is the very confound being measured).

Two extra paper claims that the main design does not cover:

  ADD-1  "The tuned ETKF uses n_ens=100 ... twice the nominal configuration
          (n_ens=50, 31.7 ms) ... at the cost of ~4x higher per-frame latency"
          (paper/wavesplat.tex:1374-1378).  results/etkf_tuning_sweep.json says
          ne100/ne50 = 2.81x.  Measure it here under the pinned protocol.

  ADD-2  The BLAS-thread sensitivity itself (debate A-2, 370%).  The paper's
          "on a single CPU thread" is the condition under which its own number is
          NOT reproducible, so the thread count must be reported as provenance.
          Re-measured here inside this study's own harness so the two regimes are
          produced by identical code paths.

ADD-2 deliberately does NOT pin threads: that is the measurement.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = ['episode_BF3_2comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
       'episode_BF4_30comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

CHILD = r'''
import os, sys, time, json, warnings
NT = os.environ["NT"]
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = NT
import numpy as np, torch
torch.set_num_threads(int(NT)); warnings.filterwarnings("ignore")
sys.path.insert(0, "%s")
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.etkf_grid import ETKFGrid
DATA = os.path.join("%s", "data_finemesh_2026-07-23")
out = []
for ep in json.loads(os.environ["EPS"]):
    for ne in json.loads(os.environ["NENS"]):
        np.random.seed(42)
        L = EpisodeDataLoader(os.path.join(DATA, ep))
        gx = L.gt_grid_x.cpu().numpy(); gy = L.gt_grid_y.cpu().numpy()
        egx = np.linspace(gx[0], gx[-1], 40); egy = np.linspace(gy[0], gy[-1], 30)
        b = ETKFGrid(egx, egy, depth=L.depth, n_ensemble=ne,
                     mult_inflation=1.0, add_inflation=0.005)
        t = []
        t_end = L.get_init_window(tp_factor=2.0)[-1].t
        for i, f in enumerate(L):
            x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy(); z = f.z_obs.cpu().numpy()
            a = time.perf_counter(); b.update(x, y, z); c = time.perf_counter()
            if i >= 5 and f.t >= t_end:
                t.append((c - a) * 1e3)
        out.append({"episode": ep, "n_ensemble": ne, "threads": int(NT),
                    "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
                    "mean_ms": float(np.mean(t)), "n": len(t)})
        print(f'  ne={ne} thr={NT} {ep[:34]:<34s} med {np.median(t):8.2f} ms', flush=True)
json.dump(out, open(os.environ["OUTF"], "w"), indent=2)
'''


def run(tag, nt, nens, eps):
    outf = os.path.join(HERE, f'addendum_{tag}.json')
    env = dict(os.environ, NT=str(nt), NENS=json.dumps(nens),
               EPS=json.dumps(eps), OUTF=outf)
    repo = os.path.abspath(os.path.join(HERE, '..', '..'))
    print(f'== {tag}: threads={nt} n_ens={nens} ==', flush=True)
    subprocess.run([sys.executable, '-u', '-c', CHILD % (repo, repo)],
                   env=env, check=True)
    return json.load(open(outf))


if __name__ == '__main__':
    import numpy as np
    a1 = run('ne_sweep', 1, [50, 100], EPS)
    a2 = run('threads16', 16, [100], EPS)
    a2b = run('threads4', 4, [100], EPS)

    def med(rows, ne, thr):
        v = [r['median_ms'] for r in rows if r['n_ensemble'] == ne and r['threads'] == thr]
        return float(np.median(v))

    rep = {
        'ADD1_ensemble_size_1thread': {
            'ne50_median_ms': med(a1, 50, 1), 'ne100_median_ms': med(a1, 100, 1),
            'ratio_ne100_over_ne50': med(a1, 100, 1) / med(a1, 50, 1),
            'paper_claim': '~4x (11.0/132.6 framing); etkf_tuning_sweep.json says 2.81x',
            'rows': a1},
        'ADD2_thread_sensitivity_ne100': {
            't1_median_ms': med(a1, 100, 1), 't4_median_ms': med(a2b, 100, 4),
            't16_median_ms': med(a2, 100, 16),
            'ratio_t16_over_t1': med(a2, 100, 16) / med(a1, 100, 1),
            'ratio_t4_over_t1': med(a2b, 100, 4) / med(a1, 100, 1),
            'rows': a2 + a2b},
        'episodes': EPS,
        'note': 'ADD-2 runs are separate processes by necessity (thread count is set '
                'before numpy import); they are NOT part of the same-session primary '
                'comparison and are reported only as a sensitivity provenance.',
    }
    json.dump(rep, open(os.path.join(HERE, 'addendum_summary.json'), 'w'), indent=2)
    print('\n=== ADDENDUM ===')
    d = rep['ADD1_ensemble_size_1thread']
    print(f"ADD-1 ETKF ne=50 {d['ne50_median_ms']:.2f} ms -> ne=100 "
          f"{d['ne100_median_ms']:.2f} ms = {d['ratio_ne100_over_ne50']:.2f}x "
          f"(paper implies ~4x, sweep says 2.81x)")
    d = rep['ADD2_thread_sensitivity_ne100']
    print(f"ADD-2 ETKF ne=100: 1 thread {d['t1_median_ms']:.2f} ms | 4 threads "
          f"{d['t4_median_ms']:.2f} ms ({d['ratio_t4_over_t1']:.2f}x) | 16 threads "
          f"{d['t16_median_ms']:.2f} ms ({d['ratio_t16_over_t1']:.2f}x)")
