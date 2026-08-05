"""CONTENTION PROBE (2026-08-05) -- run LAST, after the primary measurement is saved.

WHY. The 08-05 debate (§A-2) attributed ETKF's 115.88 ms to BLAS *thread count*
("default 16 threads unpinned"), a 370% effect.  This study's addendum measured 16 threads
on an IDLE box at 12.82 ms -- only 1.27x over 1 thread.  So thread count ALONE cannot
produce 115.88 ms, and the debate's causal attribution is not established.

This probe crosses the two candidate causes:

    threads-per-process  x  concurrent processes

If the ~116 ms regime appears only in oversubscribed cells (threads x procs >> cores),
the cause is CPU OVERSUBSCRIPTION -- a property of the batch harness and of what else was
running on the box, not of the algorithm and not of the thread flag by itself.

Deliberately parallel: parallelism is the OBJECT of measurement here, not a confound.
Work is capped at N_FRAMES per child so the oversubscribed cells terminate.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
EP = 'episode_BF4_5comp_seed0_uniform.h5'
N_FRAMES = 60          # timed frames per child (median is stable well below this)
SETTLE_LOAD = 2.0      # wait until 1-min loadavg drops below this before each cell
CELLS = [(1, 1), (16, 1), (1, 4), (4, 4), (16, 4), (16, 8)]

CHILD = r'''
import os, sys, time, json, warnings
NT = os.environ["NT"]
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = NT
import numpy as np, torch
torch.set_num_threads(int(NT)); warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ["REPO"])
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.etkf_grid import ETKFGrid
np.random.seed(42)
L = EpisodeDataLoader(os.path.join(os.environ["REPO"], "data_finemesh_2026-07-23",
                                   os.environ["EP"]))
gx = L.gt_grid_x.cpu().numpy(); gy = L.gt_grid_y.cpu().numpy()
egx = np.linspace(gx[0], gx[-1], 40); egy = np.linspace(gy[0], gy[-1], 30)
b = ETKFGrid(egx, egy, depth=L.depth, n_ensemble=100,
             mult_inflation=1.0, add_inflation=0.005)
t_end = L.get_init_window(tp_factor=2.0)[-1].t
NF = int(os.environ["NF"]); t = []
for i, f in enumerate(L):
    x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy(); z = f.z_obs.cpu().numpy()
    a = time.perf_counter(); b.update(x, y, z); c = time.perf_counter()
    if i >= 5 and f.t >= t_end:
        t.append((c - a) * 1e3)
        if len(t) >= NF: break
sys.stderr.write(json.dumps({"median_ms": float(np.median(t)),
                             "p95_ms": float(np.percentile(t, 95)),
                             "mean_ms": float(np.mean(t)), "n": len(t)}) + "\n")
'''


def settle():
    for _ in range(120):
        if float(open('/proc/loadavg').read().split()[0]) < SETTLE_LOAD:
            return True
        time.sleep(5)
    return False


def cell(nthreads, nprocs, log):
    settled = settle()
    env = dict(os.environ, NT=str(nthreads), REPO=REPO, EP=EP, NF=str(N_FRAMES))
    procs = [subprocess.Popen([sys.executable, '-c', CHILD], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True) for _ in range(nprocs)]
    t0 = time.perf_counter()
    errs = [p.communicate(timeout=1200)[1] for p in procs]
    wall = time.perf_counter() - t0
    meds, p95s = [], []
    for e in errs:
        for line in (e or '').strip().splitlines():
            try:
                d = json.loads(line)
                meds.append(d['median_ms'])
                p95s.append(d['p95_ms'])
            except Exception:
                pass
    r = {'threads_per_proc': nthreads, 'n_procs': nprocs,
         'total_threads_requested': nthreads * nprocs,
         'logical_cpus': os.cpu_count(),
         'median_ms_per_proc': meds,
         'median_ms': float(np.median(meds)) if meds else float('nan'),
         'worst_proc_median_ms': float(np.max(meds)) if meds else float('nan'),
         'p95_ms': float(np.median(p95s)) if p95s else float('nan'),
         'wall_s': wall, 'settled_before_start': settled,
         'n_children_reporting': len(meds)}
    line = (f"threads/proc {nthreads:>2}  procs {nprocs:>2}  "
            f"(requested {r['total_threads_requested']:>3} on {r['logical_cpus']} logical) "
            f"-> median {r['median_ms']:8.2f} ms  worst-proc {r['worst_proc_median_ms']:8.2f} "
            f"p95 {r['p95_ms']:8.2f}  wall {wall:6.1f}s")
    print(line, flush=True)
    log.write(line + '\n')
    log.flush()
    return r


if __name__ == '__main__':
    with open(os.path.join(HERE, 'contention_probe.log'), 'w') as log:
        rows = [cell(nt, npc, log) for nt, npc in CELLS]
        base = rows[0]['median_ms']
        for r in rows:
            r['ratio_vs_1t1p'] = r['median_ms'] / base
        json.dump({'episode': EP, 'n_frames_per_child': N_FRAMES,
                   'baseline_1t1p_ms': base,
                   'debate_A2_claim': 'attributed 115.88 ms to thread count alone (370%)',
                   'cells': rows},
                  open(os.path.join(HERE, 'contention_probe.json'), 'w'), indent=2)
        print('\nratios vs (1 thread, 1 proc):', flush=True)
        for r in rows:
            s = (f"  {r['threads_per_proc']:>2}t x {r['n_procs']:>2}p : "
                 f"{r['ratio_vs_1t1p']:6.2f}x   ({r['median_ms']:8.2f} ms)")
            print(s, flush=True)
            log.write(s + '\n')
