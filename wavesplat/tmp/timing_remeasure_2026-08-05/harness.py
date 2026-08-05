"""Timing re-measurement harness (2026-08-05). Pre-registration: ./PREREG.md

WHY THIS FILE EXISTS
--------------------
Every timing number in the paper was produced by `time.time()` on a box whose
CLOCK_REALTIME steps ~1.25 s BACKWARD every ~32.5 s, and under an unpinned BLAS
thread count that moves ETKF per-frame cost by 370%.  This harness re-measures
from scratch with:
  * time.perf_counter()  (CLOCK_MONOTONIC)   -- immune to the realtime step
  * 1 BLAS/OMP/MKL thread, pinned before numpy import
  * ONE process, ONE worker, all arms in the SAME session
  * production code completely unmodified -- every timer wraps the call from OUTSIDE

It computes NO accuracy metrics: metric computation is real work and would
contaminate the very quantity being measured.

Output: results.jsonl  (one record per (arm, episode, rep))
"""
import os
import sys

# ---- MUST precede numpy/torch import -----------------------------------------
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import json
import platform
import subprocess
import time
import warnings

import numpy as np
import torch

torch.set_num_threads(1)
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'results.jsonl')

from wavesplat.data_loader import EpisodeDataLoader                      # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater           # noqa: E402
from wavesplat.run import DEFAULT_CONFIG, set_seed                       # noqa: E402
from wavesplat.baselines.etkf_grid import ETKFGrid                       # noqa: E402
from wavesplat.baselines.letkf_grid import LETKFGrid                     # noqa: E402
from wavesplat.baselines.gpr_baseline import GPRBaseline                 # noqa: E402
from wavesplat.baselines.gpr_dispersion import GPRDispersion             # noqa: E402
from scipy.interpolate import RegularGridInterpolator                    # noqa: E402

# P1 12-episode set -- verbatim from tmp/rerun_2026-07-25/batchA_baselines.py
EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
]
ARMS = ['WS-P1', 'ETKF', 'LETKF', 'GPR-1', 'GPR-disp']
WS_SEEDS = {1: 42, 2: 123, 3: 456}
REPS = [1, 2, 3]

WS_CFG = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
          'adaptive_n': True, 'tracking_window': 3}


class Clock:
    """Paired monotonic/realtime stopwatch. Detects CLOCK_REALTIME backward steps."""

    def __init__(self):
        self.m0 = time.perf_counter()
        self.r0 = time.time()
        self._pm = self.m0
        self._pr = self.r0
        self.steps = []          # (t_monotonic_since_start, realtime_deficit_s)

    def tick(self):
        """Call once per frame boundary; records any realtime step swallowed."""
        m = time.perf_counter()
        r = time.time()
        dm = m - self._pm
        dr = r - self._pr
        if dr - dm < -0.05:
            self.steps.append((round(m - self.m0, 4), round(dr - dm, 4)))
        self._pm, self._pr = m, r

    def totals(self):
        return (time.perf_counter() - self.m0, time.time() - self.r0)


def _finish(rec, clk, prod_ms, t_upd, t_pred, t_interp, t_frame, n_pts, t_of_frame):
    mono, real = clk.totals()
    rec.update({
        'run_monotonic_s': mono,
        'run_realtime_s': real,
        'run_clock_bias_pct': 100.0 * (real - mono) / mono if mono > 0 else float('nan'),
        'n_steps_swallowed': len(clk.steps),
        'steps': clk.steps,
        't_update_ms': t_upd,
        't_predict_ms': t_pred,
        't_interp_ms': t_interp,
        't_frame_ms': t_frame,
        'prod_elapsed_ms': prod_ms,      # what production time.time() reported
        'n_pts': n_pts,
        'frame_t': t_of_frame,
    })
    return rec


# ------------------------------------------------------------------ WS-P1 ----
def run_ws(path, seed):
    clk = Clock()
    set_seed(seed)
    loader = EpisodeDataLoader(path, device='cpu')
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(WS_CFG)
    cfg['seed'] = seed
    cfg['depth'] = loader.depth
    init_frames = loader.get_init_window(tp_factor=2.0)
    track_frames = loader.get_tracking_frames(tp_factor=2.0)
    gx, gy = loader.gt_grid_x, loader.gt_grid_y

    updater = make_updater(cfg)
    a_m, a_r = time.perf_counter(), time.time()
    init_res = updater.initialize(
        init_frames,
        x_range=(float(gx[0]), float(gx[-1])),
        y_range=(float(gy[0]), float(gy[-1])),
        verbose=False)
    init_mono = time.perf_counter() - a_m
    init_real = time.time() - a_r
    clk.tick()

    t_upd, t_pred, t_frame, prod_ms, n_pts, t_of = [], [], [], [], [], []
    for f in track_frames:
        b0 = time.perf_counter()
        ur = updater.update(f)
        b1 = time.perf_counter()
        updater.predict_field(gx, gy, f.t)
        b2 = time.perf_counter()
        t_upd.append((b1 - b0) * 1e3)
        t_pred.append((b2 - b1) * 1e3)
        t_frame.append((b2 - b0) * 1e3)
        prod_ms.append(float(ur['elapsed_ms']))
        n_pts.append(int(ur['n_pts']))
        t_of.append(float(f.t))
        clk.tick()

    rec = {'arm': 'WS-P1', 'seed': seed,
           'n_frames_total': loader.n_frames,
           'n_init_frames': len(init_frames),
           't_init_end': float(init_frames[-1].t),
           'init_monotonic_s': init_mono,
           'init_realtime_s': init_real,
           'init_prod_reported_s': float(init_res['elapsed_s']),
           'init_iters': int(init_res['n_iters']),
           'final_n': int(init_res['final_n'])}
    return _finish(rec, clk, prod_ms, t_upd, t_pred, [0.0] * len(t_upd), t_frame,
                   n_pts, t_of)


# --------------------------------------------------------- grid filters ------
def run_gridfilter(path, arm):
    clk = Clock()
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40)
    egy = np.linspace(gy[0], gy[-1], 30)
    Xg, Yg = np.meshgrid(gx, gy)
    query = np.column_stack([Yg.ravel(), Xg.ravel()])

    a_m, a_r = time.perf_counter(), time.time()
    if arm == 'ETKF':
        b = ETKFGrid(egx, egy, depth=loader.depth, n_ensemble=100,
                     mult_inflation=1.0, add_inflation=0.005)
    else:
        b = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=100,
                      mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)
    init_mono = time.perf_counter() - a_m
    init_real = time.time() - a_r
    clk.tick()

    t_upd, t_pred, t_int, t_frame, prod_ms, n_pts, t_of = [], [], [], [], [], [], []
    for f in loader:
        x = f.x_obs.cpu().numpy()
        y = f.y_obs.cpu().numpy()
        z = f.z_obs.cpu().numpy()
        b0 = time.perf_counter()
        el = b.update(x, y, z)
        b1 = time.perf_counter()
        pred = b.predict_grid()
        b2 = time.perf_counter()
        RegularGridInterpolator((egy, egx), pred, method='linear',
                                bounds_error=False, fill_value=0.0)(query)
        b3 = time.perf_counter()
        t_upd.append((b1 - b0) * 1e3)
        t_pred.append((b2 - b1) * 1e3)
        t_int.append((b3 - b2) * 1e3)
        t_frame.append((b3 - b0) * 1e3)
        prod_ms.append(float(el))
        n_pts.append(int(len(x)))
        t_of.append(float(f.t))
        clk.tick()

    rec = {'arm': arm, 'seed': 42,
           'n_frames_total': loader.n_frames,
           'n_init_frames': len(loader.get_init_window(tp_factor=2.0)),
           't_init_end': float(loader.get_init_window(tp_factor=2.0)[-1].t),
           'init_monotonic_s': init_mono,
           'init_realtime_s': init_real,
           'init_prod_reported_s': None, 'init_iters': 0, 'final_n': None}
    return _finish(rec, clk, prod_ms, t_upd, t_pred, t_int, t_frame, n_pts, t_of)


# ------------------------------------------------------------------ GPR ------
def run_gpr(path, arm):
    clk = Clock()
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()

    a_m, a_r = time.perf_counter(), time.time()
    if arm == 'GPR-1':
        g = GPRBaseline(kernel_type='matern', length_scale=20.0,
                        noise_sigma=0.02, window_frames=1)
    else:
        g = GPRDispersion(length_scale=20.0, noise_sigma=0.02,
                          window_frames=3, depth=50.0)
    init_mono = time.perf_counter() - a_m
    init_real = time.time() - a_r
    clk.tick()

    t_upd, t_pred, t_frame, prod_ms, n_pts, t_of = [], [], [], [], [], []
    for f in loader:
        x = f.x_obs.cpu().numpy()
        y = f.y_obs.cpu().numpy()
        z = f.z_obs.cpu().numpy()
        b0 = time.perf_counter()
        el = g.update(x, y, z)
        b1 = time.perf_counter()
        g.predict_grid(gx, gy)
        b2 = time.perf_counter()
        t_upd.append((b1 - b0) * 1e3)
        t_pred.append((b2 - b1) * 1e3)
        t_frame.append((b2 - b0) * 1e3)
        prod_ms.append(float(el))
        n_pts.append(int(len(x)))
        t_of.append(float(f.t))
        clk.tick()

    rec = {'arm': arm, 'seed': 42,
           'n_frames_total': loader.n_frames,
           'n_init_frames': len(loader.get_init_window(tp_factor=2.0)),
           't_init_end': float(loader.get_init_window(tp_factor=2.0)[-1].t),
           'init_monotonic_s': init_mono,
           'init_realtime_s': init_real,
           'init_prod_reported_s': None, 'init_iters': 0, 'final_n': None}
    return _finish(rec, clk, prod_ms, t_upd, t_pred, [0.0] * len(t_upd), t_frame,
                   n_pts, t_of)


def provenance():
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception as e:
            return repr(e)
    import sklearn
    import scipy
    return {
        'kind': 'provenance',
        'cpu_model': sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-").strip(),
        'nproc': sh('nproc'),
        'platform': platform.platform(),
        'uname': sh('uname -a'),
        'python': sys.version.split()[0],
        'numpy': np.__version__, 'torch': torch.__version__,
        'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
        'torch_num_threads': torch.get_num_threads(),
        'env_threads': {v: os.environ.get(v) for v in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        'blas': str(np.__config__.get_info('blas_opt') if hasattr(np.__config__, 'get_info')
                    else 'scipy-openblas 0.3.33 (see np.show_config)'),
        'workers': 1, 'process_model': 'single process, serial, all arms same session',
        'timer': 'time.perf_counter (CLOCK_MONOTONIC)',
        'data_dir': DATA, 'episodes': EPISODES, 'arms': ARMS,
        'ws_cfg': WS_CFG, 'ws_seeds': WS_SEEDS,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('kind') != 'provenance' and 'error' not in r:
                done.add(r['job_id'])

    fh = open(OUT, 'a')
    if not done:
        fh.write(json.dumps(provenance()) + '\n')
        fh.flush()

    jobs = [(rep, ep, arm) for rep in REPS for ep in EPISODES for arm in ARMS]
    t_start = time.perf_counter()
    n_done = 0
    for rep, ep, arm in jobs:
        job_id = f'{arm}|{ep}|rep{rep}'
        if job_id in done:
            continue
        path = os.path.join(DATA, ep)
        w0 = time.perf_counter()
        try:
            if arm == 'WS-P1':
                rec = run_ws(path, WS_SEEDS[rep])
            elif arm in ('ETKF', 'LETKF'):
                rec = run_gridfilter(path, arm)
            else:
                rec = run_gpr(path, arm)
        except Exception as e:
            rec = {'arm': arm, 'error': repr(e)}
        rec.update({'job_id': job_id, 'episode': ep, 'rep': rep,
                    'wall_s': time.perf_counter() - w0})
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        n_done += 1
        med = (float(np.median(rec['t_update_ms'])) if 't_update_ms' in rec
               and rec['t_update_ms'] else float('nan'))
        el = time.perf_counter() - t_start
        print(f'[{n_done}] {job_id:<58s} upd_med {med:7.2f} ms '
              f'({rec["wall_s"]:5.1f}s) elapsed {el/60:.1f}min', flush=True)
    fh.close()
    print(f'DONE {(time.perf_counter()-t_start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
