"""PAIRED evaluation: init ONCE per (episode, seed), snapshot the post-init model, then run the
tracking stage TWICE from that snapshot -- flag OFF (shipped, TRACK_PHASE_PREDICT=True) and flag
ON (=False). This guarantees the two arms share an IDENTICAL initialization (same adaptive-N,
same params), so the only difference is the tracking behaviour. Same device for both arms, so no
device-mixing confound (the CPU baseline cannot be reused because the adaptive-N path is
FP-sensitive and diverges between CPU and GPU).

avg_nRMSE is computed exactly as wavesplat/run.py does: per tracking frame, predict on the GT
grid and take nrmse(pred, gt, Hs); average over frames. (compute_all_metrics['nRMSE'] IS
nrmse(pred,gt,Hs), verified.)

Serial (no multiprocessing Pool -- the CUDA+spawn Pool crashed with BrokenPipe). Shard across
processes with WS_SHARD="i/n" (episodes[i::n]) for concurrency. Resumable by job_id.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

torch.set_num_threads(int(os.environ.get('WS_THREADS', '2')))
DEVICE = os.environ.get('WS_DEVICE', 'cuda')

import wavesplat.optimizer.sequential_update as S
from wavesplat.run import set_seed, DEFAULT_CONFIG
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import nrmse

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'ampfix_paired_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]
P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3}
PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
EPISODES = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS] + [
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_slope.h5']


def snap(m):
    return dict(n=m.n_gaussians, w=m.w.detach().clone(), kx=m.kx.detach().clone(),
                ky=m.ky.detach().clone(), phi=m.phi.detach().clone())


def restore(m, s):
    m.n_gaussians = s['n']
    m.w = torch.nn.Parameter(s['w'].clone()); m.kx = torch.nn.Parameter(s['kx'].clone())
    m.ky = torch.nn.Parameter(s['ky'].clone()); m.phi = torch.nn.Parameter(s['phi'].clone())


def track_avg_nrmse(up, frames, gx, gy, Hs, last_t):
    """Replicates run.py Phase-2 loop, returns mean per-frame full-field nRMSE."""
    up._frame_buffer = []; up._last_t = last_t
    up._build_tracking_optimizer()
    nrs = []
    for frame in frames:
        up.update(frame)  # respects module TRACK_PHASE_PREDICT
        with torch.no_grad():
            pred = up.predict_field(gx, gy, frame.t).cpu().numpy()
        nrs.append(nrmse(pred, frame.gt_field.cpu().numpy(), Hs))
    return float(np.mean(nrs)) if nrs else float('nan')


def run_pair(ep, seed):
    set_seed(seed)
    loader = EpisodeDataLoader(os.path.join(DATA, ep), device=DEVICE)
    Hs = loader.Hs_true; gx, gy = loader.gt_grid_x, loader.gt_grid_y
    cfg = DEFAULT_CONFIG.copy(); cfg.update(P1); cfg['seed'] = seed; cfg['device'] = DEVICE
    inf = loader.get_init_window(2.0); trf = loader.get_tracking_frames(2.0)
    up = SequentialUpdater(cfg)
    up.initialize(inf, x_range=(float(gx[0]), float(gx[-1])),
                  y_range=(float(gy[0]), float(gy[-1])))
    last_t = inf[-1].t
    with torch.no_grad():
        p = up.predict_field(gx, gy, last_t).cpu().numpy()
    init_nr = nrmse(p, inf[-1].gt_field.cpu().numpy(), Hs)
    N = up.model.n_gaussians
    post = snap(up.model)

    # arm OFF = shipped
    S.TRACK_PHASE_PREDICT = True
    restore(up.model, post)
    off = track_avg_nrmse(up, trf, gx, gy, Hs, last_t)
    # arm ON = fix
    S.TRACK_PHASE_PREDICT = False
    restore(up.model, post)
    on = track_avg_nrmse(up, trf, gx, gy, Hs, last_t)
    S.TRACK_PHASE_PREDICT = True  # leave module at default
    return dict(episode=ep, seed=seed, N_init=int(N), init_nrmse=float(init_nr),
                off_avg_nRMSE=float(off), on_avg_nRMSE=float(on))


def main():
    shard = os.environ.get('WS_SHARD', '0/1')
    i, n = (int(x) for x in shard.split('/'))
    eps = EPISODES[i::n]
    jobs = [(ep, s) for ep in eps for s in SEEDS]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r['episode'], r['seed']))
    todo = [j for j in jobs if j not in done]
    print(f'shard {shard} | eps {len(eps)} | jobs {len(jobs)} | done {len(done&set(jobs))} '
          f'| todo {len(todo)}', flush=True)
    t0 = time.time()
    with open(OUT, 'a') as fh:
        for k, (ep, s) in enumerate(todo, 1):
            try:
                rec = run_pair(ep, s)
            except Exception as e:
                rec = {'episode': ep, 'seed': s, 'error': repr(e)}
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            print(f'[{shard} {k}/{len(todo)}] {ep:<38} s={s} N={rec.get("N_init")} '
                  f'init={rec.get("init_nrmse", float("nan")):.4f} '
                  f'OFF={rec.get("off_avg_nRMSE", float("nan")):.4f} '
                  f'ON={rec.get("on_avg_nRMSE", float("nan")):.4f} '
                  f'({el/k:.0f}s/job)', flush=True)
    print(f'SHARD {shard} DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
