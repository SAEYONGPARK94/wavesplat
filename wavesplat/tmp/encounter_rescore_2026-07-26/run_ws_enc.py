"""Analytic arm — per-window ("still-cut") frozen-init encounter forecast. PREREG.md section 4.

init_method is PINNED to 'v2_konly' (PREREG section 7.3: the shipped default was flipped to
'auto' on 2026-07-26 11:53, so the parent W9 "analytic" arm was in fact the auto router).
No tracking. Predicts at the 6 encounter distances x 4 leads, plus a full-grid global anchor.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, 'tmp/encounter_rescore_2026-07-26')

SEEDS = (42, 1, 2, 3, 4)


def _worker(args):
    cell, ep, i_ref, seed = args
    import numpy as np
    import torch
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.run import DEFAULT_CONFIG
    from common import EpisodeGeom, NF_WS, LEADS, path_of
    t0 = time.time()
    try:
        G = EpisodeGeom(cell, ep)
        tg = G.targets(i_ref)
        L = EpisodeDataLoader(path_of(cell, ep), device='cpu')
        lo = i_ref - (NF_WS - 1)
        frames = [L[i] for i in range(lo, i_ref + 1)]
        span = float(frames[-1].t) - float(frames[0].t)
        assert len(frames) == NF_WS
        cfg = dict(DEFAULT_CONFIG)
        cfg.update({'adaptive_n': True, 'seed': seed, 'device': 'cpu',
                    'init_method': 'v2_konly'})
        up = su.SequentialUpdater(cfg)
        up.initialize(frames,
                      x_range=(float(L.gt_grid_x[0]), float(L.gt_grid_x[-1])),
                      y_range=(float(L.gt_grid_y[0]), float(L.gt_grid_y[-1])),
                      verbose=False)
        xs = torch.tensor(tg['xs'], dtype=torch.float32)
        ys = torch.tensor([tg['y_c']], dtype=torch.float32)
        pred = []
        for Ld in LEADS:
            z = up.predict_field(xs, ys, tg['t_ref'] + float(Ld))
            pred.append(z.reshape(-1).cpu().numpy().tolist())
        # within-run global-map anchor at lead 0 (same fitted model, full GT grid)
        gx = torch.tensor(G.gx, dtype=torch.float32)
        gy = torch.tensor(G.gy, dtype=torch.float32)
        zg = up.predict_field(gx, gy, tg['t_ref']).cpu().numpy()
        gtg = G.eta_grid(G.gx, G.gy, tg['t_ref'])
        return dict(arm='ws', cell=cell, episode=ep, i_ref=i_ref, seed=seed,
                    t_ref=tg['t_ref'], x_edge=tg['x_edge'], y_c=tg['y_c'],
                    pred=pred, gt=tg['gt'].tolist(),
                    map_rmse=float(np.sqrt(np.mean((zg - gtg) ** 2))),
                    map_rms0=float(np.sqrt(np.mean(gtg ** 2))),
                    n_gaussians=int(up.model.n_gaussians), win_span_s=span,
                    n_win_obs=tg['n_win_obs'],
                    frac_beyond={str(k): v for k, v in tg['frac_beyond'].items()},
                    sec=time.time() - t0, err=None)
    except Exception as e:
        return dict(arm='ws', cell=cell, episode=ep, i_ref=i_ref, seed=seed,
                    pred=None, gt=None, sec=time.time() - t0, err=repr(e))


def main():
    from common import all_jobs, CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/encounter_rescore_2026-07-26/ws_enc.jsonl')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--cells', nargs='+', default=CELLS)
    a = ap.parse_args()
    jobs = [(c, ep, i, s) for (c, ep, i) in all_jobs(a.cells) for s in SEEDS]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['cell'], d['episode'], d['i_ref'], d['seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"WS encounter arm: {len(jobs)} jobs ({len(done)} done), {a.workers} workers",
          flush=True)
    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n'); n += 1
            tag = ('ERR ' + str(res['err'])[:70]) if res['err'] else \
                  f"ng={res['n_gaussians']:2d} map_r={res['map_rmse']:.3f}"
            print(f"[{n:4d}/{len(jobs)}] {res['cell']:5s} {res['episode'][14:]:26s} "
                  f"i{res['i_ref']} s{res['seed']:<2d} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"WS encounter arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
