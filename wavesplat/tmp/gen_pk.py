import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
from concurrent.futures import ProcessPoolExecutor
from wavesim.config import load_config
from wavesim.pipeline import run_episode

OUT = 'tmp/finemesh_pk'

def one(a):
    sea, seed = a
    cfg = load_config('tmp/finemesh_bf2.yaml')      # adequate mesh 180x150
    return run_episode(sea_state=sea, n_components=1, seed=seed,
                       raydrop_model='fixedT', n_pts=8000, output_dir=OUT,
                       verbose=False, cfg=cfg, duration=cfg.episode.duration,
                       dt=cfg.episode.dt, depth=cfg.episode.depth,
                       use_gpu_raycast=True, peak_locked=True)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, sd) for s in ('BF2', 'BF3', 'BF4') for sd in range(5)]
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, _ in enumerate(ex.map(one, jobs)):
            print(f'[{i+1}/{len(jobs)}]', flush=True)
