"""E-3: regenerate the F1 clean-linear stage (35 episodes) with an adequate
raycast mesh.

Prereg: docs/E2_MESH_AUDIT_prereg_2026-07-13.md (frozen before numbers).

Intervention is the mesh alone (40x30 -> 180x150).  Seeds are unchanged, so
wave_components and gt_wave_field come out bit-identical and only the LiDAR
observation operator differs -- a fully paired coarse-vs-fine design.
"""
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.config import load_config          # noqa: E402
from wavesim.pipeline import run_episode        # noqa: E402

CFG = 'tmp/finemesh_bf2.yaml'      # identical to default.yaml except mesh 180x150
OUT = 'tmp/finemesh_clean'
LIST = 'tmp/clean_stage_episodes.txt'
WORKERS = 12

PAT = re.compile(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(\w+?)(_dir180)?\.h5$')


def parse(name):
    m = PAT.match(name)
    if not m:
        raise ValueError(f'unparsed episode name: {name}')
    sea, ncomp, seed, raydrop, d180 = m.groups()
    return dict(sea_state=sea, n_components=int(ncomp), seed=int(seed),
                raydrop_model=raydrop, direction_deg=180.0 if d180 else 0.0)


def one(spec):
    cfg = load_config(CFG)
    return run_episode(output_dir=OUT, verbose=False, cfg=cfg,
                       duration=cfg.episode.duration, dt=cfg.episode.dt,
                       depth=cfg.episode.depth, n_pts=cfg.episode.n_pts,
                       **spec)


def main():
    os.makedirs(OUT, exist_ok=True)
    names = [l.strip() for l in open(LIST) if l.strip()]
    todo = [n for n in names if not os.path.exists(os.path.join(OUT, n))]
    print(f'[E-3] {len(names)} episodes in the clean stage, '
          f'{len(names) - len(todo)} already regenerated, {len(todo)} to go',
          flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, parse(n)): n for n in todo}
        for f in as_completed(futs):
            n = futs[f]
            try:
                f.result()
                done += 1
                print(f'[{done}/{len(todo)}] {n}', flush=True)
            except Exception as e:                      # noqa: BLE001
                print(f'[FAIL] {n}: {e}', flush=True)

    have = sum(os.path.exists(os.path.join(OUT, n)) for n in names)
    print(f'[E-3] regenerated {have}/{len(names)}', flush=True)


if __name__ == '__main__':
    main()
