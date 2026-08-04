"""
generate_dataset.py
Batch generation of WaveSim episodes from config.

Usage:
  # Local (sequential)
  python generate_dataset.py --config configs/default.yaml

  # Server (parallel workers)
  python generate_dataset.py --config configs/server.yaml

  # Dry run (show plan only)
  python generate_dataset.py --config configs/server.yaml --dry-run
"""

import argparse
import os
import time
from multiprocessing import Pool, cpu_count

from wavesim.config import load_config, config_summary
from wavesim.pipeline import run_episode


def _run_one(args):
    """Worker function for multiprocessing."""
    scenario, seed, raydrop, cfg, output_dir, verbose = args

    sea_state = scenario.sea_state
    n_components = scenario.n_components

    try:
        t0 = time.time()
        filepath = run_episode(
            sea_state=sea_state,
            n_components=n_components,
            n_pts=cfg.episode.n_pts,
            seed=seed,
            raydrop_model=raydrop,
            duration=cfg.episode.duration,
            dt=cfg.episode.dt,
            depth=cfg.episode.depth,
            output_dir=output_dir,
            verbose=verbose,
            cfg=cfg,
            direction_spread_deg=getattr(scenario, 'direction_spread_deg', 30.0),
        )
        elapsed = time.time() - t0
        return (filepath, elapsed, None)
    except Exception as e:
        name = f"{scenario.name}_{sea_state}_{n_components}comp_seed{seed}_{raydrop}"
        return (name, 0, str(e))


def main():
    parser = argparse.ArgumentParser(description='Generate WaveSim dataset')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show generation plan without running')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-frame details')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip episodes whose .h5 files already exist')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(config_summary(cfg))

    # Build job list
    jobs = []
    output_dir = cfg.dataset.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for scenario in cfg.dataset.scenarios:
        for seed in cfg.dataset.seeds:
            for raydrop in cfg.dataset.raydrop_models:
                fname = (f"episode_{scenario.sea_state}_"
                         f"{scenario.n_components}comp_seed{seed}_{raydrop}.h5")
                fpath = os.path.join(output_dir, fname)

                if args.skip_existing and os.path.exists(fpath):
                    print(f"  [SKIP] {fname} (exists)")
                    continue

                jobs.append((scenario, seed, raydrop, cfg, output_dir,
                             args.verbose))

    print(f"\nJobs to run: {len(jobs)}")

    if args.dry_run:
        for job in jobs:
            sc, seed, rd, _, _, _ = job
            print(f"  {sc.name}: {sc.sea_state} {sc.n_components}comp "
                  f"seed={seed} {rd}")
        return

    # Run
    num_workers = min(cfg.dataset.num_workers, cpu_count())
    t_start = time.time()

    if num_workers <= 1:
        # Sequential
        results = []
        for i, job in enumerate(jobs):
            sc, seed, rd, _, _, _ = job
            print(f"\n[{i+1}/{len(jobs)}] {sc.name}: {sc.sea_state} "
                  f"{sc.n_components}comp seed={seed} {rd}")
            results.append(_run_one(job))
    else:
        # Parallel
        print(f"\nRunning with {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = pool.map(_run_one, jobs)

    # Summary
    t_total = time.time() - t_start
    succeeded = [(fp, t) for fp, t, e in results if e is None]
    failed = [(fp, e) for fp, _, e in results if e is not None]

    print(f"\n{'='*60}")
    print(f"Dataset generation complete")
    print(f"  Succeeded: {len(succeeded)}/{len(results)}")
    print(f"  Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    if succeeded:
        times = [t for _, t in succeeded]
        print(f"  Per episode: {sum(times)/len(times):.0f}s avg "
              f"({min(times):.0f}s min, {max(times):.0f}s max)")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for fp, e in failed:
            print(f"    {fp}: {e}")


if __name__ == '__main__':
    main()
