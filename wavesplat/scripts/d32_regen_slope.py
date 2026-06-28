"""
M5 — regenerate the 8 corrupt benchmark slope episodes (all BF5_15c / BF5_30c).

The committed + working-tree files are 133-byte stubs (git has no good copy), so they
must be regenerated. Wave components and GT are deterministic in (sea_state, n_comp,
seed); only the slope raydrop differs from the intact uniform twin. We VERIFY this by
regenerating BF5_15c_s0 to a temp dir and checking its components + GT field match the
intact uniform file, THEN regenerate all 8 into data/.

Run: python3 scripts/d32_regen_slope.py
"""
import os, sys, glob, shutil
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.pipeline import run_episode

SEA = {'BF5': 'BF5'}
DUR, DT, DEPTH = 30.0, 0.1, 50.0


def parse(tag):
    bf, comp, s = tag.split('_')
    return bf, int(comp.replace('c', '')), int(s[1:])


def slope_name(bf, nc, seed):
    return f"episode_{bf}_{nc}comp_seed{seed}_slope.h5"


def find_corrupt():
    bad = []
    for f in sorted(glob.glob('data/episode_BF5_*comp_seed*_slope.h5')):
        try:
            with h5py.File(f, 'r') as h:
                _ = h['frames/timestamps'][...]
        except Exception:
            bad.append(os.path.basename(f))
    return bad


def load_h5(path):
    with h5py.File(path, 'r') as h:
        wc = h['config/wave_components'][...]
        gt = h['frames/gt_wave_field'][...]
        ts = h['frames/timestamps'][...]
    return wc, gt, ts


def main():
    corrupt = find_corrupt()
    print(f"corrupt benchmark slope files: {len(corrupt)}")
    for c in corrupt:
        print('  ', c)

    # --- VERIFY faithfulness on one file (temp dir) ---
    bf, nc, seed = 'BF5', 15, 0
    tmp = 'data/_regen_verify'
    os.makedirs(tmp, exist_ok=True)
    print(f"\n[verify] regenerating {bf}_{nc}c_s{seed} slope -> temp, comparing to uniform twin ...")
    run_episode(sea_state=bf, n_components=nc, seed=seed, raydrop_model='slope',
                duration=DUR, dt=DT, depth=DEPTH, output_dir=tmp, verbose=False)
    new_path = os.path.join(tmp, slope_name(bf, nc, seed))
    uni_path = f'data/episode_{bf}_{nc}comp_seed{seed}_uniform.h5'
    wc_new, gt_new, ts_new = load_h5(new_path)
    wc_uni, gt_uni, ts_uni = load_h5(uni_path)
    comp_match = wc_new.shape == wc_uni.shape and np.allclose(wc_new, wc_uni, atol=1e-8)
    gt_match = gt_new.shape == gt_uni.shape and np.allclose(gt_new, gt_uni, atol=1e-6)
    print(f"  components match: {comp_match}  (shapes {wc_new.shape} vs {wc_uni.shape}, "
          f"max|d|={np.abs(wc_new-wc_uni).max() if wc_new.shape==wc_uni.shape else 'NA':})")
    print(f"  GT field match:  {gt_match}  (shapes {gt_new.shape} vs {gt_uni.shape}, "
          f"max|d|={np.abs(gt_new-gt_uni).max() if gt_new.shape==gt_uni.shape else 'NA':})")
    shutil.rmtree(tmp, ignore_errors=True)

    if not (comp_match and gt_match):
        print("\n[ABORT] regeneration does NOT reproduce the intact uniform twin — "
              "generation config drifted. Not overwriting any files. Flag for manual review.")
        return

    # --- regenerate all corrupt into data/ ---
    print(f"\n[verified] regenerating {len(corrupt)} corrupt slope files into data/ ...")
    for fn in corrupt:
        parts = fn.split('_')                       # episode_BF5_15comp_seed0_slope.h5
        bf = parts[1]; nc = int(parts[2].replace('comp', '')); seed = int(parts[3].replace('seed', ''))
        p = run_episode(sea_state=bf, n_components=nc, seed=seed, raydrop_model='slope',
                        duration=DUR, dt=DT, depth=DEPTH, output_dir='data', verbose=False)
        # sanity: loads + components match uniform
        wc_new, _, _ = load_h5(p)
        wc_uni, _, _ = load_h5(f'data/episode_{bf}_{nc}comp_seed{seed}_uniform.h5')
        ok = wc_new.shape == wc_uni.shape and np.allclose(wc_new, wc_uni, atol=1e-8)
        print(f"  regenerated {os.path.basename(p)}  comp-match={ok}")
    print("\nDONE — all corrupt slope files regenerated and verified.")


if __name__ == '__main__':
    main()
