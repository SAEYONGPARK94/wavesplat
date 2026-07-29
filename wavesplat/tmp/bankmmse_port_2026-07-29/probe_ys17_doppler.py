"""EXPLORATORY: does a GT-free-selected current Doppler term (Uc) improve the
YS17 w0 bank-MMSE field?  YS17 is the one record where the July analysis found a
genuine current signature (within-record U convergence ~6 deg); the pooled
blanket-U was REFUTED, so this is per-record selection on the fit's own held-out
val misfit (vr), never on GT.

Selection protocol (all observation-only):
  1. seed-42 fit for every Uc on a speed x direction grid, tiny Pq (no render),
     record vr;
  2. repeat argmin for the other two seeds -> is the selected U stable across
     seeds (the GT-free discriminator from the July record)?
  3. render the full protocol with the seed-consensus U, 3-seed median field,
     score vs the U=0 baseline.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

sys.path.insert(0, 'tmp/bankmmse_port_2026-07-29')
from measure_blind_vs_frozen import dense_gt_std, score

OUT = 'tmp/bankmmse_port_2026-07-29/ys17_doppler.json'
SEEDS = [42, 1042, 2042]
SPEEDS = [0.15, 0.3, 0.45, 0.6]
NDIR = 12


def main():
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate

    sc = cg.scene_real('YS17', 0)
    sg = dense_gt_std('YS17', 0)
    npx, nl = sc['npx'], len(sc['leads'])
    s0 = 0.2 * float(np.std(sc['z']))
    Pq_tiny = sc['Pq'][:2]

    def vr_of(Uc, seed):
        diag = {}
        mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], Pq_tiny, sigma0=s0,
                           kp_hint=None, seed=seed, Uc=Uc, diag_out=diag)
        return float(diag['vr'])

    grid = [(0.0, 0.0)]
    for sp in SPEEDS:
        for th in np.arange(NDIR) * (2 * np.pi / NDIR):
            grid.append((sp * np.cos(th), sp * np.sin(th)))

    t0 = time.time()
    land = {}
    for seed in SEEDS:
        vrs = [vr_of(Uc, seed) for Uc in grid]
        j = int(np.argmin(vrs))
        land[seed] = dict(vr0=vrs[0], best_vr=vrs[j], best_U=list(grid[j]),
                          best_speed=float(np.hypot(*grid[j])),
                          best_dir_deg=float(np.degrees(np.arctan2(grid[j][1],
                                                                   grid[j][0]))))
        print(f'seed {seed}: U=0 vr {vrs[0]:.4f} -> best vr {vrs[j]:.4f} at '
              f'U=({grid[j][0]:+.2f},{grid[j][1]:+.2f}) '
              f'|U|={land[seed]["best_speed"]:.2f} '
              f'dir {land[seed]["best_dir_deg"]:.0f}deg', flush=True)
    print(f'landscape {time.time()-t0:.0f}s', flush=True)

    # seed-consensus U = the seed-42 argmin (report stability separately)
    Uc_sel = tuple(land[42]['best_U'])
    fields = {}
    for tag, Uc in (('U0', (0.0, 0.0)), ('Usel', Uc_sel)):
        fs = []
        for s in SEEDS:
            fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                    sigma0=s0, kp_hint=None, seed=s, Uc=Uc)
            fs.append(fq[:npx * nl].reshape(nl, *sc['Xq'].shape))
        fields[tag] = score(np.median(np.stack(fs, 0), 0), sc, sg)
        print(f'{tag}: now {fields[tag]["nowcast_nrmse"]:.4f} '
              f'mean {fields[tag]["mean_nrmse"]:.4f}', flush=True)

    json.dump(dict(landscape={str(k): v for k, v in land.items()},
                   U_selected=list(Uc_sel), scores=fields),
              open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
