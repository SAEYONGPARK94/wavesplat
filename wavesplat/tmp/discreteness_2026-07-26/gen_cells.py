"""Generate the three discreteness cells (W9). Frozen spec: tmp/discreteness_2026-07-26/PREREG.md §2.

STANDALONE. Nothing in wavesim/ or wavesplat/ is modified: this process replaces
`wavesim.pipeline.jonswap_components` with a wrapper that calls the real library function and
then (for the spread cells) expands each parent component into M energy-preserving
sub-components. Everything downstream of the component list -- mesh build, GPU raycast, raydrop,
noise, confidence, mesh-adequacy check, HDF5 writer -- is the untouched `run_episode` path used
by tmp/regen_run_finemesh.py for data_finemesh_2026-07-23.

Usage:
  python3 tmp/discreteness_2026-07-26/gen_cells.py --cell a          # one cell
  python3 tmp/discreteness_2026-07-26/gen_cells.py --cell a --verify_only
"""
import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import h5py
import numpy as np

import wavesim.pipeline as pipeline
from wavesim.config import WaveSimConfig
from wavesim.wave_generator import (WaveComponent, check_q_constraint,
                                    jonswap_components as _real_jonswap,
                                    wavenumber_from_omega)

OUT_ROOT = 'tmp/discreteness_2026-07-26/data'
SEEDS = [10, 11, 12, 13, 14]
RAYDROPS = ['uniform', 'slope']
SEA_STATE, NPARENT, DEPTH = 'BF4', 15, 50.0
MESH_NX, MESH_NY = 200, 167
TOTAL_SPREAD_DEG = 30.0          # library default; held fixed across cells

# cell -> (index, M, sigma_sub_deg, omega_jitter)
CELLS = {
    'a': dict(idx=0, M=1,  sigma_sub=0.0,  omega_jitter=False),
    'b': dict(idx=1, M=31, sigma_sub=10.0, omega_jitter=False),
    'c': dict(idx=2, M=31, sigma_sub=28.0, omega_jitter=True),
}


def parent_spread(sigma_sub):
    """sigma_parent such that parent+sub directional variance == TOTAL_SPREAD_DEG^2."""
    v = TOTAL_SPREAD_DEG ** 2 - sigma_sub ** 2
    if v < 0:
        raise ValueError('sigma_sub exceeds the total spread budget')
    return float(np.sqrt(v))


def _apply_library_Q(components, Q_max=0.8):
    """The library's own Q rule, applied to the expanded set (wave_generator.jonswap_components).

    Q never enters the rendered field (gt_generator.evaluate_field -> single_gerstner returns Z and
    normals, neither of which uses Q); this only keeps the pipeline's Q assertion satisfied and the
    stored table self-consistent.
    """
    for c in components:
        c.Q = min(Q_max / (c.omega * c.A + 1e-12), 1.0) * 0.5
    ok, q_val = check_q_constraint(components)
    if not ok:
        s = 0.95 / q_val
        for c in components:
            c.Q *= s
    return components


def expand(parents, M, sigma_sub_deg, omega_jitter, d_omega, depth, rng):
    """Replace every parent by M energy-preserving sub-components. PREREG §2 step 3."""
    if M == 1 and sigma_sub_deg == 0.0 and not omega_jitter:
        return list(parents)                      # cell A: bit-identical passthrough
    sig = np.deg2rad(sigma_sub_deg)
    out = []
    for c in parents:
        th0 = float(np.arctan2(c.ky, c.kx))
        A_sub = c.A / np.sqrt(M)
        th = rng.normal(th0, sig, M)
        if omega_jitter:
            om = rng.uniform(c.omega - 0.5 * d_omega, c.omega + 0.5 * d_omega, M)
            kk = np.array([wavenumber_from_omega(float(w), depth) for w in om])
        else:
            om = np.full(M, c.omega)
            kk = np.full(M, c.k)
        phi = rng.uniform(0.0, 2 * np.pi, M)
        for j in range(M):
            out.append(WaveComponent(A=float(A_sub), omega=float(om[j]),
                                     kx=float(kk[j] * np.cos(th[j])),
                                     ky=float(kk[j] * np.sin(th[j])),
                                     phi=float(phi[j]), Q=0.0))
    return _apply_library_Q(out)


def make_patch(cell, seed):
    """Closure installed over wavesim.pipeline.jonswap_components for ONE episode."""
    spec = CELLS[cell]
    sig_par = parent_spread(spec['sigma_sub'])
    rng = np.random.default_rng(900000 + 1000 * spec['idx'] + seed)

    def patched(Hs, Tp, n_components=30, depth=50.0, gamma=3.3, direction_deg=0.0,
                direction_spread_deg=TOTAL_SPREAD_DEG, seed=None, Q_max=0.8,
                peak_locked=False):
        parents = _real_jonswap(Hs=Hs, Tp=Tp, n_components=n_components, depth=depth,
                                gamma=gamma, direction_deg=direction_deg,
                                direction_spread_deg=sig_par, seed=seed,
                                Q_max=Q_max, peak_locked=peak_locked)
        omega_p = 2 * np.pi / Tp
        d_omega = (3.0 - 0.5) * omega_p / n_components
        return expand(parents, spec['M'], spec['sigma_sub'], spec['omega_jitter'],
                      d_omega, depth, rng)
    return patched


# ── verification gates (PREREG §2) ───────────────────────────────────────────
def gate_identity(verbose=True):
    """G-identity: cell A's expansion must be a bit-identical passthrough of the library."""
    ok = True
    for s in SEEDS:
        pat = make_patch('a', s)
        got = pat(Hs=1.2, Tp=7.0, n_components=NPARENT, depth=DEPTH, seed=s)
        ref = _real_jonswap(Hs=1.2, Tp=7.0, n_components=NPARENT, depth=DEPTH, seed=s)
        a = np.array([[c.A, c.omega, c.kx, c.ky, c.phi, c.Q] for c in got])
        b = np.array([[c.A, c.omega, c.kx, c.ky, c.phi, c.Q] for c in ref])
        same = a.shape == b.shape and np.array_equal(a, b)
        ok &= same
        if verbose:
            print(f"  G-identity seed{s}: {'BIT-IDENTICAL' if same else 'MISMATCH'}")
    return ok


def cell_stats(cell, seed):
    """Variance / spread bookkeeping of one generated component set (no rendering)."""
    pat = make_patch(cell, seed)
    comps = pat(Hs=1.2, Tp=7.0, n_components=NPARENT, depth=DEPTH, seed=seed)
    A = np.array([c.A for c in comps]); th = np.array([np.arctan2(c.ky, c.kx) for c in comps])
    k = np.array([c.k for c in comps]); om = np.array([c.omega for c in comps])
    w = A ** 2 / np.sum(A ** 2)
    # energy-weighted circular spread
    C = np.sum(w * np.cos(th)); S = np.sum(w * np.sin(th))
    R = np.hypot(C, S)
    spread = float(np.rad2deg(np.sqrt(max(-2 * np.log(max(R, 1e-12)), 0.0))))
    return dict(cell=cell, seed=seed, n_atoms=len(comps),
                Hs_from_comps=float(4 * np.sqrt(np.sum(A ** 2) / 2)),
                dir_spread_deg=spread,
                lam_min=float(2 * np.pi / k.max()), lam_max=float(2 * np.pi / k.min()),
                omega_min=float(om.min()), omega_max=float(om.max()),
                q_sum=float(sum(c.Q * c.omega * c.A for c in comps)))


def reindex(cell):
    """Rebuild the manifest from ALREADY-WRITTEN .h5 files, running the same gate checks.

    Used because the first generation pass hit a manifest-bookkeeping TypeError (duplicate
    `n_atoms` kwarg) *after* `run_episode` had returned and the HDF5 had been written and read
    back cleanly. The episode data is unaffected; only the manifest record was lost. This
    recomputes it from the file on disk instead of regenerating 30 GPU-hours of identical data.
    """
    out_dir = os.path.join(OUT_ROOT, f'cell_{cell}')
    man_path = os.path.join(out_dir, '_manifest.json')
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    n = 0
    for seed in SEEDS:
        for raydrop in RAYDROPS:
            fname = f"episode_{SEA_STATE}_{NPARENT}comp_seed{seed}_{raydrop}.h5"
            fp = os.path.join(out_dir, fname)
            if not os.path.exists(fp):
                continue
            with h5py.File(fp, 'r') as f:
                at = dict(f['config'].attrs)
                gt = f['frames/gt_wave_field'][:]
                nfr = f['frames/lidar_points'].shape[0]
                natom = f['config/wave_components'].shape[0]
                bad = bool(np.isnan(gt).any())
            man[fname] = dict(status='ok' if not bad else 'SUSPECT',
                              elapsed_s=float(man.get(fname, {}).get('elapsed_s', float('nan'))),
                              n_frames=int(nfr), n_atoms_file=int(natom),
                              mesh_verdict=str(at.get('mesh_verdict')),
                              var_under=float(at.get('var_under_resolved', np.nan)),
                              lambda_min=float(at.get('lambda_min', np.nan)),
                              pts_per_lambda=float(at.get('pts_per_lambda', np.nan)),
                              Hs_realized=float(4 * np.std(gt)), nan_gt=bad,
                              reindexed=True, **cell_stats(cell, seed))
            n += 1
            print(f"  reindex {fname}: atoms={natom} frames={nfr} "
                  f"verdict={man[fname]['mesh_verdict']} "
                  f"var_under={man[fname]['var_under']:.4f} "
                  f"Hs={man[fname]['Hs_realized']:.3f}", flush=True)
    json.dump(man, open(man_path, 'w'), indent=1)
    print(f"cell {cell}: reindexed {n} episodes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=list(CELLS))
    ap.add_argument('--verify_only', action='store_true')
    ap.add_argument('--reindex', action='store_true',
                    help='rebuild the manifest from existing .h5 files, generate nothing')
    a = ap.parse_args()

    if a.reindex:
        reindex(a.cell)
        return

    print(f"=== cell {a.cell}: {CELLS[a.cell]}  parent_sigma="
          f"{parent_spread(CELLS[a.cell]['sigma_sub']):.4f} deg ===", flush=True)
    if a.cell == 'a':
        print("G-identity check (cell A passthrough vs library):")
        assert gate_identity(), "G-identity FAILED -- generation is not a passthrough"
    for s in SEEDS[:2]:
        print("  stats:", json.dumps(cell_stats(a.cell, s)), flush=True)
    if a.verify_only:
        return

    out_dir = os.path.join(OUT_ROOT, f'cell_{a.cell}')
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, '_manifest.json')
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}

    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY

    jobs = [(s, r) for s in SEEDS for r in RAYDROPS]
    t_start = time.time()
    for i, (seed, raydrop) in enumerate(jobs):
        fname = f"episode_{SEA_STATE}_{NPARENT}comp_seed{seed}_{raydrop}.h5"
        fp = os.path.join(out_dir, fname)
        if man.get(fname, {}).get('status') == 'ok' and os.path.exists(fp):
            print(f"[{i+1}/{len(jobs)}] SKIP {fname}", flush=True)
            continue
        print(f"[{i+1}/{len(jobs)}] {fname} ...", flush=True)
        t0 = time.time()
        orig = pipeline.jonswap_components
        try:
            pipeline.jonswap_components = make_patch(a.cell, seed)
            path = pipeline.run_episode(
                sea_state=SEA_STATE, n_components=NPARENT, n_pts=50, seed=seed,
                raydrop_model=raydrop, duration=30.0, dt=0.1, depth=DEPTH,
                output_dir=out_dir, verbose=False, cfg=cfg, use_gpu_raycast=True)
            with h5py.File(path, 'r') as f:
                at = dict(f['config'].attrs)
                gt = f['frames/gt_wave_field'][:]
                nfr = f['frames/lidar_points'].shape[0]
                natom = f['config/wave_components'].shape[0]
                bad = bool(np.isnan(gt).any())
            rec = dict(status='ok' if not bad else 'SUSPECT',
                       elapsed_s=time.time() - t0, n_frames=int(nfr),
                       n_atoms_file=int(natom),
                       mesh_verdict=str(at.get('mesh_verdict')),
                       var_under=float(at.get('var_under_resolved', np.nan)),
                       lambda_min=float(at.get('lambda_min', np.nan)),
                       pts_per_lambda=float(at.get('pts_per_lambda', np.nan)),
                       Hs_realized=float(4 * np.std(gt)), nan_gt=bad,
                       **cell_stats(a.cell, seed))
            print(f"    OK atoms={natom} frames={nfr} verdict={rec['mesh_verdict']} "
                  f"var_under={rec['var_under']:.4f} Hs_real={rec['Hs_realized']:.3f} "
                  f"[{rec['elapsed_s']:.0f}s]", flush=True)
        except Exception as e:
            rec = dict(status='FAILED', error=repr(e), tb=traceback.format_exc())
            print(f"    FAILED: {e}", flush=True)
        finally:
            pipeline.jonswap_components = orig
        man[fname] = rec
        json.dump(man, open(man_path, 'w'), indent=1)
    nok = sum(1 for v in man.values() if v.get('status') == 'ok')
    print(f"cell {a.cell} done: {nok}/{len(jobs)} ok, {(time.time()-t_start)/60:.1f} min",
          flush=True)


if __name__ == '__main__':
    main()
