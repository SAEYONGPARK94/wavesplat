"""MIXED-CURRICULUM scene generator for the beamform-conditioned set-diffusion retrain.

ADDITIVE / STANDALONE.  Nothing in wavesim/, wavesplat/ or scripts/ is modified: this module
imports `scripts/d3_beamform_cond_diffusion` (architecture + beamform front-end) and the real
`wavesim.wave_generator.jonswap_components`, and only replaces the TRAINING SCENE DISTRIBUTION.

It fixes the three MEASURED defects of the shipped checkpoint's training distribution
(results/direction3/beamform_cond_diffusion.pt, trained by `cd.make_batch`):

  D1  cardinality / sea-state OOD.  make_batch draws ONLY `jonswap_components(n_components=32)`
      with Hs~U[0.8,3.2], Tp~U[5,11].  The deployment envelope is BF2..BF5 = (Hs,Tp) of
      (0.3,4.0) (0.6,5.5) (1.2,7.0) (2.0,9.0) with n_components in {1,2,5,15,30}: the BF2 corner
      is OUTSIDE the training box in BOTH Hs and Tp, and a 1-component sea was never shown.
      Measured consequence: on BF2_1comp the net emits the max_n=30 cap of near-degenerate atoms
      ~0.74 kp away from the true swell (tmp/learned_only_2026-07-26/design_diag_q1.json).
      FIX: stratified cardinality nc in {1,2,5,15,30} with an exact energy-preserving 32-slot
      target (`replicate_to_M`), and (Hs,Tp) drawn half from the wide box, half from jittered
      BF2..BF5 anchors.

  D2  continuum tiling.  Everything the net ever saw was a 32-line discretisation of a JONSWAP
      density, so "tile ~24-30 atoms" is its only mode (tmp/discreteness_2026-07-26/RESULT.md;
      real n_eff ~ 24).  FIX: continuum/spread strata built with the FROZEN `gen_cells.py`
      sub-component expansion recipe (parents x M_sub, energy-preserving, optional omega
      jitter), observed from ALL true sub-components and supervised with a 32-atom
      energy-weighted k-means quantisation (`quantise_to_M`).

  D3  observation geometry.  make_batch pairs an (x,y) drawn i.i.d. from a pooled LiDAR
      footprint with an INDEPENDENT t~U[0,Tobs].  Deployment
      (`learned_init._gather_init_obs`) pools whole LiDAR frames: ~140 frames x ~50 valid
      returns, each return carrying ITS OWN frame time, then subsamples 400.  The (x,y) MARGINAL
      of the two is measured identical (r/az/x percentiles agree to <2%); the JOINT is not.
      tmp/wp_e2e_2026-07-25/WP_E2E_2026-07-25.md L172-174 localises the transfer leak exactly
      here.  FIX: `GeometryBank` replays real frame-structured (x,y,t) windows from donor
      episodes, so a training observation set is a genuine sweep window.

Donor geometry episodes are `data_finemesh_extras_2026-07-24/` -- disjoint from EVERY gate
episode list (see PREREG.md Sec 3).  Only (x, y, t_relative) is taken from a donor; the
elevation z is always re-rendered from the fresh synthetic scene.

Usage:
  python3 tmp/mixed_retrain_2026-07-26/gen_mixed.py --selftest
  python3 tmp/mixed_retrain_2026-07-26/gen_mixed.py --time --n 64
"""
import argparse
import os
import sys
import time

import numpy as np

# cd's import os.chdir()s to the repo root and pulls in torch/CUDA.
sys.path.insert(0, 'scripts')

DEPTH = 50.0
M = 32                      # atoms per spectrum (frozen: d3_setdiff_prior.M)
M_OBS = 400                 # obs per scene (frozen: d3_beamform_net.M_OBS)
NOISE_SIGMA = 0.02          # frozen: cd.make_batch
K_LIM_HARD = 1.1            # frozen: stats['K_LIM']

# ── frozen geometry-donor set (PREREG Sec 3.2) ───────────────────────────────
# 14 episodes, EVERY one absent from all four gate episode lists, covering both raydrop models.
#   uniform : BF6/BF7 (sea states no gate uses at all), freshly regenerated 2026-07-24
#   slope   : the 5comp seeds 2/3/4 (the 5comp family's seeds 0/1 are the only ones any
#             protocol ever used; 2/3/4 appear nowhere)
# Only (x, y, t_relative) is ever read; z is always re-rendered from the fresh synthetic scene.
DONOR_FILES = (
    [('data_finemesh_extras_2026-07-24/', f'episode_BF{bf}_{nc}comp_seed0_uniform.h5')
     for bf in (6, 7) for nc in (1, 5, 15, 30)] +
    [('data_finemesh_2026-07-23/', f'episode_BF{bf}_5comp_seed{s}_slope.h5')
     for bf in (3, 4) for s in (2, 3, 4)]
)

# ── frozen mixture (PREREG Sec 2) ────────────────────────────────────────────
STRATA = ('discrete_crowded', 'discrete_sparse', 'discrete_moderate',
          'continuum_mild', 'continuum_strong', 'intermediate_spread')
WEIGHTS = np.array([0.30, 0.20, 0.10, 0.10, 0.15, 0.15])

# deployment sea-state anchors (wavesim.pipeline.SEA_STATE, the declared BF2-BF5 envelope)
ANCHORS = ((0.3, 4.0), (0.6, 5.5), (1.2, 7.0), (2.0, 9.0))
ANCHOR_FRAC = 0.5           # half the scenes anchored, half from the wide box
HS_BOX = (0.25, 3.2)
TP_BOX = (3.8, 11.5)
ANCHOR_JITTER = 0.15        # +-15% multiplicative
TOBS_FACTOR = (1.9, 2.3)    # deployment: get_init_window(2.0); head-to-head: K_TP=2.2

# training scene seeds live in a band disjoint from every evaluation generator seed
# (all evaluation seeds ever used are < 2000; see PREREG Sec 3)
SEED_LO, SEED_HI = 2_000_000_000, 2_147_483_647


# ── observation geometry: real LiDAR sweep windows ───────────────────────────
class GeometryBank:
    """Frame-structured (x, y, t) windows replayed from donor episodes.

    Loads every donor episode's VALID lidar returns once, keeping the per-frame grouping and
    the frame timestamps.  `window(rng, Tobs)` picks a donor + a random start frame, gathers
    every valid return in [t0, t0+Tobs], shifts time so min=0 (exactly what
    learned_init.propose_atoms does) and subsamples M_OBS -- reproducing the deployment joint
    distribution of (x, y, t) instead of the i.i.d. product `cd.make_batch` used.
    """

    def __init__(self, files=DONOR_FILES, verbose=False, min_tmax=27.0):
        import h5py
        self.donors = []
        for donor_dir, fn in files:
            with h5py.File(os.path.join(donor_dir, fn), 'r') as f:
                P = f['frames/lidar_points'][:]            # (F, R, 3)
                V = f['frames/lidar_valid_mask'][:]        # (F, R)
                T = f['frames/timestamps'][:]              # (F,)
            xs, ys, ts = [], [], []
            for i in range(P.shape[0]):
                p = P[i][V[i]]
                if len(p) == 0:
                    continue
                xs.append(p[:, 0]); ys.append(p[:, 1])
                ts.append(np.full(len(p), T[i]))
            x = np.concatenate(xs).astype(np.float32)
            y = np.concatenate(ys).astype(np.float32)
            t = np.concatenate(ts).astype(np.float32)
            order = np.argsort(t, kind='stable')
            assert float(T[-1]) >= min_tmax, \
                f'{fn}: record only {T[-1]:.1f}s, shorter than the longest training window'
            self.donors.append(dict(name=fn, x=x[order], y=y[order], t=t[order],
                                    t_max=float(T[-1])))
            if verbose:
                print(f"  donor {fn}: {len(x)} valid returns over {T[-1]:.1f}s", flush=True)
        assert len(self.donors) == len(files)
        self.names = [d['name'] for d in self.donors]

    def window(self, rng, Tobs):
        """-> (ox, oy, ot) float32, M_OBS points, ot in [0, Tobs] with frame structure kept."""
        d = self.donors[rng.integers(len(self.donors))]
        span = max(d['t_max'] - Tobs, 0.0)
        t0 = rng.uniform(0.0, span) if span > 0 else 0.0
        lo = np.searchsorted(d['t'], t0, 'left')
        hi = np.searchsorted(d['t'], t0 + Tobs, 'right')
        n = hi - lo
        if n < M_OBS:                                    # degenerate: widen to the whole record
            lo, hi, n = 0, len(d['t']), len(d['t'])
            t0 = 0.0
        idx = lo + rng.choice(n, M_OBS, replace=n < M_OBS)
        return (d['x'][idx].astype(np.float32), d['y'][idx].astype(np.float32),
                (d['t'][idx] - d['t'][lo:hi].min()).astype(np.float32))


# ── 32-slot target construction ──────────────────────────────────────────────
def replicate_to_M(kx, ky, A, m=M):
    """n_true <= m: round-robin replicate to exactly m slots, splitting each component's ENERGY
    equally over its replicas (A_rep = A / sqrt(count)).  Total energy is preserved exactly and
    the m atoms occupy exactly n_true DISTINCT (kx,ky) locations -> learned_init.pool_and_select's
    dedup (min-sep 0.03) collapses the proposal back to n_true.  This is the same
    energy-preserving construction gen_cells.expand() uses with sigma_sub=0."""
    n = len(kx)
    slot = np.arange(m) % n
    cnt = np.bincount(slot, minlength=n).astype(np.float64)
    return kx[slot], ky[slot], A[slot] / np.sqrt(cnt[slot])


FILL_AMP_FRAC = 0.10        # round 2: filler amplitude = 0.10 * A_max -> energy 1% of E_max


def energy_graded2_to_M(kx, ky, A, m=M, lo_e=-15.656161308288574):
    """ROUND 3 (`TARGET_MODE='graded2'`, PREREG Addendum 2).  As `energy_graded_to_M` but with a
    33x margin below `pool_and_select`'s 3 %-of-max energy threshold instead of 3x:

        A_fill = min(0.03 * A_max, 0.5 * A_min_real), floored at exp(lo_e / 2)

    The 0.03*A_max term puts filler energy at 0.09 % of peak; the 0.5*A_min cap guarantees a filler
    is never stronger than the weakest REAL component, protecting the crowded selection ordering."""
    n = len(kx)
    slot = np.arange(m) % n
    amax = float(A.max()); amin = float(A.min())
    a_fill = max(min(0.03 * amax, 0.5 * amin), float(np.exp(lo_e / 2.0)))
    out_A = np.full(m, a_fill, np.float32)
    out_A[:n] = A
    return kx[slot], ky[slot], out_A


def energy_graded_to_M(kx, ky, A, m=M):
    """ROUND 2 (`TARGET_MODE='graded'`, PREREG Addendum 1).  n_true <= m: the first n_true slots
    carry the true components at FULL amplitude; the remaining slots repeat those LOCATIONS at
    ``A_fill = FILL_AMP_FRAC * A_max`` (energy = 1 % of the scene's strongest component, well under
    ``pool_and_select``'s 3 %-of-max threshold).

    Round 1's `replicate_to_M` was exactly energy-preserving but gave all 32 slots the SAME energy,
    so on a sparse sea every proposed atom cleared the energy filter and dedup -- defeated by the
    proposal's constant-ABSOLUTE-width scatter -- was the only thing that could reduce the count.
    Grading puts cardinality INTO the channel `pool_and_select` actually reads.  The energy channel
    is therefore a salience channel, not an exact decomposition; deployment re-fits amplitudes by
    LS so nothing downstream depends on the difference."""
    n = len(kx)
    slot = np.arange(m) % n
    out_A = np.full(m, FILL_AMP_FRAC * float(A.max()), np.float32)
    out_A[:n] = A                                     # first n_true slots = the real spectrum
    return kx[slot], ky[slot], out_A


def quantise_to_M(kx, ky, A, rng, m=M, iters=25):
    """n_true > m: energy-weighted k-means in (kx,ky) -> m cluster centroids carrying the SUMMED
    cluster energy.  This is the honest 32-atom representation of a continuum: the world has
    hundreds of sub-components, the model has 32 slots."""
    w = A.astype(np.float64) ** 2
    w = np.maximum(w, 1e-18)
    p = w / w.sum()
    # energy-weighted k-means++ style init: sample m distinct seeds proportional to energy
    n = len(kx)
    sel = rng.choice(n, size=min(m, n), replace=False, p=p) if n >= m else np.arange(n)
    cx, cy = kx[sel].astype(np.float64).copy(), ky[sel].astype(np.float64).copy()
    lab = np.zeros(n, np.int64)
    for _ in range(iters):
        d2 = (kx[:, None] - cx[None, :]) ** 2 + (ky[:, None] - cy[None, :]) ** 2
        lab = np.argmin(d2, axis=1)
        moved = 0.0
        for j in range(m):
            msk = lab == j
            if not msk.any():                            # re-seed an empty cluster on the
                i = int(rng.choice(n, p=p))              # energy distribution
                nx, ny = float(kx[i]), float(ky[i])
            else:
                ww = w[msk]
                nx = float(np.sum(ww * kx[msk]) / ww.sum())
                ny = float(np.sum(ww * ky[msk]) / ww.sum())
            moved += abs(nx - cx[j]) + abs(ny - cy[j])
            cx[j], cy[j] = nx, ny
        if moved < 1e-9:
            break
    d2 = (kx[:, None] - cx[None, :]) ** 2 + (ky[:, None] - cy[None, :]) ** 2
    lab = np.argmin(d2, axis=1)
    en = np.array([w[lab == j].sum() for j in range(m)])
    en = np.maximum(en, 1e-18)
    return cx.astype(np.float32), cy.astype(np.float32), np.sqrt(en).astype(np.float32)


# ── scene sampling ───────────────────────────────────────────────────────────
def _draw_HsTp(rng):
    if rng.uniform() < ANCHOR_FRAC:
        Hs, Tp = ANCHORS[rng.integers(len(ANCHORS))]
        j = ANCHOR_JITTER
        return (float(Hs * rng.uniform(1 - j, 1 + j)), float(Tp * rng.uniform(1 - j, 1 + j)))
    return float(rng.uniform(*HS_BOX)), float(rng.uniform(*TP_BOX))


def _expand_parents(parents, M_sub, sigma_sub_deg, omega_jitter, d_omega, rng):
    """gen_cells.expand(), reimplemented on plain arrays (no WaveComponent / Q bookkeeping --
    Q never enters the rendered field).  Energy-preserving: A_sub = A_parent / sqrt(M_sub)."""
    from wavesim.wave_generator import wavenumber_from_omega
    if M_sub == 1 and sigma_sub_deg == 0.0 and not omega_jitter:
        return parents
    kx, ky, A, om, ph = parents
    sig = np.deg2rad(sigma_sub_deg)
    n = len(kx)
    th0 = np.arctan2(ky, kx)
    th = rng.normal(th0[:, None], sig, (n, M_sub))
    if omega_jitter:
        omx = rng.uniform(om[:, None] - 0.5 * d_omega, om[:, None] + 0.5 * d_omega,
                          (n, M_sub))
        kk = np.array([[wavenumber_from_omega(float(w), DEPTH) for w in row] for row in omx])
    else:
        omx = np.repeat(om[:, None], M_sub, 1)
        kk = np.repeat(np.hypot(kx, ky)[:, None], M_sub, 1)
    A_sub = np.repeat(A[:, None], M_sub, 1) / np.sqrt(M_sub)
    phx = rng.uniform(0.0, 2 * np.pi, (n, M_sub))
    return (np.ravel(kk * np.cos(th)).astype(np.float32),
            np.ravel(kk * np.sin(th)).astype(np.float32),
            np.ravel(A_sub).astype(np.float32),
            np.ravel(omx).astype(np.float32),
            np.ravel(phx).astype(np.float32))


def sample_scene(rng, bank, stratum=None):
    """One training scene.  Returns dict(atoms (M,3) normalised-free raw kx/ky/A, obs P/z, meta)."""
    from wavesim.wave_generator import jonswap_components
    from nonlinear_forward import elevation

    if stratum is None:
        stratum = STRATA[int(rng.choice(len(STRATA), p=WEIGHTS))]
    Hs, Tp = _draw_HsTp(rng)
    gamma = float(rng.uniform(1.5, 4.0))
    dird = float(rng.uniform(-50, 50))
    seed = int(rng.integers(SEED_LO, SEED_HI))

    if stratum == 'discrete_crowded':
        n_par, M_sub, sig_sub, jit = int(rng.choice([15, 30])), 1, 0.0, False
    elif stratum == 'discrete_sparse':
        n_par, M_sub, sig_sub, jit = int(rng.choice([1, 2])), 1, 0.0, False
    elif stratum == 'discrete_moderate':
        n_par, M_sub, sig_sub, jit = 5, 1, 0.0, False
    elif stratum == 'continuum_mild':
        n_par, M_sub, sig_sub, jit = 15, 31, 10.0, False
    elif stratum == 'continuum_strong':
        n_par, M_sub, sig_sub, jit = 15, 31, 28.0, True
    elif stratum == 'intermediate_spread':
        n_par = int(rng.choice([5, 15, 30]))
        M_sub = int(rng.choice([3, 7, 15]))
        sig_sub = float(rng.uniform(2.0, 20.0))
        jit = bool(rng.uniform() < 0.5)
    else:
        raise ValueError(stratum)

    # total directional spread budget, split parent/sub exactly as gen_cells.parent_spread()
    total_spread = float(rng.uniform(max(12.0, sig_sub + 4.0), 40.0))
    par_spread = float(np.sqrt(max(total_spread ** 2 - sig_sub ** 2, 1e-6)))

    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=n_par, depth=DEPTH, gamma=gamma,
                           direction_deg=dird, direction_spread_deg=par_spread, seed=seed)
    parents = (np.array([x.kx for x in c], np.float32), np.array([x.ky for x in c], np.float32),
               np.array([x.A for x in c], np.float32), np.array([x.omega for x in c], np.float32),
               np.array([x.phi for x in c], np.float32))
    omega_p = 2 * np.pi / Tp
    d_omega = (3.0 - 0.5) * omega_p / max(n_par, 1)          # gen_cells.make_patch convention
    kx, ky, A, om, ph = _expand_parents(parents, M_sub, sig_sub, jit, d_omega, rng)

    # Representability filter.  The 32-slot atom code cannot hold |k| > K_LIM = 1.1, and the
    # beamform conditioner only spans |k| <= 0.6 anyway.  Short-Tp / high-nc JONSWAP tails run
    # past that.  Those components stay in the OBSERVATIONS (they are real waves the sensor
    # sees, exactly as at deployment) but are excluded from the TARGET -- the honest statement
    # that the representation is band-limited.  Scenes whose representable energy fraction is
    # below REPR_MIN_FRAC are rejected by the caller.
    inb = np.hypot(kx, ky) <= K_LIM_HARD
    e_tot = float(np.sum(A ** 2)) + 1e-18
    repr_frac = float(np.sum(A[inb] ** 2)) / e_tot
    if not inb.any():
        return None
    fkx, fky, fA = kx[inb], ky[inb], A[inb]

    n_true = len(fkx)
    if n_true <= M:
        if TARGET_MODE == 'graded2':
            tkx, tky, tA = energy_graded2_to_M(fkx, fky, fA)
        elif TARGET_MODE == 'graded':
            tkx, tky, tA = energy_graded_to_M(fkx, fky, fA)
        else:
            tkx, tky, tA = replicate_to_M(fkx, fky, fA)
    else:
        tkx, tky, tA = quantise_to_M(fkx, fky, fA, rng)
    # canonical order: |k| ascending, strongest first within a location (graded mode puts the
    # full-energy atom ahead of its co-located fillers)
    order = np.lexsort((-tA, np.hypot(tkx, tky)))
    tkx, tky, tA = tkx[order], tky[order], tA[order]

    Tobs = float(rng.uniform(*TOBS_FACTOR)) * Tp
    ox, oy, ot = bank.window(rng, Tobs)
    z = (elevation(kx, ky, A, om, ph, ox.astype(np.float64), oy.astype(np.float64),
                   ot.astype(np.float64), DEPTH, nonlinear=False)
         + rng.normal(0, NOISE_SIGMA, M_OBS)).astype(np.float32)

    return dict(tkx=tkx, tky=tky, tA=tA,
                P=np.column_stack([ox, oy, ot]).astype(np.float32), z=z,
                stratum=stratum, n_true=int(len(kx)), n_repr=int(n_true),
                n_distinct=int(min(n_true, M)), repr_frac=repr_frac,
                Hs=Hs, Tp=Tp, Tobs=Tobs, sigma_sub=sig_sub, n_par=n_par, M_sub=M_sub)


REPR_MIN_FRAC = 0.5          # reject scenes whose in-band energy fraction is below this
TARGET_MODE = 'replicate'    # 'replicate' = round 1 (energy-preserving); 'graded' = round 2


def sample_scene_ok(rng, bank, stratum=None, max_tries=20):
    """sample_scene + the frozen representability rejection (PREREG Sec 2.4)."""
    for _ in range(max_tries):
        s = sample_scene(rng, bank, stratum=stratum)
        if s is not None and s['repr_frac'] >= REPR_MIN_FRAC:
            return s
    return s


def batch_from_scenes(scenes, stats, device):
    """scenes -> (normalised atom tensor (B,M,3), beamform map (B,NK,NK)).  Same normalisation
    and same `cd.beamform` front-end as cd.build_from_comps -- nothing about the model changes."""
    import torch
    import d3_beamform_cond_diffusion as cd
    K_LIM = stats['K_LIM']; lo_e, hi_e = stats['lo_e'], stats['hi_e']
    B = len(scenes)
    atoms = np.zeros((B, M, 3), np.float32)
    P = np.zeros((B, M_OBS, 3), np.float32)
    Z = np.zeros((B, M_OBS), np.float32)
    for b, s in enumerate(scenes):
        atoms[b, :, 0] = s['tkx'] / K_LIM
        atoms[b, :, 1] = s['tky'] / K_LIM
        atoms[b, :, 2] = 2 * (np.log(s['tA'] ** 2 + 1e-12) - lo_e) / (hi_e - lo_e) - 1
        P[b] = s['P']; Z[b] = s['z']
    Bmap = cd.beamform(torch.tensor(P, device=device), torch.tensor(Z, device=device))
    return torch.tensor(atoms, device=device), Bmap


# ── self-test / timing ───────────────────────────────────────────────────────
def selftest():
    rng = np.random.default_rng(0)
    bank = GeometryBank(verbose=True)
    print(f"\ngeometry bank: {len(bank.donors)} donors")

    # G-energy: replicate_to_M preserves total energy and the distinct-location count
    for n in (1, 2, 5, 15, 30):
        A = rng.uniform(0.05, 0.4, n).astype(np.float32)
        kx = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        ky = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        rkx, rky, rA = replicate_to_M(kx, ky, A)
        ndist = len(np.unique(np.round(np.column_stack([rkx, rky]), 9), axis=0))
        print(f"  replicate n={n:2d}: slots={len(rkx)} distinct={ndist} "
              f"dE={float(np.sum(rA**2) - np.sum(A**2)):+.3e}")
        assert ndist == n and len(rkx) == M

    # G-quantise: k-means preserves total energy
    kx = rng.normal(0.1, 0.05, 465).astype(np.float32)
    ky = rng.normal(0.0, 0.05, 465).astype(np.float32)
    A = rng.uniform(0.005, 0.05, 465).astype(np.float32)
    qx, qy, qA = quantise_to_M(kx, ky, A, rng)
    print(f"  quantise 465->32: dE={float(np.sum(qA**2) - np.sum(A**2)):+.3e} "
          f"spread_kx={float(qx.std()):.4f}")

    # G-geometry: replayed windows must reproduce the deployment (x,y,t) JOINT structure
    ox, oy, ot = bank.window(rng, 14.0)
    ut = np.unique(np.round(ot, 4))
    print(f"\n  sweep window: {len(ox)} pts, {len(ut)} distinct times, "
          f"t[{ot.min():.2f},{ot.max():.2f}], pts/time={len(ox)/len(ut):.2f}")
    print(f"  x pct {np.round(np.percentile(ox, [1, 25, 50, 75, 99]), 1)}  "
          f"r pct {np.round(np.percentile(np.hypot(ox, oy), [1, 50, 99]), 1)}")
    assert len(ut) > 50, 'window lost its frame structure'

    # G-scene: every stratum produces finite atoms inside the k-limit and a sane Bmap
    print()
    for st in STRATA:
        for _ in range(30):
            s = sample_scene_ok(rng, bank, stratum=st)
            kmax = float(np.hypot(s['tkx'], s['tky']).max())
            nd = len(np.unique(np.round(np.column_stack([s['tkx'], s['tky']]), 6), axis=0))
            assert np.isfinite(s['z']).all() and kmax <= K_LIM_HARD, (st, kmax)
            assert np.isfinite(s['tA']).all() and (s['tA'] > 0).all()
        print(f"  {st:20s} n_true={s['n_true']:4d} n_repr={s['n_repr']:4d} "
              f"distinct_atoms={nd:2d} kmax={kmax:.3f} repr_frac={s['repr_frac']:.3f} "
              f"Hs={s['Hs']:.2f} Tp={s['Tp']:.1f} std(z)={float(s['z'].std()):.3f}")

    # G-cardinality: what the 32-slot target implies for pool_and_select's distinct count
    print()
    for st in ('discrete_sparse', 'discrete_moderate', 'discrete_crowded'):
        nds = []
        for _ in range(200):
            s = sample_scene_ok(rng, bank, stratum=st)
            nds.append(len(np.unique(np.round(np.column_stack([s['tkx'], s['tky']]), 6), axis=0)))
        print(f"  {st:20s} distinct-atom count: median {np.median(nds):.1f} "
              f"range [{min(nds)},{max(nds)}]")

    # G-footprint: the replayed bank marginal vs the deployment gate episodes' marginal
    import h5py
    pool = np.concatenate([np.column_stack([d['x'], d['y']]) for d in bank.donors])
    ev = []
    for dd, fn in (('data_heldout40_2026-07-26/', 'episode_BF4_15comp_seed777_uniform.h5'),
                   ('data_heldout40_2026-07-26/', 'episode_BF2_1comp_seed888_slope.h5')):
        with h5py.File(os.path.join(dd, fn), 'r') as f:
            P = f['frames/lidar_points'][:]; V = f['frames/lidar_valid_mask'][:]
            ev.append(P[V][:, :2])
    ev = np.concatenate(ev)
    for nm, a in (('BANK ', pool), ('GATE ', ev)):
        r = np.hypot(a[:, 0], a[:, 1])
        az = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
        print(f"  {nm} r pct {np.round(np.percentile(r, [1, 25, 50, 75, 95, 99]), 1)}  "
              f"az pct {np.round(np.percentile(az, [1, 50, 99]), 1)}")

    # G-mixture: empirical proportions
    cnt = {k: 0 for k in STRATA}
    for _ in range(4000):
        cnt[STRATA[int(rng.choice(len(STRATA), p=WEIGHTS))]] += 1
    print("\n  mixture draw (n=4000): " +
          ", ".join(f"{k}={v/4000:.3f}" for k, v in cnt.items()))
    print("\nSELFTEST OK")


def timing(n):
    rng = np.random.default_rng(1)
    bank = GeometryBank()
    t0 = time.time()
    scenes = [sample_scene_ok(rng, bank) for _ in range(n)]
    t1 = time.time()
    print(f"scene gen: {n} scenes in {t1-t0:.2f}s -> {(t1-t0)/n*1000:.1f} ms/scene, "
          f"{(t1-t0)/n*32:.2f}s per batch-of-32")
    per = {}
    for s in scenes:
        per.setdefault(s['stratum'], 0)
        per[s['stratum']] += 1
    print("  realised strata:", {k: round(v / n, 3) for k, v in sorted(per.items())})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--time', action='store_true')
    ap.add_argument('--n', type=int, default=64)
    a = ap.parse_args()
    # import for the os.chdir to the repo root (cd does it on import)
    import d3_beamform_cond_diffusion  # noqa: F401
    if a.selftest:
        selftest()
    if a.time:
        timing(a.n)
