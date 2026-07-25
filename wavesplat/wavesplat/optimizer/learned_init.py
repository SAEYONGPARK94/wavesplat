"""learned_init.py -- learned set-proposal initializer (WavePrior Stage-1).

ADDITIVE MODULE. Provides the component set (kx, ky) -- and, in the default
non-konly variant, LS-fit (w, phi) -- for ``init_method='learned'``. It replaces
the v2 dispersion-beamforming source of kx/ky with the beamform-conditioned
set-diffusion proposal + Gauss-Newton polish
(``scripts/d3_beamform_cond_diffusion.py``). The existing joint fit / adaptive-N /
tracker then run on top unchanged.

Nothing here is imported unless ``init_method == 'learned'``; the shipped default
path (``v2_konly``) never touches this file.

Two entry points, so the GPU proposal and the CPU tracker can be decoupled:
  * ``propose_atoms(frames, depth, ...)`` -- runs the diffusion proposal + polish on
    the init-window observations (needs the checkpoint / torch; used on GPU).
  * ``learned_init_params(frames, depth, config, seed)`` -- returns installable
    (kx, ky, w, phi) tensors. If ``config['learned_atoms']`` carries precomputed raw
    atoms (kx, ky, energy) it uses them (CPU-only path, no diffusion); otherwise it
    calls ``propose_atoms`` inline.

Design notes (see tmp/wp_integration_2026-07-25/PREREG.md):
  * The proposal uses the SAME init window (get_init_window tp_factor=2.0) the v2
    path uses, so both arms see identical observations. This is the aperture the
    head-to-head nowcast used too (init 2.0*Tp ~ head-to-head 2.2*Tp).
  * Obs times are shifted so min=0 for the beamform-map / polish (matches the net's
    training distribution and the head-to-head). kx/ky/energy are time-origin
    independent (polish uses the phase-marginalized likelihood). The LS fit for
    w/phi is done at ABSOLUTE frame times so phi is in the SpectralGaussianModel
    convention (phase = kx*x+ky*y-w*t+phi), verified identical to
    tnse.fit_amplitudes_phases and render_wave_field.
"""
import os
import sys
import math
import numpy as np

DEFAULT_CKPT = 'results/direction3/beamform_cond_diffusion.pt'
M_OBS_CAP = 400          # matches d3_beamform_net.M_OBS / the head-to-head cond arm
_CACHE = {}


# ── observation gathering ────────────────────────────────────────────────────
def _gather_init_obs(frames):
    """All init-window observations as flat numpy (x, y, z, t_absolute)."""
    xs, ys, zs, ts = [], [], [], []
    for f in frames:
        n = len(f.x_obs)
        if n == 0:
            continue
        xs.append(f.x_obs.detach().cpu().numpy())
        ys.append(f.y_obs.detach().cpu().numpy())
        zs.append(f.z_obs.detach().cpu().numpy())
        ts.append(np.full(n, float(f.t)))
    if not xs:
        return None
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(zs), np.concatenate(ts))


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_cond(ckpt, device):
    """Load (cd module, model, stats, abar, betas, denorm_atoms). Cached per ckpt/device.

    NOTE: importing scripts/d3_beamform_cond_diffusion os.chdir's to the repo root.
    Only ever called on the GPU proposal path.
    """
    import torch
    key = (os.path.abspath(ckpt), str(device))
    if key in _CACHE:
        return _CACHE[key]
    scripts = os.path.join(_repo_root(), 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import d3_beamform_cond_diffusion as cd
    from d3_setdiff_prior import denorm_atoms
    stats = torch.load(cd.PRIOR_CKPT, map_location=device)['stats']
    betas = cd.cosine_betas(cd.T).to(device)
    abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    _CACHE[key] = (cd, model, stats, abar, betas, denorm_atoms)
    return _CACHE[key]


# ── GPU proposal (diffusion sample + polish) ─────────────────────────────────
def propose_atoms(frames, depth, ckpt=DEFAULT_CKPT, nsamp=4, diffusion_seed=0,
                  device=None, polish_iters=80, subsample_seed=0):
    """Run the cond-diffusion proposal + polish on the init-window observations.

    Returns raw concatenated atoms across the nsamp posterior samples:
    (kx, ky, energy) numpy arrays, length nsamp*M. Selection/dedup happens later in
    ``pool_and_select`` so the inline and cached paths are byte-identical.
    """
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cd, model, stats, abar, betas, denorm_atoms = _load_cond(ckpt, device)

    obs = _gather_init_obs(frames)
    if obs is None:
        return np.array([]), np.array([]), np.array([])
    x, y, z, t = obs
    P = np.column_stack([x, y, t]).astype(np.float32)
    zf = z.astype(np.float32)
    if len(zf) > M_OBS_CAP:
        idx = np.random.default_rng(subsample_seed).choice(len(zf), M_OBS_CAP,
                                                           replace=False)
        P, zf = P[idx], zf[idx]
    t0 = P[:, 2].min()
    Ps = P.copy()
    Ps[:, 2] -= t0                       # shift for beamform-map / polish

    Bmap = cd._bmap_from_obs(Ps, zf)
    X = cd.sample(model, Bmap, abar, betas, n=nsamp, seed=int(diffusion_seed))
    Pt = torch.tensor(Ps, device=device, dtype=torch.float32)
    zt = torch.tensor(zf, device=device, dtype=torch.float32)

    kx_all, ky_all, en_all = [], [], []
    for b in range(nsamp):
        xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=polish_iters)
        kx, ky, a = denorm_atoms(xb, stats)          # a = energy = A^2
        kx_all.append(kx.detach().cpu().numpy())
        ky_all.append(ky.detach().cpu().numpy())
        en_all.append(np.clip(a.detach().cpu().numpy(), 0.0, None))
    return (np.concatenate(kx_all), np.concatenate(ky_all),
            np.concatenate(en_all))


# ── pooling / selection (identical for inline + cached) ──────────────────────
def pool_and_select(kx, ky, energy, energy_thresh=0.03, dedup_dk=0.03, max_n=30):
    """Pool the raw multi-sample atoms into a de-duplicated candidate component set.

    Threshold by relative energy, greedily keep strongest-first while enforcing a
    small min-separation (dedup_dk ~ grid step, NOT the analytic MIN_DK=0.15 floor),
    cap at max_n. Returns (kx, ky, energy) selected, strongest first.
    """
    kx = np.asarray(kx, float); ky = np.asarray(ky, float)
    en = np.asarray(energy, float)
    if en.size == 0 or en.max() <= 0:
        return np.array([]), np.array([]), np.array([])
    keep = en > energy_thresh * en.max()
    kx, ky, en = kx[keep], ky[keep], en[keep]
    order = np.argsort(-en)
    sk, sy, se = [], [], []
    for i in order:
        dup = any(abs(kx[i] - jx) < dedup_dk and abs(ky[i] - jy) < dedup_dk
                  for jx, jy in zip(sk, sy))
        if not dup:
            sk.append(kx[i]); sy.append(ky[i]); se.append(en[i])
        if len(sk) >= max_n:
            break
    return np.array(sk), np.array(sy), np.array(se)


def ls_fit_wphi(kx, ky, frames, depth):
    """LS-fit amplitude+phase for the selected (kx,ky) on the init obs (absolute t).

    Uses tnse.fit_amplitudes_phases; its convention matches SpectralGaussianModel
    exactly, so the returned (A, phi) install directly as (w, phi).
    """
    from wavesplat.model.tnse import fit_amplitudes_phases
    obs = _gather_init_obs(frames)
    if obs is None or len(kx) == 0:
        return np.zeros(len(kx)), np.zeros(len(kx))
    x, y, z, t = obs
    obs_xyz = np.column_stack([x, y, t, z]).astype(np.float64)
    A, phi = fit_amplitudes_phases(np.asarray(kx, float), np.asarray(ky, float),
                                   obs_xyz, depth, lambda_reg=0.5)
    return np.abs(A), phi


# ── installable init parameters ──────────────────────────────────────────────
def learned_init_params(frames, depth, config=None, seed=None):
    """Return (init_kx, init_ky, init_w, init_phi, n) torch tensors + component count.

    config keys (all optional):
      learned_atoms   : dict{'kx','ky','energy'} of RAW pooled atoms (skips diffusion)
      learned_ckpt    : checkpoint path (default results/direction3/beamform_cond_diffusion.pt)
      learned_nsamp   : diffusion samples (default 4)
      learned_diffusion_seed : ancestral sampling seed (default: seed, else 0)
      learned_konly   : if True, use v2_konly-style w=0.01 / random phi instead of LS fit
      learned_energy_thresh / learned_dedup_dk : selection knobs
      max_n           : component cap (default 30)
    """
    import torch
    config = config or {}
    max_n = int(config.get('max_n', 30))
    e_th = float(config.get('learned_energy_thresh', 0.03))
    ddk = float(config.get('learned_dedup_dk', 0.03))

    cache = config.get('learned_atoms')
    if cache is not None:
        rkx, rky, ren = (np.asarray(cache['kx']), np.asarray(cache['ky']),
                         np.asarray(cache['energy']))
    else:
        ckpt = config.get('learned_ckpt', DEFAULT_CKPT)
        nsamp = int(config.get('learned_nsamp', 4))
        dseed = int(config.get('learned_diffusion_seed',
                               0 if seed is None else seed))
        rkx, rky, ren = propose_atoms(frames, depth, ckpt=ckpt, nsamp=nsamp,
                                      diffusion_seed=dseed)

    kx, ky, en = pool_and_select(rkx, rky, ren, energy_thresh=e_th,
                                 dedup_dk=ddk, max_n=max_n)
    n = len(kx)
    if n == 0:
        # Degenerate fallback: a few random low-amplitude components (never expected
        # on crowded/sparse seas; the proposal always returns atoms).
        rng = np.random.RandomState(seed) if seed is not None else np.random
        n = 3
        return (torch.randn(n) * 0.05, torch.randn(n) * 0.05,
                torch.ones(n) * 0.01,
                torch.tensor(rng.uniform(0, 2 * math.pi, n), dtype=torch.float32),
                n)

    if config.get('learned_konly', False):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        w = np.ones(n) * 0.01
        phi = rng.uniform(0, 2 * math.pi, n)
    else:
        w, phi = ls_fit_wphi(kx, ky, frames, depth)

    return (torch.tensor(kx, dtype=torch.float32),
            torch.tensor(ky, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(phi, dtype=torch.float32),
            n)


def raw_field_nrmse(config, frames, depth, gt_field, grid_x, grid_y, t, Hs,
                    seed=None):
    """Diagnostic: nRMSE of the learned proposal rendered field (LS w/phi, NO joint
    fit) at (grid, t). Isolates the raw-init quality before the WaveSplat pipeline."""
    from wavesplat.model.tnse import render_wave_field
    kx, ky, w, phi, n = learned_init_params(frames, depth, config, seed=seed)
    if n == 0:
        return float('nan')
    pred = render_wave_field(kx.numpy(), ky.numpy(), w.numpy(), phi.numpy(),
                             depth, np.asarray(grid_x), np.asarray(grid_y), float(t))
    gt = np.asarray(gt_field)
    return float(np.sqrt(np.mean((pred - gt) ** 2)) / max(Hs, 1e-6))
