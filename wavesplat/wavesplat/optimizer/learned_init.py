"""learned_init.py -- learned set-proposal initializer (WavePrior Stage-1).

ADDITIVE MODULE. Provides the component set (kx, ky) -- and, in the default
non-konly variant, LS-fit (w, phi) -- for ``init_method='learned'``. It replaces
the v2 dispersion-beamforming source of kx/ky with the beamform-conditioned
set-diffusion proposal + Gauss-Newton polish
(``scripts/d3_beamform_cond_diffusion.py``). The existing joint fit / adaptive-N /
tracker then run on top unchanged.

⛔**2026-08-24 정정 — 위 문장의 뒷부분은 낡았다.** ``init_method == 'learned'`` 일 때만
임포트되는 것은 여전히 맞지만, *"the shipped default path (v2_konly) never touches this
file"* 는 **2026-08-22 경로 통일 이후 거짓**이다: ``AUTO_SWITCH_THRESH_DEFAULT = 0`` 이므로
출하 기본값(``init_method='auto'``)에서 **합성 입력은 100% 이 파일로 라우팅된다**.
⟹ ★**이 모듈이 곧 배포된 합성 경로다**(실측은 여전히 ``bank_mmse`` 로 빠진다).

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

def _repo_root():
    """Absolute repo root (…/wavesplat), independent of the caller's cwd."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── DEPLOYED CHECKPOINT (2026-07-27) ─────────────────────────────────────────
# The mixed-curriculum retrain `_mixed_v1` replaced the original
# `beamform_cond_diffusion.pt` after the SECOND confirmation round on the fresh
# reserve-seed 40 set authorised the `auto_v1_m3` configuration
# (tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §1 and §4).
#   * BOTH candidates (v1, v3) passed every pre-registered bar (P0, P1c, P2-P5).
#   * v1 was picked by the FROZEN tie-break (crowded paired 11/16). That is a
#     DECISION RULE, not evidence that v1 beats v3 -- 11/16 occurs with p=0.105
#     under H0, so "v1 is superior to v3" must not be written anywhere.
# Provenance (`sha256sum`), recorded so a swapped/corrupted file is detectable:
#   beamform_cond_diffusion_mixed_v1.pt : 636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9
#   beamform_cond_diffusion.pt (previous): d28c25dee309361f8c95ec0134cdb3e28dd7bdf02417bdac2ae11567e32968fe
# The path is anchored at the repo root rather than left relative: the estimator is
# imported from harnesses with arbitrary cwd, and a relative default silently
# degraded to the analytic arm (FileNotFoundError -> v2_konly fallback) whenever the
# cwd was not the repo root.
DEFAULT_CKPT_REL = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'
DEFAULT_CKPT_SHA256 = '636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9'
DEFAULT_CKPT = os.path.join(_repo_root(), DEFAULT_CKPT_REL)

M_OBS_CAP = 400          # matches d3_beamform_net.M_OBS / the head-to-head cond arm
_CACHE = {}


# ── 성분 수 상한 ① 「제안 풀링 상한」 — SINGLE SOURCE ─────────────────────────
# ★2026-09-05 분리. 그전까지 config 키 `max_n` **하나가 서로 무관한 두 역할**을 했다:
#   ① 여기 `pool_and_select` 의 **제안 풀링 상한** (설치되는 원자 수의 상한)
#   ② `sequential_update` 의 **성장 천장** (`GROWTH_CEILING_DEFAULT` 참조)
# 두 값이 같은 키를 읽으므로 **성분 수 축의 실험이 전부 교락**돼 있었다 — 풀링을 낮추면
# 성장 분기가 **통째로 꺼진다**(측정 12/12, `tmp/B2_REAL_2026-08-23/RESULT_B2F.md`;
# `sequential_update._adaptive_prune_grow` 의 이른 반환 주석이 그것을 축자로 적었다).
# 2026-09-05 실측: 제3자 코퍼스에서 배포 팔이 풀링 상한을 **24/24 블록에서 포화**한다
# ⟹ 그 코퍼스에서 성장 분기는 **한 번도 발화하지 않는다**.
#
# ⛔**배포 거동은 한 비트도 바뀌지 않는다** — 둘 다 기본값 30 이고, `DEFAULT_CONFIG` 는
#   여전히 `max_n` **하나만** 싣는다. 새 키는 기본 구성에 **없다**(아래 `resolve_cap` 의
#   우선순위 주석 참조) ⟹ `{**DEFAULT_CONFIG, 'max_n': X}` 로 정의된 **동결 배치 팔은
#   지금도 두 역할이 함께 X 로 간다** = 의미 불변.
POOL_CAP_DEFAULT = 30


def resolve_cap(config, key, default):
    """성분 수 상한 해소 — **두 역할이 공유하는 유일한 구현**.

    우선순위: ``config[key]`` → 레거시 ``config['max_n']`` → ``default``.

    ★레거시 `max_n` 이 **둘 다** 설정한다는 것이 하위호환의 전부다. 그래서
    `run.DEFAULT_CONFIG` 에 `pool_cap`/`growth_ceiling` 을 **넣지 않았다** — 넣었다면
    `{**DEFAULT_CONFIG, 'max_n': 10}` 인 동결 팔에서 기본 30 이 10 을 덮어써
    **기록된 배치의 의미가 조용히 바뀐다**(저장소가 λ 에서 이미 치른 대가).

    ⚠**형변환을 여기서 하지 않는다.** 종전 두 역할은 형변환이 **서로 달랐다** —
    풀링 쪽은 `int(cfg.get('max_n', 30))`, 성장 쪽은 `config.get('max_n', 30)` 로
    맨값이었다. 여기서 `int()` 를 강제하면 float `max_n` 을 쓴 구성의 성장 거동이
    바뀐다(예: 10.5 → 10). 각 역할의 래퍼가 **종전에 하던 만큼만** 한다.
    """
    if config:
        v = config.get(key)
        if v is not None:
            return v
        v = config.get('max_n')
        if v is not None:
            return v
    return default


def resolve_pool_cap(config):
    """역할 ①: 제안 풀링 상한. config 키 ``pool_cap`` (레거시 ``max_n``).

    `int()` 는 종전 두 풀링 사이트가 하던 그대로다.
    """
    return int(resolve_cap(config, 'pool_cap', POOL_CAP_DEFAULT))


def ckpt_sha256(path=None):
    """sha256 of a checkpoint file (default: the deployed one). Provenance check."""
    import hashlib
    p = path or DEFAULT_CKPT
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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
def pool_and_select(kx, ky, energy, energy_thresh=0.03, dedup_dk=0.03,
                    max_n=POOL_CAP_DEFAULT):
    """Pool the raw multi-sample atoms into a de-duplicated candidate component set.

    Threshold by relative energy, greedily keep strongest-first while enforcing a
    small min-separation (dedup_dk ~ grid step, NOT the analytic MIN_DK=0.15 floor),
    cap at max_n. Returns (kx, ky, energy) selected, strongest first.

    ⚠**인자 이름 `max_n` 은 일부러 그대로 둔다** (2026-09-05 분리). 이 함수는 풀링만
    하므로 여기서는 애초에 모호하지 않았고, 무엇보다 동결 배치
    `tmp/B22_PRIOR_ABLATION_2026-08-27/gates_b22.py:101` 이 **서명 기본값을 이름으로
    되읽는다**(`_sig_default(pool_and_select, 'max_n')`) — 개명하면 그 게이트가 죽는다.
    기본값은 이제 `POOL_CAP_DEFAULT`(=30) 라 되읽은 **값은 종전과 동일**하다.
    호출자는 `resolve_pool_cap(config)` 로 값을 만들어 넣는다.
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


def merge_atoms(kx, ky, energy, merge_frac):
    """Calibrate proposal cardinality: agglomerate near-duplicate SELECTED atoms (WavePrior
    UNIFIED sparse-parity, additive).

    The diffusion proposal over-proposes near-degenerate atoms on simple seas, inflating the
    component count fed to the tracker. This single-linkage-agglomerates the (already
    pool_and_select'd) atoms within an adaptive Euclidean radius ``r = merge_frac * kp`` in
    (kx,ky), where ``kp`` = |k| of the highest-energy atom. The kp-scaled radius is the repo's
    component-separation convention done adaptively (the absolute MIN_DK floor chains across a
    crowded spectrum; measured in tmp/unified_guard_2026-07-26/explore_signals.py). Each cluster
    collapses to its ENERGY-WEIGHTED centroid with summed energy; result re-sorted strongest
    first. GT-free. ``merge_frac`` None/<=0 returns the input unchanged (bit-identical).
    """
    kx = np.asarray(kx, float); ky = np.asarray(ky, float); en = np.asarray(energy, float)
    n = len(kx)
    if merge_frac is None or merge_frac <= 0 or n <= 1:
        return kx, ky, en
    kp = float(np.sqrt(kx[int(np.argmax(en))] ** 2 + ky[int(np.argmax(en))] ** 2))
    r = merge_frac * kp
    if r <= 0:
        return kx, ky, en
    r2 = r * r
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if (kx[i] - kx[j]) ** 2 + (ky[i] - ky[j]) ** 2 < r2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    mkx, mky, men = [], [], []
    for idxs in groups.values():
        w = en[idxs]
        wn = w / w.sum() if w.sum() > 0 else np.ones(len(idxs)) / len(idxs)
        mkx.append(float(np.sum(wn * kx[idxs])))
        mky.append(float(np.sum(wn * ky[idxs])))
        men.append(float(en[idxs].sum()))
    mkx, mky, men = np.array(mkx), np.array(mky), np.array(men)
    order = np.argsort(-men)
    return mkx[order], mky[order], men[order]


def _beamform_peak_locations(frames, depth, sig_frac=0.1, n_extract=30):
    """Significant beamform-peak (kx,ky) locations + dominant kp, v2 convention. GT-free.

    Runs the v2 coarse dispersion beamformer (identical convention to
    ``initializer.beamform_peak_ratio``) on the collected init-window observations, zeroes the
    DC block, greedily extracts MIN_DK-suppressed peaks, and returns those with power
    >= ``sig_frac`` * P1 as (kx, ky) pairs, plus kp = |k| of the dominant peak. Peak LOCATIONS
    are used ONLY as anchors to keep/discard net atoms (never installed as components).
    """
    from wavesplat.optimizer.initializer import (
        _collect_observations, _beamform_coarse, MIN_DK)
    x, y, z, t = _collect_observations(frames)
    if x is None or len(x) < 10:
        return np.empty((0, 2)), 0.0
    power, kx_axis, ky_axis = _beamform_coarse(
        x, y, z, t, depth, (-4.0, 4.0), (-4.0, 4.0), 81, 81)
    dkx = kx_axis[1] - kx_axis[0]; dky = ky_axis[1] - ky_axis[0]
    cx = int(np.argmin(np.abs(kx_axis))); cy = int(np.argmin(np.abs(ky_axis)))
    power[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.0
    supx = int(MIN_DK / dkx) + 1; supy = int(MIN_DK / dky) + 1
    peaks = []; P = power.copy()
    for _ in range(n_extract):
        idx = int(np.argmax(P.ravel())); iy, ix = np.unravel_index(idx, P.shape)
        pk = float(P[iy, ix])
        if pk < 1e-12:
            break
        peaks.append((float(kx_axis[ix]), float(ky_axis[iy]), pk))
        yl, yh = max(0, iy - supy), min(P.shape[0], iy + supy + 1)
        xl, xh = max(0, ix - supx), min(P.shape[1], ix + supx + 1)
        P[yl:yh, xl:xh] = 0.0
    if not peaks:
        return np.empty((0, 2)), 0.0
    p1 = peaks[0][2]
    locs = np.array([[kx, ky] for (kx, ky, pw) in peaks if pw >= sig_frac * p1], float)
    kp = float(np.hypot(peaks[0][0], peaks[0][1]))
    return locs, kp


def peak_anchor_filter(kx, ky, energy, frames, depth, anchor_frac, sig_frac=0.1):
    """Keep proposal atoms near a SIGNIFICANT beamform peak (WavePrior learned-only sparse-parity,
    additive). Input-conditioning WITHIN the learned pipeline: the beamform power map (already the
    net's conditioning input) is used to filter net atoms; atom LOCATIONS still come from the net.

    Keep each pool_and_select'd atom iff its (kx,ky) distance to the nearest significant beamform
    peak (power >= sig_frac*P1) is <= anchor_frac * kp, where kp = |dominant beamform peak|. On a
    single-peak map this collapses scattered over-proposals onto the swell neighborhood; on a
    multi-peak (crowded) map every true peak has nearby atoms so it is a near-no-op. If NOTHING
    survives (no net atom near any peak) the ORIGINAL atoms are returned UNCHANGED (no-op; never
    injects a beamform-located atom -> guaranteed no-harm vs the shipped learned path). GT-free.
    ``anchor_frac`` None/<=0 returns the input unchanged (bit-identical).
    """
    kx = np.asarray(kx, float); ky = np.asarray(ky, float); en = np.asarray(energy, float)
    if anchor_frac is None or anchor_frac <= 0 or len(kx) == 0:
        return kx, ky, en
    locs, kp = _beamform_peak_locations(frames, depth, sig_frac=sig_frac)
    if len(locs) == 0 or kp <= 0:
        return kx, ky, en
    r = anchor_frac * kp
    dmin = np.min(np.hypot(kx[:, None] - locs[None, :, 0],
                           ky[:, None] - locs[None, :, 1]), axis=1)
    keep = dmin <= r
    if not np.any(keep):
        return kx, ky, en                # no-op: never substitute the beamform peak
    fk, fy, fe = kx[keep], ky[keep], en[keep]
    order = np.argsort(-fe)
    return fk[order], fy[order], fe[order]


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
      learned_ckpt    : checkpoint path (default DEFAULT_CKPT =
                        <repo>/results/direction3/beamform_cond_diffusion_mixed_v1.pt)
      learned_nsamp   : diffusion samples (default 4)
      learned_diffusion_seed : ancestral sampling seed (default: seed, else 0)
      learned_konly   : if True, use v2_konly-style w=0.01 / random phi instead of LS fit
      learned_energy_thresh / learned_dedup_dk : selection knobs
      pool_cap        : 제안 풀링 상한 (default POOL_CAP_DEFAULT=30).
                        레거시 `max_n` 도 읽는다 — `resolve_pool_cap` 참조.
                        ⛔`max_n` 의 **다른 역할**(성장 천장)은 이 함수와 무관하다.
    """
    import torch
    config = config or {}
    max_n = resolve_pool_cap(config)
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
    # WavePrior learned-only sparse-parity (additive): optional peak-anchored filtering of the
    # over-proposed atoms against the v2 beamform peaks. Default (learned_peak_anchor False) is a
    # no-op -> bit-identical to the shipped learned path.
    if config.get('learned_peak_anchor', False):
        kx, ky, en = peak_anchor_filter(
            kx, ky, en, frames, depth,
            anchor_frac=config.get('learned_anchor_frac'),
            sig_frac=float(config.get('learned_anchor_sigfrac', 0.1)))
    # WavePrior UNIFIED sparse-parity (additive): optional kp-scaled agglomerative merge of the
    # over-proposed atoms. Default (learned_merge_frac None) is a no-op -> bit-identical.
    merge_frac = config.get('learned_merge_frac')
    if merge_frac:
        kx, ky, en = merge_atoms(kx, ky, en, float(merge_frac))
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
