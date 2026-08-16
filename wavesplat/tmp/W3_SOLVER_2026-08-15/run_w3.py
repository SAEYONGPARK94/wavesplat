"""W3_SOLVER runner -- implements PREREG.md (self-attested freeze; md5 in FROZEN_md5.txt).

Three faces of one question:
  P0  ORA-K-NOREG vs ORA-K@L3 / ORA-K@L4   -- is the four-null a lambda artifact?
  P1  *-FITPHI vs *                        -- what does the already-computed phase buy?
  UB  ORA-K-CVX                            -- closed-form (a,b) lstsq, zero gradient steps.

NOTHING IN THE PACKAGE IS MODIFIED.  Every factor is applied inside the worker process by
  (a) environment gates read at CALL time (initializer._rx2_gate), or
  (b) module-attribute monkey-patches undone in a finally block.
`RX2_GROW_FITPHI` is NEVER changed as a code default -- it is set per arm in the worker's
environment and fully rewritten to '0' before every job so a previous arm cannot leak.

Usage
    WORKERS=8 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full
    W3_SMOKE=1 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py smoke
    python3 tmp/W3_SOLVER_2026-08-15/run_w3.py gates      # G-1/G-7/G-9, zero estimator runs
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../wavesplat/wavesplat
sys.path.insert(0, ROOT)
os.chdir(ROOT)

BATCH = 'W3_SOLVER_2026-08-15'
OUT = os.path.join(HERE, 'results.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_results.jsonl')
FP_PATH = os.path.join(HERE, 'code_fingerprint.json')

# ── sample (PREREG section 3.1) ────────────────────────────────────────────────
CROWDED = (
    [('data_indep_crowded_2026-08-08',
      f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5')
     for bf in ('BF4', 'BF5') for nc in (15, 30) for s in (20, 21, 22, 23)]
    + [('data_s9_crowded_2026-08-11',
        f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5')
       for bf in ('BF4', 'BF5') for nc in (15, 30) for s in (48, 49, 50, 51)]
)
SPARSE = [('data_indep_sparse_2026-08-08',
           f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5')
          for bf in ('BF4', 'BF5') for nc in (2, 5) for s in (20, 21, 22, 23)]

# ── seeds (PREREG section 3.2); ledger [1592-1818] -> consumes 1592/1593/1594 ──
SEEDS = (1592, 1593, 1594)

# ── shells: SAME edges as wavesplat/run.py:358, stored with the `obs:` prefix ──
EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
BARE_KEYS = ['%d_%s' % (int(a), 'inf' if b == np.inf else int(b))
             for a, b in zip(EDGES[:-1], EDGES[1:])]
SHELL_KEYS = ['obs:' + k for k in BARE_KEYS]

# ── arms (PREREG section 2) ───────────────────────────────────────────────────
LADDER = dict(n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
              max_iters_per_stage=[200, 200, 500])
ORA = dict(init_method='oracle', loss_thresholds=[None], max_iters_per_stage=[900])
# AMENDMENT_2 B-7: SHIP / SHIP-FITPHI / SHIP-DUP are built from ONE dict object so the
# GPU-path duplicate cannot silently drift from the arm it probes.
SHIP_CFG = dict(LADDER, init_method='auto', analytic_ensemble_k=3, lambda_l1=1e-4)

# AMENDMENT_2 B-7: SHIP-DUP episodes -- 4 crowded cells covering both corpora,
# both BF, both component counts.  4 episodes x 3 seeds = 12 jobs.
DUP_EPS = ('BF4_15comp_seed20_uniform_ra_rf_ds', 'BF5_30comp_seed23_uniform_ra_rf_ds',
           'BF4_15comp_seed48_uniform_ra_rf_ds', 'BF5_30comp_seed51_uniform_ra_rf_ds')

ARMS = {
    'ORA-K@L3':     dict(kind='est', cfg=dict(ORA, lambda_l1=1e-3), fitphi=0, oracle=True),
    'ORA-K-NOREG':  dict(kind='est', cfg=dict(ORA, lambda_l1=0.0), fitphi=0, oracle=True),
    'ORA-K@L4':     dict(kind='est', cfg=dict(ORA, lambda_l1=1e-4), fitphi=0, oracle=True),
    'ORA-K@L3-DUP': dict(kind='est', cfg=dict(ORA, lambda_l1=1e-3), fitphi=0, oracle=True,
                         strata=('crowded',)),
    'SHIP':         dict(kind='est', cfg=dict(SHIP_CFG), fitphi=0, oracle=False),
    'SHIP-FITPHI':  dict(kind='est', cfg=dict(SHIP_CFG), fitphi=1, oracle=False),
    'SHIP-DUP':     dict(kind='est', cfg=dict(SHIP_CFG), fitphi=0, oracle=False,
                         strata=('crowded',), eps=DUP_EPS),
    'KONLY':        dict(kind='est', cfg=dict(LADDER, init_method='v2_konly',
                                              lambda_l1=1e-4), fitphi=0, oracle=False),
    'KONLY-FITPHI': dict(kind='est', cfg=dict(LADDER, init_method='v2_konly',
                                              lambda_l1=1e-4), fitphi=1, oracle=False),
    'ORA-K-CVX':    dict(kind='cvx'),
    'ZERO':         dict(kind='zero'),
}
# PREREG section 10: PRIMARY arms first, so a truncated run still answers P-1.
ARM_ORDER = ('ZERO', 'ORA-K-CVX', 'ORA-K@L3', 'ORA-K-NOREG', 'ORA-K@L4',
             'ORA-K@L3-DUP', 'SHIP', 'SHIP-FITPHI', 'SHIP-DUP', 'KONLY', 'KONLY-FITPHI')

BASE_CFG = dict(use_dispersion=True, use_confidence=True, adaptive_n=True,
                tracking_window=3, prune_threshold=0.10, max_n=30,
                lr=0.01, lambda_smooth=0.0001, tracking_steps=10,
                tracking_mode='w_phi', tracking_lr=0.005, device='cpu')

RX2_ENV_KEYS = ('RX2_GROW_SUBTRACT', 'RX2_GROW_RECOMPUTE',
                'RX2_GROW_TIGHT_SUPPRESS', 'RX2_GROW_FITPHI')

# ── DCC required fields (PREREG section 6) ────────────────────────────────────
REQUIRED_COMMON = ('batch', 'arm', 'episode', 'corpus', 'stratum', 'bf', 'ncomp',
                   'ep_seed', 'est_seed', 'n_gt', 'avg_nRMSE', 'zero_nRMSE',
                   'n_frames_scored', 'shell_rmse', 'zero_shell', 'shell_frac',
                   'lambda_l1_intent', 'effective_config', 'code_md5', 'code_files_n',
                   'git_head', 'git_dirty_n', 'wall_s', 'deterministic')
REQUIRED_EST = ('init_method_intent', 'init_method_resolved', 'lambda_l1_effective',
                'rx2_fitphi_env', 'fitphi_atoms', 'randphi_atoms', 'fitphi_site_hits',
                'N_final', 'decorr_mean', 'shrink_mean', 'obs_resid_rms_mean',
                'elapsed_ms_mean')
REQUIRED_CVX = ('cvx_n_obs', 'cvx_cond', 'cvx_rank', 'cvx_window_resid_rms',
                'cvx_truth_window_resid_rms', 'cvx_gtomega_nrmse', 'cvx_omega_source')

HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()
DIRTY = len(subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                           text=True).stdout.strip().splitlines())


# ── sample construction ───────────────────────────────────────────────────────
def episodes():
    out = []
    for stratum, spec in (('crowded', CROWDED), ('sparse', SPARSE)):
        for corpus, name in spec:
            m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', name)
            out.append(dict(rel=os.path.join(corpus, name), corpus=corpus,
                            key=name[len('episode_'):-3], stratum=stratum,
                            bf=m.group(1), ncomp=int(m.group(2)),
                            ep_seed=int(m.group(3))))
    return out


EPISODES = episodes()
SMOKE_KEYS = ('BF4_30comp_seed20_uniform_ra_rf_ds',   # crowded, the hard cell
              'BF4_2comp_seed20_uniform_ra_rf_ds',    # sparse
              # AUDIT_CON W-3: the 15comp crowded cell was never smoked.  There
              # n_gt=15 < max_n=30, so the oracle arms CAN enter the beamform growth
              # branch that 30comp (N == max_n) can never reach -- i.e. the only cell
              # where G-4's "OFF arms show fitphi_atoms == 0" could still break.
              'BF4_15comp_seed20_uniform_ra_rf_ds')


# ── code fingerprint (G-7); fixed file set, import-order independent ──────────
def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def code_fingerprint():
    files = {}
    for pat in ('wavesplat/**/*.py', 'wavesim/**/*.py'):
        for f in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
            files[os.path.relpath(f, ROOT)] = _md5(f)
    for extra in ('configs/estimation.yaml', 'configs/default.yaml',
                  'configs/server.yaml'):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            files[extra] = _md5(p)
    for own in ('run_w3.py', 'PREREG.md'):          # the treatment definition
        p = os.path.join(HERE, own)
        if os.path.exists(p):
            files[os.path.relpath(p, ROOT)] = _md5(p)
    return hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def preload():
    """Import every dependency BEFORE the run so code_md5 is one digest."""
    import wavesplat.run                            # noqa: F401
    import wavesplat.data_loader                    # noqa: F401
    import wavesplat.evaluation.metrics             # noqa: F401
    import wavesplat.optimizer.sequential_update    # noqa: F401
    import wavesplat.optimizer.coarse_to_fine       # noqa: F401
    import wavesplat.optimizer.initializer          # noqa: F401
    import wavesplat.model.spectral_gaussian        # noqa: F401


# ── FITPHI atom-level counter (PREREG section 2.2b) ───────────────────────────
def install_probe():
    """Count atoms actually seeded with the FITTED phase, not gate reads.

    `_linear_projection` returns the phase the shipped growth path throws away.  We record
    every phase it returns (as float32, the dtype `add_gaussians` finally stores) and then
    inspect each newly added atom's `init_phi`: a value present in that set came from the
    fit, anything else came from `torch.rand(...)*2pi`.  The collision probability of a
    U(0,2pi) float32 draw landing exactly on a recorded fitted phase is ~1e-7 per atom.

    Both FITPHI sites are covered:
      (1) coarse_to_fine._beamform_residuals -- module-level import, patched on that module
      (2) sequential_update._adaptive_prune_grow -- call-time import, patched on initializer
    """
    import torch
    import wavesplat.optimizer.initializer as I
    import wavesplat.optimizer.coarse_to_fine as C
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel as M

    P = {'lp_calls': 0, 'fitphi_atoms': 0, 'randphi_atoms': 0, 'add_calls': 0,
         'site_hits': {}, '_phi': set()}

    o_lp_I, o_lp_C = I._linear_projection, getattr(C, '_linear_projection', None)

    def lp(*a, **k):
        amp, ph = o_lp_I(*a, **k)
        P['lp_calls'] += 1
        try:
            P['_phi'].add(float(np.float32(ph)))
        except Exception:
            pass
        return amp, ph

    I._linear_projection = lp
    if o_lp_C is not None:
        C._linear_projection = lp

    o_gate = I._rx2_gate

    def gate(name):
        v = o_gate(name)
        if name == 'RX2_GROW_FITPHI':
            fr = sys._getframe(1)
            key = '%s:%d|%s' % (os.path.basename(fr.f_code.co_filename),
                                fr.f_lineno, 'T' if v else 'F')
            P['site_hits'][key] = P['site_hits'].get(key, 0) + 1
        return v

    I._rx2_gate = gate

    o_add = M.add_gaussians

    def add(self, n_new, init_kx=None, init_ky=None, init_w=None, init_phi=None):
        if init_phi is not None:
            P['add_calls'] += 1
            for v in np.asarray(init_phi.detach().cpu().numpy(),
                                dtype=np.float32).ravel().tolist():
                if float(v) in P['_phi']:
                    P['fitphi_atoms'] += 1
                else:
                    P['randphi_atoms'] += 1
        return o_add(self, n_new, init_kx=init_kx, init_ky=init_ky,
                     init_w=init_w, init_phi=init_phi)

    M.add_gaussians = add

    def undo():
        I._linear_projection = o_lp_I
        if o_lp_C is not None:
            C._linear_projection = o_lp_C
        I._rx2_gate = o_gate
        M.add_gaussians = o_add

    return P, undo


# ── effective_config readback FROM THE OBJECT (PREREG section 2.2a) ───────────
_UPDATER_BOX = {}


def install_updater_capture():
    """`run_estimation` never returns the updater, so capture it at construction."""
    import wavesplat.run as R
    orig = R.make_updater

    def cap(config):
        u = orig(config)
        _UPDATER_BOX['u'] = u
        return u

    R.make_updater = cap

    def undo():
        R.make_updater = orig
        _UPDATER_BOX.pop('u', None)

    return undo


_RB_KEYS = ('lambda_l1', 'lambda_smooth', 'lr', 'tracking_lr', 'tracking_mode',
            'tracking_steps', 'init_method', 'adaptive_n', 'prune_threshold',
            'grow_residual_threshold', 'max_n', 'stages', 'loss_thresholds',
            'max_iters_per_stage', 'use_dispersion', 'use_confidence', 'depth',
            'seed', 'device', 'tracking_window', 'max_steepness')


def _readback(u):
    """Read the LIVE object, never the intent dict (S13's discipline-4 failure)."""
    if u is None:
        return {'_absent': True}
    members = getattr(u, 'members', None)
    core = members[0] if members else u
    out = {'updater_class': type(u).__name__, 'core_class': type(core).__name__,
           'n_members': len(members) if members else 1}
    for k in _RB_KEYS:
        v = getattr(core, k, '<absent>')
        if isinstance(v, (list, tuple)):
            v = [float(x) if isinstance(x, (int, float)) else x for x in v]
        elif isinstance(v, np.generic):
            v = v.item()
        elif not isinstance(v, (int, float, str, bool, type(None))):
            v = str(v)
        out[k] = v
    out['auto_selected'] = getattr(core, '_auto_selected', None)
    out['auto_real_gate'] = getattr(core, '_auto_real_gate', None)
    # every member's lambda, so an ensemble cannot hide a mismatched member
    if members:
        out['member_lambda_l1'] = [float(getattr(m, 'lambda_l1', float('nan')))
                                   for m in members]
    return out


# ── shared scoring helpers ────────────────────────────────────────────────────
def _shell_masks(Xg, Yg, ox, oy):
    md = np.sqrt((Xg[None, :, :] - ox[:, None, None]) ** 2
                 + (Yg[None, :, :] - oy[:, None, None]) ** 2).min(axis=0)
    return [((md >= a) & (md < b)) for a, b in zip(EDGES[:-1], EDGES[1:])]


def _mean_shells(d):
    out = {}
    for k, v in d.items():
        a = np.asarray(v, dtype=float)
        out[k] = float(np.nanmean(a)) if a.size and np.isfinite(a).any() else float('nan')
    return out


def _k_drift(sp, gtc, oracle):
    """AMENDMENT_2 B-6(a): does the oracle arm still HOLD the true k at the end?

    `ORA-K*` fixes k only at INITIALISATION -- `init_k_lr_scale` defaults to 1.0
    (`sequential_update.py:262`), so k is a free parameter for all 900 steps, while
    `ORA-K-CVX` freezes k at the truth.  That is an undeclared oracle factor in the
    CVX comparison unless the drift is measured, so measure it (runtime cost 0: the
    final spectrum is already in `run_estimation`'s return, `run.py:480`).

    Ordering: `initialize_from_frames_oracle_k` installs GT (kx,ky) in DESCENDING
    AMPLITUDE order whenever n_gaussians <= n_gt (`initializer.py:625-628`).  The
    docstring above that branch says "order as stored", which is NOT what the code
    does -- the stored components are not amplitude-sorted (checked on the corpus).
    """
    out = dict(k_drift_rms=None, k_drift_max=None, k_drift_rel_median=None,
               k_drift_valid=False, k_drift_note='')
    if not oracle or not sp:
        out['k_drift_note'] = 'not-an-oracle-arm' if not oracle else 'no-spectrum'
        return out
    try:
        kx = np.asarray(sp['kx'], dtype=np.float64).ravel()
        ky = np.asarray(sp['ky'], dtype=np.float64).ravel()
        if len(kx) != len(gtc):
            out['k_drift_note'] = 'N_final(%d)!=n_gt(%d): prune/grow broke the 1:1 map' \
                                  % (len(kx), len(gtc))
            return out
        order = np.argsort(gtc[:, 0])[::-1][:len(kx)]      # initializer.py:625
        kx0, ky0 = gtc[order, 2], gtc[order, 3]
        d = np.sqrt((kx - kx0) ** 2 + (ky - ky0) ** 2)
        kmag = np.sqrt(kx0 ** 2 + ky0 ** 2)
        out.update(k_drift_rms=float(np.sqrt(np.mean(d ** 2))),
                   k_drift_max=float(np.max(d)),
                   k_drift_rel_median=float(np.median(d / np.maximum(kmag, 1e-12))),
                   k_drift_valid=True, k_drift_note='amplitude-desc order')
    except Exception as e:                                 # never kill a job for a diagnostic
        out['k_drift_note'] = 'error: %s' % repr(e)[:80]
    return out


def _episode_static(rel):
    """Everything a row needs that does not depend on the arm.  Loads the episode once."""
    from wavesplat.data_loader import EpisodeDataLoader
    loader = EpisodeDataLoader(os.path.join(ROOT, rel))
    gt = np.asarray(loader.wave_components, dtype=np.float64)
    return loader, gt


def _zero_and_shape(loader):
    """ZERO field metrics + zero shells over the SAME tracking frames every arm is scored on."""
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    frames = loader.get_tracking_frames(tp_factor=2.0)
    zr, sh_z, fr = [], {k: [] for k in SHELL_KEYS}, {k: [] for k in SHELL_KEYS}
    for f in frames:
        gt = f.gt_field.cpu().numpy()
        zr.append(float(np.sqrt(np.mean(gt ** 2)) / loader.Hs_true))
        x = f.x_obs.cpu().numpy()
        y = f.y_obs.cpu().numpy()
        if len(x):
            for k, msk in zip(SHELL_KEYS, _shell_masks(Xg, Yg, x, y)):
                fr[k].append(float(np.mean(msk)))
                sh_z[k].append(float(np.sqrt(np.mean(gt[msk] ** 2)))
                               if np.any(msk) else np.nan)
        else:
            for k in SHELL_KEYS:
                fr[k].append(np.nan)
                sh_z[k].append(np.nan)
    return (float(np.mean(zr)), _mean_shells(sh_z), _mean_shells(fr), len(frames),
            frames, gx, gy, Xg, Yg)


# ── arm: ORA-K-CVX (PREREG section 2.1) -- zero gradient steps ────────────────
def run_cvx(ep):
    """Closed-form (a, b) least squares on the SAME observations the estimator sees."""
    from wavesplat.evaluation.metrics import compute_all_metrics
    loader, gtc = _episode_static(ep['rel'])
    zero_n, zero_sh, frac, n_fr, frames, gx, gy, Xg, Yg = _zero_and_shape(loader)

    g = 9.81
    kx = gtc[:, 2]
    ky = gtc[:, 3]
    n = len(kx)
    kmag = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om_disp = np.sqrt(g * kmag * np.tanh(kmag * loader.depth))
    om_gt = gtc[:, 1]

    # ---- fit window == the estimator's Phase-1 window.  NOT the GT grid. -------
    win = loader.get_init_window(tp_factor=2.0)
    X = np.concatenate([f.x_obs.cpu().numpy() for f in win])
    Y = np.concatenate([f.y_obs.cpu().numpy() for f in win])
    Z = np.concatenate([f.z_obs.cpu().numpy() for f in win]).astype(np.float64)
    T = np.concatenate([np.full(len(f.x_obs), f.t) for f in win])

    def solve(om):
        th = (kx[None, :] * X[:, None] + ky[None, :] * Y[:, None]
              - om[None, :] * T[:, None])
        Phi = np.concatenate([np.cos(th), np.sin(th)], axis=1)
        c, _res, rank, sv = np.linalg.lstsq(Phi, Z, rcond=None)
        a, b = c[:n], c[n:]
        resid = float(np.sqrt(np.mean((Phi @ c - Z) ** 2)))
        cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float('inf')
        return a, b, int(rank), cond, resid

    a, b, rank, cond, resid = solve(om_disp)
    a2, b2, _, _, _ = solve(om_gt)

    # truth's own residual on the same window (the representation floor)
    th_t = (kx[None, :] * X[:, None] + ky[None, :] * Y[:, None]
            - om_gt[None, :] * T[:, None] + gtc[:, 4][None, :])
    z_truth = (gtc[:, 0][None, :] * np.cos(th_t)).sum(axis=1)
    truth_resid = float(np.sqrt(np.mean((z_truth - Z) ** 2)))

    def field(a_, b_, om, gx_, gy_, t):
        Xg_, Yg_ = np.meshgrid(gx_, gy_)
        th = (kx[None, :] * Xg_.ravel()[:, None] + ky[None, :] * Yg_.ravel()[:, None]
              - om[None, :] * t)
        return ((a_[None, :] * np.cos(th) + b_[None, :] * np.sin(th))
                .sum(axis=1).reshape(Xg_.shape))

    nr, nr2 = [], []
    sh_a = {k: [] for k in SHELL_KEYS}
    for f in frames:
        pf = field(a, b, om_disp, gx, gy, f.t)
        gt = f.gt_field.cpu().numpy()
        x = f.x_obs.cpu().numpy()
        y = f.y_obs.cpu().numpy()
        m = compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None,
                                obs_y=y if len(y) else None)
        nr.append(float(m['nRMSE']))
        pf2 = field(a2, b2, om_gt, gx, gy, f.t)
        nr2.append(float(np.sqrt(np.mean((pf2 - gt) ** 2)) / loader.Hs_true))
        if len(x):
            for k, msk in zip(SHELL_KEYS, _shell_masks(Xg, Yg, x, y)):
                sh_a[k].append(float(np.sqrt(np.mean((pf[msk] - gt[msk]) ** 2)))
                               if np.any(msk) else np.nan)
        else:
            for k in SHELL_KEYS:
                sh_a[k].append(np.nan)

    w = np.sqrt(a ** 2 + b ** 2)
    return dict(avg_nRMSE=float(np.mean(nr)), zero_nRMSE=zero_n,
                n_frames_scored=n_fr, shell_rmse=_mean_shells(sh_a),
                zero_shell=zero_sh, shell_frac=frac, n_gt=n,
                cvx_n_obs=int(len(Z)), cvx_cond=cond, cvx_rank=rank,
                cvx_window_resid_rms=resid,
                cvx_truth_window_resid_rms=truth_resid,
                cvx_gtomega_nrmse=float(np.mean(nr2)),
                cvx_omega_source='dispersion',
                cvx_w_median=float(np.median(w)),
                effective_config={'solver': 'numpy.linalg.lstsq', 'grad_steps': 0,
                                  'lambda_l1': 0.0, 'weighting': 'none',
                                  'tracking': 'none', 'n_atoms': int(n),
                                  'omega': 'dispersion', 'fit_window': 'init_window',
                                  'rcond': None},
                Hs=float(loader.Hs_true), Tp=float(loader.Tp_true),
                depth=float(loader.depth))


# ── arm: ZERO ─────────────────────────────────────────────────────────────────
def run_zero(ep):
    loader, gtc = _episode_static(ep['rel'])
    zero_n, zero_sh, frac, n_fr, *_ = _zero_and_shape(loader)
    return dict(avg_nRMSE=zero_n, zero_nRMSE=zero_n, n_frames_scored=n_fr,
                shell_rmse=dict(zero_sh), zero_shell=zero_sh, shell_frac=frac,
                n_gt=int(len(gtc)),
                effective_config={'solver': 'none', 'field': 'identically zero'},
                Hs=float(loader.Hs_true), Tp=float(loader.Tp_true),
                depth=float(loader.depth))


# ── arm: estimator ────────────────────────────────────────────────────────────
def run_est(ep, arm, seed):
    from wavesplat.run import run_estimation
    import wavesplat.optimizer.initializer as I

    spec = ARMS[arm]
    loader, gtc = _episode_static(ep['rel'])
    zero_n, zero_sh, frac, n_fr, *_ = _zero_and_shape(loader)

    cfg = dict(BASE_CFG)
    cfg.update(spec['cfg'])
    if spec['oracle']:
        cfg['n_gaussians'] = [int(len(gtc))]
    cfg['seed'] = seed
    assert 'init_method' in cfg and 'lambda_l1' in cfg   # PREREG: never implicit

    # env gates: FULLY rewritten so a previous job in this worker cannot leak
    for k in RX2_ENV_KEYS:
        os.environ[k] = '0'
    os.environ['RX2_GROW_FITPHI'] = str(spec['fitphi'])
    for k in ('S4_ORACLE_WPHI', 'S5_WINDOW', 'S2_ALPHA',
              'S8_L1_TRACK_ONLY', 'S8_L1_PHASE1_ONLY'):
        os.environ[k] = '0'

    P, undo_probe = install_probe()
    undo_cap = install_updater_capture()
    env_seen = {}
    try:
        I.ORACLE_COMPONENTS = gtc if spec['oracle'] else None
        try:
            r = run_estimation(os.path.join(ROOT, ep['rel']), cfg, verbose=False)
        finally:
            I.ORACLE_COMPONENTS = None
            # read the gates BEFORE the finally below rewrites them to '0'
            env_seen = {k: os.environ.get(k, '<unset>') for k in RX2_ENV_KEYS}
        eff = _readback(_UPDATER_BOX.get('u'))
    finally:
        undo_cap()
        undo_probe()
        for k in RX2_ENV_KEYS:
            os.environ[k] = '0'

    fr_ = r['frame_results']
    nr = np.array([f['nRMSE'] for f in fr_ if 'nRMSE' in f], float)
    rho = np.array([f['Pearson_rho'] for f in fr_ if 'Pearson_rho' in f], float)
    vr = np.array([f['Variance_Ratio'] for f in fr_ if 'Variance_Ratio' in f], float)
    with np.errstate(invalid='ignore'):
        decorr = 1.0 - rho ** 2
        shrink = (np.sqrt(np.clip(vr, 0, None)) - rho) ** 2

    sh = {k: [] for k in SHELL_KEYS}
    for f in fr_:
        d = f.get('shell_rmse') or {}
        for bare, k in zip(BARE_KEYS, SHELL_KEYS):
            sh[k].append(d.get(bare, np.nan))

    def _m(key):
        v = [f[key] for f in fr_ if key in f and np.isfinite(f.get(key, np.nan))]
        return float(np.mean(v)) if v else float('nan')

    return dict(avg_nRMSE=float(np.nanmean(nr)), zero_nRMSE=zero_n,
                **_k_drift(r.get('spectrum_params'), gtc, spec['oracle']),
                n_frames_scored=len(fr_), shell_rmse=_mean_shells(sh),
                zero_shell=zero_sh, shell_frac=frac, n_gt=int(len(gtc)),
                init_method_intent=cfg['init_method'],
                init_method_resolved=eff.get('init_method'),
                lambda_l1_effective=eff.get('lambda_l1'),
                rx2_fitphi_env=env_seen.get('RX2_GROW_FITPHI', '<unset>'),
                rx2_env_seen=env_seen,
                fitphi_atoms=int(P['fitphi_atoms']),
                randphi_atoms=int(P['randphi_atoms']),
                fitphi_site_hits=dict(P['site_hits']),
                lp_calls=int(P['lp_calls']), add_calls=int(P['add_calls']),
                N_final=int(r.get('n_gaussians', -1)),
                decorr_mean=float(np.nanmean(decorr)) if rho.size else float('nan'),
                shrink_mean=float(np.nanmean(shrink)) if rho.size else float('nan'),
                obs_resid_rms_mean=_m('obs_resid_rms'),
                elapsed_ms_mean=_m('elapsed_ms'),
                init_nrmse=float(r.get('init_nrmse', float('nan'))),
                effective_config=eff,
                Hs=float(loader.Hs_true), Tp=float(loader.Tp_true),
                depth=float(loader.depth))


# ── job plumbing ──────────────────────────────────────────────────────────────
def build_jobs(arms=None, keys=None):
    jobs = []
    for arm in (arms or ARM_ORDER):
        spec = ARMS[arm]
        strata = spec.get('strata', ('crowded', 'sparse'))
        seeds = SEEDS if spec['kind'] == 'est' else (None,)
        for ep in EPISODES:
            if ep['stratum'] not in strata:
                continue
            if spec.get('eps') and ep['key'] not in spec['eps']:   # AMENDMENT_2 B-7
                continue
            if keys and ep['key'] not in keys:
                continue
            for s in seeds:
                jobs.append(dict(job_id='%s|%s|%s' % (arm, ep['key'], s),
                                 arm=arm, ep=ep, seed=s))
    return jobs


def run_one(job):
    sys.path.insert(0, ROOT)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    arm, ep, seed = job['arm'], job['ep'], job['seed']
    kind = ARMS[arm]['kind']
    t0 = time.time()
    base = dict(batch=BATCH, job_id=job['job_id'], arm=arm, episode=ep['key'],
                corpus=ep['corpus'], stratum=ep['stratum'], bf=ep['bf'],
                ncomp=ep['ncomp'], ep_seed=ep['ep_seed'], est_seed=seed,
                lambda_l1_intent=(ARMS[arm]['cfg']['lambda_l1'] if kind == 'est'
                                  else (0.0 if kind == 'cvx' else None)),
                deterministic=(kind != 'est'),
                git_head=HEAD, git_dirty_n=DIRTY)
    try:
        preload()
        dg, dgf = code_fingerprint()
        if kind == 'est':
            body = run_est(ep, arm, seed)
        elif kind == 'cvx':
            body = run_cvx(ep)
        else:
            body = run_zero(ep)
        return dict(base, **body, code_md5=dg, code_files_n=len(dgf),
                    wall_s=time.time() - t0)
    except Exception as e:
        import traceback
        dg, dgf = code_fingerprint()
        return dict(base, error=repr(e)[:300], tb=traceback.format_exc()[-2000:],
                    code_md5=dg, code_files_n=len(dgf), wall_s=time.time() - t0)


# ── schema check (DCC, used by the smoke and by analyze_w3) ───────────────────
def missing_fields(row):
    kind = ARMS.get(row.get('arm'), {}).get('kind')
    miss = [k for k in REQUIRED_COMMON if k not in row]
    if kind == 'est':
        miss += [k for k in REQUIRED_EST if k not in row]
    if kind == 'cvx':
        miss += [k for k in REQUIRED_CVX if k not in row]
    for fld in ('shell_rmse', 'zero_shell', 'shell_frac'):
        d = row.get(fld)
        if not isinstance(d, dict):
            miss.append(fld + '{}')
        else:
            miss += ['%s[%s]' % (fld, k) for k in SHELL_KEYS if k not in d]
    return miss


# ── gates that need no estimator run (G-1, G-7, G-9) ──────────────────────────
def gates():
    ok = True
    miss = [e['rel'] for e in EPISODES if not os.path.exists(os.path.join(ROOT, e['rel']))]
    n_c = sum(1 for e in EPISODES if e['stratum'] == 'crowded')
    n_s = len(EPISODES) - n_c
    g1 = (not miss) and n_c == 32 and n_s == 16
    print('G-1 sample     : crowded=%d sparse=%d missing=%d -> %s'
          % (n_c, n_s, len(miss), 'PASS' if g1 else 'FAIL'))
    if miss:
        print('    missing:', miss[:5])
    ok &= g1

    dg1, files = code_fingerprint()
    dg2, _ = code_fingerprint()
    g7 = dg1 == dg2
    print('G-7 fingerprint: digest=%s files=%d stable=%s (git %s dirty %d) -> %s'
          % (dg1, len(files), dg1 == dg2, HEAD[:8], DIRTY, 'PASS' if g7 else 'FAIL'))
    with open(FP_PATH, 'w') as fh:
        json.dump({'digest': dg1, 'git_head': HEAD, 'git_dirty_n': DIRTY,
                   'files': files}, fh, indent=1)
    ok &= g7

    used = set()
    for p in glob.glob(os.path.join(ROOT, 'tmp', '**', '*.jsonl'), recursive=True):
        if p.startswith(HERE):
            continue
        try:
            for line in open(p, errors='ignore'):
                for v in re.findall(r'"(?:seed|est_seed|filter_seed)"\s*:\s*(\d+)', line):
                    if 1500 <= int(v) <= 1900:
                        used.add(int(v))
        except OSError:
            pass
    clash = sorted(set(SEEDS) & used)
    g9 = not clash
    print('G-9 seed ledger: consume=%s ledger_max_used=%s clash=%s -> %s'
          % (list(SEEDS), max(used) if used else None, clash, 'PASS' if g9 else 'FAIL'))
    ok &= g9

    print('\nplan: %d arms x episodes x seeds = %d jobs'
          % (len(ARM_ORDER), len(build_jobs())))
    for a in ARM_ORDER:
        print('   %-14s %s' % (a, len(build_jobs(arms=[a]))))
    print('\nGATES(no-run) %s' % ('PASS' if ok else 'FAIL'))
    return ok


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import multiprocessing as mp
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    if mode == 'gates':
        sys.exit(0 if gates() else 1)

    smoke = mode == 'smoke'
    out = SMOKE_OUT if smoke else OUT
    arms = (sys.argv[2].split(',') if len(sys.argv) > 2 else None)
    jobs = build_jobs(arms=arms, keys=(SMOKE_KEYS if smoke else None))

    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    n_workers = int(os.environ.get('WORKERS', '8'))
    dg, files = code_fingerprint()
    print('%s | mode=%s | total %d | done %d | todo %d | workers %d | seeds %s'
          % (BATCH, mode, len(jobs), len(done), len(todo), n_workers, list(SEEDS)),
          flush=True)
    print('code_md5=%s files=%d HEAD=%s dirty=%d' % (dg, len(files), HEAD[:8], DIRTY),
          flush=True)
    if not todo:
        print('nothing to do')
        return

    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers) as pool, open(out, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            el = time.time() - t0
            tag = rec.get('avg_nRMSE', rec.get('error', '?'))
            tag = ('%.4f' % tag) if isinstance(tag, float) else str(tag)[:80]
            print('[%d/%d] %-52s %s  fit=%s  %.1fs  ETA %.0fmin'
                  % (i, len(todo), rec['job_id'], tag,
                     rec.get('fitphi_atoms', '-'), rec.get('wall_s', float('nan')),
                     el / i * (len(todo) - i) / 60), flush=True)
    print('DONE %.1f min' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
