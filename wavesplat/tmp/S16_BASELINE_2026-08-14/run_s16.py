#!/usr/bin/env python3
"""S16 runner -- implements PREREG.md (frozen; md5 in FROZEN_md5.txt) exactly.

    python3 tmp/S16_BASELINE_2026-08-14/run_s16.py gates    # G-1..G-4, G-7 iso, G-8, G-10 (run-free)
    python3 tmp/S16_BASELINE_2026-08-14/run_s16.py isotropy # AMENDMENT_1 B1 probe (run-free)
    python3 tmp/S16_BASELINE_2026-08-14/run_s16.py smoke    # 2 episodes x 5 arms + G-5/G-6/G-7/G-9
    python3 tmp/S16_BASELINE_2026-08-14/run_s16.py run                       # all 5 arms, 640 runs
    python3 tmp/S16_BASELINE_2026-08-14/run_s16.py run SPEC25 SPEC50 SPEC100 # Phase 1
    python3 tmp/S16_BASELINE_2026-08-14/run_s16.py run WHITER50 WHITE50      # Phase 2

One row per (episode, arm) appended to results.jsonl; **resumable** (non-error rows are
skipped).  Smoke output goes to smoke_results.jsonl and never mixes with the verdict data.

Every row carries (PREREG section 2.1): effective_config READ BACK FROM THE FILTER OBJECT,
`obs:`-prefixed shells for the arm AND for the zero field AND the area fractions,
code_md5 over every imported repository file, git_head/git_dirty_n, and the GT-free
Hs estimate so the size of the oracle in PREREG section 4.2 stays auditable.
"""
import collections
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
sys.path.insert(0, HERE)
os.chdir(ROOT)

OUT = os.path.join(HERE, 'results.jsonl')
# AMENDMENT_1: a tag keeps the post-fix re-verification smoke in its own files so the
# pre-fix (frozen, superseded) smoke record is never overwritten.
SMOKE_TAG = os.environ.get('S16_SMOKE_TAG', '')
SMOKE_OUT = os.path.join(HERE, 'smoke%s_results.jsonl' % SMOKE_TAG)
SMOKE_REPORT = os.path.join(HERE, 'smoke%s_report.json' % SMOKE_TAG)
FP_PATH = os.path.join(HERE, 'code_fingerprint.json')

S13 = os.path.join(ROOT, 'tmp', 'S13_oos_2026-08-12', 'results.jsonl')
S13B = os.path.join(ROOT, 'tmp', 'S13b_seed_robust_2026-08-13', 'results_s13b.jsonl')

# ── sample (PREREG section 3): identical to run_s13.py:32-33 ────────────────────
SPARSE_DIR = ['data_s9_sparse_2026-08-11']
CROWDED_DIR = ['data_s8_crowded_2026-08-10', 'data_s9_crowded_2026-08-11']
SMOKE_KEYS = ('BF4_15comp_seed32_uniform_ra_rf_ds', 'BF4_2comp_seed48_uniform_ra_rf_ds')

# ── shells: SAME edges as wavesplat/run.py:358, stored with the `obs:` prefix ───
EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
SHELL_KEYS = ['obs:%d_%s' % (int(a), 'inf' if b == np.inf else int(b))
              for a, b in zip(EDGES[:-1], EDGES[1:])]

# ── arms (PREREG section 2) ────────────────────────────────────────────────────
N_ENSEMBLE = 600            # identical to S13 / HOSGATE LETKF_PLUS
FILTER_SEED = 42            # identical to S13; G-6 may escalate to median-of-3
# PREREG G-6 escalation, FIRED: re-measured after the AMENDMENT_1 B1 isotropy fix the
# filter-seed relative sd is 8.216% / 13.549% (was 0.986% / 0.586%), i.e. above the frozen
# 2% threshold -> "the whole batch goes to a median-of-3 baseline (3x the cost)".
BASELINE_SEEDS = (42, 43, 44)
ARMS = {
    'SPEC25':   dict(kind='spec',  background='jonswap', loc=25.0),
    'SPEC50':   dict(kind='spec',  background='jonswap', loc=50.0),
    'SPEC100':  dict(kind='spec',  background='jonswap', loc=100.0),
    'WHITER50': dict(kind='spec',  background='white',   loc=50.0),
    'WHITE50':  dict(kind='stock', background='white',   loc=50.0),
}
ARM_ORDER = ('SPEC25', 'SPEC50', 'SPEC100', 'WHITER50', 'WHITE50')

# PREREG section 2.1 -- the gate checks these EXIST; analyze_s16.py refuses a verdict
# if any row is missing one.  S13/HOSGATE died twice on unrecorded baseline shells.
REQUIRED_LETKF = ('episode', 'arm', 'stratum', 'bf', 'ncomp', 'seed', 'wall_s',
                  'avg_nRMSE', 'zero_nRMSE', 'n_match', 'n_all', 'ms_median',
                  'Hs', 'Tp', 'depth', 'shell_rmse', 'zero_shell', 'shell_frac',
                  'effective_config', 'loc_radius', 'n_ensemble', 'background',
                  'filter_class', 'filter_seed', 'hs_obs4std',
                  'git_head', 'git_dirty_n', 'code_md5')
REQUIRED_SHIP = ('episode', 'arm', 'stratum', 'avg_nRMSE', 'shell_rmse',
                 'effective_config')

HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()
DIRTY = len(subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                           text=True).stdout.strip().splitlines())


# ── sample ─────────────────────────────────────────────────────────────────────
def episodes():
    out = []
    for stratum, dirs in (('sparse', SPARSE_DIR), ('crowded', CROWDED_DIR)):
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(d, '*.h5'))):
                b = os.path.basename(p)
                m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', b)
                out.append(dict(path=p, key=b[len('episode_'):-3], stratum=stratum,
                                bf=m.group(1), ncomp=int(m.group(2)),
                                seed=int(m.group(3)), suffix=m.group(4)))
    return out


# ── code fingerprint (G-8) ─────────────────────────────────────────────────────
def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def code_fingerprint():
    """AMENDMENT_1 (neutral audit A9): the old version walked `sys.modules`, so the
    digest depended on WHICH modules happened to be imported -- the smoke wrote TWO
    distinct `code_md5` values for ten rows while PREREG G-8 requires a SINGLE one.
    The file set is now FIXED (every repository .py + the three configs + the two batch
    scripts that define the treatment), so the digest is import-order independent and
    strictly stronger: it also covers repository files this process never imported."""
    files = {}
    for pat in ('wavesplat/**/*.py', 'wavesim/**/*.py'):
        for f in glob.glob(os.path.join(ROOT, pat), recursive=True):
            files[os.path.relpath(f, ROOT)] = _md5(f)
    for extra in ('configs/estimation.yaml', 'configs/default.yaml', 'configs/server.yaml'):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            files[extra] = _md5(p)
    for own in ('letkf_spec.py', 'run_s16.py'):           # the treatment definition
        p = os.path.join(HERE, own)
        if os.path.exists(p):
            files[os.path.relpath(p, ROOT)] = _md5(p)
    return hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def preload():
    """Import every arm's dependencies BEFORE the run so code_md5 is one digest."""
    import wavesplat.run                            # noqa: F401
    import wavesplat.data_loader                    # noqa: F401
    import wavesplat.baselines.letkf_grid           # noqa: F401
    import wavesplat.evaluation.metrics             # noqa: F401
    import wavesplat.optimizer.sequential_update    # noqa: F401
    import scipy.interpolate                        # noqa: F401
    import letkf_spec                               # noqa: F401


# ── SHIP import: S13 (seed 42) + S13b (seeds 43, 44) -> median-of-3 ────────────
def _load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def import_ship():
    """{episode: {seed: row}} for the shipping arm.  ZERO new runs (PREREG section 2)."""
    out = {}
    for r in _load_jsonl(S13):
        if r.get('arm') == 'SHIP' and 'error' not in r:
            out.setdefault(r['episode'], {})[42] = r
    for r in _load_jsonl(S13B):
        # AMENDMENT_1 (CON W4): S13b rows are labelled `SHIP@43` / `SHIP@44`; without
        # this arm filter any baseline row later appended to that file would enter the
        # SHIP column silently.  Currently a no-op (that file holds SHIP rows only).
        if 'error' in r or not str(r.get('arm', '')).startswith('SHIP'):
            continue
        s = r.get('est_seed', r.get('effective_config', {}).get('seed'))
        if s in (43, 44):
            out.setdefault(r['episode'], {})[int(s)] = r
    return out


def ship_med3(rows_by_seed):
    """median-of-3 over the estimator seed, for the scalar AND for every shell."""
    seeds = sorted(rows_by_seed)
    v = [rows_by_seed[s]['avg_nRMSE'] for s in seeds]
    sh = {}
    for k in SHELL_KEYS:
        bare = k.split(':', 1)[1]
        vals = [(rows_by_seed[s].get('shell_rmse') or {}).get(bare) for s in seeds]
        vals = [x for x in vals if x is not None and np.isfinite(x)]
        sh[k] = float(np.median(vals)) if vals else float('nan')
    return dict(avg_nRMSE=float(np.median(v)), shell_rmse=sh, seeds=seeds,
                per_seed={str(s): rows_by_seed[s]['avg_nRMSE'] for s in seeds})


# ── scoring ────────────────────────────────────────────────────────────────────
def _shell_masks(Xg, Yg, ox, oy):
    md = np.sqrt((Xg[None, :, :] - ox[:, None, None]) ** 2
                 + (Yg[None, :, :] - oy[:, None, None]) ** 2).min(axis=0)
    return [((md >= a) & (md < b)) for a, b in zip(EDGES[:-1], EDGES[1:])]


def run_letkf(path, arm, filter_seed=FILTER_SEED):
    """One LETKF episode.  Mirrors S13 run_s13.py:run_letkf exactly for WHITE50
    (so it reproduces the stored column bit-for-bit) and swaps only the filter class
    for the repaired arms.  Records `obs:` shells for the arm AND the zero field."""
    from scipy.interpolate import RegularGridInterpolator
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.baselines.letkf_grid import LETKFGrid
    from wavesplat.evaluation.metrics import compute_all_metrics
    import letkf_spec

    spec = ARMS[arm]
    np.random.seed(filter_seed)                     # identical to S13's np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40)
    egy = np.linspace(gy[0], gy[-1], 30)

    if spec['kind'] == 'stock':
        f = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=N_ENSEMBLE,
                      mult_inflation=1.0, add_inflation=0.005, loc_radius=spec['loc'])
    else:
        f = letkf_spec.LETKFGridSpec(
            egx, egy, background=spec['background'],
            Hs_bg=float(loader.Hs_true), Tp_bg=float(loader.Tp_true),
            filter_seed=filter_seed, depth=loader.depth, n_ensemble=N_ENSEMBLE,
            mult_inflation=1.0, add_inflation=0.005, loc_radius=spec['loc'])
    diag = (f.background_diagnostics() if hasattr(f, 'background_diagnostics') else None)

    Xg, Yg = np.meshgrid(gx, gy)
    q = np.column_stack([Yg.ravel(), Xg.ravel()])
    nr, zr, ms, t, nobs = [], [], [], [], []
    sh_a = {k: [] for k in SHELL_KEYS}
    sh_z = {k: [] for k in SHELL_KEYS}
    fr = {k: [] for k in SHELL_KEYS}
    z_pool = []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        if i < 20 and len(z):
            z_pool.append(z)
        ms.append(float(f.update(x, y, z)))
        pred = f.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear',
                                         bounds_error=False, fill_value=0.0)
        pf = interp(q).reshape(len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None,
                                obs_y=y if len(y) else None)
        nr.append(float(m['nRMSE']))
        zr.append(float(np.sqrt(np.mean(gt ** 2)) / loader.Hs_true))
        t.append(float(loader.timestamps[i]))
        nobs.append(int(len(x)))
        if len(x):
            for k, msk in zip(SHELL_KEYS, _shell_masks(Xg, Yg, x, y)):
                fr[k].append(float(np.mean(msk)))
                sh_a[k].append(float(np.sqrt(np.mean((pf[msk] - gt[msk]) ** 2)))
                               if np.any(msk) else np.nan)
                sh_z[k].append(float(np.sqrt(np.mean(gt[msk] ** 2)))
                               if np.any(msk) else np.nan)
        else:
            for k in SHELL_KEYS:
                fr[k].append(np.nan); sh_a[k].append(np.nan); sh_z[k].append(np.nan)

    t = np.array(t)
    sel = t > 2.0 * float(loader.Tp_true)

    def _sm(d):
        out = {}
        for k, v in d.items():
            a = np.array(v, dtype=float)[sel]
            out[k] = float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')
        return out

    z_pool = np.concatenate(z_pool) if z_pool else np.zeros(1)
    return dict(avg_nRMSE=float(np.mean(np.array(nr)[sel])),
                zero_nRMSE=float(np.mean(np.array(zr)[sel])),
                n_match=int(sel.sum()), n_all=len(nr),
                ms_median=float(np.median(ms)),
                n_obs_median=float(np.median(nobs)),
                shell_rmse=_sm(sh_a), zero_shell=_sm(sh_z), shell_frac=_sm(fr),
                effective_config=(letkf_spec.readback(f) if spec['kind'] == 'spec'
                                  else _stock_readback(f)),
                background_diag=diag,
                loc_radius=float(f.loc_radius), n_ensemble=int(f.n_ensemble),
                background=spec['background'], filter_class=type(f).__name__,
                filter_seed=int(filter_seed),
                hs_obs4std=float(4.0 * np.std(z_pool)),
                Tp=float(loader.Tp_true), Hs=float(loader.Hs_true),
                depth=float(loader.depth))


def _stock_readback(f):
    keys = ('loc_radius', 'n_ensemble', 'mult_inflation', 'add_inflation', 'obs_noise',
            'max_obs', 'depth', 'dt', 'max_eta', 'nx', 'ny', 'n_grid', 'n_state',
            'dx', 'dy', 'g')
    out = {'filter_class': type(f).__name__, 'background': 'white',
           'white_amp': 0.1, 'Hs_bg': None, 'Tp_bg': None, 'filter_seed': None}
    for k in keys:
        v = getattr(f, k, '<absent>')
        out[k] = (v if isinstance(v, (int, float, str, bool, type(None)))
                  else (float(v) if isinstance(v, np.floating) else str(v)))
    return out


def run_ship_rerun(path, est_seed=42):
    """G-5 only: re-run the shipping arm today and compare with S13's stored value."""
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    cfg['seed'] = est_seed
    r = run_estimation(path, config=cfg, verbose=False)
    return float(r['avg_nRMSE'])


# ── schema check (PREREG G-3 / discipline 2) ───────────────────────────────────
def check_row(row):
    miss = [k for k in REQUIRED_LETKF if k not in row or row[k] is None]
    for fld in ('shell_rmse', 'zero_shell', 'shell_frac'):
        d = row.get(fld) or {}
        miss += ['%s[%s]' % (fld, k) for k in SHELL_KEYS if k not in d]
    return miss


# ── G-7 isotropy probe (AMENDMENT_1 B1; RUN-FREE, objects only) ────────────────
def isotropy_probe(verbose=True):
    """Build the background objects and measure the two propagation-isotropy
    diagnostics.  No episode is loaded, no estimator or LiDAR is called."""
    import letkf_spec
    egx = np.linspace(10.0, 250.0, 40)
    egy = np.linspace(-100.0, 100.0, 30)
    cases = [('SPEC-BF4', 'jonswap', 1.2, 7.0, 42), ('SPEC-BF5', 'jonswap', 2.0, 9.0, 42),
             ('SPEC-BF4', 'jonswap', 1.2, 7.0, 43), ('SPEC-BF4', 'jonswap', 1.2, 7.0, 44),
             ('WHITER',   'white',   1.2, 7.0, 42)]
    out, ok = [], True
    for name, bg, hs, tp, sd in cases:
        f = letkf_spec.LETKFGridSpec(
            egx, egy, background=bg, Hs_bg=hs, Tp_bg=tp, filter_seed=sd,
            depth=50.0, n_ensemble=N_ENSEMBLE, mult_inflation=1.0,
            add_inflation=0.005, loc_radius=50.0)
        d = f.background_diagnostics()
        ok &= bool(d['isotropy_ok'])
        out.append(dict(case=name, Hs=hs, Tp=tp, filter_seed=sd, **d))
        if verbose:
            print(f'  {name:9s} Hs={hs} Tp={tp} seed={sd}  '
                  f'corr(etadot,d eta/dx)={d["corr_detadot_dx"]:+.4f}  '
                  f'corr(etadot,d eta/dy)={d["corr_detadot_dy"]:+.4f}  '
                  f'(tol {letkf_spec.ISOTROPY_TOL})  '
                  f'eta_std={d["eta_std"]:.5f}  lag1={d["lag1_corr_x"]:.3f}/'
                  f'{d["lag1_corr_y"]:.3f}  ratio={d["detadot_over_deta"]:.4f} '
                  f'vs w_rms={d["omega_rms_spectrum"]:.4f} '
                  f'-> {"PASS" if d["isotropy_ok"] else "FAIL"}', flush=True)
    return ok, out


# ── gates ──────────────────────────────────────────────────────────────────────
def gates():
    ok = True
    eps = episodes()

    n_sp = sum(e['stratum'] == 'sparse' for e in eps)
    n_cr = sum(e['stratum'] == 'crowded' for e in eps)
    suff = {e['suffix'] for e in eps}
    dup = len({e['key'] for e in eps}) != len(eps)
    ship = import_ship()
    keys_s13 = set(ship)
    g1 = (n_sp == 64 and n_cr == 64 and suff == {'uniform_ra_rf_ds'} and not dup
          and {e['key'] for e in eps} == keys_s13)
    print(f'G-1 sample: sparse={n_sp} crowded={n_cr} suffix={suff} dup={dup} '
          f'keyset==S13:{ {e["key"] for e in eps} == keys_s13 }  -> {"PASS" if g1 else "FAIL"}')
    ok &= g1

    n3 = sum(1 for v in ship.values() if set(v) == {42, 43, 44})
    shell_ok = all((r.get('shell_rmse') or {}) and
                   len(r['shell_rmse']) == 9 for v in ship.values() for r in v.values())
    g2 = (len(ship) == 128 and n3 == 128 and shell_ok)
    print(f'G-2 SHIP import: episodes={len(ship)} with 3 seeds={n3} '
          f'rows={sum(len(v) for v in ship.values())} shells9={shell_ok} '
          f'-> {"PASS" if g2 else "FAIL"}')
    ok &= g2

    print('G-3 required-field schema (PREREG 2.1):')
    print(f'    LETKF arms: {", ".join(REQUIRED_LETKF)}')
    print(f'    shells    : {", ".join(SHELL_KEYS)}  (obs: prefix, discipline 9)')
    print(f'    SHIP rows : {", ".join(REQUIRED_SHIP)}')
    print('    -> enforced by check_row() in smoke and by analyze_s16.py before any verdict')

    # NOTE (found by this gate, 2026-08-14): the two SHIP sources record
    # `effective_config` with DIFFERENT SEMANTICS.  S13 stored the INTENT dict
    # (init_method='auto'); S13b stored the OBJECT read-back, where `init_method`
    # already holds the RESOLVED route ('learned'/'v2_konly').  The PREREG G-4 intent
    # -- "every imported SHIP row is the shipping config with lambda_l1=1e-4" -- is
    # therefore checked on the intent table where one exists, and the resolved route is
    # printed separately.  (S13 recording intent as `effective_config` is the discipline-4
    # violation the 2026-08-13 audit already logged; this batch reads back from objects.)
    bad, routes = [], collections.Counter()
    for k, v in ship.items():
        for s, r in v.items():
            eff = r.get('effective_config') or {}
            intent = r.get('intent_config') or eff
            routes[(eff.get('init_method') if 'intent_config' in r else r.get('route'))] += 1
            if (eff.get('lambda_l1') != 1e-4 or intent.get('lambda_l1') != 1e-4
                    or intent.get('init_method') != 'auto' or eff.get('seed') != s):
                bad.append((k, s, eff.get('lambda_l1'), intent.get('init_method'),
                            eff.get('seed')))
    g4 = not bad
    print(f'G-4 SHIP config: lambda_l1==1e-4 & intent init_method==auto & seed matches '
          f'on {sum(len(v) for v in ship.values())} rows, violations={len(bad)} '
          f'-> {"PASS" if g4 else "FAIL"}')
    print(f'    resolved router route over the 384 rows: {dict(routes)}')
    if bad[:3]:
        print('    ', bad[:3])
    ok &= g4

    preload()
    print('G-7 isotropy (AMENDMENT_1 B1, run-free -- objects only):')
    g7i, iso = isotropy_probe()
    print(f'    -> {"PASS" if g7i else "FAIL -- undeclared directional oracle"}')
    ok &= g7i
    json.dump(iso, open(os.path.join(HERE, 'isotropy_probe.json'), 'w'), indent=1)

    dg, files = code_fingerprint()
    dg2, _ = code_fingerprint()
    json.dump(dict(digest=dg, git_head=HEAD, git_dirty_n=DIRTY, files=files),
              open(FP_PATH, 'w'), indent=1)
    print(f'G-8 code fingerprint: digest={dg} files={len(files)} stable={dg == dg2} '
          f'(git {HEAD[:8]}, dirty {DIRTY}) -> code_fingerprint.json  '
          f'{"PASS" if dg == dg2 else "FAIL"}')
    ok &= (dg == dg2)

    h5 = sorted(glob.glob(os.path.join(HERE, '*.h5')))
    g10 = not h5
    print(f'G-10 zero generation: new .h5 in batch dir={len(h5)}  seed ledger consumed=0 '
          f'-> {"PASS" if g10 else "FAIL"}')
    ok &= g10

    print(f'\nGATES {"PASS -- cleared for smoke" if ok else "FAIL -- batch halted"}')
    print('(G-5 bit-reproduction, G-6 filter-seed sensitivity, G-7 repair proof, '
          'G-9 zero-field identity are measured by `smoke`.)')
    return ok


# ── smoke ──────────────────────────────────────────────────────────────────────
def smoke(arms=None, skip_g6=False):
    """RESUMABLE: rows already in smoke_results.jsonl and diagnostics already in
    smoke_report.json are skipped, so the smoke can be completed across several calls
    (the first attempt was reaped by the harness while another batch saturated the box)."""
    preload()
    eps = [e for e in episodes() if e['key'] in SMOKE_KEYS]
    assert len(eps) == 2, [e['key'] for e in eps]
    ship = import_ship()
    s13 = {r['episode']: r for r in _load_jsonl(S13) if r.get('arm') == 'LETKF_PLUS'}
    rep_path = SMOKE_REPORT
    rep = (json.load(open(rep_path)) if os.path.exists(rep_path)
           else {'g5_ship': [], 'g5_white': [], 'g6': [], 'g7': [], 'g9': [], 'timing': {}})

    def save():
        json.dump(rep, open(rep_path, 'w'), indent=1)

    rows = ([json.loads(l) for l in open(SMOKE_OUT)] if os.path.exists(SMOKE_OUT) else [])
    done = {(r['episode'], r['arm']) for r in rows if 'error' not in r}
    print(f'=== resume: {len(done)} arm rows already present, '
          f'{len(rep["g5_ship"])} G-5 SHIP, {len(rep["g6"])} G-6 ===', flush=True)

    print('=== G-5 bit reproduction (justifies reusing S13/S13b SHIP columns) ===', flush=True)
    for e in eps:
        if any(x['key'] == e['key'] for x in rep['g5_ship']):
            continue
        t0 = time.time()
        got = run_ship_rerun(e['path'], 42)
        ref = ship[e['key']][42]['avg_nRMSE']
        d = abs(got - ref)
        print(f'  SHIP  {e["key"]:44s} rerun={got:.12f} S13={ref:.12f} |d|={d:.3e} '
              f'({time.time()-t0:.0f}s) {"OK" if d == 0 else "MISMATCH"}', flush=True)
        rep['g5_ship'].append(dict(key=e['key'], rerun=got, stored=ref, d=d))
        save()
    for x in rep['g5_ship']:
        print(f'  [cached] SHIP {x["key"]:44s} |d|={x["d"]:.3e}', flush=True)

    print('\n=== arms (2 episodes x 5 arms) ===', flush=True)
    with open(SMOKE_OUT, 'a') as fh:
        for arm in (arms or ARM_ORDER):
            for e in eps:
                if (e['key'], arm) in done:
                    print(f'  [cached] {arm:9s} {e["key"]}', flush=True)
                    continue
                t0 = time.time()
                res = run_letkf(e['path'], arm)
                dg, dgf = code_fingerprint()
                row = dict(episode=e['key'], arm=arm, stratum=e['stratum'], bf=e['bf'],
                           ncomp=e['ncomp'], seed=e['seed'], wall_s=time.time() - t0,
                           git_head=HEAD, git_dirty_n=DIRTY, code_md5=dg,
                           code_files_n=len(dgf), **res)
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                rows.append(row)
                miss = check_row(row)
                print(f'  {arm:9s} {e["key"]:44s} nRMSE={row["avg_nRMSE"]:.6f} '
                      f'zero={row["zero_nRMSE"]:.6f} ({row["wall_s"]:.0f}s) '
                      f'schema={"OK" if not miss else "MISSING " + str(miss)}')
                if arm == 'WHITE50':
                    ref = s13[e['key']]['avg_nRMSE']
                    d = abs(row['avg_nRMSE'] - ref)
                    print(f'      G-5 WHITE50 vs S13 LETKF_PLUS: {row["avg_nRMSE"]:.12f} '
                          f'vs {ref:.12f}  |d|={d:.3e} {"OK" if d == 0 else "MISMATCH"}',
                          flush=True)
                    rep['g5_white'].append(dict(key=e['key'], rerun=row['avg_nRMSE'],
                                                stored=ref, d=d))
                    save()

    # G-5 for WHITE50 rows that were produced by an earlier (reaped) invocation
    for r in rows:
        if r['arm'] == 'WHITE50' and not any(x['key'] == r['episode']
                                             for x in rep['g5_white']):
            d = abs(r['avg_nRMSE'] - s13[r['episode']]['avg_nRMSE'])
            rep['g5_white'].append(dict(key=r['episode'], rerun=r['avg_nRMSE'],
                                        stored=s13[r['episode']]['avg_nRMSE'], d=d))
    for x in rep['g5_white']:
        print(f'  G-5 WHITE50 {x["key"]:44s} |d|={x["d"]:.3e} '
              f'{"OK" if x["d"] == 0 else "MISMATCH"}', flush=True)
    save()

    print('\n=== G-9 zero-field identity across arms ===', flush=True)
    rep['g9'] = []
    for e in eps:
        v = [r['zero_nRMSE'] for r in rows if r['episode'] == e['key']]
        sp = (max(v) - min(v)) if v else float('nan')
        print(f'  {e["key"]:44s} arms={len(v)} zero spread = {sp:.3e} '
              f'{"OK" if sp < 1e-12 else "FAIL"}', flush=True)
        rep['g9'].append(dict(key=e['key'], n_arms=len(v), spread=sp))

    print('\n=== G-7 repair proof (background diagnostics from the object) ===', flush=True)
    rep['g7'] = []
    iso_bad = []
    for r in rows:
        d = r.get('background_diag')
        if d and r['arm'] in ('SPEC50', 'WHITER50'):
            print(f'  {r["arm"]:9s} {r["episode"]:44s} bg={d["background"]:8s} '
                  f'eta_std={d["eta_std"]:.4f} (target {d["eta_target"]}) '
                  f'lag1_x={d["lag1_corr_x"]:.3f} lag1_y={d["lag1_corr_y"]:.3f} '
                  f'|mean|max={d["ens_mean_abs"]:.2e} '
                  f'std(detadt)/std(eta)={d["detadot_over_deta"]:.4f} '
                  f'w_rms_spec={d["omega_rms_spectrum"]:.4f} modes={d["n_modes_nonzero"]}',
                  flush=True)
            # AMENDMENT_1 B1: the fifth and sixth G-7 diagnostics.  The four above are
            # invariant to the sign of omega and cannot see a one-way background.
            cdx, cdy = d.get('corr_detadot_dx'), d.get('corr_detadot_dy')
            ok = d.get('isotropy_ok')
            print(f'            ISOTROPY corr(etadot,d eta/dx)={cdx:+.4f} '
                  f'corr(etadot,d eta/dy)={cdy:+.4f}  tol={d.get("isotropy_tol")}  '
                  f'-> {"PASS" if ok else "FAIL (directional oracle)"}', flush=True)
            if not ok:
                iso_bad.append((r['arm'], r['episode'], cdx, cdy))
            rep['g7'].append(dict(arm=r['arm'], key=r['episode'], **d))
    rep['g7_isotropy_fail'] = iso_bad
    save()
    if iso_bad:
        raise SystemExit('G-7 ISOTROPY FAIL -- batch halted (AMENDMENT_1 B1): %s' % iso_bad)

    print('\n=== G-6 filter-seed sensitivity (SPEC50 x seeds 42/43/44) ===', flush=True)
    for e in eps:
        if skip_g6 or any(x['key'] == e['key'] for x in rep['g6']):
            continue
        vals = []
        for s in (42, 43, 44):
            v = run_letkf(e['path'], 'SPEC50', filter_seed=s)['avg_nRMSE']
            vals.append(v)
        rel = float(np.std(vals, ddof=1) / np.mean(vals))
        rep['g6'].append(dict(key=e['key'], vals=vals, rel_sd=rel))
        save()
    for x in rep['g6']:
        print(f'  {x["key"]:44s} {["%.6f" % v for v in x["vals"]]} '
              f'rel_sd={100*x["rel_sd"]:.3f}% '
              f'{"single seed OK" if x["rel_sd"] <= 0.02 else "ESCALATE to median-of-3"}',
              flush=True)

    # timing is recomputed from the rows themselves; wall_s under CPU contention is
    # NOT usable for cost estimation (see NOTES.md 11:20 -- another batch inflated it 11.7x)
    rep['timing'] = {}
    for r in rows:
        rep['timing'].setdefault(r['arm'], []).append(r['wall_s'])
    save()
    print('\n-> smoke_results.jsonl, smoke_report.json', flush=True)
    for a, v in sorted(rep['timing'].items()):
        print(f'  timing {a:9s} n={len(v)} min={min(v):.0f}s median={np.median(v):.0f}s '
              f'(min-based 128 ep = {128*min(v)/3600:.2f} h)', flush=True)
    tot = sum(min(v) for v in rep['timing'].values()) * 128 / 3600
    print(f'  FULL BATCH ESTIMATE (5 arms x 128 ep, sequential, UNCONTENDED) = {tot:.2f} h',
          flush=True)


# ── run ────────────────────────────────────────────────────────────────────────
def main_run(arms, seeds=BASELINE_SEEDS):
    preload()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['episode'], r['arm'], r.get('filter_seed')))
    eps = episodes()
    # seed-major order: the whole arm set at seed 42 completes first, so R-1 (which needs
    # only the SPEC arms and the co-recorded zero field) can be read provisionally before
    # the remaining two seeds are spent.  Order does not touch any frozen rule.
    jobs = [(e, a, s) for s in seeds for a in arms for e in eps]
    todo = [j for j in jobs if (j[0]['key'], j[1], j[2]) not in done]
    dg, files = code_fingerprint()
    print(f'# arms={arms} seeds={seeds} total={len(jobs)} remaining={len(todo)} '
          f'HEAD={HEAD[:8]} dirty={DIRTY} code_md5={dg}', flush=True)
    t_start = time.time()
    with open(OUT, 'a') as fh:
        for i, (e, a, sd) in enumerate(todo, 1):
            t0 = time.time()
            row = None
            for attempt in (1, 2):
                try:
                    res = run_letkf(e['path'], a, filter_seed=sd)
                    dg, dgf = code_fingerprint()
                    row = dict(episode=e['key'], arm=a, stratum=e['stratum'], bf=e['bf'],
                               ncomp=e['ncomp'], seed=e['seed'], attempt=attempt,
                               wall_s=time.time() - t0, git_head=HEAD,
                               git_dirty_n=DIRTY, code_md5=dg, code_files_n=len(dgf),
                               **res)
                    break
                except Exception as ex:
                    err = f'{type(ex).__name__}: {ex}'
                    print(f'    ! attempt {attempt} failed: {err}', flush=True)
                    row = dict(episode=e['key'], arm=a, stratum=e['stratum'],
                               filter_seed=sd, attempt=attempt, error=err,
                               wall_s=time.time() - t0, git_head=HEAD, code_md5=dg)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            el = time.time() - t_start
            print(f'[{i}/{len(todo)}] {a:9s} s{sd} {e["key"]:44s} '
                  f'{row.get("avg_nRMSE", row.get("error"))} '
                  f'({time.time()-t0:.0f}s, ETA {el/i*(len(todo)-i)/60:.0f}m)', flush=True)
    json.dump(dict(digest=dg, git_head=HEAD, git_dirty_n=DIRTY, files=files),
              open(FP_PATH, 'w'), indent=1)


if __name__ == '__main__':
    cmd = (sys.argv[1:] or ['gates'])[0]
    if cmd == 'gates':
        sys.exit(0 if gates() else 1)
    elif cmd == 'isotropy':                       # AMENDMENT_1 B1, run-free
        preload()
        good, _ = isotropy_probe()
        sys.exit(0 if good else 1)
    elif cmd == 'smoke':
        rest = [a for a in sys.argv[2:] if a != '--skip-g6']
        smoke(arms=(rest or None), skip_g6=('--skip-g6' in sys.argv))
    elif cmd == 'run':
        rest = sys.argv[2:]
        sds = BASELINE_SEEDS
        if any(a.startswith('--seeds=') for a in rest):
            sds = tuple(int(x) for a in rest if a.startswith('--seeds=')
                        for x in a.split('=', 1)[1].split(','))
            rest = [a for a in rest if not a.startswith('--seeds=')]
        main_run(rest or list(ARM_ORDER), sds)
    else:
        print(__doc__)
