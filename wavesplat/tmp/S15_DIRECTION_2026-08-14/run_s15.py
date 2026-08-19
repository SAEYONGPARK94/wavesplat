#!/usr/bin/env python3
"""S15_DIRECTION 러너 — `PREREG.md` (md5 는 `FROZEN_md5.txt`) 를 그대로 구현.

    python3 tmp/S15_DIRECTION_2026-08-14/run_s15.py gates            # G-1..G-11 (G-0 제외)
    python3 tmp/S15_DIRECTION_2026-08-14/run_s15.py gates --partial  # 표본수 게이트 완화
    python3 tmp/S15_DIRECTION_2026-08-14/run_s15.py g0               # ★G-0 S4 재현 (16ep)
    python3 tmp/S15_DIRECTION_2026-08-14/run_s15.py g0 --only 2      # G-0 축소 (스모크)
    python3 tmp/S15_DIRECTION_2026-08-14/run_s15.py smoke            # 2쌍 전 팔
    python3 tmp/S15_DIRECTION_2026-08-14/run_s15.py run              # 전량
    python3 tmp/S15_DIRECTION_2026-08-14/run_s15.py run ORA-WPHI-NT@L3,...   # 팔 선택

`results.jsonl` 에 `(episode, arm, est_seed)` 당 한 줄 append. **재개 가능**.
스모크는 `smoke_results.jsonl`, G-0 는 `g0_results.jsonl` 로 **분리**한다.

각 행이 지고 다니는 것 (PREREG §7):
  · `effective_config`  = 추정기/LETKF **객체에서 되읽은** 값 (의도표 금지)
  · `shell_rmse`        = obs: 9구간, **전 팔** 동일 헬퍼
  · `ship_ring_rmse`    = ship: 7구간, **전 팔** 동일 헬퍼
  · `code_md5`(임포트된 저장소 .py 전부) · `import_md5`(핵심 임포트 경로 파일) · `git_head`
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

# ★스레드 고정은 **numpy/torch 임포트보다 먼저** 와야 한다.
# `spawn` 자식은 `__main__`(이 파일)을 먼저 import 하므로 워커 initializer 에서 세면 이미 늦다.
# 2026-08-14 스모크 실측: 워커당 260% CPU x 6워커 = 초과구독으로 `SHIP` 이 19 s(비경쟁,
# S13/HOSGATE 실측) 대신 108 s 가 됐다.
#
# ⚠팔마다 최적값이 다르다 (SMOKE.md):
#   wavesplat 팔 (ORA-*, AUTO-*, SHIP)  : 스레드 1 + 워커 6   (프로세스 병렬이 이긴다)
#   LETKF_PLUS                          : 스레드 8 + 워커 2   (BLAS 병렬이 이긴다;
#                                         스레드 1 이면 40 s -> 수분으로 늘어난다)
# 값은 `S15_THREADS` 로 주고 **각 행에 `n_threads` 로 기록**한다 (부동소수 축약 순서가
# 스레드 수에 의존할 수 있으므로 재현에 필요한 정보다).
S15_THREADS = os.environ.get('S15_THREADS', '1')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = S15_THREADS

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../wavesplat/wavesplat
sys.path.insert(0, ROOT)
os.chdir(ROOT)

S4_DIR = os.path.join(ROOT, 'tmp', 'S4_phase_oracle_2026-08-08')
DIR0 = {'crowded': os.path.join(ROOT, 'data_indep_crowded_2026-08-08'),
        'sparse': os.path.join(ROOT, 'data_indep_sparse_2026-08-08')}
DIR180 = os.path.join(HERE, 'data_dir180')

OUT = os.path.join(HERE, 'results.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_results.jsonl')
G0_OUT = os.path.join(HERE, 'g0_results.jsonl')
FP_PATH = os.path.join(HERE, 'code_fingerprint.json')

EP_SEEDS = (20, 21, 22, 23)
EST_SEEDS = (1589, 1590, 1591)          # PREREG §2 시드 원장. 잔여 [1592-1818]
G0_SEED = 1557                          # S4 가 이미 소모. 재현 목적 외 사용 금지
SMOKE_KEYS = ('BF4_15comp_seed20', 'BF4_2comp_seed20')

# obs: 셸 경계 -- run.py:358 과 **같은 리스트** (G-8b 가 일치를 증명한다)
OBS_EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
# ship: 고리 경계 -- 센서 원점까지의 거리 (PREREG §6 T-2)
SHIP_EDGES = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, np.inf]

# PREREG G-10: 잡마다 **전부 명시적으로 재기록**한다
ENV_KEYS = ('S2_ALPHA', 'S4_ORACLE_WPHI', 'S5_WINDOW', 'S8_L1_PHASE1_ONLY',
            'S8_L1_TRACK_ONLY', 'RX1_CONF_NORM', 'RX1_PHASE2_NOREG',
            'RX2_GROW_SUBTRACT', 'RX2_GROW_RECOMPUTE', 'RX2_GROW_TIGHT_SUPPRESS',
            'RX2_GROW_FITPHI')

HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()
DIRTY = len(subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                           text=True).stdout.strip().splitlines())

# ── 팔 정의 (PREREG §3) ─────────────────────────────────────────────────────
# S4 BASE_CFG 승계 + 모든 키 명시 (lambda_l1 미명시가 S5-S8 을 재실행 불가로 만들었다)
BASE = dict(use_dispersion=True, use_confidence=True, adaptive_n=True,
            tracking_window=3, prune_threshold=0.10, grow_residual_threshold=0.05,
            max_n=30, lr=0.01, lambda_smooth=1e-4, tracking_steps=10,
            tracking_mode='w_phi', tracking_lr=0.0, depth=50.0, device='cpu')
AUTO_LADDER = dict(n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
                   max_iters_per_stage=[200, 200, 500])

L3, L4 = 1e-3, 1e-4
ALL_STRATA = ('crowded', 'sparse')

ARMS = {
    # name:            kind    lambda  wphi   oracle  steps  strata
    'ORA-WPHI-0STEP@L3': dict(kind='ws', lam=L3, wphi=True, oracle=True, steps=0,
                              strata=ALL_STRATA),
    'ORA-WPHI-NT@L3':    dict(kind='ws', lam=L3, wphi=True, oracle=True, steps=900,
                              strata=ALL_STRATA),
    'ORA-WPHI-0STEP@L4': dict(kind='ws', lam=L4, wphi=True, oracle=True, steps=0,
                              strata=ALL_STRATA),
    'ORA-WPHI-NT@L4':    dict(kind='ws', lam=L4, wphi=True, oracle=True, steps=900,
                              strata=ALL_STRATA),
    'ORA-K-NT@L4':       dict(kind='ws', lam=L4, wphi=False, oracle=True, steps=900,
                              strata=('crowded',)),
    'AUTO-NT@L4':        dict(kind='ws', lam=L4, wphi=False, oracle=False, steps=None,
                              strata=ALL_STRATA),
    'AUTO-NT-DUP@L4':    dict(kind='ws', lam=L4, wphi=False, oracle=False, steps=None,
                              strata=('crowded',)),
    'SHIP':              dict(kind='ship', lam=L4, wphi=False, oracle=False, steps=None,
                              strata=ALL_STRATA),
    'LETKF_PLUS':        dict(kind='letkf', strata=ALL_STRATA),
    'ZERO':              dict(kind='zero', strata=ALL_STRATA),
}
ARMS_DEFAULT = tuple(ARMS)
SINGLE_SEED_ARMS = ('LETKF_PLUS', 'ZERO')      # 결정론적/고정시드 -- 1회만
G0_ARMS = ('ORA-WPHI-0STEP@L3', 'ORA-WPHI-NT@L3')
LETKF_KW = dict(n_ensemble=600, mult_inflation=1.0, add_inflation=0.005,
                loc_radius=50.0)


def n_gt(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return int(np.asarray(f['config/wave_components']).shape[0])


def arm_config(arm, path):
    """팔의 run_estimation config. **전 키 명시** (폴백 의존 금지)."""
    spec = ARMS[arm]
    if spec['kind'] == 'ship':
        from wavesplat.run import DEFAULT_CONFIG
        cfg = dict(DEFAULT_CONFIG)                  # 출하 기본값 전 키 복사
        assert cfg['lambda_l1'] == L4, cfg['lambda_l1']
        return cfg
    cfg = dict(BASE)
    cfg['lambda_l1'] = spec['lam']
    if spec['oracle']:
        cfg.update(init_method='oracle', n_gaussians=[n_gt(path)],
                   loss_thresholds=[None], max_iters_per_stage=[spec['steps']])
    else:
        cfg.update(AUTO_LADDER)
        cfg.update(init_method='auto', analytic_ensemble_k=3)
    assert 'init_method' in cfg and 'lambda_l1' in cfg
    return cfg


# ── 표본 ────────────────────────────────────────────────────────────────────
def episodes():
    out = []
    for st, d in DIR0.items():
        for p in sorted(glob.glob(os.path.join(d, '*.h5'))):
            b = os.path.basename(p)
            m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_uniform_ra_rf_ds\.h5', b)
            if not m or int(m.group(3)) not in EP_SEEDS:
                continue
            out.append(dict(path=p, direction=0, stratum=st, bf=m.group(1),
                            ncomp=int(m.group(2)), ep_seed=int(m.group(3))))
    for p in sorted(glob.glob(os.path.join(DIR180, '*.h5'))):
        b = os.path.basename(p)
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_uniform_dir180_ra_rf_ds\.h5', b)
        if not m or int(m.group(3)) not in EP_SEEDS:
            continue
        nc = int(m.group(2))
        out.append(dict(path=p, direction=180, stratum=('crowded' if nc in (15, 30)
                                                        else 'sparse'),
                        bf=m.group(1), ncomp=nc, ep_seed=int(m.group(3))))
    for e in out:
        e['pair'] = f"{e['bf']}_{e['ncomp']}comp_seed{e['ep_seed']}"
        e['key'] = f"dir{e['direction']}/{e['pair']}"
    return out


# ── 지문 ────────────────────────────────────────────────────────────────────
def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


IMPORT_PATHS = ('wavesplat/run.py', 'wavesplat/optimizer/sequential_update.py',
                'wavesplat/optimizer/initializer.py', 'wavesplat/optimizer/learned_init.py',
                'wavesplat/optimizer/coarse_to_fine.py', 'wavesplat/model/spectral_gaussian.py',
                'wavesplat/evaluation/metrics.py', 'wavesplat/data_loader.py',
                'wavesplat/baselines/letkf_grid.py', 'configs/estimation.yaml')


def import_md5():
    """★임포트 경로 파일 지문. `git_head` 는 더러운 작업트리를 못 잡는다."""
    d = {p: (_md5(os.path.join(ROOT, p)) if os.path.exists(os.path.join(ROOT, p)) else None)
         for p in IMPORT_PATHS}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest(), d


def code_fingerprint():
    """저장소 코드 전체의 단일 digest.

    ⚠**임포트된 모듈 목록으로 만들면 안 된다** — 팔마다 임포트하는 모듈이 달라
    같은 배치 안에서 digest 가 갈린다(2026-08-14 스모크에서 실측: 2종).
    그래서 `wavesplat/**` · `wavesim/**` 의 **모든** `.py` + `configs/*.yaml` 이라는
    **고정 파일집합**을 해시한다. 임포트 순서와 무관하게 항상 같은 값이 나온다.
    """
    files = {}
    for sub in ('wavesplat', 'wavesim'):
        base = os.path.join(ROOT, sub)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for fn in filenames:
                if fn.endswith('.py'):
                    p = os.path.join(dirpath, fn)
                    files[os.path.relpath(p, ROOT)] = _md5(p)
    for extra in ('configs/estimation.yaml', 'configs/default.yaml', 'configs/server.yaml'):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            files[extra] = _md5(p)
    return hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


# ── 셸·고리: **한 헬퍼, 전 팔 동일** (PREREG §7 / G-8) ──────────────────────
_RING_CACHE = {}


def _ring_masks(gx, gy):
    key = (float(gx[0]), float(gx[-1]), len(gx), float(gy[0]), float(gy[-1]), len(gy))
    if key not in _RING_CACHE:
        Xg, Yg = np.meshgrid(np.asarray(gx), np.asarray(gy))
        r = np.hypot(Xg, Yg)
        _RING_CACHE[key] = [((r >= a) & (r < b),
                             f'{int(a)}_{"inf" if np.isinf(b) else int(b)}')
                            for a, b in zip(SHIP_EDGES[:-1], SHIP_EDGES[1:])]
    return _RING_CACHE[key]


def ship_rings(pred, gt, gx, gy):
    out = {}
    for m, k in _ring_masks(gx, gy):
        out['ship:' + k] = (float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)))
                            if np.any(m) else float('nan'))
    return out


def obs_shells(pred, gt, gx, gy, ox, oy):
    """run.py:350-367 과 **같은 정의**. G-8b 가 wavesplat 팔에서 일치를 증명한다."""
    if ox is None or len(ox) == 0:
        return {}
    Xg, Yg = np.meshgrid(np.asarray(gx), np.asarray(gy))
    md = np.full(Xg.shape, np.inf)
    for j in range(len(ox)):
        md = np.minimum(md, np.sqrt((Xg - ox[j]) ** 2 + (Yg - oy[j]) ** 2))
    out = {}
    for a, b in zip(OBS_EDGES[:-1], OBS_EDGES[1:]):
        m = (md >= a) & (md < b)
        k = f'obs:{int(a)}_{"inf" if np.isinf(b) else int(b)}'
        out[k] = (float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)))
                  if np.any(m) else float('nan'))
    return out


def install_metric_wrapper():
    """`compute_all_metrics` 를 감싸 셸·고리를 **모든 호출자**에 붙인다.

    숫자는 바꾸지 않고 키만 더한다. `wavesplat.run` 은 심볼을 **모듈 로드시 import** 했으므로
    거기도 같이 덮어야 한다(rx2b 가 `_beamform_coarse` 에서 겪은 것과 같은 함정).
    """
    import wavesplat.evaluation.metrics as M
    import wavesplat.run as WR
    if getattr(M, '_s15_wrapped', False):
        return
    orig = M.compute_all_metrics

    def wrapped(pred, gt, gx, gy, Hs, *a, **kw):
        m = orig(pred, gt, gx, gy, Hs, *a, **kw)
        m['ship_ring_rmse'] = ship_rings(np.asarray(pred), np.asarray(gt), gx, gy)
        m['shell_rmse_s15'] = obs_shells(np.asarray(pred), np.asarray(gt), gx, gy,
                                         kw.get('obs_x'), kw.get('obs_y'))
        return m

    wrapped._s15_orig = orig
    M.compute_all_metrics = wrapped
    WR.compute_all_metrics = wrapped
    M._s15_wrapped = True


def _mean_dicts(frames, key):
    acc = {}
    for f in frames:
        for k, v in (f.get(key) or {}).items():
            if v is not None and np.isfinite(v):
                acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _panel(frames):
    rho = np.array([f['Pearson_rho'] for f in frames if 'Pearson_rho' in f], float)
    vr = np.array([f['Variance_Ratio'] for f in frames if 'Variance_Ratio' in f], float)
    with np.errstate(invalid='ignore'):
        decorr = 1.0 - rho ** 2
        shrink = (np.sqrt(np.clip(vr, 0, None)) - rho) ** 2
    return (float(np.nanmean(decorr)) if rho.size else float('nan'),
            float(np.nanmean(shrink)) if rho.size else float('nan'))


# ── 환경 위생 (G-10) ────────────────────────────────────────────────────────
def set_env(wphi, s5_window=True):
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ[k] = '0'
    os.environ['S4_ORACLE_WPHI'] = '1' if wphi else '0'
    os.environ['S5_WINDOW'] = '1' if s5_window else '0'
    return saved


def restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── 팔 실행 ─────────────────────────────────────────────────────────────────
READBACK_KEYS = ('seed', 'lambda_l1', 'lr', 'init_method', 'use_confidence',
                 'use_dispersion', 'adaptive_n', 'max_n', 'tracking_mode',
                 'prune_threshold', 'grow_residual_threshold', 'lambda_smooth',
                 'tracking_steps', 'tracking_lr', 'depth', 'device',
                 'loss_thresholds', 'max_iters_per_stage', 'ensemble_k')


def readback(u):
    out = {'updater_class': type(u).__name__}
    for k in READBACK_KEYS:
        v = getattr(u, k, '<absent>')
        out[k] = (v if isinstance(v, (int, float, str, bool, type(None)))
                  else (list(v) if isinstance(v, (list, tuple)) else str(v)))
    return out


def run_ws(path, arm, est_seed, s5_window=True):
    """wavesplat 팔 (`ORA-*`, `AUTO-*`, `SHIP`)."""
    import torch
    import wavesplat.run as WR
    import wavesplat.optimizer.initializer as I
    from wavesplat.data_loader import EpisodeDataLoader

    spec = ARMS[arm]
    cfg = arm_config(arm, path)
    cfg['seed'] = est_seed
    captured = {}
    orig_mk = WR.make_updater

    def _capture(config):
        u = orig_mk(config)
        captured['u'] = u
        captured['torch_initial_seed'] = int(torch.initial_seed())
        captured['np_state0'] = int(np.random.get_state()[1][0])
        return u

    saved = set_env(bool(spec.get('wphi')), s5_window=s5_window)
    WR.make_updater = _capture
    try:
        if spec.get('oracle'):
            I.ORACLE_COMPONENTS = np.asarray(EpisodeDataLoader(path).wave_components)
        r = WR.run_estimation(path, config=cfg, verbose=False)
    finally:
        I.ORACLE_COMPONENTS = None
        WR.make_updater = orig_mk
        restore_env(saved)

    fr = r['frame_results']
    decorr, shrink = _panel(fr)
    u = captured.get('u')
    native = _mean_dicts(fr, 'shell_rmse')
    mine = _mean_dicts(fr, 'shell_rmse_s15')
    match = max((abs(native[k] - mine['obs:' + k])
                 for k in native if 'obs:' + k in mine), default=float('nan'))
    return dict(
        avg_nRMSE=float(r['avg_nRMSE']),
        avg_obs_loss=float(r.get('avg_obs_loss', np.nan)),
        avg_obs_resid_rms=float(np.nanmean([f['obs_resid_rms'] for f in fr
                                            if 'obs_resid_rms' in f])),
        avg_obs_resid_rms_oos=float(np.nanmean([f.get('obs_resid_rms_oos', np.nan)
                                                for f in fr])),
        decorr_mean=decorr, shrink_mean=shrink,
        n_frames=len(fr),
        init_nrmse=float(r.get('init_nrmse', np.nan)),
        window_resid=float(r.get('window_resid', np.nan)),
        window_n_obs=int(r.get('window_n_obs', 0) or 0),
        n_gaussians=(int(r['n_gaussians']) if r.get('n_gaussians') is not None else -1),
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        shell_rmse=mine, ship_ring_rmse=_mean_dicts(fr, 'ship_ring_rmse'),
        shell_rmse_native={('obs:' + k): v for k, v in native.items()},
        shell_native_max_abs=(float(match) if match == match else None),
        effective_config=(readback(u) if u is not None else None),
        intent_config={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                           else str(v)) for k, v in cfg.items()},
        torch_initial_seed=captured.get('torch_initial_seed'),
        np_state0=captured.get('np_state0'),
        lambda_l1=float(cfg['lambda_l1']))


def run_letkf(path, arm=None, est_seed=None, s5_window=True):
    """S13/HOSGATE 와 동일한 LETKF_PLUS. 셸·고리도 **같은 헬퍼**로 기록한다."""
    from scipy.interpolate import RegularGridInterpolator
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.baselines.letkf_grid import LETKFGrid
    from wavesplat.evaluation.metrics import compute_all_metrics
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40)
    egy = np.linspace(gy[0], gy[-1], 30)
    letkf = LETKFGrid(egx, egy, depth=loader.depth, **LETKF_KW)
    Xg, Yg = np.meshgrid(gx, gy)
    q = np.column_stack([Yg.ravel(), Xg.ravel()])
    rows, zr, ms, ts = [], [], [], []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        ms.append(float(letkf.update(x, y, z)))
        pred = letkf.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear',
                                         bounds_error=False, fill_value=0.0)
        pf = interp(q).reshape(len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        rows.append(compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                        obs_x=x if len(x) else None,
                                        obs_y=y if len(y) else None))
        # ⚠float32 함정: `np.float32 / python float` 은 NEP-50 에서 float32 로 남는다.
        # `metrics.nrmse` 는 `float(rmse)/Hs` 로 **float64 에서** 나누므로, 여기서도
        # 먼저 float() 로 올린 뒤 나눠야 G-9(문턱 1e-12)가 성립한다.
        # (스모크에서 실측: 안 맞추면 |Δ| = 1.1e-08 로 게이트가 FAIL 한다.)
        zr.append(float(np.sqrt(np.mean(gt ** 2))) / float(loader.Hs_true))
        ts.append(float(loader.timestamps[i]))
    sel = np.array(ts) > 2.0 * float(loader.Tp_true)
    fr = [r for r, s in zip(rows, sel) if s]
    decorr, shrink = _panel(fr)
    eff = {k: float(getattr(letkf, k)) for k in
           ('n_ensemble', 'loc_radius', 'mult_inflation', 'add_inflation',
            'obs_noise', 'max_obs', 'depth', 'dt')}
    eff['grid'] = f'{letkf.nx}x{letkf.ny}'
    eff['class'] = type(letkf).__name__
    return dict(
        avg_nRMSE=float(np.mean([f['nRMSE'] for f in fr])),
        zero_nRMSE=float(np.mean(np.array(zr)[sel])),
        decorr_mean=decorr, shrink_mean=shrink,
        n_frames=len(fr), n_all=len(rows),
        ms_median=float(np.median(ms)),
        shell_rmse=_mean_dicts(fr, 'shell_rmse_s15'),
        ship_ring_rmse=_mean_dicts(fr, 'ship_ring_rmse'),
        shell_rmse_native=None, shell_native_max_abs=None,
        effective_config=eff, intent_config=dict(LETKF_KW, np_seed=42),
        lambda_l1=None, Tp=float(loader.Tp_true), Hs=float(loader.Hs_true))


def run_zero(path, arm=None, est_seed=None, s5_window=True):
    """제로필드 바닥. **다른 팔과 같은 채점기**를 통과시킨다 (G-9)."""
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.evaluation.metrics import compute_all_metrics
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    rows, zr, ts = [], [], []
    for i, frame in enumerate(loader):
        gt = frame.gt_field.cpu().numpy()
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        rows.append(compute_all_metrics(np.zeros_like(gt), gt, gx, gy, loader.Hs_true,
                                        obs_x=x if len(x) else None,
                                        obs_y=y if len(y) else None))
        # ⚠float32 함정: `np.float32 / python float` 은 NEP-50 에서 float32 로 남는다.
        # `metrics.nrmse` 는 `float(rmse)/Hs` 로 **float64 에서** 나누므로, 여기서도
        # 먼저 float() 로 올린 뒤 나눠야 G-9(문턱 1e-12)가 성립한다.
        # (스모크에서 실측: 안 맞추면 |Δ| = 1.1e-08 로 게이트가 FAIL 한다.)
        zr.append(float(np.sqrt(np.mean(gt ** 2))) / float(loader.Hs_true))
        ts.append(float(loader.timestamps[i]))
    sel = np.array(ts) > 2.0 * float(loader.Tp_true)
    fr = [r for r, s in zip(rows, sel) if s]
    decorr, shrink = _panel(fr)
    a = float(np.mean([f['nRMSE'] for f in fr]))
    b = float(np.mean(np.array(zr)[sel]))
    return dict(avg_nRMSE=a, zero_formula_nRMSE=b, zero_gap=abs(a - b),
                decorr_mean=decorr, shrink_mean=shrink,
                n_frames=len(fr), n_all=len(rows),
                shell_rmse=_mean_dicts(fr, 'shell_rmse_s15'),
                ship_ring_rmse=_mean_dicts(fr, 'ship_ring_rmse'),
                shell_rmse_native=None, shell_native_max_abs=None,
                effective_config={'class': 'ZeroField', 'frame_mask': 't > 2*Tp'},
                intent_config={'class': 'ZeroField'}, lambda_l1=None,
                Tp=float(loader.Tp_true), Hs=float(loader.Hs_true))


RUNNERS = {'ws': run_ws, 'ship': run_ws, 'letkf': run_letkf, 'zero': run_zero}


def preload():
    """세 계열의 임포트를 **런 전에** 끝낸다 (안 하면 `code_md5` 가 팔 순서에 의존한다)."""
    import wavesplat.run                                    # noqa: F401
    import wavesplat.data_loader                            # noqa: F401
    import wavesplat.baselines.letkf_grid                   # noqa: F401
    import wavesplat.evaluation.metrics                     # noqa: F401
    import wavesplat.optimizer.sequential_update            # noqa: F401
    import wavesplat.optimizer.initializer                  # noqa: F401
    import scipy.interpolate                                # noqa: F401
    import torch
    torch.set_num_threads(int(S15_THREADS))                 # 초과구독 차단 (위 주석 참조)
    install_metric_wrapper()


# ── 잡 ──────────────────────────────────────────────────────────────────────
def build_jobs(eps, arms, seeds=EST_SEEDS, s5_window=True):
    jobs = []
    for a in arms:
        for e in eps:
            if e['stratum'] not in ARMS[a]['strata']:
                continue
            ss = (None,) if a in SINGLE_SEED_ARMS else seeds
            for s in ss:
                jobs.append(dict(arm=a, seed=s, s5_window=s5_window, **e))
    return jobs


def row_id(j):
    return (j['key'], j['arm'], j['seed'])


def run_job(j):
    preload()
    t0 = time.time()
    spec = ARMS[j['arm']]
    base = dict(episode=j['key'], arm=j['arm'], direction=j['direction'],
                stratum=j['stratum'], bf=j['bf'], ncomp=j['ncomp'],
                ep_seed=j['ep_seed'], est_seed=j['seed'], pair=j['pair'],
                path=os.path.relpath(j['path'], ROOT))
    row = None
    for attempt in (1, 2):
        try:
            res = RUNNERS[spec['kind']](j['path'], j['arm'], j['seed'],
                                        s5_window=j['s5_window'])
            dg, _ = code_fingerprint()
            im, _ = import_md5()
            row = dict(base, attempt=attempt, wall_s=time.time() - t0,
                       git_head=HEAD, git_dirty_n=DIRTY, code_md5=dg, import_md5=im,
                       s5_window=bool(j['s5_window']), n_threads=int(S15_THREADS),
                       **res)
            break
        except Exception as ex:
            import traceback
            err = f'{type(ex).__name__}: {ex}'
            row = dict(base, attempt=attempt, error=err,
                       tb=traceback.format_exc()[-1200:], wall_s=time.time() - t0,
                       git_head=HEAD, code_md5=None, import_md5=None)
    return row


def _init_worker():
    sys.path.insert(0, ROOT)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    preload()


def execute(jobs, out_path, workers):
    # ★잠금: 러너를 실수로 두 번 띄우면 같은 셀이 두 줄 들어가 median-of-3 이 조용히
    # median-of-6 이 된다 (2026-08-14 설계 중 실제로 한 번 발생). 파일 하나로 막는다.
    lock = out_path + '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{os.getpid()} {time.strftime("%F %T")}\n'.encode())
        os.close(fd)
    except FileExistsError:
        print(f'!! 잠금 파일이 이미 있다: {lock}\n'
              f'   다른 러너가 돌고 있거나 비정상 종료했다. 확인 후 지우고 다시 실행할 것.')
        sys.exit(2)
    try:
        _execute(jobs, out_path, workers)
    finally:
        os.path.exists(lock) and os.remove(lock)


def _execute(jobs, out_path, workers):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['episode'], r['arm'], r.get('est_seed')))
    todo = [j for j in jobs if row_id(j) not in done]
    print(f'# total={len(jobs)} done={len(done)} todo={len(todo)} workers={workers} '
          f'threads={S15_THREADS} HEAD={HEAD[:8]} dirty={DIRTY}', flush=True)
    if not todo:
        print('nothing to do')
        return
    t0 = time.time()
    with open(out_path, 'a') as fh:
        if workers <= 1:
            it = (run_job(j) for j in todo)
        else:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(workers, initializer=_init_worker)
            it = pool.imap_unordered(run_job, todo, chunksize=1)
        for i, row in enumerate(it, 1):
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            el = time.time() - t0
            tag = row.get('avg_obs_loss', row.get('avg_nRMSE', row.get('error')))
            tag = f'{tag:.6f}' if isinstance(tag, float) else str(tag)[:60]
            print(f'[{i}/{len(todo)}] {row["arm"]:<19s} {row["episode"]:<28s} '
                  f'seed={row["est_seed"]} {tag} ({row["wall_s"]:.0f}s, '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}m)', flush=True)
        if workers > 1:
            pool.close()
            pool.join()
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


# ── G-0 : S4 재현 ───────────────────────────────────────────────────────────
def s4_reference():
    """S4 `results.jsonl` 에서 (cell, episode) -> (avg_obs_loss, nRMSE_recheck), seed 1557."""
    ref = {}
    p = os.path.join(S4_DIR, 'results.jsonl')
    if not os.path.exists(p):
        return ref
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r or r.get('seed') != G0_SEED:
            continue
        cell = r['cell']
        if cell not in ('ORA-WPHI-NT', 'ORA-WPHI-0STEP'):
            continue
        key = os.path.basename(r['episode'])[len('episode_'):-len('_uniform_ra_rf_ds.h5')]
        ref[(cell + '@L3', key)] = (r.get('avg_obs_loss'), r.get('nRMSE_recheck'),
                                    r.get('nRMSE'))
    return ref


def gate_g0(only=None, workers=1):
    """★S4 재현 게이트. dir0 · lambda=1e-3 · seed 1557 이 S4 저장값과 max|Δ| <= 1e-9."""
    ref = s4_reference()
    print(f'G-0 S4 참조행 {len(ref)} 개 (seed {G0_SEED})')
    if not ref:
        print('G-0 FAIL: S4 참조를 읽지 못했다')
        return False
    eps = [e for e in episodes() if e['direction'] == 0 and e['stratum'] == 'crowded']
    eps.sort(key=lambda e: e['pair'])
    if only:
        eps = eps[:only]
    jobs = build_jobs(eps, G0_ARMS, seeds=(G0_SEED,))
    execute(jobs, G0_OUT, workers)

    rows = [json.loads(l) for l in open(G0_OUT)]
    ok, worst = True, 0.0
    seen = 0
    for r in rows:
        if 'error' in r:
            print(f'    ! {r["arm"]} {r["episode"]} ERROR {r["error"]}')
            ok = False
            continue
        key = r['episode'].split('/')[-1]
        arm = r['arm'].replace('@L3', '') + '@L3'
        cell = arm.replace('@L3', '')
        t = ref.get((cell + '@L3', key))
        if t is None:
            continue
        seen += 1
        d_loss = abs(float(r['avg_obs_loss']) - float(t[0]))
        d_nrmse = abs(float(r['avg_nRMSE']) - float(t[1]))
        worst = max(worst, d_loss, d_nrmse)
        good = d_loss <= 1e-9 and d_nrmse <= 1e-9
        ok &= good
        print(f'    {r["arm"]:<19s} {key:<22s} obs_loss {r["avg_obs_loss"]:.8f} '
              f'vs S4 {t[0]:.8f}  |Δ|={d_loss:.2e}  nRMSE |Δ|={d_nrmse:.2e}  '
              f'{"OK" if good else "FAIL"}')
    print(f'G-0 대조 {seen} 행, worst |Δ| = {worst:.3e} (문턱 1e-9) -> '
          f'{"PASS" if ok and seen else "FAIL"}')
    return bool(ok and seen)


# ── 게이트 G-1..G-11 ────────────────────────────────────────────────────────
REQUIRED_COMMON = ('episode', 'arm', 'direction', 'stratum', 'bf', 'ncomp', 'ep_seed',
                   'avg_nRMSE', 'n_frames', 'wall_s', 'attempt', 'shell_rmse',
                   'ship_ring_rmse', 'effective_config', 'intent_config',
                   'code_md5', 'git_head', 'import_md5')
REQUIRED_WS = ('avg_obs_loss', 'avg_obs_resid_rms', 'decorr_mean', 'shrink_mean',
               'window_resid', 'window_n_obs', 'n_gaussians', 'init_nrmse',
               'torch_initial_seed', 'np_state0', 'lambda_l1')
REQUIRED_LETKF = ('zero_nRMSE', 'ms_median')
REQUIRED_ZERO = ('zero_formula_nRMSE', 'zero_gap')


def gates(require_full=True, smoke_path=SMOKE_OUT):
    import h5py
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    preload()
    ok = True
    eps = episodes()
    if not eps:
        print('게이트 중단: 에피소드가 0개다 (gen_s15.py 를 먼저 돌려라)')
        return False
    pairs = {}
    for e in eps:
        pairs.setdefault(e['pair'], {})[e['direction']] = e

    # G-1 팔별 lambda 명시
    g1 = True
    for a, spec in ARMS.items():
        if spec['kind'] in ('letkf', 'zero'):
            continue
        p = next((e['path'] for e in eps if e['stratum'] in spec['strata']), None)
        if p is None:
            continue
        cfg = arm_config(a, p)
        u_ok = ('lambda_l1' in cfg and 'init_method' in cfg
                and SequentialUpdater(cfg).lambda_l1 == cfg['lambda_l1'])
        if a == 'SHIP':
            from wavesplat.run import DEFAULT_CONFIG
            u_ok &= (set(cfg) == set(DEFAULT_CONFIG))
        print(f'G-1 {a:<19s} lambda_l1={cfg["lambda_l1"]:<8} init={cfg["init_method"]:<8} '
              f'키 {len(cfg):2d}개 -> {"PASS" if u_ok else "FAIL"}')
        g1 &= u_ok
    ok &= g1

    # G-2/G-3 짝 무결성 + 방향이 실제로 바뀌었다
    g2 = g3 = True
    n_full = sum(1 for v in pairs.values() if len(v) == 2)
    print(f'G-2/G-3 짝 {n_full}/{len(pairs)} 완비')
    for k in sorted(pairs):
        v = pairs[k]
        if len(v) != 2:
            print(f'    ! 불완전 쌍 {k}: dir {sorted(v)}')
            g2 = False
            continue
        with h5py.File(v[0]['path'], 'r') as f:
            c0 = np.asarray(f['config/wave_components'])
            z0 = f['frames/gt_wave_field'][:]
            t0 = f['frames/timestamps'][:]
        with h5py.File(v[180]['path'], 'r') as f:
            c1 = np.asarray(f['config/wave_components'])
            z1 = f['frames/gt_wave_field'][:]
            t1 = f['frames/timestamps'][:]
        dA = float(np.abs(c0[:, 0] - c1[:, 0]).max())
        dw = float(np.abs(c0[:, 1] - c1[:, 1]).max())
        dphi = float(np.abs(c0[:, 4] - c1[:, 4]).max())
        dk = float(np.abs(c0[:, 2:4] + c1[:, 2:4]).max())
        same_t = z0.shape == z1.shape and np.allclose(t0, t1)
        dz = float(np.abs(z0 - z1).max()) if same_t else float('nan')
        flip = float(c1[:, 2].mean()) < 0.0 < float(c0[:, 2].mean())
        p2 = (dA <= 1e-12 and dw <= 1e-12 and dphi <= 1e-12 and dk <= 1e-12 and same_t)
        p3 = (dz > 0) and flip
        g2 &= p2
        g3 &= p3
        print(f'    {k:<22s} dA={dA:.0e} dphi={dphi:.0e} |k180+k0|={dk:.0e} '
              f'frames={z0.shape[0]} max_t|Δη|={dz:.4f} m  mean_kx '
              f'{c0[:,2].mean():+.4f}->{c1[:,2].mean():+.4f}  '
              f'{"OK" if (p2 and p3) else "FAIL"}')
    print(f'    G-2 {"PASS" if g2 else "FAIL"}   G-3 {"PASS" if g3 else "FAIL"}')
    ok &= (g2 and g3)

    # G-5 되읽기 배선
    g5 = hasattr(SequentialUpdater(arm_config('AUTO-NT@L4', eps[0]['path'])), 'lambda_l1')
    print(f'G-5 effective_config 되읽기 배선 -> {"PASS" if g5 else "FAIL"}')
    ok &= g5

    # G-6 코드 지문
    dg, files = code_fingerprint()
    im, imf = import_md5()
    json.dump(dict(digest=dg, import_digest=im, git_head=HEAD, git_dirty_n=DIRTY,
                   files=files, import_files=imf), open(FP_PATH, 'w'), indent=1)
    print(f'G-6 코드 지문 digest={dg} ({len(files)} 파일) import_md5={im} '
          f'(git {HEAD[:8]}, dirty {DIRTY}) -> code_fingerprint.json  PASS')

    # G-7 시드 충돌 + 파일명 충돌
    used = set()
    for d in glob.glob(os.path.join(ROOT, 'data*')):
        for p in glob.glob(os.path.join(d, '*.h5')):
            m = re.search(r'seed(\d+)', os.path.basename(p))
            if m:
                used.add(int(m.group(1)))
    mine = set(EST_SEEDS)
    names = {os.path.basename(p) for p in glob.glob(os.path.join(DIR180, '*.h5'))}
    clash = [n for n in names
             if any(os.path.exists(os.path.join(d, n))
                    for d in glob.glob(os.path.join(ROOT, 'data*')))]
    g7 = (not (used & mine)) and not clash
    print(f'G-7 추정시드 {sorted(mine)} vs 기존 {len(used)}개 -> 교집합 '
          f'{sorted(used & mine)} | 파일명 충돌 {len(clash)} -> '
          f'{"PASS" if g7 else "FAIL"}')
    ok &= g7

    # G-4/G-8/G-9 는 스모크 산출물이 필요하다
    if os.path.exists(smoke_path):
        rows = [json.loads(l) for l in open(smoke_path) if l.strip()]
        rows = [r for r in rows if 'error' not in r]
        ok &= gate_fields(rows)
        ok &= gate_oracle(rows)
        ok &= gate_zero(rows)
    else:
        print('G-4/G-8/G-9: 스모크 산출물 없음 -> 스모크 후 다시 돌릴 것 (SKIP)')

    if require_full:
        n_cr = sum(1 for v in pairs.values() if len(v) == 2
                   and v[0]['stratum'] == 'crowded')
        n_sp = sum(1 for v in pairs.values() if len(v) == 2
                   and v[0]['stratum'] == 'sparse')
        gN = (n_cr == 16 and n_sp == 16)
        print(f'G-11 표본수: 혼잡 {n_cr} 쌍 / 희소 {n_sp} 쌍 (사전등록 16/16) -> '
              f'{"PASS" if gN else "FAIL (생성 미완)"}')
        ok &= gN

    print(f'\nGATES {"PASS -- 실행 인가" if ok else "FAIL -- 중단"}')
    return ok


def gate_fields(rows):
    """★G-8 : **모든 팔**이 §7 필수 필드를 갖는가. 셸 키 집합이 팔 간 동일한가."""
    ok = True
    seen_arms = sorted({r['arm'] for r in rows})
    print(f'G-8 필드 존재 검사: 팔 {len(seen_arms)}개 / 행 {len(rows)}개')
    shell_keysets, ring_keysets = {}, {}
    for r in rows:
        miss = [k for k in REQUIRED_COMMON if r.get(k) is None]
        kind = ARMS[r['arm']]['kind']
        if kind in ('ws', 'ship'):
            miss += [k for k in REQUIRED_WS if r.get(k) is None]
        elif kind == 'letkf':
            miss += [k for k in REQUIRED_LETKF if r.get(k) is None]
        elif kind == 'zero':
            miss += [k for k in REQUIRED_ZERO if r.get(k) is None]
        nan = [k for k in ('avg_nRMSE',) if not np.isfinite(r.get(k, np.nan))]
        if not r.get('shell_rmse'):
            miss.append('shell_rmse(빈dict)')
        if not r.get('ship_ring_rmse'):
            miss.append('ship_ring_rmse(빈dict)')
        if miss or nan:
            print(f'    ! {r["arm"]:<19s} {r["episode"]:<28s} 결측 {miss} NaN {nan}')
            ok = False
        # ★에피소드 안에서 팔 간 비교한다 (팔마다 담당 층이 달라 union 비교는 위양성)
        shell_keysets.setdefault(r['episode'], {})[r['arm']] = frozenset(
            r.get('shell_rmse') or {})
        ring_keysets.setdefault(r['episode'], {})[r['arm']] = frozenset(
            r.get('ship_ring_rmse') or {})
    same = True
    for ep in sorted(shell_keysets):
        s = set(shell_keysets[ep].values())
        rr = set(ring_keysets[ep].values())
        if len(s) != 1 or len(rr) != 1:
            same = False
            print(f'    ! {ep}: 팔 간 셸 키 집합 불일치 obs {len(s)}종 ship {len(rr)}종')
            for arm, ks in shell_keysets[ep].items():
                print(f'        {arm:<19s} obs {sorted(ks)}')
    print(f'    셸/고리 키 집합 에피소드 안 팔 간 동일: {"PASS" if same else "FAIL"} '
          f'({len(shell_keysets)} 에피소드)')
    ok &= same
    # G-8b: 내 헬퍼 == run.py 원본 정의 (wavesplat 팔에서만 검증 가능)
    worst = max((r['shell_native_max_abs'] for r in rows
                 if r.get('shell_native_max_abs') is not None), default=None)
    if worst is None:
        print('    G-8b 원본 셸 대조 불가 (wavesplat 팔 행 없음) -> FAIL')
        ok = False
    else:
        g8b = worst <= 1e-12
        print(f'    G-8b 헬퍼 vs run.py 원본 셸 worst |Δ| = {worst:.3e} -> '
              f'{"PASS" if g8b else "FAIL"}')
        ok &= g8b
    print(f'G-8 -> {"PASS" if ok else "FAIL"}')
    return ok


def gate_oracle(rows):
    """G-4 : 오라클 배선 (파라미터 전달 무결성 + 게이트 누출 없음)."""
    ok = True
    z = [r for r in rows if r['arm'].startswith('ORA-WPHI-0STEP')]
    if not z:
        print('G-4 SKIP (0STEP 행 없음)')
        return True
    worst = max(abs(float(r['avg_nRMSE'])) for r in z)
    g = worst <= 1e-6
    print(f'G-4 0STEP avg_nRMSE worst = {worst:.3e} (문턱 1e-6, 파라미터 전달 무결성) -> '
          f'{"PASS" if g else "FAIL"}')
    ok &= g
    k = [r for r in rows if r['arm'] == 'ORA-K-NT@L4']
    if k:
        bad = [r for r in k if abs(float(r['avg_nRMSE'])) <= 1e-6]
        print(f'    누출 프로브: ORA-K-NT 가 0STEP 처럼 완벽한 행 {len(bad)}개 '
              f'(0 이어야 한다) -> {"PASS" if not bad else "FAIL"}')
        ok &= (not bad)
    return ok


def gate_zero(rows):
    """G-9 : ZERO 의 채점기 경로 == 닫힌형."""
    z = [r for r in rows if r['arm'] == 'ZERO']
    if not z:
        print('G-9 SKIP (ZERO 행 없음)')
        return True
    worst = max(float(r['zero_gap']) for r in z)
    g = worst <= 1e-12
    print(f'G-9 ZERO 채점기 vs 닫힌형 worst |Δ| = {worst:.3e} -> {"PASS" if g else "FAIL"}')
    return g


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', default='gates',
                    choices=['gates', 'g0', 'smoke', 'run'])
    ap.add_argument('arms', nargs='?', default=None)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '6')))
    ap.add_argument('--only', type=int, default=None)
    ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()

    if a.cmd == 'gates':
        sys.exit(0 if gates(require_full=not a.partial) else 1)
    if a.cmd == 'g0':
        sys.exit(0 if gate_g0(only=a.only, workers=a.workers) else 1)

    arms = a.arms.split(',') if a.arms else list(ARMS_DEFAULT)
    eps = episodes()
    if a.cmd == 'smoke':
        eps = [e for e in eps if e['pair'] in SMOKE_KEYS]
        execute(build_jobs(eps, arms), SMOKE_OUT, a.workers)
    else:
        execute(build_jobs(eps, arms), OUT, a.workers)


if __name__ == '__main__':
    main()
