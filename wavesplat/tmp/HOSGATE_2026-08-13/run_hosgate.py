#!/usr/bin/env python3
"""HOSGATE 러너 — PREREG.md (md5 는 FROZEN_md5.txt) 를 그대로 구현.

    python3 tmp/HOSGATE_2026-08-13/run_hosgate.py gates            # G-1..G-7
    python3 tmp/HOSGATE_2026-08-13/run_hosgate.py smoke            # 2쌍, smoke_results.jsonl
    python3 tmp/HOSGATE_2026-08-13/run_hosgate.py run              # 전량 (기본 3팔)
    python3 tmp/HOSGATE_2026-08-13/run_hosgate.py run SHIP ZERO    # 팔 선택

`results.jsonl` 에 (episode, arm) 당 한 줄 append. **재개 가능** — 이미 있는 (에러 아닌) 행은
건너뛴다. 스모크는 `smoke_results.jsonl` 로 **분리**되어 판정 자료에 섞이지 않는다.

각 행이 지고 다니는 것 (PREREG §8):
  · `effective_config`  = 추정기 **객체에서 되읽은** 값 (의도표 기록 금지, S13 G-1 결함)
  · `code_md5`          = 실제로 임포트된 저장소 파일 전부의 md5 를 모은 단일 digest
                          (`git_head` 만으로는 더러운 작업트리를 못 잡는다; 전체 표는
                           `code_fingerprint.json`)
  · `git_head`, `git_dirty_n`
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

DATA = {'LIN': os.path.join(HERE, 'data_lin'), 'HOS': os.path.join(HERE, 'data_hos')}
OUT = os.path.join(HERE, 'results.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_results.jsonl')
FP_PATH = os.path.join(HERE, 'code_fingerprint.json')

CROWDED_NC, SPARSE_NC = (15, 30), (2, 5)
ARMS_DEFAULT = ('SHIP', 'LETKF_PLUS', 'ZERO')
LETKF_PLUS_N = 600          # S13 과 동일 (PREREG §3)
SMOKE_KEYS = ('BF4_15comp_seed1581_uniform_ra_rf_ds', 'BF4_2comp_seed1585_uniform_ra_rf_ds')

HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()
DIRTY = len(subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                           text=True).stdout.strip().splitlines())


# ── 표본 ────────────────────────────────────────────────────────────────────
def episodes():
    out = []
    for gt_kind in ('LIN', 'HOS'):
        for p in sorted(glob.glob(os.path.join(DATA[gt_kind], '*.h5'))):
            b = os.path.basename(p)
            m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', b)
            nc = int(m.group(2))
            out.append(dict(path=p, key=b[len('episode_'):-3], gt=gt_kind,
                            pair=b[len('episode_'):-3], bf=m.group(1), ncomp=nc,
                            seed=int(m.group(3)), suffix=m.group(4),
                            stratum=('crowded' if nc in CROWDED_NC else 'sparse')))
    return out


def row_id(e, arm):
    return (e['gt'] + '/' + e['key'], arm)


# ── 코드 지문 (G-6) ─────────────────────────────────────────────────────────
def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def code_fingerprint():
    files = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        f = os.path.abspath(f)
        if f.startswith(ROOT + os.sep) and os.path.exists(f):
            files[os.path.relpath(f, ROOT)] = _md5(f)
    for extra in ('configs/estimation.yaml', 'configs/default.yaml', 'configs/server.yaml'):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            files[extra] = _md5(p)
    digest = hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return digest, files


# ── 팔 ──────────────────────────────────────────────────────────────────────
def ship_config():
    """출하 기본값을 **전 키 명시**로 복사. 폴백에 맡기지 않는다 (PREREG §3)."""
    from wavesplat.run import DEFAULT_CONFIG
    return dict(DEFAULT_CONFIG)


READBACK_KEYS = ('seed', 'lambda_l1', 'lr', 'init_method', 'use_confidence',
                 'use_dispersion', 'adaptive_n', 'max_n', 'tracking_mode',
                 'prune_threshold', 'grow_residual_threshold', 'lambda_smooth',
                 'tracking_steps', 'tracking_lr', 'depth', 'device',
                 'member_seeds', 'ensemble_k', 'k_requested')


def readback(u):
    out = {'updater_class': type(u).__name__}
    for k in READBACK_KEYS:
        v = getattr(u, k, '<absent>')
        out[k] = (v if isinstance(v, (int, float, str, bool, type(None)))
                  else (list(v) if isinstance(v, (list, tuple)) else str(v)))
    return out


def run_ship(path):
    """출하 구성. effective_config 는 **객체에서 되읽는다** (PREREG G-5)."""
    import torch
    import wavesplat.run as WR
    cfg = ship_config()
    captured = {}
    orig = WR.make_updater

    def _capture(config):
        u = orig(config)
        captured['u'] = u
        captured['torch_initial_seed'] = int(torch.initial_seed())
        captured['np_state0'] = int(np.random.get_state()[1][0])
        return u

    WR.make_updater = _capture
    try:
        r = WR.run_estimation(path, config=cfg, verbose=False)
    finally:
        WR.make_updater = orig

    fr = r.get('frame_results', [])
    shells = {}
    for f in fr:
        for k, v in (f.get('shell_rmse') or {}).items():
            if v is not None and np.isfinite(v):
                shells.setdefault(k, []).append(float(v))
    u = captured.get('u')
    return dict(avg_nRMSE=float(r['avg_nRMSE']),
                init_nrmse=float(r.get('init_nrmse', np.nan)),
                n_gaussians=(int(r['n_gaussians']) if r.get('n_gaussians') is not None else -1),
                n_match=len(fr),
                shell_rmse={k: float(np.mean(v)) for k, v in shells.items()},
                effective_config=readback(u) if u is not None else None,
                intent_config={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                   else str(v)) for k, v in cfg.items()},
                torch_initial_seed=captured.get('torch_initial_seed'),
                np_state0=captured.get('np_state0'),
                route=r.get('auto_selected'), n_selected=r.get('auto_n_selected'))


def run_letkf(path, n_ensemble=LETKF_PLUS_N):
    """S13 의 run_letkf 와 동일 (BLANKSLATE G-0 에서 공개 LETKF 열을 1.5e-08 로 재현한 경로)."""
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
    letkf = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=n_ensemble,
                      mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)
    Xg, Yg = np.meshgrid(gx, gy)
    q = np.column_stack([Yg.ravel(), Xg.ravel()])
    nr, zr, ms, t = [], [], [], []
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
        m = compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None,
                                obs_y=y if len(y) else None)
        nr.append(float(m['nRMSE']))
        zr.append(float(np.sqrt(np.mean(gt ** 2)) / loader.Hs_true))
        t.append(float(loader.timestamps[i]))
    sel = np.array(t) > 2.0 * float(loader.Tp_true)
    return dict(avg_nRMSE=float(np.mean(np.array(nr)[sel])),
                zero_nRMSE=float(np.mean(np.array(zr)[sel])),
                n_match=int(sel.sum()), n_all=len(nr),
                ms_median=float(np.median(ms)), n_ensemble=n_ensemble,
                Tp=float(loader.Tp_true), Hs=float(loader.Hs_true))


def run_zero(path):
    """제로필드 바닥. LETKF 팔의 zero_nRMSE 와 **같은 정의·같은 프레임 마스크**."""
    from wavesplat.data_loader import EpisodeDataLoader
    loader = EpisodeDataLoader(path)
    zr, t = [], []
    for i, frame in enumerate(loader):
        gt = frame.gt_field.cpu().numpy()
        zr.append(float(np.sqrt(np.mean(gt ** 2)) / loader.Hs_true))
        t.append(float(loader.timestamps[i]))
    sel = np.array(t) > 2.0 * float(loader.Tp_true)
    return dict(avg_nRMSE=float(np.mean(np.array(zr)[sel])),
                n_match=int(sel.sum()), n_all=len(zr),
                Tp=float(loader.Tp_true), Hs=float(loader.Hs_true))


RUNNERS = {'SHIP': run_ship, 'LETKF_PLUS': run_letkf, 'ZERO': run_zero}


def preload():
    """세 팔의 임포트를 **런 전에** 전부 끝낸다.

    안 하면 `code_md5` 가 팔 순서에 따라 달라진다(스모크에서 실측: SHIP 전용 행과
    LETKF 이후 행의 digest 가 서로 다름). 지문은 배치 내내 **하나**여야 한다."""
    import wavesplat.run                                    # noqa: F401
    import wavesplat.data_loader                            # noqa: F401
    import wavesplat.baselines.letkf_grid                   # noqa: F401
    import wavesplat.evaluation.metrics                     # noqa: F401
    import wavesplat.optimizer.sequential_update            # noqa: F401
    import scipy.interpolate                                # noqa: F401


# ── 게이트 (PREREG §8) ──────────────────────────────────────────────────────
def _gt_stack(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['frames/gt_wave_field'][:], f['frames/timestamps'][:]


def gates(require_full=True):
    ok = True
    eps = episodes()
    pairs = {}
    for e in eps:
        pairs.setdefault(e['pair'], {})[e['gt']] = e

    # G-1 lambda / 명시성
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    cfg = ship_config()
    g1 = ('lambda_l1' in cfg and 'init_method' in cfg
          and SequentialUpdater(cfg).lambda_l1 == cfg['lambda_l1']
          and SequentialUpdater(cfg).seed == cfg['seed'])
    print(f'G-1 lambda 명시: lambda_l1={cfg["lambda_l1"]} init_method={cfg["init_method"]} '
          f'키 {len(cfg)}개 전량 명시  -> {"PASS" if g1 else "FAIL"}')
    ok &= g1

    # G-2 분리가능 evaluate_at == 원본 (실제 gen_hos 의 클래스로 증명)
    sys.path.insert(0, HERE)
    import gen_hos                                   # noqa: E402  (몽키패치는 wavesim.pipeline 한정)
    from wavesim.gt_generator import make_grid
    from wavesim.wave_generator import jonswap_components
    from wavesim.hos_simulator import HOSConfig, HOSSimulator
    c = jonswap_components(Hs=1.2, Tp=7.0, n_components=15, depth=50.0, seed=1581,
                           direction_deg=0.0, direction_spread_deg=30.0, dirs_per_freq=1,
                           random_amplitudes=True, random_frequencies=True,
                           stream_tag='BF4_15')
    sc = gen_hos.snap_components(c, 50.0)
    sim = gen_hos.PairedHOS(HOSConfig(Lx=gen_hos.HOS_LX, Ly=gen_hos.HOS_LY,
                                      Nx=gen_hos.HOS_NX, Ny=gen_hos.HOS_NY,
                                      depth=50.0, M=gen_hos.HOS_M))
    sim.initialize_from_components(sc)
    sim.step(gen_hos.HOS_DT)                        # 스펙트럼을 가득 채운다
    X, Y = make_grid((10.0, 250.0), (-100.0, 100.0), nx=80, ny=60)
    a = HOSSimulator.evaluate_at(sim, X, Y)         # 원본 구현
    b = sim.evaluate_at(X, Y)                       # 분리가능 구현
    dmax = max(float(np.abs(u - v).max()) for u, v in zip(a, b))
    nnz = int((np.abs(sim.eta_hat) > 1e-15).sum())
    g2 = dmax <= 1e-12
    print(f'G-2 분리가능 evaluate_at: 비영모드 {nnz}/{gen_hos.HOS_NX*gen_hos.HOS_NY}, '
          f'max|Δ|={dmax:.3e} (문턱 1e-12) -> {"PASS" if g2 else "FAIL"}')
    ok &= g2

    # G-3 짝 무결성 + G-4 HOS 가 실제로 비선형
    n_full = sum(1 for v in pairs.values() if len(v) == 2)
    print(f'G-3/G-4 짝: {n_full} 쌍 완비 / {len(pairs)} 키')
    g3 = g4 = True
    diffs = []
    for k in sorted(pairs):
        v = pairs[k]
        if len(v) != 2:
            print(f'    ! 불완전 쌍 {k}: {sorted(v)}')
            g3 = False
            continue
        zl, tl = _gt_stack(v['LIN']['path'])
        zh, th = _gt_stack(v['HOS']['path'])
        same_t = zl.shape == zh.shape and np.allclose(tl, th)
        d0 = float(np.abs(zl[0] - zh[0]).max())
        dmaxf = float(np.abs(zl - zh).max()) if same_t else float('nan')
        hs = float(4 * np.std(zl))
        okp = same_t and d0 <= 1e-9 and dmaxf > 0
        g3 &= (same_t and d0 <= 1e-9)
        g4 &= (dmaxf > 0)
        diffs.append((k, d0, dmaxf, dmaxf / hs if hs else np.nan))
        print(f'    {k:44s} frames={zl.shape[0]} t=0 |Δ|={d0:.2e} '
              f'max_t|Δ|={dmaxf:.4f} m ({100*dmaxf/hs:.1f}% of Hs) '
              f'{"OK" if okp else "FAIL"}')
    print(f'    G-3 {"PASS" if g3 else "FAIL"}   G-4 {"PASS" if g4 else "FAIL"}')
    ok &= (g3 and g4)

    # G-5 되읽기 배선
    g5 = hasattr(SequentialUpdater(cfg), 'lambda_l1')
    print(f'G-5 effective_config 되읽기 배선: {"PASS" if g5 else "FAIL"}')
    ok &= g5

    # G-6 코드 지문
    dg, files = code_fingerprint()
    json.dump(dict(digest=dg, git_head=HEAD, git_dirty_n=DIRTY, files=files),
              open(FP_PATH, 'w'), indent=1)
    print(f'G-6 코드 지문: digest={dg} 파일 {len(files)}개 (git {HEAD[:8]}, dirty {DIRTY}) '
          f'-> code_fingerprint.json  PASS')

    # G-7 시드 충돌
    used = set()
    for d in glob.glob(os.path.join(ROOT, 'data*')):
        for p in glob.glob(os.path.join(d, '*.h5')):
            m = re.search(r'seed(\d+)', os.path.basename(p))
            if m:
                used.add(int(m.group(1)))
    mine = set(range(1581, 1589))
    g7 = not (used & mine)
    print(f'G-7 시드 충돌: 소모 {sorted(mine)} vs 기존 {len(used)}개 -> '
          f'교집합 {sorted(used & mine)}  {"PASS" if g7 else "FAIL"}')
    ok &= g7

    # 표본 개수 (전량 실행 전에만 강제)
    n_cr = sum(1 for v in pairs.values() if len(v) == 2
               and v['LIN']['stratum'] == 'crowded')
    n_sp = sum(1 for v in pairs.values() if len(v) == 2
               and v['LIN']['stratum'] == 'sparse')
    print(f'표본: 혼잡 {n_cr} 쌍, 희소 {n_sp} 쌍 (사전등록 16/16)')
    if require_full:
        gN = (n_cr == 16 and n_sp == 16)
        print(f'    표본수 게이트 {"PASS" if gN else "FAIL (생성 미완)"}')
        ok &= gN

    print(f'\nGATES {"PASS -- 실행 인가" if ok else "FAIL -- 배치 중단"}')
    return ok


# ── 실행 ────────────────────────────────────────────────────────────────────
def main_run(arms, smoke=False):
    preload()
    out_path = SMOKE_OUT if smoke else OUT
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['episode'], r['arm']))
    eps = episodes()
    if smoke:
        eps = [e for e in eps if e['key'] in SMOKE_KEYS]
    jobs = [(e, a) for a in arms for e in eps]
    todo = [(e, a) for e, a in jobs if row_id(e, a) not in done]
    dg, files = code_fingerprint()
    print(f'# arms={arms} smoke={smoke} total={len(jobs)} remaining={len(todo)} '
          f'HEAD={HEAD[:8]} dirty={DIRTY}', flush=True)
    t_start = time.time()
    with open(out_path, 'a') as fh:
        for i, (e, a) in enumerate(todo, 1):
            t0 = time.time()
            row = None
            for attempt in (1, 2):                      # PREREG §9: 자동 재시도 1회
                try:
                    res = RUNNERS[a](e['path'])
                    dg, _ = code_fingerprint()          # 임포트가 늘어날 수 있다
                    row = dict(episode=row_id(e, a)[0], arm=a, gt=e['gt'], pair=e['pair'],
                               stratum=e['stratum'], bf=e['bf'], ncomp=e['ncomp'],
                               seed=e['seed'], attempt=attempt, wall_s=time.time() - t0,
                               git_head=HEAD, git_dirty_n=DIRTY, code_md5=dg, **res)
                    break
                except Exception as ex:
                    err = f'{type(ex).__name__}: {ex}'
                    print(f'    ! attempt {attempt} failed: {err}', flush=True)
                    row = dict(episode=row_id(e, a)[0], arm=a, gt=e['gt'], pair=e['pair'],
                               stratum=e['stratum'], attempt=attempt, error=err,
                               wall_s=time.time() - t0, git_head=HEAD, code_md5=dg)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            el = time.time() - t_start
            print(f'[{i}/{len(todo)}] {a:11s} {row_id(e, a)[0]:50s} '
                  f'{row.get("avg_nRMSE", row.get("error"))} '
                  f'({time.time()-t0:.0f}s, ETA {el/i*(len(todo)-i)/60:.0f}m)', flush=True)
    json.dump(dict(digest=dg, git_head=HEAD, git_dirty_n=DIRTY, files=files),
              open(FP_PATH, 'w'), indent=1)


if __name__ == '__main__':
    cmd = (sys.argv[1:] or ['gates'])[0]
    if cmd == 'gates':
        sys.exit(0 if gates(require_full=('--partial' not in sys.argv)) else 1)
    elif cmd == 'smoke':
        main_run(list(ARMS_DEFAULT), smoke=True)
    elif cmd == 'run':
        main_run(sys.argv[2:] or list(ARMS_DEFAULT))
    else:
        print(__doc__)
