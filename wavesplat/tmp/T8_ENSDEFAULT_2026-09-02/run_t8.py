#!/usr/bin/env python3
"""`T8-ENSDEFAULT` 러너.

⚠**스레드 고정은 `import torch` 보다 먼저.** 규칙·집계·게이트는 `gates_t8` 단일출처.
⛔**단일 프로세스·1 스레드·직렬** — 정확도와 **시간**을 같은 행에서 재기 때문이다
  (`T6` 와 같은 이유). 샤딩하지 않는다.

모드
----
    --anchor    앵커 4 에피소드 x M1 x 시드 1626  (동결 `S-ENS` 와 비트 대조)
    --smoke     신선 1 에피소드 x 두 팔 x 시드 1개 (비용·게이트 발화 확인)
    (기본)      신선 64 x 두 팔 x 시드 3
    --timing    신선 8 x 두 팔 — `T6` 방식의 **프레임당** 계측 (SEC-B)
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import argparse    # noqa: E402
import glob        # noqa: E402
import hashlib     # noqa: E402
import json        # noqa: E402
import subprocess  # noqa: E402
import time        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
T5 = os.path.abspath(os.path.join(HERE, '..', 'T5_ENS_2026-09-01'))
for _p in (HERE, T5, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_t8 as G  # noqa: E402

_PATHS = {}
for _p in glob.glob(os.path.join(ROOT, '**', 'episode_*.h5'), recursive=True):
    _PATHS.setdefault(os.path.basename(_p)[len('episode_'):-3], _p)


def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       text=True).strip()[:12]
    except Exception:                                        # noqa: BLE001
        return None


def _code_md5():
    out = {}
    for rel in ('wavesplat/optimizer/sequential_update.py',
                'wavesplat/optimizer/learned_init.py', 'wavesplat/run.py'):
        out[os.path.basename(rel)] = hashlib.md5(
            open(os.path.join(ROOT, rel), 'rb').read()).hexdigest()[:12]
    out['gates_t8.py'] = hashlib.md5(
        open(os.path.join(HERE, 'gates_t8.py'), 'rb').read()).hexdigest()[:12]
    return out


def run_one(arm, episode, seed):
    import torch
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    from wavesplat.optimizer.sequential_update import DEFAULT_LEARNED_ENSEMBLE_K
    import gates_t5  # noqa: F401  (배포 핀 재사용)

    t0 = time.time()
    base = dict(batch=G.BATCH, arm=arm, episode=episode, seed=int(seed))
    try:
        gates_t5.assert_shipping_defaults()
        # ⛔`{}` 로 팔을 정의하지 않는다 — 배포 기본값에서 시작하고 한 키만 바꾼다.
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(G.arm_config(arm, seed))
        torch.manual_seed(int(seed))
        r = run_estimation(_PATHS[episode], cfg)
        return {**base,
                'avg_nRMSE': float(r['avg_nRMSE']),
                'auto_selected': r.get('auto_selected'),
                'auto_n_selected': r.get('auto_n_selected'),
                'ensemble_k': r.get('ensemble_k'),
                'init_time_s': float(r.get('init_time_s', float('nan'))),
                'varpro_fired': bool(r.get('varpro_fired')),
                'avg_RMSE_near': r.get('avg_RMSE_near'),
                'learned_ensemble_k_cfg': cfg.get('learned_ensemble_k'),
                'learned_ensemble_k_default': DEFAULT_LEARNED_ENSEMBLE_K,
                'deploy_pin_ok': True,
                'threads': torch.get_num_threads(),
                'pid': os.getpid(),
                'omp_threads': int(os.environ.get('OMP_NUM_THREADS', '0')),
                'code_md5': _code_md5(), 'git_head': _git_head(),
                'wall_s': round(time.time() - t0, 3)}
    except Exception as e:                                   # noqa: BLE001
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1200:],
                'wall_s': round(time.time() - t0, 3)}


def run_timing(arm, episode, seed):
    """`T6` 방식 — 초기화/프레임을 **직접** 계측한다(SEC-B)."""
    import torch
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.run import DEFAULT_CONFIG
    from wavesplat.optimizer.sequential_update import make_updater

    loader = EpisodeDataLoader(_PATHS[episode])
    init_w = loader.get_init_window(tp_factor=2.0)
    track = loader.get_tracking_frames(tp_factor=2.0)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(G.arm_config(arm, seed))
    cfg['depth'] = float(loader.depth)
    torch.manual_seed(int(seed))
    upd = make_updater(cfg)
    t0 = time.perf_counter()
    upd.initialize(init_w)
    init_s = time.perf_counter() - t0
    prim = getattr(upd, '_primary', upd)
    fms = []
    for fr in track:
        a = time.perf_counter()
        upd.update(fr)
        fms.append((time.perf_counter() - a) * 1e3)
    return dict(batch=G.BATCH, arm=arm, episode=episode, seed=int(seed),
                init_s=init_s, frame_ms=fms, n_frames=len(fms),
                route_effective=(getattr(prim, '_auto_selected', None)
                                 or prim.init_method),
                threads=torch.get_num_threads(), pid=os.getpid(),
                cuda_available=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--timing', action='store_true')
    a = ap.parse_args()

    if a.anchor:
        jobs = [('M1', e, G.ANCHOR_SEED) for e in G.ANCHOR_EPISODES]
        out = os.path.join(HERE, 'anchor_t8.jsonl')
    elif a.smoke:
        jobs = [(arm, G.EPISODES[0], G.EST_SEEDS[0]) for arm in G.ARM_ORDER]
        out = os.path.join(HERE, 'smoke_t8.jsonl')
    elif a.timing:
        eps = G.EPISODES[:2] + G.EPISODES[16:18] + G.EPISODES[32:34] + G.EPISODES[48:50]
        jobs = [(arm, e, G.EST_SEEDS[0]) for e in eps for arm in G.ARM_ORDER]
        out = os.path.join(HERE, 'timing_t8.jsonl')
    else:
        jobs = [(arm, e, s) for e in G.EPISODES for s in G.EST_SEEDS
                for arm in G.ARM_ORDER]
        out = os.path.join(HERE, 'results_t8.jsonl')

    done = set()
    if os.path.exists(out):
        for ln in open(out):
            if ln.strip():
                r = json.loads(ln)
                done.add((r.get('arm'), r.get('episode'), r.get('seed')))

    miss = sorted({e for _, e, _ in jobs if e not in _PATHS})
    if miss:
        print(f'⛔에피소드 파일 없음 {len(miss)}건: {miss[:3]}')
        return 2

    n = 0
    with open(out, 'a') as fh:
        for arm, ep, sd in jobs:
            if (arm, ep, sd) in done:
                continue
            t = time.time()
            row = run_timing(arm, ep, sd) if a.timing else run_one(arm, ep, sd)
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            n += 1
            extra = (f"frame_med={sorted(row['frame_ms'])[len(row['frame_ms'])//2]:.2f}ms"
                     if a.timing else f"k={row.get('ensemble_k')}")
            print(f'  {arm:3s} s{sd} {ep[:38]:38s} {time.time()-t:6.1f}s {extra} '
                  f'ok={"error" not in row}', flush=True)
    print(f'✅ {n} 행 → {os.path.relpath(out, ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
