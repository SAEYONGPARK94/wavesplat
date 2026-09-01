#!/usr/bin/env python3
"""`S-ENS` 러너 — 합성에서 **배포 코드 경로 자체**의 앙상블 이득을 잰다.

⚠**스레드 고정은 `import torch` 보다 먼저.** 규칙·집계·게이트는 `gates_sens` 단일출처.
⛔이 배치는 **실측을 쓰지 않는다** ⟹ 승인 가드 불필요(합성·기본값 무변경·패키지 무수정).

★`W-15` 와 다른 점: 팔이 `learned_ensemble_k` 하나만 다르고, 그 노브는
**패키지가 소유한 경로**를 켠다(`sequential_update._effective_k`). ⟹ 행의 `ensemble_k` 가
의미 있는 되읽기이고 `G-ENSK` 가 그것을 전건 검사한다.
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
W15 = os.path.abspath(os.path.join(HERE, '..', 'W15_DEPLOY_VERIFY_2026-08-22'))
T5 = os.path.abspath(os.path.join(HERE, '..', 'T5_ENS_2026-09-01'))
for _p in (HERE, W15, T5, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_sens as G  # noqa: E402

_PATHS = {}
for _p in glob.glob(os.path.join(ROOT, '**', 'episode_*.h5'), recursive=True):
    _PATHS.setdefault(os.path.basename(_p)[len('episode_'):-3], _p)

SMOKE = None   # main() 이 채운다 — ★가장 싼 셀 + 게이트가 발화할 셀


def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       text=True).strip()[:12]
    except Exception:
        return None


def _code_md5():
    out = {}
    for rel in ('wavesplat/optimizer/sequential_update.py',
                'wavesplat/optimizer/learned_init.py', 'wavesplat/run.py'):
        p = os.path.join(ROOT, rel)
        out[os.path.basename(rel)] = hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
    out['gates_sens.py'] = hashlib.md5(
        open(os.path.join(HERE, 'gates_sens.py'), 'rb').read()).hexdigest()[:12]
    return out


def run_one(arm, episode):
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    from wavesplat.optimizer.sequential_update import DEFAULT_LEARNED_ENSEMBLE_K
    import gates_t5  # noqa: F401  (배포 핀 재사용)

    t0 = time.time()
    base = dict(batch=G.BATCH, arm=arm, episode=episode,
                stratum=G.STRATUM.get(episode), seed=G.SEED)
    try:
        gates_t5.assert_shipping_defaults()
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(G.arm_config(arm))
        r = run_estimation(_PATHS[episode], cfg)
        return {**base,
                'avg_nRMSE': float(r['avg_nRMSE']),
                'auto_selected': r.get('auto_selected'),
                'auto_n_selected': r.get('auto_n_selected'),
                'ensemble_k': r.get('ensemble_k'),
                'member_n': r.get('member_n'),
                'member_elapsed_s': r.get('member_elapsed_s'),
                'init_time_s': float(r.get('init_time_s', float('nan'))),
                'varpro_fired': bool(r.get('varpro_fired')),
                'init_iters': r.get('init_iters'),
                'avg_RMSE_near': r.get('avg_RMSE_near'),
                'avg_shell_rmse': r.get('avg_shell_rmse'),
                'learned_ensemble_k_cfg': cfg.get('learned_ensemble_k'),
                'learned_ensemble_k_default': DEFAULT_LEARNED_ENSEMBLE_K,
                'deploy_pin_ok': True,
                'omp_threads': int(os.environ.get('OMP_NUM_THREADS', '0')),
                'code_md5': _code_md5(), 'git_head': _git_head(),
                'wall_s': round(time.time() - t0, 3)}
    except Exception as e:                                   # noqa: BLE001
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1200:],
                'wall_s': round(time.time() - t0, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    a = ap.parse_args()

    eps = G.cells()
    if a.smoke:
        # ★가장 싼 셀(혼잡 1) + 층이 다른 셀(희소 1) — 두 팔 모두, 앵커가 걸리는 셀이다.
        crowded = [e for e in eps if G.STRATUM.get(e) == 'crowded'][:1]
        sparse = [e for e in eps if G.STRATUM.get(e) == 'sparse'][:1]
        eps = crowded + sparse
        out = os.path.join(HERE, 'smoke_sens.jsonl')
    else:
        eps = [e for i, e in enumerate(eps) if i % a.nshard == a.shard]
        out = os.path.join(HERE, f'results_sens_{a.shard}.jsonl')

    done = set()
    if os.path.exists(out):
        for ln in open(out):
            if ln.strip():
                r = json.loads(ln)
                done.add((r.get('episode'), r.get('arm')))

    miss = [e for e in eps if e not in _PATHS]
    if miss:
        print(f'⛔에피소드 파일 없음 {len(miss)}건: {miss[:3]}')
        return 2

    n = 0
    with open(out, 'a') as fh:
        for ep in eps:
            for arm in G.ARM_ORDER:
                if (ep, arm) in done:
                    continue
                t = time.time()
                row = run_one(arm, ep)
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                fh.flush()
                n += 1
                print(f'  {arm:9s} {ep[:34]:34s} {time.time()-t:6.1f}s '
                      f'k={row.get("ensemble_k")} ok={"error" not in row}', flush=True)
    print(f'✅ {n} 행 → {os.path.relpath(out, ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
