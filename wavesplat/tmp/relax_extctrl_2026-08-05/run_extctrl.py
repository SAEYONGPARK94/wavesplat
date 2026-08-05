"""구성요소 수준 외부 대조 — RELAX 초기화기 vs 우리 beamforming.

사전등록: tmp/relax_extctrl_2026-08-05/PREREG.md (런 실행 전 작성)
프로토콜은 tmp/rerun_2026-08-03/rerun_stage2.py 를 그대로 복제:
  - data_finemesh_2026-07-23/
  - SEEDS = [42, 123, 456], median-of-3 + 산포
  - init_method 를 모든 config 에 명시
  - progress.jsonl 에 런마다 append (중단돼도 부분결과 보존)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from wavesplat.run import run_estimation  # noqa: E402

DATA = os.path.join(ROOT, 'data_finemesh_2026-07-23')
SEEDS = [42, 123, 456]
PROGRESS = os.path.join(HERE, 'progress.jsonl')
OUT = os.path.join(HERE, 'relax_extctrl.json')

EPISODES = [
    'BF2_1comp_seed0_uniform', 'BF3_1comp_seed0_uniform',
    'BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope',
    'BF3_5comp_seed0_uniform', 'BF3_5comp_seed0_slope',
    'BF4_1comp_seed0_uniform', 'BF4_5comp_seed0_uniform',
    'BF4_15comp_seed0_uniform', 'BF4_30comp_seed0_uniform',
    'BF5_15comp_seed0_uniform', 'BF5_30comp_seed0_uniform',
]

BASE = {'use_dispersion': True, 'use_confidence': True,
        'adaptive_n': True, 'tracking_window': 3}

ARMS = {
    'OURS':  {**BASE, 'init_method': 'v2_konly'},     # 배포 기본값 = WS-P1
    'EXT-M': {**BASE, 'init_method': 'relax_konly'},  # PRIMARY (w/phi 처리 일치)
    'EXT-F': {**BASE, 'init_method': 'relax'},        # 서술용
}


def log(rec):
    with open(PROGRESS, 'a') as f:
        f.write(json.dumps(rec, default=float) + '\n')
    print(f"  {rec.get('tag')} seed={rec.get('seed')} "
          f"nRMSE={rec.get('avg_nRMSE')} init={rec.get('init_nrmse')} "
          f"({rec.get('wall_s', 0):.1f}s)", flush=True)


def run_seeds(name, cfg, tag):
    out = []
    for seed in SEEDS:
        t0 = time.time()
        try:
            r = run_estimation(os.path.join(DATA, f'episode_{name}.h5'),
                               {**cfg, 'seed': seed}, verbose=False)
            rec = {
                'seed': seed,
                'avg_nRMSE': float(r['avg_nRMSE']),
                'avg_RMSE_z': float(r['avg_RMSE_z']),
                'avg_Hs_error': float(r['avg_Hs_error']),
                'init_nrmse': float(r['init_nrmse']),
                'init_rmse': float(r['init_rmse']),
                'T_converge': r['T_converge'],
                'avg_inference_ms': float(r['avg_inference_ms']),
                'final_n': int(r.get('n_gaussians', 15) or 15),
            }
        except Exception as e:                     # 구멍은 숨기지 않고 기록
            rec = {'seed': seed, 'error': f'{type(e).__name__}: {e}'}
        rec['wall_s'] = time.time() - t0
        rec['tag'] = tag
        rec['episode'] = name
        log(rec)
        out.append(rec)
    return out


def med_spread(seed_results, key):
    v = [s[key] for s in seed_results if 'error' not in s and s.get(key) is not None]
    if not v:
        return None
    return {'median': float(np.median(v)), 'mean': float(np.mean(v)),
            'std': float(np.std(v)), 'min': float(np.min(v)),
            'max': float(np.max(v)), 'n_ok': len(v)}


def main():
    t_start = time.time()
    out = {'_prereg': 'tmp/relax_extctrl_2026-08-05/PREREG.md',
           '_data': 'data_finemesh_2026-07-23',
           '_seeds': SEEDS, '_arms': {k: v['init_method'] for k, v in ARMS.items()},
           'episodes': {}}
    for ep in EPISODES:
        entry = {}
        for arm, cfg in ARMS.items():
            sr = run_seeds(ep, cfg, f'{ep}/{arm}')
            entry[arm] = {
                'init_method': cfg['init_method'],
                'avg_nRMSE': med_spread(sr, 'avg_nRMSE'),
                'init_nrmse': med_spread(sr, 'init_nrmse'),
                'avg_inference_ms': med_spread(sr, 'avg_inference_ms'),
                'final_n': med_spread(sr, 'final_n'),
                'avg_Hs_error': med_spread(sr, 'avg_Hs_error'),
                'seed_results': sr,
            }
        out['episodes'][ep] = entry
        with open(OUT, 'w') as f:
            json.dump(out, f, indent=2, default=float)
        print(f"[done] {ep}  ({time.time() - t_start:.0f}s elapsed)", flush=True)
    out['_wall_s'] = time.time() - t_start
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nALL DONE in {out['_wall_s']:.0f}s -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
