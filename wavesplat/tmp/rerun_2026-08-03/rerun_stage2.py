"""논문 수치 전파 2단계 — 절제 / 적응형-N(p1) / raydrop 재실행.

토론 ① C2·C4 준수:
  - init_method 를 모든 config 에 **명시** (W9 오염 사고 방지)
  - data_finemesh_2026-07-23/ 사용 (구 data/ 는 메시-Nyquist 위반)
  - **median-of-3 + 산포** 보고 (WS 시드분산 bimodal, CV~53%)
  - 베이스라인(ETKF 등)은 **재실행하지 않고 재사용** — 트래커 수정은 WS 팔에만 영향
    (근거: docs/PAPER_NUMBER_PROPAGATION_AUDIT_2026-08-01.md §2)

산출:
  ablation_seeded_rerun.json   (기존 results/ablation_seeded.json 과 동일 스키마)
  p1_comparison_rerun.json     (WS-P0/WS-P1 만 갱신, 나머지 키는 구본에서 승계)
  raydrop_comparison_rerun.json(WS-P1 만 갱신, ETKF 는 구본에서 승계)
  progress.jsonl               (한 런마다 append — 중단돼도 부분결과 보존)
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
OLD = os.path.join(ROOT, 'results')
SEEDS = [42, 123, 456]
PROGRESS = os.path.join(HERE, 'progress.jsonl')


def ep_path(name):
    return os.path.join(DATA, f'episode_{name}.h5')


def log(rec):
    with open(PROGRESS, 'a') as f:
        f.write(json.dumps(rec, default=float) + '\n')
    print(f"  {rec.get('tag')} seed={rec.get('seed')} "
          f"nRMSE={rec.get('avg_nRMSE')} ({rec.get('wall_s', 0):.1f}s)", flush=True)


def run_seeds(name, cfg, tag):
    """run_estimation over SEEDS; returns per-seed dicts (errors kept as dicts)."""
    out = []
    for seed in SEEDS:
        t0 = time.time()
        try:
            r = run_estimation(ep_path(name), {**cfg, 'seed': seed}, verbose=False)
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
        except Exception as e:  # keep going; a hole is recorded, not hidden
            rec = {'seed': seed, 'error': f'{type(e).__name__}: {e}'}
        rec['wall_s'] = time.time() - t0
        rec['tag'] = tag
        rec['episode'] = name
        log(rec)
        out.append(rec)
    return out


def med_spread(seed_results, key='avg_nRMSE'):
    v = [s[key] for s in seed_results if 'error' not in s and s.get(key) is not None]
    if not v:
        return None
    return {
        'median': float(np.median(v)),
        'mean': float(np.mean(v)),
        'std': float(np.std(v)),
        'min': float(np.min(v)),
        'max': float(np.max(v)),
        'n_ok': len(v),
    }


# ---------------------------------------------------------------- ablation
ABL_EPISODES = [
    'BF3_1comp_seed0_uniform', 'BF3_1comp_seed0_slope',
    'BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope',
    'BF3_5comp_seed0_uniform', 'BF3_5comp_seed0_slope',
]
ABL_CONFIGS = {
    # init_method 는 원본 실험과 동일하게 유지 (비교 가능성) — 단 전부 명시적
    'Full':      {'use_dispersion': True,  'use_confidence': True,  'init_method': 'v1'},
    'A1_noDisp': {'use_dispersion': False, 'use_confidence': True,  'init_method': 'v1'},
    'A2_noConf': {'use_dispersion': True,  'use_confidence': False, 'init_method': 'v1'},
    'v2+disp':   {'use_dispersion': True,  'use_confidence': True,  'init_method': 'v2'},
}


def do_ablation():
    out = []
    for ep in ABL_EPISODES:
        for cfg_name, cfg in ABL_CONFIGS.items():
            sr = run_seeds(ep, cfg, f'abl/{ep}/{cfg_name}')
            row = {'episode': ep, 'config': cfg_name, 'init_method': cfg['init_method'],
                   'seed_results': sr}
            for k in ('avg_nRMSE', 'avg_RMSE_z', 'avg_Hs_error', 'init_nrmse',
                      'avg_inference_ms'):
                row[k] = med_spread(sr, k)
            out.append(row)
            _dump('ablation_seeded_rerun.json', out)
    return out


# ---------------------------------------------------------------- p1 / adaptive N
P1_EPISODES = [
    'BF2_1comp_seed0_uniform', 'BF3_1comp_seed0_uniform',
    'BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope',
    'BF3_5comp_seed0_uniform', 'BF3_5comp_seed0_slope',
    'BF4_1comp_seed0_uniform', 'BF4_5comp_seed0_uniform',
    'BF4_15comp_seed0_uniform', 'BF4_30comp_seed0_uniform',
    'BF5_15comp_seed0_uniform', 'BF5_30comp_seed0_uniform',
]
P1_ARMS = {
    'WS-P0': {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
              'adaptive_n': False, 'tracking_window': 1},
    'WS-P1': {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
              'adaptive_n': True, 'tracking_window': 3},
}


def do_p1():
    old = json.load(open(os.path.join(OLD, 'p1_comparison.json')))
    out = {}
    for ep in P1_EPISODES:
        entry = dict(old.get(ep, {}))          # 베이스라인 키 승계
        entry['_carried_from'] = 'results/p1_comparison.json (2026-04-07)'
        for arm, cfg in P1_ARMS.items():
            sr = run_seeds(ep, cfg, f'p1/{ep}/{arm}')
            entry[arm] = {
                'avg_nRMSE': med_spread(sr, 'avg_nRMSE'),
                'avg_inference_ms': med_spread(sr, 'avg_inference_ms'),
                'final_n': med_spread(sr, 'final_n'),
                'seed_results': sr,
                '_rerun': '2026-08-03 phasefix defaults, median-of-3',
            }
        out[ep] = entry
        _dump('p1_comparison_rerun.json', out)
    return out


# ---------------------------------------------------------------- raydrop
RAY_PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
             ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
             ('BF5', '15comp'), ('BF5', '30comp')]
RAY_CFG = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
           'adaptive_n': True, 'tracking_window': 3}


def do_raydrop():
    old = json.load(open(os.path.join(OLD, 'raydrop_comparison.json')))
    out = {}
    for bf, comp in RAY_PAIRS:
        key = f'{bf}_{comp}'
        entry = {}
        for model in ('uniform', 'slope'):
            base = dict(old.get(key, {}).get(model, {}))   # ETKF 승계
            base.pop('WS-P1', None)
            base['_carried_from'] = 'results/raydrop_comparison.json (2026-04-07)'
            ep = f'{bf}_{comp}_seed0_{model}'
            sr = run_seeds(ep, RAY_CFG, f'ray/{key}/{model}')
            base['WS-P1'] = {
                'avg_nRMSE': med_spread(sr, 'avg_nRMSE'),
                'avg_inference_ms': med_spread(sr, 'avg_inference_ms'),
                'final_n': med_spread(sr, 'final_n'),
                'seed_results': sr,
                '_rerun': '2026-08-03 phasefix defaults, median-of-3',
            }
            entry[model] = base
        out[key] = entry
        _dump('raydrop_comparison_rerun.json', out)
    return out


def _dump(name, obj):
    with open(os.path.join(HERE, name), 'w') as f:
        json.dump(obj, f, indent=2, default=float)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    meta = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'data': DATA,
            'seeds': SEEDS, 'which': which}
    print(json.dumps(meta), flush=True)
    if which in ('all', 'ablation'):
        print('=== ABLATION ===', flush=True)
        do_ablation()
    if which in ('all', 'p1'):
        print('=== P1 / ADAPTIVE-N ===', flush=True)
        do_p1()
    if which in ('all', 'raydrop'):
        print('=== RAYDROP ===', flush=True)
        do_raydrop()
    print(f'DONE total_wall_s={time.time() - t0:.0f}', flush=True)
