"""확산 교차설계 평가 — WS(median-of-3) 와 LETKF 를 20 에피소드에 **같은 런**으로.

사전등록: tmp/spread_xdesign_2026-08-05/PREREG.md
프로토콜 출처: WS = tmp/rerun_2026-07-25/batchC_ws_phasefix.py,
              LETKF = tmp/rerun_2026-07-25/batchA_baselines.py (스톡, seed 42)
⛔ R1 교훈: 정확도와 타이밍이 다른 런이면 짝이 아니다 ⟹ 두 팔 모두 여기서 새로 돈다.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

CTRL_DIR = os.path.join(REPO, 'data_finemesh_2026-07-23')
TREAT_DIR = os.path.join(REPO, 'data_spread60_2026-08-05')
PROGRESS = os.path.join(HERE, 'progress.jsonl')
OUT = os.path.join(HERE, 'xdesign.json')

NCOMPS = [5, 30]
SEEDS = [0, 1, 2, 3, 4]
EST_SEEDS = [42, 123, 456]
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']

WS_CFG = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
          'adaptive_n': True, 'tracking_window': 3}


def cells():
    for nc in NCOMPS:
        for sd in SEEDS:
            fname = f'episode_BF4_{nc}comp_seed{sd}_uniform.h5'
            yield ('ctrl30', nc, sd, os.path.join(CTRL_DIR, fname))
            yield ('treat60', nc, sd, os.path.join(TREAT_DIR, fname))


def log(rec):
    with open(PROGRESS, 'a') as f:
        f.write(json.dumps(rec, default=float) + '\n')
    print(f"  {rec.get('tag')} nRMSE={rec.get('nRMSE')} ({rec.get('wall_s', 0):.0f}s)", flush=True)


def run_ws(path, seed, tag):
    from wavesplat.run import run_estimation
    import wavesplat.optimizer.sequential_update as _su
    _su.TRACK_PHASE_PREDICT = False          # 현행 기본값을 **명시**
    t0 = time.time()
    try:
        r = run_estimation(path, {**WS_CFG, 'seed': seed}, verbose=False)
        frames = r['frame_results']
        rec = {k: float(np.nanmean([f[k] for f in frames if k in f])) for k in PANEL}
        rec['elapsed_ms'] = float(np.mean([f['elapsed_ms'] for f in frames])) if frames else np.nan
        rec['N'] = int(r.get('n_gaussians', -1) or -1)
    except Exception as e:
        rec = {'error': f'{type(e).__name__}: {e}'}
    rec.update(tag=tag, seed=seed, wall_s=time.time() - t0)
    log(rec)
    return rec


def run_letkf(path, tag):
    from scipy.interpolate import RegularGridInterpolator
    from wavesplat.baselines.letkf_grid import LETKFGrid
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.evaluation.metrics import compute_all_metrics
    t0 = time.time()
    try:
        np.random.seed(42)                    # batchA 프로토콜 그대로
        loader = EpisodeDataLoader(path)
        gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
        egx = np.linspace(gx[0], gx[-1], 40); egy = np.linspace(gy[0], gy[-1], 30)
        b = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=100,
                      mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)
        fm = []
        Xg, Yg = np.meshgrid(gx, gy)
        for frame in loader:
            x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy()
            z = frame.z_obs.cpu().numpy()
            el = b.update(x, y, z)
            interp = RegularGridInterpolator((egy, egx), b.predict_grid(), method='linear',
                                             bounds_error=False, fill_value=0.0)
            pf = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(len(gy), len(gx))
            m = compute_all_metrics(pf, frame.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                    obs_x=x if len(x) else None, obs_y=y if len(y) else None)
            m['elapsed_ms'] = el
            fm.append(m)
        rec = {k: float(np.nanmean([m.get(k, np.nan) for m in fm])) for k in PANEL}
        rec['elapsed_ms'] = float(np.nanmean([m.get('elapsed_ms', np.nan) for m in fm]))
    except Exception as e:
        rec = {'error': f'{type(e).__name__}: {e}'}
    rec.update(tag=tag, wall_s=time.time() - t0)
    log(rec)
    return rec


def med(rs, key):
    v = [r[key] for r in rs if 'error' not in r and r.get(key) is not None
         and np.isfinite(r[key])]
    if not v:
        return None
    return {'median': float(np.median(v)), 'mean': float(np.mean(v)),
            'std': float(np.std(v)), 'n_ok': len(v)}


def main():
    todo = [c for c in cells() if os.path.exists(c[3])]
    missing = [c for c in cells() if not os.path.exists(c[3])]
    if missing:
        print('★ 없는 에피소드 (생성 먼저):')
        for c in missing:
            print('   ', c[3])
        return
    out = {'_prereg': 'tmp/spread_xdesign_2026-08-05/PREREG.md',
           '_est_seeds': EST_SEEDS, '_ws_cfg': WS_CFG, 'cells': {}}
    t_start = time.time()
    for arm, nc, sd, path in todo:
        key = f'{arm}|{nc}comp|seed{sd}'
        print(f'[{key}]', flush=True)
        ws = [run_ws(path, s, f'{key}|WS|s{s}') for s in EST_SEEDS]
        lk = run_letkf(path, f'{key}|LETKF')
        out['cells'][key] = {
            'arm': arm, 'n_comp': nc, 'gen_seed': sd, 'path': os.path.relpath(path, REPO),
            'WS': {k: med(ws, k) for k in PANEL + ['elapsed_ms', 'N']},
            'WS_seed_results': ws,
            'LETKF': {k: lk.get(k) for k in PANEL + ['elapsed_ms']},
            'LETKF_raw': lk,
        }
        json.dump(out, open(OUT, 'w'), indent=2, default=float)
        print(f'  done ({time.time() - t_start:.0f}s elapsed)', flush=True)
    out['_wall_s'] = time.time() - t_start
    json.dump(out, open(OUT, 'w'), indent=2, default=float)
    print(f'\nALL DONE {out["_wall_s"]:.0f}s -> {OUT}')


if __name__ == '__main__':
    main()
