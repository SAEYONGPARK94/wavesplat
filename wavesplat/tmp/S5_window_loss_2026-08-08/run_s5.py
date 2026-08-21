"""S5 driver — 학습창 잔차로 "최소화 실패 vs 창-과적합" 을 가르고, 정규화항을 시험한다.

사전등록 `tmp/S5_window_loss_2026-08-08/PREREG.md` (md5 동결) 실행.
셀 조립·개입은 frozen `run_rx2b.py` 와 S4 배선(`S4_ORACLE_WPHI` 게이트)을 import 해서 쓴다.
S4 와 달리 **한 번만 실행**하고 전 지표를 그 한 번에서 뽑는다(비용 절반).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S4DIR = os.path.abspath(os.path.join(HERE, '..', 'S4_phase_oracle_2026-08-08'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
sys.path.insert(0, REPO)
sys.path.insert(0, RX2)
sys.path.insert(0, S4DIR)

import run_rx2b as R      # noqa: E402  frozen 개입 로직
import run_s4 as S4       # noqa: E402  오라클 (w,phi) 게이트 배선 + 셀

R.DATA = REPO

# S4 가 이미 만든 셀 위에 정규화-제거 팔 하나만 추가한다.
R.CELLS.setdefault('ORA-WPHI-NOREG', dict(cfg='__oracle_k__'))
R.ORACLE_CELLS = set(R.ORACLE_CELLS) | {'ORA-WPHI-NOREG'}
S4.EXTRA['ORA-WPHI-NOREG'] = {'tracking_lr': 0.0, 'lambda_l1': 0.0, 'lambda_smooth': 0.0}
S4.WPHI_CELLS = set(S4.WPHI_CELLS) | {'ORA-WPHI-NOREG'}

# ★출하 팔에서도 정규화 제거가 통하는지 — 이게 없으면 배포에 대해 아무 말도 못 한다 (S1 G1P1 전례)
R.CELLS.setdefault('AUTO-NT-NOREG', dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
R.CELLS.setdefault('AUTO-NT-NOL1', dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
S4.EXTRA['AUTO-NT-NOREG'] = {'tracking_lr': 0.0, 'lambda_l1': 0.0, 'lambda_smooth': 0.0}
S4.EXTRA['AUTO-NT-NOL1'] = {'tracking_lr': 0.0, 'lambda_l1': 0.0}

CELLS = ['ORA-WPHI-0STEP', 'ORA-WPHI-NT', 'ORA-WPHI-NOREG',
         'AUTO-NT', 'AUTO-NT-NOREG', 'AUTO-NT-NOL1', 'AUTO-NT-DUP']
EPISODES = S4.EPISODES
SEEDS = [int(s) for s in os.environ.get('S5_SEEDS', '1560,1561,1562').split(',')]
OUT = os.environ.get('S5_OUT', os.path.join(HERE, 'results.jsonl'))


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES for s in SEEDS]


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ['S5_WINDOW'] = '1'
    os.environ['S4_ORACLE_WPHI'] = '1' if job['cell'] in S4.WPHI_CELLS else '0'
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    undo = None
    try:
        cfg = R.cell_config(job['cell'], job['episode'])
        cfg['seed'] = job['seed']
        path = os.path.join(REPO, job['episode'])
        gt = np.asarray(EpisodeDataLoader(path).wave_components)
        undo = R.apply_cell(job['cell'])
        _init.ORACLE_COMPONENTS = gt if job['cell'] in R.ORACLE_CELLS else None
        try:
            r = run_estimation(path, cfg)
        finally:
            _init.ORACLE_COMPONENTS = None
        fr = r['frame_results']

        def m(k):
            v = [f[k] for f in fr if k in f and np.isfinite(f.get(k, np.nan))]
            return float(np.mean(v)) if v else float('nan')

        vr = np.array([f['Variance_Ratio'] for f in fr if 'Variance_Ratio' in f], float)
        rho = np.array([f['Pearson_rho'] for f in fr if 'Pearson_rho' in f], float)
        with np.errstate(invalid='ignore'):
            decorr = 1.0 - rho ** 2
            shrink = (np.sqrt(np.clip(vr, 0, None)) - rho) ** 2
        return {**base, 'regime': S4.regime(job['episode']),
                'window_resid': float(r.get('window_resid', float('nan'))),
                'window_n_obs': int(r.get('window_n_obs', 0)),
                'all_resid': m('obs_resid_rms'),
                'avg_obs_loss': m('obs_loss'),
                'nRMSE': float(np.nanmean([f['nRMSE'] for f in fr if 'nRMSE' in f])),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'decorr_mean': float(np.nanmean(decorr)) if rho.size else float('nan'),
                'shrink_mean': float(np.nanmean(shrink)) if rho.size else float('nan'),
                'N': r.get('n_gaussians', -1),
                'cfg': {k: v for k, v in cfg.items() if k != 'seed'},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1200:],
                'wall_s': time.time() - t0}
    finally:
        if undo is not None:
            undo()
        os.environ['S4_ORACLE_WPHI'] = '0'


def main():
    import multiprocessing as mp
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = build_jobs(cells)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    n_workers = int(os.environ.get('WORKERS', '8'))
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | '
          f'workers {n_workers} | seeds {SEEDS}', flush=True)
    if not todo:
        print('nothing to do')
        return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=R._init_worker) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            el = time.time() - t0
            tag = (f"win={rec['window_resid']:.5f} all={rec['all_resid']:.5f}"
                   if 'error' not in rec else str(rec['error'])[:60])
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<72s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
