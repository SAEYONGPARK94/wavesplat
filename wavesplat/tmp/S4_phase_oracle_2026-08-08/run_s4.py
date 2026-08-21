"""S4 driver — 위상/진폭 오라클 사다리. 혼잡의 벽이 **탐색 문제인지 표현 문제인지** 가른다.

사전등록 `tmp/S4_phase_oracle_2026-08-08/PREREG.md` (md5 동결) 실행.

사다리 (전부 추적기 OFF — S3 가 추적기는 어느 배치에서도 해롭다고 쟀다):

    AUTO-NT          출하 초기화                                  = 배포 가능 기준선
    ORA-K-NT         오라클 (kx,ky);  (w,phi) 는 일반 초기화       = S3 재현
    ORA-WPHI-NT      오라클 (kx,ky,w,phi), 그 다음 **최적화**      = 신규
    ORA-WPHI-0STEP   오라클 (kx,ky,w,phi), **최적화 0 스텝**       = 신규 · 표현 천장
    AUTO-DUP         워커 누출 프로브

★`ORA-WPHI-0STEP` 이 이 실험의 핵심 대조다. 진실에서 출발해 **아무것도 하지 않은** 필드의
오차가 곧 **표현·관측모델이 허용하는 하한**이다. `ORA-WPHI-NT`(같은 출발점 + 최적화)가
그보다 나쁘면, 그 차이는 **옵티마이저가 진실에서 걸어 나간 양**이다.

개입·지표는 frozen `run_rx2b.py` 를 import 해서 쓴다. `S4_ORACLE_WPHI` 환경 게이트는
**셀마다 항상 명시적으로 덮어쓴다**(이전 잡의 값이 워커에 남지 못하게 — RX2_ENV_KEYS 와 같은 규율).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
sys.path.insert(0, REPO)
sys.path.insert(0, RX2)

import run_rx2b as R  # noqa: E402  frozen

R.DATA = REPO

for name in ('AUTO-NT', 'AUTO-NT-DUP'):
    R.CELLS.setdefault(name, dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
for name in ('ORA-K-NT', 'ORA-WPHI-NT', 'ORA-WPHI-0STEP'):
    R.CELLS.setdefault(name, dict(cfg='__oracle_k__'))
R.ORACLE_CELLS = set(R.ORACLE_CELLS) | {'ORA-K-NT', 'ORA-WPHI-NT', 'ORA-WPHI-0STEP'}

# cfg 덧씌우기 (frozen cell_config 호출 뒤에 적용)
EXTRA = {
    'AUTO-NT':        {'tracking_lr': 0.0},
    'AUTO-NT-DUP':    {'tracking_lr': 0.0},          # AUTO-NT 의 **바이트 복제**
    'ORA-K-NT':       {'tracking_lr': 0.0},
    'ORA-WPHI-NT':    {'tracking_lr': 0.0},
    # 0 스텝: 사다리 1단, 경사하강 0회. 초기화가 곧 최종 모델이다.
    'ORA-WPHI-0STEP': {'tracking_lr': 0.0, 'max_iters_per_stage': [0],
                       'loss_thresholds': [None]},
}
# 진폭/위상 오라클 게이트를 켜는 셀
WPHI_CELLS = {'ORA-WPHI-NT', 'ORA-WPHI-0STEP'}

_frozen_cell_config = R.cell_config


def _cell_config(cell, episode):
    cfg = _frozen_cell_config(cell, episode)
    cfg.update(EXTRA.get(cell, {}))
    assert 'init_method' in cfg
    return cfg


R.cell_config = _cell_config

CELLS = ['AUTO-NT', 'ORA-K-NT', 'ORA-WPHI-NT', 'ORA-WPHI-0STEP', 'AUTO-NT-DUP']

CROWDED_DIR = 'data_indep_crowded_2026-08-08'
SPARSE_DIR = 'data_indep_sparse_2026-08-08'
CROWDED16 = [f'{CROWDED_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in range(20, 24)]
SPARSE16 = [f'{SPARSE_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in range(20, 24)]
EPISODES = CROWDED16 + SPARSE16

SEEDS = [int(s) for s in os.environ.get('S4_SEEDS', '1557,1558,1559').split(',')]
OUT = os.environ.get('S4_OUT', os.path.join(HERE, 'results.jsonl'))


def regime(ep):
    return 'crowded' if ep.startswith(CROWDED_DIR) else 'sparse'


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES for s in SEEDS]


def run_one(job):
    """frozen `R.run_one` 에 **관측 적합도**를 덧붙인다.

    ★왜 필요한가: `ORA-WPHI-0STEP` 은 진실 그 자체이므로 필드 오차가 0 이다(동어반복 —
    `gt_wave_field` 가 `sum A cos(...)` 로 모델과 **같은 식**이다). 진짜 질문은
    "진실에서 출발해 최적화하면 왜 멀어지는가" 이고, 그건 **관측 적합도**로만 갈린다:

      obs_loss(최적화 결과) < obs_loss(진실)  ⟹ **목적함수가 더 나쁜 필드를 선호한다**
                                                (목적함수 오설정 / 식별 불가)
      obs_loss(최적화 결과) > obs_loss(진실)  ⟹ **옵티마이저가 수렴에 실패한다**

    `R.run_one` 은 PANEL 6지표만 평균하고 `obs_loss` 는 버리므로 여기서 따로 받는다.
    개입·설정은 전부 frozen 경로를 그대로 통과한 것이고, 이 함수는 **읽기만** 한다.
    """
    os.environ['S4_ORACLE_WPHI'] = '1' if job['cell'] in WPHI_CELLS else '0'
    try:
        rec = R.run_one(job)
        if 'error' not in rec:
            rec.update(_obs_fit(job))
    finally:
        os.environ['S4_ORACLE_WPHI'] = '0'
    rec['regime'] = regime(job['episode'])
    rec['wphi_oracle'] = job['cell'] in WPHI_CELLS
    return rec


def _obs_fit(job):
    """같은 셀·같은 시드를 한 번 더 돌려 프레임별 `obs_loss` 를 받는다.

    ⚠비용 2배지만 **결정적 지표**다. frozen `run_one` 을 수정하지 않기 위한 대가이고,
    `run_estimation` 은 결정론적이므로 (같은 cfg·같은 시드) 두 실행은 같은 궤적이다
    -- 그 동일성 자체를 `nRMSE_recheck` 로 **검증해서 기록**한다.
    """
    import numpy as np
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init
    undo = None
    try:
        cfg = R.cell_config(job['cell'], job['episode'])
        cfg['seed'] = job['seed']
        path = os.path.join(R.DATA, job['episode'])
        gt = np.asarray(EpisodeDataLoader(path).wave_components)
        undo = R.apply_cell(job['cell'])
        _init.ORACLE_COMPONENTS = gt if job['cell'] in R.ORACLE_CELLS else None
        try:
            r = run_estimation(path, cfg)
        finally:
            _init.ORACLE_COMPONENTS = None
        fr = r['frame_results']
        ol = [f['obs_loss'] for f in fr if 'obs_loss' in f
              and np.isfinite(f.get('obs_loss', np.nan))]
        rr = [f['obs_resid_rms'] for f in fr if 'obs_resid_rms' in f
              and np.isfinite(f.get('obs_resid_rms', np.nan))]
        return {'avg_obs_loss': float(np.mean(ol)) if ol else float('nan'),
                'avg_obs_resid_rms': float(np.mean(rr)) if rr else float('nan'),
                'nRMSE_recheck': float(np.nanmean([f['nRMSE'] for f in fr if 'nRMSE' in f]))}
    except Exception as e:
        return {'obs_fit_error': repr(e)[:200]}
    finally:
        if undo is not None:
            undo()


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
            tag = rec.get('nRMSE', rec.get('error', '?'))
            tag = f'{tag:.4f}' if isinstance(tag, float) else str(tag)[:70]
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<74s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
