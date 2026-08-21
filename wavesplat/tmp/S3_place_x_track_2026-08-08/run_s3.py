"""S3 driver — 배치 × 추적기 2x2 를 **한 배치에서** 돌린다.

사전등록 `tmp/S3_place_x_track_2026-08-08/PREREG.md` (md5 동결) 실행.

개입·지표 추출은 `tmp/rx2_init_2026-08-05/run_rx2b.py` 를 **import 해서 그대로** 쓴다
(복사 금지). 그 `run_one` 이 이미 `nRMSE` 와 직교분해 `decorr_mean`/`shrink_mean`/`m_mean`
을 저장하므로 H3 는 추가 코드 없이 나온다.

이 파일이 바꾸는 것은 셋뿐이다:
  1. `R.DATA = REPO`      -- 에피소드 식별자를 REPO 상대경로로 (두 데이터 디렉터리)
  2. 셀 3개 조립          -- `AUTO-NOTRACK` / `ORACLE-NOTRACK` / `AUTO-DUP`
  3. `R.cell_config` 래핑 -- '__oracle_k__' 센티널은 dict 가 아니라 문자열이라
     `tracking_lr` 을 얹을 자리가 없다. 래퍼가 **frozen 함수를 먼저 호출한 뒤** 덧씌운다.
frozen 파일 자체는 수정하지 않는다.
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

# --- 2x2 셀 조립 (frozen CELLS 는 파일이 아니라 메모리에서만 확장) -------------------
if 'AUTO-NOTRACK' not in R.CELLS:
    R.CELLS['AUTO-NOTRACK'] = dict(cfg=dict(R.CELLS['AUTO']['cfg']))
if 'ORACLE-NOTRACK' not in R.CELLS:
    R.CELLS['ORACLE-NOTRACK'] = dict(cfg='__oracle_k__')     # 배치 축은 ORACLE-K 와 동일
if 'AUTO-DUP' not in R.CELLS:
    R.CELLS['AUTO-DUP'] = dict(cfg=dict(R.CELLS['AUTO']['cfg']))
R.ORACLE_CELLS = set(R.ORACLE_CELLS) | {'ORACLE-NOTRACK'}    # 오라클 성분 주입 대상

# 추적기 off 는 cfg 덧씌우기로만 준다 (노브 1개: tracking_lr)
EXTRA = {'AUTO-NOTRACK': {'tracking_lr': 0.0},
         'ORACLE-NOTRACK': {'tracking_lr': 0.0}}

_frozen_cell_config = R.cell_config


def _cell_config(cell, episode):
    cfg = _frozen_cell_config(cell, episode)     # frozen 로직이 먼저
    cfg.update(EXTRA.get(cell, {}))              # 그 위에 노브 1개
    assert 'init_method' in cfg                  # W9 규율
    return cfg


R.cell_config = _cell_config

CELLS = ['AUTO', 'AUTO-NOTRACK', 'ORACLE-K', 'ORACLE-NOTRACK', 'AUTO-DUP']

CROWDED_DIR = 'data_indep_crowded_2026-08-08'
SPARSE_DIR = 'data_indep_sparse_2026-08-08'
CROWDED16 = [f'{CROWDED_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in range(20, 24)]
SPARSE16 = [f'{SPARSE_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in range(20, 24)]
EPISODES = CROWDED16 + SPARSE16

SEEDS = [int(s) for s in os.environ.get('S3_SEEDS', '1554,1555,1556').split(',')]
OUT = os.environ.get('S3_OUT', os.path.join(HERE, 'results.jsonl'))


def regime(ep):
    return 'crowded' if ep.startswith(CROWDED_DIR) else 'sparse'


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES for s in SEEDS]


def run_one(job):
    rec = R.run_one(job)                 # frozen body
    rec['regime'] = regime(job['episode'])
    return rec


def preflight():
    import h5py
    bad = []
    for ep in EPISODES:
        p = os.path.join(REPO, ep)
        if not os.path.exists(p):
            bad.append(ep)
            continue
        with h5py.File(p, 'r') as f:
            if str(dict(f['config'].attrs).get('mesh_verdict', '?')) != 'ADEQUATE':
                bad.append(f'{ep} (mesh)')
    return bad


def main():
    import multiprocessing as mp
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    bad = preflight()
    if bad:
        print(f'PREFLIGHT FAIL {len(bad)}:', bad[:10])
        sys.exit(1)
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
            tag = f'{tag:.4f}' if isinstance(tag, float) else str(tag)[:80]
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<74s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
