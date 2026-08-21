"""S6 driver — `lambda_l1` 용량-반응 스윕 (출하 팔).

사전등록 `tmp/S6_l1_sweep_2026-08-08/PREREG.md` (md5 동결) 실행.
개입 로직은 frozen `run_rx2b.py`, 지표 추출은 S5 러너를 import 해서 그대로 쓴다(복사 금지).
⚠`lambda_smooth` 는 전 셀 기본값 고정 — 이 배치의 노브는 **`lambda_l1` 하나**다.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S5DIR = os.path.abspath(os.path.join(HERE, '..', 'S5_window_loss_2026-08-08'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
sys.path.insert(0, REPO)
sys.path.insert(0, RX2)
sys.path.insert(0, S5DIR)

import run_rx2b as R   # noqa: E402  frozen
import run_s5 as S5    # noqa: E402  지표 추출 + S4 게이트 배선

R.DATA = REPO

# λ 스윕 셀. 전부 추적기 off, smooth 기본값 유지, init_method='auto'(출하).
LAMBDAS = {'AUTO-NT-L1-3e4': 3e-4, 'AUTO-NT-L1-1e4': 1e-4, 'AUTO-NT-L1-0': 0.0}
for name, lam in LAMBDAS.items():
    R.CELLS.setdefault(name, dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
    S5.S4.EXTRA[name] = {'tracking_lr': 0.0, 'lambda_l1': lam}
# 실제 출하 기본값(추적기 ON, λ 기본) = H3 기준선. EXTRA 없음 = 아무것도 덮지 않는다.
R.CELLS.setdefault('AUTO', dict(cfg=dict(R.CELLS['AUTO']['cfg'])))

CELLS = ['AUTO', 'AUTO-NT', 'AUTO-NT-L1-3e4', 'AUTO-NT-L1-1e4',
         'AUTO-NT-L1-0', 'AUTO-NT-DUP']
EPISODES = S5.EPISODES
SEEDS = [int(s) for s in os.environ.get('S6_SEEDS', '1563,1564,1565').split(',')]
OUT = os.environ.get('S6_OUT', os.path.join(HERE, 'results.jsonl'))


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES for s in SEEDS]


def run_one(job):
    return S5.run_one(job)        # 지표 추출 동일 (window_resid 포함)


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
            tag = (f"nRMSE={rec['nRMSE']:.4f}" if 'error' not in rec
                   else str(rec['error'])[:60])
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<72s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
