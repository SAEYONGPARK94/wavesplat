"""S1 driver — `G1P1` 을 신선 독립 32 에피소드(혼잡16 + 희소16)에서 재측정.

사전등록 `tmp/S1_indep_sparse_2026-08-08/PREREG.md` (md5 동결) §2 를 그대로 실행한다.

★**셀 정의를 복사하지 않는다.** `tmp/rx2_init_2026-08-05/run_rx2b.py` 를 **import 해서**
그 `CELLS` / `apply_cell` / `cell_config` / `run_one` 을 **그대로** 쓴다.
⟹ "개입이 원 실험과 동일한가"가 코드 동일성으로 보장된다(복사본 표류 불가).
E-C/E-D 에서 채점기를 **복사**했다가 원본의 결함까지 물려받은 전례가 있다.

이 파일이 원 러너에 대해 바꾸는 것은 **두 가지뿐**:
  1. `R.DATA = REPO`  ⟹ 에피소드 식별자를 **REPO 상대경로**로 줘서 두 데이터 디렉터리를 동시에 쓴다.
  2. 자체 job 목록(32 에피소드 × 7 셀 × 시드 1548/1549/1550).
`run_rx2b.py` 자체는 **읽기 전용**이며 수정하지 않는다.
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

import run_rx2b as R  # noqa: E402  (frozen cell definitions -- imported, never copied)

# (1) 에피소드 식별자를 REPO 상대경로로 쓴다. spawn 워커가 이 모듈을 import 할 때마다 적용된다.
#     ⚠논증이 아니라 `spawn_check.py` 로 **측정**해서 확인한다 (AMENDMENT 1 F6).
R.DATA = REPO

# (2) AMENDMENT 1 F5 — 출하 팔 위의 G1P1. frozen `run_rx2b.py` 파일은 수정하지 않고,
#     import 된 CELLS 딕셔너리에 **메모리에서만** 한 항목을 조립해 넣는다. 기존 셀은 불변.
#     `AUTO-G1P1 - AUTO` 가 노브 2개(krange + GROW_SUBTRACT)짜리 출하-팔 대조가 된다.
if 'AUTO-G1P1' not in R.CELLS:
    R.CELLS['AUTO-G1P1'] = dict(cfg=dict(R.CELLS['AUTO']['cfg']),
                                krange=R.NARROW,
                                env={'RX2_GROW_SUBTRACT': '1'})

CROWDED_DIR = 'data_indep_crowded_2026-08-08'
SPARSE_DIR = 'data_indep_sparse_2026-08-08'

CROWDED16 = [f'{CROWDED_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in range(20, 24)]
SPARSE16 = [f'{SPARSE_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in range(20, 24)]
EPISODES = CROWDED16 + SPARSE16

# PREREG §2 + AMENDMENT 1 F5: 8 cells. Names must exist in R.CELLS (7 frozen + 1 composed).
CELLS = ['X-BASE', 'X-BASE2', 'G1', 'P1', 'G1P1', 'AUTO', 'AUTO-G1P1', 'ORACLE-K']
assert all(c in R.CELLS for c in CELLS), [c for c in CELLS if c not in R.CELLS]

# Seed ledger [1548-1818]; S1 consumes 1548/1549/1550 (median-of-3).
SEEDS = [int(s) for s in os.environ.get('S1_SEEDS', '1548,1549,1550').split(',')]

OUT = os.environ.get('S1_OUT', os.path.join(HERE, 'results.jsonl'))


def regime(ep):
    return 'crowded' if ep.startswith(CROWDED_DIR) else 'sparse'


def build_jobs(cells=None):
    cells = cells or CELLS
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in cells for ep in EPISODES for s in SEEDS]


def run_one(job):
    """Frozen worker body + regime tag. Everything else is run_rx2b's."""
    rec = R.run_one(job)
    rec['regime'] = regime(job['episode'])
    return rec


def preflight():
    """PREREG §4 G-2/G-3: every episode exists and is readable before the pool starts."""
    import h5py
    missing = []
    for ep in EPISODES:
        p = os.path.join(REPO, ep)
        if not os.path.exists(p):
            missing.append(ep)
            continue
        with h5py.File(p, 'r') as f:
            v = str(dict(f['config'].attrs).get('mesh_verdict', '?'))
            if v != 'ADEQUATE':
                missing.append(f'{ep} (mesh_verdict={v})')
    return missing


def main():
    import multiprocessing as mp
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None

    bad = preflight()
    if bad:
        print(f'PREFLIGHT FAIL — {len(bad)} 에피소드 결측/부적합:', flush=True)
        for b in bad[:40]:
            print('   ', b, flush=True)
        sys.exit(1)
    print(f'preflight ok: {len(EPISODES)} 에피소드 (혼잡 {len(CROWDED16)} / 희소 {len(SPARSE16)})',
          flush=True)

    n_workers = int(os.environ.get('WORKERS', '8'))
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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers} '
          f'| seeds {SEEDS}', flush=True)
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
            if isinstance(tag, float):
                tag = f"{tag:.4f}"
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<72s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
