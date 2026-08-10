"""S8 driver — 출하 기본값 `lambda_l1` 배포 확증 + λ 곡선 + 근거리 손상 용량-반응.

사전등록 `PREREG.md` + `AMENDMENT_1.md` (md5 동결) 실행.
개입 로직은 frozen `run_rx2b.py`, 지표 추출은 `run_s5.run_one`, 추가 계측은 `run_s7._extra_keys`
를 **import 해서** 쓴다(복사 금지).

## S7 대비 바뀐 것
1. **에피소드** 신선 시드 32-39 (`data_s8_{crowded,sparse}_2026-08-10/`), `regime()` 재정의.
2. **셀 10개** — 배포 팔(추적기 ON) 중심으로 재구성:
   - `AUTO-L1-1e4` ★PRIMARY 처치 · `AUTO-L1-3e4`/`AUTO-L1-0` 용량-반응(출하 팔)
   - `AUTO-L1-1e4-P1ONLY` / `-TRKONLY` — **2-loci 양방향 분리** (A9)
   - `AUTO-NT` / `AUTO-TRK-QTR` — 근거리 손상 용량-반응 (A14)
   - ★`AUTO-SEED-NULL` — **무처치 널**(cfg 동일, 추정 시드만 1572-74). 꼬리 규칙이 희소에서
     시드잡음만으로도 발화하기 때문에(A2) 없으면 안전 게이트를 해석할 수 없다.
   - `AUTO-DUP` — `AUTO` 의 바이트 복제(워커 누출 프로브)
3. **추가 계측(추가 런 0)**: 프레임 단위 실패 꼬리 4키(A20) + `init_iters`/`final_n`(A7).
   `all_resid`(무가중 센서 잔차)는 S5 러너가 이미 기록한다 — ⛔`avg_obs_loss` 는 λ 를 포함하므로
   관측 적합 근거로 쓰지 않는다(A5).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S7DIR = os.path.abspath(os.path.join(HERE, '..', 'S7_confirm_n32_2026-08-10'))
S5DIR = os.path.abspath(os.path.join(HERE, '..', 'S5_window_loss_2026-08-08'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
for p in (REPO, RX2, S5DIR, S7DIR):
    sys.path.insert(0, p)

import run_rx2b as R      # noqa: E402  frozen 개입 로직
import run_s5 as S5       # noqa: E402  지표 추출
import run_s7 as R7       # noqa: E402  _patch_stash / _extra_keys (셸·라우트·α)

R.DATA = REPO

# ── 에피소드 ─────────────────────────────────────────────────────────────────
CROWDED_DIR = 'data_s8_crowded_2026-08-10'
SPARSE_DIR = 'data_s8_sparse_2026-08-10'
SEED_BLOCK = range(32, 40)
CROWDED32 = [f'{CROWDED_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in SEED_BLOCK]
SPARSE32 = [f'{SPARSE_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in SEED_BLOCK]
EPISODES = CROWDED32 + SPARSE32


def regime(ep):
    """⛔상속본(S4/S7)은 다른 디렉터리를 하드코딩한다. 모르는 경로는 조용히 넘어가지 않는다."""
    if ep.startswith(CROWDED_DIR):
        return 'crowded'
    if ep.startswith(SPARSE_DIR):
        return 'sparse'
    raise ValueError(f'S8 regime(): 알 수 없는 에피소드 경로 {ep!r}')


S5.S4.regime = regime
S5.EPISODES = EPISODES

# ── 셀 ────────────────────────────────────────────────────────────────────────
TRACK_LR = 0.005                              # 출하 기본값 (configs/estimation.yaml:16)
NEW_CELLS = {
    'AUTO-L1-1e4':          {'lambda_l1': 1e-4},                     # ★PRIMARY 처치
    'AUTO-L1-3e4':          {'lambda_l1': 3e-4},                     # 용량
    'AUTO-L1-0':            {'lambda_l1': 0.0},                      # 용량
    'AUTO-L1-1e4-P1ONLY':   {'lambda_l1': 1e-4},                     # + 게이트 (아래)
    'AUTO-L1-1e4-TRKONLY':  {'lambda_l1': 1e-4},                     # + 게이트 (아래)
    'AUTO-TRK-QTR':         {'tracking_lr': TRACK_LR * 0.25},        # 근거리 용량
    'AUTO-SEED-NULL':       {},                                      # ★무처치 널 (시드만 다름)
    'AUTO-DUP2':            {},                                      # AUTO 의 바이트 복제
}
for _name, _extra in NEW_CELLS.items():
    R.CELLS.setdefault(_name, dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
    S5.S4.EXTRA[_name] = _extra
R.CELLS.setdefault('AUTO', dict(cfg=dict(R.CELLS['AUTO']['cfg'])))   # 출하 기본값, EXTRA 없음

# 2-loci 게이트: 셀 -> 환경변수
LOCI_GATE = {'AUTO-L1-1e4-P1ONLY': 'S8_L1_PHASE1_ONLY',
             'AUTO-L1-1e4-TRKONLY': 'S8_L1_TRACK_ONLY'}
ALL_GATES = tuple(set(LOCI_GATE.values()))

CELLS = ['AUTO', 'AUTO-L1-1e4', 'AUTO-L1-3e4', 'AUTO-L1-0',
         'AUTO-L1-1e4-P1ONLY', 'AUTO-L1-1e4-TRKONLY',
         'AUTO-NT', 'AUTO-TRK-QTR', 'AUTO-SEED-NULL', 'AUTO-DUP2']

SEEDS = [int(s) for s in os.environ.get('S8_SEEDS', '1569,1570,1571').split(',')]
NULL_SEEDS = [int(s) for s in os.environ.get('S8_NULL_SEEDS', '1572,1573,1574').split(',')]
SEEDS_BY_CELL = {'AUTO-SEED-NULL': NULL_SEEDS}
OUT = os.environ.get('S8_OUT', os.path.join(HERE, 'results.jsonl'))


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES
            for s in SEEDS_BY_CELL.get(c, SEEDS)]


def _tail_keys(r):
    """프레임 단위 실패 꼬리 (A20) + 계산량 구조 (A7). 전부 SECONDARY."""
    fr = r.get('frame_results', [])
    v = np.array([f['nRMSE'] for f in fr if 'nRMSE' in f], float)
    fin = v[np.isfinite(v)]
    out = {
        'nRMSE_p95': float(np.percentile(fin, 95)) if fin.size else float('nan'),
        'nRMSE_max': float(np.max(fin)) if fin.size else float('nan'),
        'frac_frames_nRMSE_gt_0p5': float(np.mean(fin > 0.5)) if fin.size else float('nan'),
        'n_frames_nonfinite': int(v.size - fin.size),
        'init_iters': r.get('init_result', {}).get('n_iters') if isinstance(
            r.get('init_result'), dict) else r.get('init_iters'),
        'final_n': r.get('n_gaussians', -1),
    }
    return out


def run_one(job):
    cell = job['cell']
    os.environ['S2_ALPHA'] = '1'
    for g in ALL_GATES:                       # 셀마다 명시적으로 초기화 (누출 방지)
        os.environ.pop(g, None)
    gate = LOCI_GATE.get(cell)
    if gate:
        os.environ[gate] = '1'
    R7._patch_stash()
    R7._STASH.pop('r', None)
    try:
        rec = S5.run_one(job)
        rec['regime'] = regime(job['episode'])
        rec['loci_gate'] = gate or ''
        r = R7._STASH.pop('r', None)
        if r is not None and 'error' not in rec:
            rec.update(R7._extra_keys(r, job['episode']))   # 셸·안팎·라우트·α
            rec.update(_tail_keys(r))                       # 프레임 꼬리·계산량
        return rec
    finally:
        for g in ALL_GATES:
            os.environ.pop(g, None)


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
          f'workers {n_workers} | seeds {SEEDS} | null seeds {NULL_SEEDS}', flush=True)
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
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<80s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
