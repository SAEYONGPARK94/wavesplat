"""S11 driver — ⚠**초안. 사전등록 동결 전에는 돌리지 않는다.**

목적 (S10 `DECISION.md` §4 의 3·4번 + S11 `BRIEF.md` §4):
  A. `L1-I900`      — `learned_init_iters` 500→900. learned 라우트 전용 한 줄 노브.
  B. `L1-SEEDNULL`  — `AUTO-L1` 의 **시드-널 팔**(신선 추정 시드). 선택기 널 보정의 λ 교락을 지운다.
  C. `L1-REPRO`     — `AUTO-L1` 재현 프로브. **2026-08-11 배포 이후 설정이 표류하지 않았는지**
                      S9 결과와 대조한다(E1 이 잡은 "기준선이 조용히 처치값이 되는" 실패 모드).

개입 로직은 frozen `run_rx2b.py`, 지표 추출은 `run_s5.run_one`, 스태시 패치는 `run_s7._patch_stash`,
셸 정의는 `run_s7.SHELL_KEYS`, n-의존 상수는 `s9_constants` 를 **import 해서** 쓴다(복사 금지).

## S9 대비 바뀐 것
1. **에피소드·레짐 라벨·상수는 S9 것을 그대로 재사용**한다(같은 96 에피소드, 시드 48-63).
   ⟹ 이 배치는 **신선 표본이 아니다.** 확증이 아니라 **기제 분해 + 널 측정**이다.
   ⛔"확증" 문구 금지.
2. **`obs_resid_rms_oos` 를 레코드에 담는다** (S10 §4-1). `run.py:308` 이 프레임마다 계산해 두고도
   `run_s5.py:86-89` 스키마가 버리던 신호다. ⛔동결된 `run_s5.py` 는 **고치지 않는다** —
   `_extra_keys_s11` 로 **덧붙인다**(= `run_s9.py` 가 쓰는 기법). 재현 출처가 흔들리지 않는다.
3. **제어 계측 없음** (S9 승계, 사용자 지시). `S2_ALPHA` 는 매 잡마다 명시 삭제한다.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S9DIR = os.path.abspath(os.path.join(HERE, '..', 'S9_lr_x_l1_2026-08-11'))
S7DIR = os.path.abspath(os.path.join(HERE, '..', 'S7_confirm_n32_2026-08-10'))
S5DIR = os.path.abspath(os.path.join(HERE, '..', 'S5_window_loss_2026-08-08'))
S4DIR = os.path.abspath(os.path.join(HERE, '..', 'S4_phase_oracle_2026-08-08'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
for p in (REPO, RX2, S4DIR, S5DIR, S7DIR, S9DIR):
    sys.path.insert(0, p)

import run_rx2b as R      # noqa: E402  frozen 개입 로직
import run_s5 as S5       # noqa: E402  지표 추출
import run_s7 as R7       # noqa: E402  _patch_stash / SHELL_KEYS  (⛔run_one 은 쓰지 않는다)
import run_s9 as S9       # noqa: E402  ★에피소드·regime()·셀 정의를 **유도해서** 쓴다
import s9_constants as C  # noqa: E402  n-의존 상수 단일 출처

R.DATA = REPO

# ── 에피소드·레짐: S9 에서 **그대로 상속** (⛔손으로 다시 적지 않는다) ──────────
EPISODES = S9.EPISODES
regime = S9.regime
S5.S4.regime = regime
S5.EPISODES = EPISODES

# ── 셀 ────────────────────────────────────────────────────────────────────────
L1_TREAT = S9.L1_TREAT                              # 1e-4, 현 출하 기본값
I900 = 900                                          # = v2_konly 의 max_iters_per_stage 합 [200,200,500]

NEW_CELLS = {
    'L1-REPRO':     {'lambda_l1': L1_TREAT},                                   # C. 재현 프로브
    'L1-I900':      {'lambda_l1': L1_TREAT, 'learned_init_iters': I900},       # A. 한 줄 노브
    'L1-SEEDNULL':  {'lambda_l1': L1_TREAT},                                   # B. 시드-널
}
for _name, _extra in NEW_CELLS.items():
    R.CELLS.setdefault(_name, dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
    S5.S4.EXTRA[_name] = _extra

CELLS = [c for c in os.environ.get('S11_CELLS', 'L1-REPRO,L1-I900,L1-SEEDNULL').split(',') if c]

SEEDS = [int(s) for s in os.environ.get('S11_SEEDS', '1575,1576,1577').split(',')]
NULL_SEEDS = [int(s) for s in os.environ.get('S11_NULL_SEEDS', '1581,1582,1583').split(',')]
SEEDS_BY_CELL = {'L1-SEEDNULL': NULL_SEEDS}
OUT = os.environ.get('S11_OUT', os.path.join(HERE, 'results.jsonl'))

BANNED_ENV = ('S2_ALPHA', 'S8_L1_PHASE1_ONLY', 'S8_L1_TRACK_ONLY')

# ★재현 프로브는 전 에피소드가 필요 없다 — 레짐별 균등 부분집합이면 표류를 잡는다.
REPRO_EVERY = int(os.environ.get('S11_REPRO_EVERY', '8'))    # 96/8 = 12 에피소드


def episodes_for_cell(cell):
    if cell == 'L1-REPRO':
        return EPISODES[::REPRO_EVERY]
    return EPISODES


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in episodes_for_cell(c)
            for s in SEEDS_BY_CELL.get(c, SEEDS)]


def _extra_keys_s11(r):
    """S9 계측 + ★07-23 이 처방한 **held-out 선택 신호**.  전부 SECONDARY(판정 미사용).

    ★`oos_resid_f0` = **추적 첫 프레임의 표본외 잔차**.
    `run.py:245-256` 에서 이 값은 그 프레임의 `update()` **이전에** 계산된다 ⟹ 모델이 본 것은
    Phase 1 초기창뿐이고, 다음 프레임은 **아직 보지 않았다**. 배포 시점(초기화 직후, 추적 시작 전)에
    실제로 존재하는 유일한 **held-out** 잔차이며 **추가 계산이 0** 이다.
    ⟹ 07-23 `RESTART_SELECTION_RESULT` 의 정정절 처방 *"held-out 잔차로 골라라 — 훈련 misfit 은
    과적합 런을 보상한다"* 를 **처음으로 실제 집계**하는 값이다.
    ⛔`window_resid`·`all_resid`·`avg_obs_loss` 는 전부 in-sample 이라 그 처방을 만족하지 못한다.

    `oos_resid_mean` 은 전 추적 프레임 평균 = **in-sample 오염**(i>=1 은 갱신된 모델).
    ⛔선택 규칙에 쓰지 말 것. 진단용으로만 남긴다.
    """
    out = S9._extra_keys_s9(r)                     # ⛔산식 복사 금지 — 상속한다
    fr = r.get('frame_results', [])
    f0 = fr[0].get('obs_resid_rms_oos', float('nan')) if fr else float('nan')
    out['oos_resid_f0'] = float(f0) if np.isfinite(f0) else float('nan')
    v = [f['obs_resid_rms_oos'] for f in fr
         if 'obs_resid_rms_oos' in f and np.isfinite(f.get('obs_resid_rms_oos', np.nan))]
    out['oos_resid_mean'] = float(np.mean(v)) if v else float('nan')
    out['n_frames_oos'] = len(v)
    return out


def run_one(job):
    for g in BANNED_ENV:
        os.environ.pop(g, None)
    R7._patch_stash()
    R7._STASH.pop('r', None)
    try:
        rec = S5.run_one(job)                      # ⛔run_s7.run_one 은 S2_ALPHA 를 켜므로 쓰지 않는다
        rec['regime'] = regime(job['episode'])
        rec['s2_alpha_env'] = os.environ.get('S2_ALPHA', '<unset>')
        r = R7._STASH.pop('r', None)
        if r is not None and 'error' not in rec:
            rec.update(_extra_keys_s11(r))
        return rec
    finally:
        for g in BANNED_ENV:
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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers} | '
          f'seeds {SEEDS} | null {NULL_SEEDS} | cells {cells or CELLS}', flush=True)
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
            tag = (f"nRMSE={rec['nRMSE']:.4f} oos0={rec.get('oos_resid_f0', float('nan')):.5f}"
                   if 'error' not in rec else str(rec['error'])[:60])
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<84s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
