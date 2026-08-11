"""S9 driver — 희소 배포 확증(n=64) + `tracking_lr` 축 폐쇄.

사전등록 `PREREG.md` (md5 동결) 실행.
개입 로직은 frozen `run_rx2b.py`, 지표 추출은 `run_s5.run_one`, 스태시 패치는 `run_s7._patch_stash`
를 **import 해서** 쓴다(복사 금지).

## S8 대비 바뀐 것
1. **에피소드** 신선 시드 48-63. 레짐별 n 이 **비대칭**(희소 64 / 혼잡 32) — 표본을 검정력이
   모자란 쪽에만 쓴다. 목록·시드·디렉터리는 전부 `s9_constants.py` 에서 **유도**한다.
2. **셀 7개** — lr 축은 **배포 후보가 아니라 폐쇄용**으로만 싣는다(PREREG §0/§5b).
3. ★★★**제어 계측 제거.** 사용자 지시(2026-08-11)로 제어 축이 닫혔다.
   ⚠상속원 `run_s7.run_one:164` 은 `os.environ['S2_ALPHA']='1'` 을 **무조건 켠다.**
   그래서 이 러너는 `run_s7.run_one` 을 **호출하지 않고**, 자체 `_extra_keys_s9` 를 쓰며,
   매 잡마다 `S2_ALPHA` 를 **명시적으로 삭제**한다(G-7 이 이를 측정으로 검증한다).
   ⟹ `alpha_wave` 는 import 되지도, 계산되지도 않는다.
"""
import json
import os
import sys
import time

import numpy as np

import s9_constants as C

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S7DIR = os.path.abspath(os.path.join(HERE, '..', 'S7_confirm_n32_2026-08-10'))
S5DIR = os.path.abspath(os.path.join(HERE, '..', 'S5_window_loss_2026-08-08'))
S4DIR = os.path.abspath(os.path.join(HERE, '..', 'S4_phase_oracle_2026-08-08'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
for p in (REPO, RX2, S4DIR, S5DIR, S7DIR):
    sys.path.insert(0, p)

import run_rx2b as R      # noqa: E402  frozen 개입 로직
import run_s5 as S5       # noqa: E402  지표 추출
import run_s7 as R7       # noqa: E402  _patch_stash / SHELL_KEYS  (⛔run_one 은 쓰지 않는다)

R.DATA = REPO

# ── 에피소드 (⛔목록을 손으로 적지 않는다 — s9_constants 에서 유도) ──────────────
BLOCKS = {'sparse': ((2, 5), ('BF4', 'BF5')), 'crowded': ((15, 30), ('BF4', 'BF5'))}


def episodes_for(rg):
    ncomps, bfs = BLOCKS[rg]
    return [f'{C.DATA_DIR[rg]}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in bfs for nc in ncomps for s in C.SEEDS[rg]]


EPISODES = episodes_for('crowded') + episodes_for('sparse')
_EP_REGIME = {ep: rg for rg in ('crowded', 'sparse') for ep in episodes_for(rg)}


def regime(ep):
    """⛔상속본은 다른 디렉터리를 하드코딩하거나 성분수로 추론한다. 모르는 경로는 조용히 넘기지 않는다."""
    try:
        return _EP_REGIME[ep]
    except KeyError:
        raise ValueError(f'S9 regime(): 알 수 없는 에피소드 경로 {ep!r}')


S5.S4.regime = regime
S5.EPISODES = EPISODES

# ── 셀 ────────────────────────────────────────────────────────────────────────
TRACK_LR = 0.005                                   # 출하 기본값 (estimation.yaml:16 + run.py:30)
L1_SHIP = 1e-3                                     # 출하 기본값 (estimation.yaml:12 + run.py:26)
L1_TREAT = 1e-4                                    # ★PRIMARY 처치
QTR = TRACK_LR * 0.25

NEW_CELLS = {
    'AUTO-L1':       {'lambda_l1': L1_TREAT},                          # ★PRIMARY 처치
    'AUTO-QTR-L1':   {'tracking_lr': QTR, 'lambda_l1': L1_TREAT},      # lr 축 폐쇄
    'AUTO-QTR':      {'tracking_lr': QTR},                             # lr 성분 단독
    'AUTO-NT':       {'tracking_lr': 0.0},                             # 사다리 끝 + 양성 대조
    'AUTO-SEED-NULL': {},                                              # ★무처치 널 (시드만 다름)
    'AUTO-DUP':      {},                                               # AUTO 의 바이트 복제
}
for _name, _extra in NEW_CELLS.items():
    R.CELLS.setdefault(_name, dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
    S5.S4.EXTRA[_name] = _extra
R.CELLS.setdefault('AUTO', dict(cfg=dict(R.CELLS['AUTO']['cfg'])))     # 출하 기본값, EXTRA 없음

CELLS = ['AUTO', 'AUTO-L1', 'AUTO-QTR-L1', 'AUTO-QTR', 'AUTO-NT', 'AUTO-SEED-NULL', 'AUTO-DUP']

SEEDS = [int(s) for s in os.environ.get('S9_SEEDS', '1575,1576,1577').split(',')]
NULL_SEEDS = [int(s) for s in os.environ.get('S9_NULL_SEEDS', '1578,1579,1580').split(',')]
SEEDS_BY_CELL = {'AUTO-SEED-NULL': NULL_SEEDS}
OUT = os.environ.get('S9_OUT', os.path.join(HERE, 'results.jsonl'))

# ★제어 축 종료 — 이 게이트는 어느 셀에서도 켜지지 않는다 (PREREG G-7)
BANNED_ENV = ('S2_ALPHA', 'S8_L1_PHASE1_ONLY', 'S8_L1_TRACK_ONLY')


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES
            for s in SEEDS_BY_CELL.get(c, SEEDS)]


def _extra_keys_s9(r):
    """버려지던 계측을 레코드로. 전부 SECONDARY(판정 미사용).

    ⚠`run_s7._extra_keys` 의 셸/라우트 부분과 **동일한 산식**이되, 제어 물리량(`alpha_*`)
    블록은 **통째로 없다**(사용자 지시). 셸 정의는 `run_s7.SHELL_KEYS` 를 import 해서 쓰므로
    S8 과 셸 경계가 어긋날 수 없다.
    """
    fr = r.get('frame_results', [])
    out = {
        'auto_selected': r.get('auto_selected'),
        'auto_n_selected': r.get('auto_n_selected'),
        'ensemble_k': r.get('ensemble_k'),
    }

    def fmean(key):
        v = [f[key] for f in fr if key in f and np.isfinite(f.get(key, np.nan))]
        return float(np.mean(v)) if v else float('nan')

    for k in ('RMSE_obs_r30', 'RMSE_unobs_r30', 'RMSE_obs_r5', 'RMSE_unobs_r5'):
        out[k] = fmean(k)

    sh = {}
    for key in R7.SHELL_KEYS:
        v = [f['shell_rmse'][key] for f in fr
             if 'shell_rmse' in f and key in f['shell_rmse']
             and np.isfinite(f['shell_rmse'][key])]
        sh[key] = float(np.mean(v)) if v else float('nan')
    out['shell_rmse'] = sh
    fr0 = next((f for f in fr if 'shell_frac' in f), None)
    out['shell_frac'] = fr0['shell_frac'] if fr0 else None

    # 프레임 단위 실패 꼬리 + 계산량 구조 (S8 A20/A7 승계)
    v = np.array([f['nRMSE'] for f in fr if 'nRMSE' in f], float)
    fin = v[np.isfinite(v)]
    out.update({
        'nRMSE_p95': float(np.percentile(fin, 95)) if fin.size else float('nan'),
        'nRMSE_max': float(np.max(fin)) if fin.size else float('nan'),
        'frac_frames_nRMSE_gt_0p5': float(np.mean(fin > 0.5)) if fin.size else float('nan'),
        'n_frames_nonfinite': int(v.size - fin.size),
        'init_iters': (r.get('init_result', {}).get('n_iters')
                       if isinstance(r.get('init_result'), dict) else r.get('init_iters')),
        'final_n': r.get('n_gaussians', -1),
    })
    return out


def run_one(job):
    for g in BANNED_ENV:                 # ★매 잡마다 명시적으로 제거 (누출·상속 방지)
        os.environ.pop(g, None)
    R7._patch_stash()
    R7._STASH.pop('r', None)
    try:
        rec = S5.run_one(job)            # ⛔run_s7.run_one 은 S2_ALPHA 를 켜므로 쓰지 않는다
        rec['regime'] = regime(job['episode'])
        rec['s2_alpha_env'] = os.environ.get('S2_ALPHA', '<unset>')   # G-7 증거를 레코드에 남긴다
        r = R7._STASH.pop('r', None)
        if r is not None and 'error' not in rec:
            rec.update(_extra_keys_s9(r))
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
          f'seeds {SEEDS} | null {NULL_SEEDS} | eps {len(EPISODES)} '
          f'(sparse {len(episodes_for("sparse"))}, crowded {len(episodes_for("crowded"))})',
          flush=True)
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
            tag = (f"nRMSE={rec['nRMSE']:.4f}" if 'error' not in rec else str(rec['error'])[:60])
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<84s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
