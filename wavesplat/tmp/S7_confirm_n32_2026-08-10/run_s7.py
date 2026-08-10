"""S7 driver — `lambda_l1=1e-4` 확증 배치 (신선 32+32, n=32).

사전등록 `tmp/S7_confirm_n32_2026-08-10/PREREG.md` (+ `AMENDMENT_1.md`, md5 동결) 실행.
개입 로직은 frozen `run_rx2b.py`, 지표 추출은 `run_s5.run_one` 을 **import 해서** 쓴다(복사 금지).

## S6 대비 바뀐 것
1. **에피소드** — 신선 시드 블록 24-31, 두 레짐 각 32 (`data_s7_{crowded,sparse}_2026-08-10/`).
2. ★**`regime()` 재정의** — 상속되는 `S4.regime` 은 `ep.startswith('data_indep_crowded_2026-08-08')`
   라 **S7 경로를 전부 `sparse` 로 라벨링한다**(CON C-3). 여기서 덮고, 알 수 없는 경로는 **예외**로
   죽인다(조용한 오라벨 금지).
3. **셀 추가** `AUTO-L1-1e4` — 추적기 **ON**(출하) + λ=1e-4 = `configs/estimation.yaml:12` **한 줄**
   변경과 등가인 유일 노브 팔 (PRO P-1).
4. **버려지던 키를 살린다** (추가 런 0): `run.py` 가 이미 계산하는
   `shell_rmse`(9셸) · `RMSE_obs_r30` · `RMSE_unobs_r30` — 추적기 off 가 지지집합 안을 포기하고
   밖을 버는 거래인지(E-D/Rx-7) **전역 nRMSE 로는 볼 수 없다**(CON C-1, PRO P-4);
   `auto_selected` · `auto_n_selected` · `ensemble_k` — `auto` 라우터가 레짐마다 **다른 추정기**를
   고르므로(learned 면 앙상블 1, analytic 이면 median-of-3) "혼잡 전용"이 "라우트 전용"일 수
   있다(CON C-6).
5. **`S2_ALPHA=1`** — 제어 물리량 `alpha_wave`(S2). 전부 SECONDARY·판정 미사용.

⚠`lambda_smooth` 는 전 셀 기본값(1e-4) 고정 — 스윕 노브는 `lambda_l1` 하나.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S5DIR = os.path.abspath(os.path.join(HERE, '..', 'S5_window_loss_2026-08-08'))
S2DIR = os.path.abspath(os.path.join(HERE, '..', 'S2_control_metric_2026-08-08'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
for p in (REPO, RX2, S5DIR, S2DIR):
    sys.path.insert(0, p)

import run_rx2b as R      # noqa: E402  frozen 개입 로직
import run_s5 as S5       # noqa: E402  지표 추출 (window_resid 포함)
import alpha_wave as AW   # noqa: E402  S2 동결 규약

R.DATA = REPO

# ── 에피소드: 신선 시드 블록 24-31 ────────────────────────────────────────────
CROWDED_DIR = 'data_s7_crowded_2026-08-10'
SPARSE_DIR = 'data_s7_sparse_2026-08-10'
SEED_BLOCK = range(24, 32)
CROWDED32 = [f'{CROWDED_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in SEED_BLOCK]
SPARSE32 = [f'{SPARSE_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in SEED_BLOCK]
EPISODES = CROWDED32 + SPARSE32


def regime(ep):
    """★상속본을 덮는다. 알 수 없는 경로는 조용히 'sparse' 가 되지 않고 죽는다."""
    if ep.startswith(CROWDED_DIR):
        return 'crowded'
    if ep.startswith(SPARSE_DIR):
        return 'sparse'
    raise ValueError(f'S7 regime(): 알 수 없는 에피소드 경로 {ep!r}')


S5.S4.regime = regime          # `S5.run_one` 이 레코드에 박는 라벨을 교정
S5.EPISODES = EPISODES

# ── 셀 ────────────────────────────────────────────────────────────────────────
# 처치 = lambda_l1. 추적기 축은 EXTRA 의 tracking_lr 로만 갈린다.
NEW_CELLS = {
    'AUTO-NT-L1-1e4': {'tracking_lr': 0.0, 'lambda_l1': 1e-4},   # 처치 (두 PRIMARY 공통)
    'AUTO-NT-L1-3e4': {'tracking_lr': 0.0, 'lambda_l1': 3e-4},   # SECONDARY 용량반응
    'AUTO-L1-1e4':    {'lambda_l1': 1e-4},                       # ★단일 노브 배포 팔 (추적기 ON)
}
for _name, _extra in NEW_CELLS.items():
    R.CELLS.setdefault(_name, dict(cfg=dict(R.CELLS['AUTO']['cfg'])))
    S5.S4.EXTRA[_name] = _extra
# 실제 출하 기본값(추적기 ON, λ 기본). EXTRA 없음 = 아무것도 덮지 않는다.
R.CELLS.setdefault('AUTO', dict(cfg=dict(R.CELLS['AUTO']['cfg'])))

CELLS = ['AUTO', 'AUTO-NT', 'AUTO-NT-L1-1e4', 'AUTO-L1-1e4',
         'AUTO-NT-L1-3e4', 'AUTO-NT-DUP']
SEEDS = [int(s) for s in os.environ.get('S7_SEEDS', '1566,1567,1568').split(',')]
OUT = os.environ.get('S7_OUT', os.path.join(HERE, 'results.jsonl'))

SHELL_KEYS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100',
              '100_150', '150_inf')

_STASH = {}


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES for s in SEEDS]


def _patch_stash():
    """`run_estimation` 의 **전체** 반환값을 붙잡는다.

    `S5.run_one` 은 필요한 몇 개만 남기고 나머지를 버린다. 결과를 복사해 새 러너를 쓰는 대신
    (규약: 복사 금지), 원본 반환값을 가로채 **추가 키만** 덧붙인다.
    `S5.run_one` 은 함수 안에서 `from wavesplat.run import run_estimation` 하므로
    호출 시점에 모듈 속성을 다시 읽는다 ⟹ 아래 패치가 그대로 먹는다.
    """
    import wavesplat.run as WR
    if getattr(WR, '_S7_PATCHED', False):
        return
    _orig = WR.run_estimation

    def _wrapped(path, cfg):
        r = _orig(path, cfg)
        _STASH['r'] = r
        return r

    WR.run_estimation = _wrapped
    WR._S7_PATCHED = True


def _extra_keys(r, episode):
    """버려지던 계측을 레코드로. 전부 SECONDARY(판정 미사용)."""
    fr = r.get('frame_results', [])
    out = {
        'auto_selected': r.get('auto_selected'),
        'auto_n_selected': r.get('auto_n_selected'),
        'ensemble_k': r.get('ensemble_k'),
    }

    def fmean(key):
        v = [f[key] for f in fr if key in f and np.isfinite(f.get(key, np.nan))]
        return float(np.mean(v)) if v else float('nan')

    # 지지집합 안/밖 (E-D · Rx-7 축)
    for k in ('RMSE_obs_r30', 'RMSE_unobs_r30', 'RMSE_obs_r5', 'RMSE_unobs_r5'):
        out[k] = fmean(k)
    # 거리 셸 9개
    sh = {}
    for key in SHELL_KEYS:
        v = [f['shell_rmse'][key] for f in fr
             if 'shell_rmse' in f and key in f['shell_rmse']
             and np.isfinite(f['shell_rmse'][key])]
        sh[key] = float(np.mean(v)) if v else float('nan')
    out['shell_rmse'] = sh
    fr0 = next((f for f in fr if 'shell_frac' in f), None)
    out['shell_frac'] = fr0['shell_frac'] if fr0 else None

    # 제어 물리량 (S2 동결 규약)
    try:
        from wavesplat.data_loader import EpisodeDataLoader
        af = [f for f in fr if 'alpha_legs' in f and 't' in f]
        if af:
            comp = np.asarray(
                EpisodeDataLoader(os.path.join(REPO, episode)).wave_components, float)
            a = AW.episode_alpha(af, comp)
            out.update({'alpha_episode': a.get('episode'),
                        'alpha_per_lead': a.get('per_lead'),
                        'alpha_stencil_err': a.get('stencil_err'),
                        'alpha_norm': a.get('norm_episode'),
                        'n_alpha_frames': len(af)})
    except Exception as e:                       # 계측 실패가 배치를 죽이지 않게
        out['alpha_error'] = repr(e)
    return out


def run_one(job):
    os.environ['S2_ALPHA'] = '1'                 # run.py 의 opt-in 제어 계측 게이트
    _patch_stash()
    _STASH.pop('r', None)
    rec = S5.run_one(job)                        # 지표 추출은 S5 그대로
    rec['regime'] = regime(job['episode'])       # 상속 라벨 교정 (이중 안전)
    r = _STASH.pop('r', None)
    if r is not None and 'error' not in rec:
        rec.update(_extra_keys(r, job['episode']))
    return rec


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
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<78s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
