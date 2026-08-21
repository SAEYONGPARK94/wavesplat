"""S2 driver — 추적기 on/off 를 **제어 물리량 alpha_wave** 와 **전역 격자 eta** 로 동시에 잰다.

사전등록 `tmp/S2_control_metric_2026-08-08/PREREG.md` (md5 동결) 실행.

★설계의 핵심: **두 지표를 같은 런에서 뽑는다.** 그래야 차이가 나올 때 그 원인이
데이터도 팔도 아니라 **지표**임이 확정된다. 배치를 나눠 재면 그 귀속이 불가능하다.

개입 로직은 `tmp/rx2_init_2026-08-05/run_rx2b.py` 를 **import 해서** 쓴다(복사 금지).
이 파일이 새로 쓰는 것은 **지표 추출**뿐이며, 그건 어차피 신규다.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05'))
sys.path.insert(0, REPO)
sys.path.insert(0, RX2)
sys.path.insert(0, HERE)

import run_rx2b as R          # noqa: E402  frozen intervention logic
import alpha_wave as AW       # noqa: E402

R.DATA = REPO

# PREREG §3 — 노브 1개. frozen 파일은 수정하지 않고 메모리에서만 조립한다.
_AUTO = dict(R.CELLS['AUTO']['cfg'])
if 'AUTO-NOTRACK' not in R.CELLS:
    R.CELLS['AUTO-NOTRACK'] = dict(cfg=dict(_AUTO, tracking_lr=0.0))
if 'AUTO-DUP' not in R.CELLS:
    R.CELLS['AUTO-DUP'] = dict(cfg=dict(_AUTO))

CELLS = ['AUTO', 'AUTO-NOTRACK', 'AUTO-DUP']
CROWDED_DIR = 'data_indep_crowded_2026-08-08'
SPARSE_DIR = 'data_indep_sparse_2026-08-08'
CROWDED16 = [f'{CROWDED_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in range(20, 24)]
SPARSE16 = [f'{SPARSE_DIR}/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in range(20, 24)]
EPISODES = CROWDED16 + SPARSE16

SEEDS = [int(s) for s in os.environ.get('S2_SEEDS', '1551,1552,1553').split(',')]
OUT = os.environ.get('S2_OUT', os.path.join(HERE, 'results.jsonl'))


def regime(ep):
    return 'crowded' if ep.startswith(CROWDED_DIR) else 'sparse'


def build_jobs(cells=None):
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in (cells or CELLS) for ep in EPISODES for s in SEEDS]


def run_one(job):
    """M1(alpha_wave) 과 M2(전역 eta nRMSE) 를 **같은 런**에서 추출한다."""
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ['S2_ALPHA'] = '1'                     # run.py 의 opt-in 계측 게이트
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    undo = None
    try:
        cfg = R.cell_config(job['cell'], job['episode'])   # frozen
        cfg['seed'] = job['seed']
        path = os.path.join(REPO, job['episode'])
        comp = np.asarray(EpisodeDataLoader(path).wave_components, float)
        undo = R.apply_cell(job['cell'])                   # frozen
        r = run_estimation(path, cfg)
        frames = r['frame_results']

        # M2 -- 레거시 지표 (기존 네 배치가 쓴 것과 동일한 정의)
        m2 = float(np.nanmean([f['nRMSE'] for f in frames if 'nRMSE' in f]))

        # M1 -- 제어 물리량
        af = [f for f in frames if 'alpha_legs' in f and 't' in f]
        a = AW.episode_alpha(af, comp) if af else None

        return {**base, 'regime': regime(job['episode']),
                'nRMSE': m2,
                'alpha_episode': (a or {}).get('episode'),
                'alpha_per_lead': (a or {}).get('per_lead'),
                'alpha_stencil_err': (a or {}).get('stencil_err'),
                'alpha_norm': (a or {}).get('norm_episode'),
                'n_alpha_frames': len(af),
                'N': r.get('n_gaussians', -1),
                'cfg': {k: v for k, v in cfg.items() if k != 'seed'},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}
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
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in rec:
                done.add(rec['job_id'])
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
            a = (rec.get('alpha_episode') or {}).get('0.2')
            tag = (f"eta={rec['nRMSE']:.4f} a02={a:.4f}" if a is not None and 'error' not in rec
                   else rec.get('error', '?'))
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<70s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
