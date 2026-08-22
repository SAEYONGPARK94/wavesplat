"""W7 러너 — 없던 칸 `ORA-WPHI × λ=1e-4(출하)` 를 채운다.

사전등록 `PREREG.md`. 팔·게이트·판정은 전부 `gates_w10.py` **단일출처**에서 온다.
S4/S5 의 오라클 배선(`S4_ORACLE_WPHI`)과 `run_rx2b` 개입 로직을 그대로 import 한다.

★`run_s5.py` 와 다른 점 둘:
  1. **모든 팔이 λ 를 명시**한다 (미명시 러너는 재실행 시 기본값 변경으로 무효가 된다).
  2. **λ 를 실제 updater 객체에서 되읽어 기록**한다 (의도 dict 가 아니라 — S13b 교훈).
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
for p in (HERE, REPO, RX2, S4DIR):
    sys.path.insert(0, p)

import gates_w10 as G     # noqa: E402  ★단일출처
import run_rx2b as R     # noqa: E402
import run_s4 as S4      # noqa: E402

R.DATA = REPO

for name, spec in G.ARMS.items():
    R.CELLS.setdefault(name, dict(cfg='__oracle_k__'))
    R.ORACLE_CELLS = set(R.ORACLE_CELLS) | {name}
    S4.EXTRA[name] = dict(spec['extra'])
    S4.WPHI_CELLS = set(S4.WPHI_CELLS) | {name}

EPISODES = [e for e in S4.EPISODES if S4.regime(e) == G.REGIME][:G.N_EPISODES]
OUT = os.environ.get('W10_OUT', os.path.join(HERE, 'results.jsonl'))


def build_jobs():
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in G.ARM_ORDER for ep in EPISODES for s in G.SEEDS]


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ['S5_WINDOW'] = '1'
    os.environ['S4_ORACLE_WPHI'] = '1'
    import wavesplat.run as WR
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init
    import wavesplat.optimizer.sequential_update as SU
    # ★개입: 모듈 상수를 워커에서 설정한다 (config 키가 아니다).
    SU.SCALE_INVARIANT_L1 = bool(G.ARMS[job['cell']]['scale_invariant'])

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    undo = None
    seen = {}

    # ★λ 를 **실제로 만들어진 updater 객체**에서 되읽는다 (의도 dict 아님).
    real_make = WR.make_updater

    def spy(cfg):
        u = real_make(cfg)
        seen['lambda_l1'] = float(getattr(u, 'lambda_l1', float('nan')))
        seen['lambda_smooth'] = float(getattr(u, 'lambda_smooth', float('nan')))
        seen['init_method'] = getattr(u, 'init_method', None)
        seen['updater_type'] = type(u).__name__
        return u

    try:
        cfg = R.cell_config(job['cell'], job['episode'])
        cfg['seed'] = job['seed']
        path = os.path.join(REPO, job['episode'])
        gt = np.asarray(EpisodeDataLoader(path).wave_components)
        undo = R.apply_cell(job['cell'])
        _init.ORACLE_COMPONENTS = gt
        WR.make_updater = spy
        try:
            r = WR.run_estimation(path, cfg)
        finally:
            WR.make_updater = real_make
            _init.ORACLE_COMPONENTS = None
        fr = r['frame_results']

        def m(k):
            v = [f[k] for f in fr if k in f and np.isfinite(f.get(k, np.nan))]
            return float(np.mean(v)) if v else float('nan')

        return {**base, 'regime': S4.regime(job['episode']),
                'window_resid': float(r.get('window_resid', float('nan'))),
                'window_n_obs': int(r.get('window_n_obs', 0)),
                'all_resid': m('obs_resid_rms'),
                'avg_obs_loss': m('obs_loss'),
                'nRMSE': float(np.nanmean([f['nRMSE'] for f in fr if 'nRMSE' in f])),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'N': r.get('n_gaussians', -1),
                'lambda_l1_readback': seen.get('lambda_l1', float('nan')),
                # ★의도가 아니라 모듈에서 되읽은 값
                'scale_invariant_readback': bool(SU.SCALE_INVARIANT_L1),
                'lambda_smooth_readback': seen.get('lambda_smooth', float('nan')),
                'lambda_smooth_readback': seen.get('lambda_smooth', float('nan')),
                'init_method_readback': seen.get('init_method'),
                'updater_type': seen.get('updater_type'),
                'cfg_intent': {k: cfg.get(k) for k in ('lambda_l1', 'lambda_smooth',
                                                       'tracking_lr', 'n_iters')},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}
    finally:
        if undo is not None:
            undo()
        os.environ['S4_ORACLE_WPHI'] = '0'


def main():
    import multiprocessing as mp
    jobs = build_jobs()
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
    print(f'total {len(jobs)} | todo {len(todo)} | workers {n_workers} | '
          f'episodes {len(EPISODES)} | arms {G.ARM_ORDER}', flush=True)
    if todo:
        ctx = mp.get_context('spawn')
        t0 = time.time()
        with ctx.Pool(n_workers, initializer=R._init_worker) as pool, open(OUT, 'a') as fh:
            for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                el = time.time() - t0
                tag = (f"win={rec['window_resid']:.5f} l1={rec['lambda_l1_readback']:.0e}"
                       if 'error' not in rec else str(rec['error'])[:70])
                print(f'[{i}/{len(todo)}] {rec["job_id"]:<70s} {tag} '
                      f'ETA {el/i*(len(todo)-i)/60:.1f}min', flush=True)
        print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)

    rows = [json.loads(l) for l in open(OUT)]
    w9 = os.path.join(HERE, G.W9_DIR, 'results.jsonl')
    if os.path.exists(w9):
        rows += [r for r in (json.loads(l) for l in open(w9))
                 if r.get('cell') in G.W9_CELLS]
    print('\n=== 게이트 (단일출처 gates_w10.gates) ===', flush=True)
    for gid, ok, msg in G.gates(rows):
        print(f'  [{"PASS" if ok else "FAIL"}] {gid}: {msg}', flush=True)


if __name__ == '__main__':
    main()
