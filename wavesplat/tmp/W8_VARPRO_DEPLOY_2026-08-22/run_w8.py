"""W8 러너 — VARPRO 배포 검증. 팔·게이트·판정은 `gates_w8.py` 단일출처."""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
W4 = os.path.abspath(os.path.join(HERE, '..', 'W4_SOLVER_2026-08-19'))
for p in (HERE, REPO):
    sys.path.insert(0, p)
import gates_w8 as G  # noqa: E402

S4DIR = os.path.abspath(os.path.join(HERE, '..', 'S4_phase_oracle_2026-08-08'))
sys.path.insert(0, S4DIR)
import run_s4 as S4  # noqa: E402

EPISODES = [e for e in S4.EPISODES if S4.regime(e) == G.REGIME][:G.N_EPISODES]
OUT = os.environ.get('W8_OUT', os.path.join(HERE, 'results.jsonl'))


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    vp = None
    try:
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(G.ARMS[job['cell']])
        cfg['seed'] = job['seed']
        if job['cell'] == G.MONKEYPATCH_ARM:
            sys.path.insert(0, W4)
            import varpro as vp_mp
            vp_mp.reset_stats()
            vp_mp.install(steps=G.VP_STEPS, weighting='confidence')
            vp = vp_mp
        r = run_estimation(os.path.join(REPO, job['episode']), cfg)
        return {**base,
                'avg_nRMSE': float(r['avg_nRMSE']),
                'init_time_s': float(r.get('init_time_s', float('nan'))),
                'init_iters': r.get('init_iters'),
                'route': r.get('init_method_used') or r.get('route'),
                'varpro_fired': bool(r.get('varpro_fired'))
                                or (bool(vp.stats()['vp_entries']) if vp else False),
                'varpro_steps': r.get('varpro_steps'),
                'mp_entries': (vp.stats()['vp_entries'] if vp else None),
                'n_frames': r.get('n_frames'),
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1200:],
                'wall_s': time.time() - t0}
    finally:
        if vp is not None:
            vp.uninstall()


def main():
    import multiprocessing as mp
    jobs = [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in G.ARM_ORDER for ep in EPISODES for s in G.SEEDS]
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
    print(f'total {len(jobs)} | todo {len(todo)}', flush=True)
    if todo:
        ctx = mp.get_context('spawn')
        t0 = time.time()
        with ctx.Pool(int(os.environ.get('WORKERS', '8'))) as pool, open(OUT, 'a') as fh:
            for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
                fh.write(json.dumps(rec) + '\n'); fh.flush()
                tag = (f"nRMSE={rec['avg_nRMSE']:.5f} fired={rec['varpro_fired']} "
                       f"init={rec['init_time_s']:.1f}s"
                       if 'error' not in rec else str(rec['error'])[:70])
                el = time.time() - t0
                print(f'[{i}/{len(todo)}] {rec["job_id"][:66]:<66s} {tag} '
                      f'ETA {el/i*(len(todo)-i)/60:.1f}min', flush=True)
        print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    print('\n=== 게이트 ===')
    for gid, ok, msg in G.gates(rows):
        print(f'  [{"PASS" if ok else "FAIL"}] {gid}: {msg}')


if __name__ == '__main__':
    main()
