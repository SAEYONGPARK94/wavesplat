"""W11 러너 — 라우터 제거(학습 경로 통일)의 비용을 잰다. 규칙은 gates_w14 단일출처."""
import glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, REPO):
    sys.path.insert(0, p)
import gates_w14 as G  # noqa: E402

OUT = os.environ.get('W14_OUT', os.path.join(HERE, 'results.jsonl'))
_PATHS = {}
for p in glob.glob(os.path.join(REPO, '**', 'episode_*.h5'), recursive=True):
    _PATHS.setdefault(os.path.basename(p)[len('episode_'):-3], p)


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    try:
        cfg = dict(DEFAULT_CONFIG); cfg.update(G.ARMS[job['cell']]); cfg['seed'] = job['seed']
        r = run_estimation(_PATHS[job['episode']], cfg)
        return {**base, 'stratum': 'crowded',
                'avg_nRMSE': float(r['avg_nRMSE']),
                'init_time_s': float(r.get('init_time_s', float('nan'))),
                # ★라우터 산출을 되읽는다 (의도 아님). None = 라우터가 돌지 않았다.
                'auto_selected': r.get('auto_selected'),
                'auto_n_selected': r.get('auto_n_selected'),
                'varpro_fired': bool(r.get('varpro_fired')),
                'varpro_steps': r.get('varpro_steps'),
                'init_iters': r.get('init_iters'),
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1200:],
                'wall_s': time.time() - t0}


def main():
    import multiprocessing as mp
    jobs = [{'job_id': f'{c}|{e}|{s}', 'cell': c, 'episode': e, 'seed': s}
            for c in G.ARM_ORDER for e in G.EPISODES for s in G.SEEDS]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    miss = [j['episode'] for j in todo if j['episode'] not in _PATHS]
    if miss:
        print('경로 못 찾음:', sorted(set(miss))[:5]); return
    print(f'total {len(jobs)} | todo {len(todo)}', flush=True)
    ctx = mp.get_context('spawn'); t0 = time.time()
    with ctx.Pool(int(os.environ.get('WORKERS', '8'))) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            tag = (f"nRMSE={rec['avg_nRMSE']:.5f} route={rec['auto_selected']}"
                   if 'error' not in rec else str(rec['error'])[:60])
            el = time.time() - t0
            print(f'[{i}/{len(todo)}] {rec["job_id"][:58]:<58s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.1f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    print('\n=== 게이트 ===')
    for gid, ok, msg in G.gates(rows):
        print(f'  [{"PASS" if ok else "FAIL"}] {gid}: {msg}')


if __name__ == '__main__':
    main()
