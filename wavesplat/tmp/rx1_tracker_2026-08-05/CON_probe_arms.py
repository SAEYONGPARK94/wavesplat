"""CON probe C — knobs the Rx-1 prereg did NOT list, run at the SAME oracle
operating point (X-BASE = S2-T1f).  All are shipped, supported config values;
zero code change; disjoint from PRO's K1-K5 (lambda_l1, lambda_smooth,
confidence normaliser, phase-1 skip, TRACK_PHASE_PREDICT).

  C-phionly : tracking_mode='phi_only'  -> w is FROZEN during tracking, so amplitude
              shrinkage is structurally impossible.  If the harm survives, the
              prereg's named outcome ("amplitude shrinkage") is not the mechanism
              and every K1-K5 knob acts on the wrong parameter.
  C-lr25 / C-lr50 : tracking_lr 0.005 -> 0.00125 / 0.0025.  The tracker-OFF arm is
              already lr=0, so this is the SAME axis the 61% headline lives on,
              measured at two interior points.
  C-steps1  : tracking_steps 10 -> 1.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tmp', 's1s2_2026-08-05'))
import numpy as np
import run_s1s2 as S

OUT = os.path.join(HERE, 'CON_probe_results.jsonl')
EXTRA = {
    'C-phionly': dict(tracking_mode='phi_only'),
    'C-lr25':    dict(tracking_lr=0.00125),
    'C-lr50':    dict(tracking_lr=0.0025),
    'C-steps1':  dict(tracking_steps=1),
}

_orig = S.arm_config
def arm_config(arm, episode):
    if arm in EXTRA:
        cfg = _orig('S2-T1f', episode)       # X-BASE, oracle placement, k frozen
        cfg.update(EXTRA[arm])
        return cfg
    return _orig(arm, episode)
S.arm_config = arm_config
S.ORACLE_ARMS = set(S.ORACLE_ARMS) | set(EXTRA)
S.OUT = OUT

if __name__ == '__main__':
    import multiprocessing as mp
    arms = sys.argv[1].split(',') if len(sys.argv) > 1 else list(EXTRA)
    jobs = S.build_jobs(arms)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    print(f'todo {len(todo)}', flush=True)
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(int(os.environ.get('WORKERS','8')), initializer=S._init_worker) as pool, \
            open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(S.run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            tag = rec.get('nRMSE', rec.get('error','?'))
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<48s} '
                  f'{tag if not isinstance(tag,float) else f"{tag:.4f}"}', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)
