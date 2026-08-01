"""PREREG held-out real evaluation — bank-MMSE + rolling refit (2026-08-01).

Protocol: docs/PREREG_REAL_HELDOUT_BANKMMSE_2026-08-01.md (binding).
Harness: extends tmp/bankmmse_port_2026-07-29/run_rolling_refit.py (streaming
replay, replication assert, 600-obs cap, median field, sustained nRMSE) to the
prereg §2 design: 21 fresh windows x 5 arms.

NOTHING here changes a repo default; no threshold is computed at run time.
Scoring (§3.1-§3.4) lives in score_prereg.py and runs only after all arms are in.

Arms (§2.1) — all: blind sigma0 = 0.2*std(z_fit_window), seeds {42,1042,2042},
pointwise-median field, 600-obs cap via default_rng(0), init_method EXPLICIT.
  A  bank-MMSE  + refit R = 2 s          memory 40 s   kp_hint None
  B  bank-MMSE    frozen (shipped)       memory 40 s   kp_hint None
  C  v2_konly    + refit R = 2 s         memory 40 s   (analytic init)
  D  bank-MMSE    frozen                 memory 20 s   kp_hint None
  E  bank-MMSE    frozen                 memory 40 s   kp_hint = 2 * kp_blind
     kp_blind = the two-stage beamform kp the estimator itself derives from the
     window's observations (arm B's own fit.kp; identical across val seeds).
     GT-free, so arm E stays a legitimate low-cost escape.

Output: results.jsonl, one JSON line per (window, arm) plus one 'window' line
per window (append-only, so an interrupted run resumes where it stopped).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'bankmmse_port_2026-07-29'))
import run_rolling_refit as rr          # sets ROOT/scripts paths, chdir(ROOT)
from v2k import v2_member

OUT = 'tmp/prereg_heldout_2026-08-01/results.jsonl'
W_MAX = rr.W_MAX                        # 40 s, the frozen protocol memory window
SEEDS = rr.SEEDS                        # {42, 1042, 2042}
CAP = 600
REFIT_R = 2.0

WINDOWS = ([('AA14', w) for w in range(6)]
           + [('LJ18g', w) for w in range(6)]
           + [(r, w) for r in ('AA15a', 'AA15b', 'YS17') for w in (0, 2, 4)])

ARM_ORDER = ['B', 'A', 'D', 'E', 'C']   # B first: it supplies E's kp_blind
ARM_SPEC = {
    'A': dict(est='bank',     refit=REFIT_R, memory_s=40.0, kp='blind'),
    'B': dict(est='bank',     refit=None,    memory_s=40.0, kp='blind'),
    'C': dict(est='v2_konly', refit=REFIT_R, memory_s=40.0, kp='blind'),
    'D': dict(est='bank',     refit=None,    memory_s=20.0, kp='blind'),
    'E': dict(est='bank',     refit=None,    memory_s=40.0, kp='x2'),
}


# ------------------------------------------------------------------ utilities
def log(rec):
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')
        f.flush()
        os.fsync(f.fileno())


def load_done():
    done = {}
    if not os.path.exists(OUT):
        return done
    with open(OUT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('kind') == 'arm':
                done[(r['window'], r['arm'])] = r
            elif r.get('kind') == 'window':
                done[(r['window'],
                      '_meta' if r.get('status') == 'ok' else '_excluded')] = r
    return done


def fit_obs(st, t_f, memory_s):
    """Observations the deployed rolling window holds at fit time t_f, with the
    frozen 600-cap subsample (default_rng(0)) — identical to
    SequentialUpdater._bank_fit_window and to the 07-29 harness."""
    m = (st['P_all'][:, 2] >= t_f - memory_s) & (st['P_all'][:, 2] <= t_f + 1e-9)
    P, z = st['P_all'][m], st['z_all'][m]
    if len(z) > CAP:
        ii = np.random.default_rng(0).choice(len(z), CAP, replace=False)
        P, z = P[ii], z[ii]
    return P, z


def bank_fit(st, t_f, memory_s, kp_hint, cache):
    """median-of-3 bank-MMSE fit. Returns (fits, wall_m3_s, per_member_walls)."""
    key = ('bank', round(t_f, 3), memory_s,
           None if kp_hint is None else round(float(kp_hint), 6))
    if key in cache:
        return cache[key]
    P, z = fit_obs(st, t_f, memory_s)
    s0 = 0.2 * float(np.std(z))
    from wavesplat.optimizer.bank_mmse import fit_bank_mmse
    fits, walls = [], []
    for s in SEEDS:
        t = time.time()
        fits.append(fit_bank_mmse(P, z, st['depth'], s0, kp_hint=kp_hint, seed=s))
        walls.append(time.time() - t)
    val = (fits, float(sum(walls)), [float(w) for w in walls], int(len(z)))
    cache[key] = val
    return val


def bank_render(st, fits, tau, cache):
    key = (id(fits), round(tau, 3))
    if key in cache:
        return cache[key]
    Xq, Yq = np.meshgrid(st['gxq'], st['gyq'])
    Q = np.column_stack([Xq.ravel(), Yq.ravel(),
                         np.full(Xq.size, st['t0'] + tau)])
    med = np.median(np.stack([f.render(Q) for f in fits], 0), 0) \
        .reshape(len(st['gyq']), len(st['gxq']))
    cache[key] = med
    return med


def score(st, pred, tau):
    j = int(round(tau / st['dt']))
    g = st['Zq'][j]
    m = np.isfinite(g)
    return float(np.sqrt(np.mean((pred[m] - g[m]) ** 2)) / st['sg'])


def zero_curve(st, taus):
    out = []
    for tau in taus:
        j = int(round(tau / st['dt']))
        g = st['Zq'][j]
        m = np.isfinite(g)
        out.append(float(np.sqrt(np.mean(g[m] ** 2)) / st['sg']))
    return out


def t_f_for(st, tau, refit):
    if refit is None:
        return st['t0']
    return st['t0'] + refit * int(np.floor(tau / refit + 1e-9))


# ------------------------------------------------------------------- arm runs
def run_bank_arm(st, taus, spec, kp_blind, cache, rcache):
    kp_hint = None if spec['kp'] == 'blind' else 2.0 * float(kp_blind)
    curve, walls, keys, nobs = [], {}, [], None
    for tau in taus:
        t_f = t_f_for(st, tau, spec['refit'])
        fits, w_m3, w_mem, n_obs = bank_fit(st, t_f, spec['memory_s'],
                                            kp_hint, cache)
        nobs = n_obs
        k = round(t_f, 3)
        if k not in walls:
            walls[k] = (w_m3, w_mem)
            keys.append(k)
        curve.append(score(st, bank_render(st, fits, tau, rcache), tau))
    return dict(curve=curve, n_fits=len(keys),
                fit_wall_s_m3=[round(walls[k][0], 3) for k in keys],
                fit_wall_s_member=[round(w, 3) for k in keys
                                   for w in walls[k][1]],
                kp_hint=None if kp_hint is None else float(kp_hint),
                n_obs_fit=nobs)


def run_v2_arm(st, taus, spec, pool):
    """Arm C: same refit machinery, v2_konly (analytic) init instead of bank.
    The 3 seeds run in spawned workers; membership is bit-identical to running
    them in-process (test_parallel.py)."""
    groups = {}
    for tau in taus:
        groups.setdefault(round(t_f_for(st, tau, spec['refit']), 3), []).append(tau)
    curve = {}
    walls, nobs = [], None
    for t_f in sorted(groups):
        my_taus = groups[t_f]
        P, z = fit_obs(st, t_f, spec['memory_s'])
        nobs = int(len(z))
        t_shift = float(P[:, 2].min())
        rel = [st['t0'] + tau - t_shift for tau in my_taus]
        args = [(P, z, st['depth'], st['gxq'], st['gyq'], t_shift, rel, s)
                for s in SEEDS]
        t = time.time()
        members = pool.map(v2_member, args) if pool is not None else \
            [v2_member(a) for a in args]
        walls.append(time.time() - t)
        med = np.median(np.stack(members, 0), 0)     # (n_rel, ny, nx)
        for i, tau in enumerate(my_taus):
            curve[tau] = score(st, med[i], tau)
    return dict(curve=[curve[t] for t in taus], n_fits=len(groups),
                fit_wall_s_m3=[round(w, 3) for w in walls],
                fit_wall_s_member=None, kp_hint=None, n_obs_fit=nobs)


# ------------------------------------------------------------------------ main
def main():
    done = load_done()
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(3)
    t_run = time.time()
    try:
        for record, win in WINDOWS:
            wkey = f'{record}_w{win}'
            pending = [a for a in ARM_ORDER if (wkey, a) not in done]
            if (wkey, '_excluded') in done:
                print(f'{wkey}: previously EXCLUDED, skipping', flush=True)
                continue
            if not pending:
                print(f'{wkey}: complete, skipping', flush=True)
                continue

            t = time.time()
            st = rr.build_stream(record, win)
            try:
                rr.assert_replication(st, record, win)
            except AssertionError as e:
                log(dict(kind='window', window=wkey, record=record, win=win,
                         status='replication_failed', reason=str(e)))
                print(f'{wkey}: REPLICATION FAILED -- excluded ({e})', flush=True)
                done[(wkey, '_excluded')] = True
                continue
            taus = [float(v) for v in np.arange(0.0, st['lead_max'] + 1e-9, 1.0)]
            if (wkey, '_meta') not in done:
                log(dict(kind='window', window=wkey, record=record, win=win,
                         status='ok', lead_max=st['lead_max'], dt=st['dt'],
                         n_tau=len(taus), grid=[len(st['gyq']), len(st['gxq'])],
                         gt_std=st['sg'], depth=st['depth'],
                         zero_curve=zero_curve(st, taus),
                         zero_sustained=float(np.mean(zero_curve(st, taus))),
                         setup_wall_s=round(time.time() - t, 2)))

            cache, rcache = {}, {}
            kp_blind = bank_fit(st, st['t0'], 40.0, None, cache)[0][0].kp
            for arm in ARM_ORDER:
                if (wkey, arm) in done:
                    continue
                spec = ARM_SPEC[arm]
                ta = time.time()
                if spec['est'] == 'bank':
                    res = run_bank_arm(st, taus, spec, kp_blind, cache, rcache)
                else:
                    res = run_v2_arm(st, taus, spec, pool)
                rec = dict(kind='arm', window=wkey, record=record, win=win,
                           arm=arm, est=spec['est'], refit_s=spec['refit'],
                           memory_s=spec['memory_s'], kp_mode=spec['kp'],
                           seeds=SEEDS, lead_max=st['lead_max'],
                           taus=taus, kp_blind=float(kp_blind),
                           sustained_mean=float(np.mean(res['curve'])),
                           nowcast=float(res['curve'][0]),
                           arm_wall_s=round(time.time() - ta, 2), **res)
                log(rec)
                done[(wkey, arm)] = rec
                print(f"  {wkey} {arm}: sustained {rec['sustained_mean']:.4f} "
                      f"(nowcast {rec['nowcast']:.4f}) fits {res['n_fits']} "
                      f"wall {rec['arm_wall_s']:.0f}s", flush=True)
            el = time.time() - t_run
            print(f'{wkey} done | elapsed {el/60:.1f} min', flush=True)
    finally:
        pool.close()
        pool.join()
    print(f'ALL DONE in {(time.time() - t_run)/3600:.2f} h -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
