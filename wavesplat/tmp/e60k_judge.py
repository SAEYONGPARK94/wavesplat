"""E-B2K judgment -- applies the FROZEN decision rules, nothing else.

Pre-registration: tmp/E60K_PREREG_v3_FROZEN_2026-07-19.md (frozen 2026-07-19,
before the arena ran). This script implements section 4 (statistic) and
section 5 (PASS / FAIL / PARTIAL / VOID) verbatim. It must not contain any
choice that was not already fixed in that document.

  statistic  window-paired Delta corr, seeds averaged WITHIN window,
             cluster = sea (HOS) / site-day (real), failure = hand < 0.5,
             one-sided, cluster-t AND Webb 6-point null-imposed wild
             cluster bootstrap-t must BOTH converge.
"""
import itertools, json, os, sys
import numpy as np
from collections import defaultdict
from scipy import stats

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
W6 = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
SEEDS = (0, 1, 2)


def wild_p(m, t_obs, n_mc=200_000):
    k = len(m)
    if 6 ** k <= 4_000_000:
        draws = np.array(list(itertools.product(W6, repeat=k)))
    else:
        draws = np.random.default_rng(0).choice(W6, size=(n_mc, k))
    yb = m[None, :] * draws
    tb = yb.mean(1) / (yb.std(1, ddof=1) / np.sqrt(k))
    return float((tb >= t_obs).mean())


def cluster(by, label, quiet=False):
    keys = sorted(by)
    m = np.array([np.mean(by[k]) for k in keys])
    k = len(m)
    mean = float(m.mean())
    se = m.std(ddof=1) / np.sqrt(k)
    t = mean / se
    p = float(stats.t.sf(t, k - 1))
    wp = wild_p(m, t)
    npos = int((m > 0).sum())
    if not quiet:
        print(f'  {label:22s} k {k:2d}  mean {mean:+.4f}  t {t:+.2f}  '
              f'p1 {p:.4f}  wild_p1 {wp:.4f}  signs {npos}/{k}')
    return dict(k=k, mean=mean, t=float(t), p=p, wild_p=wp, n_pos=npos,
                per_cluster=dict(zip(keys, np.round(m, 4).tolist())))


def deltas(rows, a, b, cluster_key, failure_only):
    """Seeds averaged within window, then grouped by cluster."""
    by = defaultdict(list)
    for r in rows:
        if failure_only and r['hand'] >= 0.5:
            continue
        by[r[cluster_key]].append(
            np.mean([r[f'{a}_s{s}'] - r[f'{b}_s{s}'] for s in SEEDS]))
    return by


def manipulation_check():
    """FROZEN VOID rule (v3 amendment 3): VOID only on TRAINING degeneracy --
    divergence / NaN / across-seed smult std > 0.5. Healthy NLL convergence to
    smult >= 2.5 is NOT a void; it is the section-5 tie branch.

    smult is scene-dependent (head = scalar bias b + MLP over pooled features),
    so exp(b) alone is NOT the learned floor. We recompute it exactly the way
    d3_b2deep_train.val_loss reported it: mean/std of exp(head(X)) over the SAME
    first 160 validation scenes, identical for every arm."""
    import torch
    sys.path.insert(0, 'scripts')
    from d3_prior_net_taskft import load_all, DEVICE
    from d3_l1_s2head_train import SmultHead
    D = load_all('tmp/prior_train/bwtask')
    X = torch.tensor(D['X'][:160], device=DEVICE)
    print('\n[manipulation check] learned noise floor per arm '
          '(160 val scenes, identical across arms)')
    out = {}
    for arm in ('T', 'L', 'N'):
        per_seed, scene_std = [], []
        for s in SEEDS:
            p = f'results/direction3/b2deep_{arm}_s{s}.pt'
            if not os.path.exists(p):
                continue
            head = SmultHead().to(DEVICE)
            head.load_state_dict(torch.load(p, map_location=DEVICE)['head'])
            head.eval()
            with torch.no_grad():
                sm = torch.exp(head(X)).cpu().numpy().ravel()
            per_seed.append(float(np.mean(sm)))
            scene_std.append(float(np.std(sm)))
        v = np.array(per_seed, float)
        out[arm] = dict(per_seed=v.tolist(), mean=float(np.nanmean(v)),
                        std=float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                        scene_std=scene_std)
        print(f'  arm {arm}: smult per seed {np.round(v, 3).tolist()}  '
              f'mean {out[arm]["mean"]:.3f}  across-seed std '
              f'{out[arm]["std"]:.3f}')
    void = any(o['std'] > 0.5 or not np.isfinite(o['mean'])
               for k, o in out.items() if k in ('T', 'L', 'N'))
    print(f'  VOID (training degeneracy: across-seed std > 0.5): {void}')
    out['void'] = bool(void)
    return out


def main():
    res = {'prereg': 'tmp/E60K_PREREG_v3_FROZEN_2026-07-19.md'}
    hos_f = 'results/direction3/e60k_hos_arena.json'
    if not os.path.exists(hos_f):
        print(f'ABORT -- primary arena not run yet ({hos_f})')
        return 1
    rows = json.load(open(hos_f))['rows']
    nsea = len({r['sea'] for r in rows})
    print(f'=== PRIMARY ARENA (HOS) === rows {len(rows)}  seas {nsea}  '
          f'draws/sea {len(rows) // max(nsea, 1)}')
    if nsea < 32:
        print(f'  !! incomplete: {nsea}/32 seas -- judgment is PROVISIONAL')

    print('\n[failure windows: hand < 0.5]')
    nf = sum(1 for r in rows if r['hand'] < 0.5)
    print(f'  n failure windows {nf} / {len(rows)}')
    hos = {}
    if nf:
        hos['TL_fail'] = cluster(deltas(rows, 'T', 'L', 'sea', True), 'T-L failure')
        hos['TN_fail'] = cluster(deltas(rows, 'T', 'N', 'sea', True), 'T-N failure')
        hos['LN_fail'] = cluster(deltas(rows, 'L', 'N', 'sea', True), 'L-N failure')
    print('\n[all windows]')
    hos['TL_all'] = cluster(deltas(rows, 'T', 'L', 'sea', False), 'T-L all')
    hos['TN_all'] = cluster(deltas(rows, 'T', 'N', 'sea', False), 'T-N all')
    hos['LN_all'] = cluster(deltas(rows, 'L', 'N', 'sea', False), 'L-N all')
    res['hos'] = hos

    res['manipulation'] = manipulation_check()

    # ---- FROZEN decision -------------------------------------------------
    print('\n=== FROZEN DECISION (prereg section 5) ===')
    v = res['manipulation']['void']
    tl = hos.get('TL_fail')
    tn = hos.get('TN_fail')
    smult_N = res['manipulation']['N']['mean']

    if v:
        print('VOID -- training degeneracy (not a FAIL); re-train and re-run.')
        res['verdict'] = 'VOID'
    else:
        # NLL tie branch (FAIL regardless of T-L)
        tie = (tn is not None and tn['mean'] < 0.02
               and 2.5 <= smult_N <= 4.0)
        cond1 = (tl is not None and tl['mean'] > 0.02
                 and tl['p'] < 0.05 and tl['wild_p'] < 0.05)
        cond3 = hos['TL_all']['mean'] >= -0.01
        print(f'  [FAIL branch] NLL tie (T-N < +0.02 AND smult_N in [2.5,4.0]): '
              f'{tie}   (T-N {tn["mean"] if tn else float("nan"):+.4f}, '
              f'smult_N {smult_N:.2f})')
        print(f'  [PASS 1] HOS T-L > +0.02 AND t-p<0.05 AND wild-p<0.05: {cond1}')
        print(f'  [PASS 3] all-window harmlessness >= -0.01: {cond3} '
              f'({hos["TL_all"]["mean"]:+.4f})')
        hos_fail = (tl is None or tl['mean'] <= 0 or tl['wild_p'] >= 0.10)
        print(f'  [FAIL branch] HOS Delta <= 0 or wild p >= 0.10: {hos_fail}')
        if tie:
            res['verdict'] = 'FAIL (NLL tie -- non-likelihood identity dissolves)'
        elif hos_fail:
            res['verdict'] = 'FAIL (primary arena)'
        elif cond1 and cond3:
            res['verdict'] = ('PASS pending real leg (prereg 5 cond 2 + BS11a '
                              'blocker)')
        else:
            res['verdict'] = 'PARTIAL'
        print(f'\n  VERDICT (primary arena): {res["verdict"]}')
        print('  NOTE: real leg (cond 2 + BS11a <= -0.05 blocker) still required '
              'before any PASS is declared.')

    json.dump(res, open('results/direction3/e60k_judgment.json', 'w'), indent=1)
    print('\nwrote results/direction3/e60k_judgment.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
