"""B3 / Rx-3 scorer -- implements PREREG.md sec.4 + sec.7 VERBATIM. No post-hoc changes.

Path (i) of the two required scoring paths; path (ii) is rederive_b3.py, written from
scratch against the same frozen prereg.
"""
import json
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

CONTROL = 'AUTO'
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK', 'AUTO-K6', 'AUTO-K7', 'AUTO-K4']

CROWDED = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
           'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
MID = ['episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
       'episode_BF4_5comp_seed0_uniform.h5']
SPARSE = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
          'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
          'episode_BF4_1comp_seed0_uniform.h5']
ALL12 = SPARSE + MID + CROWDED
BLOCKS = [('ALL-12', ALL12), ('crowded 15-30c', CROWDED), ('mid 5c', MID),
          ('sparse 1-2c', SPARSE)]

# PREREG sec.6 -- frozen MDE table (computed BEFORE the run, from B2 control-arm variance)
MDE = {'ALL-12': 0.0134, 'crowded 15-30c': 0.0329, 'mid 5c': 0.0042, 'sparse 1-2c': 0.0014}

# PREREG sec.4 -- direction convention. +1 => lower is better (Delta<0 = intervention wins)
DIRECTION = {'nRMSE': +1, 'm_mean': +1, 'decorr_mean': +1, 'shrink_mean': +1,
             'absVR_framewise': +1, 'Pearson_rho': -1}

NULLS = {r['episode']: r['zero_null_tracking_nRMSE_mean']
         for r in json.load(open(os.path.join(
             REPO, 'tmp/B2_g1p1_ext_2026-08-06/nulls12.json')))}


def sign_p(k, n):
    """Two-sided sign test, PREREG sec.4(5). k = # wins out of n."""
    if n == 0:
        return float('nan')
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def load(path):
    rows, errs = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if 'error' in r:
            errs += 1
            continue
        rows.append(r)
    return rows, errs


def ep_median(rows, cell, ep, key):
    """PREREG sec.4(1): seeds -> episode by MEDIAN."""
    v = [r[key] for r in rows
         if r['cell'] == cell and r['episode'] == ep
         and r.get(key) is not None and np.isfinite(r[key])]
    return float(np.median(v)) if v else float('nan')


def main():
    res_path = os.path.join(HERE, 'results.jsonl')
    rows, errs = load(res_path)
    print(f"== B3 results.jsonl: {len(rows)} ok, {errs} errors, "
          f"cells {sorted({r['cell'] for r in rows})}, seeds {sorted({r['seed'] for r in rows})}")

    # ---------------------------------------------------------------- integrity (PREREG 4)
    print("\n" + "=" * 78)
    print("INTEGRITY CHECK (PREREG sec.4): Phase-1 quantities must be IDENTICAL across the")
    print("six cells for the same (episode, seed).  A mismatch VOIDS that cell.")
    print("=" * 78)
    idx = {(r['cell'], r['episode'], r['seed']): r for r in rows}
    seeds = sorted({r['seed'] for r in rows})
    voided = set()
    for key in ('init_nrmse', 'auto_selected', 'auto_n_selected', 'ensemble_k', 'N'):
        bad = []
        for ep in ALL12:
            for s in seeds:
                ref = idx.get((CONTROL, ep, s))
                if ref is None:
                    continue
                for c in CELLS:
                    r = idx.get((c, ep, s))
                    if r is None:
                        continue
                    a, b = ref.get(key), r.get(key)
                    same = (a == b) or (isinstance(a, float) and isinstance(b, float)
                                        and np.isnan(a) and np.isnan(b))
                    if not same:
                        bad.append((c, ep, s, a, b))
                        voided.add(c)
        print(f"  {key:16s} mismatches: {len(bad)}")
        for t in bad[:6]:
            print(f"      {t}")
    print(f"  ==> VOIDED cells: {sorted(voided) if voided else 'NONE (all six agree)'}")

    # router record
    print("\n-- AUTO router record (PREREG sec.2: AUTO is a ROUTER, not one estimator) --")
    for ep in ALL12:
        rr = [idx.get((CONTROL, ep, s)) for s in seeds]
        rr = [r for r in rr if r]
        sel = [r.get('auto_selected') for r in rr]
        nse = [r.get('auto_n_selected') for r in rr]
        ek = [r.get('ensemble_k') for r in rr]
        print(f"  {ep[8:-3]:32s} route={sel} n_sel={nse} ens_k={ek}")

    # ---------------------------------------------------------------- null screen (PREREG 4)
    print("\n" + "=" * 78)
    print("FLAT-SEA NULL SCREEN (PREREG sec.4).  (a) episode excluded iff CONTROL AUTO's")
    print("episode-median nRMSE >= that episode's null.  (b) cell flagged NULL-FAILING iff")
    print("its block medOfMed >= the block's median null.")
    print("=" * 78)
    print(f"{'episode':34s} {'null':>8s} {'AUTO':>8s} {'m':>7s}  keep?")
    keep = []
    for ep in ALL12:
        a = ep_median(rows, CONTROL, ep, 'nRMSE')
        m = ep_median(rows, CONTROL, ep, 'm_mean')
        ok = a < NULLS[ep]
        if ok:
            keep.append(ep)
        print(f"  {ep[8:-3]:32s} {NULLS[ep]:8.4f} {a:8.4f} {m:7.3f}  {'KEEP' if ok else '*** DROP ***'}")
    dropped = [e for e in ALL12 if e not in keep]
    print(f"\n  kept {len(keep)}/12; dropped: {[e[8:-3] for e in dropped] or 'none'}")

    # ---------------------------------------------------------------- levels + paired deltas
    def block_table(name, eps, tag):
        eps = [e for e in eps if e in ALL12]
        if not eps:
            return
        nullmed = float(np.median([NULLS[e] for e in eps]))
        print("\n" + "-" * 78)
        print(f"BLOCK {name}  [{tag}]  n={len(eps)}  median flat-sea null = {nullmed:.4f}"
              f"   MDE80={MDE.get(name, float('nan'))}")
        print("-" * 78)
        print(f"{'cell':14s} {'medOfMed':>9s} {'vsNULL':>8s} | {'dNRMSE':>9s} {'k/n':>6s} "
              f"{'p':>8s} {'dRel':>8s} | {'drho':>8s} {'d|VR-1|':>9s} {'dm':>8s} "
              f"{'ddecorr':>8s} {'dshrink':>8s}")
        for c in CELLS:
            lv = float(np.median([ep_median(rows, c, e, 'nRMSE') for e in eps]))
            flag = 'PASS' if lv < nullmed else 'FAIL'
            if c == CONTROL:
                print(f"{c:14s} {lv:9.4f} {flag:>8s} | {'(control)':>9s}")
                continue
            d = np.array([ep_median(rows, c, e, 'nRMSE') - ep_median(rows, CONTROL, e, 'nRMSE')
                          for e in eps])
            drel = np.array([(ep_median(rows, c, e, 'nRMSE') - ep_median(rows, CONTROL, e, 'nRMSE'))
                             / ep_median(rows, CONTROL, e, 'nRMSE') for e in eps])
            k = int(np.sum(d < 0))
            extra = []
            for key in ('Pearson_rho', 'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean'):
                dd = np.array([ep_median(rows, c, e, key) - ep_median(rows, CONTROL, e, key)
                               for e in eps])
                extra.append(float(np.median(dd)))
            print(f"{c:14s} {lv:9.4f} {flag:>8s} | {np.median(d):+9.4f} {k:3d}/{len(eps):<2d} "
                  f"{sign_p(k, len(eps)):8.4f} {np.median(drel):+8.2%} | "
                  + " ".join(f"{v:+8.4f}" for v in extra[:2])
                  + " " + " ".join(f"{v:+8.4f}" for v in extra[2:]))
        # per-episode detail for the primary contrast
        print(f"   per-episode Delta(AUTO-NOTRACK - AUTO) nRMSE:")
        for e in eps:
            dv = ep_median(rows, 'AUTO-NOTRACK', e, 'nRMSE') - ep_median(rows, CONTROL, e, 'nRMSE')
            print(f"      {e[8:-3]:32s} {dv:+.4f}")

    for name, eps in BLOCKS:
        block_table(name, eps, 'ALL EPISODES (unfiltered)')
        f = [e for e in eps if e in keep]
        if len(f) != len(eps):
            block_table(name, f, 'NULL-SCREENED')

    # ---------------------------------------------------------------- verdict (PREREG sec.7)
    print("\n" + "=" * 78)
    print("PRE-REGISTERED VERDICT (PREREG sec.7) -- target = AUTO-NOTRACK vs AUTO")
    print("=" * 78)
    for name, eps in [('ALL-12', [e for e in keep]),
                      ('crowded 15-30c', [e for e in CROWDED if e in keep])]:
        n = len(eps)
        if n == 0:
            print(f"  {name}: no episode survives the null screen -> NO VERDICT")
            continue
        d = np.array([ep_median(rows, 'AUTO-NOTRACK', e, 'nRMSE')
                      - ep_median(rows, CONTROL, e, 'nRMSE') for e in eps])
        med = float(np.median(d))
        k = int(np.sum(d < 0))
        mde = MDE[name]
        if name == 'ALL-12':
            unan = (k >= 11) or (k <= n - 11)
            coin = 4 <= k <= 8
        else:
            unan = (k == n) or (k == 0)
            coin = not unan
        if abs(med) >= mde and unan:
            v = 'MATTERS'
        elif abs(med) < mde and coin:
            v = 'NULL'
        else:
            v = 'INCONCLUSIVE'
        print(f"  {name:16s} n={n:2d}  Delta_med={med:+.4f}  k={k}/{n}  "
              f"p={sign_p(k, n):.4f}  MDE80={mde}  |Delta|/MDE={abs(med)/mde:.2f}"
              f"   ==> ***{v}***")


if __name__ == '__main__':
    main()
