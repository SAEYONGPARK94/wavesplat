"""C: the sample expansion the numbers demand.

B showed the window standard deviation is 0.08-0.13, larger than the seed
standard deviation it replaced. With four windows on four records we cannot
resolve a difference smaller than about +-0.06 -- and two of the three gains we
reported (AA14 +0.023, AA15a +0.029) sit exactly on that line.

So widen. Ten records instead of four, six windows instead of four, three seeds
each: up to 180 independent observation sets over 60 genuinely different pieces
of ocean.

`scene_real` reads its record table from d3_falcon_real_operator, which exposes
five of the ten. The other five are defined in d3_sa_multirecord with the same
three fields scene_real actually reads (path, depth, stride), so they are
injected here rather than duplicated.

What is being tested, in the order it would hurt to lose:

  1. adaptive band selection beats the fixed band   claimed on 3/4 records
  2. adaptive ~= oracle                             claimed in the mean, with a
                                                    worst window of +0.106
  3. the energy cap never harms                     beta reaches 0.787

Reported per record with the WINDOW standard deviation, and pooled with a
paired test across all record-window cells, because the pairing (same sea, same
observations, different band rule) is where the power is.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_a2_real_revalidation import draw, CANDS, HELD
import d3_falcon_real_operator as FOP
from d3_sa_multirecord import RECORDS as ALL_RECORDS
import d3_static_field as SF

OUT = 'results/direction3/c_expanded_sample.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = (11, 12, 13)
WINDOWS = (0, 1, 2, 3, 4, 5)


def inject_records():
    """scene_real only reads path/depth/stride from its table."""
    added = []
    for k, cfg in ALL_RECORDS.items():
        if k not in FOP.RECORDS:
            FOP.RECORDS[k] = cfg
            added.append(k)
    return added


def cell(sc, gen, t):
    Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
    eta = sc['make_eta']()
    g_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
    cx, cy = sc['center']
    inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
    m = inside.ravel() & np.isfinite(g_all)
    # B used 200 here. Raising it to 300 silently discarded every AA14 window,
    # which B had scored fine: after the dead-pixel mask that record's fan holds
    # only a couple of hundred finite query points.
    if m.sum() < 150:
        return None
    Pq = t(np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                            np.full(m.sum(), sc['t0'])]))
    gtq = t(g_all[m])
    gtq = gtq - gtq.mean()
    zero = float(gtq.std())
    if not np.isfinite(zero) or zero < 1e-6:
        return None
    sig0 = 0.05 * sc['Hs']

    out = []
    for sd in SEEDS:
        rng = np.random.default_rng(1000 + sd)
        P, Z = draw(sc, rng)
        if len(Z) < 200:
            continue
        Pt, zt = t(P), t(Z)

        def sol(band):
            pr = mmse_gpu(Pt, zt, sc['depth'], Pq, sigma0=sig0,
                          kp_hint=sc['kp'], band=band)
            return pr - pr.mean()

        def rat(pr):
            return float(torch.sqrt(((pr - gtq) ** 2).mean()) / zero)

        with torch.no_grad():
            n = len(zt)
            idx = torch.randperm(n, device=DEV, generator=gen)
            tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
            best, bb = None, CANDS[0]
            for b in CANDS:
                pv = mmse_gpu(Pt[tr], zt[tr], sc['depth'], Pt[he], sigma0=sig0,
                              kp_hint=sc['kp'], band=b)
                r = float(torch.sqrt(((pv - zt[he]) ** 2).mean()))
                if best is None or r < best:
                    best, bb = r, b
            pr_a = sol(bb)
            beta = min(1.0, float((zt - zt.mean()).std())
                       / float(pr_a.std() + 1e-9))
            r_cap = float(torch.sqrt(((beta * pr_a - gtq) ** 2).mean()) / zero)
            out.append((rat(sol(CANDS[0])), rat(pr_a),
                        min(rat(sol(b)) for b in CANDS), beta, r_cap))
    return np.array(out).mean(0) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', nargs='*', default=None)
    a_ = ap.parse_args()
    added = inject_records()
    recs = a_.recs or sorted(FOP.RECORDS.keys())
    print(f"injected into scene_real: {added}")
    print(f"records: {recs}\n", flush=True)

    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    gen = torch.Generator(device=DEV).manual_seed(0)
    res, cells = {}, []

    print(f"{'record':<8}{'win':>5}{'fixed .35':>11}{'ADAPTIVE':>10}"
          f"{'oracle':>9}{'gap':>8}{'beta':>7}{'+cap':>8}")
    for rec in recs:
        rows = []
        for w in WINDOWS:
            try:
                sc = SF.scene_real(rec, win=w)
                c = cell(sc, gen, t)
            except Exception as e:
                print(f"{rec:<8}{w:>5}   skip: {type(e).__name__}", flush=True)
                continue
            if c is None:
                print(f"{rec:<8}{w:>5}   unusable", flush=True)
                continue
            print(f"{rec:<8}{w:>5}{c[0]:>11.3f}{c[1]:>10.3f}{c[2]:>9.3f}"
                  f"{c[1]-c[2]:>+8.3f}{c[3]:>7.3f}{c[4]:>8.3f}", flush=True)
            rows.append(c)
            cells.append((rec, w, *c))
        if rows:
            r = np.array(rows)
            res[rec] = dict(nwin=len(rows), fix35=r[:, 0].mean(),
                            adaptive=r[:, 1].mean(), oracle=r[:, 2].mean(),
                            beta=r[:, 3].mean(), beta_min=r[:, 3].min(),
                            capped=r[:, 4].mean(),
                            win_sd=float(r[:, 1].std()),
                            gain=float((r[:, 0] - r[:, 1]).mean()),
                            gap=float((r[:, 1] - r[:, 2]).mean()),
                            gap_max=float((r[:, 1] - r[:, 2]).max()))

    print(f"\n{'record':<8}{'win':>5}{'gain vs fixed':>15}{'gap vs oracle':>15}"
          f"{'win sd':>9}{'beta min':>10}")
    for rec, d in res.items():
        print(f"{rec:<8}{d['nwin']:>5}{d['gain']:>+15.3f}{d['gap']:>+15.3f}"
              f"{d['win_sd']:>9.3f}{d['beta_min']:>10.3f}")

    if cells:
        a = np.array([[c[2], c[3], c[4], c[5], c[6]] for c in cells])
        d = a[:, 0] - a[:, 1]                       # fixed - adaptive, paired
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        print(f"\npaired over {n} record-window cells "
              f"({len(res)} records x up to {len(WINDOWS)} windows):")
        print(f"  adaptive beats fixed .35 by {d.mean():+.4f} ± {se:.4f} (se)"
              f"   t = {d.mean()/se:+.2f}   wins {int((d>0).sum())}/{n}")
        g = a[:, 1] - a[:, 2]
        print(f"  adaptive vs oracle:  mean {g.mean():+.4f}   "
              f"worst cell {g.max():+.4f}")
        h = a[:, 4] - a[:, 1]
        print(f"  cap cost (capped - adaptive): mean {h.mean():+.5f}   "
              f"worst {h.max():+.4f}   beta min {a[:,3].min():.3f}")
        print(f"\n  window sd of the adaptive score, pooled: "
              f"{np.mean([d_['win_sd'] for d_ in res.values()]):.3f}")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'per_record': {k: {kk: float(vv) for kk, vv in v.items()}
                                  for k, v in res.items()},
                   'cells': [[c[0], int(c[1])] + [float(x) for x in c[2:]]
                             for c in cells]}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
