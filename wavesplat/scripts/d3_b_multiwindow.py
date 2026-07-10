"""B: five seeds change the observations. Four windows change the sea.

A2 gave every real number five seeds and three claims dissolved. But a seed
only redraws which 600 returns we keep; the sea underneath is the same instant
of the same record. A record's `t0` is one snapshot of one sea state, and every
real claim in this project -- today's included -- rests on exactly one of them
per record.

So move the window. Four `t0` values spread across each record's usable span,
three seeds each: twelve independent observation sets per record, over four
genuinely different pieces of ocean.

What this can break, in order of how much it would cost us:

  1. adaptive band selector ~= oracle     the load-bearing claim, three times
                                          confirmed (synthetic, HOS, real n=1
                                          window). If it holds across windows it
                                          is real.
  2. adaptive beats the fixed band        LJ18g gave +0.082 on one window.
  3. the energy cap never fires           beta = 1.000 on 20/20 seed-records.
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
import d3_static_field as SF

OUT = 'results/direction3/b_multiwindow.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RECS = ['YS17', 'AA14', 'AA15a', 'LJ18g']
SEEDS = (11, 12, 13)
WINDOWS = (0, 2, 3, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', nargs='*', default=RECS)
    a_ = ap.parse_args()
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    gen = torch.Generator(device=DEV).manual_seed(0)
    res = {}

    print(f"{'record':<8}{'window':>8}{'t0':>7}"
          f"{'fixed .35':>10}{'ADAPTIVE':>10}{'oracle':>9}{'gap':>8}{'beta':>7}")
    for rec in a_.recs:
        rows = []
        # scene_real already exposes six windows spread across the record.
        # The first version of this file moved t0 by hand and every window but
        # the first came back "GT missing": the scene only loads the frames its
        # own window needs.
        for w in WINDOWS:
            sc = SF.scene_real(rec, win=w)
            eta = sc['make_eta']()
            Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
            g_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
            cx, cy = sc['center']
            inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
            m = inside.ravel() & np.isfinite(g_all)
            if m.sum() < 200:
                print(f"{rec:<8}{w:>8}   window unusable (GT missing)")
                continue
            Pq = t(np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                                    np.full(m.sum(), sc['t0'])]))
            gtq = t(g_all[m])
            gtq = gtq - gtq.mean()
            zero = float(gtq.std())
            if zero < 1e-6:
                continue
            sig0 = 0.05 * sc['Hs']

            cell = []
            for sd in SEEDS:
                rng = np.random.default_rng(100 * w + sd)
                P, Z = draw(sc, rng)
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
                        pv = mmse_gpu(Pt[tr], zt[tr], sc['depth'], Pt[he],
                                      sigma0=sig0, kp_hint=sc['kp'], band=b)
                        r = float(torch.sqrt(((pv - zt[he]) ** 2).mean()))
                        if best is None or r < best:
                            best, bb = r, b
                    pr_a = sol(bb)
                    beta = min(1.0, float((zt - zt.mean()).std())
                               / float(pr_a.std() + 1e-9))
                    cell.append((rat(sol(CANDS[0])), rat(pr_a),
                                 min(rat(sol(b)) for b in CANDS), beta))
            c = np.array(cell)
            print(f"{rec:<8}{w:>8}{sc['t0']:>7.0f}"
                  f"{c[:,0].mean():>10.3f}{c[:,1].mean():>10.3f}"
                  f"{c[:,2].mean():>9.3f}{c[:,1].mean()-c[:,2].mean():>+8.3f}"
                  f"{c[:,3].mean():>7.3f}", flush=True)
            rows.append(c.mean(0))
        if rows:
            r = np.array(rows)
            res[rec] = dict(fix35=r[:, 0].mean(), adaptive=r[:, 1].mean(),
                            oracle=r[:, 2].mean(), beta=r[:, 3].mean(),
                            gap=float((r[:, 1] - r[:, 2]).mean()),
                            gap_max=float((r[:, 1] - r[:, 2]).max()),
                            win_sd=float(r[:, 1].std()), nwin=len(rows))

    print("\nper record, averaged over windows (window sd in parentheses):")
    for rec, d in res.items():
        print(f"  {rec:<8} fixed .35 {d['fix35']:.3f}   adaptive "
              f"{d['adaptive']:.3f} ({d['win_sd']:.3f})   oracle "
              f"{d['oracle']:.3f}   gap {d['gap']:+.3f} (worst window "
              f"{d['gap_max']:+.3f})   beta {d['beta']:.3f}   "
              f"{d['nwin']} windows")

    gaps = [d['gap_max'] for d in res.values()]
    gains = [d['fix35'] - d['adaptive'] for d in res.values()]
    betas = [d['beta'] for d in res.values()]
    print(f"\nadaptive vs oracle, worst window of any record: {max(gaps):+.3f}")
    print(f"adaptive vs fixed .35, per record: "
          f"{', '.join(f'{g:+.3f}' for g in gains)}")
    print(f"cap fired anywhere?  min beta = {min(betas):.3f}  "
          f"(1.000 = never)")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in res.items()}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
