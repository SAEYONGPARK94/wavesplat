"""E-8: re-score the whole benchmark (Table B4) on a mesh that can render it.

Prereg: docs/E2_MESH_AUDIT_prereg_2026-07-13.md

E-3b retracted the BF2 row.  But the 55-episode benchmark's every sea state is
at best DEGRADED on the shipped 6.2 m mesh (E-2: BF4 multi-component loses
19-22% of its variance below the mesh's resolving power, and its measured
operator error is 2.4x the nominal noise).  So the whole table is measured
through the renderer, not just its worst row.

Same H5 deployment recipe as d3_h6_v2_e2e (the artifact that produced the
published table), run paired over the coarse and fine renderings of all 55.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import h5py                                            # noqa: E402
from d3_e3b_bf2_rescore import score                   # noqa: E402

COARSE, FINE = 'data', 'tmp/finemesh_clean'
LIST = 'tmp/bench55_episodes.txt'
OUT = 'results/direction3/e8_bench55_rescore.json'


def sea_of(name):
    with h5py.File(os.path.join(COARSE, name), 'r') as h:
        return str(h['config'].attrs['sea_state'])


def main():
    names = [l.strip() for l in open(LIST) if l.strip()]
    missing = [n for n in names if not os.path.exists(os.path.join(FINE, n))]
    if missing:
        raise SystemExit(f'{len(missing)} fine-mesh episodes missing, '
                         f'e.g. {missing[:2]}')

    recs = []
    for i, n in enumerate(names):
        sea = sea_of(n)
        row = dict(episode=n, sea=sea)
        for tag, root in (('coarse', COARSE), ('fine', FINE)):
            r = score(os.path.join(root, n), np.random.default_rng(11 + i))
            row[tag] = r
        recs.append(row)
        print(f"  [{i + 1}/{len(names)}] {n:44s} "
              f"fan {row['coarse']['fan']:+.3f} -> {row['fine']['fan']:+.3f}",
              flush=True)

    def agg(rows, tag):
        return dict(
            n=len(rows),
            ann=float(np.mean([r[tag]['ann'] for r in rows])),
            fan=float(np.mean([r[tag]['fan'] for r in rows])),
            enc=float(np.mean([r[tag]['enc'] for r in rows])),
            fan_ge70=int(sum(r[tag]['fan'] >= 0.7 for r in rows)),
            enc_ge70=int(sum(r[tag]['enc'] >= 0.7 for r in rows)))

    by_sea = {}
    for sea in sorted({r['sea'] for r in recs}):
        rows = [r for r in recs if r['sea'] == sea]
        d = np.array([r['fine']['fan'] - r['coarse']['fan'] for r in rows])
        by_sea[sea] = dict(
            coarse=agg(rows, 'coarse'), fine=agg(rows, 'fine'),
            fan_shift=float(d.mean()),
            fan_shift_se=float(d.std(ddof=1) / np.sqrt(len(d))))

    res = dict(date='2026-07-13',
               prereg='docs/E2_MESH_AUDIT_prereg_2026-07-13.md',
               recipe='H5 deployment recipe, identical to d3_h6_v2_e2e',
               published_B4=dict(fan=0.841, ann=0.839, enc=0.723,
                                 fan_ge70='48/55', enc_ge70='39/55'),
               overall=dict(coarse=agg(recs, 'coarse'), fine=agg(recs, 'fine')),
               by_sea=by_sea, episodes=recs)
    json.dump(res, open(OUT, 'w'), indent=1)

    c, f = res['overall']['coarse'], res['overall']['fine']
    print(f'\n{"=" * 70}\n[E-8] benchmark v2, all 55 episodes')
    print(f'{"":10s} {"ann":>7} {"fan":>7} {"enc":>7}   fan>=0.7   enc>=0.7')
    print(f'{"coarse":10s} {c["ann"]:7.3f} {c["fan"]:7.3f} {c["enc"]:7.3f}'
          f'   {c["fan_ge70"]:2d}/55     {c["enc_ge70"]:2d}/55')
    print(f'{"fine":10s} {f["ann"]:7.3f} {f["fan"]:7.3f} {f["enc"]:7.3f}'
          f'   {f["fan_ge70"]:2d}/55     {f["enc_ge70"]:2d}/55')
    print(f'\n{"sea":6s} {"n":>2}  fan coarse -> fine   (shift +- se)')
    for sea, v in by_sea.items():
        print(f'{sea:6s} {v["coarse"]["n"]:2d}  '
              f'{v["coarse"]["fan"]:+.3f} -> {v["fine"]["fan"]:+.3f}   '
              f'({v["fan_shift"]:+.3f} +- {v["fan_shift_se"]:.3f})')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
