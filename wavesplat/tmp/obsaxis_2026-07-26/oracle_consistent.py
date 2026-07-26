"""E3 fix — lane-consistent oracle. ONE script computes the deployed number and the
oracle under the SAME aggregation, so "the guard closes X of the oracle gap" becomes
quotable (or is deleted).

The bug (P3 R1, elevated by P5 R2): the shipped oracle is per-episode
min(median_lanes(v2), median_lanes(learned)) -- a min OF medians -- while every deployed
number is routed PER LANE and only then medianed. min-of-medians is not an upper bound on
a per-lane policy, which is why the crowded slice of the deployed config (0.17982) sat
BELOW its own "oracle" (0.18053).

Zero new compute. Writes oracle_consistent.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'figs_opus_diag_2026-07-26'))
import common as C  # noqa: E402

OUT = os.path.join(C.ROOT, 'tmp/obsaxis_2026-07-26/oracle_consistent.json')


def build():
    _, v2, _, learned, nsel, bfr = C.load_raw()
    eps = sorted({e for (e, _s) in v2})
    rows = {}
    for ep in eps:
        v2l = [v2.get((ep, E)) for (E, D) in C.LANES]
        lnl = [learned.get((ep, E, D)) for (E, D) in C.LANES]
        nsl = [nsel.get((ep, D)) for (E, D) in C.LANES]
        auto_lane, guard_lane, orc_lane = [], [], []
        for vv, lv, ns in zip(v2l, lnl, nsl):
            base = 'learned' if (ns is not None and ns >= C.THRESH_DEPLOYED) else 'v2'
            g = 'v2' if (base == 'learned' and bfr.get(ep, 1.0) < C.GUARD_RATIO) else base
            auto_lane.append(lv if base == 'learned' else vv)
            guard_lane.append(lv if g == 'learned' else vv)
            orc_lane.append(min(vv, lv))          # <- oracle DECIDED PER LANE
        pv, pl = C.med(v2l), C.med(lnl)
        c = C.comp(ep)
        rows[ep] = dict(comp=c, regime=C.regime(c), pure_v2=pv, pure_learned=pl,
                        auto=C.med(auto_lane), guard=C.med(guard_lane),
                        oracle_old=min(pv, pl),   # min-of-medians (as shipped)
                        oracle_lane=C.med(orc_lane))  # median-of-lane-minima (consistent)
    return rows


def main():
    rows = build()
    eps = sorted(rows)
    assert len(eps) == 100
    SL = {g: [e for e in eps if rows[e]['regime'] == g] for g in C.REGIME_ORDER}
    SL['all'] = eps
    m = lambda k, S: float(np.mean([rows[e][k] for e in S]))

    slices = {}
    for name, S in SL.items():
        d = {k: m(k, S) for k in ('pure_v2', 'pure_learned', 'auto', 'guard',
                                  'oracle_old', 'oracle_lane')}
        d['n'] = len(S)
        # the paradox: how often does the DEPLOYED value beat its own "upper bound"?
        d['n_deployed_below_oracle_old'] = int(sum(
            rows[e]['guard'] < rows[e]['oracle_old'] - 1e-12 for e in S))
        d['n_deployed_below_oracle_lane'] = int(sum(
            rows[e]['guard'] < rows[e]['oracle_lane'] - 1e-12 for e in S))
        d['paradox_old'] = bool(d['guard'] < d['oracle_old'])
        d['paradox_lane'] = bool(d['guard'] < d['oracle_lane'])
        slices[name] = d

    A = slices['all']
    def frac(orc):
        gap = A['auto'] - orc
        return dict(oracle=orc, gap_auto_to_oracle=gap,
                    guard_moved=A['auto'] - A['guard'],
                    frac_closed=(A['auto'] - A['guard']) / gap if gap > 1e-12 else None,
                    residual=A['guard'] - orc)
    # where the residual gap lives, under the CONSISTENT definition
    resid = sorted(((rows[e]['guard'] - rows[e]['oracle_lane']), e) for e in eps)[::-1]
    tot = sum(max(r, 0.0) for r, _ in resid)
    out = dict(
        protocol=dict(lanes=C.LANES, thresh=C.THRESH_DEPLOYED, guard_ratio=C.GUARD_RATIO,
                      deployed='per-lane route -> median over 5 lanes -> mean over 100 ep',
                      oracle_old='per-episode min(median_lanes(v2), median_lanes(learned))',
                      oracle_lane='per-lane min(v2, learned) -> median over lanes -> mean'),
        slices=slices,
        gap_old_way=frac(A['oracle_old']),
        gap_lane_consistent=frac(A['oracle_lane']),
        residual_concentration=dict(
            total=tot,
            top1=dict(episode=resid[0][1], share=resid[0][0] / tot),
            top8_share=sum(r for r, _ in resid[:8]) / tot,
            top=[dict(episode=e, resid=r, share=r / tot) for r, e in resid[:8]]))
    json.dump(out, open(OUT, 'w'), indent=1)

    print('=' * 84)
    print('E3 FIX — lane-consistent oracle (one script, one aggregation)')
    print('=' * 84)
    hdr = f"{'slice':9s} {'n':>3} {'pure_v2':>8} {'pure_lrn':>8} {'auto':>8} {'guard':>8} " \
          f"{'orc_OLD':>8} {'orc_LANE':>8}  paradox(old/lane)"
    print(hdr)
    for name in ('all', 'sparse', 'moderate', 'crowded'):
        d = slices[name]
        print(f"{name:9s} {d['n']:>3} {d['pure_v2']:>8.5f} {d['pure_learned']:>8.5f} "
              f"{d['auto']:>8.5f} {d['guard']:>8.5f} {d['oracle_old']:>8.5f} "
              f"{d['oracle_lane']:>8.5f}  {str(d['paradox_old']):>5s}/{str(d['paradox_lane']):<5s}"
              f"  ep_below: {d['n_deployed_below_oracle_old']:>2d}/"
              f"{d['n_deployed_below_oracle_lane']}")
    for label, g in (('OLD (min-of-medians)', out['gap_old_way']),
                     ('LANE-CONSISTENT     ', out['gap_lane_consistent'])):
        fc = g['frac_closed']
        print(f"\n  {label}: oracle={g['oracle']:.5f}  auto->oracle gap={g['gap_auto_to_oracle']:.5f}"
              f"  guard moved {g['guard_moved']:.5f}"
              f"  => closed {fc*100:.1f}%  residual={g['residual']:.5f}")
    rc = out['residual_concentration']
    print(f"\n  residual gap concentration (lane-consistent): top1 {rc['top1']['episode']} "
          f"{rc['top1']['share']*100:.0f}%, top8 {rc['top8_share']*100:.0f}%")


if __name__ == '__main__':
    main()
