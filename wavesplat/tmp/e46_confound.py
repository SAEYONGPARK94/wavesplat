"""e46 — resolve the SEARCH x BLOCK confound behind my "top-K6 manufactured the
DEAD verdicts" narrative.

WHY. I claimed: e42/e43b's two pre-registered DEAD verdicts (fit_bmaF -0.0166 on
si=100..163; v3_point -0.0137 on si=200..263) were artifacts of my own top-K=6
seeding, because a FULL 952,560-hypothesis enumeration scored +0.0671 on
si=300..347.

The E-45 arc audit ruled that **UNVERIFIABLE**: the full enumeration was NEVER run
on si=100/200, so SEARCH (topK6 vs full) and BLOCK (100/200 vs 300) are
**completely confounded**. And the four blind estimates
    [-0.0166 (topK6, si100), -0.0137 (topK6, si200),
     +0.0671 (full,  si300), -0.0067 (full,  si400)]
have Cochran Q = 16.9, p = 0.0007 -- three sit at ~-0.01 and **si=300 is the
outlier**. The parsimonious reading is therefore "si=300 is an outlier block",
NOT "top-K6 manufactured the negatives". My narrative is the less parsimonious
one and it has never been tested.

THIS SCRIPT TESTS IT. Run the FULL enumeration on the two blocks where DEAD was
measured. That breaks the confound:
  - If FULL >> topK6 on si=100/200 too  -> search WAS the bottleneck; my narrative
    survives (though the effect size is then whatever it is on those blocks).
  - If FULL ~= topK6 on si=100/200      -> search was NEVER the bottleneck;
    si=300 was simply an outlier block, and my "the DEAD verdicts measured my own
    search" story must be RETRACTED PERMANENTLY.

PRE-REGISTERED (frozen before the run):
  SEAS   si=100..163 and si=200..263 -- deliberately the ALREADY-BURNED blocks.
         That is the point: these are exactly where topK6 was measured, and the
         paired comparison FULL-vs-topK6 on the SAME seas is what breaks the
         confound. No new claim is made about those blocks' absolute level.
  ARMS   FULL   = blind full-enumeration point-commit (identical construction to
                  tmp/e43e_fresh_battery.full_blind_grid)
         cur    = incumbent (recomputed, must match the stored value)
         stored = the topK6-era arm from the original runs
                  (e42: fit_bmaF; e43b: v3_point)
  PRIMARY  paired (FULL - stored_topK6_arm) per block, t-CI.
  DECISION my narrative SURVIVES only if that paired difference is >= +0.05 with
           CI excluding 0 on BOTH blocks (it must reproduce roughly the +0.084
           swing that si=300 showed: +0.0671 vs ~-0.015).
           Otherwise: RETRACT "top-K6 manufactured DEAD" permanently.
  NOTE   This cannot rescue the blind route -- si=400 (fresh, full enumeration)
         already gave -0.0067. Whatever happens here, blind stays ~0. This is
         purely about whether my CAUSAL STORY was true.

Cost ~38 s/sea x 128 = ~80 min.
Out: results/direction3/e46_confound.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, net_gamma_map
from e43e_fresh_battery import full_blind_grid
from e43d_gate import paired

OUT = 'results/direction3/e46_confound.json'
SRC = {'si100': ('results/direction3/e42_blind_recovery.json', 'fit_bmaF'),
       'si200': ('results/direction3/e43b_v3_pipeline.json', 'v3_point')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    out = {}
    blocks = [('si100', 100, 164), ('si200', 200, 264)]
    if a.smoke:
        blocks = [('si100', 100, 103), ('si200', 200, 203)]

    for tag, lo, hi in blocks:
        path, arm = SRC[tag]
        stored = {r['si']: r for r in json.load(open(path))['rows']}
        rows = []
        for si in range(lo, hi):
            if si not in stored:
                continue
            S = build_sea_crosstp(si)
            g, th = full_blind_grid(S)
            cur = solve_with_grid(S, np.maximum(10.0 ** net_gamma_map(S), 1e-8))
            r = dict(si=si, FULL=solve_with_grid(S, g),
                     cur=float(cur), cur_stored=float(stored[si]['cur_grid']),
                     topk6=float(stored[si][arm]))
            r['cur_match'] = bool(abs(r['cur'] - r['cur_stored']) < 1e-6)
            rows.append(r)
            if si % 20 == 0 or a.smoke:
                print(f"{tag} si{si} FULL {r['FULL']:+.3f} topK6 "
                      f"{r['topk6']:+.3f} cur {r['cur']:+.3f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        nm = sum(r['cur_match'] for r in rows)
        print(f"\n--- {tag} (n={len(rows)}) --- cur reproduces {nm}/{len(rows)}")
        res = {}
        for aa, bb in [('FULL', 'cur'), ('topk6', 'cur'), ('FULL', 'topk6')]:
            res[f'{aa}-{bb}'] = paired(rows, aa, bb)
            r = res[f'{aa}-{bb}']
            star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
            print(f"  {aa:6s} - {bb:6s}: {r['mean']:+.4f} +- {r['se']:.4f} "
                  f"CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
                  f"win {r['win']}/{r['n']}")
        out[tag] = dict(rows=rows, contrasts=res, cur_match=nm)

    print('\n=== ★ DECISION: was SEARCH the bottleneck, or was si=300 an outlier? ===')
    ok = []
    for tag, _, _ in blocks:
        d = out[tag]['contrasts']['FULL-topk6']
        surv = d['mean'] >= 0.05 and d['lo'] > 0
        ok.append(surv)
        print(f"  {tag}: FULL - topK6 = {d['mean']:+.4f} CI "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}]  -> narrative {'holds' if surv else 'FAILS'} "
              f"on this block (bar: >= +0.05, CI excl 0)")
    verdict = ('SEARCH was the bottleneck -- "top-K6 manufactured DEAD" SURVIVES'
               if all(ok) else
               'RETRACT PERMANENTLY: search was NOT the bottleneck; si=300 was an '
               'outlier block. My causal story was wrong.')
    print(f'\n  -> {verdict}')
    print('  (Either way blind stays ~0: si=400 fresh + full enumeration = -0.0067.)')
    out['verdict'] = verdict
    out['narrative_survives'] = bool(all(ok))
    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
