"""DESIGN-PHASE (pre-freeze): cross-reference the routing-relevant subset.

The guard only fires when the proposal says 'many' (routes learned). So the question is
NOT 'separate all sparse from all crowded' but: among episodes that route learned
(median n_sel >= 8), does an observation-derived beamform signal separate the truly-sparse
MISROUTES (comp 1,2) from the genuine CROWDED (comp 15,30) we must not touch?

Combines explore_signals.json (per record atom signals) + explore_beamform.json (per episode
beamform signals). Uses per-episode median n_sel as the routing signal (Stage-1 convention).
"""
import json, re
import numpy as np

def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))

sig = json.load(open('tmp/unified_guard_2026-07-26/explore_signals.json'))
bf = {r['ep']: r for r in json.load(open('tmp/unified_guard_2026-07-26/explore_beamform.json'))}

# per-episode median n_sel and median dom_frac(atoms)
by_ep = {}
for r in sig:
    by_ep.setdefault(r['ep'], []).append(r)
ep_rows = []
for ep, recs in by_ep.items():
    ns = [r['n_sel'] for r in recs]
    b = bf[ep]
    ep_rows.append(dict(ep=ep, comp=comp(ep), ns_med=float(np.median(ns)),
                        ns_min=min(ns), ns_max=max(ns),
                        npk10=b.get('npk_10'), npk20=b.get('npk_20'), npk30=b.get('npk_30'),
                        p2p1=b.get('p2_over_p1'), p3p1=b.get('p3_over_p1'), p8p1=b.get('p8_over_p1')))

THRESH = 8
routed = [r for r in ep_rows if r['ns_med'] >= THRESH]
print(f"Episodes routing LEARNED (median n_sel >= {THRESH}): {len(routed)}")
sp = [r for r in routed if r['comp'] in (1,2)]
cr = [r for r in routed if r['comp'] in (15,30)]
mo = [r for r in routed if r['comp'] == 5]
print(f"  of which sparse(1,2)={len(sp)} [MISROUTES], crowded(15,30)={len(cr)} [KEEP], moderate5={len(mo)}")

print("\n--- MISROUTES (sparse, routed learned) beamform signals ---")
for r in sorted(sp, key=lambda x:x['ep']):
    print(f"  {r['ep']:34s} c{r['comp']} ns_med={r['ns_med']:.0f} npk10={r['npk10']} npk20={r['npk20']} npk30={r['npk30']} p2/p1={r['p2p1']:.3f} p8/p1={r['p8p1']:.3f}")

print("\n--- CROWDED (routed learned) beamform signal ranges (must NOT be vetoed) ---")
for key in ('npk10','npk20','npk30','p2p1','p3p1','p8p1'):
    vals = [r[key] for r in cr]
    print(f"  {key}: min={min(vals):.3f} p10={np.percentile(vals,10):.3f} median={np.median(vals):.3f} max={max(vals):.3f}")

print("\n--- SEPARATION TEST: candidate guard signals ---")
# Guard fires (veto->v2) if signal says 'few dominant peaks'. Test each threshold.
for key, ths in [('npk10',(1,2,3)),('npk20',(1,2)),('npk30',(1,)),('p2p1',(0.1,0.15,0.2,0.25,0.3))]:
    for th in ths:
        # veto if signal <= th (for counts) / < th (for ratios)
        if key.startswith('npk'):
            vetoed_sp = [r for r in sp if r[key] <= th]
            vetoed_cr = [r for r in cr if r[key] <= th]
            desc = f"{key} <= {th}"
        else:
            vetoed_sp = [r for r in sp if r[key] < th]
            vetoed_cr = [r for r in cr if r[key] < th]
            desc = f"{key} < {th}"
        print(f"  veto if {desc:14s}: catches {len(vetoed_sp)}/{len(sp)} misroutes, "
              f"FALSE-vetoes {len(vetoed_cr)}/{len(cr)} crowded "
              f"{'  <-- CLEAN' if len(vetoed_cr)==0 and len(vetoed_sp)>0 else ''}")

# also: moderate 5comp routed learned -- how would guard treat them?
print("\n--- MODERATE (5comp) routed learned: signals (context, not pass/fail) ---")
for r in sorted(mo, key=lambda x:x['ep']):
    print(f"  {r['ep']:34s} ns_med={r['ns_med']:.0f} npk10={r['npk10']} p2/p1={r['p2p1']:.3f}")
