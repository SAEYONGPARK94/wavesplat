"""PURE-V2 arm sanity statistics -- SINGLE ARM ONLY.

Deliberately computes NOTHING that involves another arm: no deltas, no oracle, no
bootstrap, no bar verdicts.  Those belong to the single scoring event in the GPU phase
(tmp/heldout40_2026-07-26/PREREG.md Sec 5).  What this prints is exactly what a run-log
would print about one arm: coverage, errors, runtime, and the arm's own per-regime
median-of-5 mean.

  python3 tmp/heldout40_2026-07-26/v2_sanity.py
"""
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, 'v2_arm.jsonl')
SEEDS = [42, 1, 2, 3, 4]


def comp(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


def slice_of(ep):
    c = comp(ep)
    return 'sparse' if c in (1, 2) else ('moderate' if c == 5 else 'crowded')


def main():
    rows = [json.loads(l) for l in open(JSONL)]
    errs = [r for r in rows if 'error' in r]
    ok = [r for r in rows if 'error' not in r]

    by_ep = {}
    for r in ok:
        by_ep.setdefault(r['episode'], {})[r['seed']] = r

    print(f"rows            : {len(rows)}  (ok {len(ok)}, errors {len(errs)})")
    print(f"episodes covered: {len(by_ep)} / 40")
    incomplete = {e: sorted(v) for e, v in by_ep.items() if len(v) != 5}
    print(f"episodes with != 5 lanes: {len(incomplete)}")
    for e, s in incomplete.items():
        print(f"   {e}: seeds {s}")
    for r in errs:
        print(f"   ERROR {r['job_id']}: {r['error'][:140]}")

    walls = [r['wall_s'] for r in ok]
    fr_ms = [r['elapsed_ms'] for r in ok if r.get('elapsed_ms') == r.get('elapsed_ms')]
    if walls:
        print(f"wall per run    : median {st.median(walls):.1f} s, "
              f"min {min(walls):.1f}, max {max(walls):.1f}, total {sum(walls)/3600:.2f} core-h")
    if fr_ms:
        print(f"per-frame       : median {st.median(fr_ms):.2f} ms")

    print("\nper-episode median-of-5 nRMSE (this arm only):")
    med = {}
    for e, v in sorted(by_ep.items()):
        vals = [v[s]['nRMSE'] for s in SEEDS if s in v]
        med[e] = st.median(vals)
        print(f"  {e:<46s} {slice_of(e):<9s} {med[e]:.4f}  (n_lane={len(vals)})")

    print("\npure-v2 arm, per-regime mean of per-episode medians (single arm, no comparison):")
    for sl in ('sparse', 'moderate', 'crowded'):
        xs = [v for e, v in med.items() if slice_of(e) == sl]
        if xs:
            print(f"  {sl:<9s} n={len(xs):2d}  mean {sum(xs)/len(xs):.4f}  "
                  f"median {st.median(xs):.4f}  min {min(xs):.4f}  max {max(xs):.4f}")
    allv = list(med.values())
    print(f"  {'ALL40':<9s} n={len(allv):2d}  mean {sum(allv)/len(allv):.4f}  "
          f"median {st.median(allv):.4f}")

    det = [r['spec']['detection_rate'] for r in ok
           if isinstance(r.get('spec'), dict) and 'detection_rate' in r['spec']]
    if det:
        print(f"\nspectral detection_rate (this arm): median {st.median(det):.3f}, "
              f"n={len(det)}")


if __name__ == '__main__':
    main()
