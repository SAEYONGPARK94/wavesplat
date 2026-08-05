"""Opus independent fairness control on design C's gate-0 ceiling.

C's oracle_field sweeps blur{0,1} x smult{0.5,1,3,10} and picks the best BY GT
CORR (per sea). The incumbent `cur` gets only its learned SmultHead. So part of
the reported headroom (+0.226) could be smult tuning rather than prior knowledge.

This control re-measures the ceiling with NO per-sea GT selection:
  - oracle_fixed  : blur=0, smult=3.0 fixed for every sea (modal choice)
  - oracle_fixed1 : blur=0, smult=1.0 fixed for every sea
  - oracle_best   : C's GT-selected version (reproduction)
The honest headroom is oracle_fixed - cur. If that collapses, C's headline dies.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import build_sea_c, oracle_field, corr

OUT = 'results/direction3/e29_opus_gate0_fairness.json'


def main(n_seas=16):
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        o_best, ob, osm = oracle_field(S)                              # C's version
        o_fix3, _, _ = oracle_field(S, blurs=(0.0,), smults=(3.0,))    # fixed 3
        o_fix1, _, _ = oracle_field(S, blurs=(0.0,), smults=(1.0,))    # fixed 1
        cur = ref[si]['cur']
        rows.append(dict(si=si, cur=cur, o_best=float(o_best),
                         o_fix3=float(o_fix3), o_fix1=float(o_fix1),
                         sm=float(osm),
                         h_best=float(o_best - cur),
                         h_fix3=float(o_fix3 - cur),
                         h_fix1=float(o_fix1 - cur)))
        r = rows[-1]
        print(f"sea{si:2d} cur {cur:+.3f} | o_best {o_best:+.3f}(sm{osm:g}) "
              f"h {r['h_best']:+.3f} | o_fix3 {o_fix3:+.3f} h {r['h_fix3']:+.3f} "
              f"| o_fix1 {o_fix1:+.3f} h {r['h_fix1']:+.3f}", flush=True)

    a = lambda k: np.array([r[k] for r in rows])
    print("\n=== Opus fairness control (n=%d) ===" % len(rows))
    print(f"  C 재현      h_best : mean {a('h_best').mean():+.4f} "
          f"median {np.median(a('h_best')):+.4f} pos {int((a('h_best')>0).sum())}/{len(rows)}")
    print(f"  고정 smult=3 h_fix3 : mean {a('h_fix3').mean():+.4f} "
          f"median {np.median(a('h_fix3')):+.4f} pos {int((a('h_fix3')>0).sum())}/{len(rows)}")
    print(f"  고정 smult=1 h_fix1 : mean {a('h_fix1').mean():+.4f} "
          f"median {np.median(a('h_fix1')):+.4f} pos {int((a('h_fix1')>0).sum())}/{len(rows)}")
    infl3 = a('h_best').mean() - a('h_fix3').mean()
    print(f"\n  ★GT-peek 인플레 (best - fix3) = {infl3:+.4f}  "
          f"(C 자체 추정 +0.02~0.03)")
    print(f"  ★정직한 headroom (fix3) = {a('h_fix3').mean():+.4f}")
    json.dump(dict(rows=rows, h_best=float(a('h_best').mean()),
                   h_fix3=float(a('h_fix3').mean()),
                   h_fix1=float(a('h_fix1').mean()),
                   inflation=float(infl3)), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
