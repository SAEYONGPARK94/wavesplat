"""E-30f: power. The blend (+0.013, CI[-0.027,+0.061]) is the ONLY arm that
came out above the incumbent, and n=16 cannot resolve it (MDE ~ +-0.066 --
the same power trap the E-28 audit caught at n=10). Extend to n=32 by
computing `cur` directly (self-checked against e29_gate0_ceiling.json on the
first 16 seas, which must match bit-for-bit).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import build_sea_c, incumbent_field, corr, _MASK
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from e30_speckle_control import bf_grid
from e30_dof_and_bank import net_gamma_grid
from e30_exchange_rate import robust_mu

OUT = 'results/direction3/e30_blend_n32.json'
WS = [0.25, 0.5]


def main(n_seas=32):
    kmag_u, en_u = universal_radial()
    ref = {i: r['cur'] for i, r in enumerate(
           json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows'])}
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        kp = S['kp']
        fq = incumbent_field(S['P'], S['z'], S['Pq'], kp, S['sigma0'])
        cur = float(corr(fq, S['gt_u']))
        chk = ''
        if si in ref:
            d = abs(cur - ref[si])
            chk = f" [self-check vs ceiling.json: {d:.2e}]"
            assert d < 2e-6, f"cur mismatch sea{si}: {cur} vs {ref[si]}"
        ens_g = ensemble_grid(kmag_u, en_u, robust_mu(bf_grid(S['P'], S['z'], kp)), kp)
        net_g = net_gamma_grid(S)
        nz = lambda g: np.maximum(g, 1e-12) / max(np.maximum(g, 0).sum(), 1e-12)
        ng, eg = nz(net_g), nz(ens_g)
        r = dict(si=si, cur=cur, net_s3=solve_with_grid(S, ng),
                 ens_rob=solve_with_grid(S, eg))
        for w in WS:
            r[f'geo{w:g}'] = solve_with_grid(S, np.exp((1 - w) * np.log(ng) + w * np.log(eg)))
        rows.append(r)
        print(f"sea{si:2d} cur {cur:+.3f} net_s3 {r['net_s3']:+.3f} ens {r['ens_rob']:+.3f} "
              f"geo.25 {r['geo0.25']:+.3f} geo.5 {r['geo0.5']:+.3f}{chk}", flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')

    def stat(d):
        m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        bs = np.array([np.random.default_rng(s).choice(d, len(d)).mean() for s in range(4000)])
        mde = 2.9 * se
        return m, (m / se if se > 0 else 0), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d > 0).sum()), mde

    print(f"\n=== E-30f blend @ n={len(rows)} ===")
    for k in ('net_s3', 'ens_rob', 'geo0.25', 'geo0.5'):
        m, t, lo, hi, npos, mde = stat(a(k) - cur)
        print(f"  {k:9s} {a(k).mean():+.4f} | vs cur {m:+.4f} t{t:+.2f} "
              f"CI[{lo:+.3f},{hi:+.3f}] 양수 {npos}/{len(rows)} | MDE +-{mde:.3f}")
    pick = np.maximum(a('net_s3'), a('ens_rob'))
    m, t, lo, hi, npos, mde = stat(pick - cur)
    print(f"  ★ORACLE_pick {pick.mean():+.4f} | vs cur {m:+.4f} t{t:+.2f} "
          f"CI[{lo:+.3f},{hi:+.3f}] 양수 {npos}/{len(rows)} | MDE +-{mde:.3f}")
    json.dump(dict(rows=rows), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
