"""E-30e: the honest survivor probe. cur loses BADLY on some seas (sea1 -0.122
vs hand ens +0.061; sea6 +0.125 vs ens_rob +0.388). If a zero-training blend of
the learned gamma with the hand ensemble gamma beats the incumbent, the
direction is OPEN and no amount of ceiling-talk matters.

Arms (all through the SAME matched solver: free arm, smult=3 -- so net_s3 is the
apples-to-apples learned baseline, and cur is the full deployed path):
  net_s3        : incumbent gamma, matched solver
  ens_rob       : hand universal-radial x N(mu_robust, 30deg)
  geo(w)        : gamma_net^(1-w) * gamma_ens^w     (natural: net emits log-gamma)
  ari(w)        : (1-w)*gamma_net + w*gamma_ens
  ORACLE_pick   : per-sea max(net_s3, ens_rob)  -- the gating CEILING (E-25/E-28
                  showed the gate itself is unlearnable; this bounds it anyway)
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import build_sea_c, REGIMES, _MASK
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from e30_speckle_control import bf_grid
from e30_dof_and_bank import net_gamma_grid
from e30_exchange_rate import robust_mu
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/e30_blend_probe.json'
WS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        cur, kp = ref[si]['cur'], S['kp']
        mu_rob = robust_mu(bf_grid(S['P'], S['z'], kp))
        ens_g = ensemble_grid(kmag_u, en_u, mu_rob, kp)
        net_g = net_gamma_grid(S)
        nz = lambda g: np.maximum(g, 1e-12) / max(np.maximum(g, 0).sum(), 1e-12)
        ng, eg = nz(net_g), nz(ens_g)
        geo, ari = {}, {}
        for w in WS:
            geo[f'{w:g}'] = solve_with_grid(S, np.exp((1 - w) * np.log(ng) + w * np.log(eg)))
            ari[f'{w:g}'] = solve_with_grid(S, (1 - w) * ng + w * eg)
        rows.append(dict(si=si, cur=cur, net_s3=geo['0'], ens_rob=geo['1'],
                         geo=geo, ari=ari))
        print(f"sea{si:2d} cur {cur:+.3f} net_s3 {geo['0']:+.3f} ens_rob {geo['1']:+.3f} "
              f"| geo " + " ".join(f"{w:g}:{geo[f'{w:g}']:+.3f}" for w in WS), flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur, net_s3 = a('cur'), a('net_s3')

    def stat(d):
        m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        bs = np.array([np.random.default_rng(s).choice(d, len(d)).mean() for s in range(2000)])
        return m, (m / se if se > 0 else 0), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d > 0).sum())

    print(f"\n=== E-30e blend probe (n={len(rows)}) ===")
    print(f"  matched baseline net_s3 {net_s3.mean():+.4f} | 배포 cur {cur.mean():+.4f} "
          f"(차 {(net_s3-cur).mean():+.4f} = SmultHead+구속파팔의 값)")
    for tag, D in (('geo', 'geo'), ('ari', 'ari')):
        print(f"  --- {tag} blend ---")
        for w in WS:
            v = np.array([r[D][f'{w:g}'] for r in rows])
            m, t, lo, hi, npos = stat(v - cur)
            m2 = (v - net_s3).mean()
            print(f"   w={w:4.2f}: {v.mean():+.4f} | vs cur {m:+.4f} t{t:+.2f} "
                  f"CI[{lo:+.3f},{hi:+.3f}] 양수 {npos}/{len(rows)} | vs net_s3 {m2:+.4f}")
    pick = np.maximum(net_s3, a('ens_rob'))
    m, t, lo, hi, npos = stat(pick - cur)
    print(f"\n  ★ORACLE_pick max(net_s3, ens_rob): {pick.mean():+.4f} vs cur {m:+.4f} "
          f"t{t:+.2f} CI[{lo:+.3f},{hi:+.3f}] 양수 {npos}/{len(rows)}")
    print(f"     (= 게이팅의 천장. E-25 하드게이트/E-28 소프트게이트·FiLM 둘 다 실현 실패)")
    json.dump(dict(rows=rows, ws=WS), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
