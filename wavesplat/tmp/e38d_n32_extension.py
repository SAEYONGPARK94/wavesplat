"""e38d — n=32 extension of the two decisive e38b contrasts (seeds 16..31):
  (1) H(T4n4, bank201) - H(base, bank51)   [does the prize persist/grow?]
  (2) X_eb at T4n4 bank201                  [is any of it extractable?]
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import DEPTH, KREL
from e30_conditionality_probe import radial_of
from e38_aperture_sweep import build_sea_ap
from e38b_bank_control import fine_mask, deposit_fine, ens_fine, solve_fine
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top

OUT = 'results/direction3/e38d_n32_extension.json'


def arms(S, kn, en, nf):
    KX, KY, m = fine_mask(nf)
    kp = S['kp']
    bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                          DEPTH, klim=3.5 * kp, n=120)
    mu_hat = np.arctan2(by, bx)
    g_ens = ens_fine(kn, en, mu_hat, kp, 30.0, nf)
    g_real = deposit_fine(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp, nf)
    _, _, pf = beamform_power(S['P'], S['z'], DEPTH, KREL * kp, nf)
    pf = np.maximum((pf / max(pf.max(), 1e-12)).reshape(nf, nf), 1e-8)
    return dict(ens=solve_fine(S, g_ens, KX, KY, m, kp),
                real=solve_fine(S, g_real, KX, KY, m, kp),
                eb=solve_fine(S, g_ens * pf ** 0.5, KX, KY, m, kp))


def main(si0=16, si1=32):
    kn, en = radial_of(7.0, 3.3)
    rows = []
    for si in range(si0, si1):
        Sb = build_sea_ap(si, 70.0, 400)
        rb = arms(Sb, kn, en, 51)
        St = build_sea_ap(si, 280.0, 1600)
        rt = arms(St, kn, en, 201)
        rows.append(dict(si=si,
                         b_ens=rb['ens'], b_real=rb['real'], b_eb=rb['eb'],
                         t_ens=rt['ens'], t_real=rt['real'], t_eb=rt['eb']))
        r = rows[-1]
        print(f"sea{si:2d} base51: ens {r['b_ens']:+.3f} real {r['b_real']:+.3f} "
              f"eb {r['b_eb']:+.3f} | T4n4x201: ens {r['t_ens']:+.3f} "
              f"real {r['t_real']:+.3f} eb {r['t_eb']:+.3f}", flush=True)
        json.dump(rows, open(OUT, 'w'), indent=1)
    a = lambda k: np.array([x[k] for x in rows], float)
    Hb = a('b_real') - a('b_ens'); Ht = a('t_real') - a('t_ens')
    Xb = a('b_eb') - a('b_ens'); Xt = a('t_eb') - a('t_ens')
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n(new seeds only, n={len(rows)})")
    print(f"  H base51 {Hb.mean():+.4f}±{se(Hb):.4f} | H T4n4x201 {Ht.mean():+.4f}±{se(Ht):.4f} "
          f"| dH {(Ht-Hb).mean():+.4f}±{se(Ht-Hb):.4f}")
    print(f"  X_eb base51 {Xb.mean():+.4f}±{se(Xb):.4f} | X_eb T4n4x201 {Xt.mean():+.4f}±{se(Xt):.4f}")


if __name__ == '__main__':
    main()
