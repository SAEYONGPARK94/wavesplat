"""e38c — LOOKS-AS-CURRENCY test at 4x observation budget.

e38b: at T=280s/n=1600 with a fair bank the oracle realization headroom GROWS
(+0.216 -> +0.378) but single-look data priors still extract ~0.

Agent 2's mechanism (speckle ~ looks) makes a POSITIVE prediction: with 4x
budget you can form L=4 independent 70-s looks, each at base resolution;
multilook averaging cuts speckle var by 4, so the admix prior's gain should
scale up from the base-arena +0.012 (e35, L2p0.5) toward ~+0.04-0.05.
If instead the gain stays ~+0.01, looks do not cash either and the wall is
deeper (constant prior ~ Bayes for anything measurable from z).

Arms at T=280s / n=1600 / bank101, smult=3 fixed:
  ens          constant hand prior
  real         oracle realization (ceiling)
  mlXp025/05   ens * (multilook_L4 map)^p, p in {0.25, 0.5}
  ml1          pure multilook map ^1
  sl05         ens * (single-look full map)^0.5   (= e38b eb101, for pairing)
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import corr, DEPTH, KREL
from e30_conditionality_probe import radial_of
from e38_aperture_sweep import build_sea_ap
from e38b_bank_control import fine_mask, deposit_fine, ens_fine, solve_fine
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top

OUT = 'results/direction3/e38c_multilook_budget.json'
NF = 101
L = 4


def multilook_fine(P, z, kp, nf, looks):
    order = np.argsort(P[:, 2])
    acc = np.zeros(nf * nf)
    for idx in np.array_split(order, looks):
        _, _, p = beamform_power(P[idx], z[idx], DEPTH, KREL * kp, nf)
        acc += p / max(p.max(), 1e-12)
    return (acc / looks).reshape(nf, nf)


def main(n_seas=16):
    kn, en = radial_of(7.0, 3.3)
    KX, KY, m = fine_mask(NF)
    rows = []
    for si in range(n_seas):
        S = build_sea_ap(si, 280.0, 1600)
        kp = S['kp']
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * kp, n=120)
        mu_hat = np.arctan2(by, bx)
        g_ens = ens_fine(kn, en, mu_hat, kp, 30.0, NF)
        g_real = deposit_fine(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp, NF)
        ml = np.maximum(multilook_fine(S['P'], S['z'], kp, NF, L), 1e-8)
        _, _, sl = beamform_power(S['P'], S['z'], DEPTH, KREL * kp, NF)
        sl = np.maximum((sl / max(sl.max(), 1e-12)).reshape(NF, NF), 1e-8)
        r = dict(si=si)
        r['ens'] = solve_fine(S, g_ens, KX, KY, m, kp)
        r['real'] = solve_fine(S, g_real, KX, KY, m, kp)
        r['ml025'] = solve_fine(S, g_ens * ml ** 0.25, KX, KY, m, kp)
        r['ml05'] = solve_fine(S, g_ens * ml ** 0.5, KX, KY, m, kp)
        r['ml1'] = solve_fine(S, ml, KX, KY, m, kp)
        r['sl05'] = solve_fine(S, g_ens * sl ** 0.5, KX, KY, m, kp)
        rows.append(r)
        print(f"sea{si:2d} ens {r['ens']:+.3f} real {r['real']:+.3f} "
              f"ml025 {r['ml025']:+.3f} ml05 {r['ml05']:+.3f} "
              f"ml1 {r['ml1']:+.3f} sl05 {r['sl05']:+.3f}", flush=True)
    a = lambda k: np.array([x[k] for x in rows], float)
    ens = a('ens')
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n=== e38c (T=280s n=1600 bank101, L={L} looks) n={n_seas} ===")
    print(f"  ens {ens.mean():+.4f}  real {a('real').mean():+.4f} "
          f"(H {(a('real')-ens).mean():+.4f})")
    for k in ('ml025', 'ml05', 'ml1', 'sl05'):
        d = a(k) - ens
        print(f"  {k:6s} - ens: {d.mean():+.4f} ± {se(d):.4f} "
              f"(t {d.mean()/se(d):+.2f}, pos {(d>0).sum()}/{len(d)})")
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
