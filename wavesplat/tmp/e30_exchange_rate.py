"""E-30d: the EXCHANGE RATE + the direction thread.

(1) EXCHANGE RATE. rung4 - rung2 is knowing the 200 realized angles. Interpolate
    honestly: for a fraction f of the 200 modes use the TRUE angle, for the rest
    a fresh N(mu_true,30deg) draw. f=0 -> PERM, f=1 -> rung4. Measure field corr
    AND the speckle partial-corr p(f). This converts every arm into a common
    currency: "how many of the 200 components must you correctly identify to
    gain X corr?" Then read off where the incumbent (p_net=+0.168) and the raw
    periodogram (p_bf=+0.069) actually sit.

(2) DIRECTION. sea6's 158.6deg error has flip_ratio 0.06 (the data prefers the
    TRUE side by 16x) => beamform_top slipped; it is not an ambiguity. Does a
    robust mu_hat (annulus power-weighted circular centroid) close rung3-rung2,
    and does the resulting HAND prior beat the incumbent?
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, corr, DEPTH, _MASK,
                          REGIMES, _KXN, _KYN)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from e30_speckle_control import bf_grid
from e30_dof_and_bank import pcorr, net_gamma_grid
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/e30_exchange_rate.json'
FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def robust_mu(bf_g):
    """annulus power-weighted circular centroid of the beamform map."""
    w = np.maximum(bf_g[_MASK], 0) ** 2
    th = np.arctan2(_KYN[_MASK], _KXN[_MASK])
    return float(np.angle(np.sum(w * np.exp(1j * th))))


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    lad = {r['si']: r for r in
           json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']}
    R = REGIMES['spread']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        cur, kp = ref[si]['cur'], S['kp']
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        kmag = np.hypot(S['m_kx'], S['m_ky'])
        th_true = np.arctan2(S['m_ky'], S['m_kx'])
        en = S['m_A'] ** 2 / 2
        gt_g = deposit_grid(S['m_kx'], S['m_ky'], en, kp)
        bf_g = bf_grid(S['P'], S['z'], kp)
        mu_rob = robust_mu(bf_g)
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * kp, n=120)
        mu_top = np.arctan2(by, bx)
        ens_top = ensemble_grid(kmag_u, en_u, mu_top, kp)
        ens_rob = ensemble_grid(kmag_u, en_u, mu_rob, kp)
        r_rob = solve_with_grid(S, ens_rob)
        # --- (1) exchange-rate ladder
        f_corr, f_p = {}, {}
        for f in FRACS:
            cs, ps = [], []
            for rep in range(3):
                pr = np.random.default_rng(50000 + 13 * si + rep)
                keep = pr.random(len(kmag)) < f
                th = np.where(keep, th_true, pr.normal(mu_true, np.deg2rad(30.0), len(kmag)))
                g = deposit_grid(kmag * np.cos(th), kmag * np.sin(th), en, kp)
                cs.append(solve_with_grid(S, g))
                ps.append(pcorr(g[_MASK].astype(float), gt_g[_MASK].astype(float),
                                ens_top[_MASK].astype(float)))
            f_corr[f'{f:g}'] = float(np.mean(cs))
            f_p[f'{f:g}'] = float(np.mean(ps))
        net_g = net_gamma_grid(S)
        p_net = pcorr(net_g[_MASK].astype(float), gt_g[_MASK].astype(float),
                      ens_top[_MASK].astype(float))
        derr_rob = np.rad2deg(abs(np.angle(np.exp(1j * (mu_rob - mu_true)))))
        rows.append(dict(si=si, cur=cur, ens_top=lad[si]['ens_blind'],
                         ens_rob=r_rob, ens_oracle=lad[si]['ens_oracle'],
                         real_free=lad[si]['real_free'], p_net=p_net,
                         derr_top=lad[si]['dir_err_deg'], derr_rob=float(derr_rob),
                         f_corr=f_corr, f_p=f_p))
        print(f"sea{si:2d} cur {cur:+.3f} | ens_top {lad[si]['ens_blind']:+.3f}"
              f"({lad[si]['dir_err_deg']:5.1f}d) ens_rob {r_rob:+.3f}({derr_rob:5.1f}d) "
              f"ens_or {lad[si]['ens_oracle']:+.3f} | f: "
              + " ".join(f"{f:g}:{f_corr[f'{f:g}']:+.2f}" for f in FRACS), flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')

    def stat(d):
        m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else 0.0
        bs = np.array([np.random.default_rng(s).choice(d, len(d)).mean()
                       for s in range(2000)])
        return m, t, np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d > 0).sum())

    print(f"\n=== E-30d (n={len(rows)}) ===")
    print("(2) 방향 thread:")
    for k, lab in [('ens_top', 'ens_blind (beamform_top mu)'),
                   ('ens_rob', 'ens_robust (annulus centroid mu)'),
                   ('ens_oracle', 'ens_oracle (mu_true)')]:
        m, t, lo, hi, npos = stat(a(k) - cur)
        print(f"   {lab:34s} {a(k).mean():+.4f} | vs cur {m:+.4f} t{t:+.2f} "
              f"CI[{lo:+.3f},{hi:+.3f}] 양수 {npos}/{len(rows)}")
    print(f"   방향오차: top {a('derr_top').mean():5.1f}deg -> robust {a('derr_rob').mean():5.1f}deg "
          f"(>90deg: top {int((a('derr_top')>90).sum())}, robust {int((a('derr_rob')>90).sum())})")

    print(f"\n(1) 교환율 — 200성분 중 f 비율을 '정확히 식별'했을 때:")
    print(f"   {'f':>5} {'field corr':>11} {'vs cur':>9} {'speckle p':>10}")
    for f in FRACS:
        c = np.array([r['f_corr'][f'{f:g}'] for r in rows])
        p = np.array([r['f_p'][f'{f:g}'] for r in rows])
        print(f"   {f:5.2f} {c.mean():+11.4f} {(c-cur).mean():+9.4f} {p.mean():+10.4f}")
    pn = a('p_net').mean()
    fp = np.array([np.mean([r['f_p'][f'{f:g}'] for r in rows]) for f in FRACS])
    fc = np.array([np.mean([r['f_corr'][f'{f:g}'] for r in rows]) for f in FRACS])
    f_at_net = np.interp(pn, fp, FRACS)
    print(f"\n   ★현 net의 speckle 회수 p_net = {pn:+.4f}  ->  f_equiv = {f_at_net:.3f} "
          f"(= 200개 중 {f_at_net*200:.0f}개 상당)")
    f_cross = np.interp(0.0, fc - cur.mean(), FRACS)
    print(f"   ★f-사다리가 cur를 넘는 지점: f = {f_cross:.3f} (= {f_cross*200:.0f}/200 성분 정확 식별)")
    print(f"      -> p 기준 필요치 = {np.interp(f_cross, FRACS, fp):+.4f} (현 net {pn:+.4f})")
    for tgt in (0.05, 0.10):
        fx = np.interp(tgt, fc - cur.mean(), FRACS)
        print(f"   +{tgt:.2f} corr 얻으려면: f = {fx:.3f} ({fx*200:.0f}/200), "
              f"p = {np.interp(fx, FRACS, fp):+.4f}")
    json.dump(dict(rows=rows, fracs=FRACS), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
