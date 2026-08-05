"""e29 CLOSURE red-team checks (Opus, 2026-07-16).

Candidate-killing / candidate-sparing measurements, all inference-only:
 [S] ladder stats: paired t on rung2-cur; rung3-rung2 with/without sea6, sea1
 [B] rung 2.5 "beamform-directional ensemble": universal radial x DATA-DRIVEN
     directional profile w(theta) from the two-leg min beamform map (the exact
     input a prior net sees). Variants p in {1,2,4} + Gaussian-windowed.
     -> upper bound on directional realization info a net could learn to use.
 [F] 180-flip arm: ens grid at mu_hat vs mu_hat+180, selected BLIND by
     (i) cross-leg predictive error, (ii) log-evidence on the 400 obs.
     -> is the sea6-type flip fixable without GT? candidate-2 verdict.
 [X] exact-atom oracle: diagonal gamma at the TRUE 200 (kx,ky[,2k bw]) atoms
     with true energies (off-grid, no deposit) vs rung4/4b grid deposit.
     -> is the 51x51 bank a binding constraint even for the oracle? cand-3.
 [W] bw_sc inflation factor + bound-wave variance share per sea. cand-4.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')

from e29_gate0_ladder import (universal_radial, ensemble_grid, solve_with_grid,
                              SMULT, SPREAD_DEG, NTH)
from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, REGIMES, maps6,
                          KXN_NP, KYN_NP, NB, NGRID)
from d18_relax_estimator import beamform_top
from d3_sa_nomp import split_masks
from d3_sa_fim_identifiability import omega_of
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/e29_opus_close_checks.json'
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


def solve_grid_general(S, g_free_grid, fit_idx=None, Q=None, g_bw_grid=None,
                       smult=SMULT, want_evidence=False):
    """gate-0 ladder solver, generalized to obs subsets / arbitrary query."""
    P, z = S['P'], S['z']
    if fit_idx is not None:
        P, z = P[fit_idx], z[fit_idx]
    if Q is None:
        Q = S['Pq']
    kp, s0 = S['kp'], S['sigma0']
    gb = np.zeros_like(g_free_grid) if g_bw_grid is None else g_bw_grid
    g = np.concatenate([g_free_grid[_MASK], gb[_MASK]])
    g = g / g.sum()
    kxb, kyb, om = bank_atoms(kp)
    Phi = design_t(T(kxb), T(kyb), T(om), T(P))
    Phi_q = design_t(T(kxb), T(kyb), T(om), T(Q))
    gam = T(np.concatenate([g, g]) * float(np.var(S['z'])))
    K = (Phi * gam[None]) @ Phi.T + (smult * s0) ** 2 * torch.eye(
        len(z), device=DEVICE)
    zt = T(z)
    w = Phi.T @ torch.linalg.solve(K, zt)
    pred = ((Phi_q * gam[None]) @ w).cpu().numpy()
    if not want_evidence:
        return pred
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(zt[:, None], L)[:, 0]
    ev = float(-0.5 * (zt @ alpha) - torch.log(torch.diag(L)).sum())
    return pred, ev


def dir_profile(S, kp, nbins=90):
    """directional weight from the two-leg MIN beamform map (shipped input),
    ring 0.6-1.6 k/kp."""
    pm_grid = maps6(S['P'], S['z'], kp)[2]             # (51,51) min(p1,p2)
    th = np.arctan2(KYN_NP, KXN_NP)
    km = np.hypot(KXN_NP, KYN_NP)
    ring = (km > 0.6) & (km < 1.6)
    vals = pm_grid[_MASK]                              # aligned with KXN_NP
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    h = np.histogram(th[ring], bins=bins, weights=vals[ring])[0]
    c = np.histogram(th[ring], bins=bins)[0]
    w = h / np.maximum(c, 1)
    thc = 0.5 * (bins[1:] + bins[:-1])
    return thc, w


def dirgrid(kmag_u, en_u, thc, w, kp):
    w = w / w.sum()
    kxs = np.concatenate([k * np.cos(thc) for k in kmag_u])
    kys = np.concatenate([k * np.sin(thc) for k in kmag_u])
    ens = np.concatenate([e * w for e in en_u])
    return deposit_grid(kxs, kys, ens, kp)


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    lad = json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']
    R = REGIMES['spread']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        kp = S['kp']
        row = dict(si=si, cur=lad[si]['cur'], ens_blind=lad[si]['ens_blind'],
                   ens_oracle=lad[si]['ens_oracle'],
                   real_free=lad[si]['real_free'],
                   dir_err=lad[si]['dir_err_deg'])
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32), DEPTH,
                              klim=3.5 * kp, n=120)
        mu_hat = np.arctan2(by, bx)

        # ---- [B] rung 2.5 directional-profile arms -----------------------
        thc, w0 = dir_profile(S, kp)
        for tag, wv in [('dp1', w0), ('dp2', w0 ** 2), ('dp4', w0 ** 4)]:
            g = dirgrid(kmag_u, en_u, thc, wv, kp)
            row[tag] = corr(solve_grid_general(S, g), S['gt_u'])
        gwin = np.exp(-0.5 * ((np.angle(np.exp(1j * (thc - mu_hat))))
                              / np.deg2rad(45)) ** 2)
        g = dirgrid(kmag_u, en_u, thc, w0 ** 2 * gwin, kp)
        row['dp2w'] = corr(solve_grid_general(S, g), S['gt_u'])

        # ---- [F] blind 180-flip selection --------------------------------
        m1, m2 = split_masks(S['P'], mode='legs')
        i1, i2 = np.where(m1)[0], np.where(m2)[0]
        cand = {}
        for tag, mu in [('fwd', mu_hat), ('rev', mu_hat + np.pi)]:
            g = ensemble_grid(kmag_u, en_u, mu, kp)
            p12 = solve_grid_general(S, g, fit_idx=i1, Q=S['P'][i2])
            p21 = solve_grid_general(S, g, fit_idx=i2, Q=S['P'][i1])
            xerr = float(((p12 - S['z'][i2]) ** 2).mean()
                         + ((p21 - S['z'][i1]) ** 2).mean())
            _, ev = solve_grid_general(S, g, want_evidence=True)
            cand[tag] = dict(g=g, xerr=xerr, ev=ev,
                             corr=corr(solve_grid_general(S, g), S['gt_u']))
        pick_x = 'fwd' if cand['fwd']['xerr'] <= cand['rev']['xerr'] else 'rev'
        pick_e = 'fwd' if cand['fwd']['ev'] >= cand['rev']['ev'] else 'rev'
        true_pick = 'fwd' if abs(np.angle(np.exp(1j * (mu_hat - mu_true)))) \
            <= np.pi / 2 else 'rev'
        row.update(flip_fwd=cand['fwd']['corr'], flip_rev=cand['rev']['corr'],
                   flip_x=cand[pick_x]['corr'], flip_ev=cand[pick_e]['corr'],
                   pick_x=pick_x, pick_e=pick_e, true_pick=true_pick,
                   dxerr=cand['rev']['xerr'] - cand['fwd']['xerr'],
                   dev=cand['fwd']['ev'] - cand['rev']['ev'])

        # ---- [X] exact-atom oracle (off-grid, true energies) --------------
        kx, ky, A = S['m_kx'], S['m_ky'], S['m_A']
        om = omega_of(kx, ky, DEPTH)
        en = A ** 2 / 2
        for tag, use_bw in [('xa_free', False), ('xa_full', True)]:
            kk, kk2, oo, ee = kx, ky, om, en
            if use_bw and S['bw'] is not None:
                b = S['bw']
                kk = np.concatenate([kx, b['kx']])
                kk2 = np.concatenate([ky, b['ky']])
                oo = np.concatenate([om, b['om']])
                ee = np.concatenate([en, (S['bw_sc'] * b['A']) ** 2 / 2])
            g = ee / ee.sum()
            gam = T(np.concatenate([g, g]) * float(np.var(S['z'])))
            Phi = design_t(T(kk), T(kk2), T(oo), T(S['P']))
            Phi_q = design_t(T(kk), T(kk2), T(oo), T(S['Pq']))
            K = (Phi * gam[None]) @ Phi.T + (SMULT * S['sigma0']) ** 2 \
                * torch.eye(len(S['z']), device=DEVICE)
            wv = Phi.T @ torch.linalg.solve(K, T(S['z']))
            row[tag] = corr(((Phi_q * gam[None]) @ wv).cpu().numpy(),
                            S['gt_u'])

        # ---- [W] bound-wave injection magnitude ---------------------------
        row['bw_sc'] = S['bw_sc']
        rows.append(row)
        print(f"sea{si:2d} cur {row['cur']:+.3f} | dp1 {row['dp1']:+.3f} "
              f"dp2 {row['dp2']:+.3f} dp4 {row['dp4']:+.3f} dp2w "
              f"{row['dp2w']:+.3f} | flipX {row['flip_x']:+.3f}"
              f"({pick_x}/{true_pick}) flipEV {row['flip_ev']:+.3f}"
              f"({pick_e}) | xa {row['xa_free']:+.3f}/{row['xa_full']:+.3f} "
              f"(r4 {row['real_free']:+.3f}) | bwsc {row['bw_sc']:5.1f}",
              flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur, eb = a('cur'), a('ens_blind')

    def rep(k, base, bl):
        d = a(k) - base
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"  {k:8s}: mean {a(k).mean():+.4f}  vs {bl} {d.mean():+.4f} "
              f"(t {t:+.2f}, win {(d > 0).sum()}/{len(d)})")

    print("\n=== [S] ladder stats ===")
    d2 = eb - cur
    t2 = d2.mean() / (d2.std(ddof=1) / np.sqrt(len(d2)))
    print(f"  rung2-cur: {d2.mean():+.4f} (sd {d2.std(ddof=1):.3f}, t {t2:+.2f},"
          f" 95%CI +-{1.96 * d2.std(ddof=1) / np.sqrt(len(d2)):.4f})")
    d32 = a('ens_oracle') - eb
    keep = np.array([r['si'] != 6 for r in rows])
    print(f"  rung3-rung2: {d32.mean():+.4f}; w/o sea6 {d32[keep].mean():+.4f};"
          f" w/o sea6&1 "
          f"{d32[np.array([r['si'] not in (1, 6) for r in rows])].mean():+.4f}")
    print("\n=== [B] rung2.5 directional-profile (vs cur) ===")
    for k in ('dp1', 'dp2', 'dp4', 'dp2w'):
        rep(k, cur, 'cur')
    print("\n=== [F] blind flip selection (vs ens_blind arm) ===")
    for k in ('flip_x', 'flip_ev'):
        rep(k, eb, 'ens_blind')
    nx = sum(r['pick_x'] == r['true_pick'] for r in rows)
    ne = sum(r['pick_e'] == r['true_pick'] for r in rows)
    print(f"  correct picks: xerr {nx}/{len(rows)}, evidence {ne}/{len(rows)}")
    s6 = [r for r in rows if r['si'] == 6][0]
    print(f"  sea6: fwd {s6['flip_fwd']:+.3f} rev {s6['flip_rev']:+.3f} "
          f"pick_x {s6['pick_x']} pick_e {s6['pick_e']} true {s6['true_pick']}")
    print("\n=== [X] exact-atom oracle vs grid-deposit rung4 ===")
    rep('xa_free', a('real_free'), 'r4')
    print("\n=== [W] bw_sc (injection amplitude multiplier vs physical) ===")
    bws = a('bw_sc')
    print(f"  bw_sc: min {bws.min():.1f} med {np.median(bws):.1f} "
          f"max {bws.max():.1f} (0 = no bw sea)")
    json.dump(rows, open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
