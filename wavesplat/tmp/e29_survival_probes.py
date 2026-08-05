"""e29_survival_probes.py — inference-only probes attacking the gate-0 ladder
closure argument ("ensemble content saturated at -0.012, remaining +0.141 is
realization-specific hence unlearnable").

The closure claim conflates two different bounds:
  rung 2 bounds what an UNCONDITIONAL ensemble prior buys;
  the incumbent is a CONDITIONAL (amortized) inference operator.
  The rung2->rung4 gap is the per-scene inference problem itself. Whether any
  estimator (net or classical) can extract part of it from the 400 obs is an
  EMPIRICAL question these probes answer without training anything.

PROBES (all 16 arena seas, same solver path as the ladder, no GT selection):
  A em       : incumbent full path with use_em=True (per-scene SBL-EM gamma
               refinement, val-gated, code already shipped but switched off).
               If EM > cur, realization gamma is partially data-recoverable.
  B angeb    : universal radial x 24-bin angular histogram, weights fitted by
               per-scene marginal-likelihood (type-II ML) ascent from the
               rung-2 init (N(mu_hat, 30deg)). ZERO oracle info. Measures the
               identifiable fraction of the direction-draw axis in a
               well-conditioned 24-dim parameterization.
  C resladder: true draws smeared by s in {5,10,20,45} deg -> where does the
               +0.141 live in angular resolution?
  D flipstats: beamform direction error vs truth on 48 seas + per-sea U
               (frequency and covariate of sea6-type 180deg flips).
  E mix      : val-selected convex mixture gamma_net <-> gamma_ens_blind
               (deployable, no oracle) through the fixed smult=3 free solver.

Usage: python3 tmp/e29_survival_probes.py [--probes em,angeb,resladder,flipstats,mix]
Output: results/direction3/e29_survival_probes.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, REGIMES, maps6,
                          load_incumbent, NGRID, KREL)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid, SMULT
from d3_prior_net_eval import net_gamma_mmse
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs
from wavesim.wave_generator import jonswap_components

OUT = 'results/direction3/e29_survival_probes.json'
LADDER = json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']
CEIL = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']


def sea_params(si, regime='spread'):
    """Replicate build_sea_c's rng draw order to recover mu_true, U, nfrac."""
    R = REGIMES[regime]
    rng = np.random.default_rng(7000 + 31 * si + hashabs(regime))
    Hs = rng.uniform(*R['hs'])
    mu_true = np.deg2rad(rng.uniform(0, 360))
    U = rng.uniform(*R['U'])          # in units of cp
    nf = rng.uniform(*R['nf'])
    return dict(Hs=Hs, mu_true=mu_true, Ucp=U, nfrac=nf)


# ========================================================================
# A: per-scene SBL-EM. NOTE: the shipped use_em path is DEAD CODE whenever
# smult_head or smult_fixed is set (vr0 = NaN -> `vr < nan-1e-5` False ->
# immediate break; verified: d == 0.000 on 3 seas). So we implement the EM
# loop explicitly, init at the incumbent's own gamma (free arm, smult=3
# solver = ladder-comparable), and record the full per-iteration trajectory
# of held-out corr AND obs-val RMSE (deployable gate).
# ========================================================================
def em_traj(S, g_free_grid, iters=8):
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    kxb, kyb, om = bank_atoms(kp)
    t = lambda a: torch.tensor(a, dtype=torch.float64, device=DEVICE)
    rng = np.random.default_rng(0)
    n = len(z)
    vi = rng.choice(n, max(8, n // 4), replace=False)
    vm = np.zeros(n, bool); vm[vi] = True
    Phi_tr = design_t(t(kxb), t(kyb), t(om), t(P[~vm]))
    Phi_v = design_t(t(kxb), t(kyb), t(om), t(P[vm]))
    Phi_all = design_t(t(kxb), t(kyb), t(om), t(P))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(Pq))
    ztr, zv, zt = t(z[~vm]), t(z[vm]), t(z)
    m = int((~vm).sum())
    s2 = (SMULT * s0) ** 2
    NB2 = Phi_tr.shape[1] // 2
    g = np.concatenate([g_free_grid[_MASK], np.zeros(_MASK.sum())])
    g = g / g.sum()
    gam = t(np.concatenate([g, g]) * float(np.var(z)))

    def heldout(gam):
        K = (Phi_all * gam[None]) @ Phi_all.T \
            + s2 * torch.eye(n, dtype=torch.float64, device=DEVICE)
        w = Phi_all.T @ torch.linalg.solve(K, zt)
        return corr(((Phi_q * gam[None]) @ w).cpu().numpy(), gt)

    def valrmse(gam):
        K = (Phi_tr * gam[None]) @ Phi_tr.T \
            + s2 * torch.eye(m, dtype=torch.float64, device=DEVICE)
        w = Phi_tr.T @ torch.linalg.solve(K, ztr)
        fv = ((Phi_v * gam[None]) @ w).cpu().numpy()
        return float(np.sqrt(np.mean((fv - z[vm]) ** 2)))

    traj = [dict(it=0, corr=heldout(gam), vr=valrmse(gam))]
    for it in range(1, iters + 1):
        K = (Phi_tr * gam[None]) @ Phi_tr.T \
            + s2 * torch.eye(m, dtype=torch.float64, device=DEVICE)
        Ki_Phi = torch.linalg.solve(K, Phi_tr)
        mu = gam * (Phi_tr.T @ torch.linalg.solve(K, ztr))
        sig = gam - gam ** 2 * (Phi_tr * Ki_Phi).sum(0)
        gb = 0.5 * (mu[:NB2] ** 2 + mu[NB2:] ** 2
                    + sig[:NB2].clamp_min(0) + sig[NB2:].clamp_min(0))
        gam = torch.cat([gb, gb])
        traj.append(dict(it=it, corr=heldout(gam), vr=valrmse(gam)))
    return traj


def probe_em(n_seas=16, iters=8):
    kmag, en = universal_radial()
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        lad = LADDER[si]
        tr_n = em_traj(S, net_grid(S), iters)
        # deployable selection: best val-RMSE iterate
        sel = min(tr_n, key=lambda r: r['vr'])
        best = max(tr_n, key=lambda r: r['corr'])
        rows.append(dict(si=si, cur=lad['cur'], net_g3=tr_n[0]['corr'],
                         em_sel=sel['corr'], em_sel_it=sel['it'],
                         em_best=best['corr'], em_best_it=best['it'],
                         traj=tr_n, real_free=lad['real_free']))
        print(f"[em] sea{si:2d} cur {lad['cur']:+.3f} net_g3 "
              f"{tr_n[0]['corr']:+.3f} | " +
              " ".join(f"i{r['it']}:{r['corr']:+.3f}" for r in tr_n[1:]) +
              f" | sel(it{sel['it']}) {sel['corr']:+.3f} "
              f"best(it{best['it']}) {best['corr']:+.3f}", flush=True)
    for k in ('em_sel', 'em_best'):
        d0 = np.array([r[k] - r['net_g3'] for r in rows])
        dc = np.array([r[k] - r['cur'] for r in rows])
        print(f"[em] {k:8s} vs net_g3: {d0.mean():+.4f} "
              f"(pos {(d0>0).sum()}/{len(d0)}) | vs cur: {dc.mean():+.4f} "
              f"(pos {(dc>0).sum()}/{len(dc)})")
    return rows


# ========================================================================
# B: angular empirical-Bayes (24-bin histogram, marginal-likelihood ascent)
# ========================================================================
NBIN = 24


def bin_grids(kmag, en, kp):
    """G_b (NBIN, NGRID, NGRID): deposit of the universal radial energy with
    ALL of it placed in angular bin b (5 quadrature nodes per bin)."""
    G = []
    for b in range(NBIN):
        th0 = 2 * np.pi * b / NBIN
        th = th0 + 2 * np.pi / NBIN * (np.arange(5) + 0.5) / 5
        kxs = np.concatenate([k * np.cos(th) for k in kmag])
        kys = np.concatenate([k * np.sin(th) for k in kmag])
        ens = np.concatenate([np.full(5, e / 5) for e in en])
        G.append(deposit_grid(kxs, kys, ens, kp))
    return np.stack(G)


def init_logits(mu_rad, spread_deg=30.0):
    th_b = 2 * np.pi * (np.arange(NBIN) + 0.5) / NBIN
    dd = np.angle(np.exp(1j * (th_b - mu_rad)))
    w = np.exp(-0.5 * (dd / np.deg2rad(spread_deg)) ** 2) + 1e-4
    return np.log(w / w.sum())


def angeb_sea(S, mu_hat, kmag, en, steps=250, lr=0.08, ridge=0.0):
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    G = bin_grids(kmag, en, kp)                     # (24,51,51)
    Gm = torch.tensor(G[:, _MASK], dtype=torch.float32, device=DEVICE)
    kxb, kyb, om = bank_atoms(kp)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(P))     # (n, 2*2NB)
    zt = t(z)
    n = len(z)
    NB = Gm.shape[1]
    s2 = (SMULT * s0) ** 2
    eye = torch.eye(n, device=DEVICE)
    vz = float(np.var(z))
    lg0 = init_logits(mu_hat)
    lg = torch.tensor(lg0, dtype=torch.float32, device=DEVICE,
                      requires_grad=True)
    opt = torch.optim.Adam([lg], lr=lr)
    lg_best, nll0, nll_best = lg0.copy(), None, None
    for it in range(steps + 1):
        w = torch.softmax(lg, 0)
        gfree = (w[:, None] * Gm).sum(0)
        g = torch.cat([gfree, torch.zeros(NB, device=DEVICE)])
        g = g / g.sum().clamp_min(1e-30)
        gam = torch.cat([g, g]) * vz
        K = (Phi * gam[None]) @ Phi.T + s2 * eye
        L = torch.linalg.cholesky(K)
        a = torch.cholesky_solve(zt[:, None], L)[:, 0]
        nll = 0.5 * (zt @ a) + torch.log(torch.diag(L)).sum()
        if ridge > 0:
            nll = nll + ridge * ((lg - t(lg0)) ** 2).sum()
        if it == 0:
            nll0 = float(nll)
        if nll_best is None or float(nll) < nll_best:
            nll_best, lg_best = float(nll), lg.detach().cpu().numpy().copy()
        if it == steps:
            break
        opt.zero_grad()
        nll.backward()
        opt.step()
    w = np.exp(lg_best - lg_best.max())
    w = w / w.sum()
    gfree_np = (w[:, None, None] * G).sum(0)
    c = solve_with_grid(S, gfree_np)
    return c, nll0 - nll_best, w


def probe_angeb(n_seas=16, ridge=0.0):
    kmag, en = universal_radial()
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        mu_hat = np.arctan2(*reversed(beamform_top(
            S['P'].astype(np.float32), S['z'].astype(np.float32),
            DEPTH, klim=3.5 * S['kp'], n=120)))
        lad = LADDER[si]
        c, dnll, w = angeb_sea(S, mu_hat, kmag, en, ridge=ridge)
        rows.append(dict(si=si, cur=lad['cur'], ens_blind=lad['ens_blind'],
                         real_free=lad['real_free'], angeb=c, dnll=dnll,
                         w=w.tolist()))
        print(f"[angeb] sea{si:2d} cur {lad['cur']:+.3f} "
              f"blind {lad['ens_blind']:+.3f} angeb {c:+.3f} "
              f"(vs blind {c-lad['ens_blind']:+.3f}, vs cur {c-lad['cur']:+.3f}) "
              f"real {lad['real_free']:+.3f} dNLL {dnll:8.1f}", flush=True)
    db = np.array([r['angeb'] - r['ens_blind'] for r in rows])
    dc = np.array([r['angeb'] - r['cur'] for r in rows])
    fr = np.array([(r['angeb'] - r['ens_blind'])
                   / max(r['real_free'] - r['ens_blind'], 1e-9) for r in rows])
    print(f"[angeb] vs blind {db.mean():+.4f} (pos {(db>0).sum()}/{len(db)}) "
          f"| vs cur {dc.mean():+.4f} (pos {(dc>0).sum()}/{len(dc)}) "
          f"| captured fraction of realization gap: mean {fr.mean():.2f} "
          f"median {np.median(fr):.2f}")
    return rows


# ========================================================================
# C: resolution ladder (smear true draws by s deg)
# ========================================================================
def smear_grid(S, s_deg, nq=15):
    s = np.deg2rad(s_deg)
    off = np.linspace(-3 * s, 3 * s, nq)
    wq = np.exp(-0.5 * (off / s) ** 2)
    wq = wq / wq.sum()
    th_n = np.arctan2(S['m_ky'], S['m_kx'])
    k_n = np.hypot(S['m_kx'], S['m_ky'])
    e_n = S['m_A'] ** 2 / 2
    th = (th_n[:, None] + off[None, :]).ravel()
    kk = np.repeat(k_n, nq)
    ee = (e_n[:, None] * wq[None, :]).ravel()
    return deposit_grid(kk * np.cos(th), kk * np.sin(th), ee, S['kp'])


def probe_resladder(n_seas=16, smears=(5.0, 10.0, 20.0, 45.0)):
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        lad = LADDER[si]
        r = dict(si=si, cur=lad['cur'], ens_oracle=lad['ens_oracle'],
                 real_free=lad['real_free'])
        for s in smears:
            r[f's{s:g}'] = solve_with_grid(S, smear_grid(S, s))
        rows.append(r)
        print(f"[res] sea{si:2d} real {lad['real_free']:+.3f} | "
              + " ".join(f"s{s:g} {r[f's{s:g}']:+.3f}" for s in smears)
              + f" | ens_o {lad['ens_oracle']:+.3f} cur {lad['cur']:+.3f}",
              flush=True)
    print("[res] means: real_free "
          f"{np.mean([r['real_free'] for r in rows]):+.4f} | "
          + " ".join(f"s{s:g} {np.mean([r[f's{s:g}'] for r in rows]):+.4f}"
                     for s in smears)
          + f" | ens_oracle {np.mean([r['ens_oracle'] for r in rows]):+.4f}"
          + f" | cur {np.mean([r['cur'] for r in rows]):+.4f}")
    return rows


# ========================================================================
# D: flip stats (48 seas, no solves)
# ========================================================================
def probe_flipstats(n_seas=48):
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        p = sea_params(si)
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        mu_hat = np.arctan2(by, bx)
        derr = np.rad2deg(abs(np.angle(np.exp(1j * (mu_hat - p['mu_true'])))))
        rows.append(dict(si=si, dir_err=float(derr), Ucp=p['Ucp'],
                         nfrac=p['nfrac'], Hs=p['Hs']))
        if derr > 60:
            print(f"[flip] sea{si:2d} dir_err {derr:6.1f} U {p['Ucp']:.2f}cp "
                  f"nf {p['nfrac']:.3f}", flush=True)
    de = np.array([r['dir_err'] for r in rows])
    U = np.array([r['Ucp'] for r in rows])
    print(f"[flip] n={n_seas} err>90deg: {(de>90).sum()} "
          f"({(de>90).mean():.1%}); err>30deg: {(de>30).sum()}; "
          f"median {np.median(de):.1f}deg")
    print(f"[flip] U of flipped seas: {U[de>90]} | median U all: "
          f"{np.median(U):.2f}")
    return rows


# ========================================================================
# E: val-selected mixture gamma_net <-> gamma_ens_blind (deployable)
# ========================================================================
def net_grid(S):
    net, _ = load_incumbent()
    X = maps6(S['P'], S['z'], S['kp'])
    with torch.no_grad():
        Ylog = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
    return np.maximum(10.0 ** Ylog, 1e-8)


def solve_val_and_test(S, g_free_grid):
    """Same as solve_with_grid but also returns val-RMSE on a 25% obs split."""
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    g = np.concatenate([g_free_grid[_MASK],
                        np.zeros(_MASK.sum())])
    g = g / g.sum()
    kxb, kyb, om = bank_atoms(kp)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(0)
    n = len(z)
    vi = rng.choice(n, max(8, n // 4), replace=False)
    vm = np.zeros(n, bool); vm[vi] = True
    gam = t(np.concatenate([g, g]) * float(np.var(z)))
    Phi_tr = design_t(t(kxb), t(kyb), t(om), t(P[~vm]))
    Phi_v = design_t(t(kxb), t(kyb), t(om), t(P[vm]))
    K = (Phi_tr * gam[None]) @ Phi_tr.T \
        + (SMULT * s0) ** 2 * torch.eye(int((~vm).sum()), device=DEVICE)
    w = Phi_tr.T @ torch.linalg.solve(K, t(z[~vm]))
    fv = ((Phi_v * gam[None]) @ w).cpu().numpy()
    vr = float(np.sqrt(np.mean((fv - z[vm]) ** 2)))
    return vr, solve_with_grid(S, g_free_grid)


def probe_mix(n_seas=16, ws=(0.0, 0.25, 0.5, 0.75, 1.0)):
    kmag, en = universal_radial()
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        lad = LADDER[si]
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        ge = ensemble_grid(kmag, en, np.arctan2(by, bx), S['kp'])
        gn = net_grid(S)
        ge = ge / ge.sum()
        gn = gn / gn.sum()
        best, tests = None, {}
        for wmix in ws:
            g = (1 - wmix) * gn + wmix * ge
            vr, c = solve_val_and_test(S, g)
            tests[wmix] = c
            if best is None or vr < best[0]:
                best = (vr, wmix, c)
        oracle_w = max(tests.values())
        rows.append(dict(si=si, cur=lad['cur'], w_sel=best[1], mix=best[2],
                         mix_oracle_w=oracle_w,
                         net_g3=tests[0.0], ens_g3=tests[1.0]))
        print(f"[mix] sea{si:2d} cur {lad['cur']:+.3f} net_g3 "
              f"{tests[0.0]:+.3f} ens_g3 {tests[1.0]:+.3f} | w* {best[1]:.2f} "
              f"mix {best[2]:+.3f} (orc-w {oracle_w:+.3f})", flush=True)
    for k in ('mix', 'mix_oracle_w', 'net_g3'):
        d = np.array([r[k] - r['cur'] for r in rows])
        print(f"[mix] {k:12s} - cur: mean {d.mean():+.4f} "
              f"pos {(d>0).sum()}/{len(d)}")
    return rows


# ========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', default='em,angeb,resladder,flipstats,mix')
    ap.add_argument('--n_seas', type=int, default=16)
    ap.add_argument('--flip_seas', type=int, default=48)
    ap.add_argument('--ridge', type=float, default=0.0)
    a = ap.parse_args()
    todo = a.probes.split(',')
    out = {}
    t0 = time.time()
    if 'em' in todo:
        out['em'] = probe_em(a.n_seas)
    if 'angeb' in todo:
        out['angeb'] = probe_angeb(a.n_seas, ridge=a.ridge)
    if 'resladder' in todo:
        out['resladder'] = probe_resladder(a.n_seas)
    if 'flipstats' in todo:
        out['flipstats'] = probe_flipstats(a.flip_seas)
    if 'mix' in todo:
        out['mix'] = probe_mix(a.n_seas)
    old = {}
    if os.path.exists(OUT):
        old = json.load(open(OUT))
    old.update(out)
    json.dump(old, open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
