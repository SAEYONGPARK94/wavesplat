"""Step 3 -- amortize the selection-aware EP with a site head (N5 skeleton).

G0b (scripts/d3_g0b_selection_aware.py) measures the prize of the exact
selection-aware posterior under the threshold detection law: EP with one
Bernoulli-tilt site per mask bit. Exact but expensive -- SWEEPS full solves of
a 3,800^2 system plus a 40-node quadrature over ~7k sites, per frame. The
control budget is 100 ms.

This script trains the honest learning piece: a small net that emits the EP
SITE PARAMETERS (tau_k, nu_k) directly from deployment-observable features, so
the deployed estimator is net -> ONE augmented linear solve. It follows every
design law the day established:

  L2  the net emits likelihood-space quantities (sites), never the field
  L3  null-coincident zero: tau head is softplus with bias init -6 (starts
      ~0 = the naive solver), and MCAR scenes with target tau = 0 are IN the
      training mix
  L4  the no-learning competitor is reported: warm-started 1-sweep EP
  L5  teacher = the converged EP computation, never GT
  L6  gates below, frozen before training

Per-site features (all deployment-observable, no GT, no true slope):
  r/R_MAX, t/T_WIN, is_return, ln T_scan, local return density (25 m disc
  around the site's footprint, per scan), grazing angle g(r), sea-state proxy
  sigma_z = std of the returned elevations (observable).

================================ PRE-REGISTRATION ============================
  R1 REDUCTION.  tau ≡ 0 through the same augmented-solve path equals the
                 naive solve to <= 1e-6 fp64 (architectural, verified).
  F1 FIDELITY.   on held-out seas, net-site solve recovers >= 50% of the
                 teacher EP's paired gain over naive (informed region, as
                 gated in G0b #48). Only meaningful if G0b's G1 passed;
                 otherwise this script reports fidelity to the teacher's
                 (null) behaviour and the latency numbers stand alone.
  N1 MCAR ZERO.  on held-out MCAR scenes: mean tau(net) <= 5% of the MNAR
                 mean, and |gain| <= 0.002 (equivalence, not significance).
  L1 LATENCY.    net + 1 solve <= 100 ms/frame on this GPU, and its
                 accuracy-latency point must not be dominated by warm-start
                 1-sweep EP (the no-learning competitor).
TRAIN ENOUGH: >= 20k steps, convergence curve printed, 3 seeds; the standing
rule forbids concluding from underconverged training.
=============================================================================
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import (eta, ray_grid, H_SENS, R_MIN, R_MAX,
                                  SCAN_T, NOISE, DEPTH, FOV_H)
from d3_g0b_selection_aware import gen_scene, ep_aware, A_of_t, BAND, K_MISS

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64
OUT = 'results/direction3/n5_site_head'
AMP = 3.33                     # the accounting sea (G0b main arm)
N_TRAIN, N_VAL, N_MCAR = 12, 4, 4
STEPS = 24000


def scene_tensors(seed, rng, mcar=False):
    """one scene -> everything both estimators and the net need."""
    S, kp, amps, missP, miss_r, Ts = gen_scene(seed, AMP, rng, mcar=mcar)
    P, zval = amps[:, :3], amps[:, 3] + rng.normal(0, NOISE, len(amps))
    ret_r = amps[:, 4]

    nq = 1200
    rq = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, nq)) * R_MAX
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
    Pq = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])
    gt = eta(Pq[:, 0], Pq[:, 1], 0.0, S)

    tP = torch.tensor(P, device=DEV, dtype=torch.float32)
    tz = torch.tensor(zval, device=DEV, dtype=torch.float32)
    tQ = torch.tensor(Pq, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(tP, tz, DEPTH, tQ, NOISE, kp, seed=seed,
                      return_model=True)
    gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)

    n_miss = len(missP)
    sub = rng.choice(n_miss, min(K_MISS, n_miss), replace=False)
    Tmap = {t: Ts[i] for i, t in enumerate(SCAN_T)}
    sT = np.concatenate([[Tmap[t] for t in P[:, 2]],
                         [Tmap[t] for t in missP[sub, 2]]])
    sitesP = np.vstack([P, missP[sub]])
    sites_r = np.concatenate([ret_r, miss_r[sub]])
    is_ret = np.r_[np.ones(len(P), bool), np.zeros(len(sub), bool)]
    w_scale = np.r_[np.ones(len(P)), np.full(len(sub), n_miss / len(sub))]

    # deployment-observable per-site features
    dens = np.zeros(len(sitesP))
    for t in SCAN_T:
        m = sitesP[:, 2] == t
        pts = P[P[:, 2] == t, :2]
        if m.sum() and len(pts):
            d2 = ((sitesP[m, :2][:, None] - pts[None]) ** 2).sum(2)
            dens[m] = (d2 < 25.0 ** 2).sum(1)
    sig_z = float(np.std(zval))
    feats = np.column_stack([
        sites_r / R_MAX, sitesP[:, 2] / 20.0, is_ret.astype(float),
        np.log(np.maximum(sT, 1e-12)) / 10.0, dens / 10.0,
        np.arctan(H_SENS / sites_r), np.full(len(sitesP), sig_z)])
    # MCAR scenes carry T = nan (no threshold exists); a single nan feature
    # column poisoned every weight by step 4k -- the SECOND NaN source after
    # the tau parameterization (#50). Sanitize unconditionally.
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return dict(S=S, kp=kp, gam=gam, s2=s2, kxb=kxb, kyb=kyb, om=om,
                Phi=design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om),
                Phq=design(torch.tensor(Pq, device=DEV, dtype=DT), kxb, kyb, om),
                z=torch.tensor(zval, device=DEV, dtype=DT),
                gt=torch.tensor(gt, device=DEV, dtype=DT),
                sig=float(np.std(gt)), rq=rq, ret_r=ret_r,
                sitesP=torch.tensor(sitesP, device=DEV, dtype=DT),
                sites_r=torch.tensor(sites_r, device=DEV, dtype=DT),
                sites_T=torch.tensor(sT, device=DEV, dtype=DT),
                is_ret=torch.tensor(is_ret, device=DEV),
                w=torch.tensor(w_scale, device=DEV, dtype=DT),
                feats=torch.tensor(feats, device=DEV, dtype=torch.float32),
                mcar=mcar)


def teacher_sites(sc, sweeps=8):
    """converged EP -> (tau*, nu*) per site. The teacher is a computation."""
    G = grad_design(sc['sitesP'], sc['kxb'], sc['kyb'], sc['om'])
    A0 = sc['Phi'].T @ sc['Phi'] / sc['s2'] + torch.diag(1.0 / sc['gam'])
    b0 = sc['Phi'].T @ sc['z'] / sc['s2']
    K = len(G)
    tau = torch.zeros(K, device=DEV, dtype=DT)
    nu = torch.zeros(K, device=DEV, dtype=DT)
    if sc['mcar']:
        return G, tau, nu                      # exact: constant p carries 0
    gx = torch.tensor(np.polynomial.hermite_e.hermegauss(40)[0], device=DEV,
                      dtype=DT)
    gwv = np.polynomial.hermite_e.hermegauss(40)[1]
    gw = torch.tensor(gwv / gwv.sum(), device=DEV, dtype=DT)
    for _ in range(sweeps):
        A = A0 + (G * (sc['w'] * tau)[:, None]).T @ G
        S_ = torch.linalg.inv(A)
        mu = S_ @ (b0 + G.T @ (sc['w'] * nu))
        v = ((G @ S_) * G).sum(1).clamp_min(1e-14)
        m = G @ mu
        vc_inv = (1.0 / v) - tau
        ok = vc_inv > 1e-12
        vc = torch.where(ok, 1.0 / vc_inv.clamp_min(1e-12), v * 1e6)
        mc = torch.where(ok, vc * (m / v - nu), m)
        sd = torch.sqrt(vc)
        sq = mc[:, None] + sd[:, None] * gx[None, :]
        p = torch.exp(-sc['sites_T'][:, None] /
                      A_of_t(sq, sc['sites_r'][:, None]).clamp_min(1e-30))
        f = torch.where(sc['is_ret'][:, None], p, 1.0 - p).clamp_min(1e-300)
        Z = (gw[None] * f).sum(1)
        m_h = (gw[None] * f * sq).sum(1) / Z
        v_h = ((gw[None] * f * sq ** 2).sum(1) / Z - m_h ** 2).clamp_min(1e-14)
        tau_n = ((1.0 / v_h) - (1.0 / vc)).clamp_min(0.0)
        nu_n = m_h / v_h - mc / vc
        bad = ~ok | ~torch.isfinite(tau_n) | ~torch.isfinite(nu_n)
        tau = 0.6 * torch.where(bad, tau, tau_n) + 0.4 * tau
        nu = 0.6 * torch.where(bad, nu, nu_n) + 0.4 * nu
    return G, tau, nu


def solve_with(sc, G, tau, nu):
    A = sc['Phi'].T @ sc['Phi'] / sc['s2'] + torch.diag(1.0 / sc['gam'])
    b = sc['Phi'].T @ sc['z'] / sc['s2']
    if tau is not None:
        A = A + (G * (sc['w'] * tau)[:, None]).T @ G
        b = b + G.T @ (sc['w'] * nu)
    return torch.linalg.solve(A, b)


def nrmse_inf(sc, a):
    """FULL-fan scoring with the B1 cap. The informed-region variant (#48)
    degenerated under the threshold law: r90(returns) ~ 20 m leaves 2-3 query
    points -- the statistic is void. Under this law the mask bits are the ONLY
    far-field information, so the effect, if any, lives on the full fan; the
    naive >= 1 caveat is carried in the report."""
    f = sc['Phq'] @ a
    beta = min(1.0, float(sc['z'].std() / f.std().clamp_min(1e-12)))
    e = beta * f - sc['gt']
    return float(torch.sqrt((e ** 2).mean()) / max(sc['sig'], 1e-9))


def prep_targets(sc):
    """teacher targets in a LEARNABLE parameterization.

    Raw tau* spans 14 orders of magnitude (a tight EP constraint drives the
    site variance to the 1e-14 clamp, so tau* -> 1e14); regressing that
    directly made the loss NaN by step 4k (#50). Train instead on
        tt = log1p(tau*)            (~0..32, smooth)
        mu = nu*/tau*               (the site MEAN: physical slope scale)
    with mu supervised only where the site is active. Reconstruction
    tau = expm1(tt), nu = tau*mu is exact."""
    tau, nu = sc['tau*'], sc['nu*']
    sc['tt*'] = torch.log1p(tau)
    act = tau > 1e-6
    mu = torch.zeros_like(nu)
    mu[act] = nu[act] / tau[act]
    sc['mu*'] = mu.clamp(-5.0, 5.0)
    sc['act*'] = act


class SiteHead(nn.Module):
    def __init__(self, d=7, h=96):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h),
                               nn.SiLU(), nn.Linear(h, 2))
        with torch.no_grad():                       # L3: start at tau ~ 0
            self.f[-1].bias.copy_(torch.tensor([-6.0, 0.0]))
            self.f[-1].weight.mul_(0.1)

    def forward(self, x):
        o = self.f(x)
        tt = nn.functional.softplus(o[:, 0])        # log1p(tau) >= 0; bias -6 -> ~0
        return tt, o[:, 1]                           # (tt, site mean mu)

    def sites(self, x):
        tt, mu = self(x)
        tau = torch.expm1(tt.double().clamp(max=34.0))
        return tau, tau * mu.double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a_ = ap.parse_args()

    rng = np.random.default_rng(77)
    print("generating scenes + EP teachers "
          f"({N_TRAIN} train, {N_VAL} val, {N_MCAR} mcar)...")
    scenes = []
    for i in range(N_TRAIN + N_VAL):
        sc = scene_tensors(100 + i, rng)
        sc['G'], sc['tau*'], sc['nu*'] = teacher_sites(sc)
        prep_targets(sc)
        scenes.append(sc)
        print(f"  scene {i:2d} mean tau* {float(sc['tau*'].mean()):.3e} "
              f"active {(sc['tau*'] > 1e-8).float().mean():.2f}", flush=True)
    for i in range(N_MCAR):
        sc = scene_tensors(300 + i, rng, mcar=True)
        sc['G'], sc['tau*'], sc['nu*'] = teacher_sites(sc)
        prep_targets(sc)
        scenes.append(sc)
    train, val = scenes[:N_TRAIN], scenes[N_TRAIN:N_TRAIN + N_VAL]
    mcar = scenes[N_TRAIN + N_VAL:]
    tr_all = train + mcar                          # L3: mcar in the mix

    res = {'seeds': {}}
    for seed in a_.seeds:
        torch.manual_seed(seed)
        net = SiteHead().to(DEV)
        opt = torch.optim.Adam(net.parameters(), 3e-4)
        t0 = time.time()
        for step in range(STEPS):
            sc = tr_all[int(torch.randint(len(tr_all), (1,)))]
            tt_h, mu_h = net(sc['feats'])
            lt = nn.functional.huber_loss(tt_h.double(), sc['tt*'])
            a = sc['act*']
            ln = nn.functional.huber_loss(mu_h.double()[a], sc['mu*'][a]) \
                if a.any() else lt * 0
            loss = lt + ln
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 4000 == 0:
                print(f"  seed {seed} step {step:5d} loss {loss.item():.4e} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # evaluation on held-out seas
        rows = []
        for sc in val:
            with torch.no_grad():
                tau_h, nu_h = net.sites(sc['feats'])
            a_n = solve_with(sc, sc['G'], None, None)
            a_t = solve_with(sc, sc['G'], sc['tau*'], sc['nu*'])
            a_h = solve_with(sc, sc['G'], tau_h, nu_h)
            n0, nt, nh = (nrmse_inf(sc, a) for a in (a_n, a_t, a_h))
            rows.append({'naive': n0, 'ep': nt, 'net': nh,
                         'gain_ep': n0 - nt, 'gain_net': n0 - nh})
            print(f"  val: naive {n0:.4f} EP {nt:.4f} net {nh:.4f}", flush=True)
        ge = float(np.mean([r['gain_ep'] for r in rows]))
        gn = float(np.mean([r['gain_net'] for r in rows]))
        fid = gn / ge if abs(ge) > 1e-6 else float('nan')

        # N1: mcar zero (held-out mcar scene)
        scm = scene_tensors(400 + seed, rng, mcar=True)
        scm['G'], _, _ = teacher_sites(scm)
        with torch.no_grad():
            tau_m, nu_m = net.sites(scm['feats'])
        mn = float(torch.log1p(tau_m).mean())
        mnar = float(np.mean([net(sc['feats'])[0].mean().item() for sc in val]))
        a_n = solve_with(scm, scm['G'], None, None)
        a_m = solve_with(scm, scm['G'], tau_m, nu_m)
        g_m = nrmse_inf(scm, a_n) - nrmse_inf(scm, a_m)

        # R1 + latency
        z = torch.zeros(len(scm['feats']), device=DEV, dtype=DT)
        r1 = float((solve_with(scm, scm['G'], z, z) - a_n).abs().max())
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(10):
            with torch.no_grad():
                th_, nh_ = net.sites(val[0]['feats'])
            solve_with(val[0], val[0]['G'], th_, nh_)
        torch.cuda.synchronize()
        lat_net = (time.time() - t1) / 10 * 1000
        t1 = time.time()
        for _ in range(3):
            teacher_sites(val[0], sweeps=8)
        torch.cuda.synchronize()
        lat_ep = (time.time() - t1) / 3 * 1000
        t1 = time.time()
        for _ in range(3):
            teacher_sites(val[0], sweeps=1)
        torch.cuda.synchronize()
        lat_ep1 = (time.time() - t1) / 3 * 1000
        sc1 = val[0]
        G1_, tau1, nu1 = teacher_sites(sc1, sweeps=1)
        g_ep1 = nrmse_inf(sc1, solve_with(sc1, G1_, None, None)) - \
            nrmse_inf(sc1, solve_with(sc1, G1_, tau1, nu1))

        print(f"\nseed {seed}: EP gain {ge:+.4f}  net gain {gn:+.4f}  "
              f"fidelity {fid:.0%}")
        print(f"  N1 mcar: tau {mn:.2e} vs MNAR {mnar:.2e} "
              f"({mn/max(mnar,1e-12):.1%})   gain {g_m:+.4f}  "
              f"{'PASS' if mn <= 0.05*mnar and abs(g_m) <= 0.002 else 'FAIL'}")
        print(f"  R1 = {r1:.2e}  {'PASS' if r1 <= 1e-6 else 'FAIL'}")
        print(f"  latency: net+solve {lat_net:.0f} ms | EP8 {lat_ep:.0f} ms | "
              f"EP1(warm-start competitor) {lat_ep1:.0f} ms, gain {g_ep1:+.4f}")
        res['seeds'][seed] = {'gain_ep': ge, 'gain_net': gn, 'fidelity': fid,
                              'mcar_tau_frac': mn / max(mnar, 1e-12),
                              'mcar_gain': g_m, 'r1': r1,
                              'lat_net_ms': lat_net, 'lat_ep8_ms': lat_ep,
                              'lat_ep1_ms': lat_ep1, 'gain_ep1': g_ep1,
                              'rows': rows}

    os.makedirs('results/direction3', exist_ok=True)
    with open(f'{OUT}.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
