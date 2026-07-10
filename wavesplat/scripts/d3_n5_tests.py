"""d3_n5_tests.py -- machine-precision checks on the N5 core (d3_n5_core.py).

No training. Every test prints its numbers and PASS/FAIL; the process exits
nonzero if any test fails. The numbers printed here are copied verbatim into
docs/N5_DERIVED_NETWORK_2026-07-10.md -- if you change the code, re-run and
update the doc.

  R1  reduction        all delta = 0, no EP sites  =>  the architecture's field
                       equals the naive (kernel-form) posterior mean to <= 1e-6
                       on 1,200 query points, fp64. The analogue of E5's
                       "gain = 1 => |diff| = 0".
  ID  MCAR/zero-site   EP sites carrying tau = 0 change the solve by EXACTLY 0.
                       Plus: ep_neuron on constraints satisfied by many sigma
                       self-deactivates (tau ~ 0) -- ignorable missingness.
  EQ  phase equiv.     PhaseEquivariantBlock commutes with per-mode phase
                       rotation to <= 1e-6 for random weights; the physics
                       evolution obeys the group law exactly.
  C3  impossibility    on a 2.5 s window the sub-floor band (k < 0.35 k_p) is
                       uninformed (S1), so the energy a full-scale adversarial
                       delta can inject through the gate is <= sum c_i |delta|
                       sqrt(gam_i), a few % of the band's prior energy.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_n5_core import (ep_neuron, contraction_gate, physics_posterior,
                        GatedDelta, PhaseEquivariantBlock, phase_rotate)
from d3_door1b_field_bias import (sea, eta, ray_grid, transparent_hit,
                                  R_MIN, R_MAX, NOISE, DEPTH, FOV_H, M_PTS)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64
TOL = 1e-6


# ------------------------------------------------------------------ scene
def obs_positions(S, T_window, n_scan, rng):
    """Surface hit positions (x, y, t) over a temporal window of length T."""
    o, d, _ = ray_grid()
    pts = []
    for t in np.linspace(-T_window, 0.0, n_scan):
        ok, p = transparent_hit(o, d, t, S)
        if ok.sum() == 0:
            continue
        pts.append(np.column_stack([p[ok], np.full(ok.sum(), t)]))
    pts = np.vstack(pts)
    sel = rng.choice(len(pts), M_PTS, replace=False)
    P = np.column_stack([pts[sel, 0], pts[sel, 1], pts[sel, 3]])
    z_true = pts[sel, 2]
    return P, z_true


def query_fan(rng, nq=1200):
    rq = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, nq)) * R_MAX
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
    return np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])


def kp_of(S):
    return float(np.hypot(S[0], S[1])[np.argmax(S[3])])


# ==================================================================== R1
def test_r1():
    print("\n[R1] reduction: architecture(delta=0, no sites) == naive posterior "
          "mean")
    rng = np.random.default_rng(3)
    S = sea(1)
    kp = kp_of(S)
    P, z_true = obs_positions(S, 20.0, 12, rng)
    z = z_true + rng.normal(0, NOISE, len(z_true))
    Pq = query_fan(rng)

    # prior (gam) and noise (s2) come from the production physics estimator.
    tP = torch.tensor(P, device=DEV, dtype=torch.float32)
    tz = torch.tensor(z, device=DEV, dtype=torch.float32)
    tQ = torch.tensor(Pq, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(tP, tz, DEPTH, tQ, NOISE, kp, seed=1, return_model=True)
    gam = mdl['gam'].to(DT)
    s2 = float(mdl['s2'])

    # rebuild the bank in numpy-float64 (bit-identical atom selection to mmse_gpu)
    kxb, kyb, om = build_bank(kp, (0.35, 2.5), kp / 10.0, DEPTH, DEV, DT)
    assert len(gam) == 2 * len(kxb), (len(gam), len(kxb))
    Phi = design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om)
    Phq = design(torch.tensor(Pq, device=DEV, dtype=DT), kxb, kyb, om)
    z64 = torch.tensor(z, device=DEV, dtype=DT)

    # reference: the naive posterior mean by the INDEPENDENT kernel (dual) route,
    # a = gam (*) Phi^T K^-1 z,  the exact form d3_gpu_mmse uses.
    n = len(z)
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(n, device=DEV, dtype=DT)
    a_naive = gam * (Phi.T @ torch.linalg.solve(K, z64))
    f_naive = Phq @ a_naive

    # architecture: primal physics posterior + information-gated delta, delta=0.
    c = contraction_gate(Phi, gam, s2)
    a_phys = physics_posterior(Phi, z64, gam, s2)     # no sites
    net = GatedDelta(nn.Linear(4, 2 * len(kxb)).to(DT).to(DEV), scale=0.0)
    feats = torch.zeros(4, device=DEV, dtype=DT)
    with torch.no_grad():
        a_arch = net(a_phys, c, feats)                 # a_phys + c (*) 0
    f_arch = Phq @ a_arch

    d = float((f_arch - f_naive).abs().max())
    d_cross = float((Phq @ a_phys - f_naive).abs().max())   # primal vs dual
    ok = d <= TOL
    print(f"     primal vs dual posterior-mean field  max|diff| = {d_cross:.2e}")
    print(f"     architecture vs naive field (1200 q) max|diff| = {d:.2e}"
          f"   gate {TOL:.0e}")
    print(f"     {'PASS' if ok else 'FAIL'}")
    return ok, {'r1_arch_vs_naive': d, 'r1_primal_vs_dual': d_cross}


# ==================================================================== ID
def test_zero_site():
    print("\n[ID] MCAR / zero-site identity: tau = 0 sites change nothing")
    rng = np.random.default_rng(4)
    S = sea(2)
    kp = kp_of(S)
    P, z_true = obs_positions(S, 20.0, 12, rng)
    z = z_true + rng.normal(0, NOISE, len(z_true))
    Pq = query_fan(rng)

    tP = torch.tensor(P, device=DEV, dtype=torch.float32)
    tz = torch.tensor(z, device=DEV, dtype=torch.float32)
    tQ = torch.tensor(Pq, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(tP, tz, DEPTH, tQ, NOISE, kp, seed=2, return_model=True)
    gam = mdl['gam'].to(DT)
    s2 = float(mdl['s2'])

    kxb, kyb, om = build_bank(kp, (0.35, 2.5), kp / 10.0, DEPTH, DEV, DT)
    Phi = design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om)
    z64 = torch.tensor(z, device=DEV, dtype=DT)

    a_naive = physics_posterior(Phi, z64, gam, s2)     # no sites

    # Build genuine constraint rows: sample points along rays, well ABOVE the
    # surface (they carry a true one-sided inequality phi^T a < h).
    o, d, _ = ray_grid()
    sub = rng.choice(len(o), 400, replace=False)
    s_mid = 0.5 * (-o[sub, 2] / d[sub, 2])              # halfway to flat footprint
    pc = o[sub] + s_mid[:, None] * d[sub]
    r = np.hypot(pc[:, 0], pc[:, 1])
    keep = (r > R_MIN) & (r < R_MAX)
    Pc = np.column_stack([pc[keep, 0], pc[keep, 1], np.zeros(keep.sum())])
    hc = pc[keep, 2]
    Fc = design(torch.tensor(Pc, device=DEV, dtype=DT), kxb, kyb, om)
    K = Fc.shape[0]

    # (a) tau exactly 0 -> augmented solve identical to naive, EXACTLY.
    tau0 = torch.zeros(K, device=DEV, dtype=DT)
    nu0 = torch.zeros(K, device=DEV, dtype=DT)
    a_zero = physics_posterior(Phi, z64, gam, s2, sites=(Fc, tau0, nu0))
    d_exact = float((a_zero - a_naive).abs().max())

    # (b) ep_neuron on the SAME rows: cavity from the naive posterior, event
    # phi^T a < h satisfied by many sigma -> the neuron self-deactivates.
    A = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    S_cov = torch.linalg.inv(A)
    mc = Fc @ a_naive                       # cavity mean of x = phi^T a
    vc = ((Fc @ S_cov) * Fc).sum(1)
    hc_t = torch.tensor(hc, device=DEV, dtype=DT)
    tau_ep, nu_ep = ep_neuron(mc, vc, hc_t, sgn=+1.0)
    a_ep = physics_posterior(Phi, z64, gam, s2, sites=(Fc, tau_ep, nu_ep))
    d_ep = float((a_ep - a_naive).abs().max())
    alpha_med = float(((hc_t - mc) / torch.sqrt(vc.clamp_min(1e-12))).median())

    ok = (d_exact == 0.0)
    print(f"     {K} constraint rows, tau = 0 : max|diff| = {d_exact:.1e} "
          f"(must be exactly 0)")
    print(f"     ep_neuron on satisfied events (median alpha = {alpha_med:.1f} "
          f"sigma): max site tau = {float(tau_ep.max()):.1e}, "
          f"solve max|diff| = {d_ep:.1e}")
    print(f"     {'PASS' if ok else 'FAIL'}")
    return ok, {'zero_site_diff': d_exact, 'ep_deactivate_tau_max':
                float(tau_ep.max()), 'ep_deactivate_diff': d_ep,
                'alpha_med': alpha_med}


# ==================================================================== EQ
def test_phase_equivariance():
    print("\n[EQ] phase equivariance: block commutes with per-mode rotation")
    torch.manual_seed(0)
    B = 400
    a = torch.randn(6, B, 2, dtype=DT, device=DEV)
    om = (torch.rand(B, dtype=DT, device=DEV) * 0.8 + 0.05)   # distinct freqs
    dt = 4.3
    blk = PhaseEquivariantBlock(B, hidden=16, dtype=DT).to(DEV)
    # random (non-identity) weights so the test has teeth
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn_like(p))

    with torch.no_grad():
        lhs = blk(phase_rotate(a, om, dt))      # rotate then apply
        rhs = phase_rotate(blk(a), om, dt)      # apply then rotate
    d_block = float((lhs - rhs).abs().max())

    # physics evolution obeys the group law exactly
    g_split = phase_rotate(phase_rotate(a, om, 1.6), om, 2.7)
    g_once = phase_rotate(a, om, 4.3)
    d_group = float((g_split - g_once).abs().max())

    ok = (d_block <= TOL) and (d_group <= TOL)
    print(f"     block: max|rotate.apply - apply.rotate| = {d_block:.2e}"
          f"   gate {TOL:.0e}")
    print(f"     physics group law: max|R(s)R(t) - R(s+t)| = {d_group:.2e}")
    print(f"     {'PASS' if ok else 'FAIL'}")
    return ok, {'eq_block_diff': d_block, 'eq_group_diff': d_group}


# ==================================================================== C3
def test_c3_impossibility():
    print("\n[C3] support-memorization impossibility on a 2.5 s window")
    rng = np.random.default_rng(5)
    S = sea(2)
    kp = kp_of(S)
    xs = rng.uniform(-R_MAX, R_MAX, 4000)
    ys = rng.uniform(-R_MAX, R_MAX, 4000)
    sig_field = float(np.std(eta(xs, ys, 0.0, S)))

    # extended bank INCLUDING sub-floor atoms (k in [0.10, 0.35] k_p), flat prior
    # (as in S1); the contraction is a property of the design + prior alone.
    kxb, kyb, om = build_bank(kp, (0.10, 2.5), kp / 10.0, DEPTH, DEV, DT)
    B = len(kxb)
    gm = torch.ones(B, device=DEV, dtype=DT) / B
    gam = torch.cat([gm, gm]) * (sig_field ** 2)
    s2 = NOISE ** 2
    kmag2 = torch.cat([torch.hypot(kxb, kyb), torch.hypot(kxb, kyb)]) / kp
    sub = kmag2 < 0.35
    sqrt_g = torch.sqrt(gam)

    out = {}
    c_by_T = {}
    for T in (20.0, 2.5):
        P, _ = obs_positions(S, T, 12, np.random.default_rng(1))
        Phi = design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om)
        c = contraction_gate(Phi, gam, s2)
        c_by_T[T] = c
        cm = float(c[sub].mean())
        cx = float(c[sub].max())
        cin = float(c[(kmag2 >= 1.0) & (kmag2 < 1.6)].mean())
        print(f"     T={T:4.1f}s  sub-floor(k<0.35kp) c mean {cm:.4f} max {cx:.4f}"
              f"   in-band(1.0-1.6kp) c {cin:.3f}")
        out[f'sub_c_mean_T{T}'] = cm
        out[f'sub_c_max_T{T}'] = cx
        out[f'inband_c_T{T}'] = cin

    c = c_by_T[2.5]
    collapse = out['sub_c_mean_T20.0'] / max(out['sub_c_mean_T2.5'], 1e-9)

    # full-scale adversarial delta = +/-1 everywhere, pushed through the gate.
    raw = (torch.randint(0, 2, (2 * B,), device=DEV, dtype=DT) * 2 - 1) * 20.0

    class _Sat(nn.Module):
        def __init__(self, r):
            super().__init__()
            self.r = r

        def forward(self, x):
            return self.r

    net = GatedDelta(_Sat(raw), scale=1.0)
    a_zero = torch.zeros(2 * B, device=DEV, dtype=DT)
    inj = net(a_zero, c, None)                         # = c (*) delta, |delta|<=1

    realized = float((inj[sub].abs() * sqrt_g[sub]).sum())     # amplitude L1
    bound = float((c[sub] * 1.0 * sqrt_g[sub]).sum())          # sum c |delta| sqrt(g)
    prior_energy = float(sqrt_g[sub].sum())                    # ungated, |delta|=1
    ratio = realized / prior_energy

    ok = (realized <= bound * (1 + 1e-9)) and (ratio < 0.10) \
        and (out['sub_c_max_T2.5'] < 0.06)
    print(f"     sub-floor contraction collapse 20s -> 2.5s: {collapse:.2f}x")
    print(f"     worst-case injectable amplitude (gated)  = {realized:.4e}")
    print(f"     theoretical bound  sum c_i |delta| sqrt(gam_i) = {bound:.4e}")
    print(f"     sub-floor prior amplitude (ungated net)  = {prior_energy:.4e}")
    print(f"     injectable / prior = {ratio * 100:.2f}%   (max sub-floor c "
          f"{out['sub_c_max_T2.5']:.4f} < 0.06)")
    print(f"     {'PASS' if ok else 'FAIL'}")
    out.update({'c3_collapse': collapse, 'c3_realized': realized,
                'c3_bound': bound, 'c3_prior_energy': prior_energy,
                'c3_ratio': ratio, 'sig_field': sig_field,
                'sub_atoms_2B': int(sub.sum().item())})
    return ok, out


def main():
    print(f"device {DEV}  dtype {DT}")
    results = []
    r1_ok, r1 = test_r1()
    id_ok, idd = test_zero_site()
    eq_ok, eq = test_phase_equivariance()
    c3_ok, c3 = test_c3_impossibility()
    results = [('R1 reduction', r1_ok), ('ID zero-site', id_ok),
               ('EQ phase-equivariance', eq_ok), ('C3 impossibility', c3_ok)]

    print("\n" + "=" * 60)
    for name, ok in results:
        print(f"  {name:<26s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in results)
    print("=" * 60)
    print(f"  ALL {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
