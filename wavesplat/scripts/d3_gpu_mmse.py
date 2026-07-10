"""d3_gpu_mmse.py — P0: the physics estimator on the GPU, differentiable.

`mmse_bank_estimate` is where nearly all of yesterday's compute went, and it is
`np.linalg.solve` on the CPU. That is why the bank has stayed at ~1,900 atoms
while yesterday's biggest measured lever was the bank's BAND (-32%).

Last night's timing: at n=2400, B=200,000 the gram + Cholesky costs 111 ms on
an RTX 4080 Super. The n x n factorisation is free (5 ms); the cost is the
gram (n x B), which is what a GPU is for.

A naive port is worthless: a flat prior scored nRMSE 6.45 against numpy's 0.558
on the same bank. What makes the estimator work is not the solve, it is

    gamma(k)  <- beamform power at the bank atoms, raised to a val-selected
                 exponent, scaled by a val-selected noise multiplier, then
                 refined by val-gated SBL-EM.

So this port carries the prior, the (pexp, smult) grid search on held-out
OBSERVATIONS, and the EM — not just the linear algebra. Everything stays in the
autograd graph, so downstream work (adversarial seas, control-through-estimator,
ensemble posteriors) becomes possible.

GATE: the port is only usable if it reproduces numpy. `--validate` compares
in-fan nRMSE across scenes; the plan calls anything above 0.005 a failure and
stops the day there.

Usage:
  python3 scripts/d3_gpu_mmse.py --validate
  python3 scripts/d3_gpu_mmse.py --bench --bank 200000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_bank(kp, band, dk_c, depth, device, dtype):
    """Bank atoms are chosen in numpy float64, exactly as d3_sa_nomp does.

    Doing the annulus test in torch gave 1,906 atoms where numpy gives 1,918:
    `hypot` rounds differently at `kk == b_hi * kp`, and the strict `<` then
    keeps or drops the atoms sitting on the boundary. Twelve atoms are enough
    to flip the (pexp, smult) argmin over its 9-point grid and move the field
    by 0.012 nRMSE — larger than the validation gate. The geometry must be
    bit-identical; only the linear algebra moves to the GPU.
    """
    b_lo, b_hi = band
    ax = np.arange(-b_hi * kp, b_hi * kp + dk_c / 2, dk_c)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > b_lo * kp) & (kk < b_hi * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    k = np.hypot(kxb, kyb)
    om = np.sqrt(G * k * np.tanh(k * depth))
    tt = lambda x: torch.tensor(x, device=device, dtype=dtype)
    return tt(kxb), tt(kyb), tt(om)


def design(Q, kxb, kyb, om):
    psi = (kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2]
           - om[None] * Q[:, 2:3])
    return torch.cat([torch.cos(psi), torch.sin(psi)], 1)


def beamform_prior(P, z, kxb, kyb, om, chunk=8192):
    """|sum_i z_i e^{-i psi_i}|^2 at every bank atom — the analytic prior."""
    B = len(kxb)
    pw = torch.empty(B, device=P.device, dtype=P.dtype)
    for i in range(0, B, chunk):
        psi = (kxb[i:i + chunk, None] * P[None, :, 0]
               + kyb[i:i + chunk, None] * P[None, :, 1]
               - om[i:i + chunk, None] * P[None, :, 2])
        pw[i:i + chunk] = (z[None] * torch.cos(psi)).sum(1) ** 2 \
            + (z[None] * torch.sin(psi)).sum(1) ** 2
    return pw / pw.max().clamp_min(1e-12)


def mmse_gpu(P, z, depth, Pq, sigma0, kp_hint, seed=0, band=(0.35, 2.5),
             dk_c=None, val_frac=0.25, max_em=6, return_var=False,
             q_chunk=512, dtype=torch.float32, sigma_scale=None,
             prior_gain=None, return_bank=False):
    """torch port of d3_sa_nomp.mmse_bank_estimate (free-wave bank, no
    dual_band / bound_wave / current). Differentiable in z and in the bank.

    prior_gain: optional (B,) multiplier on the beamform prior, before the
    (pexp, smult) selection. This is the ONE place theory allows learning to
    help -- gamma(k) is a modelling choice, not a posterior the physics fixes.
    Passing all-ones reproduces the analytic estimator exactly.

    return_bank: also return (kxb, kyb) so a caller can build prior_gain from
    the bank's own wavenumbers.

    sigma_scale: multiply sigma0 by this before the smult grid. Passing
    sqrt(n_ref / n) implements yesterday's I1 fix (regularization must track
    observation density, or the solution runs away outside the observed
    region).
    """
    dev = P.device
    n = len(z)
    g = np.random.default_rng(seed)
    vi = g.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = torch.zeros(n, dtype=torch.bool, device=dev)
    vmask[torch.tensor(vi, device=dev)] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]

    kp = float(kp_hint)
    if dk_c is None:
        dk_c = kp / 10.0
    kxb, kyb, om = build_bank(kp, band, dk_c, depth, dev, dtype)
    B = len(kxb)
    if sigma_scale is not None:
        sigma0 = sigma0 * float(sigma_scale)

    Phi_tr = design(Ptr, kxb, kyb, om)
    Phi_v = design(Pv, kxb, kyb, om)
    pw = beamform_prior(P, z, kxb, kyb, om)
    m = len(ztr)
    I = torch.eye(m, device=dev, dtype=dtype)
    varz = z.var(unbiased=False)      # np.var default is ddof=0

    if prior_gain is not None:
        pw = pw * prior_gain

    best = None
    for pexp in (0.5, 1.0, 2.0):
        gm = pw ** pexp
        gm = gm / gm.sum()
        gam = torch.cat([gm, gm]) * varz
        for smult in (1.0, 3.0, 10.0):
            s2 = (smult * sigma0) ** 2
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * I
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ torch.linalg.solve(K, ztr))
            vr = torch.sqrt(((fv - zv) ** 2).mean())
            if best is None or float(vr) < best[0]:
                best = (float(vr), pexp, smult, gam)
    vr_best, pexp, smult, gam = best
    s2 = (smult * sigma0) ** 2

    best_em = (vr_best, gam.clone())
    for _ in range(max_em):
        K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * I
        Ki_Phi = torch.linalg.solve(K, Phi_tr)
        mu = gam * (Phi_tr.T @ torch.linalg.solve(K, ztr))
        sig = gam - gam ** 2 * (Phi_tr * Ki_Phi).sum(0)
        gb = 0.5 * (mu[:B] ** 2 + mu[B:] ** 2
                    + sig[:B].clamp_min(0) + sig[B:].clamp_min(0))
        gam = torch.cat([gb, gb])
        K2 = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * I
        fv = (Phi_v * gam[None]) @ (Phi_tr.T @ torch.linalg.solve(K2, ztr))
        vr = float(torch.sqrt(((fv - zv) ** 2).mean()))
        if vr < best_em[0] - 1e-5:
            best_em = (vr, gam.clone())
        else:
            break
    gam = best_em[1]

    Phi_all = design(P, kxb, kyb, om)
    K = (Phi_all * gam[None]) @ Phi_all.T + s2 * torch.eye(
        n, device=dev, dtype=dtype)
    alpha = torch.linalg.solve(K, z)
    w_pred = Phi_all.T @ alpha
    Ki = torch.linalg.inv(K) if return_var else None

    out = torch.empty(len(Pq), device=dev, dtype=dtype)
    var = torch.empty(len(Pq), device=dev, dtype=dtype) if return_var else None
    for i in range(0, len(Pq), q_chunk):
        Phi_q = design(Pq[i:i + q_chunk], kxb, kyb, om)
        A = Phi_q * gam[None]
        out[i:i + q_chunk] = A @ w_pred
        if return_var:
            Bq = A @ Phi_all.T
            var[i:i + q_chunk] = (A * Phi_q).sum(1) - ((Bq @ Ki) * Bq).sum(1)
    if return_var:
        return out, var.clamp_min(0.0)
    return out


# ------------------------------------------------------------------ validate
def _scene(sea, seed):
    import d3_static_field as SF
    import d3_storm_wall as SW
    spec = SW.SEAS[sea]
    eta, kp, Hs = SW.make_sea(spec, 1000 + 37 * seed)
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1] + 0.1, SF.DX * 2)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1] + 0.1, SF.DY * 2)
    Xq, Yq = np.meshgrid(gx, gy)
    t0 = SF.W_OBS
    gt = eta(Xq.ravel(), Yq.ravel(), t0).reshape(Xq.shape)
    rng = np.random.default_rng(11 + seed)
    P, Z = SW.observations(eta, 'fan_noisy', rng, t0)
    Pq = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0)])
    inside, _ = SF.fan_mask(Xq, Yq, 0.0, 0.0, np.array([1.0, 0.0]))
    return P, Z, Pq, Xq, gt, inside, kp, Hs, spec['depth']


def validate(seeds=3):
    from d3_sa_nomp import mmse_bank_estimate
    print(f"{'sea':<9} {'seed':>4} {'numpy':>8} {'torch fp32':>11} "
          f"{'|diff|':>8} {'np ms':>8} {'gpu ms':>8}")
    print('-' * 62)
    worst = 0.0
    for sea in ('wind', 'storm', 'bimodal'):
        for sd in range(seeds):
            P, Z, Pq, Xq, gt, inside, kp, Hs, depth = _scene(sea, sd)
            sg = gt.std()
            t = time.time()
            f_np = mmse_bank_estimate(P, Z, depth, Pq, sigma0=0.05 * Hs,
                                      kp_hint=kp, seed=sd)
            t_np = (time.time() - t) * 1e3
            n_np = float(np.sqrt(np.mean(
                (f_np.reshape(Xq.shape)[inside] - gt[inside]) ** 2)) / sg)

            Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
            Zt = torch.tensor(Z, device=DEV, dtype=torch.float32)
            Qt = torch.tensor(Pq, device=DEV, dtype=torch.float32)
            torch.cuda.synchronize(); t = time.time()
            f_t = mmse_gpu(Pt, Zt, depth, Qt, 0.05 * Hs, kp, seed=sd)
            torch.cuda.synchronize(); t_gpu = (time.time() - t) * 1e3
            f_t = f_t.cpu().numpy()
            n_t = float(np.sqrt(np.mean(
                (f_t.reshape(Xq.shape)[inside] - gt[inside]) ** 2)) / sg)
            d = abs(n_t - n_np)
            worst = max(worst, d)
            print(f"{sea:<9} {sd:>4} {n_np:>8.3f} {n_t:>11.3f} {d:>8.4f} "
                  f"{t_np:>8.0f} {t_gpu:>8.0f}")
    print(f"\nworst |diff| = {worst:.4f}   gate = 0.005   "
          f"-> {'PASS' if worst < 0.005 else 'FAIL'}")
    return worst


def bench(bank_target):
    import d3_static_field as SF
    P, Z, Pq, Xq, gt, inside, kp, Hs, depth = _scene('wind', 0)
    sg = gt.std()
    Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
    Zt = torch.tensor(Z, device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Pq, device=DEV, dtype=torch.float32)
    print(f"{'dk_c':>10} {'B':>9} {'in-fan':>8} {'ms':>8}")
    for div in (10, 20, 40, 80, 160):
        dk = kp / div
        torch.cuda.synchronize(); t = time.time()
        f = mmse_gpu(Pt, Zt, depth, Qt, 0.05 * Hs, kp, seed=0, dk_c=dk,
                     q_chunk=256)
        torch.cuda.synchronize(); ms = (time.time() - t) * 1e3
        kxb, _, _ = build_bank(kp, (0.35, 2.5), dk, depth, DEV, torch.float32)
        n = float(np.sqrt(np.mean((f.cpu().numpy().reshape(Xq.shape)[inside]
                                   - gt[inside]) ** 2)) / sg)
        print(f"{dk:>10.5f} {len(kxb):>9d} {n:>8.3f} {ms:>8.0f}")
        if len(kxb) > bank_target:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--bank', type=int, default=200000)
    a = ap.parse_args()
    print(f"device {DEV}", flush=True)
    if a.validate:
        validate()
    if a.bench:
        bench(a.bank)


if __name__ == '__main__':
    sys.exit(main())
