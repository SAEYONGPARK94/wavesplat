"""E-30b: WHY does rung4 win? sharpness (free) or correct speckle (realization)?

build_sea_c's rng touches ONLY phases and directions; the radial profile is a
universal constant. So gamma_true = {universal |k_n|} x {theta_n ~ N(mu,30deg)}
and  rung4 - rung2  ==  "knowing the 200 REALIZED angle draws" (the speckle of
the empirical direction histogram), nothing else.

THE CRITICAL CONTROL (perm): keep the sharpness, the sparsity, the radial
profile and the true mean direction -- but redraw the 200 angles from the SAME
N(mu_true, 30deg). Same prior CLASS, correct ensemble, WRONG realization.
  perm ~ rung4  => sharpness alone is the prize, it needs no oracle => OPEN
  perm ~ rung2  => the prize is correct speckle = pure realization => CLOSED

Plus the deconvolution family (C's named survivor), given every advantage:
  bf_true  : beamform map of the NOISE-FREE field  (= gamma_true (x) |PSF|^2
             exactly; infinite SNR) used as gamma, sharpening exponent p swept
             with GT-peek  -> upper bound on "sharpen the periodogram"
  rl_true  : Richardson-Lucy deconvolution of bf_true, iters swept with GT-peek
             -> upper bound on deconvolution AT INFINITE SNR
  bf_obs / rl_obs : the same from the ACTUAL noisy z (deployable)
If GT-peeked RL at infinite SNR cannot approach rung4, deconvolution is dead.
"""
import os, sys, json
import numpy as np
import torch
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_power_np, corr,
                          DEVICE, DEPTH, _MASK, REGIMES, dispersion_lam)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from d18_relax_estimator import beamform_top, omega
from d3_sa_achievability_gap import hashabs
from d3_prior_net_train import NGRID, KREL

OUT = 'results/direction3/e30_speckle_control.json'
PEXP = [0.5, 1.0, 2.0, 4.0, 8.0]
RL_IT = [0, 5, 20, 100, 500]


def bf_grid(P, r, kp):
    """beamform power of series r at every FREE bank atom -> 51x51 grid."""
    pw = bank_power_np(P.astype(np.float64), r.astype(np.float64), kp)
    pw = pw[:len(pw) // 2]
    g = np.zeros((NGRID, NGRID))
    g[_MASK] = pw
    return g


def psf_kernel(P, kp, mu):
    """|PSF|^2 on the bank, re-centred at the grid centre (shift-invariant
    approximation, used only as the RL blur kernel)."""
    k0x, k0y = kp * np.cos(mu), kp * np.sin(mu)
    om0 = float(omega(np.array([kp]), DEPTH)[0])
    r = np.cos(k0x * P[:, 0] + k0y * P[:, 1] - om0 * P[:, 2])
    g = bf_grid(P, r, kp)
    # roll peak to centre
    pk = np.unravel_index(np.argmax(g), g.shape)
    c = (NGRID // 2, NGRID // 2)
    g = np.roll(np.roll(g, c[0] - pk[0], 0), c[1] - pk[1], 1)
    return g / max(g.sum(), 1e-12)


def rl_deconv(img, psf, n_it):
    """Richardson-Lucy (non-negative, energy-preserving)."""
    if n_it <= 0:
        return img.copy()
    img = np.maximum(img, 0)
    est = np.full_like(img, img.mean() + 1e-9)
    pf = psf[::-1, ::-1]
    for _ in range(n_it):
        conv = fftconvolve(est, psf, mode='same')
        rel = img / np.maximum(conv, 1e-12)
        est = est * np.maximum(fftconvolve(rel, pf, mode='same'), 0)
        est = np.maximum(est, 0)
    return est


def sweep(S, base, exps):
    """best-over-p (GT-peek) of gamma = base**p."""
    best, bp = -2.0, None
    for p in exps:
        g = np.maximum(base, 0) ** p
        if g.sum() <= 0:
            continue
        c = solve_with_grid(S, g)
        if np.isfinite(c) and c > best:
            best, bp = c, p
    return best, bp


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    lad = {r['si']: r for r in
           json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']}
    R = REGIMES['spread']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        cur = ref[si]['cur']
        kp = S['kp']
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _hs = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        kmag = np.hypot(S['m_kx'], S['m_ky'])
        en = S['m_A'] ** 2 / 2

        # --- CRITICAL CONTROL: same sharpness/sparsity/radial, WRONG speckle
        perm = []
        for rep in range(3):
            pr = np.random.default_rng(90000 + 7 * si + rep)
            th = pr.normal(mu_true, np.deg2rad(30.0), len(kmag))
            gp = deposit_grid(kmag * np.cos(th), kmag * np.sin(th), en, kp)
            perm.append(solve_with_grid(S, gp))
        perm_m = float(np.mean(perm))

        # --- deconvolution family, infinite-SNR and deployable
        psi = (S['m_kx'][None] * S['P'][:, 0:1] + S['m_ky'][None] * S['P'][:, 1:2]
               - omega(kmag, DEPTH)[None] * S['P'][:, 2:3] + S['m_ph'][None])
        z_clean = (S['m_A'][None] * np.cos(psi)).sum(1)
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * kp, n=120)
        mu_hat = np.arctan2(by, bx)
        K = psf_kernel(S['P'], kp, mu_hat)
        out = {}
        for tag, r in (('true', z_clean), ('obs', S['z'])):
            b = bf_grid(S['P'], r, kp)
            b = b / max(b.max(), 1e-12)
            c_bf, p_bf = sweep(S, b, PEXP)
            best_rl, arg_rl = -2.0, None
            for it in RL_IT:
                d = rl_deconv(b, K, it)
                d = d / max(d.max(), 1e-12)
                c, p = sweep(S, d, PEXP)
                if c > best_rl:
                    best_rl, arg_rl = c, (it, p)
            out[f'bf_{tag}'] = c_bf
            out[f'bf_{tag}_p'] = p_bf
            out[f'rl_{tag}'] = best_rl
            out[f'rl_{tag}_arg'] = arg_rl
        rows.append(dict(si=si, cur=cur, perm=perm_m, perm_all=perm,
                         ens_blind=lad[si]['ens_blind'],
                         real_free=lad[si]['real_free'], **out))
        r = rows[-1]
        print(f"sea{si:2d} cur {cur:+.3f} | blind {r['ens_blind']:+.3f} "
              f"| PERM {perm_m:+.3f} | real4 {r['real_free']:+.3f} "
              f"|| bf_true {r['bf_true']:+.3f} rl_true {r['rl_true']:+.3f} "
              f"| bf_obs {r['bf_obs']:+.3f} rl_obs {r['rl_obs']:+.3f}", flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')
    print(f"\n=== E-30b speckle control + deconvolution ceiling (n={len(rows)}) ===")
    for k, lab in [('ens_blind', 'rung2 ens_blind (smooth ensemble)'),
                   ('perm', '★PERM (sharp, right ensemble, WRONG speckle)'),
                   ('real_free', 'rung4 real_free (RIGHT speckle)'),
                   ('bf_true', 'bf_true  (noiseless periodogram, p-swept GT-peek)'),
                   ('rl_true', 'rl_true  (RL deconv @ inf SNR, GT-peek)'),
                   ('bf_obs', 'bf_obs   (noisy periodogram, GT-peek)'),
                   ('rl_obs', 'rl_obs   (RL deconv deployable, GT-peek)')]:
        d = a(k) - cur
        print(f"  {lab:48s}: {a(k).mean():+.4f}  vs cur {d.mean():+.4f}  "
              f"(양수 {int((d>0).sum())}/{len(d)})")
    print(f"\n  ★분해: sharpness가 사는가?  PERM-blind = {(a('perm')-a('ens_blind')).mean():+.4f}")
    print(f"         speckle의 값        real4-PERM  = {(a('real_free')-a('perm')).mean():+.4f}")
    print(f"         deconv이 speckle 회수? rl_true-bf_true = {(a('rl_true')-a('bf_true')).mean():+.4f}")
    json.dump(dict(rows=rows), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
