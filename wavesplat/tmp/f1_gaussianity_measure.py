"""F-1 (assigned by coordinator, 2026-07-22) — §7 random-phase field Gaussianity,
ARGUED -> MEASURED re-derivation, independent of any prior inline computation.

Question: THESIS_FRAMING_v2 §7 argues "200 components -> CLT -> near-Gaussian field
marginal -> Gaussian-approx CRB correction small" citing Shapiro-Wilk W median 0.993
(min 0.985) and E58_AUDIT appendix citing regime-stratified W: 5comp 0.991 / 15comp
0.995 / 30comp 0.995, kurtosis -0.58 -> -0.33 -> -0.19. No script producing these
numbers exists on disk (same "no persisted artifact" pattern flagged as F3 in the
E60K audit) -> this script re-derives them from scratch, from raw episode data,
using the SAME episode pools as E-58/E-58b (tmp/e58b_audit.py pool()), so the
regime buckets are the ones the thesis actually cites.

Method: for each episode, keep amplitude/kx/ky per component (the "deterministic
amplitude" part of the sea state) and redraw phase phi ~ U(0, 2pi) per component,
R=300 times (independent draws == an ensemble of random-phase realizations of the
SAME sea state) — this operationalizes "the true field is random-phase, the
JointEngine likelihood approximates it as ensemble-Gaussian" (E-45 note). At nq=40
fixed query points (space, single mid-episode time) evaluate z(x,y,t) for each of
the R draws, then test whether that R-sample marginal distribution is close to
Gaussian (Shapiro-Wilk W, excess kurtosis). Aggregate per regime bucket.

Caveat inherited from THESIS_FRAMING §7 self-critique (not resolved by this run):
near-Gaussian marginal does NOT by itself quantitatively bound the size of the
random-phase CRB correction vs the Gaussian-approx CRB (W close to 1 supports the
DIRECTION of the argument, not a proven magnitude bound). This script measures
the marginal only — same scope as the original claim.

Usage: python3 tmp/f1_gaussianity_measure.py
Out:   results/direction3/f1_gaussianity.json + tmp/F1_gaussianity_2026-07-22.md
"""
import os
import sys
import glob
import json
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesim.data_writer import load_episode

G = 9.81
DEPTH = 50.0
R = 300
NQ = 40
OUT_JSON = 'results/direction3/f1_gaussianity.json'
OUT_MD = 'tmp/F1_gaussianity_2026-07-22.md'


def omega_of_k(k, depth):
    return np.sqrt(G * k * np.tanh(k * depth) + 1e-30)


def load_theta_grid(path):
    d = load_episode(path)
    wc = np.array(d['wave_components'])          # [A, omega, kx, ky, phi, Q]
    theta = np.stack([wc[:, 0], wc[:, 2], wc[:, 3], wc[:, 4]], axis=1)  # w,kx,ky,phi0
    gx = np.array(d['gt_grid_x']); gy = np.array(d['gt_grid_y'])
    ts = np.array(d['timestamps'])
    return theta, gx, gy, ts


def episode_gaussianity(theta, gx, gy, t_now, rng, R=R, nq=NQ, depth=DEPTH):
    w, kx, ky = theta[:, 0], theta[:, 1], theta[:, 2]
    N = theta.shape[0]
    k = np.sqrt(kx ** 2 + ky ** 2)
    omega = omega_of_k(k, depth)
    xq = rng.uniform(gx.min(), gx.max(), size=nq)
    yq = rng.uniform(gy.min(), gy.max(), size=nq)
    phi = rng.uniform(0.0, 2 * np.pi, size=(R, N))
    base = (kx[None, None, :] * xq[:, None, None]
            + ky[None, None, :] * yq[:, None, None]
            - omega[None, None, :] * t_now)                       # (nq,1,N) broadcast
    psi = base + phi[None, :, :]                                  # (nq,R,N)
    z = np.einsum('n,qrn->qr', w, np.cos(psi))                    # (nq,R)
    Ws, ks = [], []
    for q in range(nq):
        zq = z[q]
        if np.std(zq) < 1e-12:
            continue
        try:
            W, _p = stats.shapiro(zq)
        except Exception:
            continue
        kurt = stats.kurtosis(zq, fisher=True, bias=False)
        Ws.append(W); ks.append(kurt)
    return np.array(Ws), np.array(ks)


def pool(pat, k, ex=()):
    ps = sorted(glob.glob(f'data/episode_{pat}_*.h5'))
    ps = [p for p in ps if not any(e in p for e in ex)]
    return ps[:k]


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # same regime buckets as tmp/e58b_audit.py doseA (n=6/10/10/10, n=36 total)
    dose = {
        '2comp': pool('BF3_2comp', 6) + pool('BF4_2comp', 6),
        '5comp': pool('BF3_5comp', 10),
        '15comp': pool('BF4_15comp', 10),
        '30comp': pool('BF4_30comp', 10),
    }
    per_ep = []
    for lvl, paths in dose.items():
        for p in paths:
            try:
                theta, gx, gy, ts = load_theta_grid(p)
                t_now = float(ts[len(ts) // 2])
                Ws, ks = episode_gaussianity(theta, gx, gy, t_now, rng)
                if len(Ws) == 0:
                    print(f"  {os.path.basename(p)} EXCLUDED (degenerate variance)")
                    continue
                per_ep.append(dict(lvl=lvl, ncomp=int(lvl.replace('comp', '')),
                                    ep=os.path.basename(p),
                                    W_med=float(np.median(Ws)), W_min=float(np.min(Ws)),
                                    kurt_med=float(np.median(ks)), n_q=len(Ws)))
                print(f"  {lvl:7s} {os.path.basename(p):32s} W_med={np.median(Ws):.4f} "
                      f"kurt_med={np.median(ks):+.3f} (nq={len(Ws)}) [{time.time()-t0:.0f}s]")
            except Exception as ex:
                print(f"  {os.path.basename(p)} EXCLUDED ({type(ex).__name__}: {ex})")

    summary = {}
    print("\n=== per-regime summary (median across episode-medians) ===")
    for lvl in dose:
        rr = [r for r in per_ep if r['lvl'] == lvl]
        if not rr:
            print(f"  {lvl:7s} n=0"); continue
        Wm = np.array([r['W_med'] for r in rr])
        km = np.array([r['kurt_med'] for r in rr])
        summary[lvl] = dict(n_ep=len(rr), W_med=float(np.median(Wm)), W_min=float(np.min(Wm)),
                             kurt_med=float(np.median(km)))
        print(f"  {lvl:7s} n_ep={len(rr):2d}  W_med={np.median(Wm):.4f}  "
              f"W_min={np.min(Wm):.4f}  kurt_med={np.median(km):+.3f}")

    all_W = np.array([r['W_med'] for r in per_ep])
    all_k = np.array([r['kurt_med'] for r in per_ep])
    pooled = dict(n_ep=len(per_ep), W_med=float(np.median(all_W)), W_min=float(np.min(all_W)),
                  kurt_med=float(np.median(all_k)))
    print(f"\n  POOLED (all regimes, n_ep={len(per_ep)}): W_med={pooled['W_med']:.4f} "
          f"W_min={pooled['W_min']:.4f} kurt_med={pooled['kurt_med']:+.3f}")

    # monotonicity check vs comp count (the CLT direction claim)
    ncomp = np.array([r['ncomp'] for r in per_ep])
    sc_W = stats.spearmanr(ncomp, all_W)
    sc_k = stats.spearmanr(ncomp, np.abs(all_k))
    print(f"  Spearman(ncomp, W)        = {sc_W.correlation:+.3f} p={sc_W.pvalue:.4g}")
    print(f"  Spearman(ncomp, |kurt|)   = {sc_k.correlation:+.3f} p={sc_k.pvalue:.4g}  "
          f"(expect negative: more comps -> smaller |excess kurtosis|)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(dict(per_episode=per_ep, per_regime=summary, pooled=pooled,
                   spearman_ncomp_W=dict(rho=float(sc_W.correlation), p=float(sc_W.pvalue)),
                   spearman_ncomp_abskurt=dict(rho=float(sc_k.correlation), p=float(sc_k.pvalue)),
                   config=dict(R=R, nq=NQ, depth=DEPTH, seed=0)),
              open(OUT_JSON, 'w'), indent=1, default=float)
    print(f"\nsaved -> {OUT_JSON}  ({time.time()-t0:.0f}s)")
    return per_ep, summary, pooled, sc_W, sc_k


if __name__ == '__main__':
    main()
