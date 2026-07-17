"""e42 probe (Designer 3, MECHANISM SMOKE ONLY — n=6, no verdicts).

Tests two load-bearing claims of the D3 design before writing the doc:
  (1) JOINT-grid evidence (no stagewise lobe1 freezing) posterior
      concentrates near the true (d1,d2,ring2,w) on cross-Tp seas.
  (2) solve-level BMA (mixture of conditional MMSE fields) vs gamma-level
      BMA (mix grids, solve once) vs uniform smear (m2 proxy) vs cur.

Seas: build_sea_crosstp(200+j) — disjoint from scored si=0..31 and
calibration 500+/3000+. NOT for scoring claims.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import corr, DEVICE, _MASK
from e29_gate0_ladder import SMULT, solve_with_grid
from e29_gennet_C import bank_atoms, design_t
from e37_bimodal_ladder import (build_sea_crosstp, net_gamma_map, mu_hat_of,
                                ang_err_deg, true_radial_grid)
from e41_evidence_detector import EvidenceEngine, ring_lobe_atoms, true_rings

D1_OFF = np.deg2rad([0., 25., -25., 50., -50., 90., -90., 140., -140.])
A1_GRID = [0.90, 1.00, 1.10]
D2_OFF = np.deg2rad([60., 105., 150., -60., -105., -150.])
A2_GRID = [0.75, 1.00, 1.30, 1.60]
W_GRID = [0.15, 0.30, 0.45]


def solve_field(S, grid):
    """solve_with_grid clone that returns (field, corr)."""
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    g = np.concatenate([grid[_MASK], np.zeros(_MASK.sum())])
    g = g / g.sum()
    kxb, kyb, om = bank_atoms(kp)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32,
                               device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(P))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(Pq))
    gam = t(np.concatenate([g, g]) * float(np.var(z)))
    K = (Phi * gam[None]) @ Phi.T \
        + (SMULT * s0) ** 2 * torch.eye(len(z), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(z))
    f = ((Phi_q * gam[None]) @ w).cpu().numpy()
    return f, float(corr(f, gt))


def probe_sea(si):
    S = build_sea_crosstp(si)
    eng = EvidenceEngine(S)
    kp = S['kp']
    mu_hat = mu_hat_of(S)
    cache = {}

    cands, Gat, Ggrid = [], [], []
    for dd in D1_OFF:
        d1c = mu_hat + dd
        for a1 in A1_GRID:
            at1, g1 = ring_lobe_atoms(kp, a1, d1c, cache)
            for off in D2_OFF:
                d2c = d1c + off
                for a2 in A2_GRID:
                    at2, g2 = ring_lobe_atoms(kp, a2, d2c, cache)
                    for w in W_GRID:
                        cands.append((d1c, a1, d2c, a2, w))
                        Gat.append((1 - w) * at1 + w * at2)
                        Ggrid.append((1 - w) * g1 + w * g2)
    Gat = np.stack(Gat)
    loge = eng.logev(Gat)
    pw = np.exp(loge - loge.max())
    pw /= pw.sum()

    # posterior diagnostics (match fitted lobes to true by direction)
    def d2err_of(c):
        d1c, a1c, d2c, a2c, wc = c
        e_dir = ang_err_deg(d1c, S['d1']) + ang_err_deg(d2c, S['d2'])
        e_sw = ang_err_deg(d1c, S['d2']) + ang_err_deg(d2c, S['d1'])
        return ang_err_deg(d2c, S['d2']) if e_dir <= e_sw \
            else ang_err_deg(d1c, S['d2'])

    imax = int(np.argmax(loge))
    a1t, a2t = true_rings(S)
    top = cands[imax]
    ent = float(-(pw * np.log(pw + 1e-300)).sum())
    # posterior mass "near truth": d2 within 25 deg AND ring2 within 12%
    near = 0.0
    for c, p in zip(cands, pw):
        if p < 1e-6:
            continue
        d1c, a1c, d2c, a2c, wc = c
        e_dir = ang_err_deg(d1c, S['d1']) + ang_err_deg(d2c, S['d2'])
        if e_dir <= ang_err_deg(d1c, S['d2']) + ang_err_deg(d2c, S['d1']):
            de, ae = ang_err_deg(d2c, S['d2']), abs(a2c / a2t - 1)
        else:
            de, ae = ang_err_deg(d1c, S['d2']), abs(a1c / a2t - 1)
        if de <= 25.0 and ae <= 0.12:
            near += p

    row = dict(si=si, r2=S['r2'], Tp2=S['Tp2'],
               mu_err=ang_err_deg(mu_hat, S['d1']),
               top_d2err=d2err_of(top), post_near=float(near), entropy=ent,
               top_w=top[4], w_true=S['r2'])

    # arms
    Y = net_gamma_map(S)
    row['cur'] = solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8))
    Gg = np.stack(Ggrid)
    row['uni_bma'] = solve_with_grid(S, Gg.mean(0))          # smear proxy
    row['g_bma'] = solve_with_grid(S, np.tensordot(pw, Gg, axes=1))
    keep = np.where(pw > 0.005)[0][:24]
    wk = pw[keep] / pw[keep].sum()
    fmix = None
    for j, i in enumerate(keep):
        f, _ = solve_field(S, Ggrid[i])
        fmix = wk[j] * f if fmix is None else fmix + wk[j] * f
    row['s_bma'] = float(corr(fmix, S['gt_u']))
    row['n_keep'] = int(len(keep))
    go = (1 - S['r2']) * true_radial_grid(S, 7.0, S['d1']) \
        + S['r2'] * true_radial_grid(S, S['Tp2'], S['d2'])
    row['oracle'] = solve_with_grid(S, go)
    return row


def main():
    t0 = time.time()
    rows = []
    for j in range(6):
        r = probe_sea(200 + j)
        rows.append(r)
        print(f"si{r['si']} r2 {r['r2']:.2f} Tp2 {r['Tp2']:.1f} "
              f"muE {r['mu_err']:5.1f} | topD2e {r['top_d2err']:5.1f} "
              f"near {r['post_near']:.2f} H {r['entropy']:.2f} "
              f"nk {r['n_keep']:2d} | cur {r['cur']:+.3f} "
              f"uni {r['uni_bma']:+.3f} gB {r['g_bma']:+.3f} "
              f"sB {r['s_bma']:+.3f} orc {r['oracle']:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    for k in ('cur', 'uni_bma', 'g_bma', 's_bma', 'oracle'):
        print(f"  mean {k:8s} {np.mean([r[k] for r in rows]):+.4f}")
    print(f"  mean topD2e {np.mean([r['top_d2err'] for r in rows]):.1f} | "
          f"mean near-mass {np.mean([r['post_near'] for r in rows]):.2f}")


if __name__ == '__main__':
    main()
