"""N3: kill WaveKF before building it, or find out it is worth building.

An independent red-team review of the WaveKF design (docs/N2) landed six
verdicts, three of them fatal, and handed back the cheapest possible test:

    exponential forgetting w_j = lam^((t0 - t_j)/dt) is EXACTLY a scalar
    constant-Q Kalman filter for a linear-Gaussian model. lam = 1 is the batch
    estimator we already have. If a scalar lam < 1 recovers nothing, no learned
    radial Q(k/kp) will recover more -- the learned version's increment is
    bounded by what the scalar leaves on the table.

The review also caught a category error in the design's own prize. O1's
`nonstat` number (+0.0101) is the SELECTOR's regret against a twenty-candidate
oracle under non-stationarity. It is not the headroom of temporal modelling over
the batch estimator. Nobody has measured that. What O1 does show is that
non-stationarity degrades the ORACLE ITSELF from 0.146 to 0.355 -- so the
headroom, if any, is in that 0.209, not in the 0.010.

So this script measures the thing that was never measured:

    field nRMSE of the estimator, as a function of lam, on
      clean     stationary synthetic seas          (forgetting should HURT)
      nonstat   amplitude-drifting synthetic seas  (forgetting should HELP)
      real      the 30 record-window cells         (the only cells that matter)

PRE-REGISTERED, before running:
  1. lam = 1 must reproduce the batch estimator exactly (identity already
     verified: max|diff| = 0.000000 with unit weights).
  2. on nonstat seas, some lam < 1 must beat lam = 1 by >= 0.010. If not, there
     is no temporal headroom even where we INJECTED non-stationarity, and
     WaveKF is dead.
  3. on real cells, the best lam must beat lam = 1 by >= 0.005. The red team's
     power calculation says 30 cells cannot resolve +0.010 (per-cell paired sd
     ~ 0.049 => n ~ 150 for 80% power), so a null result here is NOT evidence of
     absence -- but a clear win would be evidence of presence.
  4. NEGATIVE CONTROL: on clean stationary seas, forgetting must HURT. If it
     helps there too, lam is not modelling process noise, it is modelling
     something else (e.g. reweighting toward the scored instant), and the whole
     interpretation collapses.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_l1_learned_prior import S0
from d3_o1_defect_ablation import synth_sea, observe
from d3_g3_adversarial_sea import geometry, DEV, DEPTH, KP
from d3_a2_real_revalidation import draw, CANDS, HELD
import d3_static_field as SF

OUT = 'results/direction3/n3_forgetting.json'
LAMS = (1.0, 0.995, 0.99, 0.98, 0.95, 0.90)
DT = 0.5                       # scan interval
REAL_RECS = ['AA14', 'AA15a', 'AA15b', 'LJ18g', 'YS17']
WINDOWS = (0, 1, 2, 3, 4, 5)
RSEEDS = (11, 12, 13)


def weights(P, lam, t0):
    if lam >= 1.0:
        return None
    age = (t0 - P[:, 2]) / DT
    return torch.pow(torch.tensor(lam, device=DEV), age).clamp_min(1e-3)


def synthetic(cfg, lam, n_sea, gen):
    P0, Pq = geometry()
    vals = []
    for i in range(n_sea):
        rng = np.random.default_rng(5000 + i)
        s = synth_sea(rng)
        P, z, gt = observe(P0, Pq, s, rng, gen, cfg)
        if len(z) < 150:
            continue
        w = weights(P, lam, 0.0)
        with torch.no_grad():
            pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP,
                          band=(0.15, 2.5), obs_weight=w)
            pr = pr - pr.mean()
            vals.append(float(torch.sqrt(((pr - gt) ** 2).mean()) / gt.std()))
    v = np.array(vals)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), v


def real(lam, gen):
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    cells = []
    for rec in REAL_RECS:
        for wi in WINDOWS:
            try:
                sc = SF.scene_real(rec, win=wi)
            except Exception:
                continue
            Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
            eta = sc['make_eta']()
            g_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
            cx, cy = sc['center']
            inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
            m = inside.ravel() & np.isfinite(g_all)
            if m.sum() < 150:
                continue
            Pq = t(np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                                    np.full(m.sum(), sc['t0'])]))
            gtq = t(g_all[m])
            gtq = gtq - gtq.mean()
            if float(gtq.std()) < 1e-6:
                continue
            sig0 = 0.05 * sc['Hs']
            per = []
            for ds in RSEEDS:
                rng = np.random.default_rng(1000 + ds)
                P, Z = draw(sc, rng)
                if len(Z) < 200:
                    continue
                Pt, zt = t(P), t(Z)
                w = weights(Pt, lam, sc['t0'])
                with torch.no_grad():
                    n = len(zt)
                    idx = torch.randperm(n, device=DEV, generator=gen)
                    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
                    best, bb = None, CANDS[0]
                    for b in CANDS:
                        wt = None if w is None else w[tr]
                        pv = mmse_gpu(Pt[tr], zt[tr], sc['depth'], Pt[he],
                                      sigma0=sig0, kp_hint=sc['kp'], band=b,
                                      obs_weight=wt)
                        r = float(torch.sqrt(((pv - zt[he]) ** 2).mean()))
                        if best is None or r < best:
                            best, bb = r, b
                    pr = mmse_gpu(Pt, zt, sc['depth'], Pq, sigma0=sig0,
                                  kp_hint=sc['kp'], band=bb, obs_weight=w)
                    pr = pr - pr.mean()
                    per.append(float(torch.sqrt(((pr - gtq) ** 2).mean())
                                     / gtq.std()))
            if per:
                cells.append(np.mean(per))
    return np.array(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=20)
    ap.add_argument('--skip-real', action='store_true')
    a_ = ap.parse_args()
    gen = torch.Generator(device=DEV).manual_seed(0)
    res = {}

    print(f"w_j = lam^((t0 - t_j)/{DT} s).  lam = 1 is the batch estimator.\n")
    print(f"{'lam':>7}{'clean':>18}{'nonstat':>18}")
    base = {}
    for lam in LAMS:
        c = synthetic('none', lam, a_.seas, gen)
        n = synthetic('nonstat', lam, a_.seas, gen)
        base[lam] = (c[2], n[2])
        print(f"{lam:>7.3f}{c[0]:>12.4f}±{c[1]:.4f}{n[0]:>12.4f}±{n[1]:.4f}",
              flush=True)
        res[str(lam)] = dict(clean=c[0], clean_se=c[1], nonstat=n[0],
                             nonstat_se=n[1])

    # paired, since the same seas are used at every lam
    print("\npaired vs lam = 1 (same seas):")
    for lam in LAMS[1:]:
        dc = base[1.0][0] - base[lam][0]
        dn = base[1.0][1] - base[lam][1]
        print(f"  lam {lam:.3f}   clean {dc.mean():+.4f}±"
              f"{dc.std(ddof=1)/np.sqrt(len(dc)):.4f}   "
              f"nonstat {dn.mean():+.4f}±{dn.std(ddof=1)/np.sqrt(len(dn)):.4f}")
        res[str(lam)]['d_clean'] = float(dc.mean())
        res[str(lam)]['d_nonstat'] = float(dn.mean())

    best_n = max(LAMS[1:], key=lambda l: res[str(l)]['d_nonstat'])
    g2 = res[str(best_n)]['d_nonstat'] >= 0.010
    g4 = res[str(best_n)]['d_clean'] < 0
    print(f"\n=== N3 synthetic gates ===")
    print(f"  2) some lam beats batch by >= 0.010 on nonstat : "
          f"{'PASS' if g2 else 'FAIL'}  (best lam {best_n}, "
          f"{res[str(best_n)]['d_nonstat']:+.4f})")
    print(f"  4) forgetting HURTS on clean seas             : "
          f"{'PASS' if g4 else 'FAIL'}  ({res[str(best_n)]['d_clean']:+.4f})")
    if not g2:
        print("\n  no temporal headroom even where non-stationarity was "
              "INJECTED.\n  WaveKF is dead before it is built.")

    if not a_.skip_real:
        print(f"\n{'lam':>7}{'real (30 cells)':>20}{'vs lam=1':>14}")
        cells = {}
        for lam in LAMS:
            c = real(lam, gen)
            cells[lam] = c
            d = cells[1.0] - c if lam != 1.0 else np.zeros(len(c))
            se = d.std(ddof=1) / np.sqrt(len(d)) if lam != 1.0 else 0.0
            print(f"{lam:>7.3f}{c.mean():>14.4f} (n={len(c)})"
                  f"{d.mean():>+10.4f}±{se:.4f}", flush=True)
            res[str(lam)]['real'] = float(c.mean())
            res[str(lam)]['d_real'] = float(d.mean())
        best_r = max(LAMS[1:], key=lambda l: res[str(l)]['d_real'])
        g3 = res[str(best_r)]['d_real'] >= 0.005
        print(f"\n  3) best lam beats batch by >= 0.005 on real   : "
              f"{'PASS' if g3 else 'FAIL'}  (lam {best_r}, "
              f"{res[str(best_r)]['d_real']:+.4f})")
        print("  (a null here is not evidence of absence: 30 cells cannot "
              "resolve +0.010)")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
