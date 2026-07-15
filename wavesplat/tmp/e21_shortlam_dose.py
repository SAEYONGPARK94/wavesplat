"""E-21 (closing #A) — the nonlinearity-dose test the statistician demanded.

The load-bearing physics claim ("in the well-determined short-λ regime the
learning gain IS bound-wave misspecification") rested on only 2 wavelengths and
died when a 3rd was added; the pre-registered aperture×physics interaction was
null. The clean test the referee asked for: on MANY short wavelengths (all
well-determined) with a real steepness range, is there a significant STEEPNESS
main-effect on Δm3 (= net − hand on the NONLINEAR field) — computed at the SEA
level (not on the derived difference dd, not a rank corr on a mechanical term)?

Grid: 5 short wavelengths (Tp 5.5-7.5, λ ~47-88 m, aperture-ratio 2.3-4.3, all
well-determined per E-19 CRB<0.11) × 4 steepnesses (0.01-0.04) × 3 phases.
Same fixed HOS-arena operator. Δm3 = net − hand on M3; Δlin on the linear field.
Registered test: sea-level (phase-averaged) OLS Δm3 ~ z(steepness), and Spearman,
over the 20 seas. PASS = steepness coefficient on Δm3 > 0 with p<0.05 (a genuine
nonlinearity dose in the well-determined regime).

  python3 tmp/e21_shortlam_dose.py
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from e18_decorrelated_2factor import build_cell, _lam, R_APERTURE  # noqa: E402

OUT = 'results/direction3/e21_shortlam_dose.json'
TPS = [5.5, 6.0, 6.5, 7.0, 7.5]          # λ ~47-88 m: all well-determined
STEEPS = [0.01, 0.02, 0.03, 0.04]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--draws', type=int, default=4)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    tps = TPS[:2] if a.smoke else TPS
    seeds = a.seeds[:1] if a.smoke else a.seeds
    seas = [(Tp, round(st * _lam(Tp), 3)) for Tp in tps for st in STEEPS]
    tasks = [(Tp, Hs, sd, order) for (Tp, Hs) in seas
             for sd in seeds for order in (1, 3)]
    print(f"E-21 short-λ dose: {len(seas)} seas × {len(seeds)} phases ×2 = "
          f"{len(tasks)} cells; λ {_lam(tps[0]):.0f}-{_lam(tps[-1]):.0f} m", flush=True)
    t0 = time.time()
    built = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for task, sea, note, dt in ex.map(build_cell, tasks):
            built[task] = sea
            if sea is None:
                print(f"  !! {task} -> {note}", flush=True)
    print(f"  built {sum(v is not None for v in built.values())}/{len(tasks)} in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    import torch
    from e15_hos_multidraw import one_draw, CKPT
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    net = get_net(CKPT); head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head']); head.eval()

    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp); steep = Hs / lam; ar = R_APERTURE / lam
        for sd in seeds:
            sl, s3 = built[(Tp, Hs, sd, 1)], built[(Tp, Hs, sd, 3)]
            if sl is None or s3 is None:
                continue
            dl, d3 = [], []
            for rr in range(a.draws):
                rng = np.random.default_rng(800000 + 31 * rr)
                h, n_, _ = one_draw(sl, rng, net, head); dl.append(n_ - h)
                rng = np.random.default_rng(800000 + 31 * rr)
                h, n_, _ = one_draw(s3, rng, net, head); d3.append(n_ - h)
            rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp, Hs=Hs, lam=lam,
                             steepness=steep, aperture_ratio=ar, seed=sd,
                             d_lin=float(np.mean(dl)), d_m3=float(np.mean(d3))))
            print(f"[Tp{Tp:g} Hs{Hs:g} λ{lam:.0f} steep{steep:.3f} ph{sd}] "
                  f"Δlin {np.mean(dl):+.3f} Δm3 {np.mean(d3):+.3f}", flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    # ---- sea-level nonlinearity-dose analysis ----
    rows = [r for r in rows if np.isfinite(r['d_m3']) and np.isfinite(r['d_lin'])]
    seas_u = sorted({r['sea'] for r in rows})
    sm = lambda k, s: np.mean([r[k] for r in rows if r['sea'] == s])
    dm3 = np.array([sm('d_m3', s) for s in seas_u])
    dlin = np.array([sm('d_lin', s) for s in seas_u])
    stp = np.array([[r['steepness'] for r in rows if r['sea'] == s][0] for s in seas_u])
    lam = np.array([[r['lam'] for r in rows if r['sea'] == s][0] for s in seas_u])
    zst, zlam = stats.zscore(stp), stats.zscore(lam)

    def ols(y, X):
        X = np.column_stack([np.ones(len(y))] + X)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b; dof = len(y) - X.shape[1]
        se = np.sqrt(np.diag((r @ r) / dof * np.linalg.inv(X.T @ X)))
        return b, b / se, 2 * stats.t.sf(np.abs(b / se), dof)

    print(f"\n{'='*66}\nE-21 short-λ nonlinearity dose ({len(seas_u)} seas, sea-level)"
          f"\n{'='*66}")
    print(f"  λ range {lam.min():.0f}-{lam.max():.0f} m (all well-determined)")
    b, t, p = ols(dm3, [zst, zlam])
    print(f"  Δm3 ~ z(steepness) + z(λ):")
    for nm, bi, ti, pi in zip(['int', 'steepness', 'λ'], b, t, p):
        print(f"     {nm:10s} beta {bi:+.4f}  t {ti:+.2f}  p {pi:.4f}")
    r_s, p_s = stats.spearmanr(stp, dm3)
    r_sl, p_sl = stats.spearmanr(stp, dlin)
    print(f"  Spearman(steepness, Δm3)  {r_s:+.3f} p {p_s:.4f}   <- the dose test")
    print(f"  Spearman(steepness, Δlin) {r_sl:+.3f} p {p_sl:.4f}")
    print(f"  Δm3 by steepness: " + "  ".join(
        f"{s:.2f}:{np.mean(dm3[np.abs(stp-s)<0.005]):+.3f}" for s in STEEPS))
    g = bool(b[1] > 0 and p[1] < 0.05)
    print(f"  *** dose PASS (steepness→Δm3 >0, p<0.05): {g}")
    json.dump({'rows': rows, 'summary': dict(
        n_seas=len(seas_u), lam_min=float(lam.min()), lam_max=float(lam.max()),
        dm3_steep_beta=float(b[1]), dm3_steep_t=float(t[1]), dm3_steep_p=float(p[1]),
        dm3_lam_beta=float(b[2]), dm3_lam_p=float(p[2]),
        spearman_steep_dm3=[float(r_s), float(p_s)],
        spearman_steep_dlin=[float(r_sl), float(p_sl)], dose_pass=g)},
        open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
