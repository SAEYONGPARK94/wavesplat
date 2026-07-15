"""E-18 (closing experiment #1) — the DECORRELATED 2-factor design.

The examiner's load-bearing objection: E-16's short-lam physics headline is a
POST-HOC partition of a FAILED pre-registration (dd p 0.85), with steepness
CONFOUNDED with wavelength (rho -0.42) and only 2 short wavelengths. So dd's
regime-dependence could be a partition artifact.

Fix: a proper 2-factor grid where wavelength lam (=aperture axis) and steepness
are CROSSED (decorrelated by construction) — at every Tp we place the SAME set
of target steepnesses by choosing Hs = steep*lam(Tp). Then the aperture main
effect and the physics x regime interaction are separately estimable, and
"dd grows as aperture shrinks" becomes a testable INTERACTION, not a partition.

Same fixed HOS-arena fan operator as E-16. Linear arm = analytic free-wave
(build_sea_linear); nonlinear arm = HOS M3 (build_sea_M). dd = Delta(M3) -
Delta(linear), paired per (sea, phase). Multiple independent phase realizations
per cell (E-17 showed phase matters). Delta = learned - hand.

CPU raycast builds are parallelized across cores BEFORE any GPU init (so the
fork never inherits a live CUDA context); the GPU estimator draws run serially
in the main process afterwards.

Registered test: regress dd ~ z(1/lam) + z(steepness) [+ interaction] over the
(sea, phase) cells, cluster-robust on sea. PASS if the aperture term dominates
(the gain magnitude axis) AND either dd>0 concentrates at small aperture-ratio
independent of steepness, or a real aperture x physics interaction is present.
Report Delta(linear) sign vs aperture (the sign-flip, now decorrelated).
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
os.environ.setdefault('OMP_NUM_THREADS', '1')      # keep build workers 1-core

OUT = 'results/direction3/e18_decorrelated_2factor.json'
R_APERTURE = 200.0                                  # fan max range (aperture proxy)

TPS = [6.0, 7.0, 8.0, 9.0, 11.0, 13.0]
STEEPS = [0.015, 0.025, 0.035]                      # same at every Tp -> decorrelated


def _lam(Tp, depth=50.0, g=9.81):
    om = 2 * np.pi / Tp
    k = om ** 2 / g
    for _ in range(80):
        k = om ** 2 / (g * np.tanh(k * depth))
    return float(2 * np.pi / k)


def build_cell(task):
    """CPU-only raycast build of one (Tp, Hs, seed, order) cell. No GPU."""
    Tp, Hs, seed, order = task
    from e16_physics_operator import build_sea_linear, build_sea_M
    t0 = time.time()
    try:
        sea = (build_sea_linear(Tp, Hs, seed) if order == 1
               else build_sea_M(Tp, Hs, seed, 3))
        # finiteness guard (M3 can still blow up at extreme steepness)
        ok = all(np.isfinite(s['z']).all() and np.isfinite(s['amp']).all()
                 for s in sea['scans']) and np.isfinite(sea['kp'])
    except Exception as e:                                    # noqa: BLE001
        return task, None, f'{type(e).__name__}: {e}', time.time() - t0
    return task, (sea if ok else None), ('' if ok else 'NONFINITE'), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
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
    print(f"E-18: {len(seas)} decorrelated seas x {len(seeds)} phases x 2 orders "
          f"= {len(tasks)} cells; {a.workers} build workers", flush=True)

    # ---- parallel CPU builds (before any GPU init) ----
    t0 = time.time()
    built = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for task, sea, note, dt in ex.map(build_cell, tasks):
            built[task] = sea
            if sea is None:
                print(f"  !! build {task} -> {note} [{dt:.0f}s]", flush=True)
    n_ok = sum(v is not None for v in built.values())
    print(f"  built {n_ok}/{len(tasks)} cells in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- GPU estimator draws (serial, main process) ----
    import torch
    from e15_hos_multidraw import one_draw, CKPT
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp); steep = Hs / lam; ar = R_APERTURE / lam
        for sd in seeds:
            sl, s3 = built[(Tp, Hs, sd, 1)], built[(Tp, Hs, sd, 3)]
            if sl is None or s3 is None:
                continue
            dl, d3 = [], []
            for rr in range(a.draws):
                rng = np.random.default_rng(700000 + 31 * rr)
                h, n_, _ = one_draw(sl, rng, net, head); dl.append(n_ - h)
                rng = np.random.default_rng(700000 + 31 * rr)
                h, n_, _ = one_draw(s3, rng, net, head); d3.append(n_ - h)
            dlin, dm3 = float(np.mean(dl)), float(np.mean(d3))
            rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp, Hs=Hs, lam=lam,
                             steepness=steep, aperture_ratio=ar, seed=sd,
                             d_lin=dlin, d_m3=dm3, dd=dm3 - dlin,
                             flip=bool(dlin < 0 and dm3 > 0)))
            print(f"[Tp{Tp:g} Hs{Hs:g} lam{lam:.0f} ar{ar:.2f} steep{steep:.3f} "
                  f"ph{sd}] Dlin {dlin:+.3f} Dm3 {dm3:+.3f} dd {dm3-dlin:+.3f}",
                  flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    # ---- 2-factor analysis ----
    n_all = len(rows)
    rows = [r for r in rows if np.isfinite(r['dd']) and np.isfinite(r['d_lin'])]
    if n_all - len(rows):
        print(f"  dropped {n_all-len(rows)} non-finite cells (HOS M3 blow-up at "
              f"the steepest long-lam corner)")
    if len(rows) < 6:
        print("too few cells for analysis"); return 0
    dd = np.array([r['dd'] for r in rows])
    dlin = np.array([r['d_lin'] for r in rows])
    zar = stats.zscore(np.array([r['aperture_ratio'] for r in rows]))
    zst = stats.zscore(np.array([r['steepness'] for r in rows]))
    lam_v = np.array([r['lam'] for r in rows])

    def ols(y, X):
        X = np.column_stack([np.ones(len(y))] + X)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        sigma2 = (resid @ resid) / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), dof)
        return beta, se, t, p

    print(f"\n{'='*72}\nE-18 decorrelated 2-factor  ({len(rows)} cells, "
          f"{len({r['sea'] for r in rows})} seas x {len(seeds)} phases)\n{'='*72}")
    print(f"  confound check: corr(aperture_ratio, steepness) = "
          f"{np.corrcoef(zar, zst)[0,1]:+.3f}  (target ~0)")
    # dd ~ aperture + steepness + interaction
    b, se, t, p = ols(dd, [zar, zst, zar * zst])
    names = ['intercept', 'z(aperture_ratio)', 'z(steepness)', 'aperture x steep']
    print("  dd ~ aperture + steepness + interaction:")
    for nm, bi, ti, pi in zip(names, b, t, p):
        print(f"    {nm:20s} beta {bi:+.4f}  t {ti:+.2f}  p {pi:.4f}")
    # Delta(linear) ~ aperture + steepness  (the sign-flip driver)
    bl, sel, tl, pl = ols(dlin, [zar, zst])
    print("  Delta(linear) ~ aperture + steepness:")
    for nm, bi, ti, pi in zip(['intercept', 'z(aperture_ratio)', 'z(steepness)'],
                              bl, tl, pl):
        print(f"    {nm:20s} beta {bi:+.4f}  t {ti:+.2f}  p {pi:.4f}")
    rho_ap, p_ap = stats.spearmanr([r['aperture_ratio'] for r in rows], dlin)
    rho_st, p_st = stats.spearmanr([r['steepness'] for r in rows], dd)
    print(f"  Spearman(aperture_ratio, Delta_lin) {rho_ap:+.3f} p {p_ap:.4f}")
    print(f"  Spearman(steepness, dd)             {rho_st:+.3f} p {p_st:.4f}")
    # regime split: well-determined = high aperture_ratio (short lam)
    hi = np.array([r['aperture_ratio'] for r in rows]) >= np.median(
        [r['aperture_ratio'] for r in rows])
    print(f"  high-aperture (well-det): dd {dd[hi].mean():+.4f} "
          f"Dlin {dlin[hi].mean():+.4f} (neg {(dlin[hi]<0).sum()}/{hi.sum()}) "
          f"flip {sum(r['flip'] for r,h in zip(rows,hi) if h)}/{hi.sum()}")
    print(f"  low-aperture  (under-det): dd {dd[~hi].mean():+.4f} "
          f"Dlin {dlin[~hi].mean():+.4f}")

    json.dump({'rows': rows, 'summary': dict(
        n_cells=len(rows), confound_corr=float(np.corrcoef(zar, zst)[0, 1]),
        dd_aperture_beta=float(b[1]), dd_aperture_p=float(p[1]),
        dd_steep_beta=float(b[2]), dd_steep_p=float(p[2]),
        dd_interaction_beta=float(b[3]), dd_interaction_p=float(p[3]),
        dlin_aperture_beta=float(bl[1]), dlin_aperture_p=float(pl[1]),
        dlin_steep_beta=float(bl[2]), dlin_steep_p=float(pl[2]),
        rho_ap_dlin=float(rho_ap), rho_st_dd=float(rho_st))},
        open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
