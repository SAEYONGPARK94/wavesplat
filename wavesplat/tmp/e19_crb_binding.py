"""E-19 (closing experiment #2) — bind the regime boundary to the CRB.

Elevates the aperture story from an ablation to theory: if the Delta(linear)
sign-flip boundary (aperture stops helping -> learning loses on correct physics)
coincides with the Cramer-Rao "well-determined" threshold (field-reconstruction
CRB drops below a target), then the regime map is THEORETICALLY located, not
just empirically partitioned.

Per E-18 sea (Tp, Hs): reconstruct the fan observation geometry (one linear
build + one 600-pt draw), take theta = the JONSWAP free-wave modes, and compute
the unbiased field-reconstruction nRMSE CRB (d1_crb) — a property of the
observation operator + the field, independent of the estimator. Then merge with
E-18's Delta(linear)/dd and test: does log(CRB) predict Delta(linear) and the
sign-flip, and does the flip boundary sit at a CRB threshold?

  python3 tmp/e19_crb_binding.py           # after e18 json exists
"""
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')

from e16_physics_operator import build_sea_linear                # noqa: E402
from e15_hos_multidraw import N_OBS                              # noqa: E402
from d3_f1_hos_door1 import sea_components, CX, CY, DEPTH        # noqa: E402
from d3_door1b_field_bias import R_MIN, R_MAX, FOV_H             # noqa: E402
from wavesim.raydrop_threshold import T_MANUAL_50               # noqa: E402
import d1_crb                                                    # noqa: E402
from d1_crb import fisher, field_design, field_nrmse_crb, DT, DEVICE  # noqa

NOISE = 0.05
E18 = 'results/direction3/e18_decorrelated_2factor.json'
OUT = 'results/direction3/e19_crb_binding.json'

TPS = [6.0, 7.0, 8.0, 9.0, 11.0, 13.0]
STEEPS = [0.015, 0.025, 0.035]


def _lam(Tp, depth=DEPTH, g=9.81):
    om = 2 * np.pi / Tp
    k = om ** 2 / g
    for _ in range(80):
        k = om ** 2 / (g * np.tanh(k * depth))
    return float(2 * np.pi / k)


def draw_obs(sea, rng):
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep],
                                  np.full(int(keep.sum()), sc['trel'])]))
        rr.append(sc['r'][keep])
    P, rr = np.vstack(P), np.concatenate(rr)
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-N_OBS // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    return P[np.concatenate(take)]


def crb_for_sea(Tp, Hs):
    """field-reconstruction nRMSE CRB from the fan obs geometry + JONSWAP modes."""
    sea = build_sea_linear(Tp, Hs, 0)
    P = draw_obs(sea, np.random.default_rng(700000))
    S = sea_components(0, Hs=Hs)
    theta = torch.tensor(np.stack([S['A'], S['kx'], S['ky'], S['ph']], 1),
                         dtype=DT, device=DEVICE)
    x = torch.tensor(CX + P[:, 0], dtype=DT, device=DEVICE)
    y = torch.tensor(CY + P[:, 1], dtype=DT, device=DEVICE)
    t = torch.tensor(P[:, 2], dtype=DT, device=DEVICE)
    # reconstruction grid = the fan region at t*=0
    rr = np.sqrt(np.random.default_rng(1).uniform((R_MIN / R_MAX) ** 2, 1, 400)) * R_MAX
    aa = np.random.default_rng(2).uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 400)
    gx = torch.tensor(CX + rr * np.cos(aa), dtype=DT, device=DEVICE)
    gy = torch.tensor(CY + rr * np.sin(aa), dtype=DT, device=DEVICE)
    F = fisher(theta, x, y, t, DEPTH, NOISE)
    # field_design expects a grid via meshgrid; build G directly from the samples
    Jg = d1_crb.jacobian(theta, gx, gy, torch.zeros_like(gx), DEPTH)
    G = (Jg.T @ Jg) / Jg.shape[0]
    crb, _ = field_nrmse_crb(F, G, Hs)
    return float(crb)


def main():
    seas = [(Tp, round(st * _lam(Tp), 3)) for Tp in TPS for st in STEEPS]
    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp)
        crb = crb_for_sea(Tp, Hs)
        rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp, Hs=Hs, lam=lam,
                         steepness=Hs / lam, aperture_ratio=R_MAX / lam,
                         crb_nrmse=crb))
        print(f"[Tp{Tp:g} Hs{Hs:g} lam{lam:.0f} ar{R_MAX/lam:.2f} "
              f"steep{Hs/lam:.3f}] CRB nRMSE {crb:.4f}", flush=True)
    json.dump(rows, open(OUT, 'w'), indent=2)

    # ---- merge with E-18 and test the CRB-boundary alignment ----
    if not os.path.exists(E18):
        print(f"\n(E-18 not found at {E18} — CRB saved; run merge later)")
        return 0
    e18 = json.load(open(E18))['rows']
    crb_map = {r['sea']: r['crb_nrmse'] for r in rows}
    dlin = {}, {}
    per = {}
    for r in e18:
        per.setdefault(r['sea'], []).append(r)
    sd, sc, sdd = [], [], []
    for sea, rs in per.items():
        if sea not in crb_map:
            continue
        dl = np.nanmean([x['d_lin'] for x in rs])
        if not np.isfinite(dl):
            continue
        sd.append(dl)
        sdd.append(np.nanmean([x['dd'] for x in rs]))
        sc.append(crb_map[sea])
    sd, sc, sdd = np.array(sd), np.array(sc), np.array(sdd)
    lsc = np.log(sc)
    print(f"\n{'='*66}\nE-19 CRB binding ({len(sc)} seas)\n{'='*66}")
    r1, p1 = stats.spearmanr(sc, sd)
    r2, p2 = stats.spearmanr(sc, sdd)
    print(f"  Spearman(CRB, Delta_lin) {r1:+.3f} p {p1:.4f}  "
          f"(predict: higher CRB -> more negative Delta_lin i.e. aperture helps)")
    print(f"  Spearman(CRB, dd)        {r2:+.3f} p {p2:.4f}")
    # sign-flip boundary vs CRB threshold: is there a CRB level separating
    # Delta_lin<0 (aperture helps) from Delta_lin>=0 (well-determined)?
    neg = sd < 0            # Delta_lin<0 = learning LOSES on correct physics =
    #                        the WELL-DETERMINED regime (aperture gives no help)
    auc = np.nan
    if neg.any() and (~neg).any():
        # AUC that HIGH CRB predicts the UNDER-DETERMINED regime (Delta_lin>=0,
        # where learning helps). 1.0 = the flip boundary IS a CRB threshold.
        auc = np.mean([1.0 if a > b else (0.5 if a == b else 0.0)
                       for a in sc[~neg] for b in sc[neg]])
        print(f"  well-determined (Delta_lin<0, learning loses): CRB "
              f"[{sc[neg].min():.3f},{sc[neg].max():.3f}] median {np.median(sc[neg]):.3f}")
        print(f"  under-determined (Delta_lin>=0, learning helps): CRB "
              f"[{sc[~neg].min():.3f},{sc[~neg].max():.3f}] median {np.median(sc[~neg]):.3f}")
        print(f"  CRB predicts the regime: AUC {auc:.3f} "
              f"(1.0 = flip boundary IS a CRB threshold)")
        # honest collinearity caveat: does CRB beat the raw aperture ratio?
        ar = np.array([[r['aperture_ratio'] for r in crb_rows
                        if r['sea'] == s][0] for s in
                       [s2 for s2 in per if s2 in crb_map
                        and np.isfinite(np.nanmean(
                            [x['d_lin'] for x in per[s2]]))]])
        r_ar, _ = stats.spearmanr(ar, sd)
        print(f"  (collinearity: Spearman(aperture_ratio, Delta_lin) {r_ar:+.3f}; "
              f"Spearman(CRB, aperture_ratio) {stats.spearmanr(sc, ar)[0]:+.3f} "
              f"-> CRB is a re-scaling of aperture, not an independent predictor)")
    json.dump({'crb_rows': rows, 'spearman_crb_dlin': [float(r1), float(p1)],
               'spearman_crb_dd': [float(r2), float(p2)], 'auc': float(auc)},
              open(OUT, 'w'), indent=2)
    print(f"\n-> {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
