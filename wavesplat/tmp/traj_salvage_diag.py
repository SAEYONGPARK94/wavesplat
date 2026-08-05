"""Trajectory salvage — diagnosis + one recovery attempt (coordinator-assigned,
2026-07-22). Reuses tmp/traj_step1b_relax.py's harness (make_sea/fly_path/
policies/relax_step achievability) directly -- imports, does not reimplement.

Background: Step 1b showed the oracle-wavenumber trajectory advantage
(CEM/learned beat stationary, Delta~+0.02, CI excl 0, n=24) DISAPPEARS and
REVERSES once wavenumbers are estimated by RELAX instead of given as oracle
(CEM ties, learned/greedy/straight-cross all WORSE than stationary). Obs
geometry: stationary ~20x20m patch, moving policies spread to ~100-120m.

Question (coordinator): is the RELAX failure on moving obs a (a) wavenumber-
ID quality problem, or (b) reconstruction-coverage problem (fixed 150x160m
render grid far from a sparse spread-out moving obs corridor)? And can ONE
recovery attempt fix it?

Design (frozen before running, single choice not a threshold scan):
  A. Wavenumber-ID diagnostic: for each policy, run relax_step on the SAME
     obs used in Step 1b, greedy-match estimated (kx,ky) to true sea['kx'],ky
     (nearest neighbour in k-space), report median |Delta k| and count match.
  B. Near/far split of the FIXED render grid: near = grid points within
     THRESH=20m of the nearest observation point (20m chosen BEFORE running,
     tied to stationary's own natural aperture scale from Step 1b's geom
     stats -- not tuned post-hoc). far = grid points beyond that.
     Report RELAX-achieved nRMSE separately for near vs far, both oracle and
     RELAX wavenumbers, all policies.
  C. Recovery attempt (the ONE salvage try): if near-field RELAX-achieved
     nRMSE recovers a Delta>=0.02 CEM/learned-vs-stationary advantage (the
     original Step-1 PASS bar) when scoring is restricted to near-obs points
     for EVERY policy under the SAME threshold (fair -- not policy-specific
     tuning), that's the recovery signal.

n=24 episodes (matches Step 1b for direct comparability). MEASURED/ARGUED
tags in the report, not here.

Usage: python3 tmp/traj_salvage_diag.py [--episodes 24]
Out:   results/direction1/traj_salvage_diag.json
"""
import os
import sys
import json
import math
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

from d74_closed_loop_trajectory import (make_sea, dom_dir, fly_path, T_OBS, DEPTH,
                                        SPEEDS_FULL, headings_straight, headings_greedy,
                                        headings_cem, headings_learned, train_policy, K)
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d18_relax_estimator import relax_step

GXN = np.linspace(-30, 120, 40)
GYN = np.linspace(-80, 80, 40)
POLICIES = ['stationary', 'straight-cross', 'greedy', 'CEM', 'learned']
THRESH_NEAR_M = 20.0                # frozen before running (see docstring)
OUT = 'results/direction1/traj_salvage_diag.json'


def gt_field(sea):
    Xg, Yg = np.meshgrid(GXN, GYN)
    psi = (sea['kx'][None, None] * Xg[..., None] + sea['ky'][None, None] * Yg[..., None]
           - sea['om'][None, None] * T_OBS + sea['phi'][None, None])
    return (sea['A'][None, None] * np.cos(psi)).sum(-1)


def relax_kx_ky(obs, sea):
    return relax_step([], [], obs[:, :3], obs[:, 3], DEPTH,
                      klim=3.5 * (math.hypot(sea['kx'].mean(), sea['ky'].mean()) or 0.6))


def wavenumber_id_error(kx_est, ky_est, sea):
    """Greedy nearest-neighbour match true->estimated in k-space (rad/m)."""
    kt = np.stack([sea['kx'], sea['ky']], axis=1)
    if len(kx_est) == 0:
        return dict(n_est=0, n_true=len(kt), med_dk=np.nan, frac_matched=0.0)
    ke = np.stack([kx_est, ky_est], axis=1)
    d = np.sqrt(((kt[:, None, :] - ke[None, :, :]) ** 2).sum(-1))    # (n_true, n_est)
    nn = d.min(axis=1)
    kp = float(np.hypot(sea['kx'], sea['ky']).mean())
    matched = nn < 0.3 * kp                     # "matched" = within 30% of typical |k|
    return dict(n_est=int(len(kx_est)), n_true=int(len(kt)),
               med_dk=float(np.median(nn)), med_dk_rel=float(np.median(nn) / max(kp, 1e-9)),
               frac_matched=float(matched.mean()))


def achieved_split(sea, speed, headings, oracle, thresh=THRESH_NEAR_M, seed=0):
    """Like Step1b's achieved(), but ALSO returns near/far-split nRMSE and
    the wavenumber-ID diagnostic (for oracle=False)."""
    x, y, t, z = fly_path(sea, speed, headings, gen_z=True, seed=seed)
    obs = np.column_stack([x, y, t, z])
    if len(obs) > 1500:
        obs = obs[np.random.default_rng(0).choice(len(obs), 1500, replace=False)]
    wid = None
    if oracle:
        kx, ky = sea['kx'], sea['ky']
    else:
        kx, ky = relax_kx_ky(obs, sea)
        wid = wavenumber_id_error(kx, ky, sea)
        if len(kx) == 0:
            return dict(near=np.nan, far=np.nan, full=np.nan, wid=wid)
    A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
    fld = render_wave_field(kx, ky, A, phi, DEPTH, GXN, GYN, T_OBS)
    gt = gt_field(sea)
    Hs = 4 * math.sqrt((sea['A'] ** 2 / 2).sum())
    Xg, Yg = np.meshgrid(GXN, GYN)
    d_obs = np.sqrt((Xg[..., None] - obs[None, None, :, 0]) ** 2
                    + (Yg[..., None] - obs[None, None, :, 1]) ** 2).min(axis=-1)
    near = d_obs <= thresh
    err2 = (fld - gt) ** 2
    rmse_near = float(np.sqrt(np.mean(err2[near])) / Hs) if near.any() else np.nan
    rmse_far = float(np.sqrt(np.mean(err2[~near])) / Hs) if (~near).any() else np.nan
    rmse_full = float(np.sqrt(np.mean(err2)) / Hs)
    return dict(near=rmse_near, far=rmse_far, full=rmse_full,
               frac_near=float(near.mean()), wid=wid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--train_eps', type=int, default=60)
    a = ap.parse_args()
    speed = SPEEDS_FULL[-1]
    t0 = time.time()
    print(f"training learned policy @ {speed} m/s...", flush=True)
    pol = train_policy(speed, a.train_eps)
    print(f"  policy trained [{time.time()-t0:.0f}s]", flush=True)

    def headings_for(p, sea):
        return {'stationary': np.full(K, dom_dir(sea)),
                'straight-cross': headings_straight(sea, 90),
                'greedy': headings_greedy(sea, speed),
                'CEM': headings_cem(sea, speed, seed=0),
                'learned': headings_learned(sea, speed, pol)}[p]

    rows = []
    for e in range(a.episodes):
        sea = make_sea(seed=1000 + e)
        row = {'ep': e}
        for p in POLICIES:
            spd = 0.0 if p == 'stationary' else speed
            h = headings_for(p, sea)
            row[f'{p}_oracle'] = achieved_split(sea, spd, h, oracle=True, seed=e)
            row[f'{p}_relax'] = achieved_split(sea, spd, h, oracle=False, seed=e)
        rows.append(row)
        if e % 4 == 0:
            print(f"  ep {e} [{time.time()-t0:.0f}s]: " +
                  " ".join(f"{p[:4]} near={row[f'{p}_relax']['near']:.2f} "
                          f"far={row[f'{p}_relax']['far']:.2f}" for p in POLICIES), flush=True)
        json.dump(dict(rows=rows, thresh=THRESH_NEAR_M), open(OUT, 'w'), indent=1, default=float)
    analyze(rows)


def boot_ci(d, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(d, len(d))) for _ in range(B)] if len(d) else [0]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyze(rows):
    print(f"\n=== A. Wavenumber-ID diagnostic (RELAX vs true, median over episodes) ===")
    for p in POLICIES:
        dk = np.array([r[f'{p}_relax']['wid']['med_dk_rel'] for r in rows
                       if r[f'{p}_relax']['wid'] is not None])
        nest = np.array([r[f'{p}_relax']['wid']['n_est'] for r in rows
                         if r[f'{p}_relax']['wid'] is not None])
        fm = np.array([r[f'{p}_relax']['wid']['frac_matched'] for r in rows
                       if r[f'{p}_relax']['wid'] is not None])
        print(f"  {p:16s} med|dk|/kp={np.nanmedian(dk):.3f}  n_est(med)={np.median(nest):.0f} "
              f"(true=8)  frac_matched(med)={np.median(fm):.2f}")

    print(f"\n=== B. Near/far split (RELAX-estimated wavenumbers, thresh={THRESH_NEAR_M}m) ===")
    out = {'A': {}, 'B': {}, 'C': {}}
    for p in POLICIES:
        near = np.array([r[f'{p}_relax']['near'] for r in rows])
        far = np.array([r[f'{p}_relax']['far'] for r in rows])
        full = np.array([r[f'{p}_relax']['full'] for r in rows])
        onear = np.array([r[f'{p}_oracle']['near'] for r in rows])
        ofar = np.array([r[f'{p}_oracle']['far'] for r in rows])
        print(f"  {p:16s} RELAX: near={np.nanmean(near):.3f} far={np.nanmean(far):.3f} "
              f"full={np.nanmean(full):.3f} | oracle: near={np.nanmean(onear):.3f} "
              f"far={np.nanmean(ofar):.3f}")
        out['B'][p] = dict(relax_near=float(np.nanmean(near)), relax_far=float(np.nanmean(far)),
                           relax_full=float(np.nanmean(full)), oracle_near=float(np.nanmean(onear)),
                           oracle_far=float(np.nanmean(ofar)))

    print(f"\n=== C. Recovery attempt: near-obs-restricted RELAX achieved vs stationary ===")
    stat_near = np.array([r['stationary_relax']['near'] for r in rows])
    for p in ['straight-cross', 'greedy', 'CEM', 'learned']:
        v = np.array([r[f'{p}_relax']['near'] for r in rows])
        m = np.isfinite(v) & np.isfinite(stat_near)
        d = (stat_near - v)[m]
        lo, hi = boot_ci(d)
        nbetter = int((d > 0).sum())
        pass_ = d.mean() >= 0.02 and lo > 0
        print(f"  {p:16s} Delta(near) = {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"{nbetter}/{m.sum()}  {'PASS' if pass_ else 'no'}")
        out['C'][p] = dict(delta=float(d.mean()), ci=[lo, hi], nbetter=nbetter, n=int(m.sum()),
                           pass_=bool(pass_))

    j = json.load(open(OUT)); j['analysis'] = out
    json.dump(j, open(OUT, 'w'), indent=1, default=float)
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
