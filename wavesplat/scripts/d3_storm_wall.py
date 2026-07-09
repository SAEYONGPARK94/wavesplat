"""d3_storm_wall.py — A+B: is the storm wall the model class, the aperture, or
just dirty ground truth?

C1' (STATIC_LIMIT_information_vs_model) claimed the storm record LJ18g hits a
MODEL-CLASS wall: handing the estimator observations over the whole domain
made the in-fan field WORSE (0.524 -> 0.749), the signature of a misspecified
model forced to compromise. But that "oracle" fed it a stereo GT carrying
sigma ~ 0.55 m of reconstruction noise (probe_gt_w_noise). "The ruler is too
short" was never separated from "the data are filthy".

Three confounds were riding together in LJ18g:
  (i)   GT noise         sigma = 0.55 m
  (ii)  aperture ratio   lambda_p = 201 m vs a 15-200 m fan  (~1 wavelength)
  (iii) band coverage    broadband / bimodal sea vs a single-band bank
                         [0.35 kp, 2.5 kp]

In synthesis each can be switched on alone.

  seas:   wind      Hs 1.9  Tp 6.0             (control; C1' linear case)
          storm     Hs 10.9 Tp 11.35           (aperture ratio only)
          bimodal   storm + a second peak at 90 deg, Tp 6.5   (band coverage)

  configs: fan_noisy   fan, sigma 0.02 (Falcon spec)
           fan_clean   fan, sigma 0
           fan_dense   fan, sigma 0, 4x points
           all_dense   whole domain, sigma 0            <- the true oracle
           all_dirty   whole domain, sigma 0.55         <- emulates LJ18g's GT

Reading:
  all_dense degrades in-fan  -> model class / aperture. The wall is real.
  only all_dirty degrades    -> C1's storm verdict was a GT-noise artifact
                                and must be retracted.

B (same script, --wide): rerun the offending sea with a widened bank
(band 0.15-5.0 kp) and with dual_band. If the degradation disappears, the
wall is band coverage — and the fix is bank design, not learning.

Usage: python3 scripts/d3_storm_wall.py [--wide]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate
import d3_static_field as SF

G = 9.81
OUT = 'results/direction3/storm_wall.json'
GT_NOISE = 0.55          # measured stereo GT noise of LJ18g
N_DENSE = 2400

SEAS = {
    'wind':    dict(peaks=[(1.9, 6.0, 0.0)], spread=30.0, depth=50.0),
    'storm':   dict(peaks=[(10.9, 11.35, 0.0)], spread=37.0, depth=45.0),
    'bimodal': dict(peaks=[(9.5, 11.35, 0.0), (5.0, 6.5, 90.0)],
                    spread=37.0, depth=45.0),
}
CONFIGS = ['fan_noisy', 'fan_clean', 'fan_dense', 'all_dense', 'all_dirty']


def make_sea(spec, seed):
    kx, ky, A, ph = [], [], [], []
    for Hs, Tp, ddir in spec['peaks']:
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=150,
                               depth=spec['depth'], direction_deg=180.0 + ddir,
                               direction_spread_deg=spec['spread'],
                               seed=seed + int(ddir))
        kx += [q.kx for q in c]; ky += [q.ky for q in c]
        A += [q.A for q in c]; ph += [q.phi for q in c]
    kx, ky = np.array(kx), np.array(ky)
    A, ph = np.array(A), np.array(ph)
    om = omega_of(kx, ky, spec['depth'])
    # dominant peak wavenumber (the hint the estimator is given)
    kp = (2 * np.pi / spec['peaks'][0][1]) ** 2 / G
    Hs_tot = 4.0 * np.sqrt(np.sum(A ** 2) / 2)

    def eta(x, y, t):
        psi = (kx[None] * np.atleast_1d(x)[:, None]
               + ky[None] * np.atleast_1d(y)[:, None]
               - om[None] * t + ph[None])
        return (A[None] * np.cos(psi)).sum(1)
    return eta, kp, float(Hs_tot)


def observations(eta, cfg, rng, t0):
    noise = {'fan_noisy': 0.02, 'fan_clean': 0.0, 'fan_dense': 0.0,
             'all_dense': 0.0, 'all_dirty': GT_NOISE}[cfg]
    n_target = N_DENSE if cfg in ('fan_dense', 'all_dense', 'all_dirty') \
        else SF.M_CAP
    scan_ts = np.arange(t0 - SF.W_OBS, t0 + 1e-9, SF.SCAN_DT)
    per = int(np.ceil(n_target / len(scan_ts)))
    P, Z = [], []
    for t_s in scan_ts:
        if cfg.startswith('all'):
            xs = rng.uniform(*SF.DOM_X, per)
            ys = rng.uniform(*SF.DOM_Y, per)
        else:
            rr = rng.uniform(*SF.R_FAN, per)
            aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, per)
            xs, ys = rr * np.cos(aa), rr * np.sin(aa)
            k = ((xs >= SF.DOM_X[0]) & (xs <= SF.DOM_X[1]) &
                 (ys >= SF.DOM_Y[0]) & (ys <= SF.DOM_Y[1]))
            xs, ys = xs[k], ys[k]
            if len(xs) == 0:
                continue
        z = eta(xs, ys, t_s) + rng.normal(0, noise, len(xs)) if noise \
            else eta(xs, ys, t_s)
        P.append(np.column_stack([xs, ys, np.full(len(xs), t_s)]))
        Z.append(z)
    P, Z = np.vstack(P), np.concatenate(Z)
    if len(Z) > n_target:
        ii = rng.choice(len(Z), n_target, replace=False)
        P, Z = P[ii], Z[ii]
    return P, Z


def score(gt, pred, Xq, Yq):
    inside, R = SF.fan_mask(Xq, Yq, 0.0, 0.0, np.array([1.0, 0.0]))
    beyond = (R > SF.R_FAN[1]) & ~inside
    blind = ~inside & ~beyond
    sg = float(np.std(gt))

    def nr(m):
        return float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)) / sg)
    return {'in_fan': nr(inside), 'beyond_range': nr(beyond),
            'blind_near': nr(blind)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide', action='store_true',
                    help='B: widen the bank band and enable dual_band')
    ap.add_argument('--seeds', type=int, default=5,
                    help='replicate: a single scene cannot support retracting '
                         'a published verdict')
    a = ap.parse_args()
    band = (0.15, 5.0) if a.wide else (0.35, 2.5)
    key = 'wide' if a.wide else 'base'

    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1] + 0.1, SF.DX)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1] + 0.1, SF.DY)
    Xq, Yq = np.meshgrid(gx, gy)
    t0 = SF.W_OBS

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault(key, {})
    for sea, spec in SEAS.items():
        lam = 2 * np.pi / ((2 * np.pi / spec['peaks'][0][1]) ** 2 / G)
        print(f"\n[{sea}] lam_p {lam:.0f} m  "
              f"fan/lam = {(SF.R_FAN[1]-SF.R_FAN[0])/lam:.2f}  band {band}  "
              f"seeds {a.seeds}", flush=True)
        acc = {c: {'in_fan': [], 'blind_near': [], 'beyond_range': []}
               for c in CONFIGS}
        Hs_l = []
        Pq = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0)])
        for sd in range(a.seeds):
            eta, kp, Hs = make_sea(spec, 1000 + 37 * sd)
            Hs_l.append(Hs)
            gt = eta(Xq.ravel(), Yq.ravel(), t0).reshape(Xq.shape)
            for cfg in CONFIGS:
                rng = np.random.default_rng(11 + sd)
                P, Z = observations(eta, cfg, rng, t0)
                fq = mmse_bank_estimate(P, Z, spec['depth'], Pq,
                                        sigma0=0.05 * Hs, kp_hint=kp,
                                        seed=sd, band=band,
                                        dual_band=bool(a.wide))
                s = score(gt, fq.reshape(Xq.shape), Xq, Yq)
                for k2, v in s.items():
                    acc[cfg][k2].append(v)
        ent = {'Hs': float(np.mean(Hs_l)), 'lambda_p': lam,
               'depth': spec['depth'], 'seeds': a.seeds,
               'fan_over_lambda': (SF.R_FAN[1] - SF.R_FAN[0]) / lam}
        for cfg in CONFIGS:
            ent[cfg] = {k2: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                        for k2, v in acc[cfg].items()}
            f = lambda k2: (f"{ent[cfg][k2]['mean']:.3f}"
                            f"±{ent[cfg][k2]['std']:.3f}")
            print(f"  {cfg:<11} in-fan {f('in_fan')}  "
                  f"blind {f('blind_near')}  beyond {f('beyond_range')}",
                  flush=True)
        # paired per-seed change: oracle vs baseline, in-fan
        d_clean = np.array(acc['all_dense']['in_fan']) - np.array(acc['fan_noisy']['in_fan'])
        d_dirty = np.array(acc['all_dirty']['in_fan']) - np.array(acc['fan_noisy']['in_fan'])
        ent['delta_in_fan'] = {
            'clean_oracle': {'mean': float(d_clean.mean()),
                             'std': float(d_clean.std())},
            'dirty_oracle': {'mean': float(d_dirty.mean()),
                             'std': float(d_dirty.std())},
            'note': 'positive = oracle observations made in-fan WORSE'}
        print(f"  -> in-fan delta (oracle - baseline): clean "
              f"{d_clean.mean():+.3f}±{d_clean.std():.3f}   dirty "
              f"{d_dirty.mean():+.3f}±{d_dirty.std():.3f}   "
              f"(LJ18g real: +0.225)", flush=True)
        res[key][sea] = ent
        json.dump(res, open(OUT, 'w'), indent=1)

    print(f"\n-> {OUT}")


if __name__ == '__main__':
    sys.exit(main())
