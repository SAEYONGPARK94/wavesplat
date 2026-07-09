"""d3_band_selection.py — the band, diagnosed and then chosen.

Two things, following STORM_WALL_AND_UQ_GATE_2026-07-09:

PART 1 (diagnosis). That document claimed [추정] that LJ18g's oracle pathology
(handing the estimator observations everywhere made the in-fan field WORSE,
+0.235 with the base band, +0.042 with a wide one) comes from the bank's band
being placed by a kp_hint that misses the record's energy. The band is
[0.35, 2.5] * kp, so a wrong kp is a wrong band. Sweep kp, and for each value
measure BOTH

    the pathology      delta = nRMSE_in_fan(oracle) - nRMSE_in_fan(fan)
    the in-band energy fraction of the true field

If the pathology tracks the missing energy, the [추정] becomes a measurement.

PART 2 (prescription). A wide band helps a sea whose energy leaves the narrow
band, and HURTS when observations are sparse (LJ18g fan: 0.550 -> 0.609):
more bank freedom than the data can pin down. So the ruler length must be
chosen, not fixed — and chosen without the ground truth.

    hold out 25% of the OBSERVATIONS, fit each candidate band on the rest,
    predict the held-out points, keep the band with the smallest residual.

Nothing here reads the GT. Compared against every fixed band, and against the
oracle band (the fixed band that turned out best on the GT), which no
deployable rule may use.

Usage: python3 scripts/d3_band_selection.py [--part 1|2|all]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_nomp import mmse_bank_estimate
import d3_static_field as SF
import d3_static_limit as SL
import d3_storm_wall as SW

G = 9.81
OUT = 'results/direction3/band_selection.json'
BANDS = [(0.35, 2.5), (0.25, 3.5), (0.15, 5.0)]
KP_FACTORS = [0.4, 0.55, 0.7, 0.85, 1.0, 1.25, 1.6, 2.0]


def regions(Xq, Yq, ctr, hvec):
    inside, R = SF.fan_mask(Xq, Yq, ctr[0], ctr[1], hvec)
    return inside, (R > SF.R_FAN[1]) & ~inside


def nrmse(pred, gt, m, sg):
    mm = m & np.isfinite(pred) & np.isfinite(gt)
    return float(np.sqrt(np.mean((pred[mm] - gt[mm]) ** 2)) / sg)


def inband_energy(gt, dx, dy, kp, band):
    """Fraction of the field's spectral energy inside [lo*kp, hi*kp].

    The bank can only represent what falls in its annulus; this is the
    quantity a wrong kp destroys.
    """
    z = np.nan_to_num(gt - np.nanmean(gt))
    F = np.fft.fftshift(np.abs(np.fft.fft2(z)) ** 2)
    kxs = np.fft.fftshift(np.fft.fftfreq(z.shape[1], d=dx)) * 2 * np.pi
    kys = np.fft.fftshift(np.fft.fftfreq(z.shape[0], d=dy)) * 2 * np.pi
    KX, KY = np.meshgrid(kxs, kys)
    kk = np.hypot(KX, KY)
    tot = F[kk > 0].sum()
    sel = (kk > band[0] * kp) & (kk < band[1] * kp)
    return float(F[sel].sum() / max(tot, 1e-12))


def scene_real(rec):
    sc = SF.scene_real(rec)
    Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
    eta = sc['make_eta']()
    gt = eta(Xq.ravel(), Yq.ravel(), sc['t0']).reshape(Xq.shape)
    return sc, Xq, Yq, gt


def part1(rec='LJ18g'):
    sc, Xq, Yq, gt = scene_real(rec)
    Pq = np.column_stack([Xq.ravel(), Yq.ravel(),
                          np.full(Xq.size, sc['t0'])])
    inside, beyond = regions(Xq, Yq, sc['center'], sc['hvec'])
    sg = float(np.nanstd(gt))
    dx = float(sc['gx'][1] - sc['gx'][0])
    dy = float(sc['gy'][1] - sc['gy'][0])
    band = BANDS[0]
    kp0 = sc['kp']
    print(f"\n=== PART 1: {rec}  kp_hint sweep (band {band}, kp0={kp0:.4f}) ===")
    print(f"{'kp/kp0':>7} {'kp':>8} {'in-band E':>10} {'fan':>8} "
          f"{'oracle':>8} {'pathology':>10}")
    rows = []
    for f in KP_FACTORS:
        kp = kp0 * f
        e = inband_energy(gt, dx, dy, kp, band)
        out = {}
        for cfg in ('fan_noisy', 'all_dense'):
            rng = np.random.default_rng(11)
            P, Z, _ = SL.resample(sc, cfg, rng)
            fq = mmse_bank_estimate(P, Z, sc['depth'], Pq,
                                    sigma0=0.05 * sc['Hs'], kp_hint=kp,
                                    seed=sc['seed'], band=band)
            out[cfg] = nrmse(fq.reshape(Xq.shape), gt, inside, sg)
        path = out['all_dense'] - out['fan_noisy']
        rows.append({'kp_factor': f, 'kp': kp, 'inband_energy': e,
                     'fan': out['fan_noisy'], 'oracle': out['all_dense'],
                     'pathology': path})
        print(f"{f:>7.2f} {kp:>8.4f} {e:>10.3f} {out['fan_noisy']:>8.3f} "
              f"{out['all_dense']:>8.3f} {path:>+10.3f}", flush=True)
    e = np.array([r['inband_energy'] for r in rows])
    p = np.array([r['pathology'] for r in rows])
    r = float(np.corrcoef(e, p)[0, 1])
    print(f"\ncorr(in-band energy, pathology) = {r:+.3f}   "
          f"(negative = missing energy causes the pathology)")
    return {'record': rec, 'kp0': kp0, 'band': list(band), 'rows': rows,
            'corr_energy_pathology': r}


def choose_band(P, Z, depth, Hs, kp, seed, val_frac=0.25):
    """Pick the band by held-out OBSERVATIONS. The GT is never touched."""
    rng = np.random.default_rng(1234 + seed)
    n = len(Z)
    vi = rng.choice(n, max(16, int(val_frac * n)), replace=False)
    vm = np.zeros(n, bool); vm[vi] = True
    scores = []
    for band in BANDS:
        pv = mmse_bank_estimate(P[~vm], Z[~vm], depth, P[vm],
                                sigma0=0.05 * Hs, kp_hint=kp, seed=seed,
                                band=band)
        scores.append(float(np.sqrt(np.mean((pv - Z[vm]) ** 2))))
    j = int(np.argmin(scores))
    return BANDS[j], scores


def part2():
    print("\n=== PART 2: adaptive band (chosen on held-out observations) ===")
    cases = []
    for rec in ('YS17', 'LJ18g'):
        sc, Xq, Yq, gt = scene_real(rec)
        cases.append((f'real_{rec}', sc, Xq, Yq, gt))
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1] + 0.1, SF.DX)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1] + 0.1, SF.DY)
    Xs, Ys = np.meshgrid(gx, gy)
    for sea in ('storm', 'bimodal'):
        spec = SW.SEAS[sea]
        eta, kp, Hs = SW.make_sea(spec, 1000)
        gt = eta(Xs.ravel(), Ys.ravel(), SF.W_OBS).reshape(Xs.shape)
        sc = {'depth': spec['depth'], 'Hs': Hs, 'kp': kp, 't0': SF.W_OBS,
              'seed': 0, 'center': (0.0, 0.0), 'hvec': np.array([1.0, 0.0]),
              'eta': eta, 'synth': spec}
        cases.append((sea, sc, Xs, Ys, gt))

    res = {}
    hdr = (f"{'case':<12} {'obs':<10} {'fixed narrow':>13} {'fixed wide':>11} "
           f"{'ADAPTIVE':>10} {'picked':>12} {'oracle band':>12}")
    print(hdr); print('-' * len(hdr))
    for name, sc, Xq, Yq, gt in cases:
        Pq = np.column_stack([Xq.ravel(), Yq.ravel(),
                              np.full(Xq.size, sc['t0'])])
        inside, _ = regions(Xq, Yq, sc['center'], sc['hvec'])
        sg = float(np.nanstd(gt))
        for cfg in ('fan_noisy', 'all_dense'):
            rng = np.random.default_rng(11)
            if 'synth' in sc:
                P, Z = SW.observations(sc['eta'], cfg, rng, sc['t0'])
            else:
                P, Z, _ = SL.resample(sc, cfg, rng)
            fixed = {}
            for band in BANDS:
                fq = mmse_bank_estimate(P, Z, sc['depth'], Pq,
                                        sigma0=0.05 * sc['Hs'],
                                        kp_hint=sc['kp'], seed=sc['seed'],
                                        band=band)
                fixed[band] = nrmse(fq.reshape(Xq.shape), gt, inside, sg)
            band, scores = choose_band(P, Z, sc['depth'], sc['Hs'], sc['kp'],
                                       sc['seed'])
            fq = mmse_bank_estimate(P, Z, sc['depth'], Pq,
                                    sigma0=0.05 * sc['Hs'],
                                    kp_hint=sc['kp'], seed=sc['seed'],
                                    band=band)
            adap = nrmse(fq.reshape(Xq.shape), gt, inside, sg)
            ob = min(fixed, key=fixed.get)
            res[f'{name}/{cfg}'] = {
                'fixed': {str(b): v for b, v in fixed.items()},
                'adaptive': adap, 'picked': list(band),
                'oracle_band': list(ob), 'oracle_nrmse': fixed[ob],
                'val_scores': scores}
            print(f"{name:<12} {cfg:<10} {fixed[BANDS[0]]:>13.3f} "
                  f"{fixed[BANDS[-1]]:>11.3f} {adap:>10.3f} "
                  f"{str(band):>12} {str(ob):>12}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['1', '2', 'all'])
    a = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if a.part in ('1', 'all'):
        res['part1_LJ18g'] = part1('LJ18g')
        res['part1_YS17'] = part1('YS17')
        json.dump(res, open(OUT, 'w'), indent=1)
    if a.part in ('2', 'all'):
        res['part2'] = part2()
        json.dump(res, open(OUT, 'w'), indent=1)
    print(f"\n-> {OUT}")


if __name__ == '__main__':
    sys.exit(main())
