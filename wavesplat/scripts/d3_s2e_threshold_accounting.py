"""F2 decisive experiment: the mask's information under THRESHOLD detection.

A3 (docs/F2_DETECTION_PHYSICS_2026-07-10.md) established that the real Falcon
declares a return when received power crosses a threshold (manual p.13 spec
language, p.105 confidence/multi-return), while our simulator's raydrop_uniform
is a different family: Bernoulli-with-gain p = eta*A(s). The two families put
different information in the mask:

    multiplicative:  d ln p/ds =            d ln A/ds     (rate-independent)
    threshold (M=1): d ln p/ds = (-ln p) *  d ln A/ds     (amplified when sparse)

Currency: mutual information I(m; s) by 1D quadrature — Fisher-at-s=0 was
self-correction #43; MI is exact for a bit and sampling-free. Accounting
follows tmp/redteam_mi.py exactly (same el-grid ray density H/(r^2+H^2), same
population-calibrated rate, same amplitude nats per return) so every number is
comparable with today's 2.5% / 17.15%.

Speckle model: X ~ Gamma(M, 1/M), p(s) = Q(M, M*T/A(s)) (regularized upper
incomplete gamma). M = 1 is fully developed speckle; pulse/aperture averaging
raises M and steepens the tail back toward the multiplicative limit. M is the
falsifier axis the manual leaves open.

================================ PRE-REGISTRATION ============================
Frozen before the first run.

  GATE (reopening confirmed): at deployed rate 1e-3, M = 1, on the S1/S2
      accounting sea (sigma_eta 1.0 m, sigma_s 0.0982, sigma0 0.2 m):
      I_mask / I_amp >= 10%.
  FALSIFIER (de-speckling): M* := smallest M in {1,2,4,8,16} with ratio < 5%.
      If the real receiver averages >= M* independent looks per point, the
      reopening is void and the multiplicative ~2.5% stands. Report M*.
  SEA-STATE CONDITIONALITY: the same table on the Door-1 reference sea
      (sigma_eta 0.30 m, sigma_s 0.0305, sigma0 0.05 m). Expected smaller;
      reported not gated. Every closure today was sea-state-conditional and
      the reopening must be stated with the same honesty.
  SANITY (quadrature vs Monte Carlo): at r = 100 m, rate 1e-2, M = 1: binned-
      empirical MI within 10% of quadrature, else the instrument is broken
      and NOTHING is reported.
  MODEL CHECK (raydrop_threshold.py behaves): on one synthetic frame the
      module returns exactly n_pts points, and its returns are more strongly
      slope-selected than raydrop_uniform's (mean s of returns, in sigma_s
      units, larger for the threshold model).

CEILING MAPPING (labelled rough, assumption: nats -> nRMSE by the same linear
map fable used, 0.70% of amplitude nats <-> +0.002 nRMSE at the E5 operating
point). Reported for scale only; the real prize measurement is the next
experiment (EP site head vs naive MMSE on threshold-law episodes).
=============================================================================
"""
import json
import os
import sys

import numpy as np
from scipy.special import gammaincc
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.raydrop_uniform import fresnel_reflectance, apply_dropout as drop_mult
from wavesim.raydrop_threshold import apply_dropout as drop_thr, received_power

OUT = 'results/direction3/s2e_threshold_accounting.json'
H_SENS, D_REF = 5.0, 125.0
SEAS = {                      # name: (sigma_eta [m], sigma_s, sigma0 [m])
    'S1S2_sea (Hs~4m)': (1.0, 0.0982, 0.2),
    'Door1_sea (Hs1.2)': (0.30, 0.0305, 0.05),
}
RATES = (1e-3, 1e-2, 1e-1)
MLOOKS = (1, 2, 4, 8, 16)

rr = np.linspace(15.0, 200.0, 400)
dens = H_SENS / (rr ** 2 + H_SENS ** 2)
dens /= np.trapezoid(dens, rr)
qs, qw = np.polynomial.hermite_e.hermegauss(301)


def A_of_s(s, r):
    g = np.arctan(H_SENS / r)
    th = np.clip(np.pi / 2 - g - s, 0.0, np.pi / 2 - 1e-9)
    return fresnel_reflectance(th) * np.cos(th) / ((r / D_REF) ** 2 + 1e-9)


def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def p_threshold(s, r, T, M):
    """P(A(s)*X > T), X ~ Gamma(M, 1/M): regularized upper incomplete gamma."""
    a = np.maximum(A_of_s(s, r), 1e-30)
    return np.clip(gammaincc(M, M * T / a), 1e-300, 1 - 1e-15)


def p_mult(s, r, eta):
    return np.clip(eta * A_of_s(s, r), 1e-300, 1 - 1e-15)


def account(pfun, sig_s):
    ss, ws = qs * sig_s, qw / np.sum(qw)
    Im = np.zeros_like(rr)
    pm = np.zeros_like(rr)
    for i, r in enumerate(rr):
        p = pfun(ss, r)
        pbar = np.sum(ws * p)
        Im[i] = h2(pbar) - np.sum(ws * h2(p))
        pm[i] = pbar
    rate = np.trapezoid(pm * dens, rr)
    mask = np.trapezoid(Im * dens, rr)
    return rate, mask


def calibrate(make_pfun, sig_s, rate, lo=1e-12, hi=1e12):
    """bisect an INCREASING knob so the achieved rate hits the target.

    v1 had the branch inverted -- achieved rate below target shrank the knob
    further, so every law was calibrated to rate ~0 and then normalized by the
    TARGET rate: both columns were garbage (mult printed 0.00% where the
    reference instrument gives 2.46%). Caught because two instruments
    disagreed (#46). The achieved rate is now returned and PRINTED so the
    calibration is auditable in the output itself.
    """
    for _ in range(90):
        mid = np.sqrt(lo * hi)
        r, _ = account(make_pfun(mid), sig_s)
        lo, hi = (mid, hi) if r < rate else (lo, mid)
    knob = np.sqrt(lo * hi)
    ach, mask = account(make_pfun(knob), sig_s)
    return knob, ach, mask


def mc_sanity(sig_s, M=1, r=100.0, rate=1e-2, n=4_000_000, seed=0):
    """binned empirical MI at one range vs the quadrature."""
    T = None
    lo, hi = 1e-12, 1e3
    ss, ws = qs * sig_s, qw / np.sum(qw)
    for _ in range(80):                                  # calibrate at this r only
        mid = np.sqrt(lo * hi)
        pbar = np.sum(ws * p_threshold(ss, r, mid, M))
        lo, hi = (mid, hi) if pbar > rate else (lo, mid)
    T = np.sqrt(lo * hi)
    quad = h2(np.sum(ws * p_threshold(ss, r, T, M))) - \
        np.sum(ws * h2(p_threshold(ss, r, T, M)))
    rng = np.random.default_rng(seed)
    s = rng.normal(0, sig_s, n)
    m = rng.random(n) < p_threshold(s, r, T, M)
    edges = np.quantile(s, np.linspace(0, 1, 41))
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, 39)
    pb = np.array([m[idx == b].mean() for b in range(40)])
    wb = np.array([(idx == b).mean() for b in range(40)])
    emp = h2(m.mean()) - np.sum(wb * h2(np.clip(pb, 1e-12, 1)))
    return quad, emp


def model_check():
    """raydrop_threshold vs raydrop_uniform, FIXED range band.

    v1 mixed all ranges; A ~ 1/r^2 varies 100x over 15..200 m, so the
    threshold top-k mostly selected NEAR points (where d ln A/ds ~ -0.8 is
    weak) and the comparison read range concentration, not slope selection
    (#46). At fixed range the confound is gone.
    """
    rng = np.random.default_rng(3)
    n = 40000
    r = rng.uniform(95.0, 105.0, n)                      # fixed band
    g = np.arctan(H_SENS / r)
    sig_s = 0.0982
    s = rng.normal(0, sig_s, n)                          # radial facet slope
    th = np.clip(np.pi / 2 - g - s, 0.0, np.pi / 2 - 1e-9)
    hm = np.ones(n, bool)
    hp = np.zeros((n, 3))
    mt = drop_thr(hm, hp, r, th, 600, rng, m_looks=1)
    mm = drop_mult(hm, hp, r, th, 600, rng)
    return (int(mt.sum()), float(s[mt].mean() / sig_s),
            int(mm.sum()), float(s[mm].mean() / sig_s))


def main():
    print(__doc__.split('=============================================================================')[0][-1500:])
    res = {'seas': {}}

    print("== sanity: quadrature vs Monte Carlo (r=100 m, rate 1e-2, M=1) ==")
    quad, emp = mc_sanity(0.0982)
    rel = abs(emp - quad) / quad
    print(f"   quad {quad:.4e}   MC {emp:.4e}   rel diff {rel:.1%}  "
          f"{'PASS' if rel < 0.10 else 'FAIL -- instrument broken, stop'}")
    res['sanity'] = {'quad': quad, 'mc': emp, 'rel': rel, 'pass': bool(rel < 0.10)}
    if rel >= 0.10:
        return 1

    print("\n== model check: raydrop_threshold.py vs raydrop_uniform ==")
    nt, st, nm, sm = model_check()
    ok = (nt == 600) and (st > sm)
    print(f"   threshold: n={nt} (want 600)  mean slope of returns {st:+.3f} sigma_s")
    print(f"   multiplicative: n={nm}        mean slope of returns {sm:+.3f} sigma_s")
    print(f"   {'PASS' if ok else 'FAIL'} (exact count & stronger slope selection)")
    res['model_check'] = {'n_thr': nt, 's_thr': st, 'n_mult': nm, 's_mult': sm,
                          'pass': bool(ok)}

    for sea, (sig_eta, sig_s, sig0) in SEAS.items():
        amp_per = 0.5 * np.log(1 + sig_eta ** 2 / sig0 ** 2)
        print(f"\n===== {sea}: sigma_eta {sig_eta} m, sigma_s {sig_s}, "
              f"sigma0 {sig0} m  (amp {amp_per:.3f} nats/return) =====")
        hdr = ''.join(f"{f'M={m}':>9}" for m in MLOOKS)
        print(f"{'rate':>7} {'mult':>9}{hdr}")
        tab = {}
        for rate in RATES:
            _, ach, mk = calibrate(lambda k: (lambda s_, r_: p_mult(s_, r_, k)),
                                   sig_s, rate)
            row = {'mult': mk / (ach * amp_per), 'mult_rate': ach}
            line = f"{rate:7.3g} {row['mult']:9.2%}"
            for M in MLOOKS:
                # knob is 1/T (increasing in rate)
                _, ach, mk = calibrate(lambda k: (lambda s_, r_:
                                       p_threshold(s_, r_, 1.0 / k, M)),
                                       sig_s, rate)
                row[f'M{M}'] = mk / (ach * amp_per)
                row[f'M{M}_rate'] = ach
                line += f"{row[f'M{M}']:>9.2%}"
            ach_all = [row['mult_rate']] + [row[f'M{m}_rate'] for m in MLOOKS]
            line += f"   [ach {min(ach_all):.2e}..{max(ach_all):.2e}]"
            print(line, flush=True)
            tab[f'{rate:g}'] = row
        res['seas'][sea] = tab

    s1 = res['seas']['S1S2_sea (Hs~4m)']['0.001']
    gate = s1['M1'] >= 0.10
    mstar = next((m for m in MLOOKS if s1[f'M{m}'] < 0.05), None)
    d1 = res['seas']['Door1_sea (Hs1.2)']['0.001']
    print("\n" + "=" * 74)
    print(f"GATE (deployed rate 1e-3, M=1, S1/S2 sea): ratio = {s1['M1']:.2%} "
          f">= 10%  ->  {'PASS: mask channel OPEN under threshold detection' if gate else 'FAIL: stays closed'}")
    print(f"de-speckling falsifier M* (ratio < 5%): "
          f"{('M* = %d' % mstar) if mstar else 'not reached by M=16'}")
    print(f"sea-state conditionality (Door-1 sea, Hs 1.2): M=1 ratio = "
          f"{d1['M1']:.2%}  (mult {d1['mult']:.2%})")
    ceil = 0.002 * (s1['M1'] / 0.0070)
    print(f"rough ceiling mapping (0.70% <-> +0.002): M=1 S1/S2 sea ~ "
          f"+{ceil:.3f} nRMSE  [assumption-laden, scale only]")
    res['gate'] = {'ratio_M1': s1['M1'], 'pass': bool(gate), 'mstar': mstar,
                   'door1_M1': d1['M1'], 'rough_ceiling': ceil}

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
