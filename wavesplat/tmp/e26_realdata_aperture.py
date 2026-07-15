"""E-26 — REAL-DATA leg of the aperture law (the examiner's "sim-only" hole).

All of E-16..E-25 are simulation. This closes the loop on the ONE claim the
examiner can most easily attack: that the learned-prior advantage tracks
observation under-determination ONLY because it was measured on synthetic seas.

  Sim result (E-18/E-19):  sea-level Spearman(aperture_ratio, Delta_lin) ~ -0.9
      (more wavelengths in the fan -> better determined -> learning loses).

  Real question: does the SAME sign hold on the 5 Bergamasco stereo "fan"
      records, where Delta = corr(net) - corr(hand-MMSE) is the *sparse-kp*
      encounter-forecast gap already computed in E-12g?

SCOPE (state honestly, do not overclaim):
  * "real" = real ocean waves + real sparse elevation geometry (stereo surface
    pushed through the Falcon-spec fan operator). NOT a real LiDAR. The
    specular-vs-diffuse radiometry category error is unaddressed here.
  * k = 5 records. This CANNOT establish a correlation; it can only test whether
    the SIGN is consistent with simulation and how fragile that sign is.

DESIGN = RE-ANALYSIS (no new compute for the headline). Both axes already exist:
  Delta per window : results/direction3/e12g_full_sensor.json  ('d_sensor',
                     the sparse-kp net-minus-hand gap; audit 6.1 fix baked in).
  aperture ratio   : tmp/record_wavelength.json  ('aperture_ratio' =
                     r_fan / lam_temporal). lam_temporal is the TRUE
                     dispersion/time-spectrum wavelength -- NOT the patch-floored
                     spatial FFT the audit (sec.1) flagged for the BS records.

Optional Stage B (--with_crb, a light ~seconds torch run, NOT run by --check):
  a geometry-only field-reconstruction CRB per record (JONSWAP modes at the
  record's own Tp/Hs/depth on a synthetic fan) + Spearman(CRB, Delta). E-19
  already showed CRB is nearly a monotone re-scaling of aperture, so this is a
  redundancy check, reported WITH the CRB-aperture collinearity exposed.

PRE-REGISTERED verdict (fixed before reading the numbers):
  FAIL     rho_between(aperture, Delta) >= 0            (wrong sign vs sim)
  DEMOTED  rho_between < 0 BUT fragile: some single-record drop flips rho>0,
           OR the pooled 60-window rho >= 0
  PASS     rho_between < 0 AND pooled rho < 0 AND no single-record drop flips
           the sign  (sign-consistent AND not a one-record artifact)
  In every case: report the exact n=5 permutation p, LORO, and the leverage
  record. With k=5 no p can be "significant"; the honest output is the sign +
  its fragility, not a star.

  python3 tmp/e26_realdata_aperture.py --check        # Stage A only (headline)
  python3 tmp/e26_realdata_aperture.py --with_crb     # + Stage B CRB axis
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E12G = os.path.join(ROOT, 'results/direction3/e12g_full_sensor.json')
WAVELEN = os.path.join(ROOT, 'tmp/record_wavelength.json')
OUT = os.path.join(ROOT, 'results/direction3/e26_realdata_aperture.json')

RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
# fan inner/outer radius [m] per record (from d3_falcon_real_operator.RECORDS)
R_FAN = {'YS17': (20., 100.), 'AA15a': (8., 40.), 'AA14': (8., 40.),
         'AA15b': (8., 40.), 'LJ18g': (20., 100.)}
FOV_H_DEG = 60.0
TWO_PI = 2 * np.pi
G = 9.81


# ----------------------------------------------------------------------------
# small stats helpers (exact where n is tiny; block-bootstrap where clustered)
# ----------------------------------------------------------------------------
def spearman(x, y):
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def exact_perm_p(x, y, alternative='less'):
    """Exact permutation p for Spearman rho at tiny n (n<=8 -> <=40320 perms).
    alternative='less' tests the pre-registered rho<0 (aperture helps learning).
    Returns (rho_obs, p_two_sided, p_directional)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho_obs, _ = spearman(x, y)
    perms = list(itertools.permutations(range(len(y))))
    rhos = np.array([spearman(x, y[list(p)])[0] for p in perms])
    p_two = float(np.mean(np.abs(rhos) >= abs(rho_obs) - 1e-12))
    if alternative == 'less':
        p_dir = float(np.mean(rhos <= rho_obs + 1e-12))
    else:
        p_dir = float(np.mean(rhos >= rho_obs - 1e-12))
    return float(rho_obs), p_two, p_dir


def cr1_se(d):
    """plain SE of a per-record mean (windows within a record are NOT indep;
    this is the within-record SE used only for display, not for a CI claim)."""
    d = np.asarray(d, float)
    return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan


def block_bootstrap_spearman(x, y, groups, nboot=5000, seed=0):
    """Cluster-respecting bootstrap of the pooled Spearman: resample the
    RECORDS with replacement, keep each drawn record's full window block."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    labs = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in labs}
    out = []
    for _ in range(nboot):
        pick = rng.choice(labs, len(labs), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(x[ii])) < 3:
            continue
        out.append(spearman(x[ii], y[ii])[0])
    out = np.array(out)
    return (float(np.nanpercentile(out, 2.5)),
            float(np.nanpercentile(out, 97.5)),
            float(np.mean(out < 0)))


# ----------------------------------------------------------------------------
# Stage A -- pure re-analysis (numpy/scipy only, < 2 s, no torch, no GPU)
# ----------------------------------------------------------------------------
def stage_a(verbose=True):
    rows = json.load(open(E12G))['rows']
    wl = {r['name']: r for r in json.load(open(WAVELEN))}

    # -- per-window records
    W = []
    for r in rows:
        nm = r['record']
        rf = R_FAN[nm][1]
        # per-window under-determination via the SENSOR's own peak wavenumber
        # (sensor-realizable, but noisy: beamform on 600 pts can grab a spurious
        #  peak). aperture_win = r_fan / lambda_win = r_fan * kp / (2 pi).
        ap_sensor = rf * r['kp_sensor'] / TWO_PI
        ap_oracle = rf * r['kp_oracle'] / TWO_PI      # clean but dense-GT (oracle)
        W.append(dict(record=nm, d=r['d_sensor'],
                      ap_sensor=ap_sensor, ap_oracle=ap_oracle,
                      kp_ratio=r['kp_ratio']))          # = kp_oracle / kp_sensor

    # -- per-record aggregates (headline axis = TRUE time-spectrum aperture)
    per = []
    for nm in RECS:
        ds = np.array([w['d'] for w in W if w['record'] == nm])
        per.append(dict(
            record=nm,
            aperture_true=float(wl[nm]['aperture_ratio']),
            lam_true=float(wl[nm]['lam_temporal']),
            depth=float(wl[nm]['depth']), Hs=float(wl[nm]['Hs']),
            Tp=float(wl[nm]['Tp']), kd=float(wl[nm]['kd']),
            n_win=int(len(ds)),
            d_mean=float(ds.mean()), d_median=float(np.median(ds)),
            d_se=cr1_se(ds)))
    per.sort(key=lambda r: r['aperture_true'])

    ap = np.array([r['aperture_true'] for r in per])
    dm = np.array([r['d_mean'] for r in per])
    dmed = np.array([r['d_median'] for r in per])

    # -- BETWEEN-record Spearman (n=5), exact permutation p, directional (rho<0)
    rho_mean, p2_mean, pdir_mean = exact_perm_p(ap, dm, 'less')
    rho_med, p2_med, pdir_med = exact_perm_p(ap, dmed, 'less')

    # -- leave-one-record-out on the between-record Spearman (mean axis)
    loro = []
    for drop in RECS:
        keep = [r for r in per if r['record'] != drop]
        rr, _ = spearman([r['aperture_true'] for r in keep],
                         [r['d_mean'] for r in keep])
        loro.append(dict(drop=drop, rho=float(rr), flips=bool(rr > 0)))
    flip_recs = [l['drop'] for l in loro if l['flips']]

    # -- 4 non-storm records alone (removes the single storm leverage point)
    #    to expose whether the sign lives entirely in that one record.

    # -- POOLED 60-window Spearman (aperture_win vs d), both lambda choices,
    #    with record-cluster block bootstrap.
    g = np.array([w['record'] for w in W])
    d_all = np.array([w['d'] for w in W])
    aps_all = np.array([w['ap_sensor'] for w in W])
    apo_all = np.array([w['ap_oracle'] for w in W])
    rho_pool_s, p_pool_s = spearman(aps_all, d_all)
    rho_pool_o, p_pool_o = spearman(apo_all, d_all)
    ci_lo_s, ci_hi_s, pneg_s = block_bootstrap_spearman(aps_all, d_all, g)
    ci_lo_o, ci_hi_o, pneg_o = block_bootstrap_spearman(apo_all, d_all, g)

    # -- WITHIN-record Spearman per record (12 windows each), sensor axis;
    #    sign test across records.
    within = []
    for nm in RECS:
        idx = [i for i, w in enumerate(W) if w['record'] == nm]
        rr, _ = spearman(aps_all[idx], d_all[idx])
        within.append(dict(record=nm, rho=float(rr)))
    n_within_neg = int(sum(w['rho'] < 0 for w in within))

    # -- kp_sensor contamination diagnostic (the "floored/wrong-lambda" analog):
    #    fraction of windows where the sparse kp is >2x off the dense-GT kp.
    kpr = np.array([w['kp_ratio'] for w in W])       # oracle/sensor
    contaminated = float(np.mean((kpr < 0.5) | (kpr > 2.0)))

    # -- PRE-REGISTERED VERDICT ------------------------------------------------
    sign_ok = rho_mean < 0
    pooled_ok = rho_pool_s < 0
    leverage = len(flip_recs) > 0
    if not sign_ok:
        verdict = 'FAIL - wrong sign vs simulation (rho_between >= 0)'
    elif leverage or not pooled_ok:
        verdict = ('DEMOTED - sign consistent with sim but FRAGILE '
                   f'(pooled_rho{"<0" if pooled_ok else">=0"}; '
                   f'leverage record(s) {flip_recs or "none"})')
    else:
        verdict = 'PASS - sign consistent AND robust to single-record drop'

    res = dict(
        design='re-analysis (Stage A): e12g d_sensor x record_wavelength '
               'aperture_ratio; no new compute',
        per_record=per,
        between_record=dict(
            axis='aperture_true = r_fan / lam_temporal (true time-spectrum)',
            rho_mean=rho_mean, p_two_mean=p2_mean, p_dir_mean=pdir_mean,
            rho_median=rho_med, p_two_median=p2_med, p_dir_median=pdir_med,
            loro=loro, flip_records=flip_recs),
        pooled_window=dict(
            n=len(W),
            sensor_axis=dict(rho=rho_pool_s, ci95=[ci_lo_s, ci_hi_s],
                             p_neg_boot=pneg_s),
            oracle_axis=dict(rho=rho_pool_o, ci95=[ci_lo_o, ci_hi_o],
                             p_neg_boot=pneg_o),
            note='sensor axis = kp from 600 sparse obs (realizable, noisy); '
                 'oracle axis = kp from dense GT FFT (clean, not realizable)'),
        within_record=dict(per=within, n_neg_of_5=n_within_neg),
        kp_sensor_contaminated_frac=contaminated,
        verdict=verdict)

    if verbose:
        _print_a(res)
    return res


def _print_a(res):
    print('=' * 78)
    print('E-26 STAGE A  real-data aperture leg (re-analysis, k=5 records)')
    print('=' * 78)
    print(f"{'record':<8}{'aperture':>10}{'lam_T[m]':>10}{'depth':>7}"
          f"{'Hs':>7}{'n':>4}{'d_mean':>9}{'d_med':>9}{'+-se':>8}")
    for r in res['per_record']:
        print(f"{r['record']:<8}{r['aperture_true']:>10.3f}{r['lam_true']:>10.0f}"
              f"{r['depth']:>7.0f}{r['Hs']:>7.2f}{r['n_win']:>4}"
              f"{r['d_mean']:>+9.4f}{r['d_median']:>+9.4f}{r['d_se']:>8.4f}")
    b = res['between_record']
    print(f"\nBETWEEN-record Spearman(aperture_true, d)  [predict rho<0]")
    print(f"  mean-d  : rho {b['rho_mean']:+.3f}  exact p2 {b['p_two_mean']:.3f}"
          f"  directional(rho<0) p {b['p_dir_mean']:.3f}")
    print(f"  median-d: rho {b['rho_median']:+.3f}  exact p2 "
          f"{b['p_two_median']:.3f}  directional p {b['p_dir_median']:.3f}")
    print(f"  LORO (drop one record, refit rho on the other 4):")
    for l in b['loro']:
        print(f"     drop {l['drop']:<7} rho {l['rho']:+.3f}"
              f"{'   <-- SIGN FLIPS (leverage)' if l['flips'] else ''}")
    pw = res['pooled_window']
    print(f"\nPOOLED {pw['n']}-window Spearman(aperture_win, d)  [predict rho<0]")
    s, o = pw['sensor_axis'], pw['oracle_axis']
    print(f"  sensor-kp axis: rho {s['rho']:+.3f}  boot95 "
          f"[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]  P(rho<0) {s['p_neg_boot']:.2f}")
    print(f"  oracle-kp axis: rho {o['rho']:+.3f}  boot95 "
          f"[{o['ci95'][0]:+.3f},{o['ci95'][1]:+.3f}]  P(rho<0) {o['p_neg_boot']:.2f}")
    w = res['within_record']
    print(f"\nWITHIN-record Spearman (sensor axis): "
          f"{w['n_neg_of_5']}/5 records negative")
    for x in w['per']:
        print(f"     {x['record']:<7} rho {x['rho']:+.3f}")
    print(f"\nkp_sensor contaminated (|kp_oracle/kp_sensor| off by >2x): "
          f"{res['kp_sensor_contaminated_frac']*100:.0f}% of windows")
    print(f"\n*** E-26 VERDICT: {res['verdict']}")


# ----------------------------------------------------------------------------
# Stage B -- per-record geometry-only field-reconstruction CRB (light, opt-in)
# ----------------------------------------------------------------------------
def _jonswap_modes(Tp, Hs, depth, n_f=60, n_dir=7, spread_s=8.0,
                   mean_dir_deg=0.0, gamma=3.3, seed=0):
    """Discrete free-wave modes from a JONSWAP x cos^2s directional spectrum at
    the record's OWN depth (dispersion solved with that depth). Returns theta
    columns [A, kx, ky, phase]. A geometry proxy for modal content -- NOT the
    record's true modes (stated as a caveat)."""
    rng = np.random.default_rng(seed)
    fp = 1.0 / Tp
    f = np.linspace(0.4 * fp, 3.0 * fp, n_f)
    df = f[1] - f[0]
    sig = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2 * sig ** 2 * fp ** 2))
    S = (f ** -5) * np.exp(-1.25 * (fp / f) ** 4) * gamma ** r
    S *= (Hs / (4 * np.sqrt(np.trapz(S, f)))) ** 2      # scale to Hs
    th = np.deg2rad(mean_dir_deg) + np.deg2rad(
        np.linspace(-60, 60, n_dir))
    D = np.cos((th - np.deg2rad(mean_dir_deg)) / 2) ** (2 * spread_s)
    D /= np.trapz(D, th)
    dth = th[1] - th[0]
    A, KX, KY, PH = [], [], [], []
    for fi, Si in zip(f, S):
        om = TWO_PI * fi
        k = om ** 2 / G
        for _ in range(80):
            k = om ** 2 / (G * np.tanh(k * depth))
        for tj, Dj in zip(th, D):
            var = Si * df * Dj * dth
            if var <= 0:
                continue
            A.append(np.sqrt(2 * var))
            KX.append(k * np.cos(tj))
            KY.append(k * np.sin(tj))
            PH.append(rng.uniform(0, TWO_PI))
    return (np.array(A), np.array(KX), np.array(KY), np.array(PH))


def _fan_points(r0, r1, fov_deg, n, seed=1):
    """n points uniformly in the +-fov annulus (single snapshot, t=0). Uniform
    in area: r = sqrt(U(r0^2,r1^2))."""
    rng = np.random.default_rng(seed)
    rr = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    aa = rng.uniform(-np.deg2rad(fov_deg), np.deg2rad(fov_deg), n)
    return rr * np.cos(aa), rr * np.sin(aa)


def stage_b(per_record, n_obs=600, n_grid=400, noise=0.05, verbose=True):
    """field-reconstruction CRB index per record from its fan geometry + depth.
    Single-snapshot SPATIAL CRB (temporal info absent) -- a relative index only,
    comparable across records. Reuses d1_crb (torch)."""
    import torch                                          # lazy, opt-in
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import d1_crb
    from d1_crb import fisher, field_nrmse_crb, DT, DEVICE

    crbs = []
    for r in per_record:
        nm = r['record']
        r0, r1 = R_FAN[nm]
        A, kx, ky, ph = _jonswap_modes(r['Tp'], r['Hs'], r['depth'])
        theta = torch.tensor(np.stack([A, kx, ky, ph], 1), dtype=DT,
                             device=DEVICE)
        ox, oy = _fan_points(r0, r1, FOV_H_DEG, n_obs, seed=1)
        gx_, gy_ = _fan_points(r0, r1, FOV_H_DEG, n_grid, seed=2)
        x = torch.tensor(ox, dtype=DT, device=DEVICE)
        y = torch.tensor(oy, dtype=DT, device=DEVICE)
        t = torch.zeros_like(x)
        gx = torch.tensor(gx_, dtype=DT, device=DEVICE)
        gy = torch.tensor(gy_, dtype=DT, device=DEVICE)
        F = fisher(theta, x, y, t, r['depth'], noise)
        Jg = d1_crb.jacobian(theta, gx, gy, torch.zeros_like(gx), r['depth'])
        Gm = (Jg.T @ Jg) / Jg.shape[0]
        crb, _ = field_nrmse_crb(F, Gm, r['Hs'])
        crbs.append(dict(record=nm, crb_nrmse=float(crb),
                         aperture_true=r['aperture_true'], d_mean=r['d_mean']))

    crbs_sorted = sorted(crbs, key=lambda z: z['aperture_true'])
    cvec = np.array([z['crb_nrmse'] for z in crbs_sorted])
    dvec = np.array([z['d_mean'] for z in crbs_sorted])
    avec = np.array([z['aperture_true'] for z in crbs_sorted])
    rho_cd, p2_cd, pdir_cd = exact_perm_p(cvec, dvec, 'greater')  # CRB up->d up
    rho_ca, _ = spearman(cvec, avec)                              # collinearity
    res = dict(per_record=crbs, rho_crb_d=rho_cd, p_two_crb_d=p2_cd,
               p_dir_crb_d=pdir_cd, rho_crb_aperture=rho_ca)
    if verbose:
        print('\n' + '=' * 78)
        print('E-26 STAGE B  geometry-only field-reconstruction CRB (opt-in)')
        print('=' * 78)
        for z in crbs_sorted:
            print(f"  {z['record']:<7} aperture {z['aperture_true']:.3f}"
                  f"  CRB {z['crb_nrmse']:.4f}  d_mean {z['d_mean']:+.4f}")
        print(f"\n  Spearman(CRB, d)        rho {rho_cd:+.3f}  exact p2 {p2_cd:.3f}"
              f"  directional(rho>0) p {pdir_cd:.3f}  [predict rho>0: higher CRB"
              f" = more under-determined = bigger learned gain]")
        print(f"  Spearman(CRB, aperture) rho {rho_ca:+.3f}  "
              f"[E-19: CRB is ~a re-scaling of aperture -> NOT independent]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='Stage A only (re-analysis headline); exits.')
    ap.add_argument('--with_crb', action='store_true',
                    help='also run Stage B CRB axis (light torch run).')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    res = stage_a(verbose=True)
    if a.with_crb and not a.check:
        res['stage_b_crb'] = stage_b(res['per_record'], verbose=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
