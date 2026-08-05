"""NEUTRAL probes for proposition H (within-frequency spread identifiability).

No wave field is simulated.  Four probes:

  P1  Is arm B's "within-frequency spread" a physical frequency-direction
      coupling, or a partition statistic of i.i.d. draws?  (generator code
      draws `directions = rng.normal(dir, spread, n_components)` with NO bin
      structure, and within-bin amplitudes are equal -> r1_within is a
      function of M and the direction law ALONE.)
  P2  Power of the pre-registered sign test: what could this design miss?
  P3  Attribution: can n=5 x 2 separate "C reproduces B" from "B is 4x C"?
  P4  Does a clean contrast exist?  Two-knob (s_within, s_across) draw at
      FIXED N, F, M -- can total spread be held while within varies?

Run: python3 tmp/identifiability_debate_2026-08-05/NEUTRAL_probes.py
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05')
S4 = os.path.join(ROOT, 'tmp/s4_within_spread_2026-08-05')
sys.path.insert(0, ROOT)

out = {}


def s1deg(r1):
    r1 = min(max(float(r1), 0.0), 1.0)
    return float(np.degrees(np.sqrt(max(2.0 * (1.0 - r1), 0.0))))


# ---------------------------------------------------------------- P1
# Arm B's within-bin weights are EXACTLY uniform: the generator sets
# A_i = A_bin/sqrt(M) repeated M times, so wb = 1/M inside every bin.
# Directions are i.i.d. N(mu, spread) over all N, independent of the bin
# index.  => r1_within is a functional of (M, direction law) only.
# Probe: draw i.i.d. directions with NO frequency at all, chop into groups
# of M in arbitrary order, and compute the identical statistic.
def p1(spread_deg=30.0, N=30, M=3, reps=200000, seed=7):
    rng = np.random.default_rng(seed)
    s = np.deg2rad(spread_deg)
    F = N // M
    th = rng.normal(0.0, s, size=(reps, F, M))
    # within: uniform weights inside a bin; bins carry different energy, but
    # r1_within is an energy-weighted mean of per-bin r1 whose distribution is
    # bin-independent -> use the plain mean over bins (unbiased for the mixture).
    z = np.exp(1j * th).mean(axis=2)          # (reps, F) per-bin resultant
    r1_within = np.abs(z).mean(axis=1)
    return dict(spread_deg=spread_deg, N=N, M=M, reps=reps,
                r1_within_mean=float(r1_within.mean()),
                sigma1_within_deg_from_mean_r1=s1deg(r1_within.mean()),
                sigma1_within_deg_mean_of_s1=float(
                    np.mean([s1deg(x) for x in r1_within[:20000]])))


mc = json.load(open(os.path.join(S4, 'manip_check.json')))
armB = mc['arms']['B_M3_N30']['median']
armA = mc['arms']['A_M1_N30']['median']
armC = mc['arms']['C_M1_N10']['median']
p1_res = p1()
p1_res['arm_B_measured_within_deg'] = armB['sigma1_within_deg']
p1_res['arm_B_per_seed_within_deg'] = [r['sigma1_within_deg']
                                       for r in mc['arms']['B_M3_N30']['per_seed']]
# Also: sanity that no-frequency i.i.d. reproduces arm B's ACROSS term too.
p1_res['arm_B_measured_across_deg'] = armB['sigma1_across_deg']
p1_res['arm_A_measured_across_deg'] = armA['sigma1_across_deg']
p1_res['verdict'] = ('r1_within depends only on (M, direction law); frequency '
                     'enters nowhere.  If the i.i.d.-no-frequency draw matches '
                     'arm B, the arm did not create a frequency-direction '
                     'coupling -- it relabelled i.i.d. draws.')
out['P1_relabelling'] = p1_res


# ---------------------------------------------------------------- P2 power
paired = json.load(open(os.path.join(S4, 'analysis.json')))['paired']


def pairs(arm, metric='nRMSE'):
    v = paired[arm][metric]
    for key in ('deltas', 'per_pair', 'diffs', 'pairwise'):
        if isinstance(v, dict) and key in v:
            return np.asarray(v[key], dtype=float)
    if isinstance(v, list):
        return np.asarray(v, dtype=float)
    raise KeyError(f'{arm}/{metric}: {v if not isinstance(v, dict) else list(v)}')


def sign_test_power(delta, sd, n=5, need=None):
    """P(all n pairs share the hypothesised sign) at true mean `delta`."""
    from math import erf, sqrt
    p = 0.5 * (1 + erf(delta / (sd * sqrt(2))))
    if need is None:
        need = n
    return p ** n, p


def t_mde(sd, n=5, alpha=0.05, power=0.80, one_sided=True):
    """MDE of a paired t-test, normal-approx noncentrality (df=n-1 correction
    applied via a small lookup so the number is not optimistic)."""
    tcrit = {4: 2.132 if one_sided else 2.776}[n - 1]
    zb = 0.8416
    lam = tcrit + zb            # crude but conservative-ish
    return lam * sd / np.sqrt(n)


ws_ba = pairs('WS|B_minus_A')
ws_ca = pairs('WS|C_minus_A')
le_ba = pairs('LETKF|B_minus_A')
le_ca = pairs('LETKF|C_minus_A')

sd = float(np.std(ws_ba, ddof=1))
mde_sign = 1.712 * sd
obs = float(np.median(ws_ba))
pw_at_obs, p_at_obs = sign_test_power(abs(obs), sd)
grid = {}
for frac in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    d = frac * mde_sign
    pw, _ = sign_test_power(d, sd)
    grid[f'{frac:g}xMDE={d:.4f}'] = round(pw, 3)

lvl = json.load(open(os.path.join(S4, 'analysis.json')))['levels']
out['P2_power'] = dict(
    ws_BA_pairs=ws_ba.tolist(), sd_pairwise=sd,
    mean=float(ws_ba.mean()), median=obs,
    mde_sign_test_80pct=mde_sign,
    mde_as_pct_of_armA_level=None,
    power_at_observed_abs_delta=pw_at_obs,
    power_grid=grid,
    mde_paired_t_test_80pct=float(t_mde(sd)),
    t_stat=float(ws_ba.mean() / (sd / np.sqrt(len(ws_ba)))),
    note=('sign test needs 5/5; MDE = z(0.8^(1/5)) * sd = 1.712*sd. '
          'A paired t-test on the same 5 numbers has a smaller MDE.'))


# ---------------------------------------------------------------- P3 attribution
bc = le_ba - le_ca          # (B-A) - (C-A) = B - C, seed-paired
sd_bc = float(np.std(bc, ddof=1))
se_bc = sd_bc / np.sqrt(len(bc))
t95 = 2.132          # one-sided .95 / two-sided .90, df=4
ci90 = (float(bc.mean() - t95 * se_bc), float(bc.mean() + t95 * se_bc))
out['P3_attribution'] = dict(
    letkf_BA=le_ba.tolist(), letkf_CA=le_ca.tolist(), letkf_BC=bc.tolist(),
    median_BA=float(np.median(le_ba)), median_CA=float(np.median(le_ca)),
    median_BC=float(np.median(bc)), mean_BC=float(bc.mean()),
    n_pos_BC=int((bc > 0).sum()),
    sd_BC=sd_bc, ci90_mean_BC=ci90,
    ratio_upper_over_median_BA=float(ci90[1] / np.median(le_ba)),
    equivalence_question=('For "C reproduces B" to be MEASURED, the CI on B-C '
                          'must exclude effects comparable to B-A itself.'))


# ---------------------------------------------------------------- P4 clean contrast
def two_knob(s_within_deg, s_across_deg, N=30, M=3, reps=40000, seed=11):
    """Frequency-conditional directional model:
         theta_bar_i ~ N(mu, s_across);  theta_ij ~ N(theta_bar_i, s_within)
       N, F=N//M and M are IDENTICAL across settings; only (s_w, s_a) move."""
    rng = np.random.default_rng(seed)
    F = N // M
    sw, sa = np.deg2rad(s_within_deg), np.deg2rad(s_across_deg)
    tb = rng.normal(0.0, sa, size=(reps, F, 1))
    th = tb + rng.normal(0.0, sw, size=(reps, F, M))
    zb = np.exp(1j * th).mean(axis=2)
    r1_within = np.abs(zb).mean(axis=1)
    r1_tot = np.abs(np.exp(1j * th).reshape(reps, -1).mean(axis=1))
    r1_ac = r1_tot / np.maximum(r1_within, 1e-12)
    return dict(s_within_deg=s_within_deg, s_across_deg=s_across_deg,
                N=N, M=M, n_freq=F,
                sigma1_within_deg=s1deg(r1_within.mean()),
                sigma1_across_deg=s1deg(np.clip(r1_ac, 0, 1).mean()),
                sigma1_total_deg=s1deg(r1_tot.mean()))


# hold total ~ arm A/B (24.2 deg): sqrt(sw^2+sa^2) approx const
ladder = []
for sw, sa in [(4.0, 23.7), (10.0, 22.0), (14.0, 19.5), (17.0, 17.0), (20.0, 13.5)]:
    ladder.append(two_knob(sw, sa))
out['P4_clean_contrast'] = dict(
    design=('N=30, M=3, n_freq=10 held FIXED in every rung; only the '
            'within/across split of the SAME total spread moves.'),
    ladder=ladder,
    code_change=('wave_generator.py:266 currently draws '
                 'directions = rng.normal(dir_rad, spread_rad, n_components) '
                 'with no bin structure.  A frequency-conditional draw is a '
                 '2-line change and does not touch N or n_freq.'))

json.dump(out, open(os.path.join(HERE, 'NEUTRAL_probes.json'), 'w'), indent=2)

# ------------------------------------------------------------------ print
print('== P1  is arm B within-spread a partition statistic of i.i.d. draws?')
print(f"   i.i.d. no-frequency draw (N=30,M=3,spread=30deg): "
      f"within = {p1_res['sigma1_within_deg_from_mean_r1']:.2f} deg")
print(f"   arm B measured within (median over 5 seeds):      "
      f"{armB['sigma1_within_deg']:.2f} deg")
print(f"   arm B per-seed: {[round(x,2) for x in p1_res['arm_B_per_seed_within_deg']]}")

print('\n== P2  power of the pre-registered sign test (WS nRMSE, B-A)')
print(f"   pairs = {np.round(ws_ba,4).tolist()}")
print(f"   sd_pairwise = {sd:.4f}   median = {obs:+.4f}   mean = {ws_ba.mean():+.4f}")
print(f"   MDE(sign,80%) = {mde_sign:.4f}   MDE(paired t,80%) = {t_mde(sd):.4f}")
print(f"   power at |observed| ({abs(obs):.3f}) = {pw_at_obs:.3f}")
for k, v in grid.items():
    print(f"     power @ {k}: {v}")
print(f"   arm A WS nRMSE level = {lvl['WS']['A_M1_N30']['nRMSE'] if isinstance(lvl['WS'].get('A_M1_N30'),dict) else lvl['WS'].get('A_M1_N30')}")

print('\n== P3  attribution: LETKF B-C (seed-paired)')
print(f"   B-A = {np.round(le_ba,4).tolist()}  median {np.median(le_ba):+.4f}")
print(f"   C-A = {np.round(le_ca,4).tolist()}  median {np.median(le_ca):+.4f}")
print(f"   B-C = {np.round(bc,4).tolist()}  median {np.median(bc):+.4f}  "
      f"{int((bc>0).sum())}/5 positive")
print(f"   90% CI on mean(B-C) = [{ci90[0]:+.4f}, {ci90[1]:+.4f}]  "
      f"(upper is {ci90[1]/np.median(le_ba):.1f}x the whole B-A median)")

print('\n== P4  does a clean contrast exist?  (N=30, M=3, n_freq=10 fixed in all rungs)')
for r in ladder:
    print(f"   s_w={r['s_within_deg']:5.1f} s_a={r['s_across_deg']:5.1f} -> "
          f"within {r['sigma1_within_deg']:6.2f}  across {r['sigma1_across_deg']:6.2f}  "
          f"total {r['sigma1_total_deg']:6.2f}   (N=30, n_freq=10, M=3)")
print(f"\nwrote {os.path.join(HERE,'NEUTRAL_probes.json')}")
