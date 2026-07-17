"""e43 — EXTERNAL regime information: dose-response.

WHY. e42 established (pre-registered, n=64 fresh seas) that the cross-Tp
regime prize is REAL (oracle - cur_grid = +0.0715 +- 0.0141, t=5.06) but that
blind exact-evidence inference of theta recovers NONE of it (fit_bmaF - cur =
-0.0166, CI upper +0.0166 < +0.02 => DEAD; and fit_bmaF ~= U_BMA, i.e. the
likelihood does not discriminate among hypotheses at all).

What is closed is INFERRING theta FROM THIS OBSERVATION. What is not closed is
theta being VALUABLE. The oracle earns +0.0715 by knowing theta, so if theta
arrives from OUTSIDE -- a WW3/WAM forecast, a buoy, a previous window -- it
pays. This is the same shape as the current(Uc)-Doppler arc, which ended in
exactly the same prescription: "blind recovery 5/5 failed -> supply Uc
externally, pass kp_hint to the net."

  => The regime axis is an EXTERNAL-INFORMATION channel, not an inference
     channel.  This script turns that sentence into a SPEC.

THE QUESTION. External regime info always arrives with error. How accurate
must it be to pay -- and at what error does it start to HARM? (e42 showed the
2-lobe family is high-gain/high-risk: right parameters +0.0715, typical blind
parameters -0.017 = PERM at the family level. So a harm threshold MUST exist
somewhere between.)

DESIGN. Same fresh seas as e42 (arena X, si=100..163, n=64). Supply theta with
controlled error, score through the identical solver, compare to cur_grid.
  ARM D  direction dose: d1 and d2 each perturbed by N(0, sd), sd in
         {0,5,10,15,20,30,45,60,90} deg. ring/weight true.
  ARM R  ring dose: Tp2 perturbed by relative N(0, sT), sT in
         {0,2,5,10,20,40} %. directions/weight true.
  ARM W  weight dose: r2 perturbed by N(0, sw), sw in {0,0.05,0.10,0.20},
         clipped to [0.05,0.6]. directions/ring true.
  ARM C  realistic bundles: 'buoy'  (d +-5 deg,  Tp +-5%,  r2 +-0.05)
                            'ww3'   (d +-15 deg, Tp +-10%, r2 +-0.10)
                            'coarse'(d +-30 deg, Tp +-20%, r2 +-0.20)

R = 5 independent perturbation draws per (sea, dose). This is deliberate: the
E-14 audit found that ONE draw per cell makes the CI measure draw variability
rather than the effect, and that a frozen rng had silently picked the 95th
percentile of 10 draws. Per-sea means are taken over the R draws BEFORE the
paired test, so n = 64 seas (not 320 pseudo-replicates).

ANCHOR (must hold or the script is wrong): at dose 0 every arm must reproduce
the e42 oracle contrast, +0.0715 +- 0.0141.

READOUTS
  sd*  = direction accuracy at which gain crosses +0.03 (a deployable bar)
  sd0  = direction accuracy at which gain crosses 0     (the harm boundary)
  same for sT, sw. Reported by linear interpolation between bracketing doses.

Usage: python3 tmp/e43_external_regime_dose.py [--smoke] [--n 64] [--R 5]
Out:   results/direction3/e43_external_regime_dose.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, net_gamma_map
from e41_regime_prize_ladder import lobe_grid

OUT = 'results/direction3/e43_external_regime_dose.json'

SD_DOSES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0]     # deg
ST_DOSES = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]                        # percent
SW_DOSES = [0.0, 0.05, 0.10, 0.20]                                  # absolute
BUNDLES = {'buoy': (5.0, 5.0, 0.05),
           'ww3': (15.0, 10.0, 0.10),
           'coarse': (30.0, 20.0, 0.20)}
# deterministic per-bundle seed offsets. NOT Python hash(): it is process-
# salted, which is exactly what disqualified E-28 design B for failing the
# "frozen JSON reproduces bit-for-bit" standard (E-14b).
BUNDLE_SEED = {'buoy': 11, 'ww3': 23, 'coarse': 37}


def supplied_grid(S, d1, d2, Tp2, r2):
    """The oracle's own construction, but with SUPPLIED theta."""
    kp = S['kp']
    L1 = lobe_grid(kp, [(7.0, 3.3, 1.0)], [(d1, 30.0, 1.0)])
    L2 = lobe_grid(kp, [(float(Tp2), 3.3, 1.0)], [(d2, 30.0, 1.0)])
    return (1 - r2) * L1 + r2 * L2


def perturb(S, rng, sd=0.0, sT=0.0, sw=0.0):
    d1 = S['d1'] + np.deg2rad(rng.normal(0, sd)) if sd > 0 else S['d1']
    d2 = S['d2'] + np.deg2rad(rng.normal(0, sd)) if sd > 0 else S['d2']
    Tp2 = S['Tp2'] * (1 + rng.normal(0, sT / 100.0)) if sT > 0 else S['Tp2']
    Tp2 = float(np.clip(Tp2, 3.0, 16.0))
    r2 = float(np.clip(S['r2'] + rng.normal(0, sw), 0.05, 0.6)) if sw > 0 \
        else S['r2']
    return d1, d2, Tp2, r2


def cross(xs, ys, level):
    """First x at which the curve ys(xs) crosses `level` going down.
    Linear interpolation; None if it never crosses inside the swept range."""
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if (a - level) * (b - level) <= 0 and a != b:
            f = (a - level) / (a - b)
            return float(xs[i] + f * (xs[i + 1] - xs[i]))
    return None


def paired(per_sea_a, per_sea_b):
    d = np.asarray(per_sea_a) - np.asarray(per_sea_b)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=float(d.mean()), se=float(se), t=float(t), p=float(p),
                lo=float(lo), hi=float(hi), win=int((d > 0).sum()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--R', type=int, default=5)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.R = 4, 2
    t0 = time.time()

    seas, cur = [], []
    print(f'--- building {a.n} seas + incumbent (arena X, si=100..) ---',
          flush=True)
    for si in range(100, 100 + a.n):
        S = build_sea_crosstp(si)
        seas.append(S)
        Y = net_gamma_map(S)
        cur.append(solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8)))
    cur = np.array(cur)
    print(f'  cur_grid mean {cur.mean():+.4f}  [{time.time()-t0:.0f}s]',
          flush=True)

    out = {'cur_mean': float(cur.mean()), 'n': a.n, 'R': a.R, 'arms': {}}

    def run_dose(tag, doses, kw_of):
        rows = []
        for dv in doses:
            per_sea = []
            for i, S in enumerate(seas):
                rng = np.random.default_rng(90000 + 7 * i + int(dv * 1000))
                vals = []
                for r in range(a.R):
                    d1, d2, Tp2, r2 = perturb(S, rng, **kw_of(dv))
                    g = supplied_grid(S, d1, d2, Tp2, r2)
                    vals.append(solve_with_grid(S, g))
                per_sea.append(np.mean(vals))
            st = paired(per_sea, cur)
            rows.append(dict(dose=float(dv), gain=st['mean'], se=st['se'],
                             lo=st['lo'], hi=st['hi'], win=st['win'],
                             mean_abs=float(np.mean(per_sea))))
            print(f"  {tag} dose {dv:6.2f} : gain {st['mean']:+.4f} "
                  f"+- {st['se']:.4f} CI [{st['lo']:+.4f},{st['hi']:+.4f}] "
                  f"win {st['win']}/{st['n']} [{time.time()-t0:.0f}s]",
                  flush=True)
        out['arms'][tag] = rows
        xs = [r['dose'] for r in rows]
        ys = [r['gain'] for r in rows]
        out['arms'][tag + '_cross'] = dict(
            at_003=cross(xs, ys, 0.03), at_000=cross(xs, ys, 0.0))
        return rows

    print('\n--- ARM D: direction dose (deg) ---', flush=True)
    run_dose('D', SD_DOSES, lambda v: dict(sd=v))
    print('\n--- ARM R: ring/Tp2 dose (%) ---', flush=True)
    run_dose('R', ST_DOSES, lambda v: dict(sT=v))
    print('\n--- ARM W: weight dose (abs) ---', flush=True)
    run_dose('W', SW_DOSES, lambda v: dict(sw=v))

    print('\n--- ARM C: realistic bundles ---', flush=True)
    brows = []
    for name, (sd, sT, sw) in BUNDLES.items():
        per_sea = []
        for i, S in enumerate(seas):
            rng = np.random.default_rng(91000 + 7 * i + BUNDLE_SEED[name])
            vals = []
            for r in range(a.R):
                d1, d2, Tp2, r2 = perturb(S, rng, sd=sd, sT=sT, sw=sw)
                vals.append(solve_with_grid(S, supplied_grid(S, d1, d2, Tp2, r2)))
            per_sea.append(np.mean(vals))
        st = paired(per_sea, cur)
        brows.append(dict(name=name, sd=sd, sT=sT, sw=sw, **st))
        star = '*' if st['lo'] > 0 else ' '
        print(f"  {name:7s} (d+-{sd:.0f}deg Tp+-{sT:.0f}% r2+-{sw:.2f}): "
              f"gain {st['mean']:+.4f} +- {st['se']:.4f} "
              f"CI [{st['lo']:+.4f},{st['hi']:+.4f}]{star} "
              f"win {st['win']}/{st['n']}", flush=True)
    out['arms']['C'] = brows

    print('\n=== SPEC READOUT ===')
    for tag, unit in [('D', 'deg'), ('R', '% Tp2'), ('W', 'abs r2')]:
        c = out['arms'][tag + '_cross']
        a3 = c['at_003']; a0 = c['at_000']
        print(f"  {tag}: gain >= +0.03 while error <= "
              f"{('%.1f' % a3) if a3 is not None else 'n/a'} {unit}"
              f"   |  harm (gain<0) beyond "
              f"{('%.1f' % a0) if a0 is not None else 'never in range'} {unit}")
    anchor = out['arms']['D'][0]['gain']
    print(f"\n  ANCHOR dose-0 gain = {anchor:+.4f} "
          f"(must match e42 oracle-cur = +0.0715 +- 0.0141)")
    ok = abs(anchor - 0.0715) < 0.03
    print(f"  anchor OK: {ok}")
    out['anchor_ok'] = bool(ok)

    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
