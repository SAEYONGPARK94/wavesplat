"""Assemble the paired old(=flag OFF, shipped) vs flag-ON evaluation table.

Baseline (flag OFF) is reused from tmp/rerun_2026-07-24/task13_runs.jsonl (finemesh, P1) which
is bit-identical to ws_runs.jsonl P1 (verified). Flag-ON runs come from ampfix_runs.jsonl.

Both are median-of-5 seeds [42,1,2,3,4], same 12 Table-1 episodes, so the comparison is paired
at the episode level. Applies the frozen adoption rule (see TRACKER_AMPDECAY_PREREG_2026-07-25.md).
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RERUN = os.path.abspath(os.path.join(HERE, '..', 'rerun_2026-07-24'))
BASELINE = os.path.join(RERUN, 'task13_runs.jsonl')
FLAGON = os.path.join(HERE, 'ampfix_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]
PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
EPISODES = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS] + [
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_slope.h5']

# flag-sweep 6-regime subset (for the literal >=4/6 rule)
FLAGSWEEP6 = ['episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_uniform.h5',
              'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_uniform.h5',
              'episode_BF4_30comp_seed0_uniform.h5', 'episode_BF5_15comp_seed0_uniform.h5']
CALM = ['episode_BF3_1comp_seed0_uniform.h5', 'episode_BF4_1comp_seed0_uniform.h5']

SEED_SD_FLOOR = 0.031  # frozen repo flag-sweep floor


def load_baseline():
    d = {}
    init = {}
    for l in open(BASELINE):
        r = json.loads(l)
        if r.get('mesh') == 'finemesh' and 'avg_nRMSE' in r:
            d[(r['episode'], r['seed'])] = r['avg_nRMSE']
            init[(r['episode'], r['seed'])] = r.get('init_nrmse', float('nan'))
    return d, init


def load_flagon():
    d = {}
    init = {}
    if not os.path.exists(FLAGON):
        return d, init
    for l in open(FLAGON):
        r = json.loads(l)
        if 'avg_nRMSE' in r:
            d[(r['episode'], r['seed'])] = r['avg_nRMSE']
            init[(r['episode'], r['seed'])] = r.get('init_nrmse', float('nan'))
    return d, init


def med(d, ep):
    vals = [d[(ep, s)] for s in SEEDS if (ep, s) in d]
    return (float(np.median(vals)), float(np.std(vals)),
            float(np.min(vals)), float(np.max(vals)), len(vals)) if vals else (None,)*5


def main():
    base, base_init = load_baseline()
    flag, flag_init = load_flagon()
    print(f"baseline runs {len(base)} | flag-on runs {len(flag)}\n")

    print(f"{'episode':<38} {'OFF med(sd)':<16} {'ON med(sd)':<16} {'delta':<9} dir")
    rows = []
    for ep in EPISODES:
        bm = med(base, ep)
        fm = med(flag, ep)
        if bm[0] is None or fm[0] is None:
            print(f"{ep:<38} {'?':<16} {'?':<16}  (incomplete)")
            continue
        delta = fm[0] - bm[0]  # negative = improvement
        dirn = 'IMPROVE' if delta < 0 else ('worse' if delta > 0 else 'same')
        rows.append({'ep': ep, 'off': bm, 'on': fm, 'delta': delta})
        print(f"{ep:<38} {bm[0]:.4f}({bm[1]:.3f})    {fm[0]:.4f}({fm[1]:.3f})    "
              f"{delta:+.4f}  {dirn}")

    if not rows:
        print("\nNo complete rows yet.")
        return

    deltas = np.array([r['delta'] for r in rows])
    n_improve = int((deltas < 0).sum())
    print(f"\n--- 12-episode aggregate ---")
    print(f"mean delta = {deltas.mean():+.4f}  (improve on {n_improve}/{len(rows)})")
    print(f"OFF agg median-of-medians = {np.median([r['off'][0] for r in rows]):.4f}, "
          f"mean = {np.mean([r['off'][0] for r in rows]):.4f}")
    print(f"ON  agg median-of-medians = {np.median([r['on'][0] for r in rows]):.4f}, "
          f"mean = {np.mean([r['on'][0] for r in rows]):.4f}")

    # flag-sweep 6-regime subset rule
    sub = [r for r in rows if r['ep'] in FLAGSWEEP6]
    sd = np.array([r['delta'] for r in sub])
    n_imp6 = int((sd < 0).sum())
    print(f"\n--- flag-sweep 6-regime subset (literal >=4/6 rule) ---")
    for r in sub:
        print(f"  {r['ep']:<38} {r['delta']:+.4f}")
    print(f"  improve on {n_imp6}/6, mean delta {sd.mean():+.4f}, "
          f"|mean| vs floor {SEED_SD_FLOOR}: "
          f"{'EXCEEDS' if abs(sd.mean())>SEED_SD_FLOOR else 'within noise'}")

    # calm regression shrink
    print(f"\n--- calm-episode regression (init -> track gap) ---")
    for ep in CALM:
        bm = med(base, ep); fm = med(flag, ep)
        bi = med(base_init, ep); fi = med(flag_init, ep)
        if bm[0] is None or fm[0] is None:
            print(f"  {ep}: incomplete"); continue
        gap_off = bm[0] - bi[0]
        gap_on = fm[0] - fi[0]
        shrink = (gap_off - gap_on) / gap_off * 100 if gap_off else float('nan')
        print(f"  {ep}")
        print(f"    OFF: init {bi[0]:.4f} -> track {bm[0]:.4f}  (gap {gap_off:+.4f})")
        print(f"    ON : init {fi[0]:.4f} -> track {fm[0]:.4f}  (gap {gap_on:+.4f})")
        print(f"    gap shrink = {shrink:.0f}%  {'PASS(>50%)' if shrink>50 else 'FAIL'}")


if __name__ == '__main__':
    main()
