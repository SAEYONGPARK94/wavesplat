"""Apply the frozen adoption rule to ampfix_paired_runs.jsonl (both arms, same device, identical
init per seed). Rule (tmp/TRACKER_AMPDECAY_PREREG_2026-07-25.md):
  R1  improve on >=4/6 flag-sweep regimes (paired median)
  R2  mean delta over 12 episodes < -0.031
  R3  both calm episodes' init->track gap shrinks >50%
Adopt iff R1 AND R2 AND R3.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'ampfix_paired_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]
PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
EPISODES = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS] + [
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_slope.h5']
FLAGSWEEP6 = ['episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_uniform.h5',
              'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_uniform.h5',
              'episode_BF4_30comp_seed0_uniform.h5', 'episode_BF5_15comp_seed0_uniform.h5']
CALM = ['episode_BF3_1comp_seed0_uniform.h5', 'episode_BF4_1comp_seed0_uniform.h5']
FLOOR = 0.031


def load():
    rows = {}
    for l in open(RUNS):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        rows[(r['episode'], r['seed'])] = r
    return rows


def stat(vals):
    a = np.array(vals, float)
    return dict(median=float(np.median(a)), sd=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                mn=float(a.min()), mx=float(a.max()), n=len(a))


def main():
    rows = load()
    short = lambda e: e.replace('episode_', '').replace('_seed0', '').replace('.h5', '')
    print(f"loaded {len(rows)} paired runs\n")
    print(f"{'episode':<22} {'OFF med(sd)':<15} {'ON med(sd)':<15} {'delta':<9} dir  init")
    table = []
    for ep in EPISODES:
        offs = [rows[(ep, s)]['off_avg_nRMSE'] for s in SEEDS if (ep, s) in rows]
        ons = [rows[(ep, s)]['on_avg_nRMSE'] for s in SEEDS if (ep, s) in rows]
        inits = [rows[(ep, s)]['init_nrmse'] for s in SEEDS if (ep, s) in rows]
        if len(offs) < 5 or len(ons) < 5:
            print(f"{short(ep):<22} incomplete ({len(offs)}/5)")
            continue
        so, sn, si = stat(offs), stat(ons), stat(inits)
        d = sn['median'] - so['median']
        table.append(dict(ep=ep, off=so, on=sn, init=si, delta=d))
        print(f"{short(ep):<22} {so['median']:.4f}({so['sd']:.3f})   {sn['median']:.4f}({sn['sd']:.3f})   "
              f"{d:+.4f}  {'IMPR' if d<0 else 'wors' if d>0 else 'same'}  {si['median']:.4f}")

    if len(table) < 12:
        print(f"\n[{len(table)}/12 complete -- partial]");
    deltas = np.array([t['delta'] for t in table])
    n_impr = int((deltas < 0).sum())
    print(f"\n=== 12-episode aggregate ({len(table)}/12) ===")
    print(f"mean delta = {deltas.mean():+.4f}   (improve {n_impr}/{len(table)})")
    print(f"OFF mean-of-medians {np.mean([t['off']['median'] for t in table]):.4f} | "
          f"ON {np.mean([t['on']['median'] for t in table]):.4f}")
    print(f"OFF median-of-medians {np.median([t['off']['median'] for t in table]):.4f} | "
          f"ON {np.median([t['on']['median'] for t in table]):.4f}")

    # R1
    sub = [t for t in table if t['ep'] in FLAGSWEEP6]
    n6 = int((np.array([t['delta'] for t in sub]) < 0).sum())
    print(f"\n--- R1 (6-regime >=4/6) ---")
    for t in sub:
        print(f"   {short(t['ep']):<20} {t['delta']:+.4f} {'IMPR' if t['delta']<0 else 'wors'}")
    R1 = (n6 >= 4) and len(sub) == 6
    print(f"   improve {n6}/6  -> R1 {'PASS' if R1 else ('FAIL' if len(sub)==6 else 'partial')}")

    # R2
    R2 = deltas.mean() < -FLOOR
    print(f"\n--- R2 (mean delta < -{FLOOR}) ---")
    print(f"   mean delta {deltas.mean():+.4f}  -> R2 {'PASS' if R2 else 'FAIL'}")

    # R3
    print(f"\n--- R3 (calm gap shrink >50%) ---")
    R3flags = []
    for ep in CALM:
        t = next((x for x in table if x['ep'] == ep), None)
        if not t:
            print(f"   {short(ep)}: incomplete"); R3flags.append(False); continue
        gap_off = t['off']['median'] - t['init']['median']
        gap_on = t['on']['median'] - t['init']['median']
        shrink = (gap_off - gap_on) / gap_off * 100 if abs(gap_off) > 1e-9 else float('nan')
        ok = shrink > 50
        R3flags.append(ok)
        print(f"   {short(ep):<14} init {t['init']['median']:.4f} | OFF gap {gap_off:+.4f} "
              f"ON gap {gap_on:+.4f} | shrink {shrink:.0f}%  {'PASS' if ok else 'FAIL'}")
    R3 = all(R3flags) and len(R3flags) == 2

    print(f"\n===== VERDICT =====")
    print(f"R1(>=4/6)={R1}  R2(mean<-0.031)={R2}  R3(calm>50%)={R3}")
    print(f"ADOPT = {R1 and R2 and R3}")


if __name__ == '__main__':
    main()
