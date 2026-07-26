"""Analysis for tmp/blind_auto_real_2026-07-26/PREREG.md. Frozen bars B1-B4 only."""
import os, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D = os.path.join(ROOT, 'tmp/blind_auto_real_2026-07-26')
J = json.load(open(os.path.join(D, 'blind_auto_real.json')))
rows = J['rows']

lines = []; P = lines.append
out = {}

# ---- B4 lint ----
P('== B4  one-policy lint (one callable, shipped constants, no arena branch) ==')
for k, v in J['lint'].items():
    P(f'  {k:32s} {v}')
P(f"  shipped constants used: {J['shipped']}")
b4 = (J['lint'].get('DEFAULT_CONFIG_init_method') == 'auto'
      and J['lint'].get('seed_matches_default') is True
      and all(J['lint'].get(k) for k in ('auto_switch_thresh', 'auto_guard', 'auto_guard_ratio')))
P(f'  ==> B4 {"PASS" if b4 else "FAIL"}')
out['B4'] = dict(verdict='PASS' if b4 else 'FAIL', lint=J['lint'], shipped=J['shipped'])

# ---- route distribution ----
P('\n== ROUTE DISTRIBUTION (primary diffusion seed 42 = shipped DEFAULT_CONFIG seed) ==')
n_learn = sum(r['route'] == 'learned' for r in rows)
n_v2 = sum(r['route'] == 'v2_konly' for r in rows)
n_guard = sum(bool(r['guard_fired']) for r in rows)
P(f'  learned {n_learn}/30   v2_konly {n_v2}/30   guard fired {n_guard}/30')
norm = [r for r in rows if not r['collapse']]
P(f'  on the 27 NORMAL windows: learned {sum(r["route"]=="learned" for r in norm)}/27')
ns = [r['n_selected'] for r in rows]
bf = [r['bf_peak_ratio'] for r in rows if r['bf_peak_ratio'] is not None]
P(f'  n_selected: min {min(ns)} median {int(np.median(ns))} max {max(ns)}  (thresh 8)')
P(f'  bf_peak_ratio: min {min(bf):.4f} median {np.median(bf):.4f} max {max(bf):.4f}  (veto <0.05)')
agree = [sum(r['per_seed'][str(d)]['route'] == r['route'] for d in J['diff_seeds']) for r in rows]
P(f'  6-seed route agreement: {sum(a==len(J["diff_seeds"]) for a in agree)}/30 windows unanimous')
out['routes'] = dict(learned=n_learn, v2_konly=n_v2, guard_fired=n_guard, n=30,
                     learned_on_normals=sum(r['route'] == 'learned' for r in norm),
                     n_selected_min=min(ns), n_selected_max=max(ns),
                     bf_ratio_min=float(min(bf)), bf_ratio_max=float(max(bf)),
                     unanimous_seeds=int(sum(a == len(J['diff_seeds']) for a in agree)))

# ---- per-window table ----
P('\n== PER-WINDOW ==')
P(f"{'window':10s} {'route':9s} {'nsel':>4} {'bfr':>7} {'hand':>7} {'learned':>8} {'policy':>7} {'d=hand-pol':>11}")
for r in rows:
    P(f"{r['record']+'_w'+str(r['win']):10s} {r['route']:9s} {r['n_selected']:4d} "
      f"{r['bf_peak_ratio']:7.4f} {r['hand_now']:7.4f} {r['learned_now']:8.4f} "
      f"{r['policy_now']:7.4f} {r['hand_now']-r['policy_now']:+11.4f}"
      f"{'  [collapse]' if r['collapse'] else ''}")

# ---- B1 primary + secondary ----
def med(v):
    return float(np.median(v))

P('\n== B1  PRIMARY: median(policy) <= median(hand) + 0.010, all 30 windows ==')
res = {}
for name, sel in (('all30', rows), ('normals27', norm)):
    mp = med([r['policy_now'] for r in sel]); mh = med([r['hand_now'] for r in sel])
    ml = med([r['learned_now'] for r in sel])
    d = [r['hand_now'] - r['policy_now'] for r in sel]
    res[name] = dict(n=len(sel), median_policy=mp, median_hand=mh, median_learned=ml,
                     diff=mp - mh, median_paired_delta=med(d),
                     mean_policy=float(np.mean([r['policy_now'] for r in sel])),
                     mean_hand=float(np.mean([r['hand_now'] for r in sel])))
    P(f'  {name:10s} n={len(sel):2d}  median hand {mh:.4f}  median learned {ml:.4f}  '
      f'median policy {mp:.4f}   policy-hand {mp-mh:+.4f}   median paired d {med(d):+.4f}')
b1 = res['all30']['diff'] <= 0.010
P(f"  ==> B1 {'PASS' if b1 else 'FAIL'}  (policy-hand {res['all30']['diff']:+.4f} "
  f"vs allowance +0.0100)")
out['B1'] = dict(verdict='PASS' if b1 else 'FAIL', **res)

# ---- B2 P1 scoping clause ----
P('\n== B2  P1 scoping clause: routed-learned on >=20/27 normals AND median paired d < -0.03 ==')
rl = [r for r in norm if r['route'] == 'learned']
d_rl = med([r['hand_now'] - r['policy_now'] for r in rl]) if rl else float('nan')
c1 = len(rl) >= 20; c2 = d_rl < -0.03
P(f'  routed-learned normals {len(rl)}/27 (>=20? {c1});  median paired d on them {d_rl:+.4f} '
  f'(< -0.03? {c2})')
b2 = c1 and c2
P(f'  ==> B2 {"FIRES" if b2 else "does not fire"}')
if b2:
    P('  pre-committed consequence: the loss is REPORTABLE and the thesis abstract must '
      'scope the unified ("one estimator") claim to SYNTHETIC data.')
out['B2'] = dict(fires=bool(b2), n_routed_learned_normals=len(rl),
                 median_paired_delta=d_rl, cond_count=bool(c1), cond_delta=bool(c2))

# ---- B3 P3 failure wording ----
P('\n== B3  P3 pre-committed wording: median(policy) - median(hand) > +0.02 (all 30) ==')
b3 = res['all30']['diff'] > 0.02
P(f"  median(policy)-median(hand) = {res['all30']['diff']:+.4f}  ==> "
  f"{'FIRES' if b3 else 'does not fire'}")
if b3:
    P('  pre-committed wording, verbatim: the thesis must report "two policies".')
out['B3'] = dict(fires=bool(b3), diff=res['all30']['diff'])

# ---- record-unit robustness (declared secondary) ----
P('\n== DECLARED SECONDARY (never a verdict): record-unit medians, n=5 ==')
for rec in ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']:
    sel = [r for r in rows if r['record'] == rec]
    P(f'  {rec:6s} hand {med([r["hand_now"] for r in sel]):.4f}  '
      f'learned {med([r["learned_now"] for r in sel]):.4f}  '
      f'policy {med([r["policy_now"] for r in sel]):.4f}')
rec_h = [med([r['hand_now'] for r in rows if r['record'] == rc])
         for rc in ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']]
rec_p = [med([r['policy_now'] for r in rows if r['record'] == rc])
         for rc in ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']]
P(f'  record-median-of-medians: hand {med(rec_h):.4f}  policy {med(rec_p):.4f}  '
  f'({med(rec_p)-med(rec_h):+.4f});  policy worse on '
  f'{sum(p>h for p,h in zip(rec_p,rec_h))}/5 records')
out['record_unit'] = dict(hand=rec_h, policy=rec_p,
                          worse_records=int(sum(p > h for p, h in zip(rec_p, rec_h))))

# ---- win/loss counts (diagnostic) ----
P('\n== DIAGNOSTIC (not a bar) ==')
P(f'  policy worse than hand on {sum(r["policy_now"]>r["hand_now"] for r in rows)}/30 windows'
  f' ({sum(r["policy_now"]>r["hand_now"] for r in norm)}/27 normals)')
P(f'  windows where policy nRMSE >= 1.0 (unusable): '
  f'{sum(r["policy_now"]>=1.0 for r in rows)}/30 vs hand {sum(r["hand_now"]>=1.0 for r in rows)}/30')
out['diagnostic'] = dict(
    policy_worse_n=int(sum(r['policy_now'] > r['hand_now'] for r in rows)),
    policy_worse_normals=int(sum(r['policy_now'] > r['hand_now'] for r in norm)),
    policy_unusable=int(sum(r['policy_now'] >= 1.0 for r in rows)),
    hand_unusable=int(sum(r['hand_now'] >= 1.0 for r in rows)))

txt = '\n'.join(lines)
open(os.path.join(D, 'analysis.txt'), 'w').write(txt + '\n')
json.dump(out, open(os.path.join(D, 'results.json'), 'w'), indent=1)
print(txt)
