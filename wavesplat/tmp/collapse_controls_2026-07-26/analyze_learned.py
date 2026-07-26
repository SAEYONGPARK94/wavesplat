"""Analysis for PREREG_GPU_ADDENDUM.md — learned arm vs cheap escapes on the 18 collapse
windows. Frozen criteria only; no new endpoint. Writes results_learned.json + analysis_learned.txt."""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D = os.path.join(ROOT, 'tmp/collapse_controls_2026-07-26')

esc = json.load(open(os.path.join(D, 'escapes.json')))
lrn = json.load(open(os.path.join(D, 'learned_arm.json')))
wpr = json.load(open(os.path.join(ROOT, 'tmp/wp_real_2026-07-25/wp_real.json')))

E = {(r['record'], r['win']): r for r in esc['rows']}
L = {(r['record'], r['win']): r for r in lrn['rows']}
W = {(r['record'], r['win']): r for r in wpr['results']}

ORDER = [('BS11b', 4), ('BS11c', 3), ('BS11c', 4), ('BS13a', 0), ('BS13a', 1),
         ('BS13a', 2), ('BS13a', 4), ('BS13a', 5), ('BS13c', 2), ('BS13c', 3),
         ('LJ18n', 0), ('LJ18n', 1), ('LJ18n', 3), ('LJ18n', 4), ('LJ18n', 5),
         ('AA14', 0), ('AA15a', 3), ('LJ18g', 3)]
DESIGN = [('AA14', 0), ('AA15a', 3), ('LJ18g', 3)]
FIRED = [('AA14', 0), ('LJ18g', 3), ('LJ18n', 0), ('LJ18n', 1), ('LJ18n', 4)]
C_BEST = {'a': 6, 'b': 11}          # measured, RESULT.md s2 (both E2-double)

out = {}
lines = []
P = lines.append

# ---- 1. bit-identity control (design-3 vs wp_real.json) ----
P('== 1. VALIDITY CONTROL: design-3 learned must reproduce wp_real.json to 1e-9 ==')
maxd = 0.0; bit = []
for k in DESIGN:
    a = L[k]['learned_now']; b = W[k]['learned_now']
    d = abs(a - b); maxd = max(maxd, d)
    sd_a = np.array(L[k]['learned_now_seeds']); sd_b = np.array(W[k]['learned_now_seeds'])
    dseed = float(np.max(np.abs(sd_a - sd_b))) if len(sd_a) == len(sd_b) else float('inf')
    maxd = max(maxd, dseed)
    bit.append(dict(window=f'{k[0]}_w{k[1]}', live=a, stored=b, d_median=d, d_seeds=dseed))
    P(f'  {k[0]}_w{k[1]:<2d}  live {a:.15f}  stored {b:.15f}  |d| {d:.3e}  seeds |d| {dseed:.3e}')
verdict_bit = 'PASS' if maxd <= 1e-9 else 'FAIL'
P(f'  -> max |delta| = {maxd:.3e}  ==> {verdict_bit}')
out['bit_identity'] = dict(rows=bit, max_abs_delta=maxd, verdict=verdict_bit, tol=1e-9)

# ---- 2. per-window table ----
P('\n== 2. PER-WINDOW: baseline hand | learned (median-of-5) | E2-double ==')
P(f"{'window':11s} {'arena':8s} {'base':>7} {'learned':>8} {'sd':>6} {'E2dbl':>7} "
  f"{'d_learn':>8} {'d_E2d':>7} {'npx':>4}")
rows = []
for k in ORDER:
    b = E[k]['baseline_recheck']; l = L[k]['learned_now']; e2 = E[k]['E2']['kp_x2.0']
    r = dict(window=f'{k[0]}_w{k[1]}', record=k[0], win=k[1], arena=E[k]['arena'],
             base=b, learned=l, learned_sd=L[k]['learned_now_sd'],
             learned_seeds=L[k]['learned_now_seeds'], n_eff=L[k]['n_eff'],
             npx=L[k]['npx_scored'], e2_double=e2,
             e1_median=float(np.median(list(E[k]['E1'].values()))),
             e2_peak2=E[k]['E2']['peak2'], e2_half=E[k]['E2']['kp_x0.5'],
             e3_half=E[k]['E3']['sig_x0.5'], e3_double=E[k]['E3']['sig_x2.0'],
             learned_a=bool(b - l >= 0.5), learned_b=bool(l < 1.0),
             e2d_a=bool(b - e2 >= 0.5), e2d_b=bool(e2 < 1.0))
    rows.append(r)
    P(f"{r['window']:11s} {r['arena']:8s} {b:7.4f} {l:8.4f} {r['learned_sd']:6.3f} "
      f"{e2:7.4f} {b-l:+8.4f} {b-e2:+7.4f} {r['npx']:4d}")
out['rows'] = rows

# ---- 3. the two counts (never merged) ----
def counts(sel, key_a, key_b):
    return sum(r[key_a] for r in sel), sum(r[key_b] for r in sel)

P('\n== 3. FROZEN COUNTS, never merged.  (a) base-esc >= 0.5   (b) esc < 1.0 ==')
la, lb = counts(rows, 'learned_a', 'learned_b')
ea, eb = counts(rows, 'e2d_a', 'e2d_b')
P(f'  learned    (a) {la}/18   (b) {lb}/18')
P(f'  E2-double  (a) {ea}/18   (b) {eb}/18   [matches stored C_best {C_BEST}]')
for arena in ('heldout', 'design'):
    sel = [r for r in rows if r['arena'] == arena]
    a1, b1 = counts(sel, 'learned_a', 'learned_b'); a2, b2 = counts(sel, 'e2d_a', 'e2d_b')
    P(f'    {arena:8s} n={len(sel):2d}  learned (a) {a1}/{len(sel)} (b) {b1}/{len(sel)}'
      f'   |  E2-double (a) {a2}/{len(sel)} (b) {b2}/{len(sel)}')
out['counts'] = dict(learned=dict(a=la, b=lb, n=18), e2_double=dict(a=ea, b=eb, n=18),
                     by_arena={ar: dict(
                         n=len([r for r in rows if r['arena'] == ar]),
                         learned_a=int(counts([r for r in rows if r['arena'] == ar],
                                              'learned_a', 'learned_b')[0]),
                         learned_b=int(counts([r for r in rows if r['arena'] == ar],
                                              'learned_a', 'learned_b')[1]))
                         for ar in ('heldout', 'design')})

# ---- 4. P2 bar ----
P('\n== 4. P2 BAR: learned_count >= 1.5 * C_best, per criterion, never merged ==')
bar = {}
for crit, cb, lc in (('a', C_BEST['a'], la), ('b', C_BEST['b'], lb)):
    need = int(np.ceil(1.5 * cb))
    v = 'PASS' if lc >= need else 'FAIL'
    bar[crit] = dict(C_best=cb, need=need, learned=lc, verdict=v)
    P(f'  ({crit})  C_best {cb}/18  -> need >= {need}/18  |  learned {lc}/18  ==> {v}')
out['p2_bar'] = bar
if bar['a']['verdict'] == 'FAIL' and bar['b']['verdict'] == 'FAIL':
    P("  ==> pre-committed consequence, BOTH criteria: Ch.7's interlock is "
      "\"a paragraph about initialization robustness\".")

# ---- 5. head-to-head on the 5 detector-fired windows ----
P('\n== 5. OPERATIONAL SUBSET (post-hoc, defined by deployment): the 5 windows tau=1.5 fires on ==')
P(f"{'window':11s} {'base':>7} {'learned':>8} {'E2dbl':>7} {'winner':>9}")
h2h = []
for k in FIRED:
    r = [x for x in rows if (x['record'], x['win']) == k][0]
    w = 'E2-double' if r['e2_double'] < r['learned'] else 'learned'
    h2h.append(dict(window=r['window'], base=r['base'], learned=r['learned'],
                    e2_double=r['e2_double'], winner=w,
                    learned_a=r['learned_a'], learned_b=r['learned_b'],
                    e2d_a=r['e2d_a'], e2d_b=r['e2d_b']))
    P(f"{r['window']:11s} {r['base']:7.4f} {r['learned']:8.4f} {r['e2_double']:7.4f} {w:>9}")
fa = sum(x['learned_a'] for x in h2h); fb = sum(x['learned_b'] for x in h2h)
ga = sum(x['e2d_a'] for x in h2h); gb = sum(x['e2d_b'] for x in h2h)
P(f'  learned   (a) {fa}/5  (b) {fb}/5   |  E2-double (a) {ga}/5  (b) {gb}/5')
P(f"  head-to-head wins: E2-double {sum(x['winner']=='E2-double' for x in h2h)}/5, "
  f"learned {sum(x['winner']=='learned' for x in h2h)}/5")
out['fired_subset'] = dict(rows=h2h, learned=dict(a=fa, b=fb, n=5),
                           e2_double=dict(a=ga, b=gb, n=5))

# ---- 6. own-collapse / paired diagnostics (declared diagnostic, not a bar) ----
P('\n== 6. DIAGNOSTIC (not a bar): paired learned vs E2-double over all 18 ==')
dl = np.array([r['base'] - r['learned'] for r in rows])
de = np.array([r['base'] - r['e2_double'] for r in rows])
wins_e2 = int(sum(r['e2_double'] < r['learned'] for r in rows))
P(f'  median reduction: learned {np.median(dl):+.4f}   E2-double {np.median(de):+.4f}')
P(f'  E2-double numerically better on {wins_e2}/18 windows')
P(f'  learned worse than baseline on {int(sum(r["learned"] > r["base"] for r in rows))}/18')
out['diagnostic'] = dict(median_reduction_learned=float(np.median(dl)),
                         median_reduction_e2double=float(np.median(de)),
                         e2double_better_n=wins_e2,
                         learned_worse_than_base=int(sum(r['learned'] > r['base'] for r in rows)))

txt = '\n'.join(lines)
open(os.path.join(D, 'analysis_learned.txt'), 'w').write(txt + '\n')
json.dump(out, open(os.path.join(D, 'results_learned.json'), 'w'), indent=1)
print(txt)
