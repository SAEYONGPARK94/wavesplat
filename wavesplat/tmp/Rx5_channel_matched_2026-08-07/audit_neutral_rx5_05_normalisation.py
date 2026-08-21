"""NEUTRAL audit 05 -- is the sec.3 'matching' rule the right one?

The PREREG matches RELATIVE STEP:
    phase : lr / (2*pi)        <- normalised by a PERIOD
    ampl. : lr / median|w|     <- normalised by a MAGNITUDE
Those are different kinds of normaliser.  A first-order FIELD-SENSITIVITY match is
derivable from the model itself:  eta = sum_i w_i cos(k_i.x - omega_i t + phi_i)
    d eta / d w_i    = cos(...)            -> field change per unit dw   ~ 1
    d eta / d phi_i  = -w_i sin(...)       -> field change per unit dphi ~ w_i
so equal field perturbation requires   lr_w = median|w| * lr_phi   (radians, no 2*pi).
This script reports both, and locates the lr_w that empirically reproduces the phase
channel's damage.
"""
import json, os, statistics as st
from math import comb, log10

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
EPS = sorted({r['episode'] for r in recs})
CELLS = sorted({r['cell'] for r in recs})
M = {c: {e: st.median([r['nRMSE'] for r in recs if r['cell'] == c and r['episode'] == e])
         for e in EPS} for c in CELLS}
ws = json.load(open(os.path.join(HERE, 'wscale.json')))
MEDW = ws['median_abs_w']

def signtest(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0); n = pos + neg
    if n == 0: return n, neg, pos, 1.0
    k = max(pos, neg)
    return n, neg, pos, min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def paired(a, b, label):
    d = [M[a][e] - M[b][e] for e in EPS]
    n, neg, pos, p = signtest(d)
    print(f'  {label:46s} median={st.median(d):+.5f}  wins {max(pos,neg)}/{n} '
          f'(thr 13)  p={p:.5f}   {"MEETS" if max(pos,neg)>=13 else "short"}')
    return st.median(d)

print('=' * 88)
print('1. THE TWO CANDIDATE MATCHING RULES')
print('=' * 88)
lr_phi = 0.005
print(f'  median|w| (frozen, Phase-1 end)                = {MEDW:.6f} m')
print(f'  PREREG rule   lr_w = (lr_phi/2pi) * median|w|  = {ws["RX5_LR_W_MATCHED"]:.6e}   '
      f'<- the W-MATCH cell')
fm = lr_phi * MEDW
print(f'  field-sens.   lr_w =  lr_phi      * median|w|  = {fm:.6e}')
print(f'  ratio between them = {fm / ws["RX5_LR_W_MATCHED"]:.4f}   (= 2*pi exactly: '
      f'{fm / ws["RX5_LR_W_MATCHED"] / (2*3.141592653589793):.6f} x 2pi)')
print('  -> the PREREG rule divides the PHASE channel by a period (2pi) but the')
print('     AMPLITUDE channel by a magnitude.  The field-matched lr_w is 2pi LARGER.')
print(f'  The sweep BRACKETS the field-matched value but never ran it:')
print(f'     W-5e4 = 5.0e-04   >  field-matched {fm:.2e}  >  W-MATCH = 3.51e-05')

print('\n' + '=' * 88)
print('2. RESIDUAL TRACKING DAMAGE PER CELL  (cell - AUTO-NOTRACK), as % of total D')
print('=' * 88)
D = st.median([M['AUTO'][e] - M['AUTO-NOTRACK'][e] for e in EPS])
print(f'  total tracking damage D = median(AUTO - AUTO-NOTRACK) = {D:+.5f}\n')
print(f"  {'cell':12s} {'lr_w':>10} {'rel.step(PREREG)':>17} {'residual damage':>16} {'% of D':>8} {'wins/16':>8}")
rows = []
for c, lr in [('W-5e3', 5e-3), ('W-5e4', 5e-4), ('W-MATCH', ws['RX5_LR_W_MATCHED']),
              ('W-5e6', 5e-6)]:
    d = [M[c][e] - M['AUTO-NOTRACK'][e] for e in EPS]
    n, neg, pos, p = signtest(d)
    med = st.median(d)
    rows.append((lr, med))
    print(f'  {c:12s} {lr:10.2e} {lr/MEDW*100:16.4f}% {med:+16.5f} {med/D*100:7.1f}% '
          f'{max(pos,neg):5d}/16')
for c in ['PHI-5e3', 'AUTO']:
    d = [M[c][e] - M['AUTO-NOTRACK'][e] for e in EPS]
    n, neg, pos, p = signtest(d)
    med = st.median(d)
    print(f'  {c:12s} {"-":>10} {"(0.0796%)" if c=="PHI-5e3" else "-":>17} {med:+16.5f} '
          f'{med/D*100:7.1f}% {max(pos,neg):5d}/16')

print('\n  >> W-MATCH retains only ~6.5% of the tracking damage: it has essentially')
print('     COLLAPSED ONTO AUTO-NOTRACK.  PHI-5e3 retains ~110%.')

print('\n' + '=' * 88)
print('3. WHICH lr_w REPRODUCES THE PHASE CHANNEL\'S DAMAGE?  (log-linear interpolation)')
print('=' * 88)
target = st.median([M['PHI-5e3'][e] - M['AUTO-NOTRACK'][e] for e in EPS])
print(f'  phase-channel residual damage (target) = {target:+.5f}')
rows.sort()
sol = None
for (l1, d1), (l2, d2) in zip(rows, rows[1:]):
    if (d1 - target) * (d2 - target) <= 0 and d1 != d2:
        f = (target - d1) / (d2 - d1)
        sol = 10 ** (log10(l1) + f * (log10(l2) - log10(l1)))
        break
print(f'  lr_w that matches it empirically  ~= {sol:.3e}' if sol else '  no bracket')
if sol:
    print(f'  field-sensitivity prediction         = {fm:.3e}   '
          f'(ratio {sol/fm:.2f}x)')
    print(f'  PREREG W-MATCH value                 = {ws["RX5_LR_W_MATCHED"]:.3e}   '
          f'(ratio {sol/ws["RX5_LR_W_MATCHED"]:.1f}x)')
    print('\n  -> The empirically damage-matched lr_w agrees with the FIELD-SENSITIVITY')
    print('     rule to within a factor ~2, and is ~13x the PREREG W-MATCH value.')

print('\n' + '=' * 88)
print('4. THE VERDICT IS NORMALISATION-DEPENDENT')
print('=' * 88)
paired('W-MATCH', 'PHI-5e3', 'PREREG matching   : W-MATCH vs PHI-5e3')
paired('W-5e4', 'PHI-5e3', 'nearest to field-match: W-5e4 vs PHI-5e3')
paired('W-5e3', 'PHI-5e3', 'shipped lr both   : W-5e3 vs PHI-5e3 (= Rx-4 contrast)')
print('\n  -> With the PREREG normaliser the amplitude arm looks ROBUST (negative median).')
print('     One notch up the dose curve (W-5e4, closest cell to the field-matched lr)')
print('     the two channels are INDISTINGUISHABLE.  The sign of PRIMARY-A is a')
print('     property of the chosen normaliser, not only of the system.')
