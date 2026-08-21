#!/usr/bin/env python3
"""CON: the adversarial battery.  Each block is one attempt to kill a positive claim."""
import json, math, os, random, statistics as st
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
ROWS = [json.loads(l) for l in open(os.path.join(D, 'results.jsonl')) if l.strip()]
NULLS = {r['episode']: r['zero_null_tracking_nRMSE_mean']
         for r in json.load(open(os.path.join(D, 'nulls_full.json')))}
CROWDED = [f'episode_BF{b}_{c}comp_seed{s}_uniform.h5'
           for b in (4, 5) for c in (15, 30) for s in range(4)]
ALL_EP = sorted(set(r['episode'] for r in ROWS))
SPARSE = [e for e in ALL_EP if e not in CROWDED]
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK', 'T-PHIONLY', 'T-WONLY', 'T-EARLY20', 'T-K4REF']
BY = defaultdict(list)
for r in ROWS: BY[(r['cell'], r['episode'])].append(r)

def med(xs):
    xs = sorted(xs); n = len(xs)
    return float('nan') if n == 0 else (xs[n//2] if n % 2 else .5*(xs[n//2-1]+xs[n//2]))
def epmed(c, e, m='nRMSE'): return med([x[m] for x in BY[(c, e)]])
def signp(k, n):
    pm = lambda i: math.comb(n, i)/2.0**n
    o = pm(k); return min(1.0, sum(pm(i) for i in range(n+1) if pm(i) <= o*(1+1e-12)))
def deltas(c, ref, eps, m='nRMSE'): return [epmed(c, e, m)-epmed(ref, e, m) for e in eps]

EXC = [e for e in CROWDED if epmed('AUTO', e) >= NULLS[e]]
KEEP = [e for e in CROWDED if e not in EXC]
SKEEP = SPARSE[:]
line = lambda t: print('\n' + '=' * 96 + f'\n{t}\n' + '=' * 96)

# ------------------------------------------------------------------ A4: n is not 14
line('ATTACK 4 — episode independence.  Two crossed dependence axes, only one declared.')
print('measured in con_depends2/3.py (raw HDF5):')
print('  within a config, episode-seeds 0..3 share amplitude(col0), omega(col1), |k| (<=1.1e-16)')
print('    and gt_params/spectrum_1d bit-for-bit;  only direction theta and phase differ.')
print('  UNDECLARED second axis: at a fixed episode-seed, theta AND phase are IDENTICAL')
print('    across BF4/BF5 at the same component count (BF4_15 s0 theta == BF5_15 s0 theta).')
print('  => block C = 4 amplitude/omega/|k| draws  x  8 direction/phase draws, crossed.')
print()
CFG = {k: [e for e in KEEP if k in e] for k in ('BF4_15', 'BF4_30', 'BF5_15', 'BF5_30')}
SEEDG = {f'epseed{s}': [e for e in KEEP if f'seed{s}_' in e] for s in range(4)}
for name, groups in (('CONFIG margin (prereg n=4)', CFG), ('EPISODE-SEED margin (NOT in prereg)', SEEDG)):
    print(f'-- {name} --   sizes: ' + ', '.join(f'{k}:{len(v)}' for k, v in groups.items()))
    for c in ['AUTO-NOTRACK', 'T-PHIONLY', 'T-WONLY', 'T-EARLY20', 'T-K4REF']:
        ds = [med(deltas(c, 'AUTO', v)) for v in groups.values() if v]   # median of within-cluster deltas
        w = sum(1 for d in ds if d < 0)
        print(f'   {c:14s} medD={med(ds):+.4f}  {w}/{len(ds)}  p={signp(w,len(ds)):.4f}  ds={[round(d,4) for d in ds]}')
    print()
print('cluster-level sign-flip floor: 4 clusters -> min two-sided p = 2/2^4 = 0.125.')
print('=> NO cluster-robust analysis of block C can reach p<=0.05.  The n=14 p=0.0018 is')
print('   an episode-level p under an assumption of 14 independent draws that the raw data denies.')
print()
# intra-cluster correlation of the T-EARLY20 delta
for c in ['T-EARLY20', 'AUTO-NOTRACK', 'T-PHIONLY']:
    allds = {e: epmed(c, e)-epmed('AUTO', e) for e in KEEP}
    gm = st.mean(allds.values())
    for gname, groups in (('config', CFG), ('epseed', SEEDG)):
        within, between = 0.0, 0.0
        for v in groups.values():
            if len(v) < 2: continue
            m = st.mean(allds[e] for e in v)
            within += sum((allds[e]-m)**2 for e in v)
            between += len(v)*(m-gm)**2
        tot = within+between
        print(f'   ICC-ish({c},{gname}) = between/total = {between/tot:.3f}')
print()

# ------------------------------------------------------------------ A2: null exclusion
line('ATTACK 2 — did the null exclusion manufacture the conclusion?')
print(f'excluded: {[e.replace("episode_","").replace("_uniform.h5","") for e in EXC]}')
print('AUTO rank of the excluded episodes among the 16 (1 = worst nRMSE):')
order = sorted(CROWDED, key=lambda e: -epmed('AUTO', e))
for i, e in enumerate(order, 1):
    tag = ' <-- EXCLUDED' if e in EXC else ''
    print(f'   {i:2d}. {e.replace("episode_","").replace("_uniform.h5",""):22s} AUTO={epmed("AUTO",e):.4f} '
          f'null={NULLS[e]:.4f}{tag}')
print()
print('effect of the exclusion on each cell (medD vs AUTO, nRMSE):')
print(f'  {"cell":14s} {"n=16":>10s} {"w/n":>7s} | {"n=14":>10s} {"w/n":>7s}   sign flip?')
for c in ['AUTO-NOTRACK', 'T-PHIONLY', 'T-WONLY', 'T-EARLY20', 'T-K4REF']:
    d16, d14 = med(deltas(c, 'AUTO', CROWDED)), med(deltas(c, 'AUTO', KEEP))
    w16 = sum(1 for d in deltas(c, 'AUTO', CROWDED) if d < 0)
    w14 = sum(1 for d in deltas(c, 'AUTO', KEEP) if d < 0)
    print(f'  {c:14s} {d16:+10.4f} {w16:3d}/16  | {d14:+10.4f} {w14:3d}/14   '
          f'{"YES" if d16*d14 < 0 else "no"}')
print()
print('sensitivity: raise the null margin required to keep an episode')
for thr in (0.0, 0.01, 0.02, 0.03, 0.05):
    k = [e for e in CROWDED if NULLS[e]-epmed('AUTO', e) > thr]
    if len(k) < 4: continue
    row = []
    for c in ['AUTO-NOTRACK', 'T-EARLY20', 'T-PHIONLY']:
        ds = deltas(c, 'AUTO', k); w = sum(1 for d in ds if d < 0)
        row.append(f'{c}: {med(ds):+.4f} {w}/{len(k)} p={signp(w,len(k)):.4f}')
    print(f'  margin>{thr:.2f} -> n={len(k):2d}   ' + ' | '.join(row))
print()

# ------------------------------------------------------------------ A3: router
line('ATTACK 3 — AUTO is a router: are the cells comparing different estimator mixtures?')
for bn, eps in (('block C', CROWDED), ('block S', SPARSE)):
    cnt = defaultdict(int)
    for e in eps:
        for c in CELLS:
            for r in BY[(c, e)]: cnt[(c, r['auto_selected'])] += 1
    tot = defaultdict(int)
    for (c, s), n in cnt.items(): tot[s] += n
    print(f'  {bn}: pooled over all 7 cells -> {dict(tot)}')
    for c in CELLS:
        d = {s: cnt[(c, s)] for s in ('learned', 'v2_konly') if cnt[(c, s)]}
        print(f'     {c:14s} {d}')
print()
print('  => the 7 cells have IDENTICAL routing (sec.6.2 mismatches = 0, 72 groups x 5 fields).')
print('     So within a block the comparison is NOT confounded by the mixture.  ATTACK FAILS.')
print('  BUT the mixture is nearly degenerate in block C and dominated by v2_konly in block S:')
cC = sum(1 for e in CROWDED for r in BY[('AUTO', e)] if r['auto_selected'] == 'learned')
cS = sum(1 for e in SPARSE for r in BY[('AUTO', e)] if r['auto_selected'] == 'learned')
print(f'     block C AUTO runs learned {cC}/48 ({100*cC/48:.0f}%);  block S learned {cS}/24 ({100*cS/24:.0f}%)')
print('     => any CROSS-BLOCK statement (e.g. "T-K4REF wins in S but not in C") is confounded')
print('        by the initialiser arm, not only by regime.')
print()

# ------------------------------------------------------------------ A1: T-EARLY20
line('ATTACK 1 — T-EARLY20')
nfr = {}
for e in ALL_EP: nfr[e] = len(BY[('AUTO', e)][0]['vr_frames'])
print('tracking-window length per crowded episode:', sorted(set(nfr[e] for e in CROWDED)))
base = med(deltas('AUTO-NOTRACK', 'AUTO', KEEP))
obsR = med(deltas('T-EARLY20', 'AUTO', KEEP))/base
print(f'  medD(NOTRACK-AUTO) = {base:+.4f}   medD(EARLY20-AUTO) = {med(deltas("T-EARLY20","AUTO",KEEP)):+.4f}'
      f'   R = {100*obsR:.1f}%')
print()
print('(1a) the FLOOR is numerically better and EARLY20 never beats it:')
dN = deltas('T-EARLY20', 'AUTO-NOTRACK', KEEP)
w = sum(1 for d in dN if d < 0)
print(f'     medD(EARLY20 - NOTRACK) = {med(dN):+.4f}   EARLY20 wins {w}/14  p={signp(w,14):.4f}')
print(f'     i.e. the floor is better in {14-w}/14 episodes; the point estimate is {abs(med(dN))/abs(base)*100:.0f}%'
      f' of the whole tracker damage, in the floor\'s favour.')
print(f'     a deployer who wants the best number sets tracking_lr=0, not rx4_track_frames=20.')
print()
print('(1b) frame-fraction null: if damage accrued at a constant rate per tracked frame,')
print('     freezing after 20 of T frames would recover 1 - 20/T of the floor.')
for e in KEEP:
    T = nfr[e]
    pred = 1 - 20.0/T
    obs = (epmed('T-EARLY20', e)-epmed('AUTO', e))/(epmed('AUTO-NOTRACK', e)-epmed('AUTO', e)) \
        if (epmed('AUTO-NOTRACK', e)-epmed('AUTO', e)) != 0 else float('nan')
    print(f'     {e.replace("episode_","").replace("_uniform.h5",""):22s} T={T:3d}  predicted R={pred:.3f}  observed R={obs:+.3f}')
print(f'     median predicted R = {med([1-20.0/nfr[e] for e in KEEP]):.3f} ; observed pooled R = {obsR:.3f}')
print('     => EARLY20 recovers ~1/3 LESS than the constant-rate model predicts.')
print('        the first 20 frames (9-17% of the window) carry ~40% of the tracker damage.')
print('        that is FRONT-LOADING, which contradicts the "no jumps, constant slope" reading of')
print('        diag_analysis.txt sec.2 -- OR freezing does not stop degradation.  The data here')
print('        cannot separate those two.  Either way, "20 frames = a proportional dose of the')
print('        floor" is NOT what happened, so EARLY20 is not simply "off, weakly".')
print()
print('(1c) 20 is one pre-picked point.  Cells actually run:', sorted(set(r['cell'] for r in ROWS)))
print('     no other rx4_track_frames value exists in results.jsonl -> zero sweep evidence.')

# ------------------------------------------------------------------ A5: multiplicity
line('ATTACK 5 — multiplicity over the whole search volume actually looked at')
METRICS = ['nRMSE', 'Pearson_rho', 'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean']
NEG = {'nRMSE', 'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean'}
fam = []
for bn, eps in (('C', KEEP), ('S', SKEEP)):
    for c in CELLS:
        if c in ('AUTO', 'AUTO-DUP'): continue
        for m in METRICS:
            ds = deltas(c, 'AUTO', eps, m)
            w = sum(1 for d in ds if (d < 0 if m in NEG else d > 0))
            fam.append((signp(w, len(ds)), bn, c, m, med(ds), w, len(ds)))
fam.sort()
K = len(fam)
print(f'family = 2 blocks x 5 intervention cells x 6 metrics = {K} paired sign tests')
print(f'  {"rk":>3s} {"p":>8s} {"BHq":>8s} {"Bonf":>8s}  blk cell / metric / medD / w-n')
for i, (p, bn, c, m, d, w, n) in enumerate(fam[:14], 1):
    print(f'  {i:3d} {p:8.4f} {min(1,p*K/i):8.4f} {min(1,p*K):8.4f}   {bn}  {c:13s} {m:16s} {d:+.4f} {w}/{n}')
print('  ...')
for i, (p, bn, c, m, d, w, n) in enumerate(fam, 1):
    if m == 'nRMSE' and c in ('T-EARLY20', 'T-K4REF'):
        print(f'  ->  rank {i:2d}: block {bn} {c:11s} nRMSE p={p:.4f}  BHq={min(1,p*K/i):.4f}  Bonf={min(1,p*K):.4f}')
print()
print('  note: T-EARLY20 nRMSE and T-K4REF(S) nRMSE were pre-registered directional hypotheses')
print('  (H3 / block-S do-no-harm), so Bonferroni over the whole panel is the harsh reading,')
print('  BH is the fair one.  Both are reported.')

# ------------------------------------------------------------------ A7: T-K4REF in block S
line('ATTACK 7 — T-K4REF, block S, 8/8, p=0.0078, R=123.6%')
print('block S episode list and the distinct-wave-field clusters:')
FIELD = {}
for e in SPARSE:
    FIELD.setdefault(e.replace('_uniform.h5', '').replace('_slope.h5', ''), []).append(e)
for k, v in FIELD.items():
    print(f'   field {k.replace("episode_",""):24s} -> {len(v)} episodes: '
          f'{[x.split("seed0_")[-1].replace(".h5","") for x in v]}')
print()
print('  PREREG sec.3 excluded slope twins from block C precisely because the gt field is')
print('  bit-identical (verified here in raw HDF5: gt_wave_field, spectrum_1d, wave_components')
print('  all bit-identical for BF3_2comp and BF3_5comp uniform-vs-slope).')
print('  BLOCK S KEEPS BOTH TWINS.  So block S n=8 is 6 distinct wave fields, and the same')
print('  rule the prereg applied to block C, applied to block S, gives n=6.')
dK = {e: epmed('T-K4REF', e)-epmed('AUTO', e) for e in SPARSE}
print('  per-episode T-K4REF - AUTO on block S:')
for e in SPARSE:
    print(f'    {e.replace("episode_","").replace(".h5",""):32s} {dK[e]:+.5f}')
cl = [med([dK[e] for e in v]) for v in FIELD.values()]
w = sum(1 for d in cl if d < 0)
print(f'  field-clustered: medD={med(cl):+.5f}  {w}/{len(cl)}  p={signp(w,len(cl)):.4f}  '
      f'(floor p at n=6 = {signp(6,6):.4f})')
print()
dKN = deltas('T-K4REF', 'AUTO-NOTRACK', SPARSE)
wn = sum(1 for d in dKN if d < 0)
print(f'  vs the FLOOR: medD(T-K4REF - NOTRACK) = {med(dKN):+.6f}  wins {wn}/8  p={signp(wn,8):.4f}')
print('  => "exceeds the floor (R=123.6%)" is a ratio of two numbers whose paired test is p=1.0.')
print(f'  R is -0.0024/-0.0020; jackknife the ratio over the 8 episodes:')
Rj = []
for i in range(len(SPARSE)):
    sub = [e for j, e in enumerate(SPARSE) if j != i]
    b = med(deltas('AUTO-NOTRACK', 'AUTO', sub))
    Rj.append(med(deltas('T-K4REF', 'AUTO', sub))/b)
print('     jackknife R range = %.1f%% .. %.1f%%  (median %.1f%%)' % (100*min(Rj), 100*max(Rj), 100*med(Rj)))
print(f'  absolute size: block-S levels are AUTO {med([epmed("AUTO",e) for e in SPARSE]):.4f} vs '
      f'null {med([NULLS[e] for e in SPARSE]):.4f} -- {med([NULLS[e] for e in SPARSE])/med([epmed("AUTO",e) for e in SPARSE]):.0f}x below null.')
print('  PREREG sec.5: "block S is a do-no-harm check ONLY; it does not overturn PRIMARY."')

# ------------------------------------------------------------------ A6: D1 contamination
line('ATTACK 6 — D1 refuted: how far does the contamination reach?')
print('the pre-registered D1 statistic (sec.5) is |medD(PHIONLY-AUTO)| vs |medD(WONLY-AUTO)|.')
dP, dW = med(deltas('T-PHIONLY', 'AUTO', KEEP)), med(deltas('T-WONLY', 'AUTO', KEEP))
print(f'   medD PHIONLY={dP:+.4f}  WONLY={dW:+.4f}   |.|: {abs(dP):.4f} < {abs(dW):.4f} -> "REFUTED"')
print('   BUT both deltas are POSITIVE (both interventions are WORSE than AUTO).  With the')
print('   absolute value, the cell that did MORE DAMAGE wins the test.  The statistic is')
print('   sign-degenerate outside the case it was written for.')
print(f'   prereg sec.1.1 states D1 in words as "T-PHIONLY recovers MORE of the floor".')
print(f'   recovery: PHIONLY R={100*dP/base:+.1f}%   WONLY R={100*dW/base:+.1f}%  -> PHIONLY recovers MORE.')
print('   => sec.1.1 (words) says SUPPORTED, sec.5 (formula) says REFUTED.  Internal contradiction.')
print()
dh = deltas('T-PHIONLY', 'T-WONLY', KEEP)
wh = sum(1 for d in dh if d < 0)
print(f'   direct head-to-head paired: medD(PHIONLY - WONLY) = {med(dh):+.4f}, PHIONLY wins {wh}/14, '
      f'p={signp(wh,14):.4f}  (prereg needs 12/14)')
print()
print('   per-episode check on the 4 episodes the counterfactual was actually measured on')
print('   (diag used run-seeds 1521-1523; the intervention used 1524-1526 => out-of-sample in seed):')
CF = {'BF4_15comp_seed0': (0.0278, 0.0017), 'BF4_30comp_seed0': (0.0220, 0.0299),
      'BF5_15comp_seed0': (0.0278, 0.0074), 'BF5_30comp_seed0': (0.0470, 0.0035)}
agree = 0
for k, (dw, dp) in CF.items():
    e = f'episode_{k}_uniform.h5'
    pred = 'PHIONLY' if dw > dp else 'WONLY'
    dpp, dww = epmed('T-PHIONLY', e)-epmed('AUTO', e), epmed('T-WONLY', e)-epmed('AUTO', e)
    obs = 'PHIONLY' if dpp < dww else 'WONLY'
    ok = pred == obs; agree += ok
    print(f'     {k:20s} dmg_w={dw:.4f} dmg_phi={dp:.4f} -> predicts {pred:7s} | '
          f'PHIONLY{dpp:+.4f} WONLY{dww:+.4f} -> {obs:7s}  {"OK" if ok else "MISS"}')
print(f'   agreement {agree}/4  (sign-test floor at n=4 is p=0.125, so this cannot be "significant")')
print()
print('   what in diag_analysis.txt is a DIRECT MEASUREMENT (untouched by D1):')
print('     sec.1 dispersion violation 4.8e-07, ||dk||=0, |domega|=0   -- direct, survives')
print('     sec.3 decorr/VR/rho per cell                               -- direct, survives')
print('     sec.5 ||dw||/||w0||, rms dphi, sum|w| ratio                -- direct, survives')
print('     sec.6 r^2(dphi_rms, frame) = 0.954-0.979                   -- direct, survives')
print('   what is INFERENCE and is now in question:')
print('     sec.4 counterfactual channel split  -- the prereg itself warned it is a decomposition')
print('           on the realised joint trajectory, and it was measured on n=4 episodes, one of')
print('           which (BF4_30comp_seed0) already had dmg_phi > dmg_w.  The "84% amplitude"')
print('           headline is a median of 4 with a 1/4 reversal.')
print('     sec.2 "monotone accumulation, no jumps"  -- quantitatively undercut by attack 1b:')
print('           a constant slope predicts R(EARLY20)=0.91, observed 0.60.')

# ------------------------------------------------------------------ power / MDE
line('POWER — is T-PHIONLY a null or an underpowered miss?')
for c in ['T-PHIONLY', 'T-EARLY20', 'AUTO-NOTRACK']:
    ds = deltas(c, 'AUTO', KEEP)
    random.seed(7); bs = []
    for _ in range(20000):
        bs.append(med([random.choice(ds) for _ in ds]))
    bs.sort()
    print(f'  {c:14s} medD={med(ds):+.5f}  episode-bootstrap 95% CI = '
          f'[{bs[int(.025*len(bs))]:+.5f}, {bs[int(.975*len(bs))]:+.5f}]')
    # cluster bootstrap over the 4 config clusters
    random.seed(7); bs2 = []
    keys = list(CFG)
    for _ in range(20000):
        pick = [random.choice(keys) for _ in keys]
        v = [epmed(c, e)-epmed('AUTO', e) for k in pick for e in CFG[k]]
        bs2.append(med(v))
    bs2.sort()
    print(f'  {"":14s} config-CLUSTER bootstrap 95% CI = '
          f'[{bs2[int(.025*len(bs2))]:+.5f}, {bs2[int(.975*len(bs2))]:+.5f}]')
print()
print('  MDE: with n=14 and a paired sign test at alpha=0.05 you need 12/14.  T-PHIONLY got 5/14')
print('  -- the direction is not merely unresolved, it is on the wrong side of 50%.  The')
print('  bootstrap CI above brackets zero tightly relative to the floor effect, so this is a')
print('  measured NULL for the nRMSE endpoint, not a power failure.  (It is still power-limited')
print('  for any effect smaller than roughly the CI half-width.)')

# ------------------------------------------------------------------ final-init
line('bonus — does EARLY20 remove the initialisation damage, or something else?')
for c in ['AUTO', 'AUTO-NOTRACK', 'T-EARLY20', 'T-PHIONLY', 'T-WONLY', 'T-K4REF']:
    v = [epmed(c, e) - med([x['init_nrmse'] for x in BY[(c, e)]]) for e in KEEP]
    print(f'  {c:14s} med(final - init) on block C keep = {med(v):+.4f}')
print('  AUTO adds damage on top of Phase 1; NOTRACK is slightly negative; EARLY20 is ~0.')
print('  => EARLY20 = "stop before the tracker has eaten the initialisation", which is')
print('     the same mechanism B3 already reported, dosed at 20 frames.')
