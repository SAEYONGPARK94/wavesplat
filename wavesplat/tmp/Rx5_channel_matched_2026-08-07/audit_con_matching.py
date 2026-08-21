#!/usr/bin/env python3
"""CON audit part 2 — (a) the matching constant is convention-dependent,
(b) R(E) definitions, (c) multiplicity, (d) K6 seed comparison."""
import json, os, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
K6   = '/home/vexit/wavesplat/wavesplat/tmp/K6_run_2026-08-07'

def load(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append({k: r.get(k) for k in
                    ('cell','episode','seed','nRMSE','init_nrmse','Pearson_rho',
                     'absVR_framewise','m_mean','decorr_mean','shrink_mean','N')})
    return out

recs = load(os.path.join(HERE,'results.jsonl'))
eps   = sorted({r['episode'] for r in recs})
seeds = sorted({r['seed'] for r in recs})
cells = sorted({r['cell'] for r in recs})
by = {(r['cell'],r['episode'],r['seed']): r for r in recs}

def med(xs):
    xs=sorted(xs); n=len(xs)
    return xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])
def block(c,m='nRMSE'):
    return {e: med([by[(c,e,s)][m] for s in seeds]) for e in eps}
def signtest(ds):
    nz=[d for d in ds if d!=0.0]; n=len(nz)
    npos=sum(1 for d in nz if d>0); nneg=n-npos; k=min(npos,nneg)
    if n==0: return 0,0,0,1.0
    p=min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/2**n)
    return n,nneg,npos,p
def paired(a,b,m='nRMSE'):
    A=block(a,m); B=block(b,m); ds=[A[e]-B[e] for e in eps]
    n,nn,np_,p=signtest(ds)
    return med(ds), nn, np_, p, ds

print("="*78)
print("A. THE MATCHING CONSTANT IS CONVENTION-DEPENDENT")
print("="*78)
MEDW = 0.044095
print(f"  model:  z = SUM_i w_i * cos(k_i.x - w_i t + phi_i)   (spectral_gaussian.py:127)")
print(f"  dz/dw_i  = cos(.)          -> |dz| = |dw|")
print(f"  dz/dphi_i= -w_i sin(.)     -> |dz| = |w_i|*|dphi|")
print(f"  Adam step magnitude ~= lr for each parameter.")
print()
print(f"  PREREG matching  : lr_w = lr_phi/(2*pi) * median|w| = {0.005/(2*math.pi)*MEDW:.6e}   <- used (W-MATCH)")
print(f"  FIELD  matching  : lr_w = lr_phi        * median|w| = {0.005*MEDW:.6e}   <- 2*pi = 6.283x LARGER")
LR_FIELD = 0.005*MEDW
print(f"  ratio = {LR_FIELD/(0.005/(2*math.pi)*MEDW):.4f}  (exactly 2*pi)")
print()
print("  The prereg's phase 'natural scale' is 2*pi rad (the PERIOD).  But the field")
print("  responds LINEARLY to phi with gain |w| per RADIAN, not per 2*pi.  Normalising")
print("  phase by its period silently under-drives the amplitude arm by exactly 2*pi.")
print()
print("  Is the field-matched point INSIDE the measured sweep?")
LRS = {'W-5e3':5e-3,'W-5e4':5e-4,'W-MATCH':3.508993677838184e-05,'W-5e6':5e-6}
print(f"    W-5e4  lr={LRS['W-5e4']:.3e}  >  FIELD {LR_FIELD:.3e}  >  W-MATCH lr={LRS['W-MATCH']:.3e}   YES, bracketed")
print()
print("  Channel contrast (W-cell minus PHI-5e3) as a function of lr:")
rows=[]
for c in ['W-5e3','W-5e4','W-MATCH','W-5e6']:
    d,nn,np_,p,_ = paired(c,'PHI-5e3')
    rows.append((LRS[c],d,nn,np_,p,c))
    print(f"    {c:8s} lr={LRS[c]:.3e}  medD={d:+.5f}  W-better/PHI-better={nn}/{np_}  p={p:.4f}")
# log-linear interpolation between W-5e4 and W-MATCH
(l1,d1)=(LRS['W-5e4'], [r[1] for r in rows if r[5]=='W-5e4'][0])
(l2,d2)=(LRS['W-MATCH'],[r[1] for r in rows if r[5]=='W-MATCH'][0])
f = (math.log10(l1)-math.log10(LR_FIELD))/(math.log10(l1)-math.log10(l2))
d_field = d1 + f*(d2-d1)
lr_cross = 10**(math.log10(l1) - (d1/(d1-d2))*(math.log10(l1)-math.log10(l2)))
print()
print(f"  log-linear interpolation at the FIELD-matched lr {LR_FIELD:.3e}:")
print(f"    estimated channel contrast = {d_field:+.5f}   (vs reported {d2:+.5f} => {abs(d2/d_field):.1f}x smaller)")
print(f"  sign-change (crossover) lr where W-cell == PHI-5e3 : {lr_cross:.3e}")
print(f"    field-matched lr / crossover lr = {LR_FIELD/lr_cross:.3f}   (1.0 == contrast is exactly null at field matching)")
print()
print("  => The verdict INVERTED vs NO-DIFFERENCE vs CHANNEL-REAL is decided by whether you")
print("     divide the phase scale by 2*pi.  The data DO NOT determine the sign at the")
print("     physically-matched point; the crossover sits inside the 6.3x window.")

# also level-based
print("\n  (same conclusion on block levels)")
for c in ['W-5e3','W-5e4','W-MATCH','W-5e6','PHI-5e3','AUTO-NOTRACK','AUTO']:
    b=block(c); print(f"    {c:14s} block-median nRMSE = {med(list(b.values())):.4f}")

print("\n" + "="*78)
print("B. R(E) UNDER FOUR DEFINITIONS")
print("="*78)
AU=block('AUTO'); NT=block('AUTO-NOTRACK')
den_paired = med([NT[e]-AU[e] for e in eps])
print(f"  denominator med[Delta(NOTRACK-AUTO)] = {den_paired:+.5f}")
print(f"{'cell':6s} {'E':>4s} {'R=medD/medD':>12s} {'R=block-lvl':>12s} {'R=med(perEp)':>13s} {'R=mean(perEp)':>14s} | {'E vs NOTRACK paired':>22s}")
ECELLS=[('E-2',2),('E-5',5),('E-10',10),('E-20',20),('E-40',40),('E-80',80)]
for c,E in ECELLS:
    B=block(c)
    d_c,_,_,_,_ = paired(c,'AUTO')
    R1 = d_c/den_paired
    R2 = (med(list(AU.values()))-med(list(B.values())))/(med(list(AU.values()))-med(list(NT.values())))
    per=[(AU[e]-B[e])/(AU[e]-NT[e]) for e in eps if abs(AU[e]-NT[e])>1e-12]
    R3 = med(per); R4 = sum(per)/len(per)
    dn,nn,np_,p,_ = paired(c,'AUTO-NOTRACK')
    print(f"{c:6s} {E:4d} {100*R1:11.1f}% {100*R2:11.1f}% {100*R3:12.1f}% {100*R4:13.1f}% |"
          f" medD={dn:+.5f} {nn}/{np_} p={p:.3f}")
print()
print("  R1 (ratio of two medians of paired deltas) is what gives 125.9% at E=10.")
print("  It is a RATIO OF AGGREGATES -- the same class of object PREREG 4.3 bans.")
print("  Per-episode ratios (R3/R4) do not reproduce it, and the paired test of")
print("  E-10 vs AUTO-NOTRACK is the only thing that can support 'better than off'.")

print("\n  --- what happens if a single episode is dropped (LOO on R1, E=10) ---")
B10=block('E-10')
vals=[]
for drop in eps:
    sub=[e for e in eps if e!=drop]
    num=med([B10[e]-AU[e] for e in sub]); dn2=med([NT[e]-AU[e] for e in sub])
    vals.append((100*num/dn2, drop))
vals.sort()
print(f"    LOO range of R1(E=10): {vals[0][0]:.1f}% .. {vals[-1][0]:.1f}%")
for v,d in vals[:3]+vals[-3:]:
    print(f"      {v:8.1f}%  (dropped {d[:44]})")

print("\n" + "="*78)
print("C. MULTIPLICITY")
print("="*78)
METRICS=['nRMSE','init_nrmse','Pearson_rho','absVR_framewise','m_mean','decorr_mean','shrink_mean']
fam=[]
for c in cells:
    if c in ('AUTO','AUTO-DUP'): continue
    for m in METRICS:
        d,nn,np_,p,_ = paired(c,'AUTO',m)
        fam.append((p,f"{c} vs AUTO [{m}]",d,nn,np_))
# add the prespecified / reported contrasts
extra=[('W-MATCH','PHI-5e3','nRMSE'),('W-5e3','PHI-5e3','nRMSE'),('W-5e4','PHI-5e3','nRMSE'),
       ('W-5e6','PHI-5e3','nRMSE'),('W-MATCH','AUTO-NOTRACK','nRMSE'),
       ('W-5e4','W-5e3','nRMSE'),('W-MATCH','W-5e3','nRMSE'),('W-5e6','W-5e3','nRMSE')]
for a,b,m in extra:
    d,nn,np_,p,_ = paired(a,b,m)
    fam.append((p,f"{a} vs {b} [{m}]",d,nn,np_))
fam.sort()
N=len(fam)
print(f"  family size N = {N} (13 cells x 7 metrics vs AUTO, + 8 reported pairwise)")
print(f"  Bonferroni alpha = {0.05/N:.5f}; sign-test floor at n=16 is p=3.05e-05")
print(f"  => at n=16 a test can only clear Bonferroni if it is 16/16 or 15/16.")
# BH
surv_bh=[]
for i,(p,name,d,nn,np_) in enumerate(fam,1):
    thr = 0.05*i/N
    if p<=thr: surv_bh.append((p,name,d,nn,np_,thr))
kmax = max([i for i,(p,_,_,_,_) in enumerate(fam,1) if p<=0.05*i/N], default=0)
print(f"\n  {'p':>10s}  {'BH thr':>9s}  {'medD':>9s}  {'w/l':>7s}  test")
for i,(p,name,d,nn,np_) in enumerate(fam,1):
    if i<=max(kmax,0)+6:
        tag = 'BH-PASS' if i<=kmax else '       '
        bon = 'BONF' if p<=0.05/N else '    '
        print(f"  {p:10.2e}  {0.05*i/N:9.2e}  {d:+9.5f}  {nn}/{np_:<4} {tag} {bon}  {name}")
print(f"\n  BH(0.05) survivors: {kmax} of {N}")
print(f"  Bonferroni survivors: {sum(1 for p,_,_,_,_ in fam if p<=0.05/N)}")
print("\n  PRIMARY-A (W-MATCH vs PHI-5e3 [nRMSE], p=0.0768) rank:",
      [i for i,(p,n_,_,_,_) in enumerate(fam,1) if n_.startswith('W-MATCH vs PHI-5e3')])

print("\n" + "="*78)
print("D. SEED / DATASET COMPARISON vs K6")
print("="*78)
k6 = load(os.path.join(K6,'results.jsonl'))
k6by = defaultdict(list)
for r in k6:
    k6by[(r['cell'],r['episode'],r['seed'])] = r
k6cells = sorted({r['cell'] for r in k6}); k6seeds=sorted({r['seed'] for r in k6})
k6eps = sorted({r['episode'] for r in k6})
print(f"  K6 cells={k6cells} seeds={k6seeds} eps={len(k6eps)}")
print(f"  episode sets identical: {set(k6eps)==set(eps)}")
k6auto = {e: med([k6by[('AUTO',e,s)]['nRMSE'] for s in k6seeds]) for e in k6eps}
rx5auto = block('AUTO')
print(f"  K6  AUTO block-median (16 ep, seeds 1527-29) = {med(list(k6auto.values())):.6f}")
print(f"  Rx5 AUTO block-median (16 ep, seeds 1530-32) = {med(list(rx5auto.values())):.6f}")
ds=[rx5auto[e]-k6auto[e] for e in eps]
n,nn,np_,p = signtest(ds)
print(f"  paired Delta(Rx5.AUTO - K6.AUTO): med={med(ds):+.6f}  {nn}/{np_}  p={p:.4f}")
print(f"    per-episode |Delta| : min={min(abs(d) for d in ds):.5f} med={med([abs(d) for d in ds]):.5f} max={max(abs(d) for d in ds):.5f}")
print("  => block medians agree to 1e-4 ONLY because the block median is a rank statistic;")
print("     the per-episode disagreement is an order of magnitude larger.")
# how many episodes bit-identical?
ident=sum(1 for e in eps if abs(rx5auto[e]-k6auto[e])<1e-12)
print(f"    bit-identical episodes: {ident}/16 (0 expected -- different seeds)")
print("\n  Six-seed pooled AUTO spread (1527..1532), per episode:")
allseed = {}
for e in eps:
    v=[k6by[('AUTO',e,s)]['nRMSE'] for s in k6seeds]+[by[('AUTO',e,s)]['nRMSE'] for s in seeds]
    allseed[e]=v
sp=[max(v)-min(v) for v in allseed.values()]
print(f"    per-episode 6-seed range: med={med(sp):.5f} max={max(sp):.5f}")
print(f"    headline |PRIMARY-A| = 0.02162  ->  median 6-seed episode range / headline = {med(sp)/0.02162:.2f}x")
# block-level bootstrap over seed triples
import itertools
allseeds=k6seeds+seeds
trip=[t for t in itertools.combinations(allseeds,3)]
lv=[]
for t in trip:
    vals=[]
    for e in eps:
        v=[]
        for s in t:
            v.append((k6by[('AUTO',e,s)] if s in k6seeds else by[('AUTO',e,s)])['nRMSE'])
        vals.append(med(v))
    lv.append(med(vals))
print(f"\n  AUTO block level over all C(6,3)={len(trip)} seed-triples: min={min(lv):.4f} med={med(lv):.4f} max={max(lv):.4f}")
print(f"    spread = {max(lv)-min(lv):.4f}  vs |PRIMARY-A| = 0.02162  ->  {(max(lv)-min(lv))/0.02162:.2f}x")
print("DONE")
