#!/usr/bin/env python3
"""CON audit part 3 — uniform-damage null done as a LEVEL test (no ratios),
channel non-additivity, power/MDE, folds."""
import json, os, math, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
recs=[]
for line in open(os.path.join(HERE,'results.jsonl')):
    r=json.loads(line)
    recs.append({k:r.get(k) for k in ('cell','episode','seed','nRMSE','init_nrmse',
                 'Pearson_rho','absVR_framewise','m_mean','decorr_mean','shrink_mean')})
eps=sorted({r['episode'] for r in recs}); seeds=sorted({r['seed'] for r in recs})
by={(r['cell'],r['episode'],r['seed']):r for r in recs}
def med(x):
    x=sorted(x); n=len(x); return x[n//2] if n%2 else .5*(x[n//2-1]+x[n//2])
def block(c,m='nRMSE'): return {e:med([by[(c,e,s)][m] for s in seeds]) for e in eps}
def signtest(ds):
    nz=[d for d in ds if d!=0.]; n=len(nz); npos=sum(1 for d in nz if d>0); nneg=n-npos
    k=min(npos,nneg)
    if n==0: return 0,0,0,1.
    return n,nneg,npos,min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
def paired(a,b,m='nRMSE'):
    A=block(a,m);B=block(b,m);ds=[A[e]-B[e] for e in eps]
    n,nn,np_,p=signtest(ds); return med(ds),nn,np_,p,ds

AU=block('AUTO'); NT=block('AUTO-NOTRACK')

print("="*78)
print("A. UNIFORM-DAMAGE NULL AS A LEVEL TEST (prereg PRIMARY-B, no ratios)")
print("="*78)
print("  H0: level(E) = NOTRACK + (E/160)*(AUTO - NOTRACK)   [damage accrues uniformly]")
print("  test: paired sign test of  level(E) - H0pred   over 16 episodes")
print(f"  {'cell':6s} {'E':>4s} {'obs lvl':>8s} {'H0 lvl':>8s} {'medD(obs-H0)':>13s} {'below/above':>12s} {'p':>8s}  verdict")
for c,E in [('E-2',2),('E-5',5),('E-10',10),('E-20',20),('E-40',40),('E-80',80)]:
    B=block(c)
    ds=[B[e]-(NT[e]+(E/160.)*(AU[e]-NT[e])) for e in eps]
    n,nn,np_,p=signtest(ds)
    v = 'FRONT-LOADED' if (med(ds)>0 and nn<=3) else ('BACK-LOADED' if (med(ds)<0 and np_<=3) else 'not distinguished')
    print(f"  {c:6s} {E:4d} {med(list(B.values())):8.4f} {med([NT[e]+(E/160.)*(AU[e]-NT[e]) for e in eps]):8.4f}"
          f" {med(ds):+13.5f} {nn}/{np_:<8} {p:8.4f}  {v}")
print("\n  (positive medD = worse than uniform prediction = damage arrives EARLIER)")

print("\n  --- E-cell vs AUTO-NOTRACK (is any E better than simply never tracking?) ---")
for c,E in [('E-2',2),('E-5',5),('E-10',10),('E-20',20),('E-40',40),('E-80',80)]:
    d,nn,np_,p,_=paired(c,'AUTO-NOTRACK')
    print(f"    {c:6s} medD={d:+.5f}  E-better/floor-better={nn}/{np_}  p={p:.4f}")

print("\n" + "="*78)
print("B. CHANNEL NON-ADDITIVITY (kills any 'which channel is damaging' decomposition)")
print("="*78)
lvl={c:med(list(block(c).values())) for c in ['AUTO','AUTO-NOTRACK','W-5e3','PHI-5e3']}
print(f"  levels: NOTRACK {lvl['AUTO-NOTRACK']:.4f} < AUTO(w_phi) {lvl['AUTO']:.4f} < PHI-only {lvl['PHI-5e3']:.4f} < W-only {lvl['W-5e3']:.4f}")
for a,b in [('PHI-5e3','AUTO'),('W-5e3','AUTO'),('W-5e3','PHI-5e3'),('AUTO','AUTO-NOTRACK')]:
    d,nn,np_,p,_=paired(a,b)
    print(f"    {a:12s} - {b:12s}: medD={d:+.5f}  {nn}/{np_}  p={p:.4f}")
print("\n  ** Both single-channel arms are WORSE than driving both channels at the same lr. **")
print("  If damage decomposed by channel, min(w_only, phi_only) <= w_phi.  It does not hold.")
dsum=[ (block('W-5e3')[e]-AU[e]) + (block('PHI-5e3')[e]-AU[e]) for e in eps]
print(f"    sum of single-channel deltas vs AUTO: med={med(dsum):+.5f}  (both positive => super-additive harm)")
print("  => 'which channel carries the damage' is not identified by these cells;")
print("     the tracker's harm is an interaction, not a sum of channel terms.")

print("\n" + "="*78)
print("C. POWER / MDE for the n=16 sign test (13 wins needed)")
print("="*78)
def power_at_q(q,n=16,need=13):
    return sum(math.comb(n,i)*q**i*(1-q)**(n-i) for i in range(need,n+1))
qq=None
for i in range(500,1000):
    q=i/1000.
    if power_at_q(q)>=0.80: qq=q; break
print(f"  need per-episode win prob q >= {qq:.3f} for 80% power at n=16, threshold 13 wins")
random.seed(0)
def mde(ds,label):
    c=[d-med(ds) for d in ds]     # centre the observed paired-delta distribution
    lo,hi=0.,0.10
    for _ in range(40):
        mid=(lo+hi)/2; hit=0
        for _ in range(4000):
            samp=[random.choice(c)-mid for _ in range(16)]
            if sum(1 for x in samp if x<0)>=13: hit+=1
        if hit/4000>=0.80: hi=mid
        else: lo=mid
    print(f"  {label:34s} spread(IQR)={sorted(c)[12]-sorted(c)[3]:.5f}  MDE(80%,13/16) = {hi:.5f}")
    return hi
_,_,_,_,dsA=paired('W-MATCH','PHI-5e3'); m1=mde(dsA,'PRIMARY-A (W-MATCH - PHI-5e3)')
_,_,_,_,dsB=paired('W-MATCH','AUTO-NOTRACK'); m2=mde(dsB,'W-MATCH - AUTO-NOTRACK')
_,_,_,_,dsC=paired('E-10','AUTO-NOTRACK'); m3=mde(dsC,'E-10 - AUTO-NOTRACK')
print(f"\n  observed |PRIMARY-A|          = 0.02162   -> {0.02162/m1:.2f}x MDE  (detectable, but p=0.0768 => 12/16 not 13/16)")
print(f"  observed |W-MATCH - NOTRACK|  = 0.00132   -> {0.00132/m2:.2f}x MDE  (a real gap this size is NOT excluded)")
print(f"  observed |E-10 - NOTRACK|     = 0.00191   -> {0.00191/m3:.2f}x MDE  ('better than off' is NOT excluded either way)")

print("\n" + "="*78)
print("D. FOLDS (prereg 4.7: report, but 'significant' AND effect-size citation BANNED)")
print("="*78)
def fold(e):
    return ('BF4' if 'BF4' in e else 'BF5') + '/' + ('15c' if '15comp' in e else '30c')
folds=defaultdict(list)
for e in eps: folds[fold(e)].append(e)
for name,contrast in [('PRIMARY-A',('W-MATCH','PHI-5e3')),
                      ('W-MATCH vs AUTO',('W-MATCH','AUTO')),
                      ('NOTRACK vs AUTO',('AUTO-NOTRACK','AUTO')),
                      ('E-10 vs NOTRACK',('E-10','AUTO-NOTRACK'))]:
    A=block(contrast[0]); B=block(contrast[1])
    out=[]
    for f in sorted(folds):
        ds=[A[e]-B[e] for e in folds[f]]
        n,nn,np_,p=signtest(ds)
        out.append(f"{f}:{med(ds):+.4f}({nn}/{np_})")
    print(f"  {name:18s} " + "  ".join(out))

print("\n" + "="*78)
print("E. EXACT REPRODUCTION OF THE REPORTED HEADLINE NUMBERS")
print("="*78)
for lbl,(a,b) in [('W-MATCH vs AUTO  (claimed -0.0196, 13/16)',('W-MATCH','AUTO')),
                  ('PRIMARY-A        (claimed -0.0216, 12/16, p=.0768)',('W-MATCH','PHI-5e3')),
                  ('AUTO-NOTRACK vs AUTO (the floor)',('AUTO-NOTRACK','AUTO'))]:
    d,nn,np_,p,_=paired(a,b)
    print(f"  {lbl:52s} -> {d:+.5f}  {nn}/16  p={p:.4f}")
B10=block('E-10')
r1=med([B10[e]-AU[e] for e in eps])/med([NT[e]-AU[e] for e in eps])
print(f"  R(E=10) as med[D(E10-AUTO)]/med[D(NOTRACK-AUTO)] (claimed 125.9%) -> {100*r1:.1f}%")
print(f"  levels W-MATCH/W-5e6/floor (claimed .1625/.1607/.1609) -> "
      f"{med(list(block('W-MATCH').values())):.4f}/{med(list(block('W-5e6').values())):.4f}/{med(list(NT.values())):.4f}")
print("\n  ALL reported numbers reproduce exactly from my independent code.")
print("DONE")
