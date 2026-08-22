#!/usr/bin/env python3
"""B-1 design diagnostic: re-derive S16/S17 numbers from raw jsonl.  READ-ONLY, 0 estimation runs."""
import json, os, numpy as np
from math import comb
R = '/home/vexit/wavesplat/wavesplat/tmp'
S13 = f'{R}/S13_oos_2026-08-12/results.jsonl'
S13B = f'{R}/S13b_seed_robust_2026-08-13/results_s13b.jsonl'
S16 = f'{R}/S16_BASELINE_2026-08-14/results.jsonl'
S17 = f'{R}/S17_WHITE25_2026-08-18/results.jsonl'
SH = ['0_5','5_10','10_20','20_30','30_50','50_75','75_100','100_150','150_inf']

def L(p): return [json.loads(l) for l in open(p)]

# ---- SHIP median-of-3 (old deploy config, S13 + S13b) ----
ship = {}
for r in L(S13):
    if r.get('arm') == 'SHIP' and 'error' not in r: ship.setdefault(r['episode'], {})[42] = r
for r in L(S13B):
    if 'error' in r or not str(r.get('arm','')).startswith('SHIP'): continue
    s = r.get('est_seed', r.get('effective_config',{}).get('seed'))
    if s in (43,44): ship.setdefault(r['episode'], {})[int(s)] = r
def med3(d, key=None):
    ss = sorted(d)
    if key is None: return float(np.median([d[s]['avg_nRMSE'] for s in ss]))
    v = [(d[s].get('shell_rmse') or {}).get(key, (d[s].get('shell_rmse') or {}).get('obs:'+key)) for s in ss]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')
S = {e: med3(d) for e,d in ship.items()}
Ssh = {e: {k: med3(d,k) for k in SH} for e,d in ship.items()}

# ---- baselines median-of-3 over filter seed ----
A, Ash, Z, Zsh, FRAC = {}, {}, {}, {}, {}
for src in (S16, S17):
    for r in L(src):
        if 'error' in r: continue
        a, e = r['arm'], r['episode']
        A.setdefault(a, {}).setdefault(e, []).append(r['avg_nRMSE'])
        Ash.setdefault(a, {}).setdefault(e, {})
        for k in SH:
            Ash[a][e].setdefault(k, []).append((r.get('shell_rmse') or {}).get('obs:'+k, (r.get('shell_rmse') or {}).get(k)))
        Z.setdefault(e, []).append(r['zero_nRMSE'])
        zs = r.get('zero_shell') or {}
        Zsh.setdefault(e, {})
        for k in SH: Zsh[e].setdefault(k, []).append(zs.get('obs:'+k, zs.get(k)))
        fr = r.get('shell_frac') or {}
        FRAC[e] = {k: fr.get('obs:'+k, fr.get(k)) for k in SH}
def m(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')
A = {a:{e:m(v) for e,v in d.items()} for a,d in A.items()}
Ash = {a:{e:{k:m(v) for k,v in kk.items()} for e,kk in d.items()} for a,d in Ash.items()}
Zs = {e:m(v) for e,v in Z.items()}
Zsh = {e:{k:m(v) for k,v in kk.items()} for e,kk in Zsh.items()}

# ---- strata ----
eps_all = sorted(set(S) & set(A['SPEC25']))
def ncomp(e):
    import re; mm = re.search(r'_(\d+)comp', e); return int(mm.group(1))
crowd = [e for e in eps_all if ncomp(e) >= 15]
sparse = [e for e in eps_all if ncomp(e) < 15]
print('eps', len(eps_all), 'crowded', len(crowd), 'sparse', len(sparse))
def p1(w,n): return sum(comb(n,k) for k in range(w,n+1))/2**n
def sgn(x,y,eps_):
    w = sum(x[e] < y[e] for e in eps_); t = sum(x[e]==y[e] for e in eps_)
    return w, t, len(eps_), p1(w,len(eps_))
def rel(x,y,eps_): return float(np.median([(x[e]-y[e])/y[e] for e in eps_]))

print('\n== reproduce S16 gates (crowded n=%d) ==' % len(crowd))
for lbl,x,y in [('R-2  SPEC50<WHITER50', A['SPEC50'], A['WHITER50']),
                ('R-1  SPEC25<ZERO',     A['SPEC25'], Zs),
                ('PRIM SHIP<SPEC25',     S,           A['SPEC25']),
                ('     SHIP<ZERO',       S,           Zs)]:
    w,t,n,p = sgn(x,y,crowd); print(f'  {lbl}: {w}/{n} ties={t} p={p:.2e} med(rel)={100*rel(x,y,crowd):+.2f}%')

print('\n== per-shell: does baseline SPEC25 have skill vs ZERO? (crowded) ==')
skill = []
for k in SH:
    x = {e:Ash['SPEC25'][e][k] for e in crowd}; z = {e:Zsh[e][k] for e in crowd}
    w,t,n,p = sgn(x,z,crowd)
    s = {e:Ssh[e][k] for e in crowd}
    w2,_,_,p2 = sgn(s,x,crowd)
    fr = float(np.median([FRAC[e][k] for e in crowd if FRAC[e][k] is not None]))
    print(f'  {k:9s} area={100*fr:5.1f}%  SPEC25<ZERO {w:2d}/{n} p={p:.1e} | SHIP<SPEC25 {w2:2d}/{n} p={p2:.1e} med(rel)={100*rel(s,x,crowd):+6.1f}%')
    if w >= 41: skill.append(k)
print('  skillful shells (>=41/64):', skill)

print('\n== area-weighted restriction to skillful shells (crowded) ==')
def wavg(d, eps_, keys):
    out={}
    for e in eps_:
        num=den=0.0
        for k in keys:
            f=FRAC[e][k]; v=d[e][k]
            if f is None or v is None or not np.isfinite(v): continue
            num+=f*v*v; den+=f
        out[e]=float(np.sqrt(num/den)) if den>0 else float('nan')
    return out
for keys,lbl in [(skill,'skillful'), (SH,'all 9')]:
    sS=wavg(Ssh,crowd,keys); sA=wavg(Ash['SPEC25'],crowd,keys); sZ=wavg(Zsh,crowd,keys)
    w,t,n,p=sgn(sS,sA,crowd); wz,_,_,pz=sgn(sA,sZ,crowd)
    print(f'  [{lbl}] SHIP<SPEC25 {w}/{n} p={p:.2e} med(rel)={100*rel(sS,sA,crowd):+.2f}% | SPEC25<ZERO {wz}/{n} p={pz:.1e}')

print('\n== sparse stratum (descriptive only) ==')
w,t,n,p = sgn(S,A['SPEC25'],sparse); print(f'  SHIP<SPEC25 {w}/{n} p={p:.2e} med(rel)={100*rel(S,A["SPEC25"],sparse):+.1f}%')

print('\n== arm means (crowded) ==')
for a in sorted(A): print(f'  {a:11s} {np.mean([A[a][e] for e in crowd]):.4f}')
print(f'  {"ZERO":11s} {np.mean([Zs[e] for e in crowd]):.4f}')
print(f'  {"SHIP(old)":11s} {np.mean([S[e] for e in crowd]):.4f}')
