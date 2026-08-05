import json
import numpy as np
from collections import defaultdict
from math import comb

def signp(k,n):
    kk=min(k,n-k); return min(1.0, 2*sum(comb(n,i) for i in range(kk+1))/2**n)

base=[json.loads(l) for l in open('tmp/s1s2_2026-08-05/results.jsonl')]
new =[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/CON_probe_results.jsonl')]
rows=[r for r in base+new if 'error' not in r]
c=defaultdict(list)
for r in rows: c[(r['arm'],r['episode'])].append(r)
eps=sorted(set(r['episode'] for r in base))
def m(a,e,k):
    v=[r[k] for r in c[(a,e)] if k in r]
    return float(np.median(v)) if v else float('nan')
def absvr(a,e): return abs(m(a,e,'Variance_Ratio')-1)

ARMS=['S2-T1f','X-T1f-notrack','C-phionly','C-steps1','C-lr50','C-lr25']
have=[a for a in ARMS if all(len(c[(a,e)])>=3 for e in eps)]
print('='*104)
print('CON probe C — knobs OUTSIDE the pre-registered K1-K5, all at X-BASE (oracle, k frozen)')
print('  complete arms:', have)
print('='*104)
print(f'{"arm":16s}{"nRMSE":>9s}{"init":>9s}{"rho":>9s}{"VR":>9s}{"|VR-1|":>9s}{"Hs_err":>9s}{"Skill":>9s}{"amp_err":>9s}')
for a in have:
    v=[np.mean([m(a,e,k) for e in eps]) for k in ['nRMSE','init_nrmse','Pearson_rho','Variance_Ratio']]
    av=np.mean([absvr(a,e) for e in eps])
    hs=np.mean([m(a,e,'Hs_error') for e in eps]); sk=np.mean([m(a,e,'Skill_Score') for e in eps])
    ae=np.mean([float(np.median([r['spec']['amp_rel_error_mean'] for r in c[(a,e)]])) for e in eps])
    print(f'{a:16s}{v[0]:9.4f}{v[1]:9.4f}{v[2]:9.4f}{v[3]:9.4f}{av:9.4f}{hs:9.4f}{sk:9.4f}{ae:9.4f}')

print()
print('paired vs X-BASE (S2-T1f), median of paired episode deltas, two-sided sign test, n=4 (floor p=0.125)')
print(f'{"arm":16s}{"d_nRMSE":>10s}{"k/n":>7s}{"p":>8s}{"d|VR-1|":>10s}{"k/n":>7s}{"d_rho":>10s}{"k/n":>7s}'
      f'{"% of tracker term removed":>28s}')
harm=np.mean([m('S2-T1f',e,'nRMSE')-m('X-T1f-notrack',e,'nRMSE') for e in eps])
for a in have:
    if a=='S2-T1f': continue
    d=np.array([m(a,e,'nRMSE')-m('S2-T1f',e,'nRMSE') for e in eps])
    dv=np.array([absvr(a,e)-absvr('S2-T1f',e) for e in eps])
    dr=np.array([m(a,e,'Pearson_rho')-m('S2-T1f',e,'Pearson_rho') for e in eps])
    kw=int((d>0).sum()); kv=int((dv>0).sum()); kr=int((dr>0).sum())
    print(f'{a:16s}{np.median(d):+10.4f}{kw:>4d}/4{signp(kw,4):8.4f}{np.median(dv):+10.4f}{kv:>4d}/4'
          f'{np.median(dr):+10.4f}{kr:>4d}/4{-np.median(d)/harm*100:>26.0f}%')
print(f'  (tracker term at X-BASE = {harm:+.4f}; MDE of this design = 0.0269 = 26% of it)')

print()
print('per-episode nRMSE:')
print(f'{"arm":16s}' + ''.join(f'{e[8:18]:>13s}' for e in eps))
for a in have:
    print(f'{a:16s}' + ''.join(f'{m(a,e,"nRMSE"):13.4f}' for e in eps))

print()
print('='*104)
print('COLLINEARITY: do all these knobs slide along ONE axis?  (PCA over arms x [rho, VR, amp_err])')
print('='*104)
X=[]
for a in have:
    X.append([np.mean([m(a,e,'Pearson_rho') for e in eps]),
              np.mean([m(a,e,'Variance_Ratio') for e in eps]),
              np.mean([float(np.median([r['spec']['amp_rel_error_mean'] for r in c[(a,e)]])) for e in eps])])
X=np.array(X); Xc=(X-X.mean(0))/(X.std(0)+1e-12)
s=np.linalg.svd(Xc,compute_uv=False)
print(f'  n arms = {len(have)};  PC1 variance share = {s[0]**2/np.sum(s**2)*100:.1f}%')
y=np.array([np.mean([m(a,e,'nRMSE') for e in eps]) for a in have])
for j,nm in enumerate(['rho','VR','amp_rel_error']):
    print(f'  r(nRMSE, {nm:14s}) = {np.corrcoef(y,X[:,j])[0,1]:+.4f}')
