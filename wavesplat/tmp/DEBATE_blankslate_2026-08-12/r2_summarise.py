#!/usr/bin/env python3
"""R2/P3 rolling summary of the three R2 jobs."""
import os, json, numpy as np
from scipy import stats
R='/home/vexit/wavesplat/wavesplat'
H=os.path.join(R,'tmp','DEBATE_blankslate_2026-08-12')
FM=json.load(open(os.path.join(R,'tmp','BLANKSLATE_2026-08-12','letkf_framematch.json')))['per_episode']
SD=json.load(open(os.path.join(R,'tmp','BLANKSLATE_2026-08-12','shipping_default.json')))
def L(f):
    p=os.path.join(H,f); return json.load(open(p)) if os.path.exists(p) else {}
PH=L('r2_phase_attrib.json'); A70=L('r2_auto_rest70.json')
def st(w,n): return stats.binomtest(w,n,0.5).pvalue if n else float('nan')

print('=== C. TRACK_PHASE_PREDICT attribution (HEAD code, flag forced True) ===')
ks=[k.split('|')[0] for k in PH if k.endswith('|v2k_phaseTrue') and 'avg_nRMSE' in PH[k]]
if ks:
    pub=np.array([FM[k]['ws'] for k in ks]); v2kT=np.array([PH[f'{k}|v2k_phaseTrue']['avg_nRMSE'] for k in ks])
    v2kH=np.array([SD[f'{k}|v2_konly_100ep']['avg_nRMSE'] for k in ks])
    autH=np.array([SD[f'{k}|shipping_default']['avg_nRMSE'] for k in ks])
    lk=np.array([FM[k]['letkf_match'] for k in ks])
    print(f'n={len(ks)}')
    print(f'  published (06-24, v2k+flagTrue)  median {np.median(pub):.4f}')
    print(f'  HEAD v2k + flag TRUE             median {np.median(v2kT):.4f}   med|rel diff to published| {100*np.median(np.abs(v2kT/pub-1)):.1f}%  sign vs pub {int((v2kT<pub).sum())}/{len(ks)}')
    print(f'  HEAD v2k (flag False, shipped)   median {np.median(v2kH):.4f}')
    print(f'  HEAD auto (flag False, shipped)  median {np.median(autH):.4f}')
    print(f'  LETKF frame-matched              median {np.median(lk):.4f}')
    print(f'  -> flag alone explains: {100*np.median((v2kT-v2kH)/v2kH):+.1f}% (med rel), sign {int((v2kT>v2kH).sum())}/{len(ks)} worse with flag ON')
    aT=[k for k in PH if k.endswith('|auto_phaseTrue') and 'avg_nRMSE' in PH[k]]
    ks2=[k.split('|')[0] for k in aT if f'{k.split("|")[0]}|shipping_default' in SD]
    if ks2:
        a1=np.array([PH[f'{k}|auto_phaseTrue']['avg_nRMSE'] for k in ks2])
        a0=np.array([SD[f'{k}|shipping_default']['avg_nRMSE'] for k in ks2])
        l2=np.array([FM[k]['letkf_match'] for k in ks2])
        print(f'  auto+flagTRUE (n={len(ks2)}) median {np.median(a1):.4f} vs auto shipped {np.median(a0):.4f}'
              f'  | auto+flagTRUE beats LETKF {int((a1<l2).sum())}/{len(ks2)} p={st(int((a1<l2).sum()),len(ks2)):.4f}')

print()
print('=== A. held-out 70 episodes, shipping auto vs frame-matched LETKF ===')
ks=[k for k in A70 if 'avg_nrmse' in A70[k]]
if ks:
    ws=np.array([A70[k]['avg_nrmse'] for k in ks]); lk=np.array([FM[k]['letkf_match'] for k in ks])
    ze=np.array([FM[k]['zero_match'] for k in ks]); nc=np.array([FM[k]['ncomp'] for k in ks])
    nf=np.array([A70[k]['n_tracking_frames'] for k in ks]); nm=np.array([FM[k]['n_match'] for k in ks])
    w=int((ws<lk).sum()); n=len(ks)
    print(f'  n so far {n}/70   frame-set match {int((nf==nm).sum())}/{n}')
    print(f'  WS wins {w}/{n}  p={st(w,n):.2e}  med_rel {100*np.median(ws/lk-1):+.1f}%')
    print(f'  vs ZERO: {int((ws<ze).sum())}/{n}  p={st(int((ws<ze).sum()),n):.2e}')
    for lab,m in [('sparse<=5',nc<=5),('crowded>=15',nc>=15)]:
        if m.sum():
            print(f'    {lab:<12s} n={int(m.sum())} wins {int((ws[m]<lk[m]).sum())} p={st(int((ws[m]<lk[m]).sum()),int(m.sum())):.4f} med_rel {100*np.median(ws[m]/lk[m]-1):+.1f}%')
    import collections
    cc=collections.defaultdict(lambda:[0,0])
    for k,a,b in zip(ks,ws,lk):
        c='_'.join(k.split('_')[:2]); cc[c][1]+=1; cc[c][0]+= int(a<b)
    print('  per cell:', ' '.join(f'{c}:{v[0]}/{v[1]}' for c,v in sorted(cc.items())))

print()
print('=== A2. FULL 100 episodes, shipping auto vs frame-matched LETKF ===')
allk = {}
for k in A70:
    if 'avg_nrmse' in A70[k]: allk[k] = A70[k]['avg_nrmse']
for k, v in SD.items():
    ep, arm = k.split('|')
    if arm == 'shipping_default' and 'avg_NRMSE'.lower() in [x.lower() for x in v]:
        allk[ep] = v['avg_nRMSE']
ks = sorted(allk)
ws = np.array([allk[k] for k in ks]); lk = np.array([FM[k]['letkf_match'] for k in ks])
ze = np.array([FM[k]['zero_match'] for k in ks]); nc = np.array([FM[k]['ncomp'] for k in ks])
pub = np.array([FM[k]['ws'] for k in ks])
print(f'  n={len(ks)}')
for lab, a, b in [('auto vs LETKF', ws, lk), ('auto vs ZERO', ws, ze),
                  ('pub  vs LETKF', pub, lk)]:
    w = int((a < b).sum()); n = len(ks)
    print(f'  {lab:<16s} {w}/{n}  p={st(w,n):.3e}  med(rel)={100*np.median(a/b-1):+6.1f}%  '
          f'rel(med)={100*(np.median(a)/np.median(b)-1):+6.1f}%  rel(mean)={100*(a.mean()/b.mean()-1):+6.1f}%')
for lab, m in [('sparse<=5', nc <= 5), ('crowded>=15', nc >= 15)]:
    w = int((ws[m] < lk[m]).sum()); n = int(m.sum())
    print(f'    {lab:<12s} n={n} auto beats LETKF {w} p={st(w,n):.2e} | beats ZERO '
          f'{int((ws[m]<ze[m]).sum())} | published beat LETKF {int((pub[m]<lk[m]).sum())}')
import collections
cc = collections.defaultdict(lambda: [0, 0])
for k, a, b in zip(ks, ws, lk):
    c = '_'.join(k.split('_')[:2]); cc[c][1] += 1; cc[c][0] += int(a < b)
print('  per cell:', ' '.join(f'{c}:{v[0]}/{v[1]}' for c, v in sorted(cc.items())))
# held-out only, against the frozen prediction
ho = [k for k in ks if k in A70]
w70 = sum(1 for k in ho if allk[k] < FM[k]['letkf_match'])
print(f'  HELD-OUT 70 realised: {w70}/{len(ho)}   frozen prediction mean 54.3, 90% PI [47, 61]')
