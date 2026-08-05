"""CON-2 (CORRECTED): score the SELECTED learned atoms (pool_and_select, max_n=30) --
the set actually installed -- on the same ruler as S1-A0's `spec`.
v1 of this script scored the 128 RAW pooled atoms and got det=0.983: a coverage artifact.
Recorded here deliberately."""
import json, os, statistics as st
import numpy as np, h5py
import sys; sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')

def pool_and_select(kx, ky, energy, energy_thresh=0.03, dedup_dk=0.03, max_n=30):
    kx = np.asarray(kx, float); ky = np.asarray(ky, float); en = np.asarray(energy, float)
    if en.size == 0 or en.max() <= 0: return np.array([]), np.array([]), np.array([])
    keep = en > energy_thresh * en.max(); kx, ky, en = kx[keep], ky[keep], en[keep]
    order = np.argsort(-en); sk, sy, se = [], [], []
    for i in order:
        if not any(abs(kx[i]-jx) < dedup_dk and abs(ky[i]-jy) < dedup_dk for jx, jy in zip(sk, sy)):
            sk.append(kx[i]); sy.append(ky[i]); se.append(en[i])
        if len(sk) >= max_n: break
    return np.array(sk), np.array(sy), np.array(se)

HERE = '/home/vexit/wavesplat/wavesplat/tmp'
DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'
EPS = ['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']

def match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A):
    N_est=len(est_kx); N_gt=len(gt_kx)
    if N_est==0 or N_gt==0: return []
    order=np.argsort(-np.asarray(gt_A)); used=set(); pairs=[]
    for gi in order:
        d=np.hypot(np.asarray(est_kx)-gt_kx[gi], np.asarray(est_ky)-gt_ky[gi])
        for ei in np.argsort(d):
            if ei in used: continue
            kmag=np.hypot(gt_kx[gi],gt_ky[gi])
            if d[ei] <= 0.5*kmag: used.add(int(ei)); pairs.append((int(gi),int(ei)))
            break
    return pairs

def spec_of(kx,ky,amp,gt):
    gk,gy,gA = gt['kx'],gt['ky'],gt['A']
    pairs=match_components(kx,ky,amp,gk,gy,gA); n_gt=len(gA); E=gA**2
    r=dict(n_est=len(kx), n_matched=len(pairs), detection_rate=len(pairs)/n_gt)
    if pairs:
        ke,de=[],[]
        for gi,ei in pairs:
            gm=np.hypot(gk[gi],gy[gi]); em=np.hypot(kx[ei],ky[ei])
            ke.append(abs(em-gm)/max(gm,1e-9))
            ga=np.degrees(np.arctan2(gy[gi],gk[gi])); ea=np.degrees(np.arctan2(ky[ei],kx[ei]))
            de.append(abs((ea-ga+180)%360-180))
        r['k_mag_rel_error_mean']=float(np.mean(ke)); r['direction_error_deg_mean']=float(np.mean(de))
        r['energy_detection_rate']=float(E[[g for g,_ in pairs]].sum()/E.sum())
    else:
        r['k_mag_rel_error_mean']=float('nan'); r['direction_error_deg_mean']=float('nan'); r['energy_detection_rate']=0.0
    return r

GT={}
for ep in EPS:
    with h5py.File(os.path.join(DATA,ep),'r') as f: wc=np.array(f['config/wave_components'])
    GT[ep]=dict(A=wc[:,0],w=wc[:,1],kx=wc[:,2],ky=wc[:,3])

atoms={}
for l in open(f'{HERE}/wp_integration_2026-07-25/atoms_cache.jsonl'):
    d=json.loads(l)
    if d['episode'] in EPS and not d.get('err'): atoms.setdefault(d['episode'],{})[d['diff_seed']]=d

s1s2={}
for l in open(f'{HERE}/s1s2_2026-08-05/results.jsonl'):
    d=json.loads(l); s1s2.setdefault((d['arm'],d['episode']),{})[d['seed']]=d
medspec=lambda arm,ep,k: st.median([v['spec'][k] for v in s1s2[(arm,ep)].values()])

wp={}
for l in open(f'{HERE}/wp_integration_2026-07-25/arm_runs.jsonl'):
    d=json.loads(l)
    if not d.get('err'): wp.setdefault((d['arm'],d['episode']),{})[(d['est_seed'],d['diff_seed'])]=d

KEYS=['n_est','detection_rate','energy_detection_rate','k_mag_rel_error_mean','direction_error_deg_mean']
print(f"{'episode':26s} {'arm':18s} {'n_est':>5s} {'det':>6s} {'Edet':>6s} {'k_err':>7s} {'dir_deg':>8s}")
res={}
for ep in EPS:
    rows=[]
    for a in atoms[ep].values():
        sk,sy,se = pool_and_select(a['kx'],a['ky'],a['energy'],max_n=30)
        rows.append(spec_of(sk,sy,se,GT[ep]))
    L={k: st.median([r[k] for r in rows]) for k in KEYS}
    L['n_sel_raw']=st.median([a['n_selected'] for a in atoms[ep].values()])
    A0={k: medspec('S1-A0',ep,k) for k in KEYS}
    OR={k: medspec('S2-T1f',ep,k) for k in KEYS}
    res[ep]=dict(learned_selected=L,S1_A0=A0,oracle=OR,
                 n_gt=len(GT[ep]['A']),
                 final_n_learned=st.median([v['final_n'] for v in wp[('learned',ep)].values()]),
                 final_n_v2=st.median([v['final_n'] for v in wp[('v2_konly',ep)].values()]))
    for tag,r in [('learned-SELECTED',L),('S1-A0 (v2 beamf)',A0),('oracle T1f',OR)]:
        print(f"{ep[8:-3]:26s} {tag:18s} {r['n_est']:5.0f} {r['detection_rate']:6.3f} "
              f"{r['energy_detection_rate']:6.3f} {r['k_mag_rel_error_mean']:7.3f} {r['direction_error_deg_mean']:8.2f}")
    print(f"{'':26s} {'final_n installed':18s} learned {res[ep]['final_n_learned']:.0f} / v2 {res[ep]['final_n_v2']:.0f} / N_gt {res[ep]['n_gt']}\n")

mo=lambda tag,k: st.mean([res[ep][tag][k] for ep in EPS])
print('MEAN4:')
for tag in ['learned_selected','S1_A0','oracle']:
    print(f"  {tag:18s} n_est {mo(tag,'n_est'):5.1f}  det {mo(tag,'detection_rate'):.3f}  "
          f"Edet {mo(tag,'energy_detection_rate'):.3f}  k_err {mo(tag,'k_mag_rel_error_mean'):.3f}  dir {mo(tag,'direction_error_deg_mean'):.2f}")
for k,lab in [('detection_rate','detection'),('energy_detection_rate','energy-detection')]:
    l,a=mo('learned_selected',k),mo('S1_A0',k)
    print(f"  PLACEMENT-axis recovery ({lab}): ({l:.3f}-{a:.3f})/(1-{a:.3f}) = {100*(l-a)/(1-a):.1f}%")
for k,lab in [('k_mag_rel_error_mean','k-magnitude err'),('direction_error_deg_mean','direction err')]:
    l,a=mo('learned_selected',k),mo('S1_A0',k)
    print(f"  PLACEMENT-axis recovery ({lab}): ({a:.3f}-{l:.3f})/({a:.3f}-0) = {100*(a-l)/a:.1f}%")
json.dump(res,open(f'{HERE}/waveprior_debate_2026-08-05/CON_axis_decomp.json','w'),indent=1,default=float)
