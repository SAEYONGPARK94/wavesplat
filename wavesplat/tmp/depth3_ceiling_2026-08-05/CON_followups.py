"""CON depth-3 follow-ups: paired within/across, analytic bias law, N_eff vs gamma,
and the exact statement of what IS and IS NOT reachable."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

def sig_from_r1(r1): return np.degrees(np.sqrt(np.clip(2.0*(1.0-np.clip(r1,0,1)),0,None)))
def r1_from_sig(s):  return 1.0 - np.radians(np.asarray(s,float))**2/2.0

raw = json.load(open(os.path.join(ROOT,'tmp/cdip_support_2026-08-03/cdip_support_raw.json')))
B,T,A,ST = [],[],[],[]
for st in raw['real']:
    b=np.array(st['spread_band_deg'],float); t=np.array(st['spread_total_deg'],float)
    m=np.isfinite(b)&np.isfinite(t); b,t=b[m],t[m]
    rb,rt=r1_from_sig(b),r1_from_sig(t)
    ok=(rb>1e-6)&(rt>0)&(rt<=rb+1e-9)
    B+=b[ok].tolist(); T+=t[ok].tolist()
    A+=sig_from_r1(np.clip(rt[ok]/rb[ok],0,1)).tolist(); ST+=[st['station']]*int(ok.sum())
B,T,A=np.array(B),np.array(T),np.array(A)

OUT={}
OUT['paired_within_vs_across']={
 'n':int(B.size),
 'median_within':float(np.median(B)),'median_across':float(np.median(A)),
 'median_total':float(np.median(T)),
 'frac_records_within_GT_across':float((B>A).mean()),
 'median_ratio_across_over_within':float(np.median(A/np.maximum(B,1e-9))),
 'reading':('in the real ocean the DOMINANT term is WITHIN-frequency; our generator '
            'sets that term to exactly 0 and must overdrive the subordinate term'),
}

# our generator's allocation, at the frozen default and at the "matched-total" setting
def pop(s): return float(sig_from_r1(np.exp(-np.radians(s)**2/2)))
OUT['allocation_table']={
 'real_ocean_CDIP_median': {'within':float(np.median(B)),'across':float(np.median(A)),'total':float(np.median(T))},
 'ours_s30_frozen_default': {'within':0.0,'across':pop(30.0),'total':pop(30.0)},
 'ours_s37.2_total_matched':{'within':0.0,'across':pop(37.2),'total':pop(37.2)},
 'ours_s60_ALREADY_GENERATED':{'within':0.0,'across':pop(60.0),'total':pop(60.0),
                               'realized_30comp_measured':46.33},
 'across_overdrive_factor':{'s30':pop(30.0)/float(np.median(A)),
                            's37.2':pop(37.2)/float(np.median(A)),
                            's60':pop(60.0)/float(np.median(A))},
}

# analytic finite-N bias law:  E[r1_N^2] = r1_inf^2 + (1-r1_inf^2)/N_eff
def neff(N,Tp=7.0,gamma=3.3,Hs=1.2):
    op=2*np.pi/Tp; e=np.linspace(0.5*op,3.0*op,N+1)
    oc,do=0.5*(e[:-1]+e[1:]),np.diff(e); sg=np.where(oc<=op,0.07,0.09)
    S=(0.3125*Hs**2*op**4/oc**5*np.exp(-1.25*(op/oc)**4)*gamma**np.exp(-0.5*((oc/op-1)/sg)**2))
    w=(S*do); w=w/w.sum(); return float(1/(w**2).sum())

rows={}
for s in (30.0,37.2,43.5):
    r1i=np.exp(-np.radians(s)**2/2); rr={}
    for N in (5,15,30,50,100,200,500,1000,2000,10000):
        ne=neff(N); r1n=np.sqrt(r1i**2+(1-r1i**2)/ne)
        rr[N]={'N_eff':round(ne,2),'sigma1_pred_deg':round(float(sig_from_r1(r1n)),3),
               'gap_to_asymptote_deg':round(float(sig_from_r1(r1i)-sig_from_r1(r1n)),3)}
    rows[f's={s:g}']={'asymptote_deg':round(pop(s),3),'by_N':rr}
OUT['analytic_bias_law']={'formula':'E[r1_N^2]=r1_inf^2+(1-r1_inf^2)/N_eff ; N_eff=1/sum(w_i^2)',
                          'rows':rows}

OUT['N_eff_vs_gamma']={g:{N:round(neff(N,gamma=g),2) for N in (5,8,15,30,100,300,1000)}
                       for g in (1.0,2.0,3.3,7.0)}
OUT['N_eff_slope_gamma3.3']=round(neff(10000)/10000,4)
OUT['N_eff_nonmonotonic_small_N']={N:round(neff(N),3) for N in range(1,17)}

# what N does OUR EXACT RULE need to clear the real median at each s?
from wavesim.wave_generator import jonswap_components
def gen_sig(N,s,seed):
    cs=jonswap_components(Hs=1.2,Tp=7.0,n_components=N,depth=50.0,
                          direction_deg=0.0,direction_spread_deg=s,seed=seed)
    Ai=np.array([c.A for c in cs]); th=np.array([np.arctan2(c.ky,c.kx) for c in cs])
    E=0.5*Ai**2; e=E.sum()
    return float(sig_from_r1(np.hypot((E*np.cos(th)).sum()/e,(E*np.sin(th)).sum()/e)))

need={}
for s in (30.0,35.0,37.2,40.0,43.5,50.0,60.0):
    hit_med=None; hit_full=None
    for N in (5,10,15,20,30,50,75,100,150,200,300,500):
        m=float(np.median([gen_sig(N,s,2000+r) for r in range(200)]))
        if hit_med is None and m>=35.349: hit_med=N
        if hit_full is None and m>=40.550: hit_full=N
        if hit_med is not None and hit_full is not None: break
    need[f's={s:g}']={'asymptote':round(pop(s),2),
                      'min_N_for_CDIP_total_median_35.35':hit_med,
                      'min_N_for_028p1_full_median_40.55':hit_full}
OUT['min_N_our_exact_rule']=need

json.dump(OUT,open(os.path.join(HERE,'CON_followups.json'),'w'),indent=1)
print(json.dumps(OUT,indent=1))
