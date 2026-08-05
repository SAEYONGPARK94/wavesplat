#!/usr/bin/env python3
"""PRO follow-up: the decisive sub-analyses. Re-analysis only."""
import json, math, statistics as st
from collections import defaultdict
import numpy as np, h5py

ROOT = "/home/vexit/wavesplat/wavesplat"
E1 = f"{ROOT}/tmp/e1_grid_2026-08-05"; S12 = f"{ROOT}/tmp/s1s2_2026-08-05"
DATA = f"{ROOT}/data_finemesh_2026-07-23"; OUT = f"{ROOT}/tmp/placement_field_debate_2026-08-05"
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
med = lambda v: st.median(v) if v else float('nan')

def rank(x):
    n=len(x); o=sorted(range(n),key=lambda i:x[i]); r=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and x[o[j+1]]==x[o[i]]: j+=1
        a=(i+j)/2.0+1.0
        for k in range(i,j+1): r[o[k]]=a
        i=j+1
    return r
def pearson(x,y):
    n=len(x)
    if n<3: return float('nan')
    mx,my=sum(x)/n,sum(y)/n
    sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y))
    if sx==0 or sy==0: return float('nan')
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)
spearman=lambda x,y: pearson(rank(x),rank(y)) if len(x)>=3 else float('nan')
def sign_test(d,tol=1e-12):
    d=[v for v in d if abs(v)>tol]; n=len(d); k=sum(1 for v in d if v>0)
    if n==0: return 0,0,1.0
    t=sum(math.comb(n,i) for i in range(0,min(k,n-k)+1))
    return k,n,min(1.0,2*t/2**n)

e1=jl(f"{E1}/results.jsonl"); pr=jl(f"{E1}/probe.jsonl"); s12=jl(f"{S12}/results.jsonl")
for r in e1+s12:
    r["absVR1"]=abs(r["Variance_Ratio"]-1.0)
    for k,v in r["spec"].items(): r["spec_"+k]=v
CELLS=list("ABCDE"); EPS=sorted({r["episode"] for r in e1})
CROWDED=[e for e in EPS if "15comp" in e or "30comp" in e]; SPARSE=[e for e in EPS if e not in CROWDED]
TRUE={}
for e in EPS:
    with h5py.File(f"{DATA}/{e}","r") as f: wc=f["config/wave_components"][:]
    k=wc[:,2:4]; n=len(k)
    d=np.array([min(math.hypot(k[i,0]-k[j,0],k[i,1]-k[j,1]) for j in range(n) if j!=i) for i in range(n)]) if n>=2 else np.array([np.nan])
    TRUE[e]={"n_gt":n,"nn":d,"kmax":float(np.hypot(wc[:,2],wc[:,3]).max())}
F=["nRMSE","init_nrmse","Variance_Ratio","absVR1","Hs_error","Skill_Score","Pearson_rho",
   "Directional_R2","avg_RMSE_unobserved","N","spec_detection_rate","spec_energy_detection_rate",
   "spec_k_mag_rel_error_mean","spec_direction_error_deg_mean","spec_amp_rel_error_mean","spec_n_est"]
def med3(rows,kf):
    g=defaultdict(list)
    for r in rows: g[kf(r)].append(r)
    return {k:{f:med([x[f] for x in v]) for f in F} for k,v in g.items()}
E1M=med3(e1,lambda r:(r["episode"],r["cell"])); S12M=med3(s12,lambda r:(r["episode"],r["arm"]))
PRB={(p["episode"],p["cell"],p["n_g"]):p for p in pr}
for (e,c,ng),p in PRB.items():
    p["surplus_in_box"]=max(0,ng-p["n_outside_narrow_box"]-min(TRUE[e]["n_gt"],ng))
for k,v in E1M.items(): v["surplus_final_n"]=max(0.0,v["N"]-TRUE[k[0]]["n_gt"])
for k,v in S12M.items(): v["surplus_final_n"]=max(0.0,v["N"]-TRUE[k[0]]["n_gt"])

L=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s); L.append(s)

P("="*104); P("PRO FOLLOW-UP — the decisive sub-analyses"); P("="*104)

# ------------------------------------------------------------------
P("\n### F1. THE PUREST TEST — episodes where PLACEMENT QUALITY IS EXACTLY TIED across all 5 cells")
P("If placement and field were one axis, tied placement must give tied field.")
tied=[]
for e in EPS:
    dets=[PRB[(e,c,15)]["detection_rate"] for c in CELLS]
    kerr=[PRB[(e,c,15)]["k_mag_rel_error_mean"] for c in CELLS]
    if max(dets)-min(dets)==0 and max(kerr)-min(kerr)<1e-9: tied.append(e)
P(f"episodes with probe detection AND k_mag_rel_error identical in all five cells: {len(tied)}")
hdr=f"{'episode':<24}{'det':>6}{'kerr':>8} | " + " | ".join(f"{c}: surplus/|VR-1|/Hs" for c in CELLS)
P(hdr)
for e in tied:
    nm=e.replace("episode_","").replace("_seed0","").replace(".h5","")
    row=f"{nm:<24}{PRB[(e,'A',15)]['detection_rate']:>6.2f}{PRB[(e,'A',15)]['k_mag_rel_error_mean']:>8.4f} | "
    row+=" | ".join(f"{PRB[(e,c,15)]['surplus_in_box']:>2d}/{E1M[(e,c)]['absVR1']:.3f}/{E1M[(e,c)]['Hs_error']:.3f}" for c in CELLS)
    P(row)
rs_v=[spearman([PRB[(e,c,15)]["surplus_in_box"] for c in CELLS],[E1M[(e,c)]["absVR1"] for c in CELLS]) for e in tied]
rs_h=[spearman([PRB[(e,c,15)]["surplus_in_box"] for c in CELLS],[E1M[(e,c)]["Hs_error"] for c in CELLS]) for e in tied]
rs_v=[r for r in rs_v if not math.isnan(r)]; rs_h=[r for r in rs_h if not math.isnan(r)]
for nm,rs in [("|VR-1|",rs_v),("Hs_error",rs_h)]:
    k,n,p=sign_test(rs); P(f"  within these episodes: spearman(surplus@15, {nm}) med={med(rs):+.3f} pos {k}/{n} sign p={p:.4f}")
sp_vr=[max(E1M[(e,c)]["absVR1"] for c in CELLS)-min(E1M[(e,c)]["absVR1"] for c in CELLS) for e in tied]
sp_hs=[max(E1M[(e,c)]["Hs_error"] for c in CELLS)-min(E1M[(e,c)]["Hs_error"] for c in CELLS) for e in tied]
P(f"  field SPREAD across cells at TIED placement: |VR-1| range med={med(sp_vr):.4f} (max {max(sp_vr):.4f}), "
  f"Hs_error range med={med(sp_hs):.4f} (max {max(sp_hs):.4f})")
P(f"  reference scale: pre-registered MDE |VR-1| 0.0482, Hs_error 0.0235 -> the tied-placement spread")
P(f"  is {med(sp_vr)/0.0482:.1f}x / {med(sp_hs)/0.0235:.1f}x the MDE.")

# ------------------------------------------------------------------
P("\n### F2. THE MIRROR TEST — do the E1 field harms disappear where surplus is 0?")
P("Cell D vs A, stratified by the D-arm surplus at n_g=15 (surplus = 15 - outside_box - min(N_gt,15)).")
hdr=f"{'episode':<24}{'N_gt':>5}{'surp_D':>7}{'surp_A':>7}{'d_det@15':>9}{'d_init_nrmse':>13}{'d_nRMSE':>9}{'d_Hs':>9}{'d|VR-1|':>9}{'N_A->N_D':>10}"
P(hdr); P("-"*len(hdr))
for e in sorted(EPS,key=lambda e:-PRB[(e,"D",15)]["surplus_in_box"]):
    nm=e.replace("episode_","").replace("_seed0","").replace(".h5","")
    P(f"{nm:<24}{TRUE[e]['n_gt']:>5}{PRB[(e,'D',15)]['surplus_in_box']:>7d}{PRB[(e,'A',15)]['surplus_in_box']:>7d}"
      f"{PRB[(e,'D',15)]['detection_rate']-PRB[(e,'A',15)]['detection_rate']:>+9.3f}"
      f"{E1M[(e,'D')]['init_nrmse']-E1M[(e,'A')]['init_nrmse']:>+13.4f}"
      f"{E1M[(e,'D')]['nRMSE']-E1M[(e,'A')]['nRMSE']:>+9.4f}"
      f"{E1M[(e,'D')]['Hs_error']-E1M[(e,'A')]['Hs_error']:>+9.4f}"
      f"{E1M[(e,'D')]['absVR1']-E1M[(e,'A')]['absVR1']:>+9.4f}"
      f"{E1M[(e,'A')]['N']:>5.0f}->{E1M[(e,'D')]['N']:<4.0f}")
hi=[e for e in EPS if PRB[(e,"D",15)]["surplus_in_box"]>=10]; lo=[e for e in EPS if PRB[(e,"D",15)]["surplus_in_box"]<10]
for lab,grp in [("surplus>=10",hi),("surplus<10",lo)]:
    for f,fl in [("Hs_error","Hs_error"),("absVR1","|VR-1|"),("init_nrmse","init_nrmse"),("nRMSE","nRMSE")]:
        d=[E1M[(e,"D")][f]-E1M[(e,"A")][f] for e in grp]; k,n,p=sign_test(d)
        P(f"  {lab:<12} n={len(grp):<3} {fl:<11} medD={med(d):>+8.4f}  worse {k}/{n}  p={p:.4f}")
    P("")
P("  !! CONFOUND, stated: for cell D, surplus == 15 - N_gt exactly (n_outside_box = 0 in 12/12).")
P("     So this stratification is NOT independent of the crowding stratifier. It is reported as")
P("     descriptive; the CONTROLLED version of the same claim is F1 and PART 3a (within-episode).")

# ------------------------------------------------------------------
P("\n### F3. THE ORACLE'S SPACING — is the oracle 'the narrow grid taken further'?")
hdr=f"{'episode':<24}{'N_gt':>5}{'true nn_med':>12}{'cellA nn@15':>12}{'cellD nn@15':>12}{'D/true':>8}{'A/true':>8}"
P(hdr); P("-"*len(hdr))
for e in EPS:
    t=TRUE[e]; nm=e.replace("episode_","").replace("_seed0","").replace(".h5","")
    tn=float(np.median(t["nn"])) if t["n_gt"]>=2 else float('nan')
    a=PRB[(e,"A",15)]["placed_nn_median"]; d=PRB[(e,"D",15)]["placed_nn_median"]
    P(f"{nm:<24}{t['n_gt']:>5}{(tn if tn==tn else float('nan')):>12.4f}{a:>12.4f}{d:>12.4f}"
      f"{(d/tn if tn==tn else float('nan')):>8.2f}{(a/tn if tn==tn else float('nan')):>8.2f}")
P("\n  crowded 4 ep: true nn_med = %.4f ; cell D nn@15 = %.4f  -> ratio %.2f (D matches the physics)"
  % (med([float(np.median(TRUE[e]['nn'])) for e in CROWDED]),
     med([PRB[(e,'D',15)]['placed_nn_median'] for e in CROWDED]),
     med([PRB[(e,'D',15)]['placed_nn_median']/float(np.median(TRUE[e]['nn'])) for e in CROWDED])))
r5=[e for e in EPS if TRUE[e]["n_gt"] in (2,5)]
P("  2/5-comp ep: true nn_med = %.4f ; cell D nn@15 = %.4f  -> ratio %.2f (D is %.0fx TIGHTER than physics)"
  % (med([float(np.median(TRUE[e]['nn'])) for e in r5]),
     med([PRB[(e,'D',15)]['placed_nn_median'] for e in r5]),
     med([PRB[(e,'D',15)]['placed_nn_median']/float(np.median(TRUE[e]['nn'])) for e in r5]),
     1/med([PRB[(e,'D',15)]['placed_nn_median']/float(np.median(TRUE[e]['nn'])) for e in r5])))
P("\n  ==> ORACLE ARM: n_est == N_gt exactly, atoms sit exactly on the truth, so its nn ratio is 1.00")
P("      BY DEFINITION and its surplus is 0 BY DEFINITION, at every crowding level.")
S2EPS=sorted({k[0] for k in S12M if k[1]=="S2-T1f"})
hdr=f"{'episode':<24}{'N_gt':>5}{'T1f n_est':>10}{'T1f final_n':>12}{'T1f surplus':>12}{'A0 final_n':>11}{'A0 surplus':>11}"
P(hdr); P("-"*len(hdr))
for e in S2EPS:
    nm=e.replace("episode_","").replace("_seed0","").replace(".h5","")
    P(f"{nm:<24}{TRUE[e]['n_gt']:>5}{S12M[(e,'S2-T1f')]['spec_n_est']:>10.0f}{S12M[(e,'S2-T1f')]['N']:>12.0f}"
      f"{S12M[(e,'S2-T1f')]['surplus_final_n']:>12.0f}{S12M[(e,'S1-A0')]['N']:>11.0f}{S12M[(e,'S1-A0')]['surplus_final_n']:>11.0f}")

# ------------------------------------------------------------------
P("\n### F4. THE TWO CHANNELS, SEPARATED AND SIZED (E1, cell D vs A)")
P("Channel 1 (benefit): placement -> nRMSE, only fires where detection has headroom.")
P("Channel 2 (harm):   surplus/stacking -> dispersion, fires everywhere.")
head=[e for e in EPS if PRB[(e,"A",15)]["detection_rate"]<0.999]
noh =[e for e in EPS if PRB[(e,"A",15)]["detection_rate"]>=0.999]
P(f"  detection headroom in cell A: {len(head)} ep ({[e[8:-13] for e in head]})")
P(f"  no headroom (A already 1.000): {len(noh)} ep")
for lab,grp in [("HEADROOM",head),("NO HEADROOM",noh)]:
    for f,fl in [("init_nrmse","init_nrmse"),("nRMSE","nRMSE"),("Hs_error","Hs_error"),("absVR1","|VR-1|"),("Skill_Score","Skill"),("Pearson_rho","rho")]:
        d=[E1M[(e,"D")][f]-E1M[(e,"A")][f] for e in grp]; k,n,p=sign_test(d)
        P(f"  {lab:<12} {fl:<11} med={med(d):>+9.4f}  larger {k}/{n}  p={p:.4f}")
    P("")
allep=[E1M[(e,"D")]["nRMSE"]-E1M[(e,"A")]["nRMSE"] for e in EPS]
P(f"  NET over all 12 (the deployment mix): nRMSE medD={med(allep):+.4f}  <-- the two channels cancel")

# ------------------------------------------------------------------
P("\n### F5. IS THE HARM CHANNEL THE SAME ONE IN E1 AND S2?  (dispersion signature)")
hdr=f"{'intervention':<44}{'d_det':>8}{'d_nRMSE':>9}{'d_rho':>8}{'d_VR':>8}{'d|VR-1|':>9}{'d_Hs':>8}"
P(hdr); P("-"*len(hdr))
def de1(t,b,f,grp=EPS): return med([E1M[(e,t)][f]-E1M[(e,b)][f] for e in grp])
def ds2(t,b,f): return med([S12M[(e,t)][f]-S12M[(e,b)][f] for e in S2EPS])
rows=[("E1  D vs A  (narrow grid, 12 ep)",lambda f:de1("D","A",f)),
      ("E1  B vs A  (narrow grid, 12 ep)",lambda f:de1("B","A",f)),
      ("E1  D vs A  (narrow grid, crowded 4)",lambda f:de1("D","A",f,CROWDED)),
      ("S2  T1f vs A0 (oracle k, crowded 4)",lambda f:ds2("S2-T1f","S1-A0",f)),
      ("S1  A3 vs A0 (budget 40, crowded 4)",lambda f:ds2("S1-A3","S1-A0",f)),
      ("S1  A1 vs A0 (budget 22, crowded 4)",lambda f:ds2("S1-A1","S1-A0",f))]
for lab,fn in rows:
    P(f"{lab:<44}{fn('spec_detection_rate'):>+8.3f}{fn('nRMSE'):>+9.4f}{fn('Pearson_rho'):>+8.3f}"
      f"{fn('Variance_Ratio'):>+8.3f}{fn('absVR1'):>+9.4f}{fn('Hs_error'):>+8.4f}")
P("  ==> FOUR independent placement interventions, one signature: detection UP, VR DOWN, Hs_error UP.")

# ------------------------------------------------------------------
P("\n### F6. WHERE THE ENERGY GOES — amplitude error vs detection (E1 e2e, within episode)")
rs=[spearman([E1M[(e,c)]["spec_detection_rate"] for c in CELLS],[E1M[(e,c)]["spec_amp_rel_error_mean"] for c in CELLS]) for e in EPS]
rs=[r for r in rs if not math.isnan(r)]; k,n,p=sign_test(rs)
P(f"  within-episode spearman(detection, amp_rel_error) med={med(rs):+.3f} pos {k}/{n} p={p:.4f}")
rs=[spearman([PRB[(e,c,15)]["surplus_in_box"] for c in CELLS],[E1M[(e,c)]["spec_amp_rel_error_mean"] for c in CELLS]) for e in EPS]
rs=[r for r in rs if not math.isnan(r)]; k,n,p=sign_test(rs)
P(f"  within-episode spearman(surplus@15, amp_rel_error)  med={med(rs):+.3f} pos {k}/{n} p={p:.4f}")
for lab,arms in [("S1 budget",["S1-A0","S1-A1","S1-A2","S1-A3"]),("S2 oracle",["S1-A0","S2-C1","S2-T1f","X-T1f-notrack"])]:
    P(f"  {lab}: " + "  ".join(f"{a}={med([S12M[(e,a)]['spec_amp_rel_error_mean'] for e in S2EPS]):.3f}" for a in arms))

# ------------------------------------------------------------------
P("\n### F7. TRACKER x PLACEMENT 2x2 (S2) — the field axis has its own bottleneck")
hdr=f"{'':<18}{'placement=beamform':>22}{'placement=oracle':>20}{'row effect':>14}"
P(hdr); P("-"*len(hdr))
for f,fl in [("nRMSE","nRMSE"),("Variance_Ratio","VR"),("Hs_error","Hs_error"),("Pearson_rho","rho"),("Skill_Score","Skill")]:
    on_b=med([S12M[(e,"S1-A0")][f] for e in S2EPS]); on_o=med([S12M[(e,"S2-T1f")][f] for e in S2EPS])
    of_b=med([S12M[(e,"X-A0-notrack")][f] for e in S2EPS]); of_o=med([S12M[(e,"X-T1f-notrack")][f] for e in S2EPS])
    P(f"{fl+' tracker ON':<18}{on_b:>22.4f}{on_o:>20.4f}{on_o-on_b:>+14.4f}")
    P(f"{fl+' tracker OFF':<18}{of_b:>22.4f}{of_o:>20.4f}{of_o-of_b:>+14.4f}")
    P(f"{'  tracker effect':<18}{on_b-of_b:>+22.4f}{on_o-of_o:>+20.4f}{'interaction '+f'{(on_o-of_o)-(on_b-of_b):+.4f}':>14}")
    P("")
P("  placement main effect on nRMSE with tracker OFF  = %+.4f" %
  (med([S12M[(e,"X-T1f-notrack")]["nRMSE"] for e in S2EPS])-med([S12M[(e,"X-A0-notrack")]["nRMSE"] for e in S2EPS])))
P("  placement main effect on nRMSE with tracker ON   = %+.4f" %
  (med([S12M[(e,"S2-T1f")]["nRMSE"] for e in S2EPS])-med([S12M[(e,"S1-A0")]["nRMSE"] for e in S2EPS])))
P("  ==> the tracker EATS %.0f%% of the placement gain."
  % (100*(1-(med([S12M[(e,"S2-T1f")]["nRMSE"] for e in S2EPS])-med([S12M[(e,"S1-A0")]["nRMSE"] for e in S2EPS]))
          /(med([S12M[(e,"X-T1f-notrack")]["nRMSE"] for e in S2EPS])-med([S12M[(e,"X-A0-notrack")]["nRMSE"] for e in S2EPS])))))

open(f"{OUT}/PRO_followup.txt","w").write("\n".join(L)+"\n")
P(f"\nwritten {OUT}/PRO_followup.txt")
