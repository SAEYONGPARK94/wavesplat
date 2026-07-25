import numpy as np
from scipy import stats as ss
import d3_prior_net_data as G
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
net=get_net('results/direction3/prior_net_2ndlobeTF.pt')
def nr(fq,gt): return float(np.sqrt(np.mean((fq-gt)**2))/max(np.std(gt),1e-9))
print("=== PART 1: complexity sweep (gen_sample in-dist, hand vs learned, gt_q) ===",flush=True)
for nc in [5,15,30]:
    H,F=[],[]; rng=np.random.default_rng(70000+nc)
    for r in range(20):
        s=G.gen_sample(rng,0.0,0.0,0.15,nc); P,z,Q,gt=s['P'],s['z'],s['Q'],s['gt_q']; kp=s['kp_e']; s0=0.05*s['Hs']
        H.append(nr(mmse_bank_estimate(P,z,G.DEPTH,Q,sigma0=s0,kp_hint=kp,seed=r),gt))
        F.append(nr(net_gamma_mmse(net,P,z,G.DEPTH,Q,sigma0=s0,seed=r,kp_hint=kp),gt))
    H,F=np.array(H),np.array(F); d=H-F
    print(f"  nc={nc:2d}: hand {H.mean():.3f} learned {F.mean():.3f} d {d.mean():+.3f} ({int((F<H).sum())}/20 p={ss.wilcoxon(d).pvalue:.4f})",flush=True)
print("=== PART 2: UQ calibration (spread-vs-error corr, nc=30 n=20) ===",flush=True)
ch,cf=[],[]; rng=np.random.default_rng(88)
for r in range(20):
    s=G.gen_sample(rng,0.0,0.0,0.15,30); P,z,Q,gt=s['P'],s['z'],s['Q'],s['gt_q']; kp=s['kp_e']; s0=0.05*s['Hs']
    fh,vh=mmse_bank_estimate(P,z,G.DEPTH,Q,sigma0=s0,kp_hint=kp,seed=r,return_var=True)
    ff,vf=net_gamma_mmse(net,P,z,G.DEPTH,Q,sigma0=s0,seed=r,kp_hint=kp,return_var=True)
    if np.std(vh)>0: ch.append(np.corrcoef(np.sqrt(np.abs(vh)),np.abs(fh-gt))[0,1])
    if np.std(vf)>0: cf.append(np.corrcoef(np.sqrt(np.abs(vf)),np.abs(ff-gt))[0,1])
print(f"  spread-vs-error corr: analytic {np.nanmean(ch):+.3f}  learned {np.nanmean(cf):+.3f}",flush=True)
