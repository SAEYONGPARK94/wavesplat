import numpy as np, sys
sys.path.insert(0,'scripts')
import d3_compare_gifs as cg
cg.NET_CKPT='results/direction3/prior_net_2ndlobeTF.pt'
def enc_curve(kind, N):
    EW,EP,EGT,leads=[],[],[],None
    for r in range(N):
        try: sc=cg.scene_real('YS17',win=r) if kind=='real' else cg.scene_sim(kind,seed=r)
        except Exception: continue
        sc['n_obs']=len(sc['z']); sol=cg.solve(sc)
        leads=sc['leads']; EW.append(sol['ws']['enc']); EP.append(sol['wp']['enc']); EGT.append(sc['enc_gt'])
    EW,EP,EGT=map(np.array,(EW,EP,EGT)); sg=np.nanstd(EGT)
    nw=np.sqrt(np.nanmean((EW-EGT)**2,axis=0))/sg; npd=np.sqrt(np.nanmean((EP-EGT)**2,axis=0))/sg
    return leads,nw,npd,len(EW)
for kind,N in [('linear',8),('hos',6),('real',6)]:
    leads,nw,npd,n=enc_curve(kind,N)
    print(f"\n=== {kind} encounter forecast (n={n}), nRMSE vs lead (analytic/learned; 1.0=zero-pred) ===",flush=True)
    for i,L in enumerate(leads):
        if L in (0,2,4,6,8,10,12,14,16,18,20): print(f"  lead {L:4.1f}s: analytic {nw[i]:.2f} / learned {npd[i]:.2f}",flush=True)
    hw=leads[nw<0.7].max() if (nw<0.7).any() else 0; hp=leads[npd<0.7].max() if (npd<0.7).any() else 0
    print(f"  predictable horizon (nRMSE<0.7): analytic {hw:.1f}s / learned {hp:.1f}s",flush=True)
