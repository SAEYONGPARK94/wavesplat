"""NEUTRAL §5.6 — verify on REAL run data that RMSE_observed / RMSE_unobserved
partition the grid exactly, and that the weighted quadratic combination
reproduces the full-grid RMSE.  CPU only, no estimator run, no GPU."""
import os, sys, numpy as np
REPO='/home/vexit/wavesplat/wavesplat'
sys.path.insert(0,REPO)
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.evaluation.metrics import rmse, rmse_observed, rmse_unobserved, compute_all_metrics

DATA=os.path.join(REPO,'data_indep4_2026-08-07')
eps=[f for f in sorted(os.listdir(DATA)) if f.endswith('_ra_rf_ds.h5')]
print(f'episodes in indep4 = {len(eps)}')
rng=np.random.default_rng(0)
worst_part=0; worst_comb=0.0; nframe=0; frac=[]
for ep in eps[:4]:
    L=EpisodeDataLoader(os.path.join(DATA,ep),device='cpu')
    gx=L.gt_grid_x.cpu().numpy(); gy=L.gt_grid_y.cpu().numpy()
    tf=L.get_tracking_frames(tp_factor=2.0)
    print(f'\n{ep[8:-17]:24s} grid={len(gy)}x{len(gx)}  tracking frames={len(tf)}')
    for fi,f in enumerate(tf):
        gt=f.gt_field.cpu().numpy()
        ox=f.x_obs.cpu().numpy(); oy=f.y_obs.cpu().numpy()
        # a non-trivial "prediction": smoothed + noisy GT, so RMSE is spatially varying
        pred=gt*0.7+rng.normal(0,0.1,gt.shape)
        Xg,Yg=np.meshgrid(gx,gy)
        d=np.full(Xg.shape,np.inf)
        for j in range(len(ox)):
            d=np.minimum(d,np.sqrt((Xg-ox[j])**2+(Yg-oy[j])**2))
        mo=d<10.0; mu=d>=10.0
        # (1) exact partition of the grid
        part_err=int(np.sum(mo&mu))+int(mo.sum()+mu.sum()-gt.size)
        worst_part=max(worst_part,abs(part_err))
        ro=rmse_observed(pred,gt,gx,gy,ox,oy); ru=rmse_unobserved(pred,gt,gx,gy,ox,oy)
        rt=rmse(pred,gt)
        comb=np.sqrt((mo.sum()*ro**2+mu.sum()*ru**2)/gt.size)
        worst_comb=max(worst_comb,abs(comb-rt))
        frac.append(mo.sum()/gt.size)
        nframe+=1
        if fi<2:
            print(f'   frame{fi}: n_obs_pts={len(ox)}  |obs cells|={mo.sum()} |unobs cells|={mu.sum()} '
                  f'sum={mo.sum()+mu.sum()} (grid={gt.size})  overlap={int((mo&mu).sum())}')
            print(f'            RMSE_obs={ro:.6f} RMSE_unobs={ru:.6f}  weighted={comb:.9f} '
                  f'full RMSE={rt:.9f}  |diff|={abs(comb-rt):.2e}')
        # (2) confirm compute_all_metrics emits both and they equal the direct calls
        m=compute_all_metrics(pred,gt,gx,gy,L.Hs_true,obs_x=ox,obs_y=oy)
        assert abs(m['RMSE_observed']-ro)<1e-12 and abs(m['RMSE_unobserved']-ru)<1e-12
print(f'\n=== §5.6 VERDICT over {nframe} REAL tracking frames (4 episodes) ===')
print(f'  max partition error (overlap + coverage) = {worst_part}   -> {"PASS" if worst_part==0 else "FAIL"}')
print(f'  max |weighted-combination - full RMSE|   = {worst_comb:.3e} -> {"PASS" if worst_comb<1e-12 else "FAIL"}')
print(f'  observed-cell fraction: med={np.median(frac):.4f} min={min(frac):.4f} max={max(frac):.4f}')
print(f'  => the observed half carries ~{100*np.median(frac):.1f}% of the grid; '
      f'unobserved ~{100*(1-np.median(frac)):.1f}%')
print(f'  NOTE units: rmse_observed/unobserved return metres (no /Hs); nRMSE is /Hs.')
