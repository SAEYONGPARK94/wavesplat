"""NEUTRAL — how much of the grid is the 'observed' half, per episode?  Needed to
read PRIMARY-B's two effect sizes on a common footing (they are METRES on
non-equal-area sub-grids)."""
import os,sys,numpy as np,statistics as st,json,collections
REPO='/home/vexit/wavesplat/wavesplat'; sys.path.insert(0,REPO)
from wavesplat.data_loader import EpisodeDataLoader
DATA=os.path.join(REPO,'data_indep4_2026-08-07')
H=os.path.dirname(os.path.abspath(__file__))
eps=[f for f in sorted(os.listdir(DATA)) if f.endswith('_ra_rf_ds.h5')]
FR={}
for ep in eps:
    L=EpisodeDataLoader(os.path.join(DATA,ep),device='cpu')
    gx=L.gt_grid_x.cpu().numpy(); gy=L.gt_grid_y.cpu().numpy()
    Xg,Yg=np.meshgrid(gx,gy); P=np.stack([Xg.ravel(),Yg.ravel()],1)
    fr=[]
    for f in L.get_tracking_frames(tp_factor=2.0):
        ox=f.x_obs.cpu().numpy(); oy=f.y_obs.cpu().numpy()
        if len(ox)==0: continue
        d=np.sqrt(((P[:,None,0]-ox[None,:])**2+(P[:,None,1]-oy[None,:])**2)).min(1)
        fr.append(float((d<10.0).mean()))
    FR[ep]=fr
    print(f'{ep[8:-17]:24s} frames={len(fr)} observed-cell fraction med={st.median(fr):.4f} '
          f'min={min(fr):.4f} max={max(fr):.4f}')
allf=[x for v in FR.values() for x in v]
print(f'\nALL {len(allf)} frames: observed fraction med={st.median(allf):.4f} '
      f'p05={np.percentile(allf,5):.4f} p95={np.percentile(allf,95):.4f}')
json.dump({k:st.median(v) for k,v in FR.items()},open(H+'/audit_neutral_obsfrac.json','w'),indent=1)

# ---- mechanism: MSE-budget decomposition per episode (exact algebra, per-episode Hs) ----
rows=[json.loads(l) for l in open(H+'/results.jsonl') if l.strip()]
ok=[r for r in rows if 'error' not in r]
NULL={n['episode']:n for n in json.load(open(H+'/nulls_rx7.json'))}
IDX=collections.defaultdict(list)
for r in ok: IDX[(r['cell'],r['episode'])].append(r)
def med(c,e,k): return st.median([r[k] for r in IDX[(c,e)]])
kept=[e for e in sorted({r['episode'] for r in ok})
      if med('AUTO',e,'nRMSE')<NULL[e]['zero_null_tracking_nRMSE_mean']]
print('\n### MSE budget: which half carries the tracker damage?  (per episode, f = its own median observed fraction)')
print(f"  {'episode':24s} {'f':>6s} {'obs MSE Δ':>11s} {'unobs MSE Δ':>12s} {'unobs/|obs|':>11s}")
ratios=[]
for e in kept:
    f=st.median(FR[e])
    do=f*(med('AUTO',e,'avg_RMSE_observed')**2-med('AUTO-NOTRACK',e,'avg_RMSE_observed')**2)
    du=(1-f)*(med('AUTO',e,'avg_RMSE_unobserved')**2-med('AUTO-NOTRACK',e,'avg_RMSE_unobserved')**2)
    ratios.append(du/abs(do))
    print(f'  {e[8:-17]:24s} {f:6.4f} {do:+11.6f} {du:+12.6f} {du/abs(do):+11.2f}')
print(f'  median(unobs MSE Δ / |obs MSE Δ|) over episodes = {st.median(ratios):+.2f}')
print('  (>1 means the unobserved half dominates the net effect despite the smaller per-cell Δ)')
