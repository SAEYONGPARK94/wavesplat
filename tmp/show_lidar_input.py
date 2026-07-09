"""Visualize the INPUT: real ship-LiDAR points (3D) on the wave surface. Shows the actual sparse forward scan."""
import os, sys, numpy as np, h5py
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
EP='data/episode_BF4_15comp_seed0_uniform.h5'
h=h5py.File(EP,'r')
gx=h['frames/gt_grid_x'][:]; gy=h['frames/gt_grid_y'][:]; Xg,Yg=np.meshgrid(gx,gy)
lp=h['frames/lidar_points']; vm=h['frames/lidar_valid_mask']; gtf=h['frames/gt_wave_field']; ts=h['frames/timestamps'][:]
f=150; pts=lp[f][vm[f].astype(bool)]; gt=gtf[f]
fig=plt.figure(figsize=(18,5.2))
# 1) 3D: one-frame LiDAR on wave surface
ax=fig.add_subplot(1,3,1,projection='3d')
ax.plot_surface(Xg,Yg,gt,cmap='RdBu_r',alpha=0.35,linewidth=0,antialiased=True,rcount=60,ccount=80)
ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=pts[:,2],cmap='viridis',s=28,edgecolor='k',linewidth=0.3,depthshade=False)
ax.set_title(f'1 frame: {len(pts)} LiDAR pts on wave surface',fontsize=10); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('η [m]'); ax.view_init(28,-60)
# 2) 3D: accumulated window (spatiotemporal input), colored by time
ax2=fig.add_subplot(1,3,2,projection='3d')
for i in range(130,151):
    p=lp[i][vm[i].astype(bool)]
    ax2.scatter(p[:,0],p[:,1],p[:,2],c=[ts[i]]*len(p),cmap='plasma',vmin=ts[130],vmax=ts[150],s=10,depthshade=False)
ax2.set_title('window (21 frames ~2s): spatiotemporal input\n(color=time)',fontsize=10); ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('η'); ax2.view_init(28,-60)
# 3) top-down coverage vs full domain
ax3=fig.add_subplot(1,3,3)
ax3.add_patch(plt.Rectangle((gx.min(),gy.min()),gx.max()-gx.min(),gy.max()-gy.min(),fill=False,ls='--',ec='gray',label='reconstruction domain'))
sc=ax3.scatter(pts[:,0],pts[:,1],c=pts[:,2],cmap='RdBu_r',s=30,edgecolor='k',linewidth=0.3)
ax3.set_xlim(0,gx.max()+10); ax3.set_ylim(gy.min()-10,gy.max()+10); ax3.set_aspect('equal')
ax3.set_title('top-down: sparse forward-cone LiDAR\nvs full reconstruction domain',fontsize=10); ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.legend(fontsize=8)
plt.colorbar(sc,ax=ax3,label='η [m]',shrink=0.8)
fig.suptitle(f'INPUT: real ship-LiDAR scan ({EP.split("/")[-1]}) — ~44 pts/frame, forward cone x∈[10,57]m, over 240×200m domain',fontsize=12)
plt.tight_layout(); os.makedirs('results/gifs',exist_ok=True); plt.savefig('results/gifs/lidar_input_3d.png',dpi=115,bbox_inches='tight')
print('saved results/gifs/lidar_input_3d.png')
print(f'per-frame valid pts: {[int(vm[i].sum()) for i in range(130,151,5)]}')
print(f'window total pts (21 frames): {sum(int(vm[i].sum()) for i in range(130,151))}')
