"""
Does over-componentization (the La Jument tuning issue) also inflate the SYNTHETIC
headline? Compare default iterative_relax vs model-order-controlled RELAX on episodes
with KNOWN true component count.
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
from wavesplat.evaluation.metrics import pearson_correlation, variance_ratio

def iterative_default(P, z, depth, target, n_outer=3):
    kx, ky = np.zeros(0), np.zeros(0); fld=None
    for _ in range(n_outer):
        kx, ky = relax_step(kx, ky, P, z, depth)                 # default max_comp=32, prune 0.05
        kx, ky, fld = refine_params(kx, ky, P, z, depth, target)
    return fld, len(kx)

def iterative_controlled(P, z, depth, target, true_N, n_outer=3):
    # cap model order near a modest multiple + stronger prune (proxy for BIC/CV control)
    kx, ky = np.zeros(0), np.zeros(0); fld=None
    for _ in range(n_outer):
        kx, ky = relax_step(kx, ky, P, z, depth, max_comp=min(true_N+2, 12), prune_frac=0.12)
        kx, ky, fld = refine_params(kx, ky, P, z, depth, target, iters=400)
    return fld, len(kx)

def score(fld, gt, Hs):
    return (float(np.sqrt(np.mean((fld-gt)**2))/Hs), pearson_correlation(fld,gt), float(variance_ratio(fld,gt)))

tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
true_N = {'BF4_15c_s0':15,'BF4_30c_s0':30,'BF5_30c_s1':30}
i_star = 250; K_TP=2.2
print(f"{'episode':12s} {'true_N':>6} | {'default N':>9} {'nRMSE':>6} {'corr':>5} {'VR':>5} | {'ctrl N':>6} {'nRMSE':>6} {'corr':>5} {'VR':>5}")
for tag in tags:
    L = EpisodeDataLoader('data/'+tag_to_file(tag)); depth=L.depth; Hs=L.Hs_true
    n_back = int(round(K_TP*L.Tp_true/0.1))
    fstar = L[i_star]
    target = {'gx':L.gt_grid_x.numpy(),'gy':L.gt_grid_y.numpy(),'t':float(fstar.t),'gt':fstar.gt_field.numpy(),'Hs':Hs}
    P,z = gather_frames(L, i_star, n_back)
    fd, Nd = iterative_default(P.astype(np.float32), z.astype(np.float32), depth, target)
    nr_d,cr_d,vr_d = score(fd, target['gt'], Hs)
    fc, Nc = iterative_controlled(P.astype(np.float32), z.astype(np.float32), depth, target, true_N[tag])
    nr_c,cr_c,vr_c = score(fc, target['gt'], Hs)
    print(f"{tag:12s} {true_N[tag]:6d} | {Nd:9d} {nr_d:6.3f} {cr_d:5.2f} {vr_d:5.2f} | {Nc:6d} {nr_c:6.3f} {cr_c:5.2f} {vr_c:5.2f}")
