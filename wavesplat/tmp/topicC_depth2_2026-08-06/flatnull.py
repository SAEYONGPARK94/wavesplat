import h5py, numpy as np, os, json
eps=['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
     'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
base='data_finemesh_2026-07-23'; out={}
for e in eps:
    with h5py.File(os.path.join(base,e),'r') as f:
        Hs=float(f['gt_params'].attrs['Hs_true']); Tp=float(f['gt_params'].attrs['Tp_true'])
        ts=f['frames/timestamps'][:]; gt=f['frames/gt_wave_field']; n=len(ts); tinit=2.0*Tp
        flat=np.array([np.sqrt(np.mean(gt[i]**2))/Hs for i in range(n)])
        trk=[i for i in range(n) if ts[i]>tinit]; li=[i for i in range(n) if ts[i]<=tinit][-1]
        out[e]=dict(flat_trk=flat[trk].tolist(), flat_init=float(flat[li]), Hs=Hs, Tp=Tp, n_trk=len(trk))
json.dump(out,open('tmp/topicC_depth2_flatframes.json','w'))
