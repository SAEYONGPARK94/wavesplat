#!/usr/bin/env python3
"""CON: column-level check of wave_components across episode-seeds 0..3 within a config."""
import h5py, numpy as np, os
DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'

def comps(fn):
    with h5py.File(os.path.join(DATA, fn), 'r') as f:
        return np.array(f['config/wave_components']), np.array(f['gt_params/spectrum_1d'])

for cfg in ['BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']:
    print('=' * 90)
    print(cfg)
    C = []; S = []
    for s in range(4):
        c, sp = comps(f'episode_{cfg}_seed{s}_uniform.h5')
        C.append(c); S.append(sp)
    C = np.stack(C)   # (4, N, 6)
    print('  wave_components shape per seed:', C[0].shape)
    for col in range(C.shape[2]):
        vals = C[:, :, col]
        allsame = np.allclose(vals[0], vals[1]) and np.allclose(vals[0], vals[2]) and np.allclose(vals[0], vals[3])
        sortsame = all(np.allclose(np.sort(vals[0]), np.sort(vals[i])) for i in range(1, 4))
        rng = (vals.min(), vals.max())
        print(f'   col{col}: identical-across-seeds={allsame}  identical-as-SET={sortsame}  '
              f'range=({rng[0]:.4f},{rng[1]:.4f})  seed0[:4]={np.round(vals[0][:4],4)}')
    sp_same = all(np.array_equal(S[0], S[i]) for i in range(1, 4))
    print(f'  gt_params/spectrum_1d identical across seeds: {sp_same}')
    if not sp_same:
        d = max(np.abs(S[0] - S[i]).max() for i in range(1, 4))
        print(f'    max |diff| = {d:.6g}   (col0 = freq? col1 = S(f)?)')
        print(f'    freq col identical: {all(np.array_equal(S[0][:,0], S[i][:,0]) for i in range(1,4))}')
        print(f'    S(f) col max diff : {max(np.abs(S[0][:,1]-S[i][:,1]).max() for i in range(1,4)):.6g}')
        print(f'    S(f) seed0 total  : {S[0][:,1].sum():.6g}  '
              f'others: {[round(float(S[i][:,1].sum()),6) for i in range(1,4)]}')
