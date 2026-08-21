#!/usr/bin/env python3
"""CON: is |k| shared across episode-seeds (only direction varies)?  Is the phase column
shared ACROSS configs at the same episode-seed (a second, undeclared dependence axis)?"""
import h5py, numpy as np, os
DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'
def comps(fn):
    with h5py.File(os.path.join(DATA, fn), 'r') as f:
        return np.array(f['config/wave_components'])

CFGS = ['BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']
print('=== |k| and direction across episode-seeds within a config ===')
for cfg in CFGS:
    C = np.stack([comps(f'episode_{cfg}_seed{s}_uniform.h5') for s in range(4)])
    kmag = np.hypot(C[:, :, 2], C[:, :, 3])
    th = np.degrees(np.arctan2(C[:, :, 3], C[:, :, 2]))
    same_kmag = all(np.allclose(kmag[0], kmag[i]) for i in range(1, 4))
    print(f'  {cfg}: |k| identical across seeds = {same_kmag}   '
          f'max|dkmag| = {max(np.abs(kmag[0]-kmag[i]).max() for i in range(1,4)):.3g}')
    print(f'     theta[deg] seed0[:5]={np.round(th[0][:5],1)}  seed1[:5]={np.round(th[1][:5],1)}  '
          f'seed3[:5]={np.round(th[3][:5],1)}')

print()
print('=== is col4 (phase) shared ACROSS configs at the same episode-seed? ===')
for s in range(4):
    ph = [comps(f'episode_{c}_seed{s}_uniform.h5')[:, 4] for c in CFGS]
    n = min(len(p) for p in ph)
    pairs = [(CFGS[i], CFGS[j], bool(np.allclose(ph[i][:n], ph[j][:n])))
             for i in range(4) for j in range(i + 1, 4)]
    print(f'  seed{s}: first {n} phases equal across config pairs -> '
          + ', '.join(f'{a[:6]}~{b[:6]}:{v}' for a, b, v in pairs))
    print(f'     BF4_15 phases[:5]={np.round(ph[0][:5],4)}   BF5_30 phases[:5]={np.round(ph[3][:5],4)}')
