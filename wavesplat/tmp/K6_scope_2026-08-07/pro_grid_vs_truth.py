"""PRO (6): quantify the COARSE-GRID / SUPPRESSION geometry against the true k-spectra.
Pure geometry from the episode HDF5 + the shipped constants. No estimation run. Read-only."""
import os, re
import numpy as np, h5py

DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'
CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
ALL12 = NEW8 + CROWDED4

# shipped constants, read off the source (initializer.py / coarse_to_fine.py)
SITES = {'init v2 (initializer:456)': dict(rng=4.0, n=81),
         'growth  (coarse_to_fine:218)': dict(rng=4.0, n=61)}
MIN_DK = 0.15
NARROW = 0.8   # G1

def dk(rng, n): return 2 * rng / (n - 1)

print('SHIPPED COARSE-GRID GEOMETRY')
for nm, s in SITES.items():
    d = dk(s['rng'], s['n'])
    sup = int(MIN_DK / d) + 1
    print(f'  {nm:30s} range +-{s["rng"]}  n={s["n"]}  dk={d:.4f}  '
          f'suppress=+-{sup} cells = +-{sup*d:.3f} rad/m  (box {2*sup*d:.3f} wide)')
d_narrow = dk(NARROW, 81)
sup_n = int(MIN_DK / d_narrow) + 1
print(f'  {"G1 NARROW at init site":30s} range +-{NARROW}   n=81  dk={d_narrow:.4f}  '
      f'suppress=+-{sup_n} cells = +-{sup_n*d_narrow:.3f} rad/m  (box {2*sup_n*d_narrow:.3f} wide)')

print('\n' + '=' * 122)
print('TRUE SPECTRUM vs GRID   —   "how many true components can the greedy possibly emit?"')
print('  kmax      = max |k| of true components  (the whole spectrum lives in a disc of this radius)')
print('  cells_occ = distinct COARSE CELLS (dk=0.1) the true components fall in  <= resolvable atoms')
print('  packbnd   = disc area / suppression-box area = HARD CAP on greedy picks at that site')
print('=' * 122)
hdr = (f'{"episode":30s}{"Ngt":>4s}{"kmax":>7s}{"cells@.100":>11s}{"cells@.133":>11s}'
       f'{"cells@.020":>11s}{"pack@.100":>10s}{"pack@.133":>10s}{"pack@.020":>10s}')
print(hdr)
rowsum = []
for e in ALL12:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        wc = np.array(f['config/wave_components'])
    kx, ky = wc[:, 2], wc[:, 3]
    n = len(kx); kmax = float(np.hypot(kx, ky).max())
    out = [f'{e[8:-3]:30s}{n:4d}{kmax:7.3f}']
    cells, packs = [], []
    for d in (0.1, 8/60, dk(NARROW, 81)):
        occ = len({(int(round(a / d)), int(round(b / d))) for a, b in zip(kx, ky)})
        cells.append(occ)
        sup = int(MIN_DK / d) + 1
        box = 2 * sup * d
        packs.append(np.pi * kmax ** 2 / box ** 2)
    out += [f'{c:11d}' for c in cells] + [f'{p:10.1f}' for p in packs]
    print(''.join(out))
    rowsum.append((e, n, kmax, cells, packs))

print('\n' + '=' * 122)
print('READING')
print('=' * 122)
for e, n, kmax, cells, packs in rowsum:
    if n >= 15:
        print(f'  {e[8:-3]:28s} Ngt={n:2d} : shipped init grid resolves only {cells[0]:2d} distinct cells '
              f'({cells[0]/n*100:4.0f}% of Ngt); suppression box alone caps greedy at ~{packs[0]:.1f} picks. '
              f'NARROW: {cells[2]:2d} cells / cap ~{packs[2]:.1f}.')
