"""PRO (6): the LEARNED arm (which is what actually ships in crowded seas) pools its
diffusion atoms through `pool_and_select(dedup_dk=0.03)` -- an L-inf box min-separation.
Question: does that box merge components that are genuinely distinct in the truth?

Pure geometry on the true spectra + the shipped constant. No estimation, no GT leakage
into any arm -- this is an audit of a constant against the physics it is applied to."""
import os, re
import numpy as np, h5py

DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'
DEDUP = 0.03            # learned_init.pool_and_select default
MAXN = 30
EPS = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
       'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_uniform.h5',
       'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
       'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

print('=' * 112)
print(f'SHIPPED LEARNED-ARM DEDUP: keep atom only if |dkx| >= {DEDUP} OR |dky| >= {DEDUP}'
      f'  (L-inf box, half-width {DEDUP})')
print('  survivors = greedy strongest-first application of that SAME rule to the TRUE'
      ' components (upper bound on what the pooler can emit even from a PERFECT proposal)')
print('=' * 112)
print(f'{"episode":30s}{"Ngt":>4s}{"pairs merged":>14s}{"survivors":>11s}{"surv%":>8s}'
      f'{"lost E%":>9s}{"medNN":>9s}')
for e in EPS:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        wc = np.array(f['config/wave_components'])
    A, kx, ky = wc[:, 0], wc[:, 2], wc[:, 3]
    n = len(A)
    dx = np.abs(kx[:, None] - kx[None, :]); dy = np.abs(ky[:, None] - ky[None, :])
    box = (dx < DEDUP) & (dy < DEDUP)
    np.fill_diagonal(box, False)
    merged_pairs = int(box.sum() // 2)
    # greedy strongest-first, exactly as pool_and_select does
    order = np.argsort(-A)
    keep = []
    for i in order:
        if any(abs(kx[i] - kx[j]) < DEDUP and abs(ky[i] - ky[j]) < DEDUP for j in keep):
            continue
        keep.append(i)
        if len(keep) >= MAXN:
            break
    lostE = 1 - (A[keep] ** 2).sum() / (A ** 2).sum()
    P = np.stack([kx, ky], 1)
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    nn = np.median(D.min(1)) if n > 1 else float('nan')
    print(f'{e[8:-3]:30s}{n:4d}{merged_pairs:14d}{len(keep):11d}'
          f'{len(keep)/n*100:7.0f}%{lostE*100:8.2f}%{nn:9.4f}')

print('\n' + '=' * 112)
print('READING: the dedup box is a HARD CEILING on the shipped crowded arm.')
print('  Even a PERFECT diffusion proposal cannot emit more components than "survivors".')
print('  Compare to the measured AUTO detection_rate on the same episodes (B2 raw):')
meas = {'BF4_15comp_seed0_uniform': 80.0, 'BF4_30comp_seed0_uniform': 50.0,
        'BF5_15comp_seed0_uniform': 80.0, 'BF5_30comp_seed0_uniform': 66.7}
for k, v in meas.items():
    print(f'    {k:30s} AUTO det = {v:5.1f}%')
