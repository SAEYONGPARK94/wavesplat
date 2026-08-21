#!/usr/bin/env python3
"""CON: verify the dependence structure of the 16 block-C episodes and the 8 block-S episodes."""
import h5py, numpy as np, os, itertools, json

DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'

def dump(fn):
    p = os.path.join(DATA, fn)
    out = {}
    with h5py.File(p, 'r') as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj
        f.visititems(walk)
        keys = sorted(out.keys())
        attrs = dict(f.attrs)
        arrs = {}
        for k in keys:
            try:
                arrs[k] = np.array(out[k])
            except Exception:
                pass
    return keys, attrs, arrs

fn0 = 'episode_BF4_15comp_seed0_uniform.h5'
keys, attrs, arrs = dump(fn0)
print('=== structure of', fn0, '===')
for k in keys:
    print(f'  {k:40s} shape={arrs[k].shape} dtype={arrs[k].dtype}')
print('  attrs:', {k: (v if np.ndim(v) == 0 else f'<{np.shape(v)}>') for k, v in attrs.items()})
print()

# --- claim A: episode-seeds 0..3 share amplitude & frequency spectrum, differ only in direction
CAND = [k for k in keys if any(t in k.lower() for t in
        ('amp', 'freq', 'omega', 'dir', 'theta', 'phase', 'phi', 'k_', 'wavenum', 'comp'))]
print('=== claim A: do episode-seeds 0..3 share amplitude/frequency and differ in direction? ===')
for cfg in ['BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']:
    print(f'-- {cfg} --')
    ref = None
    for s in range(4):
        fn = f'episode_{cfg}_seed{s}_uniform.h5'
        _, a, ar = dump(fn)
        row = {}
        for k in CAND:
            if k in ar and ar[k].ndim <= 2 and ar[k].size < 200:
                row[k] = ar[k]
        if ref is None:
            ref = row
            print(f'   seed0 fields: ' + ', '.join(f'{k}={np.round(np.ravel(v)[:6],4)}' for k, v in row.items()))
        else:
            same = [k for k in row if k in ref and ref[k].shape == row[k].shape
                    and np.allclose(ref[k], row[k])]
            diff = [k for k in row if k not in same]
            print(f'   seed{s}: identical-to-seed0 = {same} ; differs = {diff}')
print()

# --- claim B: slope twin has bit-identical gt_wave_field
print('=== claim B: slope vs uniform twin, gt field identical? ===')
for base in ['BF3_2comp_seed0', 'BF3_5comp_seed0', 'BF4_15comp_seed0']:
    _, _, au = dump(f'episode_{base}_uniform.h5')
    _, _, asl = dump(f'episode_{base}_slope.h5')
    for k in sorted(set(au) & set(asl)):
        if au[k].shape != asl[k].shape:
            print(f'  {base:20s} {k:30s} SHAPE DIFF {au[k].shape} vs {asl[k].shape}'); continue
        try:
            eq = np.array_equal(au[k], asl[k])
        except Exception:
            eq = None
        print(f'  {base:20s} {k:30s} bit-identical={eq}')
    print()
