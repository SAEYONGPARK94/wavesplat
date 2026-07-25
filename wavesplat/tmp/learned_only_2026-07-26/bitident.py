"""Bit-identity re-verify: (a) default learned path (no anchor flag) reproduces the cached arm
value; (b) peak_anchor_filter with anchor_frac=None is a strict no-op."""
import os, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import wavesplat.optimizer.sequential_update as su
su.TRACK_PHASE_PREDICT = False
from wavesplat.run import run_estimation
from wavesplat.optimizer.learned_init import peak_anchor_filter, pool_and_select

FINE = 'data_finemesh_2026-07-23/'
ep = 'episode_BF3_2comp_seed0_slope.h5'; est_seed = 42; diff_seed = 0
REF = 0.045643081643

# cached raw atoms
cache = {}
for src in ('tmp/wp_integration_2026-07-25/atoms_cache.jsonl',
            'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'):
    for l in open(src):
        d = json.loads(l)
        if d.get('kx') is not None and d.get('err') is None:
            cache[(d['episode'], d['diff_seed'])] = dict(kx=d['kx'], ky=d['ky'], energy=d['energy'])
entry = cache[(ep, diff_seed)]

# (a) default learned (no anchor) -> must reproduce REF bit-identically
cfg = {'adaptive_n': True, 'seed': est_seed, 'device': 'cpu', 'init_method': 'learned',
       'learned_diffusion_seed': diff_seed, 'learned_init_iters': 500, 'learned_atoms': entry}
r = run_estimation(FINE + ep, config=cfg)
print(f"(a) default learned avg_nRMSE = {r['avg_nRMSE']:.12f}  ref = {REF:.12f}  "
      f"|diff| = {abs(r['avg_nRMSE']-REF):.2e}  -> {'PASS' if abs(r['avg_nRMSE']-REF)<1e-9 else 'FAIL'}")

# (a2) explicit anchor flag OFF -> also reproduce
cfg2 = dict(cfg); cfg2['learned_peak_anchor'] = False
r2 = run_estimation(FINE + ep, config=cfg2)
print(f"(a2) anchor=False        avg_nRMSE = {r2['avg_nRMSE']:.12f}  "
      f"|diff| = {abs(r2['avg_nRMSE']-REF):.2e}  -> {'PASS' if abs(r2['avg_nRMSE']-REF)<1e-9 else 'FAIL'}")

# (b) peak_anchor_filter anchor_frac=None is a strict no-op on the pooled atoms
sk, sy, se = pool_and_select(np.asarray(entry['kx']), np.asarray(entry['ky']), np.asarray(entry['energy']))
fk, fy, fe = peak_anchor_filter(sk, sy, se, None, None, anchor_frac=None)
noop = np.array_equal(sk, fk) and np.array_equal(sy, fy) and np.array_equal(se, fe)
print(f"(b) anchor_frac=None no-op: {'PASS' if noop else 'FAIL'} (n_in={len(sk)} n_out={len(fk)})")
