"""Additive-code validation: (A) learned path with no merge is bit-identical to recorded
learned lanes; (B) init_method='auto' + guard routes/flips as designed and a guard-flip
reproduces the pure-v2 lane bit-identically.

  OMP_NUM_THREADS=1 python3 tmp/unified_guard_2026-07-26/validate_code.py
"""
import os, json, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np
import wavesplat.optimizer.sequential_update as su
su.TRACK_PHASE_PREDICT = False
from wavesplat.run import run_estimation

FINE = 'data_finemesh_2026-07-23/'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

# caches (raw atoms) + recorded arms
CACHE = {}
for src in ('tmp/wp_integration_2026-07-25/atoms_cache.jsonl',
            'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'):
    for l in open(src):
        d = json.loads(l)
        if d.get('kx') is not None and d.get('err') is None:
            CACHE[(d['episode'], d['diff_seed'])] = dict(kx=d['kx'], ky=d['ky'], energy=d['energy'])
LEARNED = {}
for src in ('tmp/wp_integration_2026-07-25/arm_runs.jsonl',
            'tmp/unified_2026-07-26/arm_runs_extra.jsonl'):
    for l in open(src):
        d = json.loads(l)
        if d.get('arm') == 'learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
            LEARNED[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
V2 = {}
for l in open('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'):
    d = json.loads(l); V2[(d['episode'], d['seed'])] = d['nRMSE']


def learned_run(ep, est_seed, diff_seed, merge_frac=None):
    cfg = {'adaptive_n': True, 'seed': est_seed, 'device': 'cpu', 'init_method': 'learned',
           'learned_diffusion_seed': diff_seed, 'learned_init_iters': 500,
           'learned_atoms': CACHE[(ep, diff_seed)]}
    if merge_frac is not None:
        cfg['learned_merge_frac'] = merge_frac
    return run_estimation(FINE + ep, config=cfg)


print("="*80)
print("(A) LEARNED (no merge) bit-identity vs recorded arm_runs")
print("="*80)
checks = [('episode_BF4_15comp_seed0_uniform.h5', 42, 0),
          ('episode_BF4_30comp_seed1_slope.h5', 1, 1),
          ('episode_BF3_5comp_seed0_uniform.h5', 2, 2)]
allok = True
for ep, es, ds in checks:
    if (ep, es, ds) not in LEARNED:
        print(f"  skip {ep} e{es}d{ds} (no recorded lane)"); continue
    out = learned_run(ep, es, ds)
    rec = LEARNED[(ep, es, ds)]
    ok = f"{float(out['avg_nRMSE']):.10f}" == f"{rec:.10f}"
    allok &= ok
    print(f"  {'OK ' if ok else 'DIFF'} {ep:34s} e{es}d{ds} rec={rec:.10f} got={float(out['avg_nRMSE']):.10f}")
print("LEARNED-NO-MERGE BIT-IDENTICAL:", allok)

print("\n" + "="*80)
print("(B) init_method='auto' + guard: routing + guard-flip reproduces pure-v2 lane")
print("="*80)
# BF2_1comp_seed0_slope: proposal n_sel~30, bf_ratio~0.024 -> guard should FIRE -> v2
# BF4_15comp_seed0_uniform: crowded, n_sel high, bf_ratio high -> route learned, guard ok
cases = [('episode_BF2_1comp_seed0_slope.h5', 42, 0, 'v2_konly', True),
         ('episode_BF4_15comp_seed0_uniform.h5', 42, 0, 'learned', False)]
for ep, es, ds, exp_sel, exp_fire in cases:
    from wavesplat.data_loader import EpisodeDataLoader
    L = EpisodeDataLoader(FINE + ep)
    cfg = {'adaptive_n': True, 'seed': es, 'device': 'cpu', 'init_method': 'auto',
           'learned_diffusion_seed': ds, 'learned_init_iters': 500,
           'learned_atoms': dict(CACHE[(ep, ds)])}
    upd = su.SequentialUpdater(cfg)
    fr = L.get_init_window(2.0)
    upd.initialize(fr)
    sel = upd._auto_selected; fired = upd._auto_guard_fired
    bfr = upd._auto_bf_peak_ratio; nsel = upd._auto_n_selected
    ok = (sel == exp_sel) and (fired == exp_fire)
    print(f"  {'OK ' if ok else 'BAD'} {ep:34s} n_sel={nsel} bf_ratio={bfr:.4f} "
          f"selected={sel} guard_fired={fired} (expected {exp_sel}, fire={exp_fire})")

# guard-flip result == pure-v2 lane bit-identical (full run_estimation through auto)
print("\n  guard-flip -> pure-v2 lane bit-identity (full run_estimation):")
ep = 'episode_BF2_1comp_seed0_slope.h5'
allok2 = True
for es, ds in LANES:
    cfg = {'adaptive_n': True, 'seed': es, 'device': 'cpu', 'init_method': 'auto',
           'learned_diffusion_seed': ds, 'learned_init_iters': 500,
           'learned_atoms': dict(CACHE[(ep, ds)])}
    out = run_estimation(FINE + ep, config=cfg)
    ref = V2.get((ep, es))
    ok = ref is not None and f"{float(out['avg_nRMSE']):.10f}" == f"{ref:.10f}"
    allok2 &= ok
    print(f"    {'OK ' if ok else 'DIFF'} e{es}d{ds} auto={float(out['avg_nRMSE']):.10f} pure_v2={ref:.10f}")
print("  GUARD-FLIP == PURE-V2 BIT-IDENTICAL:", allok2)
