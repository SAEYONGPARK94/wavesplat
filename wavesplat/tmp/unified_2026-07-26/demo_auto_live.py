"""UNIFIED demo/validation: run init_method='auto' live and confirm the single-config
path (a) routes correctly from observations alone and (b) reproduces the assembled arm.

Two modes:
  CACHED: auto with precomputed atoms (learned_diffusion_seed=0, est_seed=42). For crowded
    episodes it must route 'learned' and its avg_nRMSE must match the Stage-1 learned (42,0)
    lane bit-identically; for sparse it must route 'v2_konly' and match batchC v2 seed=42.
  LIVE  : auto with NO cache -> the router runs the GPU diffusion proposal itself (the true
    single-entry-point path), diffusion_seed=0. Must reproduce the cached routing + result.

TRACK_PHASE_PREDICT=False (arm protocol). Writes demo_auto.json.
  python3 tmp/unified_2026-07-26/demo_auto_live.py
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

FINE = 'data_finemesh_2026-07-23/'
S1_ARMS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
S1_CACHE = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'

CROWDED_DEMO = [
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_slope.h5',
    'episode_BF4_30comp_seed3_slope.h5',  'episode_BF5_30comp_seed3_uniform.h5',
    'episode_BF5_15comp_seed4_slope.h5',  'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed1_uniform.h5', 'episode_BF4_15comp_seed2_uniform.h5',
    'episode_BF5_30comp_seed0_slope.h5',  'episode_BF4_30comp_seed1_slope.h5',
]
SPARSE_DEMO = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
]
LIVE_DEMO = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_uniform.h5']


def main():
    # references
    s1_learned = {}   # (ep) -> avg_nRMSE at (42,0)
    for l in open(S1_ARMS):
        d = json.loads(l)
        if d.get('arm') == 'learned' and d.get('est_seed') == 42 and d.get('diff_seed') == 0 \
                and d.get('err') is None:
            s1_learned[d['episode']] = d['avg_nRMSE']
    cache0 = {}
    for l in open(S1_CACHE):
        d = json.loads(l)
        if d.get('diff_seed') == 0 and d.get('kx') is not None:
            cache0[d['episode']] = (dict(kx=d['kx'], ky=d['ky'], energy=d['energy']),
                                    d['n_selected'])
    v2_42 = {}
    for l in open(BATCHC):
        d = json.loads(l)
        if d['seed'] == 42:
            v2_42[d['episode']] = d['nRMSE']

    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False
    from wavesplat.run import run_estimation

    rows = []

    def run_auto(ep, cached):
        cfg = {'adaptive_n': True, 'seed': 42, 'device': 'cpu', 'init_method': 'auto',
               'learned_diffusion_seed': 0}
        if cached:
            cfg['learned_atoms'] = dict(**cache0[ep][0])   # copy raw atoms
        t0 = time.time()
        r = run_estimation(FINE + ep, config=cfg)
        return r, time.time() - t0

    print("=== CACHED-atom auto routing validation (est_seed=42, diff_seed=0) ===")
    for ep in CROWDED_DEMO + SPARSE_DEMO:
        r, sec = run_auto(ep, cached=True)
        sel = r['auto_selected']; ns = r['auto_n_selected']
        ref = s1_learned.get(ep) if sel == 'learned' else v2_42.get(ep)
        diff = abs(r['avg_nRMSE'] - ref) if ref is not None else None
        rows.append(dict(mode='cached', episode=ep, auto_selected=sel, auto_n_selected=ns,
                         avg_nRMSE=r['avg_nRMSE'], ref=ref, ref_diff=diff, sec=sec))
        print(f"  {ep:40s} n_sel={ns:2d} -> {sel:9s} avg={r['avg_nRMSE']:.4f}"
              f" ref={ref:.4f} |diff|={diff:.2e}")

    print("\n=== FULLY-LIVE auto (router runs GPU proposal itself, no cache) ===")
    for ep in LIVE_DEMO:
        r, sec = run_auto(ep, cached=False)
        sel = r['auto_selected']; ns = r['auto_n_selected']
        ref = s1_learned.get(ep) if sel == 'learned' else v2_42.get(ep)
        diff = abs(r['avg_nRMSE'] - ref) if ref is not None else None
        rows.append(dict(mode='live', episode=ep, auto_selected=sel, auto_n_selected=ns,
                         avg_nRMSE=r['avg_nRMSE'], ref=ref, ref_diff=diff, sec=sec))
        print(f"  {ep:40s} n_sel={ns:2d} -> {sel:9s} avg={r['avg_nRMSE']:.4f}"
              f" ref={ref:.4f} |diff|={diff:.2e}  ({sec:.0f}s, GPU proposal in-path)")

    # summary: routing correctness + reproduction
    crowded_ok = all(r['auto_selected'] == 'learned' for r in rows
                     if 'comp' not in r and ('15comp' in r['episode'] or '30comp' in r['episode']))
    sparse_ok = all(r['auto_selected'] == 'v2_konly' for r in rows
                    if ('1comp' in r['episode'] or '2comp' in r['episode']))
    max_cached_diff = max((r['ref_diff'] for r in rows if r['mode'] == 'cached'
                           and r['ref_diff'] is not None), default=None)
    max_live_diff = max((r['ref_diff'] for r in rows if r['mode'] == 'live'
                         and r['ref_diff'] is not None), default=None)
    summary = dict(crowded_all_route_learned=crowded_ok, sparse_all_route_v2=sparse_ok,
                   max_cached_reproduction_diff=max_cached_diff,
                   max_live_reproduction_diff=max_live_diff)
    json.dump(dict(rows=rows, summary=summary), open('tmp/unified_2026-07-26/demo_auto.json', 'w'),
              indent=1)
    print("\nSUMMARY:", summary)


if __name__ == '__main__':
    main()
