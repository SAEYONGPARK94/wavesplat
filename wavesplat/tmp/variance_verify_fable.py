"""
variance_verify_fable.py -- independent re-derivation / attack on
tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md (author's estimator_variance.py).

Does NOT touch tmp/estimator_variance_results.json (author's raw data) or
tmp/restart_selection.py (concurrent, do-not-duplicate). Writes its own
files: tmp/variance_verify_fable_*.json / .log

Runs, in order (each gated so it can be re-invoked; results accumulate in JSON):
  D. reproducibility: same episode+seed run twice, bit-for-bit?
  C. mechanism probe: capture init_nrmse, final n_gaussians, spectrum_params,
     per-frame nRMSE trajectory for a few named seeds (good vs bad per author's
     table) on the worst episode.
  A. phase-freeze ablation: monkeypatch torch.rand to a FIXED generator
     (independent of config['seed']) and re-run the same 8 seeds. If phase
     draws are the only seed-driven randomness, CV should collapse.
  B. adaptive_n=False ablation: same 8 seeds, adaptive_n off.
  E. bimodality resolution: 16 additional fresh seeds on the worst episode.
"""
import sys, os, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from wavesplat.run import run_estimation, set_seed, DEFAULT_CONFIG

WORST = 'data/episode_BF3_2comp_seed0_uniform.h5'
SEEDS8 = [42, 1, 2, 3, 4, 5, 6, 7]
OUT = 'tmp/variance_verify_fable_results.json'

BASE_CFG = {
    'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
    'adaptive_n': True, 'tracking_window': 3,
}

res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, 'w'), indent=2, default=str)


def run_full(ep, seed, cfg_extra=None, tag=''):
    cfg = dict(BASE_CFG)
    if cfg_extra:
        cfg.update(cfg_extra)
    cfg['seed'] = seed
    t0 = time.time()
    r = run_estimation(ep, config=cfg)
    dt = time.time() - t0
    print(f'  [{tag}] {os.path.basename(ep)} seed={seed} adaptive_n={cfg.get("adaptive_n")} '
          f'-> nRMSE={r["avg_nRMSE"]:.4f} init_nrmse={r["init_nrmse"]:.4f} '
          f'n_gauss_final={r["n_gaussians"]} obs_loss={r["avg_obs_loss"]:.4f} [{dt:.0f}s]',
          flush=True)
    return r


# ---------------------------------------------------------------------------
# D. Reproducibility: same seed, same episode, run twice -> bit-identical?
# ---------------------------------------------------------------------------
if 'D_repro' not in res:
    print('=== D. Reproducibility check (seed=1, WORST episode, run x2) ===', flush=True)
    r1 = run_full(WORST, 1, tag='repro-run1')
    r2 = run_full(WORST, 1, tag='repro-run2')
    res['D_repro'] = {
        'run1_avg_nRMSE': float(r1['avg_nRMSE']), 'run2_avg_nRMSE': float(r2['avg_nRMSE']),
        'identical': bool(r1['avg_nRMSE'] == r2['avg_nRMSE']),
        'diff': float(r1['avg_nRMSE'] - r2['avg_nRMSE']),
        'run1_n_gaussians': int(r1['n_gaussians']), 'run2_n_gaussians': int(r2['n_gaussians']),
        'run1_spectrum': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                           for k, v in r1['spectrum_params'].items()},
        'run2_spectrum': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                           for k, v in r2['spectrum_params'].items()},
    }
    save()
print('D_repro:', json.dumps({k: v for k, v in res['D_repro'].items()
                               if 'spectrum' not in k}, indent=2))

# ---------------------------------------------------------------------------
# C. Mechanism probe: good vs bad seeds on WORST episode, full detail
#    author's table: seed42=0.0524 (good), seed1=0.1682 (bad), seed6=0.1676 (bad),
#    seed7=0.0590 (good)
# ---------------------------------------------------------------------------
if 'C_mechanism' not in res:
    print('=== C. Mechanism probe (good vs bad seeds, full detail) ===', flush=True)
    res['C_mechanism'] = {}
    for s, label in [(42, 'good'), (7, 'good'), (1, 'bad'), (6, 'bad')]:
        r = run_full(WORST, s, tag=f'mech-{label}-{s}')
        frame_nrmse = [float(fr.get('nRMSE')) for fr in r['frame_results']]
        res['C_mechanism'][str(s)] = {
            'label': label,
            'avg_nRMSE': float(r['avg_nRMSE']),
            'init_nrmse': float(r['init_nrmse']),
            'init_iters': int(r['init_iters']),
            'n_gaussians_final': int(r['n_gaussians']),
            'avg_obs_loss': float(r['avg_obs_loss']),
            'T_converge': r['T_converge'],
            'frame_nrmse_first5': frame_nrmse[:5],
            'frame_nrmse_last5': frame_nrmse[-5:],
            'frame_nrmse_min': min(frame_nrmse), 'frame_nrmse_max': max(frame_nrmse),
            'spectrum_params': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                 for k, v in r['spectrum_params'].items()},
        }
        save()
print('C_mechanism summary:')
for s, d in res['C_mechanism'].items():
    print(f"  seed={s:<3} ({d['label']:<4}) avg_nRMSE={d['avg_nRMSE']:.4f} "
          f"init_nrmse={d['init_nrmse']:.4f} n_gauss_final={d['n_gaussians_final']} "
          f"obs_loss={d['avg_obs_loss']:.4f} T_conv={d['T_converge']}")

# ---------------------------------------------------------------------------
# A. Phase-freeze ablation: patch torch.rand to a fixed, seed-INDEPENDENT
#    generator. If phase draws are the sole random-per-seed ingredient in
#    v2_konly, CV should collapse toward 0.
# ---------------------------------------------------------------------------
if 'A_phase_freeze' not in res:
    print('=== A. Phase-freeze ablation (torch.rand pinned, seed varies) ===', flush=True)
    _real_rand = torch.rand

    def _fixed_rand(*args, **kwargs):
        gen = torch.Generator()
        gen.manual_seed(20260723)
        device = kwargs.pop('device', None)
        dtype = kwargs.pop('dtype', None)
        kwargs.pop('generator', None)
        t = _real_rand(*args, generator=gen, **kwargs)
        if device is not None:
            t = t.to(device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    torch.rand = _fixed_rand
    vals = {}
    try:
        for s in SEEDS8:
            r = run_full(WORST, s, tag='phase-freeze')
            vals[str(s)] = r['avg_nRMSE']
            save_partial = res.setdefault('A_phase_freeze', {})
            save_partial[str(s)] = r['avg_nRMSE']
            save()
    finally:
        torch.rand = _real_rand
    a = np.array(list(vals.values()))
    print(f'A_phase_freeze: mean={a.mean():.4f} sd={a.std(ddof=1):.4f} '
          f'cv={100*a.std(ddof=1)/a.mean():.1f}% min={a.min():.4f} max={a.max():.4f}')
print('A_phase_freeze raw:', res.get('A_phase_freeze'))

# ---------------------------------------------------------------------------
# B. adaptive_n=False ablation, same 8 seeds, real randomness intact
# ---------------------------------------------------------------------------
if 'B_no_adaptive' not in res:
    print('=== B. adaptive_n=False ablation ===', flush=True)
    vals = {}
    for s in SEEDS8:
        r = run_full(WORST, s, cfg_extra={'adaptive_n': False}, tag='no-adaptive')
        vals[str(s)] = {'avg_nRMSE': r['avg_nRMSE'], 'n_gaussians': r['n_gaussians']}
        res.setdefault('B_no_adaptive', {})[str(s)] = vals[str(s)]
        save()
    a = np.array([v['avg_nRMSE'] for v in vals.values()])
    print(f'B_no_adaptive: mean={a.mean():.4f} sd={a.std(ddof=1):.4f} '
          f'cv={100*a.std(ddof=1)/a.mean():.1f}% min={a.min():.4f} max={a.max():.4f}')
print('B_no_adaptive raw:', res.get('B_no_adaptive'))

# ---------------------------------------------------------------------------
# E. Extended-n bimodality resolution: 16 more fresh seeds on WORST episode
# ---------------------------------------------------------------------------
EXTRA_SEEDS = list(range(10, 26))
if 'E_extended' not in res:
    res['E_extended'] = {}
if len(res['E_extended']) < len(EXTRA_SEEDS):
    print('=== E. Extended bimodality resolution (16 more seeds) ===', flush=True)
    for s in EXTRA_SEEDS:
        if str(s) in res['E_extended']:
            continue
        r = run_full(WORST, s, tag='extended')
        res['E_extended'][str(s)] = r['avg_nRMSE']
        save()
orig = json.load(open('tmp/estimator_variance_results.json'))
orig_worst = orig[os.path.basename(WORST)]  # author's original 8 seeds, read-only
all_e = {**orig_worst, **res.get('E_extended', {})}
print('E combined n =', len(all_e), 'sorted values:', sorted(all_e.values()))

print('\nALL DONE.')
