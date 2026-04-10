"""
run_q4_dispersion_residual.py
Q4 evidence: A1 (noDisp) estimates (kx, ky, ω) that drift off the
dispersion manifold, while Full stays on it by construction.

Output: amplitude-weighted relative residual
    R = Σᵢ |wᵢ| · |ωᵢ − √(g|kᵢ|tanh(|kᵢ|d))| / (ωᵢ · Σⱼ |wⱼ|)

For Full (use_dispersion=True) R ≡ 0 by construction.
For A1 a non-zero R quantifies the physical-validity cost of the
nRMSE gain reported in Table III.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from wavesplat.run import run_estimation

G = 9.81
DEPTH = 50.0

EPISODES = []
for bf in ['BF4', 'BF5']:
    for ncomp in ['5comp', '15comp', '30comp']:
        EPISODES.append(f'data/episode_{bf}_{ncomp}_seed0_uniform.h5')

SEEDS = [42, 123]

def dispersion_residual(params):
    kx = np.asarray(params['kx'])
    ky = np.asarray(params['ky'])
    w  = np.abs(np.asarray(params['w']))
    om = np.asarray(params['omega'])
    kmag = np.sqrt(kx**2 + ky**2) + 1e-12
    om_disp = np.sqrt(G * kmag * np.tanh(kmag * DEPTH))
    rel = np.abs(om - om_disp) / (om_disp + 1e-12)
    abs_hz = np.abs(om - om_disp) / (2*np.pi)
    wsum = w.sum() + 1e-12
    return {
        'rel_weighted': float(np.sum(w * rel) / wsum),
        'rel_mean':     float(np.mean(rel)),
        'rel_max':      float(np.max(rel)),
        'abs_weighted_hz': float(np.sum(w * abs_hz) / wsum),
        'n_components': int(len(kx)),
    }

results = {'Full': [], 'A1_noDisp': []}
print(f"Q4 dispersion residual: {len(EPISODES)} eps × {len(SEEDS)} seeds")
print('=' * 78)
for ep in EPISODES:
    if not os.path.exists(ep):
        continue
    name = os.path.basename(ep).replace('episode_', '').replace('.h5', '')
    for cfg_name, use_disp in [('Full', True), ('A1_noDisp', False)]:
        for seed in SEEDS:
            cfg = {
                'use_dispersion': use_disp, 'use_confidence': True,
                'init_method': 'v2_konly', 'adaptive_n': True,
                'tracking_window': 3, 'seed': seed,
            }
            try:
                r = run_estimation(ep, cfg, verbose=False)
                res = dispersion_residual(r['spectrum_params'])
                res.update({
                    'episode': name, 'seed': seed,
                    'avg_nRMSE': float(r['avg_nRMSE']),
                })
                results[cfg_name].append(res)
                print(f"{name:30s} {cfg_name:10s} s{seed} "
                      f"nRMSE={res['avg_nRMSE']:.3f} "
                      f"rel_w={res['rel_weighted']:.3f} "
                      f"abs_w={res['abs_weighted_hz']:.3f}Hz "
                      f"N={res['n_components']}", flush=True)
            except Exception as e:
                print(f"{name} {cfg_name} s{seed} ERROR: {e}", flush=True)

out = {
    'summary': {
        cfg: {
            'n_runs': len(v),
            'rel_weighted_mean': float(np.mean([r['rel_weighted'] for r in v])) if v else None,
            'rel_weighted_std':  float(np.std([r['rel_weighted'] for r in v])) if v else None,
            'abs_weighted_hz_mean': float(np.mean([r['abs_weighted_hz'] for r in v])) if v else None,
            'nRMSE_mean': float(np.mean([r['avg_nRMSE'] for r in v])) if v else None,
        }
        for cfg, v in results.items()
    },
    'runs': results,
}
path = 'results/q4_dispersion_residual.json'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print('\n=== SUMMARY ===')
print(json.dumps(out['summary'], indent=2))
print(f"\nSaved: {path}")
