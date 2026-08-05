"""Bit-identity control: X-BASE @ seeds {42,123,456} must reproduce tmp/s1s2_2026-08-05
arm S1-A0 exactly, proving the Rx-2 code edits are inert at their defaults."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1S2 = os.path.join(HERE, '..', 's1s2_2026-08-05', 'results.jsonl')
NEW = os.path.join(HERE, 'bitid.jsonl')
FIELDS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2',
          'Hs_error', 'init_nrmse', 'init_rmse', 'N', 'avg_RMSE_unobserved']
SPEC = ['detection_rate', 'energy_detection_rate', 'n_est', 'n_matched',
        'k_mag_rel_error_mean', 'amp_rel_error_mean', 'direction_error_deg_mean',
        'omega_rel_error_mean']

old = {}
for line in open(S1S2):
    r = json.loads(line)
    if r.get('arm') == 'S1-A0' and 'error' not in r:
        old[(r['episode'], r['seed'])] = r
new = {}
for line in open(NEW):
    r = json.loads(line)
    if r.get('cell') == 'X-BASE' and 'error' not in r:
        new[(r['episode'], r['seed'])] = r

keys = sorted(set(old) & set(new))
print(f'paired runs: {len(keys)}  (old {len(old)}, new {len(new)})')
n_cmp = 0
bad = []
for k in keys:
    a, b = old[k], new[k]
    for f in FIELDS:
        if f not in a or f not in b:
            continue
        n_cmp += 1
        if repr(a[f]) != repr(b[f]):
            bad.append((k, f, a[f], b[f]))
    for f in SPEC:
        if f in a.get('spec', {}) and f in b.get('spec', {}):
            n_cmp += 1
            if repr(a['spec'][f]) != repr(b['spec'][f]):
                bad.append((k, 'spec.' + f, a['spec'][f], b['spec'][f]))
print(f'compared {n_cmp} values across {len(keys)} runs')
if bad:
    print(f'MISMATCH {len(bad)}')
    for x in bad[:25]:
        print('  ', x)
    sys.exit(1)
print('BIT-IDENTITY PASS: 0 mismatches (repr-level exact)')
for k in keys:
    print(f"  {k[0].replace('episode_','').replace('_seed0_uniform.h5',''):<12} "
          f"seed {k[1]:<4} nRMSE {new[k]['nRMSE']:.10f}  init {new[k]['init_nrmse']:.10f}")
