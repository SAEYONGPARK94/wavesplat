"""NEUTRAL A4 — bit-identity check of the CURRENT working tree (which carries uncommitted
RX2_GROW_* gates in initializer/coarse_to_fine/sequential_update) against the recorded
s1s2 rows produced BEFORE those edits.  All gates unset == shipped behaviour must hold.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1S2 = os.path.abspath(os.path.join(HERE, '..', 's1s2_2026-08-05'))
sys.path.insert(0, S1S2)
for g in ('RX2_GROW_SUBTRACT', 'RX2_GROW_RECOMPUTE', 'RX2_GROW_TIGHT_SUPPRESS',
          'RX2_GROW_FITPHI'):
    os.environ.pop(g, None)

import run_s1s2 as R

CASES = [('S1-A0', 'episode_BF4_15comp_seed0_uniform.h5', 42),
         ('S1-A0', 'episode_BF5_30comp_seed0_uniform.h5', 123),
         ('S2-T1f', 'episode_BF4_15comp_seed0_uniform.h5', 42)]

ref = {}
for line in open(os.path.join(S1S2, 'results.jsonl')):
    r = json.loads(line)
    ref[r['job_id']] = r

out = []
for arm, ep, seed in CASES:
    jid = f'{arm}|{ep}|{seed}'
    got = R.run_one({'job_id': jid, 'arm': arm, 'episode': ep, 'seed': seed})
    exp = ref.get(jid, {})
    rec = {'job_id': jid}
    for k in ('nRMSE', 'init_nrmse', 'Pearson_rho', 'Variance_Ratio', 'N'):
        rec[f'ref_{k}'] = exp.get(k)
        rec[f'new_{k}'] = got.get(k)
        try:
            rec[f'bitid_{k}'] = (repr(float(exp[k])) == repr(float(got[k])))
        except Exception:
            rec[f'bitid_{k}'] = None
    print(json.dumps(rec), flush=True)
    out.append(rec)

json.dump(out, open(os.path.join(HERE, 'NEUTRAL_A4_bitid.json'), 'w'), indent=1)
