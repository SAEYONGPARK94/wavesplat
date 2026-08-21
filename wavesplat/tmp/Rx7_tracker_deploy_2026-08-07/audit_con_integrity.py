"""CON audit — integrity checks re-run independently, plus pre-registration compliance."""
import hashlib, json, os, subprocess
import statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(f'{HERE}/results.jsonl') if l.strip()]
ok = [r for r in rows if 'error' not in r]
R = {(r['cell'], r['episode'], r['seed']): r for r in ok}
EPS = sorted({r['episode'] for r in ok})
SEEDS = sorted({r['seed'] for r in ok})
CELLS = sorted({r['cell'] for r in ok})

print('rows', len(rows), 'ok', len(ok), 'cells', CELLS, 'seeds', SEEDS, 'eps', len(EPS))

print('\n1. AUTO-DUP == AUTO over EVERY key (not just the 9 analyze_rx7 checks)')
skip = {'job_id', 'cell', 'wall_s', 'elapsed_ms'}
bad = []
for e in EPS:
    for s in SEEDS:
        a, b = R[('AUTO', e, s)], R[('AUTO-DUP', e, s)]
        for k in set(a) | set(b):
            if k in skip:
                continue
            if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True):
                bad.append((e, s, k))
print(f'   mismatches over {len(EPS)*len(SEEDS)} (ep,seed) x ~29 keys: {len(bad)}'
      f'{"" if not bad else " -> " + str(bad[:5])}')

print('\n2. Phase-1 invariance across all 6 cells (atoms, not just scalars)')
bad2 = 0
for e in EPS:
    for s in SEEDS:
        ref = R[('AUTO-NOTRACK', e, s)]
        for c in CELLS:
            r = R[(c, e, s)]
            if (r['init_nrmse'] != ref['init_nrmse'] or r['N'] != ref['N'] or
                    r['auto_selected'] != ref['auto_selected'] or
                    r['ensemble_k'] != ref['ensemble_k'] or
                    r['kx'] != ref['kx'] or r['ky'] != ref['ky']):
                bad2 += 1
print(f'   violations: {bad2}/{len(EPS)*len(SEEDS)*len(CELLS)}')

print('\n3. PREREG freeze vs results mtime')
h = hashlib.md5(open(f'{HERE}/PREREG.md', "rb").read()).hexdigest()
print('   PREREG.md md5   :', h)
print('   frozen record   :', open(f'{HERE}/PREREG_frozen_md5.txt').read().split()[0])
for f in ('PREREG.md', 'PREREG_frozen_md5.txt', 'run_rx7.py', 'nulls_rx7.json',
          'independence_gate16.txt', 'results.jsonl', 'analyze_rx7.py'):
    print(f'   {f:26s} mtime {os.path.getmtime(f"{HERE}/{f}"):.0f}')

print('\n4. PREREG deliverables actually produced')
print(f'   sec.1 "existing 32 (indep2+indep3) scored as a SEPARATE companion block":')
print(f'      episodes in results.jsonl: {len(EPS)}, all from indep4 (episode-seeds 8-11)? '
      f'{all("seed"+str(i) in e for e in EPS for i in range(8,12) if "seed"+str(i) in e)}')
print(f'      episode-seed set = {sorted({e.split("seed")[1].split("_")[0] for e in EPS})}')
print('      -> the companion 32-episode lane was NOT run.  PRIMARY-A/PRIMARY-B have')
print('         no companion block at all (Rx-6 never recorded avg_RMSE_observed).')
print(f'   sec.5.6 "re-verify the partition in the actual runs": analyze_rx7.py contains'
      f' "RMSE_observed +" reconstruction? '
      f'{"partition" in open(f"{HERE}/analyze_rx7.py").read().lower()}')
print('   sec.3.7 "report MDE and MDE / incumbent slack": present in analysis.txt? '
      f'{"MDE" in open(f"{HERE}/analysis.txt").read()}')
print('   sec.3.4 "report the DIRECTION of the exclusions": present in analysis.txt? '
      f'{"배제 방향" in open(f"{HERE}/analysis.txt").read()}')
print('   sec.3.5 "table of per-cell null-pass and m>1 counts": m>1 counts in analysis.txt? '
      f'{"m>1" in open(f"{HERE}/analysis.txt").read()}')

print('\n5. m>1 (worse than a flat sea) counts, which sec.3.5 asked for and analysis.txt omits')
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean']
        for n in json.load(open(f'{HERE}/nulls_rx7.json'))}
for c in ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']:
    nm = sum(1 for e in EPS for s in SEEDS if R[(c, e, s)]['nRMSE'] >= NULL[e])
    nme = sum(1 for e in EPS
              if st.median([R[(c, e, s)]['nRMSE'] for s in SEEDS]) >= NULL[e])
    mm = st.median([R[(c, e, s)]['m_mean'] for e in EPS for s in SEEDS])
    print(f'   {c:14s} run-level nRMSE>=null {nm:3d}/{len(EPS)*len(SEEDS)}  '
          f'episode-level {nme}/{len(EPS)}   median m_mean={mm:.4f}')
