"""PREREG sec.6.1 -- bit-identity of the Rx-4 runner copy against B3.
Compares repro.jsonl (AUTO x crowded4 x seeds 1521-1523) field-by-field with the same 12
runs in tmp/B3_rx3_2026-08-06/results.jsonl.  Any numeric mismatch invalidates Rx-4."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

new = {}
for l in open(os.path.join(HERE, 'repro.jsonl')):
    r = json.loads(l)
    if 'error' not in r:
        new[r['job_id']] = r
old = {}
for l in open(os.path.join(REPO, 'tmp/B3_rx3_2026-08-06/results.jsonl')):
    r = json.loads(l)
    if 'error' not in r:
        old[r['job_id']] = r

SKIP = {'wall_s', 'elapsed_ms', 'cfg'}   # wall clock + config echo (paths/seeds), not results
print(f'repro runs = {len(new)}')
missing = [k for k in new if k not in old]
print(f'missing from B3 = {missing}')

nbad = 0
ncmp = 0
worst = ('', 0.0)


def walk(a, b, path, jid):
    global nbad, ncmp, worst
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            if k not in a or k not in b:
                print(f'  KEY-DIFF {jid} {path}.{k}')
                nbad += 1
                continue
            walk(a[k], b[k], f'{path}.{k}', jid)
    elif isinstance(a, list):
        if len(a) != len(b):
            print(f'  LEN-DIFF {jid} {path}: {len(a)} vs {len(b)}')
            nbad += 1
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{path}[{i}]', jid)
    elif isinstance(a, float) or isinstance(b, float):
        ncmp += 1
        if a != b and not (a != a and b != b):     # NaN == NaN treated as equal
            d = abs(a - b)
            if d > worst[1]:
                worst = (f'{jid} {path}', d)
            nbad += 1
    else:
        ncmp += 1
        if a != b:
            print(f'  DIFF {jid} {path}: {a!r} vs {b!r}')
            nbad += 1


for jid, r in sorted(new.items()):
    if jid in old:
        walk(r, old[jid], '', jid)

print(f'scalar comparisons = {ncmp}')
print(f'mismatches         = {nbad}')
if worst[1]:
    print(f'worst float diff   = {worst[1]:.3e} at {worst[0]}')
print('\nBIT-IDENTITY: ' + ('PASS' if nbad == 0 and not missing else 'FAIL'))
