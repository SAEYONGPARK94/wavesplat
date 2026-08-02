"""Triage the FRF projects/ catalog crawl for datasets that could matter to us.

Reads projects_tree.txt (written by crawl_projects2.py) and buckets nodes by
keyword. No claims -- this is an index, not a measurement.
"""
import re
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else 'projects_tree.txt'

# what would actually change our situation, in priority order
BUCKETS = [
    ('A_3D_SURFACE_LIDAR', r'lidarwavescan|waveshape|velodyne|pointcloud|point_cloud|3dscan|scan'),
    ('B_LIDAR_OTHER',      r'lidar|riegl|laser'),
    ('C_STEREO_VIDEO',     r'stereo|argus|video|camera|imagery|photogram'),
    ('D_RADAR',            r'radar|xband|x-band|marine_radar'),
    ('E_ARRAY_DIRECTIONAL', r'array|awac|adcp|adop|directional|dwg|buoy'),
    ('F_MOVING_PLATFORM',  r'crab|lark|vessel|ship|boat|uav|uas|drone|aerial|airborne'),
]

lines = open(PATH, encoding='utf-8').read().splitlines()

nodes = []          # (indent, name, kind, extra)
for ln in lines:
    m = re.match(r'^(\s*)(\S.*?)\s{2,}\[(DIR|FILES|YEARS)\]\s*(.*)$', ln)
    if m:
        nodes.append((len(m.group(1)), m.group(2).strip(), m.group(3), m.group(4).strip()))

hits = defaultdict(list)
for indent, name, kind, extra in nodes:
    low = name.lower()
    for bucket, pat in BUCKETS:
        if re.search(pat, low):
            hits[bucket].append((indent, name, kind, extra))
            break

print(f'lines={len(lines)}  parsed_nodes={len(nodes)}')
dirs = [n for n in nodes if n[2] == 'DIR']
print(f'DIR nodes={len(dirs)}  FILES nodes={sum(1 for n in nodes if n[2]=="FILES")}  '
      f'YEARS nodes={sum(1 for n in nodes if n[2]=="YEARS")}')

# top-level projects (indent 2 under root)
tops = sorted({n[1] for n in nodes if n[0] <= 2 and n[2] == 'DIR'})
print(f'\ntop-level projects ({len(tops)}):')
for t in tops:
    print(f'  - {t}')

print('\n=== keyword buckets ===')
for bucket, _ in BUCKETS:
    rows = hits.get(bucket, [])
    uniq = sorted({(r[1], r[2]) for r in rows})
    print(f'\n[{bucket}] {len(rows)} node-hits / {len(uniq)} unique names')
    for name, kind in uniq[:60]:
        ex = next((r[3] for r in rows if r[1] == name and r[3]), '')
        print(f'    {name:<34} [{kind}] {ex[:70]}')

fails = [l for l in lines if l.startswith('!! FETCH FAILED')]
print(f'\nfetch failures: {len(fails)}')
for f in fails[:20]:
    print('   ', f[:140])
