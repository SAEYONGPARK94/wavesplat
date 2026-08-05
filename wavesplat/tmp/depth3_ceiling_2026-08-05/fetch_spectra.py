"""Fetch per-frequency a1(f), b1(f), E(f) from CDIP -- the thing that
cdip_support_raw.json THREW AWAY (it stored only per-record scalars).

Identical fetch/QC/stride rules as tmp/cdip_support_2026-08-03/cdip_support.py
so the record sets are comparable record-for-record.
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DODS = 'https://thredds.cdip.ucsd.edu/thredds/dodsC/'
TARGET_REC = 1000
STATIONS = ['028p1', '094p1', '142p1', '194p1', '243p1', '278p1']

sys.path.insert(0, os.path.join(HERE, '..', 'cdip_support_2026-08-03'))
from cdip_support import parse_ascii, get, last_deployment  # noqa: E402


def fetch(st):
    dep = last_deployment(st)
    base = f'{DODS}cdip/archive/{st}/{dep}'
    dds = get(base + '.dds')
    m = re.search(r'waveEnergyDensity\[waveTime = (\d+)\]\[waveFrequency = (\d+)\]', dds)
    nt, nf = int(m.group(1)), int(m.group(2))
    stride = max(1, nt // TARGET_REC)
    sl = f'[0:{stride}:{nt - 1}]'
    q = (f'waveFrequency[0:1:{nf - 1}],waveHs{sl},waveTp{sl},waveFlagPrimary{sl},'
         f'waveEnergyDensity{sl}[0:1:{nf - 1}],waveA1Value{sl}[0:1:{nf - 1}],'
         f'waveB1Value{sl}[0:1:{nf - 1}]')
    d = parse_ascii(get(base + '.ascii?' + q, timeout=600))
    return dep, d


out = {}
for st in STATIONS:
    t0 = time.time()
    try:
        dep, d = fetch(st)
        np.savez_compressed(
            os.path.join(HERE, f'spec_{st}.npz'),
            f=d['waveFrequency'], E=np.atleast_2d(d['waveEnergyDensity']),
            a1=np.atleast_2d(d['waveA1Value']), b1=np.atleast_2d(d['waveB1Value']),
            Hs=d['waveHs'].ravel(), Tp=d['waveTp'].ravel(),
            flag=d['waveFlagPrimary'].ravel())
        out[st] = {'deployment': dep, 'nt': int(np.atleast_2d(d['waveEnergyDensity']).shape[0]),
                   'nf': int(d['waveFrequency'].size), 'secs': round(time.time() - t0, 1)}
        print(f'{st}/{dep}: {out[st]["nt"]} x {out[st]["nf"]}  ({out[st]["secs"]}s)', flush=True)
    except Exception as e:
        out[st] = {'error': f'{type(e).__name__}: {e}'}
        print(f'{st}: FAIL {type(e).__name__}: {e}', flush=True)

json.dump(out, open(os.path.join(HERE, 'fetch_manifest.json'), 'w'), indent=2)
