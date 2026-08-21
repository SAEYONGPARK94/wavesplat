"""Which D formula reproduces the scorer's ratios 0.963/0.999/1.437/2.259/3.085?
PREREG sec.1.1 literal:  D_w = rms(dw)   D_phi = rms(w_NOTRACK (x) wrap(dphi))
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ['AUTO', 'AUTO-NOTRACK'] + [f'{c}-S{i}' for c in 'WP' for i in range(1, 6)]
SLEV = ['S1', 'S2', 'S3', 'S4', 'S5']
MEDW = 0.04409531503915787
TARGET = [0.963, 0.999, 1.437, 2.259, 3.085]

raw = defaultdict(dict)
for line in open(os.path.join(HERE, 'results.jsonl')):
    r = json.loads(line)
    raw[r['cell']][(r['episode'], r['seed'])] = r
eps = sorted({k[0] for k in raw['AUTO']})
seeds = sorted({k[1] for k in raw['AUTO']})


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def arrs(c, e, s):
    b, u = raw['AUTO-NOTRACK'][(e, s)], raw[c][(e, s)]
    w0 = np.asarray(b['w'], float)
    p0 = np.asarray(b['phi'], float)
    return w0, p0, np.asarray(u['w'], float), np.asarray(u['phi'], float)


VAR = {}
VAR['prereg_literal'] = (
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w1 - w0) ** 2))),
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w0 * wrap(p1 - p0)) ** 2))))
VAR['phi_unwrapped'] = (
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w1 - w0) ** 2))),
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w0 * (p1 - p0)) ** 2))))
VAR['phi_uses_medianw'] = (
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w1 - w0) ** 2))),
    lambda w0, p0, w1, p1: float(MEDW * np.sqrt(np.mean(wrap(p1 - p0) ** 2))))
VAR['mean_abs'] = (
    lambda w0, p0, w1, p1: float(np.mean(np.abs(w1 - w0))),
    lambda w0, p0, w1, p1: float(np.mean(np.abs(w0 * wrap(p1 - p0)))))
VAR['phi_raw_rad_no_gain'] = (
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w1 - w0) ** 2))),
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean(wrap(p1 - p0) ** 2))))
VAR['phi_over_2pi'] = (
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w1 - w0) ** 2))),
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w0 * wrap(p1 - p0)) ** 2)) / (2 * np.pi)))
VAR['w_relative'] = (
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean(((w1 - w0) / np.maximum(np.abs(w0), 1e-9)) ** 2))),
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean(wrap(p1 - p0) ** 2))))
VAR['w_rel_medianw'] = (
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean((w1 - w0) ** 2)) / MEDW),
    lambda w0, p0, w1, p1: float(np.sqrt(np.mean(wrap(p1 - p0) ** 2))))

print(f'{"variant":22s} ' + ' '.join(f'{s:>8s}' for s in SLEV) + '   err vs scorer')
for name, (fw, fp) in VAR.items():
    ratios = []
    for S in SLEV:
        dw = np.median([np.median([fw(*arrs('W-' + S, e, s)) for s in seeds]) for e in eps])
        dp = np.median([np.median([fp(*arrs('P-' + S, e, s)) for s in seeds]) for e in eps])
        ratios.append(dw / dp)
    err = float(np.mean([abs(a - b) / b for a, b in zip(ratios, TARGET)]))
    print(f'{name:22s} ' + ' '.join(f'{r:8.4f}' for r in ratios) + f'   {err * 100:6.1f}%')
print(f'{"SCORER REPORTED":22s} ' + ' '.join(f'{t:8.4f}' for t in TARGET))

# what would D_phi have to be to give the scorer's ratio?
print('\nimplied D_phi (given prereg D_w) that would produce the scorer ratios:')
fw = VAR['prereg_literal'][0]
fp = VAR['prereg_literal'][1]
for S, t in zip(SLEV, TARGET):
    dw = np.median([np.median([fw(*arrs('W-' + S, e, s)) for s in seeds]) for e in eps])
    dp = np.median([np.median([fp(*arrs('P-' + S, e, s)) for s in seeds]) for e in eps])
    print(f'  {S}: D_w {dw:.6f}  prereg D_phi {dp:.6f}  implied D_phi {dw / t:.6f} '
          f'(factor {dp / (dw / t):.3f} vs prereg)')

# sanity: does phase actually saturate?  report rms|dphi| in radians
print('\nraw phase drift rms|wrap(dphi)| [rad] and rms|dw| [m], block medians:')
for S in SLEV:
    rp = np.median([np.median([float(np.sqrt(np.mean(wrap(arrs('P-' + S, e, s)[3] - arrs('P-' + S, e, s)[1]) ** 2)))
                               for s in seeds]) for e in eps])
    ru = np.median([np.median([float(np.sqrt(np.mean(np.abs(arrs('P-' + S, e, s)[3] - arrs('P-' + S, e, s)[1]) ** 2)))
                               for s in seeds]) for e in eps])
    rw = np.median([np.median([float(np.sqrt(np.mean((arrs('W-' + S, e, s)[2] - arrs('W-' + S, e, s)[0]) ** 2)))
                               for s in seeds]) for e in eps])
    print(f'  {S}: P rms|dphi|wrapped {rp:7.4f} rad  unwrapped {ru:8.4f} rad  |  W rms|dw| {rw:.6f} m')
