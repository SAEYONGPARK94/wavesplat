"""Read out the flag sweep under the rule pre-registered in flagjobs.py's docstring.

Rule (fixed before the numbers): an arm counts as an improvement only if the paired
per-episode median moves in the same direction on >= 4 of the 6 episodes AND the mean move
exceeds the mean per-episode seed sd of the default arm. Otherwise: "not resolvable at n=6".
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'flag_runs.jsonl')


def main():
    by = defaultdict(list)
    with open(RUNS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r:
                continue
            by[(r['arm'], r['episode'])].append(r['avg_nRMSE'])

    arms = sorted({a for a, _ in by})
    eps = sorted({e for _, e in by})
    if 'default' not in arms:
        print('no default arm yet')
        return

    med = {k: float(np.median(v)) for k, v in by.items()}
    sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in by.items()}
    noise = np.mean([sd[('default', e)] for e in eps if ('default', e) in sd])

    print(f'default arm mean per-episode seed sd = {noise:.4f}  '
          f'(the bar an arm must clear)\n')
    hdr = f'{"episode":<40}' + ''.join(f'{a:>22}' for a in arms)
    print(hdr)
    print('-' * len(hdr))
    for e in eps:
        row = f'{e.replace("episode_", "").replace(".h5", ""):<40}'
        for a in arms:
            k = (a, e)
            row += f'{med[k]:>15.4f}+-{sd[k]:.3f}' if k in med else f'{"---":>22}'
        print(row)

    print()
    for a in arms:
        if a == 'default':
            continue
        d = [med[(a, e)] - med[('default', e)] for e in eps
             if (a, e) in med and ('default', e) in med]
        if not d:
            continue
        d = np.array(d)
        better = int((d < 0).sum())
        mean_move = float(d.mean())
        passes = (better >= 4 or better <= len(d) - 4) and abs(mean_move) > noise
        direction = 'IMPROVES' if mean_move < 0 else 'DEGRADES'
        verdict = f'{direction} (pre-registered rule met)' if passes \
            else 'not resolvable at this n'
        print(f'{a:<22} mean delta {mean_move:+.4f}  better on {better}/{len(d)}  '
              f'-> {verdict}')


if __name__ == '__main__':
    main()
