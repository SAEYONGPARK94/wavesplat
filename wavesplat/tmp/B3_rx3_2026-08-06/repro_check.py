"""B3 SECONDARY-0: does the COPIED runner reproduce the ORIGINAL bit-for-bit?

PREREG sec.5. Compares tmp/B3_rx3_2026-08-06/repro.jsonl (cell AUTO, seeds 1518-1520,
run through run_b3.py) against BOTH prior raw files that carry the same cell+seeds:
   tmp/rx2_init_2026-08-05/results.jsonl   (crowded 4)
   tmp/B2_g1p1_ext_2026-08-06/results.jsonl (all 12)
A deterministic check: no p-values, tolerance is exactly 0.
Also checks AUTO-DUP == AUTO inside the main batch (worker env-residue probe).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2',
        'Hs_error', 'absVR_framewise', 'decorr_mean', 'shrink_mean', 'm_mean',
        'avg_RMSE_unobserved', 'init_nrmse', 'init_rmse', 'N']


def load(p):
    d = {}
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if 'error' in r:
            continue
        d[(r['cell'], r['episode'], r['seed'])] = r
    return d


def compare(new, old, oldname, cell_new='AUTO', cell_old='AUTO'):
    common = [k for k in new if k[0] == cell_new
              and (cell_old, k[1], k[2]) in old]
    if not common:
        print(f"  vs {oldname}: NO OVERLAP")
        return
    worst = {}
    for k in common:
        a, b = new[k], old[(cell_old, k[1], k[2])]
        for key in KEYS:
            if key not in a or key not in b:
                continue
            va, vb = a[key], b[key]
            if va is None or vb is None:
                continue
            if isinstance(va, float) and np.isnan(va) and np.isnan(vb):
                continue
            d = abs(float(va) - float(vb))
            if d > worst.get(key, (-1, None))[0]:
                worst[key] = (d, k)
    print(f"  vs {oldname}: {len(common)} overlapping runs")
    allzero = True
    for key in KEYS:
        if key in worst:
            d, k = worst[key]
            if d != 0.0:
                allzero = False
            print(f"     {key:22s} max|diff| = {d:.3e}" + ("" if d == 0 else f"   at {k}"))
    print(f"     ==> {'BIT-IDENTICAL' if allzero else '*** MISMATCH ***'}")


def main():
    repro = load(os.path.join(HERE, 'repro.jsonl'))
    print(f"== repro.jsonl: {len(repro)} runs, seeds {sorted({k[2] for k in repro})}")
    compare(repro, load(os.path.join(REPO, 'tmp/rx2_init_2026-08-05/results.jsonl')),
            'rx2_init_2026-08-05/results.jsonl')
    compare(repro, load(os.path.join(REPO, 'tmp/B2_g1p1_ext_2026-08-06/results.jsonl')),
            'B2_g1p1_ext_2026-08-06/results.jsonl')

    main_path = os.path.join(HERE, 'results.jsonl')
    if os.path.exists(main_path):
        m = load(main_path)
        print(f"\n== AUTO-DUP vs AUTO inside the main batch (env-residue probe), "
              f"{len(m)} runs ==")
        compare(m, m, 'same-batch AUTO', cell_new='AUTO-DUP', cell_old='AUTO')


if __name__ == '__main__':
    main()
