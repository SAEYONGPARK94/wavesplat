"""E2 pre-run verification (PREREG section 7). Checks 1-4."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, '..', 'e1_grid_2026-08-05')

E2E_FIELDS = ['init_nrmse', 'nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
              'Directional_R2', 'Hs_error', 'N', 'avg_RMSE_unobserved']
SPEC_FIELDS = ['detection_rate', 'energy_detection_rate', 'k_mag_rel_error_mean',
               'direction_error_deg_mean', 'n_est', 'n_matched']
PROBE_FIELDS = ['detection_rate', 'energy_detection_rate', 'k_mag_rel_error_mean',
                'direction_error_deg_mean', 'n_matched', 'n_outside_narrow_box',
                'kmag_max_placed', 'placed_nn_median', 'n_placed_pairs_lt_002']


def load(p, key):
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        out[key(r)] = r
    return out


def cmp_sets(a, b, fields, spec=False, label=''):
    n_pair = n_val = n_bad = 0
    bad = []
    for k, ra in sorted(a.items()):
        rb = b.get(k)
        if rb is None:
            continue
        n_pair += 1
        for f in fields:
            va, vb = ra.get(f), rb.get(f)
            n_val += 1
            if repr(va) != repr(vb):
                n_bad += 1
                if len(bad) < 8:
                    bad.append(f'{k} {f}: {va!r} != {vb!r}')
        if spec:
            sa, sb = ra.get('spec', {}), rb.get('spec', {})
            for f in SPEC_FIELDS:
                n_val += 1
                if repr(sa.get(f)) != repr(sb.get(f)):
                    n_bad += 1
                    if len(bad) < 8:
                        bad.append(f'{k} spec.{f}: {sa.get(f)!r} != {sb.get(f)!r}')
    status = 'PASS' if (n_pair and n_bad == 0) else ('NO-PAIRS' if not n_pair else 'FAIL')
    print(f'  {label}: {n_pair} pairs x fields = {n_val} values, mismatches {n_bad} '
          f'-> {status}')
    for b_ in bad:
        print('      ', b_)
    return status


def main():
    ok = True
    print('CHECK 1 -- e2e cell A0 vs E1 cell A (bit-identity)')
    e1 = load(os.path.join(E1, 'results.jsonl'),
              lambda r: (r['episode'], r['seed']) if r['cell'] == 'A' else None)
    e1.pop(None, None)
    e2 = load(os.path.join(HERE, 'results.jsonl'),
              lambda r: (r['episode'], r['seed']) if r['cell'] == 'A0' else None)
    e2.pop(None, None)
    ok &= cmp_sets(e2, e1, E2E_FIELDS, spec=True, label='A0 == E1.A') in ('PASS', 'NO-PAIRS')

    print('CHECK 2 -- probe cell A0 vs E1 probe cell A (bit-identity)')
    p1 = load(os.path.join(E1, 'probe.jsonl'),
              lambda r: (r['episode'], r['n_g']) if r['cell'] == 'A' else None)
    p1.pop(None, None)
    p2 = load(os.path.join(HERE, 'probe.jsonl'),
              lambda r: (r['episode'], r['n_g']) if r['cell'] == 'A0' else None)
    p2.pop(None, None)
    ok &= cmp_sets(p2, p1, PROBE_FIELDS, label='probe A0 == E1 probe A') == 'PASS'

    print('CHECK 3 -- probe U0 == probe A0 (design prediction: MIN_DK=(c-0.5)dk trick)')
    p0 = load(os.path.join(HERE, 'probe.jsonl'),
              lambda r: (r['episode'], r['n_g']) if r['cell'] == 'U0' else None)
    p0.pop(None, None)
    ok &= cmp_sets(p0, p2, PROBE_FIELDS, label='probe U0 == probe A0') == 'PASS'

    print('CHECK 4 -- MINSEP invariant: placed_nn_min >= M in every M-arm probe run')
    from run_e2 import CELLS
    n_chk = n_bad = 0
    for line in open(os.path.join(HERE, 'probe.jsonl')):
        r = json.loads(line)
        if 'error' in r:
            continue
        m = CELLS[r['cell']]['minsep']
        if m is None:
            continue
        n_chk += 1
        v = r.get('placed_nn_min')
        if v is not None and v == v and v < m - 1e-9:
            n_bad += 1
            print(f'      VIOLATION {r["job_id"]}: nn_min={v:.5f} < M={m}')
    print(f'  {n_chk} M-arm probe runs, violations {n_bad} -> '
          f'{"PASS" if n_chk and n_bad == 0 else "FAIL"}')
    ok &= (n_chk > 0 and n_bad == 0)
    print('\nOVERALL:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    sys.exit(main())
