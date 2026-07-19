"""E-B2K step 0 -- current position under the FROZEN statistic (no new compute).

Frozen spec (fable prereg, tmp/E60_B2_headline_fable_review_2026-07-19.md sec 6):
  unit      = window, paired Delta corr (T - L)
  seeds     = averaged WITHIN record (repeated measure, not a cluster)
  cluster   = record
  failure   = hand < 0.5 (07-12 frozen definition)
  test      = record-mean, t(k-1), one-sided; Webb 6-point null-imposed wild
              cluster bootstrap-t, one-sided, full enumeration when 6^k feasible
This script only RE-READS existing results. It establishes where we stand at k=5
before any expansion, so the k=9 run has a frozen reference point.
"""
import json, itertools
import numpy as np
from collections import defaultdict
from scipy import stats

R3 = 'results/direction3/'
W6 = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def load_arms():
    """Return rows: (record, wi, hand, {seed: (T, L)})."""
    d1 = json.load(open(R3 + 'b2deep_eval.json'))['rows']
    d2 = json.load(open(R3 + 'b2deep2_eval.json'))['seed3']['rows']
    s2 = {(r['record'], r['wi']): (r['T_s2'], r['L_s2']) for r in d2}
    out = []
    for r in d1:
        key = (r['record'], r['wi'])
        arms = {0: (r['T_s0'], r['L_s0']), 1: (r['T_s1'], r['L_s1'])}
        if key in s2:
            arms[2] = s2[key]
        out.append(dict(record=r['record'], wi=r['wi'], hand=r['hand'], arms=arms))
    return out


def wild_p_onesided(m, t_obs):
    """Webb 6-point null-imposed wild cluster bootstrap-t, one-sided (H1: mean>0)."""
    k = len(m)
    if 6 ** k > 4_000_000:
        rng = np.random.default_rng(0)
        draws = rng.choice(W6, size=(200_000, k))
    else:
        draws = np.array(list(itertools.product(W6, repeat=k)))
    yb = m[None, :] * draws
    mb = yb.mean(1)
    sb = yb.std(1, ddof=1) / np.sqrt(k)
    tb = mb / sb
    return float((tb >= t_obs).mean())


def cluster_test(by_rec, label):
    recs = sorted(by_rec)
    m = np.array([np.mean(by_rec[r]) for r in recs])
    k = len(recs)
    mean = m.mean()
    se = m.std(ddof=1) / np.sqrt(k)
    t = mean / se
    p1 = stats.t.sf(t, k - 1)
    wp = wild_p_onesided(m, t)
    pos = int((m > 0).sum())
    print(f'{label:34s} mean {mean:+.4f}  t({k-1}) {t:+.2f}  p1 {p1:.4f}  '
          f'wild_p1 {wp:.4f}  signs {pos}/{k}')
    for r, v in zip(recs, m):
        print(f'{"":36s}{r:8s} {v:+.4f}')
    return dict(records=recs, per_record=m.tolist(), mean=float(mean), k=k,
                t=float(t), p_onesided=float(p1), wild_p_onesided=wp, n_pos=pos)


def main():
    rows = load_arms()
    nseed = {len(r['arms']) for r in rows}
    print(f'rows {len(rows)}  seeds/row {sorted(nseed)}  '
          f'records {sorted({r["record"] for r in rows})}')

    out = {}
    for tag, keep in (('all windows', lambda h: True),
                      ('failure windows (hand<0.5)', lambda h: h < 0.5)):
        by_rec = defaultdict(list)
        n = 0
        for r in rows:
            if not keep(r['hand']):
                continue
            n += 1
            # seeds averaged WITHIN window, then window is the paired unit
            d = np.mean([t - l for t, l in r['arms'].values()])
            by_rec[r['record']].append(d)
        print(f'\n--- T - L, {tag}  (n windows {n}) ---')
        out[tag] = cluster_test(by_rec, 'T - L')
        out[tag]['n_windows'] = n

    # per-seed direction check (the 07-12 gate's own criterion)
    print('\n--- per-seed failure-window record-mean (direction agreement) ---')
    for s in (0, 1, 2):
        by_rec = defaultdict(list)
        for r in rows:
            if r['hand'] < 0.5 and s in r['arms']:
                t, l = r['arms'][s]
                by_rec[r['record']].append(t - l)
        if by_rec:
            m = np.array([np.mean(by_rec[r]) for r in sorted(by_rec)])
            print(f'  seed {s}: mean {m.mean():+.4f}  signs {(m>0).sum()}/{len(m)}  '
                  f'per-rec {np.round(m,3)}')

    json.dump(out, open('tmp/e60k_step0_frozen_stat.json', 'w'), indent=1)
    print('\nwrote tmp/e60k_step0_frozen_stat.json')


if __name__ == '__main__':
    main()
