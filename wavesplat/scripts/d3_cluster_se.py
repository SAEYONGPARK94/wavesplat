"""Cluster-robust standard errors for the headline deltas.

The prereg promised these and the code never delivered them: every summary so far
divides by sqrt(n) as if 35 synthetic episodes (10-seed sea groups, two of them
double-counted) and 60 real windows (12 per record) were independent draws.  They
are not.  A reviewer who knows this will not trust any CI in the paper.

Cluster-robust SE (CR1, the standard small-sample correction):

    Var(dbar) = c * sum_g (dbar_g - dbar)^2 * (n_g/n)^2 / (G - 1)
    c = G/(G-1) * (n-1)/(n-G)   [finite-cluster correction]

Clusters: synthetic = sea state x n_components (seeds within are the same sea
law); real = record (windows within a record share the sea, the reconstruction,
and the ship track).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

OUT = 'results/direction3/cluster_se.json'


def cluster_se(d, g):
    """CR1 cluster-robust SE of the mean of d, clustered on labels g."""
    d = np.asarray(d, float)
    g = np.asarray(g)
    n, mu = len(d), float(d.mean())
    labs = np.unique(g)
    G = len(labs)
    acc = 0.0
    for lab in labs:
        di = d[g == lab]
        acc += (di.sum() - len(di) * mu) ** 2
    c = (G / (G - 1)) * ((n - 1) / (n - G)) if G > 1 and n > G else 1.0
    var = c * acc / (n ** 2)
    return float(np.sqrt(var)), int(G)


def report(name, d, g, naive_note=''):
    d = np.asarray(d, float)
    mu = float(d.mean())
    se_iid = float(d.std(ddof=1) / np.sqrt(len(d)))
    se_cl, G = cluster_se(d, g)
    infl = se_cl / se_iid if se_iid > 0 else float('nan')
    print(f'{name}')
    print(f'   n={len(d)}  clusters={G}   mean {mu:+.4f}')
    print(f'   SE  iid {se_iid:.4f}  ->  cluster-robust {se_cl:.4f}  '
          f'(x{infl:.2f})')
    print(f'   t   iid {mu / se_iid:+.2f}   ->  cluster-robust '
          f'{mu / se_cl:+.2f}')
    print(f'   95% CI (cluster): [{mu - 1.96 * se_cl:+.4f}, '
          f'{mu + 1.96 * se_cl:+.4f}]   {naive_note}')
    print()
    return dict(n=len(d), n_clusters=G, mean=mu, se_iid=se_iid,
                se_cluster=se_cl, inflation=infl,
                t_iid=mu / se_iid, t_cluster=mu / se_cl,
                ci_lo=mu - 1.96 * se_cl, ci_hi=mu + 1.96 * se_cl)


def main():
    res = {}

    # ---- the gradient's left anchor (synthetic), at both budgets -----------
    e3 = json.load(open('results/direction3/e3_clean_rejudge.json'))
    for arm, label in (('fine', 'clean-linear stage, 2000 obs'),
                       ('fine_sparse', 'clean-linear stage, 600 obs (SPARSE)')):
        eps = e3['episodes'][arm]
        d = [e['delta'] for e in eps]
        g = [f"{e['sea']}_{e['n_comp']}" for e in eps]   # sea law = the cluster
        res[f'gradient_clean_{arm}'] = report(label, d, g)

    # ---- the real headline -------------------------------------------------
    fw = json.load(open('results/direction3/field_wide_eval.json'))
    d = [r['field_learned'] - r['field_hand'] for r in fw]
    g = [r['record'] for r in fw]                        # record = the cluster
    res['real_field_delta'] = report('real stereo, 60 windows (+0.033 headline)',
                                     d, g)

    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'-> {OUT}')
    print('\nRule for the paper: quote the cluster-robust CI.  If a claim only '
          'survives the iid CI, it does not survive.')


if __name__ == '__main__':
    main()
