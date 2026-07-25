"""Assemble the raydrop table for every available ETKF config + the WS arm, with paired stats.

Reads the median-of-5 method-seed jsonl for each config and reports, per config:
  per-pair U/S medians, aggregate U/S, paired Delta (S-U), Delta%, direction count (S<U),
  paired t-test and Wilcoxon signed-rank on the 10 per-pair (S-U) deltas.
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['BF2_1comp', 'BF3_1comp', 'BF3_2comp', 'BF3_5comp', 'BF4_1comp',
         'BF4_5comp', 'BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']


def key_of(ep):
    return ep.replace('episode_', '').replace('_seed0_uniform.h5', '').replace('_seed0_slope.h5', '')


def load(path, nrmse_key='avg_nRMSE'):
    """Return {pair: {cond: [vals]}} and {pair:{cond:[rho]}}."""
    d = defaultdict(lambda: defaultdict(list))
    rho = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(path):
        return None, None
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if nrmse_key not in r:
            continue
        ep = r['episode']
        cond = 'slope' if '_slope' in ep else 'uniform'
        d[key_of(ep)][cond].append(r[nrmse_key])
        if 'avg_rho' in r:
            rho[key_of(ep)][cond].append(r['avg_rho'])
    return d, rho


def report(name, path, ws=False):
    d, rho = load(path)
    if d is None:
        print(f'\n### {name}: FILE MISSING ({path})')
        return None
    # WS arm jsonl has table4 filter + different fields; handle separately
    print(f'\n{"="*78}\n### {name}')
    print(f'{"pair":<12} {"U":>8} {"S":>8} {"S-U":>9} {"Δ%":>7} {"ρU":>7} {"ρS":>7} {"nU/nS":>7}')
    U, S, deltas = [], [], []
    complete = True
    for k in ORDER:
        if 'uniform' not in d[k] or 'slope' not in d[k]:
            print(f'{k:<12} INCOMPLETE u={len(d[k].get("uniform",[]))} s={len(d[k].get("slope",[]))}')
            complete = False
            continue
        u = float(np.median(d[k]['uniform']))
        s = float(np.median(d[k]['slope']))
        ru = float(np.median(rho[k]['uniform'])) if rho[k].get('uniform') else float('nan')
        rs = float(np.median(rho[k]['slope'])) if rho[k].get('slope') else float('nan')
        U.append(u); S.append(s); deltas.append(s - u)
        print(f'{k:<12} {u:8.4f} {s:8.4f} {s-u:+9.4f} {(s-u)/u*100:+7.1f} '
              f'{ru:7.3f} {rs:7.3f} {len(d[k]["uniform"])}/{len(d[k]["slope"])}')
    if not complete or len(U) < 10:
        print('  (incomplete — stats deferred)')
        return None
    U = np.array(U); S = np.array(S); deltas = np.array(deltas)
    uavg, savg = U.mean(), S.mean()
    print(f'{"AVG":<12} {uavg:8.4f} {savg:8.4f} {savg-uavg:+9.4f} {(savg-uavg)/uavg*100:+7.1f}')
    # paired stats on per-pair (S-U)
    t = stats.ttest_rel(S, U)
    try:
        w = stats.wilcoxon(S, U)
        wp = w.pvalue
    except ValueError:
        wp = float('nan')
    nbetter = int((S < U).sum())
    print(f'  aggregate Δ%(mean-of-means) = {(savg-uavg)/uavg*100:+.1f}%   '
          f'mean paired Δ% = {np.mean(deltas/U)*100:+.1f}%')
    print(f'  slope better than uniform on {nbetter}/10 pairs')
    print(f'  paired t(9) = {t.statistic:+.2f}, p = {t.pvalue:.4f}   '
          f'Wilcoxon p = {wp:.4f}')
    print(f'  median ρ over all 20 = {np.nanmedian([v for k in ORDER for c in ("uniform","slope") for v in rho[k].get(c,[])]):.3f}')
    return dict(name=name, U=U.tolist(), S=S.tolist(), uavg=uavg, savg=savg,
                deltapct=(savg-uavg)/uavg*100, nbetter=nbetter,
                t=t.statistic, p=t.pvalue, wilcoxon_p=wp, deltas=deltas.tolist())


def report_ws():
    """WS arm from ws_runs.jsonl (table4 tag, median over 5 estimator seeds)."""
    path = os.path.join(HERE, 'ws_runs.jsonl')
    d = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = r.get('episode', '')
        if 'seed0' not in ep or 'table4' not in r.get('tables', []):
            continue
        cond = 'slope' if '_slope' in ep else 'uniform'
        d[key_of(ep)][cond].append(r['avg_nRMSE'])
    print(f'\n{"="*78}\n### WaveSplat-P1 (median over 5 estimator seeds) [current Table 4 WS col]')
    U, S = [], []
    for k in ORDER:
        u = float(np.median(d[k]['uniform'])); s = float(np.median(d[k]['slope']))
        U.append(u); S.append(s)
        print(f'{k:<12} {u:8.4f} {s:8.4f} {s-u:+9.4f} {(s-u)/u*100:+7.1f}')
    U = np.array(U); S = np.array(S)
    print(f'{"AVG":<12} {U.mean():8.4f} {S.mean():8.4f} {(S.mean()-U.mean()):+9.4f} '
          f'{(S.mean()-U.mean())/U.mean()*100:+7.1f}')
    print(f'  slope better on {int((S<U).sum())}/10')


if __name__ == '__main__':
    report_ws()
    report('ETKF as-published (40x30 Ne50 ai.005 R.05) — CURRENT PAPER Table 4',
           os.path.join(HERE, 'etkf_raydrop_runs.jsonl'))
    report('ETKF paper-tuned (40x30 Ne100 ai.005 R.05) — the paper\'s "tuned ETKF"',
           os.path.join(HERE, 'etkf_raydrop_papertuned_runs.jsonl'))
    report('ETKF retuned hold-out (160x120 Ne50 ai0 R.20 max20) — TASK PRIMARY',
           os.path.join(HERE, 'etkf_raydrop_retuned_runs.jsonl'))
