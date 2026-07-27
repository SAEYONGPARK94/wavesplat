"""Score the confirmation-2 run against PREREG_CONFIRM2 (frozen bars P0-P5).

Inputs:
  atoms_confirm2_v1.jsonl / atoms_confirm2_v3.jsonl   (n_selected per ep x diff_seed)
  arm_runs_confirm2_v1.jsonl / _v3.jsonl              (learned lanes)
  arm_runs_v2m3.jsonl                                 (m3 lanes + member diagnostics)
v2 single-seed baseline = member[0] of each m3 lane (PREREG Sec 2, no separate runs).

  python3 tmp/confirm2_2026-07-27/score_confirm2.py
"""
import json
import os
import re
import sys
from statistics import median

BASE = 'tmp/confirm2_2026-07-27'
DATA = 'data_confirm2_2026-07-27'
TH = 8
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def load_rows(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        key = (r['episode'], r['est_seed'], r['diff_seed'])
        if key in rows:
            sys.exit(f'DUPLICATE ROW {key} in {path} -- abort (PREREG Sec 2 err rule)')
        rows[key] = r
    return rows


def val(rows, key):
    r = rows[key]
    if r.get('err') is not None:
        return float('inf')          # err lane is final and counts against P1c
    return r['avg_nRMSE']


def load_nsel(path):
    ns = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('err') is None:
            ns[(r['episode'], r['diff_seed'])] = r['n_selected']
    return ns


def ncomp_of(ep):
    return int(re.search(r'_(\d+)comp_', ep).group(1))


def slice_of(ep):
    n = ncomp_of(ep)
    return 'sparse' if n <= 2 else ('moderate' if n == 5 else 'crowded')


def mean(xs):
    return sum(xs) / len(xs)


def main():
    m3_rows = load_rows(f'{BASE}/arm_runs_v2m3.jsonl')
    l_rows = {'v1': load_rows(f'{BASE}/arm_runs_confirm2_v1.jsonl'),
              'v3': load_rows(f'{BASE}/arm_runs_confirm2_v3.jsonl')}
    nsel = {'v1': load_nsel(f'{BASE}/atoms_confirm2_v1.jsonl'),
            'v3': load_nsel(f'{BASE}/atoms_confirm2_v3.jsonl')}
    episodes = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
    assert len(episodes) == 40, len(episodes)

    sl = {s: [e for e in episodes if slice_of(e) == s]
          for s in ('sparse', 'moderate', 'crowded')}
    two = [e for e in episodes if ncomp_of(e) == 2]
    slices = {'all': episodes, 'sparse': sl['sparse'], 'moderate': sl['moderate'],
              'crowded': sl['crowded'], '2comp': two}

    # arm lane values
    m3 = {k: val(m3_rows, k) for k in m3_rows}
    v2s = {k: (m3_rows[k]['member_nRMSE'][0] if m3_rows[k].get('err') is None
               else float('inf')) for k in m3_rows}     # single-seed = member[0]

    def ep_med(lanevals, ep):
        vs = [lanevals[(ep, es, ds)] for es, ds in LANES]
        assert len(vs) == 5
        return median(vs)

    m3_ep = {e: ep_med(m3, e) for e in episodes}
    v2_ep = {e: ep_med(v2s, e) for e in episodes}
    l_ep = {ck: {e: ep_med({k: val(l_rows[ck], k) for k in l_rows[ck]}, e)
                 for e in episodes} for ck in ('v1', 'v3')}

    out = {'prereg': 'PREREG_CONFIRM2.md', 'th': TH, 'candidates': {}, 'reports': {}}

    # ---- P0 spec harmlessness (v2_m3 vs v2 single, all slices) ----
    p0_detail = {s: (mean([m3_ep[e] for e in eps_]), mean([v2_ep[e] for e in eps_]))
                 for s, eps_ in slices.items()}
    p0 = all(m <= v + 0.005 for m, v in p0_detail.values())
    out['P0'] = {'pass': p0, 'slice_means_m3_vs_v2': p0_detail}

    # ---- candidates ----
    for ck in ('v1', 'v3'):
        lv = {k: val(l_rows[ck], k) for k in l_rows[ck]}
        cand_lane, routed_flag = {}, {}
        for e in episodes:
            for es, ds in LANES:
                r = nsel[ck][(e, ds)] >= TH
                cand_lane[(e, es, ds)] = lv[(e, es, ds)] if r else m3[(e, es, ds)]
                routed_flag[(e, es, ds)] = r
        c_ep = {e: ep_med(cand_lane, e) for e in episodes}

        sparse_lanes = [(k, cand_lane[k]) for k in cand_lane if slice_of(k[0]) == 'sparse']
        viol = [(k, v) for k, v in sparse_lanes if v > 0.10]
        p1c = len(viol) == 0
        p1a = all(c_ep[e] <= 0.10 for e in sl['sparse'])
        learned_sparse = [v for k, v in sparse_lanes if routed_flag[k]]
        p1b = all(v <= 0.10 for v in learned_sparse) if learned_sparse else True
        attribution = [{'lane': list(k), 'value': v,
                        'source': 'learned' if routed_flag[k] else 'v2_m3_fallback'}
                       for k, v in viol]

        diffs_v2 = [c_ep[e] - v2_ep[e] for e in episodes]
        p2 = (mean([c_ep[e] for e in episodes]) <= mean([v2_ep[e] for e in episodes]) + 0.005
              and median(diffs_v2) <= 0.002)
        cr_impr_m3 = sum(1 for e in sl['crowded'] if c_ep[e] < m3_ep[e])
        p3 = (mean([c_ep[e] for e in sl['crowded']])
              <= mean([m3_ep[e] for e in sl['crowded']]) - 0.03) and cr_impr_m3 >= 12
        p4 = mean([c_ep[e] for e in sl['moderate']]) <= mean([v2_ep[e] for e in sl['moderate']]) + 0.005
        p5 = mean([c_ep[e] for e in two]) <= mean([v2_ep[e] for e in two]) + 0.005

        out['candidates'][ck] = dict(
            episode_medians=c_ep,
            means={s: mean([c_ep[e] for e in eps_]) for s, eps_ in slices.items()},
            bars={'P1c': p1c, 'P1a(diag)': p1a, 'P1b(diag)': p1b,
                  'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5},
            p1c_violations=attribution,
            paired_median_vs_v2=median(diffs_v2),
            crowded_improved_vs_m3=f'{cr_impr_m3}/16',
            crowded_margin_vs_m3=mean([c_ep[e] for e in sl['crowded']])
            - mean([m3_ep[e] for e in sl['crowded']]),
            crowded_margin_vs_v2=mean([c_ep[e] for e in sl['crowded']])
            - mean([v2_ep[e] for e in sl['crowded']]),
            routed_census={s: sum(1 for e in eps_
                                  if sum(routed_flag[(e, es, ds)] for es, ds in LANES) >= 3)
                           for s, eps_ in (('sparse', sl['sparse']),
                                           ('moderate', sl['moderate']),
                                           ('crowded', sl['crowded']))},
            all_pass=p0 and p1c and p2 and p3 and p4 and p5,
        )

    # ---- frozen decision rule ----
    a1, a3 = out['candidates']['v1'], out['candidates']['v3']
    if a1['all_pass'] and a3['all_pass']:
        wins = sum(1 for e in sl['crowded']
                   if a1['episode_medians'][e] < a3['episode_medians'][e])  # ties not counted
        out['crowded_v1_vs_v3'] = f'{wins}/16'
        out['decision'] = 'auto_v1_m3' if wins >= 11 else 'auto_v3_m3'
    elif a1['all_pass'] != a3['all_pass']:
        out['decision'] = 'auto_v1_m3' if a1['all_pass'] else 'auto_v3_m3'
        if a1['bars']['P1c'] != a3['bars']['P1c']:
            out['p1c_divergence_note'] = ('P1c divergence is routing luck, not accuracy '
                                          'evidence (PREREG Sec 4-4) -- see p1c_violations')
    else:
        out['decision'] = 'NO_DEPLOY'
    if not p0:
        out['decision'] = 'NO_DEPLOY_SPEC'

    # ---- reports ----
    rep = out['reports']
    rep['R1_pure_means'] = {arm: {s: mean([d[e] for e in eps_]) for s, eps_ in slices.items()}
                            for arm, d in (('learned_v1', l_ep['v1']),
                                           ('learned_v3', l_ep['v3']),
                                           ('v2_m3', m3_ep), ('v2', v2_ep))}
    for base_name, base in (('v2', v2_ep), ('v2_m3', m3_ep)):
        d = [l_ep['v1'][e] - base[e] for e in episodes]
        rep[f'R2_pure_v1_vs_{base_name}'] = {'median_delta': median(d),
                                             'wins': sum(1 for x in d if x < 0)}
    sp_lanes = sorted(v for k, v in m3.items() if slice_of(k[0]) == 'sparse')
    rep['R3_m3_sparse_lanes'] = {'gt_0.10': sum(1 for v in sp_lanes if v > 0.10),
                                 'q90': sp_lanes[int(0.90 * len(sp_lanes))],
                                 'q95': sp_lanes[int(0.95 * len(sp_lanes))],
                                 'max': sp_lanes[-1]}
    disag = repair = better_all = 0
    for k, r in m3_rows.items():
        if r.get('err') is not None:
            continue
        mem = r['member_nRMSE']
        if max(mem) - min(mem) > 0.05:
            disag += 1
        if r['avg_nRMSE'] < min(mem):
            better_all += 1
        if min(mem) < 0.10 < max(mem) and r['avg_nRMSE'] <= 0.10:
            repair += 1
    rep['R8_member_disagreement_lanes'] = disag
    rep['R8_repairs_triggered'] = repair
    rep['R9_m3_better_than_all_members'] = better_all
    rep['R9_delta_by_slice'] = {s: mean([m3_ep[e] for e in eps_]) - mean([v2_ep[e] for e in eps_])
                                for s, eps_ in slices.items()}
    # R7 best-of-3 by obs_loss (GT-free) alternative scoring
    bo3 = {}
    for k, r in m3_rows.items():
        if r.get('err') is not None:
            bo3[k] = float('inf')
            continue
        ols = r.get('member_obs_loss', [])
        if ols and all(o == o for o in ols):
            bo3[k] = r['member_nRMSE'][ols.index(min(ols))]
        else:
            bo3[k] = r['avg_nRMSE']
    bo3_ep = {e: ep_med(bo3, e) for e in episodes}
    rep['R7_bestof3_obsloss'] = {
        'sparse_mean_vs_m3': (mean([bo3_ep[e] for e in sl['sparse']]),
                              mean([m3_ep[e] for e in sl['sparse']])),
        'sparse_lanes_gt_0.10': sum(1 for k, v in bo3.items()
                                    if slice_of(k[0]) == 'sparse' and v > 0.10),
        'all_mean_vs_m3': (mean(list(bo3_ep.values())), mean(list(m3_ep.values())))}

    json.dump(out, open(f'{BASE}/confirm2_verdict.json', 'w'), indent=1)

    print('=== PREREG_CONFIRM2 verdict ===')
    print(f"P0 spec harmlessness: {'PASS' if p0 else 'FAIL'} "
          + str({s: f'{m:.4f}/{v:.4f}' for s, (m, v) in p0_detail.items()}))
    for ck in ('v1', 'v3'):
        c = out['candidates'][ck]
        print(f"auto_{ck}_m3: {c['bars']}  all_pass={c['all_pass']}")
        print(f"  means {[f'{s}={v:.4f}' for s, v in c['means'].items()]}")
        print(f"  crowded margin vs m3 {c['crowded_margin_vs_m3']:+.4f} ({c['crowded_improved_vs_m3']})"
              f" | vs v2 {c['crowded_margin_vs_v2']:+.4f} | pairedΔ vs v2 {c['paired_median_vs_v2']:+.5f}")
        if c['p1c_violations']:
            print(f"  P1c violations: {c['p1c_violations']}")
        print(f"  routed {c['routed_census']}")
    if 'crowded_v1_vs_v3' in out:
        print('crowded v1 vs v3:', out['crowded_v1_vs_v3'])
    print('DECISION:', out['decision'])
    print(json.dumps(rep, indent=1, default=str))


if __name__ == '__main__':
    main()
