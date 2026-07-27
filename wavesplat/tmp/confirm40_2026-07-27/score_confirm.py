"""Score the confirmation-40 run against PREREG_CONFIRM.md Sec 3-5 (frozen bars).

Inputs (produced by precompute_confirm.py / run_arms_confirm.py):
  atoms_confirm_v1.jsonl / atoms_confirm_v3.jsonl        (n_selected per episode x diff_seed)
  arm_runs_confirm_v1.jsonl / _v3.jsonl / _v2.jsonl      (5 lanes per episode)

Candidates: auto_v1, auto_v3 = per-lane router (n_selected >= 8 -> learned lane, else v2
lane; no guard). Episode value = median of 5 assembled lanes.

  python3 tmp/confirm40_2026-07-27/score_confirm.py
"""
import json
import os
import re
from statistics import median

BASE = 'tmp/confirm40_2026-07-27'
TH = 8                       # auto_switch_thresh, shipped value (PREREG Sec 2)
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def load_arm(path):
    lanes = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('err') is not None:
            continue
        lanes[(r['episode'], r['est_seed'], r['diff_seed'])] = r['avg_nRMSE']
    return lanes


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


def main():
    v2 = load_arm(f'{BASE}/arm_runs_confirm_v2.jsonl')
    learned = {'v1': load_arm(f'{BASE}/arm_runs_confirm_v1.jsonl'),
               'v3': load_arm(f'{BASE}/arm_runs_confirm_v3.jsonl')}
    nsel = {'v1': load_nsel(f'{BASE}/atoms_confirm_v1.jsonl'),
            'v3': load_nsel(f'{BASE}/atoms_confirm_v3.jsonl')}

    episodes = sorted(f for f in os.listdir('data_confirm40_2026-07-27') if f.endswith('.h5'))
    assert len(episodes) == 40, len(episodes)

    # per-episode medians for every arm
    def ep_median(lanes, ep):
        vals = [lanes[(ep, es, ds)] for (es, ds) in LANES if (ep, es, ds) in lanes]
        assert len(vals) == 5, (ep, len(vals))
        return median(vals)

    def assemble(ck, ep):
        vals = []
        for (es, ds) in LANES:
            routed_learned = nsel[ck][(ep, ds)] >= TH
            src = learned[ck] if routed_learned else v2
            vals.append((src[(ep, es, ds)], routed_learned))
        return vals

    out = {'prereg': 'PREREG_CONFIRM.md', 'th': TH, 'candidates': {}, 'report_only': {}}
    v2_ep = {ep: ep_median(v2, ep) for ep in episodes}

    def mean(xs):
        return sum(xs) / len(xs)

    for ck in ('v1', 'v3'):
        cand = {}
        auto_ep, auto_lane_sparse_max, routed_census = {}, 0.0, {'sparse': 0, 'moderate': 0, 'crowded': 0}
        for ep in episodes:
            lanes = assemble(ck, ep)
            auto_ep[ep] = median([v for v, _ in lanes])
            if slice_of(ep) == 'sparse':
                auto_lane_sparse_max = max(auto_lane_sparse_max, max(v for v, _ in lanes))
            if sum(r for _, r in lanes) >= 3:           # majority of lanes routed learned
                routed_census[slice_of(ep)] += 1
        sl = {s: [ep for ep in episodes if slice_of(ep) == s]
              for s in ('sparse', 'moderate', 'crowded')}
        two = [ep for ep in episodes if ncomp_of(ep) == 2]

        # P1 safety
        p1_bad = [ep for ep in sl['sparse'] if auto_ep[ep] > 0.10]
        p1 = (len(p1_bad) == 0) and (auto_lane_sparse_max <= 0.10)
        # P2 non-inferiority
        diffs = [auto_ep[ep] - v2_ep[ep] for ep in episodes]
        p2 = (mean([auto_ep[e] for e in episodes]) <= mean([v2_ep[e] for e in episodes]) + 0.005) \
            and (median(diffs) <= 0.002)
        # P3 crowded value
        cr_impr = sum(1 for ep in sl['crowded'] if auto_ep[ep] < v2_ep[ep])
        p3 = (mean([auto_ep[e] for e in sl['crowded']])
              <= mean([v2_ep[e] for e in sl['crowded']]) - 0.03) and cr_impr >= 12
        # P4 moderate harmless
        p4 = mean([auto_ep[e] for e in sl['moderate']]) <= mean([v2_ep[e] for e in sl['moderate']]) + 0.005
        # P5 2comp harmless
        p5 = mean([auto_ep[e] for e in two]) <= mean([v2_ep[e] for e in two]) + 0.005

        cand.update(
            episode_medians=auto_ep,
            means={'all': mean([auto_ep[e] for e in episodes]),
                   **{s: mean([auto_ep[e] for e in sl[s]]) for s in sl},
                   '2comp': mean([auto_ep[e] for e in two])},
            v2_means={'all': mean([v2_ep[e] for e in episodes]),
                      **{s: mean([v2_ep[e] for e in sl[s]]) for s in sl},
                      '2comp': mean([v2_ep[e] for e in two])},
            paired_median_delta=median(diffs),
            crowded_improved=f'{cr_impr}/16',
            sparse_lane_max=auto_lane_sparse_max,
            p1_violations=p1_bad,
            routed_census=routed_census,
            bars={'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5},
            all_pass=all([p1, p2, p3, p4, p5]),
        )
        out['candidates'][ck] = cand

    # frozen decision rule (PREREG Sec 4)
    a1, a3 = out['candidates']['v1'], out['candidates']['v3']
    if a1['all_pass'] and a3['all_pass']:
        cr = [ep for ep in episodes if slice_of(ep) == 'crowded']
        v1_wins = sum(1 for ep in cr
                      if a1['episode_medians'][ep] < a3['episode_medians'][ep])
        out['crowded_v1_vs_v3'] = f'{v1_wins}/16'
        out['decision'] = 'auto_v1' if v1_wins >= 11 else 'auto_v3'
    elif a1['all_pass']:
        out['decision'] = 'auto_v1'
    elif a3['all_pass']:
        out['decision'] = 'auto_v3'
    else:
        out['decision'] = 'NO_DEPLOY'

    # report-only
    l_ep = {ck: {ep: ep_median(learned[ck], ep) for ep in episodes} for ck in ('v1', 'v3')}
    ro = out['report_only']
    ro['pure_means'] = {
        arm: {'all': mean(list(d.values())),
              **{s: mean([d[e] for e in episodes if slice_of(e) == s])
                 for s in ('sparse', 'moderate', 'crowded')},
              '2comp': mean([d[e] for e in episodes if ncomp_of(e) == 2])}
        for arm, d in (('v1', l_ep['v1']), ('v3', l_ep['v3']), ('v2', v2_ep))}
    d12 = [l_ep['v1'][e] - v2_ep[e] for e in episodes]
    ro['R2_pure_v1_vs_v2'] = {'median_delta': median(d12),
                              'wins': sum(1 for x in d12 if x < 0)}
    for ck in ('v1', 'v3'):
        spread = sum(1 for ep in episodes
                     if max(learned[ck][(ep, es, ds)] for es, ds in LANES)
                     - min(learned[ck][(ep, es, ds)] for es, ds in LANES) > 0.02)
        ro[f'R3_lane_spread_gt02_{ck}'] = f'{spread}/40'
        ro[f'R4_nsel_{ck}'] = {ep: [nsel[ck][(ep, ds)] for _, ds in LANES]
                               for ep in episodes if ncomp_of(ep) == 1}

    json.dump(out, open(f'{BASE}/confirm_verdict.json', 'w'), indent=1)

    print('=== PREREG_CONFIRM verdict ===')
    for ck in ('v1', 'v3'):
        c = out['candidates'][ck]
        print(f"auto_{ck}: bars {c['bars']}  all_pass={c['all_pass']}")
        print(f"  means {['%s=%.4f' % (k, v) for k, v in c['means'].items()]}")
        print(f"  v2    {['%s=%.4f' % (k, v) for k, v in c['v2_means'].items()]}")
        print(f"  pairedΔmedian {c['paired_median_delta']:+.5f}  crowded_improved {c['crowded_improved']}"
              f"  sparse_lane_max {c['sparse_lane_max']:.4f}  routed {c['routed_census']}")
    if 'crowded_v1_vs_v3' in out:
        print('crowded v1 vs v3:', out['crowded_v1_vs_v3'])
    print('DECISION:', out['decision'])
    print('report-only pure means:', json.dumps(ro['pure_means'], indent=1))
    print('R2 pure v1 vs v2:', ro['R2_pure_v1_vs_v2'],
          '| lane spread>0.02:', ro['R3_lane_spread_gt02_v1'], ro['R3_lane_spread_gt02_v3'])


if __name__ == '__main__':
    main()
