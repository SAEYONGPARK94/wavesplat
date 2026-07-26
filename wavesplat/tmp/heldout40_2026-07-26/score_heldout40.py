"""THE SINGLE SCORING EVENT -- W4 held-out forty.

Frozen protocol : tmp/heldout40_2026-07-26/PREREG.md  (bars S1..S7 + reporting rule)
Recovered defs  : tmp/heldout40_2026-07-26/SCORING_DEFINITIONS.md (S6, S7 formulas; frozen
                  from in-sample data before any held-out number was computed)

Run ONCE.  Emits every S1..S7 verdict in code (judge-doc C13: evidence maps are generated,
never hand-typed) to tmp/heldout40_2026-07-26/score_results.json and to stdout.

  python3 tmp/heldout40_2026-07-26/score_heldout40.py
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

R = 'tmp/heldout40_2026-07-26/'
V2 = R + 'v2_arm.jsonl'
LRN = R + 'arm_runs.jsonl'
CACHE = R + 'atoms_cache.jsonl'
BF = R + 'bf_ratio.json'
AUTO_LIVE = R + 'arm_runs_auto_live.jsonl'
V2_CTRL = R + 'arm_runs_v2_control.jsonl'
OUT = R + 'score_results.json'

# ---- FROZEN constants (PREREG Sec 1) ---------------------------------------
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
THRESH = 8
GUARD_RATIO = 0.05
# ---- FROZEN bar parameters (PREREG Sec 4.2 + SCORING_DEFINITIONS.md) -------
S1B_WORDING = 0.095
S1B_SURVIVAL = 0.1011
S2_MARGIN = 0.005
S3_MARGIN = 0.005
S4_MAX = 0.015
S6_COST = 0.03          # "damaging" misroute cost, calibrated to in-sample 5/100
S6_MAX = 2
DISCORDANT_MIN = 8
NBOOT = 20000
RNG = np.random.default_rng(20260726)


def comp(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


def regime(c):
    return 'sparse' if c in (1, 2) else ('moderate' if c == 5 else 'crowded')


def realization(ep):
    """uniform/slope share a wave realization for a given (scenario, seed) -- PREREG 2.3."""
    return re.sub(r'_(uniform|slope)\.h5$', '', ep)


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


# ---------------------------------------------------------------- load ------
def load():
    v2, eps = {}, set()
    for l in open(V2):
        d = json.loads(l)
        v2[(d['episode'], d['seed'])] = d['nRMSE']
        eps.add(d['episode'])
    learned = {}
    for l in open(LRN):
        d = json.loads(l)
        if d.get('arm') == 'learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
            learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
    nsel = {}
    for l in open(CACHE):
        d = json.loads(l)
        if d.get('n_selected') is not None:
            nsel[(d['episode'], d['diff_seed'])] = d['n_selected']
    bfr = json.load(open(BF))
    return sorted(eps), v2, learned, nsel, bfr


def assemble(eps, v2, learned, nsel, bfr):
    rows, crowded_flips, flip_eps = {}, 0, []
    for ep in eps:
        v2l = [v2.get((ep, E)) for (E, D) in LANES]
        lnl = [learned.get((ep, E, D)) for (E, D) in LANES]
        nsl = [nsel.get((ep, D)) for (E, D) in LANES]
        auto_lane, guard_lane, orc_lane, routes = [], [], [], []
        for (E, D), vv, lv, ns in zip(LANES, v2l, lnl, nsl):
            base = 'learned' if (ns is not None and ns >= THRESH) else 'v2'
            g = base
            if base == 'learned' and bfr.get(ep, 1.0) < GUARD_RATIO:
                g = 'v2'
            auto_lane.append(lv if base == 'learned' else vv)
            guard_lane.append(lv if g == 'learned' else vv)
            orc_lane.append(min(vv, lv))
            routes.append(g)
            if g != base:
                crowded_flips += (comp(ep) in (15, 30))
                if comp(ep) in (15, 30):
                    flip_eps.append((ep, E, D))
        pv, pl = med(v2l), med(lnl)
        c = comp(ep)
        rows[ep] = dict(comp=c, regime=regime(c), pure_v2=pv, pure_learned=pl,
                        auto_noguard=med(auto_lane), guard=med(guard_lane),
                        oracle_lane=med(orc_lane), oracle_old=min(pv, pl),
                        bfr=bfr.get(ep), ns_med=med([x for x in nsl if x is not None]),
                        n_learned_routes=sum(1 for r in routes if r == 'learned'),
                        v2_lanes=v2l, lrn_lanes=lnl, ns_lanes=nsl, routes=routes)
    return rows, crowded_flips, flip_eps


# ------------------------------------------------------------- statistics ---
def bca_ci(d, nboot=NBOOT, alpha=0.05, blocks=None):
    """BCa CI for the mean of d.  blocks=None -> resample elements (episodes);
    otherwise resample the listed block ids and take all their elements."""
    d = np.asarray(d, float)
    n = len(d)
    theta = float(d.mean())
    if blocks is None:
        idx = RNG.integers(0, n, size=(nboot, n))
        boot = d[idx].mean(axis=1)
        jack = np.array([np.delete(d, i).mean() for i in range(n)])
    else:
        blocks = np.asarray(blocks)
        uniq = np.unique(blocks)
        members = [np.where(blocks == b)[0] for b in uniq]
        B = len(uniq)
        boot = np.empty(nboot)
        for i in range(nboot):
            pick = RNG.integers(0, B, size=B)
            boot[i] = d[np.concatenate([members[j] for j in pick])].mean()
        jack = np.array([d[np.concatenate([m for k, m in enumerate(members) if k != j])].mean()
                         for j in range(B)])
    from scipy.stats import norm
    prop = float(np.mean(boot < theta))
    prop = min(max(prop, 1.0 / (2 * nboot)), 1 - 1.0 / (2 * nboot))
    z0 = norm.ppf(prop)
    jm = jack.mean()
    num = float(np.sum((jm - jack) ** 3))
    den = 6.0 * float(np.sum((jm - jack) ** 2)) ** 1.5
    a = num / den if den > 0 else 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        return float(norm.cdf(z0 + (z0 + z) / max(1e-12, (1 - a * (z0 + z)))))
    lo_q, hi_q = adj(zl), adj(zu)
    lo = float(np.quantile(boot, min(max(lo_q, 0.0), 1.0)))
    hi = float(np.quantile(boot, min(max(hi_q, 0.0), 1.0)))
    return dict(mean=theta, lo=lo, hi=hi, z0=float(z0), a=float(a), nboot=nboot,
                n=int(n), n_blocks=(None if blocks is None else int(len(np.unique(blocks)))))


def paired_wilcoxon(a, b, alt='less'):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = int(np.sum(np.abs(d) > 1e-12))
    if nz == 0:
        return dict(n=len(a), n_nonzero=0, median_delta=0.0, p=None, note='all ties')
    try:
        st, p = wilcoxon(a, b, alternative=alt, zero_method='wilcox')
    except Exception as e:
        return dict(n=len(a), n_nonzero=nz, median_delta=float(np.median(d)), p=None,
                    note=repr(e))
    return dict(n=len(a), n_nonzero=nz, median_delta=float(np.median(d)),
                p=float(p), alternative=alt)


def discordant_report(rows, S, key, ref):
    D = [e for e in S if abs(rows[e][key] - rows[e][ref]) > 1e-12]
    rec = dict(n_slice=len(S), n_discordant=len(D),
               underpowered=bool(len(D) < DISCORDANT_MIN))
    if D:
        a = [rows[e][key] for e in D]
        b = [rows[e][ref] for e in D]
        rec['mean_deployed'] = float(np.mean(a))
        rec['mean_ref'] = float(np.mean(b))
        rec['median_delta'] = float(np.median(np.array(a) - np.array(b)))
        rec['wilcoxon_less'] = paired_wilcoxon(a, b, 'less')
        rec['wilcoxon_greater'] = paired_wilcoxon(a, b, 'greater')
        rec['n_win'] = int(sum(1 for e in D if rows[e][key] < rows[e][ref] - 1e-12))
        rec['n_loss'] = int(sum(1 for e in D if rows[e][key] > rows[e][ref] + 1e-12))
    if rec['underpowered']:
        rec['direction'] = 'UNDERPOWERED - no direction claimed'
    return rec


# ------------------------------------------------------------------- main ---
def main():
    eps, v2, learned, nsel, bfr = load()
    assert len(eps) == 40, f'expected 40 held-out episodes, got {len(eps)}'
    missing_v2 = [(e, E) for e in eps for (E, D) in LANES if (e, E) not in v2]
    missing_l = [(e, E, D) for e in eps for (E, D) in LANES if (e, E, D) not in learned]
    missing_n = [(e, D) for e in eps for (E, D) in LANES if (e, D) not in nsel]
    assert not (missing_v2 or missing_l or missing_n), \
        f'incomplete lanes: v2 {len(missing_v2)}, learned {len(missing_l)}, nsel {len(missing_n)}'

    rows, crowded_flips, flip_eps = assemble(eps, v2, learned, nsel, bfr)
    SL = {g: [e for e in eps if rows[e]['regime'] == g]
          for g in ('sparse', 'moderate', 'crowded')}
    SL['all'] = eps
    m = lambda k, S: float(np.mean([rows[e][k] for e in S]))

    means = {name: {k: m(k, S) for k in ('pure_v2', 'pure_learned', 'auto_noguard',
                                         'guard', 'oracle_lane', 'oracle_old')}
             for name, S in SL.items()}
    for name in means:
        means[name]['n'] = len(SL[name])

    # ---------------- S1 : BCa CI of the paired mean difference, both contrasts
    dl = [rows[e]['guard'] - rows[e]['pure_learned'] for e in eps]
    dv = [rows[e]['guard'] - rows[e]['pure_v2'] for e in eps]
    blocks = [realization(e) for e in eps]
    s1 = {}
    for tag, d in (('vs_pure_learned', dl), ('vs_pure_v2', dv)):
        ep_ci = bca_ci(d)
        rl_ci = bca_ci(d, blocks=blocks)
        agree = (ep_ci['hi'] < 0) == (rl_ci['hi'] < 0)
        quoted = ep_ci if agree else rl_ci
        s1[tag] = dict(episode_bootstrap=ep_ci, realization_blocked=rl_ci,
                       bootstraps_agree=bool(agree),
                       quoted='episode' if agree else 'realization_blocked',
                       ci_entirely_below_zero=bool(quoted['hi'] < 0),
                       wilcoxon_less=paired_wilcoxon(
                           [rows[e]['guard'] for e in eps],
                           [rows[e]['pure_learned' if tag == 'vs_pure_learned' else 'pure_v2']
                            for e in eps], 'less'))
    S1 = bool(s1['vs_pure_learned']['ci_entirely_below_zero'] and
              s1['vs_pure_v2']['ci_entirely_below_zero'])

    # ---------------- S1b : absolute wording / survival gates
    agg = means['all']['guard']
    S1b = dict(mean=agg, wording_gate=S1B_WORDING, survival_gate=S1B_SURVIVAL,
               wording_pass=bool(agg <= S1B_WORDING),
               capstone_dead=bool(agg > S1B_SURVIVAL),
               band=('<=0.095 headline wording retained' if agg <= S1B_WORDING else
                     ('>0.1011 capstone dead outright' if agg > S1B_SURVIVAL else
                      'between 0.095 and 0.1011: "matches the better specialist per regime",'
                      ' no aggregate win')))

    # ---------------- S2 / S3
    S2_val, S2_ref = means['sparse']['guard'], means['sparse']['pure_v2']
    S3_val, S3_ref = means['crowded']['guard'], means['crowded']['pure_learned']
    S2 = bool(S2_val <= S2_ref + S2_MARGIN)
    S3 = bool(S3_val <= S3_ref + S3_MARGIN)

    # ---------------- S4 : deployed - oracle, lane-consistent (primary)
    S4_gap_lane = means['all']['guard'] - means['all']['oracle_lane']
    S4_gap_old = means['all']['guard'] - means['all']['oracle_old']
    S4 = bool(S4_gap_lane <= S4_MAX)

    # ---------------- S5 : guard fires on 0 crowded AND min bfr | crowded > 0.05
    crowded_bfr = [rows[e]['bfr'] for e in SL['crowded']]
    S5_min_bfr = float(np.min(crowded_bfr))
    S5 = bool(crowded_flips == 0 and S5_min_bfr > GUARD_RATIO)

    # ---------------- S6 : damaging misroutes  (guard - oracle_lane > 0.03)
    dm_lane = sorted(((rows[e]['guard'] - rows[e]['oracle_lane']), e) for e in eps)[::-1]
    dmg = [dict(episode=e, comp=rows[e]['comp'], cost=c, guard=rows[e]['guard'],
                oracle_lane=rows[e]['oracle_lane'], pure_v2=rows[e]['pure_v2'],
                pure_learned=rows[e]['pure_learned'], bfr=rows[e]['bfr'],
                ns_med=rows[e]['ns_med'], n_learned_routes=rows[e]['n_learned_routes'])
           for c, e in dm_lane if c > S6_COST]
    dmg_old = int(sum(1 for e in eps if rows[e]['guard'] - rows[e]['oracle_old'] > S6_COST))
    S6 = bool(len(dmg) <= S6_MAX)

    # ---------------- S7 : moderate cell must not lose more than it wins
    MOD = SL['moderate']
    W7 = int(sum(1 for e in MOD if rows[e]['guard'] < rows[e]['pure_v2'] - 1e-12))
    L7 = int(sum(1 for e in MOD if rows[e]['guard'] > rows[e]['pure_v2'] + 1e-12))
    S7 = bool(L7 <= W7)

    # ---------------- reporting rule : discordance per slice
    disc = dict(
        sparse_vs_pure_v2=discordant_report(rows, SL['sparse'], 'guard', 'pure_v2'),
        crowded_vs_pure_learned=discordant_report(rows, SL['crowded'], 'guard', 'pure_learned'),
        moderate_vs_pure_v2=discordant_report(rows, SL['moderate'], 'guard', 'pure_v2'),
        all_vs_pure_v2=discordant_report(rows, eps, 'guard', 'pure_v2'),
        all_vs_pure_learned=discordant_report(rows, eps, 'guard', 'pure_learned'))

    # ---------------- free paired Wilcoxons
    wil = {}
    for name, S in SL.items():
        wil[name] = dict(
            guard_vs_pure_v2=paired_wilcoxon([rows[e]['guard'] for e in S],
                                             [rows[e]['pure_v2'] for e in S], 'less'),
            guard_vs_pure_learned=paired_wilcoxon([rows[e]['guard'] for e in S],
                                                  [rows[e]['pure_learned'] for e in S], 'less'))

    # ---------------- routing census
    route_learned_lanes = int(sum(rows[e]['n_learned_routes'] for e in eps))
    census = {g: dict(n=len(S),
                      lanes_routed_learned=int(sum(rows[e]['n_learned_routes'] for e in S)),
                      lanes_total=5 * len(S),
                      episodes_majority_learned=int(sum(rows[e]['n_learned_routes'] >= 3
                                                        for e in S)),
                      min_bfr=float(np.min([rows[e]['bfr'] for e in S])),
                      median_ns=float(np.median([rows[e]['ns_med'] for e in S])))
              for g, S in SL.items()}

    # ---------------- optional live-router cross-check
    live = None
    if os.path.exists(AUTO_LIVE):
        lv = {}
        for l in open(AUTO_LIVE):
            d = json.loads(l)
            if d.get('arm') == 'auto' and d.get('err') is None and d.get('avg_nRMSE') is not None:
                lv[(d['episode'], d['est_seed'], d['diff_seed'])] = d
        if len(lv) >= 5 * len(eps):
            live_ep = {e: med([lv[(e, E, D)]['avg_nRMSE'] for (E, D) in LANES]) for e in eps}
            diffs = [abs(live_ep[e] - rows[e]['guard']) for e in eps]
            live = dict(n=len(lv), mean_live=float(np.mean([live_ep[e] for e in eps])),
                        mean_reassembled=agg, max_abs_episode_diff=float(np.max(diffs)),
                        n_episodes_differ=int(sum(d > 1e-9 for d in diffs)))

    ctrl = None
    if os.path.exists(V2_CTRL):
        cv = {}
        for l in open(V2_CTRL):
            d = json.loads(l)
            if d.get('arm') == 'v2_konly' and d.get('err') is None:
                cv[(d['episode'], d['est_seed'])] = d['avg_nRMSE']
        pairs = [(cv[k], v2[k]) for k in cv if k in v2]
        if pairs:
            dd = [abs(a - b) for a, b in pairs]
            ctrl = dict(n=len(pairs), max_abs_diff=float(np.max(dd)),
                        agrees_1e10=bool(np.max(dd) < 1e-10))

    bars = dict(
        S1=dict(PASS=S1, detail=s1),
        S1b=dict(PASS=bool(S1b['wording_pass']), detail=S1b),
        S2=dict(PASS=S2, value=S2_val, bar=S2_ref + S2_MARGIN, ref_pure_v2=S2_ref,
                margin=S2_MARGIN),
        S3=dict(PASS=S3, value=S3_val, bar=S3_ref + S3_MARGIN, ref_pure_learned=S3_ref,
                margin=S3_MARGIN),
        S4=dict(PASS=S4, gap_lane_consistent=S4_gap_lane, gap_old_min_of_medians=S4_gap_old,
                bar=S4_MAX, oracle_lane=means['all']['oracle_lane'],
                oracle_old=means['all']['oracle_old'], deployed=agg),
        S5=dict(PASS=S5, crowded_guard_flips=crowded_flips, flip_lanes=flip_eps,
                min_bf_peak_ratio_crowded=S5_min_bfr, guard_ratio=GUARD_RATIO),
        S6=dict(PASS=S6, n_damaging=len(dmg), bar=S6_MAX, cost_threshold=S6_COST,
                n_damaging_old_oracle=dmg_old, misroutes=dmg),
        S7=dict(PASS=S7, wins=W7, losses=L7, ties=len(MOD) - W7 - L7,
                n_moderate=len(MOD)))

    out = dict(
        protocol=dict(lanes=LANES, thresh=THRESH, guard_ratio=GUARD_RATIO,
                      TRACK_PHASE_PREDICT=False, n_episodes=len(eps), nboot=NBOOT,
                      oracle='lane-consistent: median_lanes(min(v2_lane, learned_lane))',
                      s6_definition='guard - oracle_lane > 0.03 (in-sample calibrated 5/100)',
                      s7_definition='moderate: W=#(guard<pure_v2), L=#(guard>pure_v2), '
                                    'PASS iff L<=W (in-sample 4W/6L)'),
        means=means, bars=bars, discordance=disc, wilcoxon=wil, routing_census=census,
        live_router_crosscheck=live, v2_protocol_control=ctrl,
        per_episode={e: {k: v for k, v in rows[e].items()
                         if k not in ('v2_lanes', 'lrn_lanes', 'ns_lanes', 'routes')}
                     for e in eps})
    json.dump(out, open(OUT, 'w'), indent=1)

    # ------------------------------------------------------------- printout --
    P = lambda b: 'PASS' if b else 'FAIL'
    print('=' * 90)
    print('W4 HELD-OUT FORTY -- THE SINGLE SCORING EVENT (one shot, frozen bars)')
    print('=' * 90)
    print(f"{'slice':10s} {'n':>3} {'pure_v2':>9} {'pure_lrn':>9} {'auto(no g)':>10} "
          f"{'AUTO+GUARD':>10} {'orc_lane':>9} {'orc_old':>9}")
    for name in ('all', 'sparse', 'moderate', 'crowded'):
        d = means[name]
        print(f"{name:10s} {d['n']:>3} {d['pure_v2']:>9.5f} {d['pure_learned']:>9.5f} "
              f"{d['auto_noguard']:>10.5f} {d['guard']:>10.5f} {d['oracle_lane']:>9.5f} "
              f"{d['oracle_old']:>9.5f}")

    print('\n--- BARS ---')
    for tag in ('vs_pure_learned', 'vs_pure_v2'):
        c = s1[tag]['episode_bootstrap']; r = s1[tag]['realization_blocked']
        print(f"S1 {tag:16s}: mean d={c['mean']:+.5f}  episode BCa [{c['lo']:+.5f},{c['hi']:+.5f}]"
              f"  realiz-blocked [{r['lo']:+.5f},{r['hi']:+.5f}]"
              f"  -> {P(s1[tag]['ci_entirely_below_zero'])} (quoted: {s1[tag]['quoted']})"
              f"  wilcoxon p={s1[tag]['wilcoxon_less']['p']}")
    print(f"S1  (PRIMARY, both required)                       : {P(S1)}")
    print(f"S1b mean={agg:.5f} (wording<=0.095, dead>0.1011)   : {P(S1b['wording_pass'])}"
          f"  [{S1b['band']}]")
    print(f"S2  sparse {S2_val:.5f} <= pure_v2 {S2_ref:.5f}+0.005={S2_ref+S2_MARGIN:.5f} : {P(S2)}")
    print(f"S3  crowded {S3_val:.5f} <= pure_lrn {S3_ref:.5f}+0.005={S3_ref+S3_MARGIN:.5f}: {P(S3)}")
    print(f"S4  deployed-oracle(lane) {S4_gap_lane:+.5f} <= 0.015 : {P(S4)}"
          f"   [old-oracle {S4_gap_old:+.5f}]")
    print(f"S5  crowded guard flips={crowded_flips}, min bfr|crowded={S5_min_bfr:.4f} > 0.05 : {P(S5)}")
    print(f"S6  damaging misroutes (cost>0.03) = {len(dmg)} <= 2 : {P(S6)}"
          f"   [old-oracle count {dmg_old}]")
    for d0 in dmg:
        print(f"      {d0['episode']:44s} c{d0['comp']:2d} guard={d0['guard']:.4f} "
              f"orc={d0['oracle_lane']:.4f} cost=+{d0['cost']:.4f} bfr={d0['bfr']:.4f} "
              f"ns={d0['ns_med']:.1f} learned_lanes={d0['n_learned_routes']}/5")
    print(f"S7  moderate W={W7} L={L7} T={len(MOD)-W7-L7}, PASS iff L<=W : {P(S7)}")

    print('\n--- REPORTING RULE: discordance per slice ---')
    for k, v in disc.items():
        line = (f"  {k:26s} n_slice={v['n_slice']:>2d} n_discordant={v['n_discordant']:>2d}"
                f" {'UNDERPOWERED' if v['underpowered'] else ''}")
        if v['n_discordant']:
            line += (f"  medΔ={v['median_delta']:+.5f} W/L={v.get('n_win')}/{v.get('n_loss')}"
                     f" p_less={v['wilcoxon_less']['p']}")
        print(line)

    print('\n--- routing census ---')
    for g in ('all', 'sparse', 'moderate', 'crowded'):
        c = census[g]
        print(f"  {g:9s} lanes routed learned {c['lanes_routed_learned']:>3d}/{c['lanes_total']:>3d}"
              f"  episodes majority-learned {c['episodes_majority_learned']:>2d}/{c['n']:<2d}"
              f"  min bfr={c['min_bfr']:.4f}  median n_sel={c['median_ns']:.1f}")
    if live:
        print(f"\n  live-router cross-check: mean_live={live['mean_live']:.5f} vs "
              f"re-assembled={live['mean_reassembled']:.5f}, "
              f"max |Δ| per episode={live['max_abs_episode_diff']:.2e}, "
              f"episodes differing={live['n_episodes_differ']}")
    if ctrl:
        print(f"  v2 protocol control: n={ctrl['n']} max|Δ|={ctrl['max_abs_diff']:.3e} "
              f"agrees@1e-10={ctrl['agrees_1e10']}")

    n_pass = sum(bars[k]['PASS'] for k in ('S1', 'S1b', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'))
    print(f"\nBARS PASSED: {n_pass}/8  "
          f"({', '.join(k + '=' + P(bars[k]['PASS']) for k in ('S1','S1b','S2','S3','S4','S5','S6','S7'))})")
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
