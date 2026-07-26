"""Apply the FROZEN bars of tmp/heldout_real_2026-07-26/PREREG.md §3. Mechanical.

Inputs (read-only):
  tmp/unified_2026-07-26/hand_amp.json      -- 30 DESIGN windows (bit-identity controlled)
  tmp/heldout_real_2026-07-26/harvest_heldout.json -- 48 HELD-OUT windows (this study)
  tmp/wp_real_2026-07-25/wp_real.json       -- for W6 record-blocked re-analysis
Writes tmp/heldout_real_2026-07-26/results.json
"""
import json, itertools
import numpy as np
from scipy.stats import wilcoxon, binomtest, ttest_1samp

TAU = 1.5             # FROZEN detector threshold
COLLAPSE_NRMSE = 1.0  # FROZEN collapse definition
ADEQ_NPX = 25         # FROZEN grid-adequacy subset bar
DESIGN_COLLAPSE = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}

fires = lambda r: r['amp_ratio_grid'] > TAU
coll = lambda r: r['hand_now'] >= COLLAPSE_NRMSE


def load():
    d = json.load(open('tmp/unified_2026-07-26/hand_amp.json'))
    assert abs(d['hand_now_maxdiff']) < 1e-9, 'design bit-identity control failed'
    design = [dict(record=r['record'], win=r['win'], arena='design',
                   hand_now=r['hand_now'], amp_ratio_grid=r['amp_ratio_grid'],
                   amp_ratio_obs=r['amp_ratio_obs'], Hs=r['Hs'], npx=None,
                   learned_now=r['learned_now']) for r in d['rows']]
    h = json.load(open('tmp/heldout_real_2026-07-26/harvest_heldout.json'))
    held = [dict(record=r['record'], win=r['win'], arena='heldout',
                 hand_now=r['hand_now'], amp_ratio_grid=r['amp_ratio_grid'],
                 amp_ratio_obs=r['amp_ratio_obs'], Hs=r['Hs'], npx=r['npx'],
                 lam_p=r['lam_p'], learned_now=None) for r in h['rows']]
    assert len(design) == 30 and len(held) == 48
    # frozen collapse definition must reproduce the hard-coded design COLLAPSE set
    assert {(r['record'], r['win']) for r in design if coll(r)} == DESIGN_COLLAPSE
    return design, held


def exact_ci(k, n):
    return [float(x) for x in binomtest(k, n).proportion_ci(method='exact')]


def r1(held):
    normal = [r for r in held if not coll(r)]
    ff = [r for r in normal if fires(r)]
    recs = sorted({r['record'] for r in held})
    per_rec = {rc: sum(1 for r in ff if r['record'] == rc) for rc in recs}
    n_rec_ff = sum(1 for v in per_rec.values() if v > 0)
    max_per_rec = max(per_rec.values()) if per_rec else 0
    if len(ff) == 0:
        verdict = 'PASS'
    elif len(ff) == 1 and max_per_rec <= 1:
        verdict = 'PARTIAL'
    else:
        verdict = 'FAIL'
    if max_per_rec > 1:
        verdict = 'FAIL'
    return dict(n_windows=len(held), n_normal=len(normal), n_false_fire=len(ff),
                false_fired=[f"{r['record']}_w{r['win']}({r['amp_ratio_grid']:.3f})" for r in ff],
                per_record_false_fires=per_rec, n_records=len(recs),
                n_records_with_ff=n_rec_ff, max_ff_in_one_record=max_per_rec,
                cluster_robust_ci_records=exact_ci(n_rec_ff, len(recs)),
                window_level_ci=exact_ci(len(ff), len(normal)) if normal else None,
                verdict=verdict)


def r2(rows, label):
    normal = [r for r in rows if not coll(r)]
    fired_coll = [r for r in rows if coll(r) and fires(r)]
    mn = min((r['amp_ratio_grid'] for r in fired_coll), default=None)
    mx = max((r['amp_ratio_grid'] for r in normal), default=None)
    over = [f"{r['record']}_w{r['win']}({r['amp_ratio_grid']:.3f})"
            for r in normal if r['amp_ratio_grid'] > TAU]
    return dict(arena=label, n=len(rows), n_normal=len(normal),
                n_collapse=sum(1 for r in rows if coll(r)), n_collapse_fired=len(fired_coll),
                min_amp_fired_collapse=mn, max_amp_normal=mx,
                margin=(mn - mx) if (mn is not None and mx is not None) else None,
                normals_above_tau=over, separation_broken=bool(over))


def r3(design, held):
    new = [r for r in held if coll(r)]
    det = [r for r in new if fires(r)]
    allc = [r for r in design + held if coll(r)]
    alld = [r for r in allc if fires(r)]
    per_rec = {}
    for r in new:
        per_rec.setdefault(r['record'], []).append(r['win'])
    return dict(n_new_collapse=len(new), n_new_detected=len(det),
                new_events=[dict(window=f"{r['record']}_w{r['win']}", hand_now=r['hand_now'],
                                 amp=r['amp_ratio_grid'], fires=fires(r),
                                 npx=r.get('npx'), Hs=r['Hs']) for r in
                            sorted(new, key=lambda x: -x['hand_now'])],
                per_record=per_rec,
                collapse_rate_heldout=len(new) / len(held),
                pooled_n_collapse=len(allc), pooled_n_detected=len(alld),
                pooled_detection_rate=len(alld) / len(allc) if allc else None,
                sensitivity_ci_new=exact_ci(len(det), len(new)) if new else None,
                underpowered=len(new) < 3,
                count_a_large_reduction='PENDING-GPU (learned arm not run)',
                count_b_below_line='PENDING-GPU (learned arm not run)',
                nonlearned_escape_controls='NOT RUN (out of scope, W8 remainder)')


def w6():
    d = json.load(open('tmp/wp_real_2026-07-25/wp_real.json'))['results']
    normal = [r for r in d if (r['record'], r['win']) not in DESIGN_COLLAPSE]
    assert len(normal) == 27
    out = {}
    for label, rows in (('normal27', normal), ('all30', d)):
        recs = sorted({r['record'] for r in rows})
        dl = {rc: [r['hand_now'] - r['learned_now'] for r in rows if r['record'] == rc]
              for rc in recs}
        m = np.array([np.mean(dl[rc]) for rc in recs])
        obs = float(m.mean())
        perm = [abs(float((m * np.array(s)).mean()))
                for s in itertools.product([1, -1], repeat=len(recs))]
        p_perm = float(np.mean(np.array(perm) >= abs(obs) - 1e-12))
        allw = np.array([v for rc in recs for v in dl[rc]])
        w_win = wilcoxon(allw)
        try:
            w_rec = wilcoxon(m)
            p_wrec = float(w_rec.pvalue)
        except Exception:
            p_wrec = None
        t = ttest_1samp(m, 0.0)
        out[label] = dict(
            n_windows=len(rows), n_records=len(recs),
            record_means={rc: float(np.mean(dl[rc])) for rc in recs},
            record_n={rc: len(dl[rc]) for rc in recs},
            signs_all_same=bool(np.all(m < 0) or np.all(m > 0)),
            n_records_learned_worse=int(np.sum(m < 0)),
            window_level_mean=float(allw.mean()), window_level_p=float(w_win.pvalue),
            record_level_stat=obs, perm_p_two_sided=p_perm,
            perm_p_floor=2.0 / 2 ** len(recs),
            wilcoxon_record_p=p_wrec,
            ttest_record_p=float(t.pvalue), ttest_record_t=float(t.statistic))
    n27 = out['normal27']
    n27['verdict'] = ('PASS-in-direction' if (n27['signs_all_same'] and
                      abs(n27['perm_p_two_sided'] - n27['perm_p_floor']) < 1e-9)
                      else 'FAIL-in-direction')
    return out


def main():
    design, held = load()
    allrows = design + held
    adeq = [r for r in held if r['npx'] >= ADEQ_NPX]
    res = dict(tau=TAU, collapse_def=COLLAPSE_NRMSE,
               R1_primary=r1(held), R1_adequacy_subset=r1(adeq),
               R2=dict(pooled=r2(allrows, 'pooled(design+heldout)'),
                       design=r2(design, 'design(in-sample)'),
                       heldout=r2(held, 'heldout'),
                       heldout_adequacy=r2(adeq, 'heldout_adequacy(npx>=25)')),
               R3=r3(design, held), W6=w6(),
               heldout_table=sorted(
                   [dict(w=f"{r['record']}_w{r['win']}", hand=round(r['hand_now'], 3),
                         amp=round(r['amp_ratio_grid'], 3),
                         amp_obs=round(r['amp_ratio_obs'], 3),
                         coll=coll(r), fire=fires(r), npx=r['npx'],
                         Hs=round(r['Hs'], 2), lam_p=round(r['lam_p'], 1))
                    for r in held], key=lambda x: -x['amp']))
    json.dump(res, open('tmp/heldout_real_2026-07-26/results.json', 'w'), indent=1)

    p = print
    p(f"=== R1 held-out specificity (tau={TAU}) ===")
    for k in ('n_windows', 'n_normal', 'n_false_fire', 'false_fired',
              'per_record_false_fires', 'n_records_with_ff', 'cluster_robust_ci_records',
              'window_level_ci', 'verdict'):
        p(f"  {k:26s} {res['R1_primary'][k]}")
    p(f"  -- adequacy subset (npx>=25): n_normal={res['R1_adequacy_subset']['n_normal']} "
      f"ff={res['R1_adequacy_subset']['n_false_fire']} "
      f"verdict={res['R1_adequacy_subset']['verdict']}")
    p("\n=== R2 margin ===")
    for k, v in res['R2'].items():
        p(f"  {v['arena']:32s} nC={v['n_collapse']} fired={v['n_collapse_fired']} "
          f"minAmpFired={v['min_amp_fired_collapse']} maxAmpNormal={v['max_amp_normal']} "
          f"margin={v['margin']} broken={v['separation_broken']} {v['normals_above_tau']}")
    p("\n=== R3 collapse harvest ===")
    R = res['R3']
    p(f"  new collapse events (held-out): {R['n_new_collapse']} / {len(held)} windows "
      f"(rate {R['collapse_rate_heldout']:.1%})")
    p(f"  detected by frozen detector   : {R['n_new_detected']}/{R['n_new_collapse']} "
      f"CI {R['sensitivity_ci_new']}")
    p(f"  pooled (design+heldout)       : {R['pooled_n_detected']}/{R['pooled_n_collapse']} "
      f"= {R['pooled_detection_rate']:.2f}")
    p(f"  UNDERPOWERED (<3 new)         : {R['underpowered']}")
    for e in R['new_events']:
        p(f"    {e['window']:12s} hand={e['hand_now']:7.3f} amp={e['amp']:6.3f} "
          f"fires={str(e['fires']):5s} npx={e['npx']}")
    p(f"  (a) Delta>=0.5 : {R['count_a_large_reduction']}")
    p(f"  (b) learned<1.0: {R['count_b_below_line']}")
    p("\n=== W6 record-blocked re-analysis of the 27-window primary ===")
    for lab in ('normal27', 'all30'):
        w = res['W6'][lab]
        p(f"  [{lab}] window-level mean {w['window_level_mean']:+.4f} "
          f"p_wilcoxon(window)={w['window_level_p']:.4g}")
        p(f"    record means: " + "  ".join(
            f"{k}={v:+.4f}(n={w['record_n'][k]})" for k, v in w['record_means'].items()))
        p(f"    all same sign={w['signs_all_same']}  learned-worse in "
          f"{w['n_records_learned_worse']}/{w['n_records']} records")
        p(f"    record-blocked permutation p={w['perm_p_two_sided']:.4f} "
          f"(attainable floor {w['perm_p_floor']:.4f}); "
          f"wilcoxon(record) p={w['wilcoxon_record_p']}; "
          f"t(record) p={w['ttest_record_p']:.4f}")
    p(f"  VERDICT: {res['W6']['normal27']['verdict']}")
    p("\n=== held-out windows sorted by amp_ratio_grid ===")
    p(f"  {'window':12s} {'hand':>7s} {'amp':>6s} {'ampobs':>7s} {'coll':>5s} "
      f"{'fire':>5s} {'npx':>4s} {'Hs':>6s} {'lam_p':>6s}")
    for r in res['heldout_table']:
        p(f"  {r['w']:12s} {r['hand']:7.3f} {r['amp']:6.3f} {r['amp_obs']:7.3f} "
          f"{str(r['coll']):>5s} {str(r['fire']):>5s} {r['npx']:4d} {r['Hs']:6.2f} "
          f"{r['lam_p']:6.1f}")


if __name__ == '__main__':
    main()
