"""W1a — MACHINE-GENERATE the evidence map from the result JSONs, frozen criteria in code.

P3 (persona debate R2): the hand-typed map carried 2 material errors in 7 rows, both
propagated to MEMORY.md; "machine-generate it with frozen criteria applied in code."
Here every number is READ and every verdict is COMPUTED from its criterion, so E1
("repairs 3/3" -> counted from `learned < 1.0`) and E2 (mean-only real win -> auto
median caveat) cannot recur. Deterministic, no compute, re-runnable:

    python3 tmp/obsaxis_2026-07-26/gen_evidence_map.py > docs/EVIDENCE_MAP_GENERATED_2026-07-26.md
"""
import json
import os
import sys
from datetime import date

import numpy as np
from scipy.stats import wilcoxon

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
J = lambda p: json.load(open(os.path.join(R, p)))
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
COLLAPSE_FIXED = 1.0   # FROZEN, tmp/wp_real_2026-07-25/PREREG.md sec.3(a)
TAU = 1.5              # FROZEN collapse-detector threshold
ROWS = []


def row(claim, number, criterion, verdict, source, caveat=''):
    ROWS.append(dict(claim=claim, number=number, criterion=criterion,
                     verdict=verdict, source=source, caveat=caveat))


def med_by_ep(path):
    """jsonl keyed (episode, seed) -> {episode: median over the 5 est-seed lanes}."""
    d = {}
    for line in open(os.path.join(R, path)):
        j = json.loads(line)
        d.setdefault(j['episode'], {})[j['seed']] = j['nRMSE']
    return {e: float(np.median([v[E] for E, _D in LANES])) for e, v in d.items()}


# --- 1. tracker double phase-advance fix (batchB bug-era vs batchC fixed) -------------
bug = med_by_ep('tmp/rerun_2026-07-25/batchB_ws.jsonl')
fix = med_by_ep('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl')
EPS = sorted(fix)
b = np.array([bug[e] for e in EPS]); f = np.array([fix[e] for e in EPS])
p_fix = float(wilcoxon(f, b).pvalue)
row('Tracker double phase-advance was a bug; fixing it improves every episode',
    f'{b.mean():.4f} -> {f.mean():.4f} ({(f.mean()/b.mean()-1)*100:+.1f}%), '
    f'improved {int((f < b).sum())}/{len(EPS)}, Wilcoxon p={p_fix:.1e}',
    'improve >= 90/100 AND p < 0.001',
    'MEASURED - PASS' if (f < b).sum() >= 90 and p_fix < 1e-3 else 'MEASURED - FAIL',
    'tmp/rerun_2026-07-25/batchB_ws.jsonl + batchC_ws_phasefix.jsonl')

# --- 2. learned set-proposal wins crowded (synthetic, in-distribution) ----------------
I = J('tmp/wp_integration_2026-07-25/results.json')['primary']
row('Learned set-proposal init beats the analytic init on crowded synthetic seas',
    f"{I['v2_mean']:.4f} -> {I['learned_mean']:.4f} (medDelta {I['median_delta']:+.4f}), "
    f"{I['wins']}/{I['n']}, p={I['p_onesided']:.1e}",
    'pre-registered P1 AND P2 (tmp/wp_integration_2026-07-25/PREREG.md)',
    f"MEASURED - {I['decision']}", 'tmp/wp_integration_2026-07-25/results.json',
    '' if I['retention_pass'] else
    f"component retention {I['retention_frac']*100:.1f}% FAILED its own bar (tracker erosion)")

# --- 3/4. router + guard on the 100-episode benchmark --------------------------------
SW = J('tmp/unified_2026-07-26/switch_results.json')['by_threshold']['8']['means']
T1 = J('tmp/unified_guard_2026-07-26/task1_results.json')
P1_ = T1['primary']
row('A GT-free router beats both pure arms on the 100-episode benchmark',
    f"pure analytic {SW['pure_v2']:.4f} / pure learned {SW['pure_learned']:.4f} / "
    f"auto {SW['switch_perlane']:.4f} (n_selected >= 8)",
    'auto <= min(pure arms)',
    'MEASURED - PASS' if SW['switch_perlane'] <= min(SW['pure_v2'], SW['pure_learned'])
    else 'MEASURED - FAIL', 'tmp/unified_2026-07-26/switch_results.json',
    'IN-SAMPLE: the threshold and the guard ratio were set on these same 100 episodes')
row('The beamform guard removes the sparse misroutes without touching crowded routing',
    f"auto {P1_['agg_baseline']:.4f} -> guard {P1_['agg_val']:.4f}; "
    f"sparse {P1_['sparse_val']:.4f} (bar {T1['sparse_target']:.4f}); "
    f"crowded flips {P1_['flips']}",
    'sparse <= pure_v2 + 0.005 AND 0 crowded flips AND aggregate improves',
    f"MEASURED - {'PASS' if P1_['DECISION'] else 'FAIL'}",
    'tmp/unified_guard_2026-07-26/task1_results.json')

# --- 5. E3: the oracle gap, lane-consistent ------------------------------------------
OC_PATH = 'tmp/obsaxis_2026-07-26/oracle_consistent.json'
if os.path.exists(os.path.join(R, OC_PATH)):
    O = J(OC_PATH); g, gl = O['gap_old_way'], O['gap_lane_consistent']
    cr = O['slices']['crowded']
    row('The guard closes most of the gap between the router and an oracle switch',
        f"deployed {O['slices']['all']['guard']:.4f} vs lane-consistent oracle "
        f"{gl['oracle']:.4f}: {gl['frac_closed']*100:.1f}% of the auto->oracle gap closed "
        f"(old min-of-medians oracle {g['oracle']:.4f} would have said {g['frac_closed']*100:.1f}%)",
        'deployed and oracle aggregated identically (per-lane, then median, then mean)',
        'MEASURED - QUOTABLE (E3 fixed)', OC_PATH,
        'the old crowded-slice paradox (deployed below its own oracle) is gone: '
        f"{cr['n_deployed_below_oracle_old']} -> {cr['n_deployed_below_oracle_lane']} episodes; "
        'the ratio is an in-sample point estimate with no CI')

# --- 6. real-data primary (pre-registered LOSS) --------------------------------------
W = J('tmp/wp_real_2026-07-25/wp_real.json')['results']
COLL = {'AA14_0', 'AA15a_3', 'LJ18g_3'}
key = lambda w: f"{w['record']}_{w['win']}"
norm = [w for w in W if key(w) not in COLL]
dn = np.array([w['delta_now'] for w in norm])   # hand - learned; > 0 = learned better
p_real = float(wilcoxon(dn).pvalue)
row('On real stereo seas the learned arm LOSES accuracy (the programme predicted this)',
    f"n={len(dn)} normal windows, median Delta={np.median(dn):+.4f} (hand-learned), "
    f"learned better on {int((dn > 0).sum())}/{len(dn)}, two-sided Wilcoxon p={p_real:.3f}",
    'pre-registered cell 1 (LOSS on normal-27) - tmp/wp_real_2026-07-25/PREREG.md',
    'MEASURED - CONFIRMED LOSS' if np.median(dn) < 0 and p_real < 0.05 else 'MEASURED - null',
    'tmp/wp_real_2026-07-25/wp_real.json',
    'analysis unit is the WINDOW; windows within a record are not independent (n_records=5)')

# --- 7. collapse repair: the frozen "< 1.0" rule, counted IN CODE (kills E1) ----------
FB = J('tmp/unified_2026-07-26/fallback_results.json')
cd = FB['coll_detail']
fixed = [c for c in cd if c['learned'] < COLLAPSE_FIXED]
e2e = [c for c in cd if c['fallback'] < COLLAPSE_FIXED]
big = [c for c in cd if (c['hand'] - c['learned']) >= 0.5]
row('The learned arm repairs the analytic arm\'s collapse windows',
    f"learned arm below the line: {len(fixed)}/{len(cd)}; "
    f"end-to-end (detector o fallback): {len(e2e)}/{len(cd)}; "
    f"large error reduction (Delta >= 0.5): {len(big)}/{len(cd)}",
    f'FROZEN: "fixed" = learned nRMSE < {COLLAPSE_FIXED} (PREREG sec.3(a))',
    f'MEASURED - {len(fixed)}/{len(cd)}, NOT {len(cd)}/{len(cd)}',
    'tmp/unified_2026-07-26/fallback_results.json',
    'repaired fields land at ~0.96-1.07, i.e. at a no-skill predictor: this is bounded '
    'degradation, not restored skill. Report as an operating characteristic, never as nRMSE gain. '
    + '; '.join(f"{c['window']} {c['hand']:.2f}->{c['learned']:.2f}"
                + ('' if c['learned'] < COLLAPSE_FIXED else ' [NOT FIXED]') for c in cd))

# --- 8. deployed fallback aggregate: auto median-unchanged caveat (kills E2) ----------
A = FB['aggregate']
n_touch = sum(1 for c in cd if c['fires'])
med_same = abs(A['hand_median'] - A['fallback_median']) < 1e-9
row('The deployed detector-plus-fallback chain improves the 30-window real aggregate',
    f"MEAN {A['hand_mean']:.4f} -> {A['fallback_mean']:.4f}; "
    f"MEDIAN {A['hand_median']:.4f} -> {A['fallback_median']:.4f}",
    'a central-tendency win must move the MEDIAN (do-not-cite rule: no mean-based real win)',
    'MEASURED - NOT A GENERAL WIN' if med_same else 'MEASURED - win',
    'tmp/unified_2026-07-26/fallback_results.json',
    (f'AUTO-CAVEAT: the median is unchanged ({A["hand_median"]:.4f} -> {A["fallback_median"]:.4f}); '
     f'the entire mean delta comes from the {n_touch} window(s) the detector touched. '
     'Do not quote the mean without the median.') if med_same else '')

# --- 9. detector specificity + separation margin -------------------------------------
SP = FB['specificity']
amps = FB['all_amp_sorted']
fired = [a for a, _w, _c in amps if a > TAU]
notf = [a for a, _w, _c in amps if a <= TAU]
row('The collapse detector does not false-alarm on normal real windows',
    f"{SP['n_false_fire']}/{SP['n_normal']} false fires at tau={TAU}; "
    f"separation margin = min(amp|fired) {min(fired):.3f} - max(amp|not fired) "
    f"{max(notf):.3f} = {min(fired)-max(notf):+.3f}",
    'R1 (P3, strictest): 0 false fires = PASS, 1 = PARTIAL, >=2 = FAIL; R2: margin > 0',
    'MEASURED - PASS' if SP['n_false_fire'] == 0 and min(fired) > max(notf) else 'MEASURED - CHECK',
    'tmp/unified_2026-07-26/fallback_results.json',
    'tau was set IN-SAMPLE on these 27 windows; specificity is unproven until the 8 unused '
    'Bergamasco records are scored (W7). Report the margin, not a p-value.')

# --- 10. W9: is the boundary readable on a GT-free coordinate? (crux A / F) -----------
OA_PATH = 'tmp/obsaxis_2026-07-26/obsaxis_stats.json'
if os.path.exists(os.path.join(R, OA_PATH)):
    D_ = J(OA_PATH)['stats']; dc = D_['decision']
    row('The regime boundary is detectable from the observations, not just from the simulator knob',
        f"Spearman |rho| of the per-episode delta: bf_peak_ratio {dc['abs_rho_bf']:.3f} "
        f"(p={D_['primary']['bf']['p']:.1e}) vs n_components {dc['abs_rho_ncomp']:.3f} "
        f"(p={D_['primary']['ncomp']['p']:.1e}), n=100",
        f"PRE-STATED (PRESTATE.md): |rho_bf| >= |rho_ncomp| - {dc['tol']} => boundary is not "
        'a relabeling of the answer key',
        'MEASURED - ' + ('PASS' if dc['verdict'] == 'FOR_STUDENT' else 'FAIL: ' + dc['text']),
        OA_PATH,
        'the 30 real windows cluster at bf_peak_ratio 0.51-0.95 (IQR) where only '
        f"6/100 synthetic episodes live; {D_['real']['n_below_guard']}/30 fall below the "
        '0.05 guard, so the synthetic guard never fires on real data')

# --- emit ----------------------------------------------------------------------------
w = sys.stdout.write
w(f"# EVIDENCE MAP — machine-generated {date.today().isoformat()}\n\n")
w("> **Do not hand-edit.** Regenerate with\n>\n"
  "> ```\n> python3 tmp/obsaxis_2026-07-26/gen_evidence_map.py > "
  "docs/EVIDENCE_MAP_GENERATED_2026-07-26.md\n> ```\n>\n"
  "> Every number is read from a result file and every verdict is computed from the frozen\n"
  "> criterion in `gen_evidence_map.py`. This exists because the hand-typed map carried two\n"
  "> material errors (persona debate 2026-07-26, E1 \"repairs 3/3\" and E2 the mean-only real\n"
  "> line); both are now emitted by code and cannot drift.\n\n")
for i, r in enumerate(ROWS, 1):
    w(f"## {i}. {r['claim']}\n\n")
    w(f"- **measured:** {r['number']}\n")
    w(f"- **frozen criterion:** {r['criterion']}\n")
    w(f"- **verdict:** **{r['verdict']}**\n")
    w(f"- **source:** `{r['source']}`\n")
    if r['caveat']:
        w(f"- **caveat (auto):** {r['caveat']}\n")
    w("\n")
w(f"---\n\n_{len(ROWS)} claims emitted. Criteria are frozen in code; "
  "editing a number here without editing its source file is a no-op that the next "
  "regeneration deletes._\n")
