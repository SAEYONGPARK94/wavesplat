#!/usr/bin/env python3
"""현황판(`STATUS.md`)·인계문(`NEXT.md`)이 인용하는 **모든 수를 동결 산출물에서 재도출**하고,
문서에 적힌 문자열과 어긋나면 `AssertionError` 로 죽는다.

⛔**추정기 런 0회** — 이 스크립트는 `verdict_*.json` · `premise_*.json` · `probe_*.json` ·
동결 `results*.jsonl` 만 읽는다. 채점 0회, 패키지 임포트 0회.

본보기 = `tmp/STATUS_2026-08-28/f1_table.py`(같은 규율).
실행:
    python3 /home/vexit/wavesplat/wavesplat/tmp/STATUS_2026-08-30/status_numbers.py
불일치 시 exit 1.
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict
from math import comb

ROOT = '/home/vexit/wavesplat/wavesplat'
TMP = os.path.join(ROOT, 'tmp')


def J(rel):
    with open(os.path.join(TMP, rel), encoding='utf-8') as f:
        return json.load(f)


def L(rel):
    with open(os.path.join(TMP, rel), encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def frac(d, kw='better', kn='n'):
    return '%d/%d' % (d[kw], d[kn])


def pct(x, sign=True):
    """상대 중앙값을 문서 표기(유니코드 마이너스)로."""
    s = '%+.2f%%' % x if sign else '%.2f%%' % x
    return s.replace('-', '−')


def g(x, digits=3):
    return '%.*g' % (digits, x)


def gates_of(v):
    """게이트 표기 'pass/total' — 두 스키마(리스트-of-dict / 리스트-of-list)를 모두 받는다."""
    if 'gates_pass' in v and 'gates_total' in v:
        return '%d/%d' % (v['gates_pass'], v['gates_total'])
    gs = v['gates']
    ok = sum(1 for x in gs if (x['pass'] if isinstance(x, dict) else x[1]))
    return '%d/%d' % (ok, len(gs))


# ---------------------------------------------------------------- 동결 산출물
T2 = J('T2_STATIONARITY_2026-08-30/verdict_t2stat.json')
T3 = J('T3_NOWCAST_2026-08-30/verdict_t3now.json')
T4P = J('T4_GEOMETRY_2026-08-30/premise_t4.json')
PROBE = J('T3_NOWCAST_2026-08-30/probe_mirror_b.json')
T3P = J('T3_NOWCAST_2026-08-30/premise_t3.json')
B3 = J('B3_REFIT_2026-08-23/verdict_b3.json')
B26 = J('B26_SEASTATE_2026-08-27/verdict_b26.json')
M4 = J('M4_CAPACITY_2026-08-28/verdict_m4.json')
B1 = J('B1_BASELINE_2026-08-22/verdict_b1.json')
B12 = J('B12_HOS_EXTEND_2026-08-23/verdict_b12.json')
B16 = J('B16_KGRID_2026-08-24/verdict_b16.json')
B22 = J('B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json')

# `W-15 §2`(학습 vs **배포 해석적 + 앙상블**)는 verdict json 에 없다 ⟹ 동결 행에서 재도출.
W15R = L('W15_DEPLOY_VERIFY_2026-08-22/results.jsonl')


def w15(stratum):
    def med(cell):
        d = defaultdict(list)
        for x in W15R:
            if x['cell'] == cell and x['stratum'] == stratum:
                d[x['episode']].append(x['avg_nRMSE'])
        return {k: st.median(v) for k, v in d.items()}
    a, b = med('POST'), med('ALL-ANALYTIC')
    eps = sorted(set(a) & set(b))
    n = len(eps)
    w = sum(1 for e in eps if a[e] < b[e])
    rel = st.median([(a[e] - b[e]) / b[e] * 100.0 for e in eps])
    return w, n, rel


W15C = w15('crowded')
W15S = w15('sparse')

# ★C-1 (BLOCKER): 성장 분기 발화율. 배포 **전** 구성(`W3`)과 **출하 삼종 등가** 구성
# (`W5 L-VP-ONLY` · `B_DEPLOY A-VP`)을 동결 행에서 직접 센다. ⛔손복사 아님.
W3R = L('W3_SOLVER_2026-08-15/results.jsonl')
W5R = L('W5_WHY_2026-08-19/results.jsonl')
BDR = L('B_DEPLOY_2026-08-19/results.jsonl')


def _rate(rows, key):
    n = sum(1 for a in rows if (a.get(key) or 0) > 0)
    return n, len(rows)


def w3_growth(learned_only=True):
    rows = [a for a in W3R if a.get('arm') == 'SHIP' and a.get('stratum') == 'crowded']
    if learned_only:
        rows = [a for a in rows if a.get('init_method_resolved') == 'learned']
    return _rate(rows, 'add_calls')


def w5_growth(arm):
    rows = [a for a in W5R if a['arm'] == arm]
    n = sum(1 for a in rows if (a['counters'].get('adam_steps_grow') or 0) > 0)
    return n, len(rows)


def bd_growth(arm):
    rows = [a for a in BDR if a['arm'] == arm and a['stratum'] == 'crowded'
            and a.get('route') == 'learned']
    return _rate(rows, 'adam_steps_grow')


def _fp(t):
    return '%d/%d' % t, '%.1f' % (100.0 * t[0] / t[1])


W3L, W3A = w3_growth(True), w3_growth(False)
W5S, W5V8, W5VO = w5_growth('L-SHIP-DUP2'), w5_growth('L-VP800'), w5_growth('L-VP-ONLY')
BDS, BDV = bd_growth('A-SHIP'), bd_growth('A-VP')

# `T-3` 거울 탐침: bank 멤버 퇴화 분포(`SHIP-K1M3`).
_dist = defaultdict(int)
for row in PROBE:
    _dist[row['n_distinct']] += 1
_NPROBE = len(PROBE)

# `T-4` 선별표: GO / NO-GO.
_go = [c['name'] for c in T4P['screen'] if c['go']]
_nogo = [c['name'] for c in T4P['screen'] if not c['go']]
_t4_verdicts = [f for f in os.listdir(os.path.join(TMP, 'T4_GEOMETRY_2026-08-30'))
                if f.startswith('verdict')]

# ---------------------------------------------------------------- 주장 = (키, 문서 표기, 재도출)
CLAIMS = [
    # ---- T2-STAT
    ('t2.label', 'T2-STAT :: PROP-SKILL-UNDECIDED', T2['label']),
    ('t2.label2', 'T2-STAT :: PROP-REFIT-UNDECIDED', T2['label_secondary']),
    ('t2.gates', '29/29', gates_of(T2)),
    ('t2.rows', '144', str(T2['n_rows'])),
    ('t2.PRIMARY', '14/36', frac(T2['primary'])),
    ('t2.PRIMARY.p', '0.9338', '%.4f' % T2['primary']['p_one_sided']),
    ('t2.PRIMARY.ratio', '1.204', '%.3f' % T2['primary']['ratio_median']),
    ('t2.SECA', '13/36', frac(T2['secondary_a'])),
    ('t2.SECA.rel', '+4.04%', pct(T2['secondary_a']['rel_median_pct'])),
    ('t2.SECA.p', '0.9674', '%.4f' % T2['secondary_a']['p_one_sided']),
    ('t2.SECB', '2/36', '%d/%d' % (T2['secondary_b']['prop_more'], T2['secondary_b']['n'])),
    ('t2.SECB.grade', 'EXPLORATORY', T2['secondary_b']['grade']),
    ('t2.CONFIRM', '25', str(T2['thresholds']['CONFIRM'])),
    ('t2.NOMINAL', '24', str(T2['thresholds']['NOMINAL'])),
    ('t2.NOT_EARN', '12', str(T2['thresholds']['NOT_EARN'])),
    ('t2.mde', '0.742', '%.3f' % T2['mde']['q80']),
    ('t2.now.SHIP', '33/36', frac(T2['descriptive_nowcast']['SHIP-FROZEN'])),
    ('t2.now.SHIP.ratio', '0.779', '%.3f' % T2['descriptive_nowcast']['SHIP-FROZEN']['ratio_median']),
    ('t2.now.PROP', '15/36', frac(T2['descriptive_nowcast']['PROP-FROZEN'])),
    ('t2.now.PROP.ratio', '1.049', '%.3f' % T2['descriptive_nowcast']['PROP-FROZEN']['ratio_median']),
    # ---- 같은 36셀의 bank 쪽 재적합 이득(= `B-3`, `T2 G-PC` 가 36 으로 재확인)
    ('b3.refit', '36/36', frac(B3['paired'])),
    ('b3.refit.rel', '−23.96%', pct(B3['paired']['rel_median_pct'], sign=False)),
    ('t2.GPC.refit', '36', [x['detail'] for x in T2['gates']
                            if x['name'].startswith('G-PC 짝')][0]),

    # ---- T3-NOW
    ('t3.label', 'T3-NOW :: PROP-ENSEMBLE-EARNS (CONFIRMATORY)', T3['label']),
    ('t3.label.c2', 'T3-NOW :: VARPRO-UNDECIDED', T3['label_c2']),
    ('t3.label.c3', 'T3-NOW :: PROP-NOWCAST-UNDECIDED', T3['label_c3']),
    ('t3.gates', '47/47', gates_of(T3)),
    ('t3.rows', '144', str(T3['n_rows'])),
    ('t3.runs', '360/360', [x['detail'] for x in T3['gates']
                            if x['name'].startswith('G-0e')][0]),
    ('t3.C1', '29/36', frac(T3['primary'])),
    ('t3.C1.rel', '−8.65%', pct(T3['primary']['rel_median_pct'], sign=False)),
    ('t3.C1.p', '1.56e-04', '%.2e' % T3['primary']['p_one_sided']),
    ('t3.C2', '23/36', frac(T3['c2'])),
    ('t3.C2.rel', '−5.43%', pct(T3['c2']['rel_median_pct'], sign=False)),
    ('t3.C2.p', '0.0662', '%.4f' % T3['c2']['p_one_sided']),
    ('t3.C3', '21/36', frac(T3['c3'])),
    ('t3.C3.ratio', '0.848', '%.3f' % T3['c3']['ratio_median']),
    ('t3.C3.p', '0.2025', '%.4f' % T3['c3']['p_one_sided']),
    ('t3.holm', 'C1', ','.join(k for k, v in T3['holm_survivors'].items() if v)),
    ('t3.HOLM_FIRST', '25', str(T3['thresholds']['HOLM_FIRST'])),
    ('t3.NOMINAL', '24', str(T3['thresholds']['NOMINAL'])),
    ('t3.NOT_EARN', '12', str(T3['thresholds']['NOT_EARN'])),
    ('t3.mde.family', '0.742', '%.3f' % T3['mde']['q80']),
    ('t3.mde.single', '0.716', '%.3f' % T3['mde_single']['q80']),
    ('t3.E2.bank', '15/36', frac(T3['exploratory']['bank_ensemble'])),
    ('t3.E2.ties', '17', str(T3['exploratory']['bank_ensemble']['ties'])),
    ('t3.E2.rel', '0.00%', pct(T3['exploratory']['bank_ensemble']['rel_median_pct'], sign=False)),
    ('t3.zero.SHIPK3', '33/36', frac(T3['descriptive']['vs_zero']['SHIP-K3'])),
    ('t3.zero.SHIPK3.ratio', '0.729', '%.3f' % T3['descriptive']['vs_zero']['SHIP-K3']['ratio_median']),
    ('t3.zero.V400', '21/36', frac(T3['descriptive']['vs_zero']['PROP-V400M3'])),
    ('t3.zero.V0', '27/36', frac(T3['descriptive']['vs_zero']['PROP-V0M3'])),
    ('t3.zero.V0.ratio', '0.838', '%.3f' % T3['descriptive']['vs_zero']['PROP-V0M3']['ratio_median']),
    ('t3.zero.V0.p', '0.00197', g(T3['descriptive']['vs_zero']['PROP-V0M3']['p_one_sided'])),
    ('t3.spread.V400', '0.559', '%.3f' % T3['exploratory']['spread']['PROP-V400M3']['R_median']),
    ('t3.spread.K1M3', '0.0426', '%.4f' % T3['exploratory']['spread']['SHIP-K1M3']['R_median']),
    ('t3.wall', '13.97', '%.2f' % T3['descriptive']['wall_s_median']),
    # ---- T3 거울 탐침(bank 앙상블 퇴화)
    ('t3.probe.n', '36', str(_NPROBE)),
    ('t3.probe.d3', '16/36', '%d/%d' % (_dist[3], _NPROBE)),
    ('t3.probe.d2', '11/36', '%d/%d' % (_dist[2], _NPROBE)),
    ('t3.probe.d1', '9/36', '%d/%d' % (_dist[1], _NPROBE)),
    ('t3.probe.degen', '20/36', '%d/%d' % (_dist[1] + _dist[2], _NPROBE)),

    # ---- T3 전제(동결 자료 재도출) — `NEXT.md` 가 쓰는 수
    ('t3.pre.propzero1', '17/36', frac(T3P['prop_lead0_vs_zero'])),
    ('t3.pre.propzero1.ratio', '1.0365', '%.4f' % T3P['prop_lead0_vs_zero']['ratio_median']),
    ('t3.GPC.prop', '17', [x['detail'] for x in T3['gates']
                           if x['name'].startswith('G-PC PROP-V400M3')][0]),
    ('t3.pre.ship_lt_prop', '33/36', frac(T3P['ship_lt_prop_lead0'])),
    ('t3.pre.ship_lt_prop.rel', '−25.75%', pct(T3P['ship_lt_prop_lead0']['rel_median_pct'], sign=False)),
    ('t3.pre.dev.bank_bit', '12/12', '%d/%d' % (T3P['device_sensitivity']['bank_bit_equal'],
                                                T3P['device_sensitivity']['n'])),
    ('t3.pre.dev.prop_bit', '0/12', '%d/%d' % (T3P['device_sensitivity']['prop_bit_equal'],
                                               T3P['device_sensitivity']['n'])),
    ('t3.pre.dev.prop_maxrel', '245.1', '%.1f' % T3P['device_sensitivity']['prop_max_abs_rel_pct']),
    ('t3.pre.cv.PROP', '13.57', '%.2f' % T3P['seed_cv']['PROP']['cv_median_pct']),
    ('t3.pre.cv.PROPmax', '67.3', '%.1f' % T3P['seed_cv']['PROP']['cv_max_pct']),
    ('t3.pre.cv.BANK', '1.04', '%.2f' % T3P['seed_cv']['BANK']['cv_median_pct']),
    ('t3.pre.m8l1', '5/12', '%d/%d' % (T3P['m8_l1_ensemble']['worse'], T3P['m8_l1_ensemble']['n'])),
    ('t3.pre.m8l1.rel', '0.00%', pct(T3P['m8_l1_ensemble']['rel_median_pct'], sign=False)),
    ('t3.pre.m8l6', '9/12', '%d/%d' % (T3P['m8_l6_varpro']['worse'], T3P['m8_l6_varpro']['n'])),
    ('t3.pre.cost.PROP', '9.97', '%.2f' % T3P['cost']['t2_with_update_loop']['PROP-FROZEN']['median_s']),
    ('t3.pre.cost.SHIP', '3.71', '%.2f' % T3P['cost']['t2_with_update_loop']['SHIP-FROZEN']['median_s']),

    # ---- T4-GEOM (판정 미정)
    ('t4.n', '48', str(T4P['thresholds']['n'])),
    ('t4.CONFIRM', '32', str(T4P['thresholds']['CONFIRM'])),
    ('t4.NOMINAL', '31', str(T4P['thresholds']['NOMINAL'])),
    ('t4.NOT_EARN', '17', str(T4P['thresholds']['NOT_EARN'])),
    ('t4.mde', '0.7109', '%.4f' % T4P['thresholds']['q80']),
    ('t4.primary_rung', '1.0', '%.1f' % T4P['ladder']['primary']),
    ('t4.go', 'LEADTIME', ','.join(_go)),
    ('t4.nogo_count', '9', str(len(_nogo))),
    ('t4.premise_runs', '0', str(T4P['runs'])),
    ('t4.premise_est_calls', '0', str(T4P['estimator_calls'])),
    ('t4.verdict_absent', 'YES', 'YES' if not _t4_verdicts else 'NO:' + ','.join(_t4_verdicts)),

    # ---- B-26 게이트 사후 반전
    ('b26.gates', '28/29', gates_of(B26)),
    ('b26.fail', 'G-0', [x[0] for x in B26['gates'] if not x[1]][0].split()[0]),
    ('b26.collision', '충돌 [1641]', [x[2] for x in B26['gates'] if not x[1]][0]),

    # ---- M-4 (오늘 EXPLORATORY 강등; 수치·게이트는 무수정)
    ('m4.gates', '51/51', gates_of(M4)),
    ('m4.primary', '0/12', frac(M4['primary'])),
    ('m4.primary.other_p', '0.000244', g(M4['primary']['p_other_tail'])),

    # ---- ★C-1 [BLOCKER] 성장 분기 발화율 (STATUS.md §5)
    ('c1.w3.learned', '8/87', _fp(W3L)[0]),
    ('c1.w3.learned.pct', '9.2', _fp(W3L)[1]),
    ('c1.w3.all', '17/96', _fp(W3A)[0]),
    ('c1.w5.ship_dup2', '11/64', _fp(W5S)[0]),
    ('c1.w5.ship_dup2.pct', '17.2', _fp(W5S)[1]),
    ('c1.w5.vp800', '47/192', _fp(W5V8)[0]),
    ('c1.w5.vp800.pct', '24.5', _fp(W5V8)[1]),
    ('c1.w5.vp_only', '141/192', _fp(W5VO)[0]),
    ('c1.w5.vp_only.pct', '73.4', _fp(W5VO)[1]),
    ('c1.bdeploy.a_ship', '31/173', _fp(BDS)[0]),
    ('c1.bdeploy.a_ship.pct', '17.9', _fp(BDS)[1]),
    ('c1.bdeploy.a_vp', '123/173', _fp(BDV)[0]),
    ('c1.bdeploy.a_vp.pct', '71.1', _fp(BDV)[1]),

    # ---- 학습 경로의 인용 가능한 합성 양성 (NEXT.md §1)
    ('b22.L2', '45/48', frac(B22['comparisons']['L2'])),
    ('b22.L2.rel', '−36.34%', pct(B22['comparisons']['L2']['rel_median_pct'], sign=False)),
    ('b22.L2.p', '6.56e-11', g(B22['comparisons']['L2']['p_one_sided'])),
    ('b16.primary', '46/48', frac(B16['primary'])),
    ('b16.primary.rel', '−38.61%', pct(B16['primary']['rel_median_pct'], sign=False)),
    ('b16.primary.p', '4.18e-12', g(B16['primary']['p_one_sided'])),
    ('b16.dose', '47/48', frac(B16['dose'])),
    ('b12.ship_vs_spec', '43/48', frac(B12['ship_vs_spec'])),
    ('b12.rel', '−40.19%', pct(B12['ship_vs_spec']['rel_median_pct'], sign=False)),
    ('b12.p', '6.84e-09', g(B12['ship_vs_spec']['p_one_sided'])),
    ('b1.crowded', '62/64', frac(B1['crowded'])),
    ('b1.rel', '−38.58%', pct(B1['crowded']['rel_median_pct'], sign=False)),
    ('w15.crowded', '55/64', '%d/%d' % (W15C[0], W15C[1])),
    ('w15.crowded.rel', '−44.84%', pct(W15C[2], sign=False)),
    ('w15.sparse', '28/64', '%d/%d' % (W15S[0], W15S[1])),
    ('w15.sparse.rel', '+21.42%', pct(W15S[2])),
]


def main():
    bad = []
    print('%-24s %-46s %s' % ('KEY', 'DOC', 'DERIVED'))
    print('-' * 100)
    for key, doc, derived in CLAIMS:
        derived = str(derived)
        ok = (doc == derived)
        if not ok:
            bad.append((key, doc, derived))
        print('%-24s %-46s %s  %s' % (key, doc, derived, 'OK' if ok else '<<< MISMATCH'))
    print('-' * 100)
    print('%d claims, %d mismatch' % (len(CLAIMS), len(bad)))
    if bad:
        raise AssertionError('문서 표기와 동결 산출물이 어긋난다: ' + repr(bad))
    print('STATUS_2026-08-30/status_numbers.py: PASS  (재도출 런 0회)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
