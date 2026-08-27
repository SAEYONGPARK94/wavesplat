#!/usr/bin/env python3
"""F1 표 단일출처 — STATUS_2026-08-28 §2 의 표를 동결 자료에서 다시 만들고,
문서가 인용하는 수와 불일치하면 exit 1 로 죽는다 (08-24 교훈: 인용 수치를 낸
스크립트가 저장소에 없으면 안 된다; 본보기 = tmp/B3_REFIT_2026-08-23/final6.py).

두 겹으로 확인한다:
  (1) results_b26_*.jsonl 576행을 직접 재집계 (에피소드별 시드 median-of-3 → 코너 중앙)
  (2) verdict_b26.json 의 corners.*.medians / rel_median_pct 와 대조
그 위에 문서 §2 가 인용하는 파생값(코너 간 절대 % 차)을 재계산해 핀과 대조한다.
⛔이 스크립트는 판정 라벨·문턱을 계산하지 않는다 — 판정은 gates_b26.py 단일출처다.
"""
import json, glob, math, statistics as st, sys, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'B26_SEASTATE_2026-08-27')
CORNERS = ['SS-BF4', 'SS-LOWH', 'SS-SHORT', 'SS-REAL']

# ---- (1) jsonl 재집계 ----
rows = []
for f in sorted(glob.glob(os.path.join(D, 'results_b26_*.jsonl'))):
    for line in open(f):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
assert len(rows) == 576, f"행 수 {len(rows)} != 576"
ep = {}
for r in rows:
    ep.setdefault((r['corner'], r['episode']), {}).setdefault(r['arm'], []).append(
        (r['avg_nRMSE'], r.get('zero_nRMSE')))
agg = {}
for c in CORNERS:
    keys = sorted(k for k in ep if k[0] == c)
    assert len(keys) == 24, (c, len(keys))
    ship, spec, zero, rel_spec_zero, rel_ship_zero, w_ss, w_sz = [], [], [], [], [], 0, 0
    for k in keys:
        sh = st.median(v[0] for v in ep[k]['SHIP-NOW'])
        sp = st.median(v[0] for v in ep[k]['SPEC25'])
        z = st.median(v[1] for v in ep[k]['SPEC25'] if v[1] is not None)
        ship.append(sh); spec.append(sp); zero.append(z)
        rel_spec_zero.append(100*(sp/z-1)); rel_ship_zero.append(100*(sh/z-1))
        w_ss += sh < sp; w_sz += sh < z
    agg[c] = dict(ship=st.median(ship), spec=st.median(spec), zero=st.median(zero),
                  skill_spec=st.median(rel_spec_zero), skill_ship=st.median(rel_ship_zero),
                  wins_ship_spec=w_ss, wins_ship_zero=w_sz)

# ---- (2) verdict 대조 ----
V = json.load(open(os.path.join(D, 'verdict_b26.json')))
fail = []
for c in CORNERS:
    vc = V['corners'][c]
    for name, got, want, tol in [
            ('ship', agg[c]['ship'], vc['medians']['ship'], 1e-9),
            ('spec', agg[c]['spec'], vc['medians']['spec'], 1e-9),
            ('zero', agg[c]['zero'], vc['medians']['zero'], 1e-9),
            ('skill_spec', agg[c]['skill_spec'], vc['spec_vs_zero']['rel_median_pct'], 1e-6),
            ('skill_ship', agg[c]['skill_ship'], vc['ship_vs_zero']['rel_median_pct'], 1e-6)]:
        if not math.isclose(got, want, rel_tol=0, abs_tol=tol):
            fail.append(f"{c}.{name}: jsonl {got!r} != verdict {want!r}")
    if agg[c]['wins_ship_spec'] != vc['ship_vs_spec']['better']:
        fail.append(f"{c}.wins: {agg[c]['wins_ship_spec']} != {vc['ship_vs_spec']['better']}")

# ---- (3) 문서 §2 핀 (소수 자리 = 문서 표기 그대로) ----
b = agg['SS-BF4']
pin = {
    ('SS-BF4', 'ship'): '0.1176', ('SS-LOWH', 'ship'): '0.1731',
    ('SS-SHORT', 'ship'): '0.1370', ('SS-REAL', 'ship'): '0.1366',
    ('SS-BF4', 'skill_spec'): '-16.17', ('SS-LOWH', 'skill_spec'): '-17.02',
    ('SS-SHORT', 'skill_spec'): '-4.04', ('SS-REAL', 'skill_spec'): '-3.96',
    ('SS-LOWH', 'abs_vs_bf4'): '+47.2', ('SS-SHORT', 'abs_vs_bf4'): '+16.5',
    ('SS-REAL', 'abs_vs_bf4'): '+16.1',
    ('SS-BF4', 'skill_ship'): '-53.31', ('SS-REAL', 'skill_ship'): '-45.46',
}
def chk(key, fmt, val):
    if key in pin and fmt % val != pin[key]:
        fail.append(f"핀 불일치 {key}: {fmt % val} != {pin[key]}")
print(f"{'코너':9s} {'SHIP':>7s} {'SPEC25':>7s} {'ZERO':>7s} {'절대(vs BF4)':>12s} {'기준선기술':>9s} {'SHIP<SPEC':>9s} {'SHIP기술':>8s}")
for c in CORNERS:
    a = agg[c]
    absr = 100*(a['ship']/b['ship']-1)
    print(f"{c:9s} {a['ship']:7.4f} {a['spec']:7.4f} {a['zero']:7.4f} "
          f"{('기준' if c=='SS-BF4' else '%+11.1f%%' % absr):>12s} {a['skill_spec']:8.2f}% "
          f"{a['wins_ship_spec']:>6d}/24 {a['skill_ship']:7.2f}%")
    chk((c, 'ship'), '%.4f', a['ship']); chk((c, 'skill_spec'), '%.2f', a['skill_spec'])
    chk((c, 'skill_ship'), '%.2f', a['skill_ship'])
    if c != 'SS-BF4':
        chk((c, 'abs_vs_bf4'), '%+.1f', absr)
if fail:
    print('\nFAIL:'); [print(' ', f) for f in fail]; sys.exit(1)
print('\nOK — jsonl 재집계 == verdict_b26.json == STATUS_2026-08-28 §2 핀 (전건 일치)')
