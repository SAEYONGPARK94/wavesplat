"""A3 — 세 시드 축의 자(ruler) 를 M7 동결 자료에서 **다시 만든다**.

⛔런 0회.  동결된 `results_m7_e{1,2}_*.jsonl` 과 `results_b12.jsonl` 만 읽는다.
★손복사 0: 집계 함수는 배치 자신의 동결 모듈에서 **임포트**한다
  (`gates_m7._cell_value`·`neg_control`, `gates_b11b.med_by_episode`·`paired`).
★자기검증: B-12 의 공표 판정(43/48 · -40.194069998273584%)을 먼저 재현하고,
  M7 `SHIPPED` 칸이 그 판정과 **같은 수**를 내는지 대조한다.  어긋나면 AssertionError.
"""
import json, glob, os, sys, statistics as st

ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, os.path.join(ROOT, 'tmp/B11_HOS_BASELINE_2026-08-23'))
sys.path.insert(0, os.path.join(ROOT, 'tmp/M7_SUPERVISION_2026-08-28'))
from gates_b11b import med_by_episode, paired           # noqa: E402
import gates_m7 as G7                                   # noqa: E402

out = {}

# ── 0. B-12 재현 + M7 SHIPPED 대조 ───────────────────────────────────────────
b12 = [json.loads(l) for l in open(f'{ROOT}/tmp/B12_HOS_EXTEND_2026-08-23/results_b12.jsonl')]
b12 = [r for r in b12 if 'error' not in r]
p12 = paired(med_by_episode(b12, 'SHIP-NOW'), med_by_episode(b12, 'SPEC25'))
assert p12['better'] == 43 and abs(p12['rel_median_pct'] + 40.194069998273584) < 1e-9, p12
out['b12_reproduced'] = p12

e2 = []
for f in sorted(glob.glob(f'{ROOT}/tmp/M7_SUPERVISION_2026-08-28/results_m7_e2_*.jsonl')):
    e2 += [json.loads(l) for l in open(f) if l.strip()]
e2 = [r for r in e2 if 'error' not in r]


def med_cell(cell, field='avg_nRMSE'):
    d = {}
    for r in e2:
        if r.get('cell') != cell or r.get(field) is None:
            continue
        d.setdefault(r['episode'], []).append(r[field])
    return {e: st.median(v) for e, v in d.items()}


spec = med_cell('SPEC25')
zero = med_cell('SPEC25', field='zero_nRMSE')
cells = ['BASE', 'BASE-B', 'LOSS', 'DATA', 'BOTH', 'SHIPPED']
out['e2_per_cell'] = {}
med = {}
for c in cells:
    a = med_cell(c)
    med[c] = a
    out['e2_per_cell'][c] = dict(vs_spec=paired(a, spec), vs_zero=paired(a, zero),
                                 median_nrmse=st.median(a.values()),
                                 mean_nrmse=st.mean(a.values()))
sh = out['e2_per_cell']['SHIPPED']['vs_spec']
assert (sh['better'], round(sh['rel_median_pct'], 9)) == (43, round(p12['rel_median_pct'], 9)), sh
out['shipped_equals_b12'] = True

# ── 1. 학습 시드 축 (BASE vs BASE-B) — 합성 n=48 ─────────────────────────────
eps = sorted(set(med['BASE']) & set(med['BASE-B']))
rel = sorted((med['BASE-B'][e] - med['BASE'][e]) / med['BASE'][e] for e in eps)
out['train_seed_synth'] = dict(
    n=len(eps),
    b_better=sum(1 for e in eps if med['BASE-B'][e] < med['BASE'][e]),
    rel_median=st.median(rel), rel_mean=st.mean(rel),
    abs_rel_median=st.median([abs(x) for x in rel]),
    abs_rel_p90=sorted(abs(x) for x in rel)[int(0.9 * len(rel)) - 1],
    rel_min=rel[0], rel_max=rel[-1],
    d_wins_vs_spec=out['e2_per_cell']['BASE']['vs_spec']['better']
    - out['e2_per_cell']['BASE-B']['vs_spec']['better'],
    d_rel_median_pp=out['e2_per_cell']['BASE']['vs_spec']['rel_median_pct']
    - out['e2_per_cell']['BASE-B']['vs_spec']['rel_median_pct'],
    d_arm_median_rel=(st.median(med['BASE-B'].values()) - st.median(med['BASE'].values()))
    / st.median(med['BASE'].values()))

# ── 2. 추정 시드 축 (est_seed 42/43/44) — 같은 48 에피소드, 체크포인트 고정 ──
raw = {}
for r in e2:
    if r.get('est_seed') is None or r.get('avg_nRMSE') is None:
        continue
    raw.setdefault((r['cell'], r['episode']), {})[r['est_seed']] = r['avg_nRMSE']
out['est_seed_synth'] = {}
for c in cells:
    spread, pair = [], []
    for e in eps:
        v = raw[(c, e)]
        vals = [v[s] for s in sorted(v)]
        spread.append((max(vals) - min(vals)) / st.median(vals))
        pair.append(abs(v[43] - v[42]) / v[42])
    spread.sort(); pair.sort()
    out['est_seed_synth'][c] = dict(
        n=len(eps), range_over_median_median=st.median(spread),
        range_over_median_p90=spread[int(0.9 * len(spread)) - 1], range_over_median_max=spread[-1],
        pair42v43_abs_rel_median=st.median(pair), pair42v43_abs_rel_max=pair[-1])

# ── 3. 실측 n=12 — 학습 시드 축 vs 추정(확산) 시드 축, 같은 자 ──────────────
e1 = []
for f in sorted(glob.glob(f'{ROOT}/tmp/M7_SUPERVISION_2026-08-28/results_m7_e1_*.jsonl')):
    e1 += [json.loads(l) for l in open(f) if l.strip()]
nc = G7.neg_control(e1)
out['train_seed_real'] = {k: v for k, v in nc.items()}
rel_r, base_v = {}, {}
for rec in G7.REAL_RECORDS:
    a = G7._cell_value(e1, 'BASE', rec)
    b = G7._cell_value(e1, 'BASE-B', rec)
    base_v[rec] = a
    rel_r[rec] = (b - a) / a
v = sorted(rel_r.values())
out['train_seed_real']['rel'] = dict(median=st.median(v), abs_median=st.median([abs(x) for x in v]),
                                     min=v[0], max=v[-1])
# 확산 시드(추정 축) 폭 — 같은 체크포인트 안, 같은 레코드
out['est_seed_real'] = {}
for c in ['BASE', 'BASE-B', 'SHIPPED']:
    sp = []
    for rec in G7.REAL_RECORDS:
        vals = [r['nowcast'] for r in e1
                if r.get('cell') == c and r.get('record') == rec
                and r.get('win') == G7.REAL_WIN_PRIMARY and r.get('nowcast') is not None]
        if len(vals) >= 2:
            sp.append((max(vals) - min(vals)) / st.median(vals))
    sp.sort()
    out['est_seed_real'][c] = dict(n=len(sp), range_over_median_median=st.median(sp),
                                   range_over_median_max=sp[-1])

json.dump(out, open(f'{ROOT}/tmp/STATUS_2026-08-30/seed_axis_ruler.json', 'w'),
          indent=1, ensure_ascii=False, default=float)
print(json.dumps(out, indent=1, ensure_ascii=False, default=float))

# ── 4. 부록: 배포 후보 v1 vs v3 (레시피 짝) — §4 의 "같은 자릿수" 근거 ────────
#    ⛔코퍼스가 다르다(`data_confirm40` vs `B-12`) ⟹ 직접 비교 금지, 자릿수 비교만.
cv = json.load(open(f'{ROOT}/tmp/confirm40_2026-07-27/confirm_verdict.json'))
m1 = cv['candidates']['v1']['means']
m3 = cv['candidates']['v3']['means']
out['confirm40_v1_vs_v3'] = {s: dict(v1=m1[s], v3=m3[s],
                                     v1_rel_to_v3_pct=(m1[s] - m3[s]) / m3[s] * 100)
                             for s in ('all', 'sparse', '2comp', 'moderate', 'crowded')}
out['confirm40_crowded_improved'] = {k: cv['candidates'][k]['crowded_improved']
                                     for k in ('v1', 'v3')}
json.dump(out, open(f'{ROOT}/tmp/STATUS_2026-08-30/seed_axis_ruler.json', 'w'),
          indent=1, ensure_ascii=False, default=float)
print('\n[부록] confirm40 v1 vs v3 (층별 평균, v1 의 v3 대비 상대):')
for s, v in out['confirm40_v1_vs_v3'].items():
    print(f"  {s:9s} v1 {v['v1']:.4f}  v3 {v['v3']:.4f}   {v['v1_rel_to_v3_pct']:+.2f}%")
