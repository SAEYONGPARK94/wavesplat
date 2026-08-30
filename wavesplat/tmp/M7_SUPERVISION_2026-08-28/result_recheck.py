#!/usr/bin/env python3
"""M-7 `RESULT.md` 재현자 — ★**런 0회 · 학습 0회 · GPU 0회.**

동결 jsonl **2,245행**(= 고유 2,244런 + 비트동일 중복 1행)에서 E1·E2·E3 의 수치를
**독립 코드로** 다시 만들고, ①`verdict_m7*.json` ②`RESULT.md` 축자 ③★중복 1행의
**판정 불변성** ④★논문 한계 절이 쓰는 **학습 시드 편차** 수치를 대조한다.
어긋나면 AssertionError.

⛔`gates_m7`/`analyze_m7`/`jobs_m7` 를 **import 하지 않는다** — 그러면 재현이 아니라
되풀이다(선례 `M1`·`M3` 의 `result_recheck.py`).
검정·집계 규약은 `PREREG.md §3·§4` 의 산문에서 **다시 구현**했다:
  · 레코드값 = 확산 시드의 **중앙값**(win == 3)
  · T-1 = median{LOSS−BASE, BOTH−DATA} · T-2 = median{DATA−BASE, BOTH−LOSS}
  · T-3 = (BOTH−DATA) − (LOSS−BASE) · 부호검정(T-3 만 양측) · Holm(가족 3, α=0.05)
  · 음성대조 = BASE-B − BASE (양측) · 중단 문턱 10
  · E2 = 에피소드별 추정시드 중앙값, `ZERO` 는 `SPEC25.zero_nRMSE` 열에서 합성(B-12 규약)
  · R1 바 = 이 배치에서 잰 `SHIPPED` 평균 + 0.010

사용: python3 tmp/M7_SUPERVISION_2026-08-28/result_recheck.py
"""
import glob
import json
import os
import statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))

RECORDS = ('AA14', 'AA15a', 'AA15b', 'BS11a', 'BS11b', 'BS11c', 'BS11d',
           'BS13a', 'BS13b', 'BS13c', 'LJ18a', 'YS17')
CELLS = ('BASE', 'LOSS', 'DATA', 'BOTH', 'BASE-B')
WIN = 3
FAMILY = ('T-1_loss', 'T-2_data', 'T-3_interaction')
ALPHA = 0.05
ABORT = 10          # PREREG §2.3 중단 문턱
R1_MARGIN = 0.010   # PREREG §3 E2 do-not-forget 여유
S1_NOMINAL, S1_CONFIRM, S1_NEG_MIN = 31, 34, 31   # B-12 상속 문턱


# ── 검정 ────────────────────────────────────────────────────────────────────
def p1(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def p2(k, n):
    return min(1.0, 2.0 * min(p1(k, n), p1(n - k, n)))


# ── 적재 ────────────────────────────────────────────────────────────────────
def load(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, pat))):
        for line in open(p):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


E1 = load('results_m7_e1_*.jsonl')
E2 = load('results_m7_e2_*.jsonl')
E3 = load('results_m7_e3_*.jsonl')
V = json.load(open(os.path.join(HERE, 'verdict_m7.json')))
V1 = json.load(open(os.path.join(HERE, 'verdict_m7_e1.json')))
V2 = json.load(open(os.path.join(HERE, 'verdict_m7_e2.json')))
V3 = json.load(open(os.path.join(HERE, 'verdict_m7_e3.json')))
TXT = open(os.path.join(HERE, 'RESULT.md'), encoding='utf-8').read()
bad = []

assert not [r for r in E1 + E2 + E3 if 'error' in r], '오류 행이 있다'
assert (len(E1), len(E2), len(E3)) == (229, 1008, 1008), (len(E1), len(E2), len(E3))
assert len(E1) + len(E2) + len(E3) == V['n_rows'] == 2245


# ── ★중복 1행: 존재하고, 비트 동일하고, 판정을 바꾸지 않는다 ────────────────
key = lambda r: (r['cell'], r['record'], r['diffusion_seed'])
seen, dups = {}, []
for r in E1:
    if key(r) in seen:
        dups.append((seen[key(r)], r))
    seen[key(r)] = r
assert len(dups) == 1, f'E1 중복 {len(dups)}쌍 (기대 1)'
a, b = dups[0]
assert key(a) == ('DATA', 'AA15a', 1), key(a)
for f in ('nowcast', 'leadmean', 'obs_hash', 'q_hash', 'final_n', 'n_atoms',
          'varpro_obj_before', 'varpro_obj_after'):
    assert a[f] == b[f], f'중복 행이 {f} 에서 다르다 — 재시도 기록이 아니다'
assert a['pid'] != b['pid'], '중복이 같은 프로세스에서 나왔다'
print(f'  ★E1 중복 1행 = {key(a)} · 비트 동일(nowcast {a["nowcast"]!r}) · pid {a["pid"]}≠{b["pid"]}')
print(f'  ⟹ 고유 런 {len(E1) - 1} + {len(E2)} + {len(E3)} = '
      f'{len(E1) - 1 + len(E2) + len(E3)} (원장의 "2,244런")')


def cellval(cell, rec, dedup=False):
    rs = [r for r in E1 if r['cell'] == cell and r['record'] == rec
          and r['win'] == WIN and r.get('nowcast') is not None]
    if dedup:                       # 확산 시드당 마지막 1행만
        rs = list({r['diffusion_seed']: r for r in rs}.values())
    return st.median([r['nowcast'] for r in rs]) if rs else None


def e1_stats(dedup=False):
    g = lambda c, r: cellval(c, r, dedup)
    per = {k: {} for k in FAMILY}
    for r in RECORDS:
        v = {c: g(c, r) for c in CELLS}
        per['T-1_loss'][r] = st.median([v['LOSS'] - v['BASE'], v['BOTH'] - v['DATA']])
        per['T-2_data'][r] = st.median([v['DATA'] - v['BASE'], v['BOTH'] - v['LOSS']])
        per['T-3_interaction'][r] = (v['BOTH'] - v['DATA']) - (v['LOSS'] - v['BASE'])
    res = {}
    for k, d in per.items():
        n = len(d)
        w = sum(1 for x in d.values() if x < 0)
        res[k] = dict(n=n, wins_negative=w,
                      p=(p2(w, n) if k == 'T-3_interaction' else p1(w, n)),
                      median_delta=st.median(d.values()), per_record=d)
    nc = {r: g('BASE-B', r) - g('BASE', r) for r in RECORDS}
    n = len(nc)
    w = sum(1 for x in nc.values() if x < 0)
    neg = dict(n=n, wins=w, max_side=max(w, n - w), p=p2(w, n),
               aborts_inference=max(w, n - w) >= ABORT, median_delta=st.median(nc.values()))
    return res, neg


res, neg = e1_stats(False)
res_d, neg_d = e1_stats(True)

# ★중복이 어느 수치도 움직이지 않는다(판정 불변)
for k in FAMILY:
    assert (res[k]['n'], res[k]['wins_negative'], res[k]['p']) == \
           (res_d[k]['n'], res_d[k]['wins_negative'], res_d[k]['p']), f'{k} 가 중복 제거로 바뀐다'
    assert abs(res[k]['median_delta'] - res_d[k]['median_delta']) < 1e-15
assert (neg['wins'], neg['p']) == (neg_d['wins'], neg_d['p']), '음성대조가 중복 제거로 바뀐다'
print('  ★중복 제거 재집계 == 원본 재집계 (세 대비 + 음성대조 전건 일치) ⟹ 판정 불변')

# verdict 대조
for k in FAMILY:
    v = V1['primary']['contrasts'][k]
    assert res[k]['n'] == v['n'], f'{k} n'
    assert res[k]['wins_negative'] == v['wins_negative'], f'{k} 승수'
    assert abs(res[k]['p'] - v['p']) < 1e-12, f'{k} p'
    assert abs(res[k]['median_delta'] - v['median_delta']) < 1e-12, f'{k} 중앙 Δ'
    for rec, x in res[k]['per_record'].items():
        assert abs(x - V1['primary']['per_record'][k][rec]) < 1e-12, f'{k}.{rec}'
    print(f'  {k:17s} {res[k]["wins_negative"]:2d}/{res[k]["n"]}  p={res[k]["p"]:.4f}')

vn = V1['primary']['neg_control']
assert (neg['n'], neg['wins'], neg['max_side']) == (vn['n'], vn['wins'], vn['max_side'])
assert abs(neg['p'] - vn['p']) < 1e-12 and neg['aborts_inference'] == vn['aborts_inference']
print(f'  음성대조 BASE-B<BASE {neg["wins"]}/{neg["n"]} (한쪽 최대 {neg["max_side"]}, '
      f'중단 문턱 {ABORT}) p={neg["p"]:.4f} ⟹ 중단 {"발동" if neg["aborts_inference"] else "미발동"}')

# Holm — 독립 재구현
order = sorted(FAMILY, key=lambda k: res[k]['p'])
holm, m = {}, len(FAMILY)
for i, k in enumerate(order):
    thr = ALPHA / (m - i)
    holm[k] = dict(threshold=thr, reject=res[k]['p'] <= thr)
    if not holm[k]['reject']:
        for k2 in order[i + 1:]:
            holm[k2] = dict(threshold=ALPHA / (m - order.index(k2)), reject=False)
        break
for k in FAMILY:
    vh = V1['primary']['holm'][k]
    assert holm[k]['reject'] == vh['reject'] and abs(holm[k]['threshold'] - vh['threshold']) < 1e-12
assert not any(holm[k]['reject'] for k in FAMILY), '기각된 대비가 있다 — 라벨이 다를 것'

# MDE
need = min(k for k in range(13) if p1(k, 12) <= ALPHA / 3)
vm = V1['primary']['mde']
assert (need, vm['min_wins']) == (11, 11) and abs(p1(need, 12) - vm['p_at_min']) < 1e-12
assert abs(p1(need - 1, 12) - vm['p_at_min_minus_1']) < 1e-12
print(f'  MDE: Holm 첫 문턱 {ALPHA/3:.6f} ⟹ 최소 {need}/12 (p={p1(need,12):.6f}); '
      f'10/12 는 p={p1(10,12):.6f} 로 실패')


# ── E2 ──────────────────────────────────────────────────────────────────────
def med_ep(rows, cell, field='avg_nRMSE'):
    per = {}
    for r in rows:
        if r.get('cell') == cell and r.get(field) is not None:
            per.setdefault(r['episode'], []).append(float(r[field]))
    return {k: st.median(v) for k, v in per.items()}


spec = med_ep(E2, 'SPEC25')
zero = med_ep(E2, 'SPEC25', 'zero_nRMSE')       # B-12 규약: ZERO 는 이 열에서 합성
n_zero_rows = sum(1 for r in E2 if r.get('cell') == 'SPEC25' and r.get('zero_nRMSE') is not None)
assert n_zero_rows == V2['s1']['n_zero_rows'] == 144, n_zero_rows
negc = sum(1 for k in spec if k in zero and spec[k] < zero[k])
assert (negc, len(spec)) == (V2['s1']['detail']['neg_control']['better'],
                             V2['s1']['detail']['neg_control']['n']) == (47, 48)
assert negc >= S1_NEG_MIN, '음성대조 미달 — S-1 이 무효 라벨이어야 한다'

ship = med_ep(E2, 'SHIPPED')
ship_mean = st.mean(ship.values())
bar = ship_mean + R1_MARGIN
assert abs(ship_mean - V2['s1']['r1_hard_bar']['shipped_mean']) < 1e-12
print(f'  E2 SHIPPED 평균 {ship_mean:.6f} ⟹ R1 바 {bar:.6f} (여유 +{R1_MARGIN})')
print(f'  E2 음성대조 SPEC25<ZERO {negc}/{len(spec)}')

for c in list(CELLS) + ['SHIPPED']:
    m = med_ep(E2, c)
    ks = sorted(set(m) & set(spec))
    w = sum(1 for k in ks if m[k] < spec[k])
    v = V2['s1']['detail']['per_cell'][c]
    assert (len(ks), w) == (v['n'], v['better']), f'E2 {c} 승수'
    assert abs(p1(w, len(ks)) - v['p']) < 1e-15, f'E2 {c} p'
    assert (w >= S1_CONFIRM) == v['confirmatory'] and (w >= S1_NOMINAL) == v['nominal']
    mean = st.mean(m.values())
    if c in CELLS:
        vb = V2['s1']['r1_hard_bar']['per_cell'][c]
        assert abs(mean - vb['mean']) < 1e-12 and (mean <= bar) == vb['passed'] is True
    print(f'    {c:8s} SHIP<SPEC25 {w}/{len(ks)}  p={p1(w,len(ks)):.2e}  '
          f'평균 {mean:.4f}  바통과 {mean <= bar}')

assert V2['s1']['detail']['threshold_parity_with_B12'] is True, 'B-12 문턱 패리티 실패'
assert V2['s1']['r1_hard_bar']['all_pass'] is True

# ★학습 시드만의 편차 — 논문 한계 절이 인용하는 수(단일출처는 이 스크립트다)
rel_seed = sorted(abs(cellval('BASE-B', r) - cellval('BASE', r)) / cellval('BASE', r) * 100
                  for r in RECORDS)
seed_med = st.median(rel_seed)
def contrast_rel(k):
    return st.median([abs(res[k]['per_record'][r]) / cellval('BASE', r) * 100 for r in RECORDS])
t1_rel, t2_rel = contrast_rel('T-1_loss'), contrast_rel('T-2_data')
ks = sorted(set(med_ep(E2, 'BASE')) & set(med_ep(E2, 'BASE-B')))
mb, mbb = med_ep(E2, 'BASE'), med_ep(E2, 'BASE-B')
seed_med_e2 = st.median([abs(mbb[k] - mb[k]) / mb[k] * 100 for k in ks])
print(f'  ★학습 시드 편차(E1, |상대| 중앙) {seed_med:.2f}%  vs  '
      f'T-1 {t1_rel:.2f}% · T-2 {t2_rel:.2f}%   (E2 {seed_med_e2:.2f}%)')
assert seed_med > t1_rel, '시드 편차가 T-1 보다 작다 — 한계 절 문장이 틀렸다'


# ── E3 (보고 전용) ──────────────────────────────────────────────────────────
for arm in ('LIN', 'HOS'):
    sub = [r for r in E3 if r.get('gt_arm') == arm]
    sp = med_ep(sub, 'SPEC25')
    assert abs(st.median(sp.values()) - V3['e3'][arm]['spec_median']) < 1e-12, f'E3 {arm} spec'
    line = []
    for c in list(CELLS) + ['SHIPPED']:
        m = med_ep(sub, c)
        ks = sorted(set(m) & set(sp))
        w = sum(1 for k in ks if m[k] < sp[k])
        v = V3['e3'][arm]['per_cell'][c]
        assert (len(ks), w) == (v['n'], v['better']), f'E3 {arm} {c}'
        assert abs(st.median(m.values()) - v['median']) < 1e-12, f'E3 {arm} {c} 중앙'
        line.append(f'{c} {w}/{len(ks)}')
    print(f'  E3 {arm}: ' + ' · '.join(line))


# ── 게이트 · 라벨 · 공시 ────────────────────────────────────────────────────
assert (V['gates_pass'], V['gates_total']) == (16, 16), '게이트 16/16 아님'
assert all(g['passed'] for g in V['gates']), 'FAIL 게이트가 있다'
assert V['prechecks_pass'] == 13 and all(p['passed'] for p in V['prechecks'])
assert V1['primary']['label'] == 'NEITHER-MOVES-REAL'
assert V2['s1']['label'] == 'DISTRIBUTION-SHIFT-SURVIVES'
assert V2['s1']['grade'] == 'PREREGISTERED'
assert V['grades']['T-2_data'] == 'EXPLORATORY' and V['grades']['T-1_loss'] == 'PREREGISTERED'
assert V1['primary']['data_axis_is_load_bearing'] is False
assert len(V['disclosures']) == 6, f'공시 {len(V["disclosures"])}건 (기대 6)'
assert V['excess_disclosure_ok'] is True and '9970ee91' in V['excess_disclosure_note']

# 동결 md5 — 검사기가 이미 대조하지만 여기서도 drift 0 을 요구한다
assert V['manifest']['frozen_drift'] == 0 and V['manifest']['frozen_n'] == 12
assert V['manifest']['m1_pair_key'] == ['hold', 'hold'] and V['manifest']['m1_verdict_frozen']


# ── RESULT.md 축자 대조 ─────────────────────────────────────────────────────
need_txt = [
    'NEITHER-MOVES-REAL', 'DISTRIBUTION-SHIFT-SURVIVES', 'EXPLORATORY',
    '16/16', '2,244', '2,245',
    f'{res["T-1_loss"]["wins_negative"]}/12', f'{res["T-2_data"]["wins_negative"]}/12',
    f'{res["T-3_interaction"]["wins_negative"]}/12',
    f'{neg["wins"]}/12', f'{negc}/48',
    f'{V2["s1"]["detail"]["per_cell"]["BASE"]["better"]}/48',
    f'{V2["s1"]["detail"]["per_cell"]["LOSS"]["better"]}/48',
    f'{V2["s1"]["detail"]["per_cell"]["DATA"]["better"]}/48',
    f'{V2["s1"]["detail"]["per_cell"]["BASE-B"]["better"]}/48',
    f'{ship_mean:.4f}', f'{bar:.4f}',
    f'{seed_med:.2f}', f'{t1_rel:.2f}', f'{t2_rel:.2f}',
    '0.387', '0.0730', '11/12', '재학습이 무익',
    'AMENDMENT_1', 'APPROVAL_E1_2026-08-29',
]
for s in need_txt:
    if s not in TXT:
        bad.append(f'RESULT.md 에 "{s}" 없음')
# 공시 6건이 전부 문서에 반영됐는지 — 각 공시의 식별 토큰
for tok in ('§8-1', '§8-2', '§8-3', '§8-10', 'G-6', 'BANK'):
    if tok not in TXT:
        bad.append(f'RESULT.md 에 공시 토큰 "{tok}" 없음')


# ── ★논문 축자 대조 — 학습 시드 편차의 **단일출처는 이 스크립트다** ────────────
# 사유: 이 네 수(9.14 / 6.73 / 9.13 / 10.21)는 `verdict_m7*.json` 에 없다 —
#   레코드별 **상대 크기**는 판정 대상이 아니었기 때문이다. 그래서 논문 감사
#   `audit_thesis_numbers.py` 가 그것을 verdict 에서 계산할 수 없다.
#   ⟹ 손복사를 막는 유일한 장치는 **여기서 동결 jsonl 로부터 다시 만들어 본문과 대조**하는 것이다.
TEX = '/home/vexit/wavesplat/wavesplat/thesis/thesis_main.tex'
if os.path.exists(TEX):
    tex = open(TEX, encoding='utf-8').read()
    for label, s in (('학습 시드 편차(E1 중앙)', f'{seed_med:.2f}'),
                     ('T-1 상대 크기', f'{t1_rel:.2f}'),
                     ('T-2 상대 크기', f'{t2_rel:.2f}'),
                     ('학습 시드 편차(E2 중앙)', f'{seed_med_e2:.2f}'),
                     ('학습 시드 편차(E1 최대)', f'{max(rel_seed):.2f}')):
        if s not in tex:
            bad.append(f'thesis_main.tex 에 {label} "{s}" 없음 (단일출처 = 이 스크립트)')
    # ★약화 문면이 되감기지 않았는지 — "닫혔다" 로 승격되면 여기서 깨진다
    if '단서가 \\textbf{약해졌다}' not in tex and '약해졌다' not in tex:
        bad.append('thesis_main.tex 에 M-7 약화 문면이 없다')
    print(f'  ★논문 대조: 시드 편차 {seed_med:.2f}% / {max(rel_seed):.2f}% · '
          f'T-1 {t1_rel:.2f}% · T-2 {t2_rel:.2f}% · E2 {seed_med_e2:.2f}% — 전건 본문에 존재')

assert not bad, '\n'.join(bad)
print('\n✅ M-7: 독립 재집계 == verdict == RESULT.md 축자 == 논문 축자 (불일치 0 · 런 0회)')
