#!/usr/bin/env python3
"""M-1 `RESULT.md` 재현자 — ★**런 0회.** 동결 jsonl 288행에서 판정 수치를 **독립 코드로**
다시 만들고, ①`verdict_m1.json` 과 ②`RESULT.md` 축자 문면과 대조한다. 어긋나면 AssertionError.

본보기 = `tmp/B2_REAL_2026-08-23/depth_recheck.py` · `tmp/B3_REFIT_2026-08-23/final6.py`.
⛔`gates_m1`/`analyze_m1` 을 import 하지 않는다 — 그러면 재현이 아니라 되풀이다.
사용: python3 tmp/M1_NONLIN_REAL_2026-08-28/result_recheck.py
"""
import glob, json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_m1_*.jsonl'))):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if 'error' not in r]


def med_by_ep(rows, arm, field='avg_nRMSE'):
    d = {}
    for r in rows:
        if r.get('arm') != arm:
            continue
        v = r.get(field)
        if v is not None:
            d.setdefault(r['episode'], []).append(float(v))
    return {k: st.median(v) for k, v in d.items()}


def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def paired(a, b):
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    rel = st.median([(a[k] - b[k]) / b[k] * 100 for k in ks])
    return dict(n=len(ks), better=w, rel=rel, p=sign_p(w, len(ks)))


rows = load()
assert len(rows) == 288, f'유효 행 {len(rows)} != 288'
V = json.load(open(os.path.join(HERE, 'verdict_m1.json')))
TXT = open(os.path.join(HERE, 'RESULT.md'), encoding='utf-8').read()
bad = []

for gt in ('LIN', 'HOS'):
    sub = [r for r in rows if r.get('gt_arm') == gt]
    ship = med_by_ep(sub, 'SHIP-NOW')
    spec = med_by_ep(sub, 'SPEC25')
    zero = med_by_ep(sub, 'SPEC25', 'zero_nRMSE')
    got = dict(ship_vs_spec=paired(ship, spec), spec_vs_zero=paired(spec, zero),
               ship_vs_zero=paired(ship, zero))
    for key, g in got.items():
        v = V['arms'][gt][key]
        assert (g['n'], g['better']) == (v['n'], v['better']), f'{gt}.{key} 승수 불일치'
        assert abs(g['rel'] - v['rel_median_pct']) < 1e-9, f'{gt}.{key} 크기 불일치'
        assert abs(g['p'] - v['p_one_sided']) < 1e-12, f'{gt}.{key} p 불일치'
    print(f'  {gt}: SHIP<SPEC {got["ship_vs_spec"]["better"]}/24 '
          f'{got["ship_vs_spec"]["rel"]:.2f}% p={got["ship_vs_spec"]["p"]:.2e} · '
          f'SPEC<ZERO {got["spec_vs_zero"]["better"]}/24 · '
          f'SHIP<ZERO {got["ship_vs_zero"]["better"]}/24 {got["ship_vs_zero"]["rel"]:.2f}%')
    # RESULT.md 축자 대조
    for s in (f'{got["ship_vs_spec"]["better"]}/24',
              f'{abs(got["ship_vs_spec"]["rel"]):.2f}%',
              f'{abs(got["ship_vs_zero"]["rel"]):.2f}%'):
        if s not in TXT:
            bad.append(f'{gt}: RESULT.md 에 "{s}" 없음')

# 짝 대비(탐색적)
sl = med_by_ep([r for r in rows if r.get('gt_arm') == 'LIN'], 'SHIP-NOW')
sh = med_by_ep([r for r in rows if r.get('gt_arm') == 'HOS'], 'SHIP-NOW')
rel, worse = [], 0
for k in sorted(sl):
    h = k.replace('LIN|', 'HOS|')
    if h in sh:
        rel.append((sh[h] - sl[k]) / sl[k] * 100)
        worse += sh[h] > sl[k]
pc = V['paired_contrast']
assert (worse, len(rel)) == (pc['hos_worse'], pc['n']), '짝 대비 승수 불일치'
assert abs(st.median(rel) - pc['rel_median_pct']) < 1e-9, '짝 대비 크기 불일치'
print(f'  짝(탐색적): HOS 나쁨 {worse}/{len(rel)} 상대중앙 {st.median(rel):+.2f}%')
for s in (f'{worse}/24', f'+{st.median(rel):.2f}%'):
    if s not in TXT:
        bad.append(f'짝 대비: RESULT.md 에 "{s}" 없음')

# 게이트·라벨
assert V['gates_pass'] == V['gates_total'] == 36, '게이트 36/36 아님'
assert V['pair_key'] == ['hold', 'hold'], f'짝 키 {V["pair_key"]}'
for gt in ('LIN', 'HOS'):
    assert V['arms'][gt]['label'] == 'HOLDS (CONFIRMATORY)', f'{gt} 라벨'
for s in ('36/36', 'HOLDS (CONFIRMATORY)', '(hold, hold)'):
    if s not in TXT:
        bad.append(f'RESULT.md 에 "{s}" 없음')

# ★동결 드리프트가 여전히 공시돼 있는가 (지우면 실패한다)
assert 'sequential_update.py' in TXT, 'RESULT.md 에 동결 드리프트 공시가 없다'

assert not bad, '\n'.join(bad)
print('\n✅ M-1: 독립 재집계 == verdict == RESULT.md 축자 (불일치 0)')
