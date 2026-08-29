#!/usr/bin/env python3
"""M-3 `RESULT.md` 재현자 — ★**런 0회.** 동결 jsonl 288행에서 셀별 수치를 **독립 코드로**
다시 만들고 ①`verdict_m3.json` ②`RESULT.md` 축자 ③★`G-17` 위반의 **실체**(SPEC25 의
`cap_depth` 되읽기)를 대조한다. 어긋나면 AssertionError.

⛔`gates_m3`/`analyze_m3` 을 import 하지 않는다 — 그러면 재현이 아니라 되풀이다.
사용: python3 tmp/M3_SUPPORT_2026-08-28/result_recheck.py
"""
import glob, json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_TRUE_DEPTH = {'DIR180': 50.0, 'SPR5': 50.0, 'DEP12': 12.0}


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_m3_*.jsonl'))):
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
    return dict(n=len(ks), better=w,
                rel=st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]), p=sign_p(w, len(ks)))


rows = load()
assert len(rows) == 288, f'유효 행 {len(rows)} != 288'
V = json.load(open(os.path.join(HERE, 'verdict_m3.json')))
TXT = open(os.path.join(HERE, 'RESULT.md'), encoding='utf-8').read()
bad = []

for cell in ('DIR180', 'SPR5', 'DEP12'):
    sub = [r for r in rows if r.get('cell') == cell]
    ship = med_by_ep(sub, 'SHIP-NOW')
    spec = med_by_ep(sub, 'SPEC25')
    zero = med_by_ep(sub, 'SPEC25', 'zero_nRMSE')
    got = dict(ship_vs_spec=paired(ship, spec), spec_vs_zero=paired(spec, zero),
               ship_vs_zero=paired(ship, zero))
    for key, g in got.items():
        v = V['cells'][cell][key]
        assert (g['n'], g['better']) == (v['n'], v['better']), f'{cell}.{key} 승수'
        assert abs(g['rel'] - v['rel_median_pct']) < 1e-9, f'{cell}.{key} 크기'
        assert abs(g['p'] - v['p_one_sided']) < 1e-12, f'{cell}.{key} p'
    caps = sorted({r.get('cap_depth') for r in sub if r.get('arm') == 'SPEC25'})
    print(f'  {cell}: SHIP<SPEC {got["ship_vs_spec"]["better"]}/16 '
          f'{got["ship_vs_spec"]["rel"]:.2f}% p={got["ship_vs_spec"]["p"]:.4f} · '
          f'SPEC<ZERO {got["spec_vs_zero"]["better"]}/16 · '
          f'SHIP<ZERO {got["ship_vs_zero"]["better"]}/16 {got["ship_vs_zero"]["rel"]:.2f}% · '
          f'SPEC25 cap_depth={caps} (참 {CELL_TRUE_DEPTH[cell]})')
    if cell != 'DEP12':                       # ⛔DEP12 의 수는 인용 금지 ⟹ 대조도 하지 않는다
        for s in (f'{got["ship_vs_spec"]["better"]}/16',
                  f'{abs(got["ship_vs_spec"]["rel"]):.2f}%',
                  f'{got["ship_vs_zero"]["better"]}/16',
                  f'{abs(got["ship_vs_zero"]["rel"]):.2f}%'):
            if s not in TXT:
                bad.append(f'{cell}: RESULT.md 에 "{s}" 없음')

# ★G-17 위반의 실체 — 되읽기로 확인한다(의도표 아님)
viol = [r for r in rows if r.get('arm') == 'SPEC25'
        and abs(float(r['cap_depth']) - CELL_TRUE_DEPTH[r['cell']]) > 1e-6]
assert len(viol) == 48, f'G-17 위반 행 {len(viol)} != 48'
assert {r['cell'] for r in viol} == {'DEP12'}, '위반이 DEP12 밖에도 있다'
print(f'\n  ★G-17 위반 = SPEC25 144행 중 {len(viol)}행, 전부 DEP12 '
      f'(cap_depth 50.0, 참 12.0) ⟹ 처치 미집행')

# 패키지 결함 자체 — 소스에서 (런 0회)
import re
dw = open('/home/vexit/wavesplat/wavesplat/wavesim/data_writer.py', encoding='utf-8').read()
dl = open('/home/vexit/wavesplat/wavesplat/wavesplat/data_loader.py', encoding='utf-8').read()
le = dw[dw.index('def load_episode'):]
le = le[:le.index('\ndef ', 1)] if '\ndef ' in le[1:] else le
assert "'depth'" not in le and '"depth"' not in le, 'load_episode 가 depth 를 읽는다 (수리됨?)'
assert re.search(r"self\.depth\s*=\s*float\(self\.data\.get\('depth',\s*50\.0\)\)", dl), \
    'data_loader.py 의 50.0 폴백을 못 찾았다 (수리됨?)'
print('  ★패키지 결함 재확인: load_episode 는 depth 를 읽지 않고, data_loader 는 50.0 으로 폴백한다')

# 게이트·라벨·처분
assert (V['gates_pass'], V['gates_total']) == (29, 30), '게이트 29/30 아님'
g17 = [g for g in V['gates'] if g[0].startswith('★G-17')]
assert len(g17) == 1 and g17[0][1] is False, 'FAIL 이 G-17 이 아니다'
assert V['cells']['DIR180']['label'] == 'BASELINE-NO-SKILL/SHIP-EARNS'
assert V['cells']['SPR5']['label'] == 'UNDECIDED'
for s in ('29/30', 'UNMEASURABLE', 'TREATMENT-NOT-DELIVERED', 'UNDECIDED',
          'BASELINE-NO-SKILL/SHIP-EARNS', '두 셀 미결 + 한 셀 집행 실패',
          'PREREG.md:105', '소급'):
    if s not in TXT:
        bad.append(f'RESULT.md 에 "{s}" 없음')

assert not bad, '\n'.join(bad)
print('\n✅ M-3: 독립 재집계 == verdict == RESULT.md 축자 (불일치 0)')
