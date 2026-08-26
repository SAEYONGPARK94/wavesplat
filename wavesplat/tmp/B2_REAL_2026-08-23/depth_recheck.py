"""★수심 드리프트 재검 — **B-2 의 어느 판정이 `depth 17→30` 정정을 견디는가.**

⛔이것은 분석기가 아니다(`analyze_*` 이름을 쓰지 않는다). 동결 판정을 **바꾸지 않는다.**
하는 일은 하나다: *"기록(수심 17)과 정정된 코드(수심 30)에서 같은 양을 나란히 놓고,
무엇이 바뀌고 무엇이 안 바뀌는지"* 를 **동결 결과 파일에서만** 재도출한다. 런 0회.

## 왜 필요한가 (2026-08-26)
`FROZEN_DRIFT.md` 초판은 이 드리프트를 *"측정된 효과 Δω 0.0000–0.0012% ⟹ 수치적으로 무해"*,
*"두 변경 모두 기본 경로 비트동일"* 이라 적었다. ⛔**둘 다 틀렸다.**
`depth` 는 `scene_station()` 의 관측·질의·채점(`P`·`z`·`maps_gt`·`sg`)에 **전혀 들어가지 않고**
반환 dict 의 `depth` 로만 나가 **추정기의 분산관계**로 들어간다. 따라서
"장면은 비트동일, 추정만 달라진다" 가 맞고, 그 추정 차이가 `PROP` 팔에서 **−70% ~ +31%** 다.
⟹ *"Δω 가 작다"* 는 *"결과가 같다"* 를 함의하지 않는다. 이 파일이 그 사실을 못 박는다.

## 수심 30 대조군은 **이미 존재한다** (재실행 불필요)
| 팔 | 수심 17 (기록) | 수심 30 (정정 후) |
|---|---|---|
| `SHIP`(배포 bank_mmse) | `B2/results.jsonl` cell=SHIP `nowcast` | `B3/results_b3.jsonl` cell=FROZEN win=3 `lead0` |
| `PROP`(제안 표현 강제) | `B2/results.jsonl` cell=PROP `nowcast` | `B14/results_b14b_*.jsonl` obs_model=IID |
★**대조가 정당하다는 증거 = 수심 정정이 닿지 않는 5 레코드(AA×3·LJ·YS)에서 비트 동일**
(`assert_isolation()`). 그 5개가 비트 동일하고 BS 7개만 다르다 ⟹ **유일한 차이가 수심이다.**
"""
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import gates_b2 as G                                            # noqa: E402

UNAFFECTED = [r for r in G.RECORDS if not r.startswith('BS')]   # AA×3 · LJ · YS
AFFECTED = [r for r in G.RECORDS if r.startswith('BS')]         # = G.SMALL_BOX
assert AFFECTED == G.SMALL_BOX, '★소형 층 = 수심이 정정된 레코드 집합, 이 항등식이 핵심이다'


def load(*rel):
    out = []
    for r in rel:
        p = os.path.join(ROOT, 'tmp', r)
        out += [json.loads(l) for l in open(p) if l.strip()]
    return [r for r in out if 'error' not in r]


def fisher_one_sided(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    return sum(comb(c1, k) * comb(n - c1, r1 - k)
               for k in range(a, min(r1, c1) + 1)) / comb(n, r1)


B2 = load('B2_REAL_2026-08-23/results.jsonl')
B2B = load('B2_REAL_2026-08-23/results_b2b.jsonl')
B3 = load('B3_REFIT_2026-08-23/results_b3.jsonl')
B14B = load('B14_OBSMODEL_2026-08-24/results_b14b_0.jsonl',
            'B14_OBSMODEL_2026-08-24/results_b14b_1.jsonl')
TZ = json.load(open(os.path.join(HERE, 'true_zero_field.json')))

ARMS = {
    ('SHIP', 17): {r['record']: r['nowcast'] for r in B2 if r['cell'] == 'SHIP'},
    ('PROP', 17): {r['record']: r['nowcast'] for r in B2 if r['cell'] == 'PROP'},
    ('SHIP', 30): {r['record']: r['lead0'] for r in B3
                   if r['cell'] == 'FROZEN' and r['win'] == G.WIN},
    ('PROP', 30): {r['record']: r['nowcast'] for r in B14B
                   if r['obs_model'] == 'IID' and r['noise'] == 0.02},
}
NOISE = {(nz, 17): {r['record']: r['nowcast'] for r in B2B if r['noise'] == nz}
         for nz in (0.02, 0.005, 0.0)}
NOISE.update({(nz, 30): {r['record']: r['nowcast'] for r in B14B
                         if r['obs_model'] == 'IID' and r['noise'] == nz}
              for nz in (0.02, 0.005, 0.0)})


def assert_isolation():
    """★수심이 **유일한** 차이임을 비트로 증명한다. 실패하면 아래 표를 읽지 마라."""
    bad = []
    for arm in ('SHIP', 'PROP'):
        a, b = ARMS[(arm, 17)], ARMS[(arm, 30)]
        for k in UNAFFECTED:
            if a.get(k) != b.get(k):
                bad.append(f'{arm}/{k}')
        if not any(a.get(k) != b.get(k) for k in AFFECTED):
            bad.append(f'{arm}/BS-전건동일(개입이 발화하지 않았다)')
    for nz in (0.02, 0.005, 0.0):
        a, b = NOISE[(nz, 17)], NOISE[(nz, 30)]
        for k in UNAFFECTED:
            if a.get(k) != b.get(k):
                bad.append(f'noise{nz}/{k}')
    assert not bad, f'⛔격리 실패 — 수심 말고 다른 것이 움직였다: {bad}'
    return (f'✅격리 확인: 수심 무관 5 레코드가 두 팔 × 세 잡음수준에서 **전건 비트 동일**, '
            f'BS 7 레코드만 다르다 ⟹ 유일한 차이 = depth 17→30')


def counts(v, ref):
    f = (lambda k: 1.0) if ref == 'const' else (lambda k: TZ[k])
    c = lambda sub: (sum(1 for k in sub if v[k] < f(k)), len(sub))
    a, lg, sm = c(G.RECORDS), c(G.LARGE_BOX), c(G.SMALL_BOX)
    return dict(all_w=a[0], p=G.sign_test_one_sided(*a),
                large=lg[0], small=sm[0],
                fisher=fisher_one_sided(lg[0], lg[1] - lg[0], sm[0], sm[1] - sm[0]),
                label=G.label({'beats_zero': a[0], 'n': a[1],
                               'p_one_sided': G.sign_test_one_sided(*a)}, None, None)[0])


def paired(a, b, sub=None):
    ks = sub or G.RECORDS
    w = sum(1 for k in ks if a[k] < b[k])
    return w, len(ks), st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]), \
        G.sign_test_one_sided(w, len(ks))


# ── ★결과 전이 아니라 **결과 후** 고정한 기대값 — RESULT.md 의 「부록 D」 표와 같은 수다.
#    이 파일이 그 표의 단일출처이고, 어긋나면 실패한다(문서 드리프트 방지).
EXPECT = {
    ('PROP', 'const'): dict(all_w={17: 7, 30: 9}, small={17: 2, 30: 4}, large={17: 5, 30: 5}),
    ('PROP', 'true'): dict(all_w={17: 5, 30: 7}, small={17: 1, 30: 3}, large={17: 4, 30: 4}),
    ('SHIP', 'const'): dict(all_w={17: 12, 30: 12}, small={17: 7, 30: 7}, large={17: 5, 30: 5}),
    ('SHIP', 'true'): dict(all_w={17: 10, 30: 10}, small={17: 6, 30: 6}, large={17: 4, 30: 4}),
}


def main():
    print('=' * 78)
    print('B-2 수심 드리프트 재검 (depth 17 → 30) — 런 0회, 동결 결과만 재집계')
    print('=' * 78)
    print('\n' + assert_isolation())

    print('\n## 1. BS 7 레코드에서 정정이 실제로 한 일 (상대 %)')
    print(f"   {'레코드':8s} {'PROP d17':>9s} {'PROP d30':>9s} {'Δ%':>8s}   "
          f"{'SHIP d17':>9s} {'SHIP d30':>9s} {'Δ%':>7s}")
    dp, ds = [], []
    for k in AFFECTED:
        p17, p30 = ARMS[('PROP', 17)][k], ARMS[('PROP', 30)][k]
        s17, s30 = ARMS[('SHIP', 17)][k], ARMS[('SHIP', 30)][k]
        dp.append((p30 - p17) / p17 * 100)
        ds.append((s30 - s17) / s17 * 100)
        print(f'   {k:8s} {p17:9.4f} {p30:9.4f} {dp[-1]:+7.2f}%   '
              f'{s17:9.4f} {s30:9.4f} {ds[-1]:+6.2f}%')
    print(f'   ⟹ PROP 변동폭 [{min(dp):+.1f}%, {max(dp):+.1f}%]  vs  '
          f'SHIP [{min(ds):+.1f}%, {max(ds):+.1f}%]')
    print('   ★배포 팔(SHIP)은 수심에 사실상 무감, 제안 표현 팔(PROP)만 크게 흔들린다.')

    print('\n## 2. 판정 — 무엇이 견디고 무엇이 무너지는가')
    fails = []
    for ref in ('const', 'true'):
        nm = 'ZERO_REF=1.0(원 사전등록)' if ref == 'const' else '참 제로필드(AMENDMENT)'
        print(f'\n   ── 채점 기준 = {nm} ──')
        for arm in ('SHIP', 'PROP'):
            row = {}
            for dep in (17, 30):
                c = counts(ARMS[(arm, dep)], ref)
                row[dep] = c
                print(f'   {arm:5s} d={dep:<3d} 전체 {c["all_w"]:2d}/12 p={c["p"]:.4f} | '
                      f'대형 {c["large"]}/5 · 소형 {c["small"]}/7 | '
                      f'층 Fisher p={c["fisher"]:.4f} | {c["label"]}')
                e = EXPECT[(arm, ref)]
                for key in ('all_w', 'large', 'small'):
                    if c[key] != e[key][dep]:
                        fails.append(f'{arm}/{ref}/d{dep}/{key}: {c[key]} != {e[key][dep]}')
            if row[17]['label'] != row[30]['label']:
                print(f'   ⛔{arm}: **라벨이 바뀐다** {row[17]["label"]} → {row[30]["label"]}')
            else:
                print(f'   ✅{arm}: 라벨 불변 = {row[17]["label"]}')

    print('\n## 3. 짝 비교 PROP vs SHIP (기준값 소거, 같은 수심끼리)')
    for dep in (17, 30):
        w, n, med, p = paired(ARMS[('PROP', dep)], ARMS[('SHIP', dep)])
        wl = paired(ARMS[('PROP', dep)], ARMS[('SHIP', dep)], G.LARGE_BOX)[0]
        ws = paired(ARMS[('PROP', dep)], ARMS[('SHIP', dep)], G.SMALL_BOX)[0]
        print(f'   d={dep:<3d} PROP 승 {w}/{n} (SHIP 승 {n - w}/{n}) '
              f'상대중앙 {med:+.1f}% p={p:.4f} | 대형 PROP {wl}/5 · 소형 PROP {ws}/7')
    print('   ✅부호(=SHIP 이 낫다)는 견딘다. ⛔크기(+52.2%)는 수심 17 한정이다.')

    print('\n## 4. B-2b 잡음 개입도 수심에 의존하는가 (짝, σ=0.02 대비)')
    for nz in (0.005, 0.0):
        for dep in (17, 30):
            w, n, med, p = paired(NOISE[(nz, dep)], NOISE[(0.02, dep)])
            print(f'   σ 0.02→{nz:<6} d={dep:<3d} 개선 {w}/{n} 상대중앙 {med:+.1f}% p={p:.4f}')
    print('   ⛔`B-2b` 의 짝 개입 유의성은 수심 30 에서 살아남지 못한다.')

    assert not fails, f'⛔RESULT.md 부록 D 와 불일치: {fails}'
    print('\n' + '=' * 78)
    print('✅ EXPECT 전건 일치 — RESULT.md 「부록 D」 표가 이 스크립트로 재현된다.')
    print('=' * 78)


if __name__ == '__main__':
    main()
