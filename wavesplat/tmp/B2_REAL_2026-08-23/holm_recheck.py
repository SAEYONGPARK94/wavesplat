"""★`MULTIPLICITY.md` 재검 — **수심 정정(17→30) 뒤 짝-가족 Holm 에서 무엇이 살아남는가.**

⛔**이것은 분석기가 아니다**(`analyze_*` 이름을 쓰지 않는다). **동결 판정을 바꾸지 않는다.**
런 0회 — 동결 결과 파일만 재집계한다.

## 왜 필요한가 (2026-08-26)
`MULTIPLICITY.md:13,34` 는 짝-가족 7건 Holm 에서 **`noise 0.0 vs 0.02`(11/12, p=0.0032) 하나만
생존**한다고 적었고, `:34` 가 그것을 *"살아남는 유일한 잡음 진술"* 이라 불렀다.
그런데 그 팔은 **BS 7 레코드의 `depth` 가 17.0 m(=AA 값 복사 오기)** 인 구성에서 나왔고
정정 수심(30.0 m)에서는 **8/12, p=0.1938 = 널**이다(`depth_recheck.py §4`).
⟹ ★**그 가족의 생존자가 0개가 되는지**를 손이 아니라 코드로 확인한다.

## 이 스크립트가 지키는 규율
1. **먼저 옛 표를 재현한다**(`EXPECT_D17`). 재현하지 못하면 `AssertionError` 로 죽는다 —
   즉 *"내가 새 수를 만든 것이 아니라 같은 자를 다시 댔다"* 가 기계로 증명된다.
2. **정정 대조군이 있는 팔만 교체한다.** 나머지 5팔은 `depth=17` 그대로 두고 그 사실을
   행마다 표시한다(⛔없는 자료를 만들지 않는다). 그 5팔은 **정정 전에도 전부 널**(p≥0.387)이다.
3. Holm 은 `MULTIPLICITY.md` 자신이 쓴 **가족 크기 m=7** 을 그대로 쓴다(문턱을 바꾸지 않는다).

## 부록 — `RESULT.md:195` 의 "비단조" 재집계
같은 자료에서 잡음 3수준의 **제로필드(ZERO_REF=1.0) 이김 수**와 짝 `0.005 vs 0.0` 를
두 수심에서 나란히 낸다.
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import gates_b2 as G                                            # noqa: E402

ALPHA = 0.05
M_FAMILY = 7        # ★`MULTIPLICITY.md §1` 표가 결과 전이 아니라 그 문서에서 쓴 가족 크기.


def load(*pats):
    out = []
    for p in pats:
        for f in sorted(glob.glob(os.path.join(ROOT, 'tmp', p))):
            out += [json.loads(l) for l in open(f) if l.strip()]
    return [r for r in out if 'error' not in r]


def by(rows, **kw):
    return {r['record']: r['nowcast']
            for r in rows if all(r.get(k) == v for k, v in kw.items())}


B2B = load('B2_REAL_2026-08-23/results_b2b.jsonl')
B2C = load('B2_REAL_2026-08-23/results_b2c.jsonl')
B2D = load('B2_REAL_2026-08-23/results_b2d.jsonl')
B2E = load('B2_REAL_2026-08-23/results_b2e.jsonl')
B14B = load('B14_OBSMODEL_2026-08-24/results_b14b_*.jsonl')

NOISE = {(nz, 17): by(B2B, noise=nz) for nz in (0.02, 0.005, 0.0)}
NOISE.update({(nz, 30): by(B14B, obs_model='IID', noise=nz) for nz in (0.02, 0.005, 0.0)})


def paired(a, b):
    ks = G.RECORDS
    w = sum(1 for k in ks if a[k] < b[k])
    med = st.median([(a[k] - b[k]) / b[k] * 100 for k in ks])
    return w, len(ks), med, G.sign_test_one_sided(w, len(ks))


# ── 짝-가족 7건. `depth30` = 정정 대조군이 존재하면 그 쌍, 없으면 None.
FAMILY = [
    ('noise 0.0 vs 0.02',      (NOISE[(0.0, 17)],   NOISE[(0.02, 17)]),
     (NOISE[(0.0, 30)], NOISE[(0.02, 30)])),
    ('noise 0.005 vs 0.02',    (NOISE[(0.005, 17)], NOISE[(0.02, 17)]),
     (NOISE[(0.005, 30)], NOISE[(0.02, 30)])),
    ('MAXN-10 vs MAXN-30',     (by(B2D, cell='MAXN-10'), by(B2D, cell='MAXN-30')), None),
    ('LAM-SCALE vs LAM-BASE',  (by(B2C, cell='LAM-SCALE'), by(B2C, cell='LAM-BASE')), None),
    ('LAM-ZERO vs LAM-BASE',   (by(B2C, cell='LAM-ZERO'), by(B2C, cell='LAM-BASE')), None),
    ('MAXN-4 vs MAXN-30',      (by(B2D, cell='MAXN-4'), by(B2D, cell='MAXN-30')), None),
    ('GROW-SCALED vs GROW-ABS', (by(B2E, cell='GROW-SCALED'), by(B2E, cell='GROW-ABS')), None),
]

# ★`MULTIPLICITY.md §1` 표 축자 (개선 수, 상대 중앙 %, p) — 재현하지 못하면 죽는다.
EXPECT_D17 = {
    'noise 0.0 vs 0.02':       (11, -23.40, 0.0032),
    'noise 0.005 vs 0.02':     (10, -15.97, 0.0193),
    'MAXN-10 vs MAXN-30':      (7,   -1.53, 0.3872),
    'LAM-SCALE vs LAM-BASE':   (6,   -0.07, 0.6128),
    'LAM-ZERO vs LAM-BASE':    (6,   -0.22, 0.6128),
    'MAXN-4 vs MAXN-30':       (5,   +4.40, 0.8062),
    'GROW-SCALED vs GROW-ABS': (3,   +0.00, 0.9807),
}
# ★부록: 잡음 3수준의 제로필드(1.0) 이김 수 + 짝 `0.005 vs 0.0` (`RESULT.md:195` 단일출처)
EXPECT_MONO = {17: (7, 10, 8, 6, 0.6128), 30: (9, 8, 7, 3, 0.9807)}


def holm(rows, m=M_FAMILY):
    """Holm–Bonferroni. rows = [(name, p, note)]. 반환 = 생존 이름 목록 + 인쇄용 표."""
    order = sorted(rows, key=lambda r: r[1])
    alive, stopped, table = [], False, []
    for i, (name, p, note) in enumerate(order):
        thr = ALPHA / (m - i)
        ok = (not stopped) and p <= thr
        if not ok:
            stopped = True
        else:
            alive.append(name)
        table.append((name, p, thr, ok, note))
    return alive, table


def main():
    print('=' * 84)
    print('MULTIPLICITY.md 짝-가족 재검 (depth 17 → 30) — 런 0회, 동결 결과만 재집계')
    print('=' * 84)

    print('\n## 1. 재현 검사 — `MULTIPLICITY.md §1` 표(수심 17)를 그대로 다시 만든다')
    fails, rows17, rows30 = [], [], []
    for name, (a17, b17), corr in FAMILY:
        w, n, med, p = paired(a17, b17)
        ew, emed, ep = EXPECT_D17[name]
        if w != ew or round(med, 2) != round(emed, 2) or round(p, 4) != round(ep, 4):
            fails.append(f'{name}: {w}/{n} {med:+.2f}% p={p:.4f} != 문서 {ew} {emed:+.2f}% {ep}')
        print(f'   {name:26s} {w:2d}/{n} {med:+7.2f}% p={p:.4f}   ✅문서와 일치')
        rows17.append((name, p, ''))
        if corr is None:
            rows30.append((name, p, '⚠수심 17 그대로 (정정 대조군 없음)'))
        else:
            w2, n2, med2, p2 = paired(*corr)
            rows30.append((name, p2, f'★정정 {w2}/{n2} {med2:+.2f}% (수심 17 에서는 {w}/{n})'))
    assert not fails, f'⛔MULTIPLICITY.md §1 표를 재현하지 못했다: {fails}'

    for tag, rows in (('수심 17 (기록 = MULTIPLICITY.md 가 쓴 것)', rows17),
                      ('수심 30 (정정 대조군으로 교체 가능한 팔만 교체)', rows30)):
        alive, table = holm(rows)
        print(f'\n## 2. Holm–Bonferroni (α={ALPHA}, m={M_FAMILY}) — {tag}')
        for name, p, thr, ok, note in table:
            print(f'   {name:26s} p={p:.4f}  임계={thr:.5f}  '
                  f'{"★생존" if ok else "탈락":6s} {note}')
        print(f'   ⟹ ★**생존 {len(alive)}개**' + (f': {alive}' if alive else ' = 없음'))

    print('\n## 3. `RESULT.md:195` 「비단조」 재집계 — 제로필드(ZERO_REF=1.0) 이김 수')
    print(f"   {'수심':6s} {'σ=0.02':>8s} {'σ=0.005':>9s} {'σ=0.0':>8s}   짝 `0.005 vs 0.0`")
    mono = []
    for dep in (17, 30):
        c = [sum(1 for k in G.RECORDS if NOISE[(nz, dep)][k] < G.ZERO_REF)
             for nz in (0.02, 0.005, 0.0)]
        w, n, med, p = paired(NOISE[(0.005, dep)], NOISE[(0.0, dep)])
        shape = ('봉우리(비단조)' if c[1] > c[0] and c[1] > c[2]
                 else '단조 감소' if c[0] > c[1] > c[2]
                 else '단조 증가' if c[0] < c[1] < c[2] else '기타')
        print(f'   d={dep:<4d} {c[0]:6d}/12 {c[1]:7d}/12 {c[2]:6d}/12   '
              f'{w}/12 상대중앙 {med:+.2f}% p={p:.4f}   ⟹ {shape}')
        mono.append(tuple(c) + (w, round(p, 4)))
    for dep, got in zip((17, 30), mono):
        exp = EXPECT_MONO[dep]
        assert got == exp, f'⛔d={dep} 비단조 표 불일치: {got} != {exp}'
    print('   ⛔σ 를 낮출수록 좋아지는 관계는 **어느 수심에서도** 확립되지 않는다.')
    print('   ⛔그리고 수준별 순서가 두 수심에서 **서로 반대**다 ⟹ *"모양이 유지된다"* 는 거짓.')

    print('\n' + '=' * 84)
    print('✅ 재현 검사 통과 · EXPECT 전건 일치')
    print('=' * 84)


if __name__ == '__main__':
    main()
