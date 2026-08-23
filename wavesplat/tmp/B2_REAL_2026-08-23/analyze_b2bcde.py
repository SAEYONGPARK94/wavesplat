"""B-2b/c/d/e 동결 분석기 — ⚠**결과 후에 쓰였다**(2026-08-23, RV-5 지적).

**결함**: 이 네 배치는 `verdict_*.json` 이 없었다. 라벨이 **코드가 아니라 내 손**에서 나왔다
(오늘 철회 #5 와 **같은 부류**이고, 그 처방을 적어 놓고도 같은 날 네 번 더 반복했다).

**수리 원칙**: 판정 규칙은 **각 러너의 독스트링에 결과 전 기재된 것을 축자로 옮긴다.**
새로 고르지 않는다. 규칙 출처를 각 라벨 함수 위에 적었다.
⚠**두 기준으로 모두 낸다**: 종전 `ZERO_REF=1.0`(그때 선언된 것) 과
**참 제로필드**(`zero_ref.py`, 오늘 밤 확정). 후자가 정본이다.
"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b2 as G
from zero_ref import zero_ref_nowcast

TZ = {r: zero_ref_nowcast(r, G.WIN) for r in G.RECORDS}


def _count(v, ref):
    return sum(1 for k, x in v.items() if x < (1.0 if ref == 'old' else TZ[k]))


def _label(batch, w):
    """규칙은 각 러너 독스트링 축자(결과 전 기재)."""
    if batch == 'B-2b':   # run_b2b.py: 잡음 팔이 >=10/12 면 처방 성립
        return ('NOISE-MEDIATES' if w >= 10 else 'PARTIAL' if w >= 8 else 'NOISE-DOES-NOT-MEDIATE')
    if batch == 'B-2c':   # run_b2c.py: >=10 LAMBDA-MEDIATES / 7-9 PARTIAL / <=6 DOES-NOT
        return ('LAMBDA-MEDIATES' if w >= 10 else 'PARTIAL' if w >= 7 else 'LAMBDA-DOES-NOT-MEDIATE')
    if batch == 'B-2d':   # run_b2d.py: >=10 ORDER-MEDIATES / 8-9 PARTIAL / <=7 DOES-NOT
        return ('ORDER-MEDIATES' if w >= 10 else 'PARTIAL' if w >= 8 else 'ORDER-DOES-NOT-MEDIATE')
    # run_b2e.py: >=10 SCALE-INVARIANT-GROW-WORKS / 8-9 PARTIAL / <=7 NO-EFFECT
    return ('SCALE-INVARIANT-GROW-WORKS' if w >= 10 else 'PARTIAL' if w >= 8 else 'NO-EFFECT')


SPEC = {'B-2b': ('results_b2b.jsonl', lambda r: f"NOISE-{r['noise']}", 'NOISE-0.005'),
        'B-2c': ('results_b2c.jsonl', lambda r: r['cell'], 'LAM-SCALE'),
        'B-2d': ('results_b2d.jsonl', lambda r: r['cell'], 'MAXN-10'),
        'B-2e': ('results_b2e.jsonl', lambda r: r['cell'], 'GROW-SCALED')}

for batch, (f, keyf, test_arm) in SPEC.items():
    rows = [r for r in (json.loads(l) for l in open(os.path.join(HERE, f))) if 'error' not in r]
    arms = sorted({keyf(r) for r in rows})
    summ = {}
    for a in arms:
        v = {r['record']: r['nowcast'] for r in rows if keyf(r) == a}
        summ[a] = {'n': len(v), 'beats_zero_old': _count(v, 'old'),
                   'beats_zero_true': _count(v, 'true'), 'median': st.median(v.values()),
                   'p_old': G.sign_test_one_sided(_count(v, 'old'), len(v)),
                   'p_true': G.sign_test_one_sided(_count(v, 'true'), len(v))}
    w_old, w_true = summ[test_arm]['beats_zero_old'], summ[test_arm]['beats_zero_true']
    out = {'batch': batch, 'test_arm': test_arm, 'arms': summ,
           'label': _label(batch, w_old), 'label_true_ref': _label(batch, w_true),
           'rule_source': f'{f.replace("results", "run").replace(".jsonl", ".py")} 독스트링(결과 전 기재)',
           '_amended': '2026-08-23: verdict 부재 결함 수리. 라벨은 러너 독스트링 규칙을 축자 적용.'}
    open(os.path.join(HERE, f'verdict_{batch.replace("B-2", "b2")}.json'), 'w').write(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{batch}: {test_arm}  종전기준 {w_old}/12 -> {out['label']}  |  "
          f"★참기준 {w_true}/12 -> {out['label_true_ref']}")
