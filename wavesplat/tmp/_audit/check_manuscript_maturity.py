"""★원고에 들어간 배치가 **성숙도 규칙**을 만족하는가 (박사 기여 감사 C-10).

⛔왜 (2026-09-03): `R5` 가 **결과가 나온 그날** 두 원고에 실렸고 **같은 날 철회**됐다.
그 배치는 셋을 어겼다:
  ① `verdict_r5.json` 이 자기 `FROZEN_md5.txt` 목록에 **없다**
  ② `grade = EXPLORATORY` 인데 그 등급이 **인쇄면에 병기되지 않았다**
  ③ 적대적 감사를 **거치기 전에** 원고에 들어갔다
⟹ 규칙(⚠**초판을 고쳤다**): 원고가 인용하는 배치는
  **(a) 검토 산출물**(`RESULT.md` / `AMENDMENT*` / `AUDIT*`)을 하나 이상 갖고,
  **(b) `verdict` 에 `grade` 가 있고**, 확증이 아니면 그 등급이 **배치 문서에 적혀** 있다.

⛔**초판은 불가능한 것을 요구했다** --- *"판정물이 자기 `FROZEN_md5` 목록에 있어야 한다"*.
   `verdict` 는 동결 **뒤에** 생기므로 사전 동결 목록에 들어갈 수 없다.
   ★오늘 세 번째로 나온 「검사기가 거짓/불가능을 강제한다」 부류다 ⟹ 규칙을 **참인 것**으로 바꿨다.
⚠(a) 는 *"하루가 지났는가"* 를 시간으로 재지 않는다 --- **검토가 있었다는 산출물**을 요구한다.
   시간은 재현 불가능하고 산출물은 재현 가능하기 때문이다.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
TMP = os.path.join(ROOT, 'tmp')
DOCS = [os.path.join(ROOT, 'paper', 'waveprior.tex'),
        os.path.join(ROOT, 'thesis', 'thesis_main.tex')]
REVIEW_MARK = ('AMENDMENT', 'RESULT.MD', 'REVIEW', 'AUDIT', 'RETRACT')
# ★래칫. ⛔침묵시키지 말고 **의도적으로** 내려라(사유를 주석에).
# ⚠[2026-09-03] 첫 전수 측정 = **25**. 대부분은 *"verdict 에 `grade` 가 없다"* --- 옛 배치가
#   그 필드를 안 쓴다. ⛔그 자체가 결함은 아니지만, **필드가 없으면 원고가 등급을 기계로 알 수 없다**.
#   ⟹ 새 배치는 반드시 `grade` 를 넣는다. 줄어들면 이 수를 **의도적으로** 내려라.
BASELINE = 25


def cited_batches():
    """원고가 **본문에서** 이름으로 부르는 배치(주석 제외)."""
    out = set()
    for d in DOCS:
        if not os.path.exists(d):
            continue
        for line in open(d, encoding='utf-8'):
            t = line.split('%')[0]          # ⛔주석은 인용이 아니다
            for m in re.finditer(r'\\texttt\{(R\d[a-z]?|[BSTWM]-?\d+[a-z]?)[-\w]*\}', t):
                out.add(m.group(1).replace('-', ''))
    return out


def batch_key(dirname):
    m = re.match(r'([A-Z]+\d*[A-Z]?)_', dirname)
    return m.group(1) if m else None


def main(argv):
    cited = cited_batches()
    bad = []
    for d in sorted(glob.glob(os.path.join(TMP, '*'))):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        key = batch_key(base)
        if not key or key.upper() not in {c.upper() for c in cited}:
            continue
        vs = sorted(glob.glob(os.path.join(d, 'verdict*.json')))
        if not vs:
            continue
        # ⛔대소문자 --- 초판이 'RESULT.md' 를 대문자화한 이름과 비교해 **못 봤다**.
        names = [os.path.basename(f).upper() for f in glob.glob(os.path.join(d, '*'))]
        marks = [n for n in names if any(m in n for m in REVIEW_MARK)]
        for v in vs:
            name = os.path.basename(v)
            why = []
            if not marks:
                why.append('검토 산출물(RESULT/AMENDMENT/AUDIT)이 없다')
            try:
                grade = json.load(open(v, encoding='utf-8')).get('grade')
            except Exception:
                grade = None
            if grade is None:
                why.append('verdict 에 grade 가 없다')
            elif str(grade).upper() not in ('CONFIRMATORY',) and not marks:
                why.append(f'등급 {grade} 인데 배치 문서가 없다')
            if why:
                bad.append((base, name, why))

    print('── 원고 인용 배치의 성숙도 ──────────────────────────────')
    for b, n, why in bad:
        print(f'  ⛔{b}/{n}: ' + ' · '.join(why))
    print(f'  원고가 부르는 배치 {len(cited)} · 위반 {len(bad)} (기준선 {BASELINE})')
    if len(argv) > 1:
        print('  [측정] 인자 지정 실행 --- 래칫 미적용.')
        return 0
    if len(bad) > BASELINE:
        print(f'  [FAIL] 위반 {len(bad)} > 기준선 {BASELINE}'); return 1
    if len(bad) < BASELINE:
        print(f'  [FAIL] 위반 {len(bad)} < 기준선 {BASELINE} — **개선이 기록되지 않았다.**')
        return 1
    print('  [OK  ] 래칫 유지')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
