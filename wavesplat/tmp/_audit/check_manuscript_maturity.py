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
# ⚠[2026-09-05] **25 → 27 로 의도적으로 올린다.** ⛔침묵시킨 것이 아니라 사유를 적는다:
#   학위논문에 §sec:results_blocked(라벨 없는 두 배치)를 신설하면서 **새로 이름을 부른 배치
#   셋**이 생겼다 --- `T17`(위반 0) · `T18`(+1) · `M8`(+1).
#     · `T18_FACTORIAL_2026-09-05/verdict_t18.json` = `grade` 가 `None` 이다. 그 배치는
#       **차단되어 라벨이 없고**(`label = null`) 분석기가 등급도 쓰지 않았다.
#       ⛔그 파일은 **동결물**이라 고치지 않는다(고치면 그것이 더 큰 결함이다).
#       ⚠형제 배치 `T14`·`T17` 은 차단되고도 `grade` 를 썼다 ⟹ 이것은 `T18` 분석기의
#       누락이고, **다음 배치가 반복하지 말아야 할 것**이다.
#     · `M8_LADDER_2026-08-28/verdict_m8.json` = 같은 사유(옛 배치가 그 필드를 안 쓴다).
#       원고가 그것을 **차단된 배치의 선례**로 부르면서 검사 대상에 들어왔다.
#   ★즉 늘어난 둘은 **원고가 나빠져서가 아니라 원고가 더 많이 인용해서** 생긴 것이고,
#   둘 다 「인쇄면의 결함」이 아니라 「동결 verdict 의 필드 부재」다. ⛔그래도 감춘다는 뜻은
#   아니다 --- 그래서 수를 올리고 여기에 이름을 남긴다.
# ★[2026-09-05, 같은 날 두 번째] **27 → 26 으로 의도적으로 내린다.** 백지 독자 심사가
#   *"`M8` 은 이 논문 본문에서 여기 한 번만 나온다 --- 소절 존재의 정당화 절반이 조회 불가능한
#   선례에 걸려 있다"* 라 지적해 그 인용을 **뺐다**(선례는 `T14` 하나로 충분하고 그쪽은 본문에
#   문단이 있다). ⛔래칫을 피하려고 뺀 것이 아니다 --- 사유는 **독자의 조회 가능성**이고,
#   이 수가 내려간 것은 그 편집의 부수효과다. 남는 위반 26 에는 `T18`(grade 없음)이 포함된다.
BASELINE = 26


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
