#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★**학위논문 동반 공시 검사기** --- `check_paper_disclosures.py` 의 학위논문판.

⛔왜 이것이 필요한가 (2026-09-02 적대적 감사 A6·A10·A11 이 **셋 다 같은 원인**을 지목했다):

> `scripts/check_paper_disclosures.py:28` 의 대상은 `paper/waveprior.tex` **하나**다.
> ⟹ 학위논문의 의무 공시는 **아무도 기계로 보지 않는다.**

그래서 오늘 감사가 셋을 찾았다 --- `B-1` 의 *"게이트·분석이 결과 뒤에 작성됐다"*(0회) ·
`B-26` `SS-LOWH` 의 *"명목/탐색적"*(다섯 곳 중 0곳) · `M-10` 의 *"blind 아님"*(0회).
셋 다 **동결 판정 파일이 결과 시점에 직접 요구한** 공시다.

⚠**범위를 정직하게 적는다**: 이 검사기는 논문판의 35규칙을 이식한 것이 **아니다**.
오늘 수리한 **네 건**의 공시가 다시 사라지는 것을 막는다. 규칙은 늘려도 좋으나,
늘릴 때마다 `--selftest` 의 A/B/C 를 통과시켜라.

★그리고 ⛔**요구하는 자구가 참인지**(실패 모드 D)를 값으로 붙든다: `A9` 규칙은
비교자 강도 두 수를 **동결 행에서 다시 계산**해 문서의 값과 대조한다.
"""
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEX = os.path.join(ROOT, 'thesis', 'thesis_main.tex')


def body(text):
    """주석 제거 + 공백 정규화. ⛔주석은 이 자구를 **인용해서** 설명하므로 세지 않는다."""
    out = []
    for line in text.split('\n'):
        if line.lstrip().startswith('%'):
            continue
        out.append(re.sub(r'(?<!\\)%.*$', '', line))
    return re.sub(r'\s+', ' ', ' '.join(out))


def sections(text):
    """(제목, 본문) 목록 --- ⛔**절 단위로** 본다.

    ⛔[2026-09-02 감사 M-7-1] 초판은 문서 전체를 한 문자열로 접었다. 그래서 한 곳의
    공시가 **다른 절의 무공시 인용을 면허**했다(근거를 전부 지워도 통과했다).
    ★논문판은 2026-09-01 에 정확히 이 결함을 고쳤는데 **이식되지 않았다** ---
    「같은 결함을 두 번 고치지 않으려면 처방도 두 문서에 옮겨야 한다」.
    """
    ABS = r'\\begin\{(abstract|englishabstract|onecolabstract)\}'
    parts, cur, title = [], [], '<frontmatter>'
    for line in text.split('\n'):
        env = re.match(ABS, line)
        # ⛔[2026-09-05 적대적 검사 MINOR 9] 초판은 `\subsubsection` 과 **들여쓴** 제목을
        #   경계로 세지 않았다 ⟹ 같은 편집이 `\subsection` 이면 스코프가 갈리고
        #   `\subsubsection` 이면 안 갈리는 **비일관**. 둘 다 경계로 센다.
        if re.match(r'\s*\\(?:sub)*section\*?\{|\s*\\chapter\*?\{', line) or env:
            parts.append((title, '\n'.join(cur)))
            title = line.strip()[:60] if not env else '<' + env.group(1) + '>'
            cur = []
        else:
            cur.append(line)
    parts.append((title, '\n'.join(cur)))
    return parts


# ═══════════════════════════════════════════════════════════════════════════
# ★★[2026-09-05] 적대적 검사(fresh lens)가 이 검사기에서 **BLOCKER 3 + MAJOR 5** 를 냈다.
#   아래 둘은 그 수리다. ⛔둘 다 「검사기 자신의 실패 모드」에 대한 것이다.
#
# (A) **오탐 방어** --- *"있는 것을 없다고"* 는 *"없는 것을 있다고"* 만큼 나쁘다.
#     실증된 오탐 10건: `\emph{}` 를 떼거나 붙이면 · 조사를 `아니다→아니고` 로 바꾸면 ·
#     `단일요인→단일 요인` 으로 띄면 · **한국어 어절 안에서 줄바꿈**이 나면 ·
#     LaTeX `%` 줄이음(CJK 에서 공백 억제 관용구)을 쓰면 규칙이 「누락」을 외쳤다.
#     ⟹ **평문 자구**는 글자 사이에 「공백·글꼴명령·닫는 중괄호」가 끼어도 맞도록 컴파일한다.
#     ⛔매크로 이름(`\\TFifteen...`)과 정규식은 **그대로** 엄격히 본다.
#
# (B) **우회 방어** --- 트리거가 전부 매크로 이름이라, 같은 수를 **리터럴로** 적으면
#     33규칙 중 16이 침묵했다(실증). 인쇄되는 수는 한 글자도 안 바뀐다.
#     ⟹ 트리거를 `thesis/t11t14_constants.tex` **에서 생성**해 매크로 이름과 **그 값**을
#     동시에 잡는다. ⛔값이 `23/64` 처럼 **변별적일 때만** 넣는다(맨 정수는 오탐 공장이다).
_GAP = r'(?:\s|\\(?:emph|textbf|texttt|textit|mathbf|mathrm)\{|\})*'
_META = set('|()[]*+?^$.')


def _tolerant(lit):
    """평문 자구 → 마크업·줄바꿈에 관대한 정규식."""
    return _GAP.join(re.escape(c) for c in lit if not c.isspace())


def compile_need(pat):
    """매크로/정규식은 엄격히, **평문 자구**는 관대하게."""
    if '\\' in pat or any(c in _META for c in pat):
        return re.compile(pat)
    return re.compile(_tolerant(pat))


_NEED_CACHE = {}


def need_re(pat):
    if pat not in _NEED_CACHE:
        _NEED_CACHE[pat] = compile_need(pat)
    return _NEED_CACHE[pat]


CONSTS = os.path.join(ROOT, 'thesis', 't11t14_constants.tex')
# ⛔맨 정수(`64`·`41`·`2`)는 트리거로 쓰지 않는다 --- 문서 어디에나 있다.
_DISTINCT = re.compile(r'\d+/\d+|[+-]?\d+\.\d+\\%|0\.\d{3,}|\d+\.\d+\\%')


def _macro_values():
    if not os.path.exists(CONSTS):
        return {}
    out = {}
    for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', open(CONSTS, encoding='utf-8').read()):
        out[m.group(1)] = m.group(2)
    return out


def _val_alts(v):
    """인쇄된 **값**을 잡는 대안들 --- 리터럴 우회를 막는다."""
    alts = [re.escape(v)]
    m = re.fullmatch(r'(\d+)/(\d+)', v)
    if m:
        w, n = m.groups()
        alts.append(rf'\$?{w}\s*/\s*{n}')                    # 23/64 · $23/64$
        alts.append(rf'\${n}\$[^$]{{0,8}}중\s*\${w}\$')      # $64$ 중 $23$
    return alts


# ⚠**값 대안은 공짜가 아니다.** 첫 구현이 `63/64`·`12/32`·`+8.3\\%` 로 **무관한 절 셋에서
#   오탐 13건**을 냈다 --- 같은 리터럴이 다른 배치 자리에도 있기 때문이다.
#   ⟹ 값 대안은 **충돌하지 않을 때만** 넣고, 넣지 못한 것은 `[note]` 로 **인쇄한다**
#   (⛔침묵시키지 않는다 --- 그 값은 리터럴 우회에 여전히 열려 있다).
SKIPPED_VALUE_ALTS = []
_SEC_CACHE = {}


def _doc_sections():
    if 'v' not in _SEC_CACHE:
        _SEC_CACHE['v'] = [body(c) for _, c in sections(open(TEX, encoding='utf-8').read())]
    return _SEC_CACHE['v']


def trig(*names):
    """매크로 이름 + (변별적이고 **충돌하지 않으면**) 그 값으로 트리거를 만든다."""
    vals, alts = _macro_values(), []
    for n in names:
        mac = r'\\' + n + r'(?![A-Za-z])'
        alts.append(mac)
        v = vals.get(n, '')
        if not _DISTINCT.fullmatch(v):
            continue
        cand = _val_alts(v)
        # ⛔그 리터럴이 **매크로가 없는 절**에도 나오면 오탐 공장이다 --- 통째로 버린다.
        clash = any(re.search(a, sec) and not re.search(mac, sec)
                    for a in cand for sec in _doc_sections())
        if clash:
            SKIPPED_VALUE_ALTS.append(f'{n}={v}')
        else:
            alts += cand
    return '|'.join(alts)


# (이름, 트리거 정규식, 요구 정규식 목록(하나라도 있으면 통과), 사유)
# ★[2026-09-03] 검사기가 실제로 놓쳤던 문자열. 트리거 회귀 시험용(⛔지우지 마라).
TRIGGER_REGRESSION = ('약 $39\\%$ 낮춘다($64$ 에피소드 중 $62$, 양측 $p = 2.3'
                      '\\times10^{-16}$; 배치 \\texttt{B-1}')

RULES = [
    # ★★[2026-09-03 새벽, 감사 지적] 학위논문에 `T1`·`T1b`·`C2` 규칙이 **0개**였다.
    #   그 셋이 새로 들어온 절이고 인쇄 수치를 담는다 ⟹ 최소 규칙을 건다.
    ('TOne-사다리',
     r'\\texttt\{T1b\}',
     [r'모순이 아니다', r'검정력의 진술'],
     '⛔T1b 를 적으면 **T1 과 모순이 아니라 검정력의 사다리**임을 병기'),
    ('TOne-비독립',
     r'\$2/12 \\to 9/12\$|프레임 수만',
     [r'독립이 아니므로', r'판정 갱신에 쓰지 않는다'],
     '⛔SAME 팔은 T1 과 비독립 ⟹ 판정 갱신에 쓰지 않음을 병기'),
    ('TOne-solver미명시',
     r'Zenodo 17359600',
     [r'생성 solver 를 명시한 문서가', r'위상 결합이 있다'],
     '⛔solver 미명시 ⟹ "HOS"·"완전 비선형" 이라 부르지 않음을 병기'),
    ('TOne-탐색적',
     r'순위상관 \$0\.99\$',
     [r'사전등록하지 않은', r'탐색적'],
     '⛔용량-반응은 사전등록되지 않았음을 병기'),
    ('CTwo-프레임집합',
     r'\\CTwoWin\{\}',
     [r'프레임 집합이 다르다', r'\\CTwoWinNoZero'],
     '⛔C2 의 계수와 크기는 프레임 집합이 다르다 --- t=0 공시를 병기'),
    # ★★[2026-09-03] 오늘 들어온 세 배치. 방벽 비대칭(논문 48규칙 vs 학위논문 5규칙)을
    #   **적어도 새 수치에 대해서는** 닫는다. ⛔트리거는 좁게 --- 늑대를 외치지 않는다.
    # ★★[2026-09-03] `C1` — 거리를 충분성으로 읽는 것이 유일한 오독 경로다.
    ('COne-충분성아님',
     r'\$0\.308\$--\$1\.416\$',
     [r'거리\}이지'],
     '⛔C1 은 **거리**를 잰다 --- "충분히 크다" 로 읽히지 않게 병기'),
    # ⚠[항진 수리] 초판 트리거 `C1-MODELGAP` 은 **주석 안에만** 있어 검사기가 못 본다
    #   (검사기는 주석을 지운다) ⟹ 규칙이 한 번도 발화하지 않았다. 본문 자구로 바꾼다.
    # ★★[2026-09-03] `C1c` — max 노름만 인쇄하면 과대독해된다.
    ('COne-두노름',
     r'\$0\.308\$--\$1\.416\$',
     # ⛔[2026-09-03 정정] 초판은 **재적합 뒤** 값(0.0581)을 요구했다 = 틀린 짝 강제.
     [r'중앙 \$\\mathbf\{0\.0786\}\$'],
     '⛔C1 의 max 노름을 적으면 **제곱평균(0.0581)** 을 같은 절에 --- 한 자릿수 다르다'),
    ('COne-선형유지',
     r'\\texttt\{C1\}',
     [r'정확히\} \$0\.0\$'],
     '★C1 을 인용하면 **선형 코퍼스의 대조(정확히 0)** 를 같은 절에 --- 판정은 유지된다'),
    ('REight-층갈림',
     r'R8-HELDOUT|\$0\.717 \\to 0\.979\$',
     # ⛔[항진 수리] 요구가 둘이면 **OR** 라 하나를 지워도 통과한다 --- 하중 하나만 남긴다.
     [r'\$3/5\$'],
     '⛔R8 의 10/12 는 검정력 층(박스 ≥ 40 m)에서 3/5 로 재현되지 않는다 --- 병기 의무'),
    ('RSeven-교락대조',
     r'R7-FLOOR|Spearman \$\\rho = 0\.893\$',
     [r'상관 \$0\.000\$'],
     '★R7 의 하중은 상관이 아니라 C-1 대조(무기술 기준선과 상관 0.000)다 --- 병기 의무'),
    ('RSeven-범위',
     r'R7-FLOOR|Spearman \$\\rho = 0\.893\$',
     # ⛔[항진 수리] `탐색적` 은 이 절의 다른 곳에도 있어 지워도 통과했다 --- 이 배치에만
     #   있는 자구를 요구한다(★검사기가 「없는 것을 있다」고 말하는 쪽의 항진).
     [r'사이트 \$1\$ 개에 기대고'],
     '⛔R7 = 탐색적 · blind 아님 · 사이트 1개 · 상한 --- 같은 절에 적어야 한다'),
    ('HThreeC-처방아님',
     r'H3c-LAW|\$0\.348 \\to 0\.303\$',
     [r'복사량 항 자신'],
     '⛔H3c 의 함의는 "다른 모형으로 갈아라" 가 아니다 --- 남는 용의자는 복사량 항 자신'),
    # ★[2026-09-03 박사 기여 감사 C-3] 희소 용량--반응의 아래 계단은 **무기술**이다.
    #   `14/16` 만 인용하면 계단 크기를 오도한다 ⟹ 병기를 강제한다.
    ('BNineteen-아래계단',
     r'희소 층 \$14/16\$|\$14/16\$.{0,80}?B-19|B-19.{0,80}?\$14/16\$',
     [r'KGRID2', r'\$2/16\$', r'무기술'],
     'B-19 의 14/16 인용에는 KGRID2 의 제로필드 대비 2/16(무기술) 병기 의무'),
    # ⛔[2026-09-02 감사 M-7-3] 초판 트리거 `62/64` 는 **두 초록을 못 봤다**
    #   (국문 `$64$ 중 $62$` · 영문 `$62$ of $64$`) ⟹ 가장 눈에 띄는 인용이 검사 밖이었다.
    ('BOne-사후작성',
     # ⚠[2026-09-02] `62/64` 단독은 **다른 양**에도 걸린다(W-15 의 비교자 대 제로).
     #   ⟹ **`B-1` 태그와의 근접**을 요구한다.
     # ⛔[2026-09-03] 이 정규식이 **실제 결함을 놓쳤다**: 기여 목록(`:1647`)의 문자열은
     #   `$64$ **에피소드** 중 $62$` 인데 위 패턴은 `$64$ 중 $62$` 만 잡았다 ⟹ 검사기가
     #   *"누락 0"* 을 찍는 동안 논문에서 가장 많이 인용될 수에 사후작성 공시가 없었다.
     #   ★같은 실패 모드(「있는 것을 없다고」의 역방향 = 「없는 것을 있다고」)의 재발.
     #   ⟹ 두 수 사이에 **한 낱말까지 허용**한다.
     r'(?:62/64|\$64\$(?:\s*\S{1,6})?\s*중\s*\$62\$|\$62\$ of \$64\$).{0,160}?B-1'
     r'|B-1.{0,160}?(?:62/64|\$64\$(?:\s*\S{1,6})?\s*중\s*\$62\$|\$62\$ of \$64\$)',
     # ⚠검사기는 **원문**을 본다 --- 매크로는 조판 시점에야 문장이 되므로
     #   매크로 이름 자체를 요구 목록에 넣어야 한다(안 그러면 참인 공시를 누락으로 오판한다).
     [r'\\BOnePostHoc(?:EN)?\{\}', r'게이트\$?\\cdot\$?분석 코드는 결과 뒤에 작성',
      r'결과 뒤에 작성되었다', r'written after (?:its |the )?results'],
     'B-1 의 게이트·분석은 결과 뒤에 작성됐다(verdict_b1.json `_honesty`) — 병기 의무'),
    # ⛔[2026-09-02 감사 M-7-2] 초판은 **항진**이었다 --- 트리거가 `\newcommand` 정의
    #   자신에 매치되므로 사용처를 전부 지워도 발동·통과한다.
    #   ⟹ 트리거를 **크기 자구**로 바꾸고 요구를 **매크로 사용**으로 바꾼다.
    # ⚠[2026-09-02] 초판 트리거는 **금지 문장**에도 걸렸다(*"약 $-40\%$ 를 약 $-39\%$ 와
    #   직접 비교하지 않는다"* --- 그것은 크기 주장이 아니라 그 반대다).
    #   ⟹ 크기가 **주장으로 서는 형태**(승수 병기 또는 "낮춘다"/"lowers")만 트리거한다.
    ('BOne-비교자강도',
     r'약 \$39\\%\$[^。]{0,80}?(?:낮춘다|\$62\$)|by about \$39\\%\$[^.]{0,80}?relative to',
     [r'\\CmpSkillBOne(?:EN)?\{\}'],
     'B-1 크기를 적는 자리에는 그 코퍼스의 비교자 강도를 병기한다'),
    # ⚠[2026-09-02] 의무는 **매크로**가 아니라 **공시**다 --- §결과 절은 표 칸에
    #   `SS-LOWH … 명목` 으로 이미 공시한다. 매크로만 요구하면 참인 공시를 누락으로 오판한다.
    ('SSLowh-명목',
     r'네 칸이 \\textbf\{모두 유지\}|all four cells holding|네 칸이 모두 유지',
     [r'\\SSLowhNominal', r'SS-LOWH\}?[^\n]{0,120}?명목', r'명목[^\n]{0,80}?SS-LOWH',
      r'탐색적'],
     'verdict_b26.json corners.SS-LOWH.reading 이 「가족단위 미달 = 명목」 병기를 요구한다'),
    ('MTen-blind',
     r'동반해야 하는 공시',
     [r'blind 가 아니다'],
     'verdict_m10.json disclosures[0] = 「blind 아님 + 스모크 노출」 — 병기 의무'),
    # ★★[2026-09-05] `T11`~`T14`. ⛔이 넷의 인용금지는 **자구가 아니라 병기**로만 막을 수
    #   있다(금지 목록 `## 2026-09-04/05` 절). 트리거를 **매크로 이름**으로 잡는다 ---
    #   수치는 전부 `thesis/t11t14_constants.tex` 에서 오므로 산문에 숫자가 없다.
    #   ⚠`check_spelled_ratios.py` 가 「철자/한글로 쓴 비율」 쪽 우회로를 따로 막는다.
    ('TEleven-사고실험',
     r'\\TElevenPrimary',
     [r'사고실험', r'thought experiment'],
     '⛔`RESCALE` 은 단위를 바꾼 사고실험이다 — 성능·배포 구성으로 읽히지 않게 병기'),
    ('TEleven-절대스케일',
     r'\\TElevenPrimary',
     [r'절대 스케일', r'absolute scale', r'절대 파수'],
     '⛔정본 읽기는 「배포 추정기의 **절대 스케일** 설계 범위」다 '
     '(⛔"학습 사전의 파장 밴드" 아님 — k_p 장면은 학습 밴드 83백분위 안)'),
    ('TTwelve-단일상수아님',
     r'\\TTwelvePrimary',
     [r'단일 상수', r'single constant'],
     '⛔「단일 상수」 금지 — 그 팔도 맵 칸·사전 밴드·polish 물리 스텝을 함께 옮긴다'),
    ('TThirteen-MDE',
     r'\\TThirteenPrimary',
     [r'\\TThirteenMDE'],
     '⛔`FIX-UNDETECTED` 는 검정력의 진술이다 — MDE(q80) 병기 의무 '
     '(⛔"수심은 원인이 아니다"·"무효과 확증" 금지)'),
    ('TFourteen-차단',
     r'\\TFourteenBlocker',
     [r'\\TFourteenMDE'],
     '⛔`T14` 는 차단된 배치다 — 라벨·부분 통계 금지, 음성 읽기에는 MDE(q80) 병기 의무'),

    # ═════════════════════════════════════════════════════════════════════════
    # ★★[2026-09-05] `T15-HOMEREG` + `T16-BANDFIX`.
    # ⛔**계기**: 같은 두 배치가 `paper/waveprior.tex` 에는 **26 규칙**으로 실렸는데
    #   학위논문 쪽은 **0 규칙**이었다 ⟹ 원고가 학위논문보다 **더 많이 공시**하는 상태.
    #   실측된 누락 다섯 = 중간 열람 · 게이트 완화 · 속성 노출 36/60 · 스윕 부분층 ·
    #   개입 잡음 바닥. ★공시는 기계가 보지 않으면 **한 번의 편집**으로 사라진다
    #   (이 저장소는 같은 문장을 그렇게 세 번 잃었다).
    # ⚠규칙마다 요구는 **하나**다 --- `need` 는 OR 이므로 둘을 넣으면 하나를 지워도 통과한다
    #   (이 파일의 `REight-층갈림`·`RSeven-범위` 가 그 항진을 이미 한 번 고쳤다).
    # ⚠이 검사기에는 논문판의 `MULTI_RULES`(크기 결합 금지)가 없다. `\TFifteenRel` 은
    #   **해침** 크기이고 이득 크기 옆에 놓이면 뺄셈을 부른다 ⟹ 현재 그 매크로는
    #   `sec:results_t16` 한 절에만 있고 그 절에 이득 크기가 **0개**임을 확인했다.
    #   ⛔다른 절로 옮기려면 그때 결합금지 규칙을 함께 만들어라.
    # ═════════════════════════════════════════════════════════════════════════
    # ★T15 의무 묶음 --- 트리거는 **인용가치 있는 T15 계수 전부**다(적대적 검사 BLOCKER 2:
    #   `\TFifteenCtrl`(63/64)·`\TFifteenRel`(+12.2%) 단독 인용이 무규칙이었다).
    #   ⚠논문판 트리거가 `…CrowdedWin|…Rel` 인데 학위논문판이 `…Crowded` 뿐이었다(회귀).
    ('TFifteen-학습봉투', trig('TFifteenCrowded', 'TFifteenRel', 'TFifteenCtrl',
                          'TFifteenCtrlSparse', 'TFifteenLabel'),
     [r'학습\s*(?:분포\s*)?봉투'],
     '⛔자국 코퍼스는 **배포 체크포인트의 학습 봉투**다 — '
     '"out-of-sample"·"held-out"·"blind" 자구 금지를 같은 절에'),
    ('TFifteen-문턱정확', trig('TFifteenCrowded', 'TFifteenRel'),
     ['정확히 만나'],
     '★계수가 해침 문턱을 **정확히** 만난다(w == thresh_hurt) — 편안하게 읽히지 않도록 '
     '그 사실을 같은 절에'),
    ('TFifteen-이탈공시', trig('TFifteenCrowded', 'TFifteenRel', 'TFifteenCtrl',
                          'TFifteenLabel'),
     [r'\\TFifteenSmokeExposed'],
     '⛔**사전등록 이탈** — 스모크가 판정 표본 안 2블록을 먼저 봤다(혼잡은 등록 예측과 '
     '반대 방향). 그 공시 없이 PRIMARY 계수 인용 금지'),
    ('TFifteen-중간열람', trig('TFifteenCrowded', 'TFifteenRel', 'TFifteenCtrl',
                          'TFifteenLabel'),
     [r'\\TFifteenPeekWin'],
     '⚠본실행 도중 PRIMARY 통계량의 **중간집계를 봤다**(`NOTE_INTERIM_PEEK.md`) — '
     '저장소 규율은 *"봤지만 안 말한다 는 공시가 아니다"* 다'),
    ('TFifteen-열람양성대조', trig('TFifteenPeekWin'), [r'\\TFifteenPeekCtrl'],
     '⛔중간 열람에서 **보인 세 계수 전부**를 적어라 — 양성 대조가 빠졌다'),
    ('TFifteen-열람바닥', trig('TFifteenPeekWin'), [r'\\TFifteenPeekFloor'],
     '⛔중간 열람에서 **보인 세 계수 전부**를 적어라 — 바닥 팔이 빠졌다'),
    ('TFifteen-배포구성아님', trig('TFifteenCrowded', 'TFifteenRel', 'TFifteenCtrl',
                            'TFifteenLabel'),
     ['배포 구성이 아니'],
     '⛔`T15` 의 어떤 처치 팔도 배포 구성이 아니다 — 배포 권고로 읽히지 않게 병기'),
    # ⛔[BLOCKER 3] 같은 음성을 **p 매크로로만** 인용하면 MDE 병기가 증발했다(실증).
    ('TFifteen-MDE', trig('TFifteenSparse', 'TFifteenSparseP'), [r'\\TFifteenMDE'],
     '⛔희소 층의 미검출을 *"무해하다"* 로 읽지 못하게 — MDE(q80) 병기 의무'),
    ('TFifteen-바닥', trig('TFifteenRel', 'TFifteenDoseRel', 'TFifteenSweepLoRel',
                        'TFifteenSweepMidRel', 'TFifteenSubFTwoRel'),
     [r'\\TFifteenFloor(?![A-Za-z])'],
     '⛔크기 인용에는 **결과 전 고정된 개입 잡음 바닥** 병기 의무 — 효과가 그 바닥 '
     '바로 바깥이다'),
    ('TFifteen-바닥이빌려온것', trig('TFifteenFloor'), [r'\\TFifteenFloorSparse'],
     '⛔잡음 바닥은 **빌려온 것**이고 희소 층에서는 성립하지 않는다(도움 문턱을 정확히 '
     '만난다) — 두 계수 병기 의무'),
    # ⛔⛔스윕 세 팔만 보면 단조지만 PRIMARY `f=2` 를 같은 32 부분층에 놓으면 깨진다.
    #   ⚠트리거에 **승수**(9/32 등)를 넣는다 --- 논문판에는 있고 학위논문판에 빠져 있었다.
    ('TFifteen-스윕비단조',
     trig('TFifteenDoseRel', 'TFifteenSweepLoRel', 'TFifteenSweepMidRel',
          'TFifteenSubFTwoRel', 'TFifteenDose', 'TFifteenSweepLo',
          'TFifteenSweepMid', 'TFifteenSubFTwo'),
     ['단조가 아니'],
     '⛔스윕 사다리는 **단조가 아니다**(f=2 를 같은 부분층에 넣으면 깨진다) — 병기 의무'),
    ('TFifteen-스윕표본',
     trig('TFifteenDoseRel', 'TFifteenSweepLoRel', 'TFifteenSweepMidRel',
          'TFifteenSubFTwoRel', 'TFifteenDose', 'TFifteenSweepLo',
          'TFifteenSweepMid', 'TFifteenSubFTwo'),
     ['PRIMARY 표본에서 돌지 않는다'],
     '⚠스윕 세 팔은 **32 부분층**에서만 돈다 — PRIMARY(n=64) 계수와 표본이 다르다'),
    # ⛔[MAJOR 6] 초판은 단일 일반 토큰 `재집계` 를 요구했다 --- 그 낱말은 학위논문에
    #   33회 나오고, **정반대 뜻의 문장**이 의무를 대신 만족시키는 것이 실증됐다.
    #   ⟹ **주장 전체**를 요구한다.
    ('TFifteen-스윕재집계', trig('TFifteenSubFTwo', 'TFifteenSubFTwoRel'),
     [r'동결 verdict 에[^.]{0,30}?없는\s*\\emph\{재집계\}', r'없는 \\emph\{재집계\}'],
     '⛔`f=2` 의 부분층 값은 동결 verdict 에 **없다** — 재집계임을 병기해야 한다'),
    # ⛔[MAJOR 5] *"외부 기준선 행이 0개"* 는 이 문서의 집필 관용구라 한 배치의 공시가
    #   다른 배치의 무공시를 면허했다(실증: 한쪽만 지워도 누락 0). ⟹ **배치별로 쪼갠다**.
    ('TFifteen-외부기준선없음', trig('TFifteenCrowded', 'TFifteenRel', 'TFifteenLabel'),
     ['여기에도 \\\\textbf\\{외부 기준선 행이'],
     '⛔`T15` 도 외부 기준선 행 **0개** — *"저들보다 낫다"* 로 읽히지 않게 병기'),
    ('TSixteen-외부기준선없음', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     ['이 배치에도 \\\\textbf\\{외부 기준선 행이'],
     '⛔`T16` 도 외부 기준선 행 **0개** — 비교자는 참 제로필드다'),
    ('TSixteen-옛규칙', trig('TSixteenSkill', 'TSixteenPosCtrl'),
     [r'\\TSixteenOldRule(?![A-Za-z])'],
     '⛔`S` 축은 `SEC-E`(옛 규칙도 제로필드를 이긴다) 병기 없이 인용 금지 — '
     '이긴 것은 *"자동 스케일"* 이지 *"밴드 수리"* 가 아니다'),
    ('TSixteen-등변재규정', trig('TSixteenEquivarViol'), ['다시 규정된 것이지'],
     '⛔`T14` 의 **그** 게이트가 아니다 — 저쪽 허용오차는 c=2.0 방향에만, 규약도 다르다'),
    ('TSixteen-옛등변방향', trig('TSixteenEquivarOldDown'), [r'\\TSixteenEquivarOldUp'],
     '⛔옛 규칙의 등변 편차는 **방향 의존**이다 — 한쪽만 인쇄하면 체리피킹이다'),
    ('TSixteen-D축구동', trig('TSixteenLanding', 'TSixteenLandingP'),
     [r'\\TSixteenDrive(?![A-Za-z])'],
     '⛔`D` 축은 `D_drive`(중앙 |Δ ln s| 와 그 자) 병기 없이 인용 금지'),
    ('TSixteen-D축부호', trig('TSixteenLanding', 'TSixteenLandingP'),
     [r'\\TSixteenDrivePos'],
     '⛔`D` 축의 구동량은 **부호가 섞인다**(+39/−21) — 그 분할 없이 인용 금지'),
    ('TSixteen-D축MDE', trig('TSixteenLanding', 'TSixteenLandingP'),
     [r'\\TSixteenLandingMDE'],
     '⛔`D-TREATMENT-BELOW-SCALE` 를 *"수리가 무의미하다"* 로 읽지 못하게 — MDE 병기'),
    # ★T16 의무 묶음 --- 트리거에 인용가치 높은 계수를 전부 넣는다(BLOCKER 2).
    ('TSixteen-착지점아님', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl',
                           'TSixteenPriorD'),
     ['착지점 진술이지 적응 진술이 아니다'],
     '⚠규칙이 **읽는 양**이 코퍼스 상수다 ⟹ **착지점 진술이지 적응 진술이 아니다**'),
    ('TSixteen-상수인것은읽는양', trig('TSixteenSkill', 'TSixteenOldRule'),
     ['규칙이 읽는 양이 코퍼스 상수'],
     '⛔상수인 것은 **규칙이 읽는 양**(1단 반경)이다 — *"규칙이 고르는 배율이 상수"* 는 '
     '동결 행이 반증한다(`s_rule` 고유값 41/60 · 35/60)'),
    ('TSixteen-배포권고아님', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl',
                            'TSixteenPriorD'),
     ['배포 권고가 아니다'],
     '⛔양성 라벨도 **필요조건 하나**다(단가·실측 경로·자국 무해성이 남는다)'),
    ('TSixteen-blind아님', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl',
                            'TSixteenAttrExposed'),
     ['완전 blind 가 아니다'],
     '⚠설계자가 앞 배치의 개정문과 앵커 값을 읽은 상태에서 문턱을 잡았다 — 병기 의무'),
    ('TSixteen-속성노출', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     [r'\\TSixteenAttrExposed'],
     '⚠판정 60블록은 **성과** 미노출일 뿐 36개는 앞 배치가 **속성**을 열었다 — '
     '*"아무도 열지 않은 신선한 블록"* 으로 읽히지 않게 그 계수를 같은 절에'),
    ('TSixteen-속성정체', trig('TSixteenAttrExposed'), ['비선형성과 분산관계'],
     '⛔열린 속성은 격자·기하가 아니라 **참값장의 비선형성·분산관계**다'
     '(`gates_t16.exposed_blocks()` 축자) — 노출을 무해한 쪽으로 적지 마라'),
    ('TSixteen-실측무언', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     ['말도 하지 않는다'],
     '⛔이 배치는 배포 **합성 경로**를 잰다 — 실측 경로에 대해 아무 말도 하지 않는다'),
    ('TSixteen-게이트항진', trig('TSixteenGates'), ['강도의 척도가 아니다'],
     '⚠게이트 계수는 **강도의 척도가 아니다** — 배포 상수상 구조적으로 참인 항이 여럿이다'),
    ('TSixteen-강건성아님', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     ['강건성이 아니다'],
     '⛔잰 것은 **등변성과 격자 여유**이지 강건성이 아니다(verdict §9.3-18) — '
     '*"새 규칙이 더 튼튼하다"* 로 읽히지 않게 병기'),
    ('TSixteen-단일요인아님', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     ['단일요인이 아니'],
     '⛔재척도는 **단일요인이 아니다**(verdict §9.3-3) — 기제 지목 금지를 같은 절에'),
    ('TSixteen-비대칭교락', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     ['규칙이 움직이는 방향으로 비대칭'],
     '⛔`d3_beamform_net` 의 하드코딩 수심이 **비대칭 교락**이다(verdict §9.3-4) — '
     '스케일을 키우면 그 불일치도 줄어든다'),
    ('TSixteen-사전확률', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     [r'\\TSixteenPriorD'],
     '★설계가 결과 **전에** P(S 축 양성)≈1 · P(이 착지 라벨)=0.937 이라 적었다 — '
     '⛔그 사실을 결과 뒤에 재서술하지 못하게 같은 절에 인쇄'),
    ('TSixteen-차단게이트', trig('TSixteenGridTol', 'TSixteenGridMeas'),
     ['차단 게이트다'],
     '⛔`G-GRID` 는 `GATES_BLOCKING` 이다 — *"이 게이트를 빼도 나머지는 선다"* 는 거짓'),
    ('TSixteen-게이트완화', trig('TSixteenGates'), ['허용오차가 완화되었다'],
     '⛔`G-GRID` 허용오차가 **완화**됐다(개정문 2, 판정 자료 0행 시점) — 게이트 계수를 '
     '인쇄하는 자리에 그 사실을 같이'),
    ('TSixteen-학습분포유래', trig('TSixteenSkill', 'TSixteenOldRule', 'TSixteenPosCtrl'),
     ['우리 학습 분포'],
     '⚠규칙의 목표 상수는 **우리 학습 분포**에서 나온다(verdict §9.3-8) — '
     '⛔*"제3자 자료가 규칙을 정했다"* 는 거짓'),
    ('TSixteen-모집단', trig('TSixteenShipZero', 'TSixteenSkill'),
     ['계수를 서로 비교하지 않는다'],
     '⛔형제 배치(`T11`·`T14`)의 계수와 **직접 비교 금지** — 모집단이 다르다'),
    # ⛔[백지 독자 B-1] (나)의 코퍼스가 인쇄되지 않으면 제목의 "자국" 때문에 독자가
    #   같은 배포 팔의 63/64 승리와 16/60 패배를 **모순으로** 읽는다.
    ('TSixteen-코퍼스명시', trig('TSixteenSkill', 'TSixteenShipZero'),
     ['제3자 아카이브'],
     '⛔`T16` 은 **자국이 아니라 제3자 아카이브**에서 쟀다 — 소절 제목의 "자국" 은 '
     '(가)에만 걸린다. 그 사실 없이 이 계수를 (가)의 양성 대조 옆에서 읽으면 모순이다'),
]


def value_pin():
    """★실패 모드 D --- 요구하는 자구의 **수가 참인가**를 동결 행에서 다시 계산한다."""
    p = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14', 'results.jsonl')
    if not os.path.exists(p):
        return []
    byep = defaultdict(list)
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('arm') != 'SPEC25' or 'error' in r:
            continue
        m = re.search(r'(\d+)comp', r['episode'])
        if m and int(m.group(1)) in (15, 30):
            byep[r['episode']].append(r)
    if not byep:
        return []
    win = n = 0
    rels = []
    for v in byep.values():
        a = st.median([x['avg_nRMSE'] for x in v])
        z = st.median([x['zero_nRMSE'] for x in v])
        n += 1
        win += a < z
        rels.append((a - z) / z * 100)
    med = st.median(rels)
    txt = open(TEX, encoding='utf-8').read()
    # ⛔**매크로 정의 안에서만** 찾는다. `$64/64$` 는 다른 줄(무기술 대비 열)에도 있어서
    #    문서 전체를 뒤지면 매크로의 수를 바꿔도 통과한다 = 핀이 공허해진다.
    m = re.search(r'\\newcommand\{\\CmpSkillBOne\}\{(.{0,400}?)\}\n', txt, re.S)
    if not m:
        return ['⛔`\\CmpSkillBOne` 매크로 정의를 찾지 못했다']
    mac, bad = m.group(1), []
    if f'${win}/{n}$' not in mac:
        bad.append(f'⛔`\\CmpSkillBOne` 의 승수가 재계산값 {win}/{n} 과 다르다')
    if f'{med:.2f}' not in mac:
        bad.append(f'⛔`\\CmpSkillBOne` 의 상대 중앙이 재계산값 {med:.2f}% 와 다르다')
    return bad


def run(text):
    fired, missing = [], []
    for title, chunk in sections(text):
        b = body(chunk)
        for name, trig, need, why in RULES:
            if not re.search(trig, b):
                continue
            fired.append((name, title))
            if not any(need_re(x).search(b) for x in need):
                missing.append((name, title, why))
    return fired, missing


def selftest():
    """A 위조 · B 침묵 · C 미검증 --- `check_paper_disclosures.py --selftest` 와 같은 규약."""
    text = open(TEX, encoding='utf-8').read()
    b = body(text)
    a_ok = b_ok = 0
    unfired = []
    for name, trig, need, _ in RULES:
        if not re.search(trig, b):
            unfired.append(name)
            continue
        forged = b
        for x in need:
            forged = need_re(x).sub('XXFORGEDXX', forged)
        if any(need_re(x).search(forged) for x in need):
            print(f'  ⛔A 실패 {name}: 근거를 지웠는데도 통과한다')
        else:
            a_ok += 1
        silent = re.sub(trig, 'XXNOTRIGXX', b)
        if re.search(trig, silent):
            print(f'  ⛔B 실패 {name}: 트리거를 지웠는데 여전히 발동한다')
        else:
            b_ok += 1
    # ★[2026-09-05] `TRIGGER_REGRESSION` 은 *"⛔지우지 마라"* 라 적혀 있으면서 **아무 데서도
    #   쓰이지 않았다** --- 죽은 픽스처가 살아 있는 척했다. 여기서 실제로 쓴다.
    tr = next((t for n, t, _, _ in RULES if n == 'BOne-사후작성'), None)
    if tr is None or not re.search(tr, body(TRIGGER_REGRESSION)):
        print('  ⛔D 실패 BOne-사후작성: 트리거 회귀 문자열을 더는 잡지 못한다')
        a_ok = -1
    print(f'── 학위논문 공시 항진 검사 {"─" * 26}')
    print(f'  A 위조 {a_ok}/{len(RULES) - len(unfired)} · '
          f'B 침묵 {b_ok}/{len(RULES) - len(unfired)} · C 미검증 {len(unfired)}')
    if unfired:
        print('  ⚠발동하지 않은 규칙(=미검증, 「통과」로 세지 않는다): ' + ', '.join(unfired))
    ok = (a_ok == b_ok == len(RULES) - len(unfired)) and not unfired
    print('  ' + ('[OK  ] 전부 통과' if ok else '[FAIL] 위 항목을 읽어라'))
    return 0 if ok else 1


def main():
    if '--selftest' in sys.argv:
        return selftest()
    text = open(TEX, encoding='utf-8').read()
    fired, missing = run(text)
    pin = value_pin()
    print('── 학위논문 동반 공시 ' + '─' * 32)
    for name, title, why in missing:
        print(f'  ⛔{name}  [{title}]: {why}')
    for m in pin:
        print(f'  {m}')
    if missing or pin:
        print(f'  [FAIL] 누락 {len(missing)} · 값 불일치 {len(pin)}')
        return 1
    if SKIPPED_VALUE_ALTS:
        u = sorted(set(SKIPPED_VALUE_ALTS))
        print(f'  [note] 값 대안 미적용 {len(u)}건(리터럴 우회에 열려 있다): '
              + ', '.join(u))
        print('         ⚠사유 = 같은 리터럴이 매크로 없는 다른 절에도 있다(오탐 공장).')
    print(f'  [OK  ] {len(RULES)} 규칙 중 {len(fired)} 발동 · 누락 0 · 값 핀 일치')
    return 0


if __name__ == '__main__':
    sys.exit(main())
