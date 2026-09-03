#!/usr/bin/env python3
"""★**동반 공시 의무**를 기계로 검사한다 — `gen_paper_constants.py` 의 사각지대.

배경(2026-09-01): 논문 개정에 적대적 감사를 붙였더니 `paper_constants.tex` 의 **수치 핀은
34원자 전건 통과**했는데 **BLOCKER 5건 중 넷과 MAJOR 대부분이 「공시 누락」**이었다.
감사 축자:

> *"생성기는 **수치**만 핀하고 **병기 의무**는 전혀 검사하지 않는다 --- BLOCKER 4·MAJOR 3 이
> 전부 그 사각지대다."*

⟹ 수치가 옳아도 **그 옆에 있어야 할 문장이 없으면** 그 인용은 금지목록 위반이다.
이 스크립트가 그것을 검사한다.

규약
----
`RULES` 의 각 항목은 *"이 매크로(또는 문자열)가 본문에 나타나면, 이 근거 표현들이
**같은 절 안에** 있어야 한다"* 를 뜻한다. 근거 표현은 **여러 대안** 중 하나면 된다
(문면은 바뀔 수 있으므로 의미 단서를 여러 개 둔다).

⛔이 검사기가 통과한다고 공시가 **충분**하다는 뜻은 아니다 --- **누락**만 잡는다.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TEX = os.path.join(ROOT, 'paper', 'waveprior.tex')

# (이름, 발동 정규식, [대안 근거 정규식 …], 사유)
RULES = [
    # ★[2026-09-03 절제 사슬 감사 처방 ①] 이 검사기는 `B22-우변` 규칙을 갖고 있었는데
    #   **사슬의 두 표현 계단(`B-16`·`B-19`)에는 대응 규칙이 0개**였다 ⟹ 방벽이
    #   자기 이름을 딴 실패 모드(`F1`)를 그 두 칸에서 면제하고 있었다.
    ('B16-비교자기술', r'\\BSixteenRel|\\BSixteenBetter',
     [r'\\BSixteenCmpZero', r'comparator'],
     'B-16 크기 인용에는 비교자 기술(38/48) 병기 의무'),
    ('B19-비교자기술', r'\\BNineteenRel|\\BNineteenBetter',
     [r'\\BNineteenCmpZero', r'comparator'],
     'B-19 크기 인용에는 비교자 기술(12/16) 병기 의무'),
    ('B16-두번째계단', r'\\BSixteenDose',
     [r'\\BSixteenDoseCmpZero', r'direction, not a size', r'non-working'],
     'B-16 의 두 번째 계단은 **우변이 무기술**이다 --- 크기가 아니라 방향으로만'),
    # ★[2026-09-02] 적대적 감사 M17: `R1b`·`R4b` 매크로가 규칙 0개로 들어와
    #   **오늘 두 문단이 방벽 밖**이었다. 아침에 만든 기계가 새 배치에 따라오지 않은 것이다.
    #   ⟹ 세 쌍을 여기서 강제한다(전부 **참인 것**만 요구한다 --- D 항 항진 검사 통과).
    ('R1b-다중성', r'\\ROneBShipFixed|\\ROneBP\b',
     [r'sixteen-arm Holm', r'0\.05/16', r'does not cross it'],
     'R-1b 의 p 는 이 아크의 16팔 Holm 임계 0.0031 을 넘지 못한다 --- 병기 의무'),
    ('R1b-클러스터', r'\\ROneBShipFixed|\\ROneBP\b',
     [r'four sites and ten calendar days', r'folded to sites', r'unreachable by construction'],
     'R-1b 부호검정은 12 레코드를 독립으로 다룬다 --- 사이트/일자 접기 병기 의무'),
    ('R4b-비초과', r'\\RFourBTreatPass|\\RFourBTreatMax',
     [r'non-exceedance, not an absence', r'median of its own null'],
     'R-4b 의 0/6 은 **비초과**이지 부재가 아니다 --- 병기 의무'),
    ('R4b-표본율교락', r'\\RFourBTreatPass|\\RFourBTreatMax',
     [r'0\.5~s sample interval', r'self-overlaps more', r'sample interval'],
     'R-4b 는 dt 가 군과 완전 교락한다 --- 그 공시 없이 인용 금지'),
    ('R2b-상한', r'\\RTwoBFloorMin|\\RTwoBFloorMax|\\RTwoBFloorMed',
     [r'\\emph\{upper\} bound', r'upper.{0,3}bound'],
     'R-2b 초과분산은 **상한**이다 --- 병기 의무'),
    ('R2b-얇은여유', r'\\RTwoBFloorMin|\\RTwoBFloorMax',
     [r'thinner than \$0\.03\$', r'\\RTwoBSkewMin', r'undetectable'],
     # ⛔[2026-09-03] 초판은 산문 `"four of the seven"` 을 요구했다 --- 그런데 그 계수는
     #   **어느 동결물에도 없었다**(정규성 문턱을 사후에 고른 수였다). 이제 규칙은
     #   **측정된 범위 매크로**를 요구한다: 문턱 없는 진술이라 사후 선택이 불가능하다.
     'R-2b 는 위치대응 널의 얇은 여유와 **측정된 정규성 범위**를 병기해야 한다'),
    # ★[2026-09-03] `T1` — 검출률과 왜도가 **반대 방향**을 가리킨다(`F1` 부류).
    #   검출률만 인쇄하면 독자는 "저들이 더 선형" 으로 읽는다. 두 수를 같은 절에 강제한다.
    ('T1-검정력아님', r'\\TOneWin',
     [r'\\TOneSkewPct', r'\\TOneNullRatio'],
     'T1 의 검출률을 적으면 **왜도 비(77%)와 귀무 폭 비(3.0배)**를 같은 절에 적어야 한다'),
    # ★[2026-09-03] `T1b` — 용량-반응은 **사전등록에 없다**. 강한 정합 증거라 독자가
    #   확증으로 읽기 쉽다 ⟹ `exploratory` 자구를 같은 절에 강제한다.
    ('T1b-탐색적', r'\\TOneBRho|\\TOneBSkewHi',
     [r'did not pre-register', r'exploratory'],
     'T1b 의 용량-반응을 적으면 **사전등록되지 않았음**을 같은 절에 적어야 한다'),
    ('T1b-사다리', r'\\TOneBWin',
     [r'ladder and not a contradiction', r'about our power'],
     'T1b 를 적으면 **T1 과 모순이 아니라 검정력의 사다리**임을 같은 절에 적어야 한다'),
    ('T1-solver미명시', r'\\TOneWin|\\TOneNMat',
     [r'no documentation naming the solver', r'do not call the corpus higher-order',
      r'undetermined here'],
     'T1 을 인용하면 **solver 미명시**를 같은 절에 적어야 한다'),
    ('T4-비교자붕괴', r'\\TFourRel|\\TFourBetter',
     [r'comparator (?:degrades|collapse)', r'\\TFourCmp\b', r'only \\TFourCmp'],
     'T-4 크기는 비교자 붕괴(47/48 -> 34/48) 병기 없이 인용 금지'),
    ('T4-rung범위', r'\\TFourRel|\\TFourBetter',
     [r'\$11\$ of (?:the )?\$?12', r'11 of the 12', r'lower\b.*\bend of the measured lead'],
     'T-4 rung 은 실측 밴드의 아래쪽 — "실측과 같은 지평" 금지'),
    ('B1-합성단서', r'\\BOneRel|\\BOneBetter',
     # 2차 감사: 'synthetic' 한 토큰이면 통과 = 사실상 항진이었다. 문맥을 요구한다.
     [r'synthetic\)', r'are synthetic', r'gains are synthetic', r'synthetic (?:corpus|seas|evaluation)'],
     'B-1 은 합성·선형 GT 단서 병기 의무'),
    ('B1-체크포인트', r'\\BOneRel|\\BOneBetter',
     [r'same distribution family', r'seed axis only', r'trained on the same'],
     'B-1 은 체크포인트가 학습 분포와 겹친다는 단서 병기 의무'),
    # ★[2026-09-02] 5차 감사의 가장 아픈 지적: 논문이 *"비교자가 붕괴한 상태에서 얻은
    #   마진은 추정기에 대한 증거가 아니다"* 라고 **원칙을 선언해 놓고**, 그 대조가 **없는**
    #   행으로 헤드라인을 낸다. ⟹ `B-1` 인용에 그 두 결함의 병기를 강제한다.
    # ⛔⛔[2026-09-02] 이 규칙이 **거짓을 강제하고 있었다** --- 오늘 아침 `예산-보수적칸`
    #   에서 고친 것과 **똑같은 실패 모드(D 항)**가 다른 규칙에 또 있었다.
    #   `B-1` 의 비교자 기술은 **측정돼 있다**(`S16` 행 전건에 `zero_nRMSE`, 재집계 64/64).
    #   규칙이 *"측정되지 않았다"* 는 자구를 요구하는 바람에 논문이 **자기에게 불리한
    #   거짓**을 유지했다. ⟹ 요구를 **참인 것**으로 바꾼다: 사전등록된 대조가 아니라
    #   **재집계**임을 밝힐 것.
    ('B1-대조는재집계', r'\\BOneRel|\\BOneBetter',
     [r'\\BOneCmp\b', r're-aggregation', r'not part of the design'],
     'B-1 의 음성 대조는 **사전등록된 것이 아니라 재집계**다 — 그 지위를 병기해야 한다'),
    ('B1-사후분석', r'\\BOneRel|\\BOneBetter',
     [r'written after (?:its |the )?results', r'were written after'],
     'B-1 의 게이트·분석은 결과 뒤에 작성됐다 — 병기 의무'),

    ('B1-최강아님', r'\\BOneRel|\\BOneBetter',
     [r'strongest .*we possess', r'strongest eligible', r'not the same claim as'],
     '"최강 적격" 이지 "강화된" 이 아니다'),
    # ★★[2026-09-02] 검증 감사 18: `\BOneRel` 에는 규칙이 여섯인데 **헤드라인이 옮겨간
    #   `\BTwelveRel` 에는 하나도 없었다** --- 그래서 초록이 HOS 등급 단서 없이 나갔다.
    #   ⟹ 헤드라인이 옮겨가면 **공시 의무도 함께 옮겨야 한다**.
    ('B12-실현등급', r'\\BTwelveRel|\\BTwelveBetter',
     [r'\\HosCrowdedDegraded', r'second\s+(?:rather than|order)', r'requested. of the generator'],
     'B-12 의 "HOS M=3" 은 **요청한 것** — 실현 등급(저하 17/48) 병기 의무'),
    # ★[2026-09-02] `W-15` 재집계는 **동결 verdict 에 없는 수**다. 두 가지를 강제한다:
    #   ①재집계임을 밝힐 것 ②비교자의 제로필드 기술을 병기할 것(`F1` 부류 차단).
    ('WFifteen-재집계', r'\\WFifteenSparseBetter|\\WFifteenCrowdedBetter',
     [r're-aggregation', r'do not appear in that batch.s verdict'],
     'W-15 수치는 동결 verdict 에 없다 — **재집계**임을 병기해야 한다'),
    ('WFifteen-음성대조', r'\\WFifteenSparseBetter|\\WFifteenCrowdedBetter',
     [r'\\WFifteenSparseCmpZero', r'comparator is not weak'],
     '비교자가 제로필드를 이기는 비율 병기 의무(F1 부류 차단)'),
    # ⛔⛔[2026-09-02] **실패 모드 D 의 오늘 세 번째** --- 이 규칙은 `L2` 옆에
    #   `\BTwentyTwoAnaZero`(=**15**/48)를 요구했는데, 15/48 은 **`L1a` 의 우변**(`ANA`)이고
    #   `L2` 의 우변은 **`ANA-SHIP` = 32/48 (p=0.0147)** 이다.
    #   ⟹ 검사기가 논문에 **자기에게 불리한 거짓**을 강제하고 있었다(그 병기가 참이었다면
    #     §5.1 의 자기 원칙에 의해 헤드라인 절제가 스스로 인용 불가가 된다).
    #   ★값을 붙들기 위해 원자 이름을 요구한다 --- 자구가 아니라 **원자**다.
    ('B22-우변', r'\\BTwentyTwoRel|\\BTwentyTwoBetter',
     [r'\\BTwentyTwoAnaShipZero'],
     'B-22 L2 의 우변은 ANA-SHIP 이다(32/48) — 그 원자를 병기해야 한다'),
    ('B16-교락폐쇄', r'\\BSixteenRel',
     # 2차 감사: 'confound' 한 토큰이면 통과했다(부정문도 통과). 확인 서술을 요구한다.
     [r'not confounded with', r'checked .{0,40}for confounding', r'not attributable by subtraction'],
     'B-16 크기 귀속은 B-21(교락 무시가능)이 지탱한다 — 병기 의무'),
    # ⚠[2026-09-02] 초판은 `\BSixteenRel` **하나만** 나와도 요구했다. 그것은 과대발동이다 ---
    #   크기 **하나**를 말하는 절은 결합할 수가 없다. ⟹ `MULTI_RULES` 로 옮겼다(아래).
    #   ★검사기가 늑대를 외치면 사람이 무시하기 시작한다(2026-09-01 리터럴 오탐의 교훈).
    # ⚠발동을 **우리 음성 주장**에만 건다(관련연구의 "measured records" 는 대상이 아니다).
    # 발동을 **12 레코드 음성을 서술하는 문장**에만 건다.
    # ⚠[2026-09-02] 근거 패턴 `pre-registered .*threshold` 가 **탐욕 매칭**이라
    #   두 단어가 한 절 어디에 있든 통과했다 = 사실상 항진. ★항진 검사가 잡았다
    #   (내가 다른 뜻으로 "pre-registered threshold" 를 쓰자마자 규칙이 죽었다).
    #   ⟹ **인접 어구**만 인정한다.
    ('실측음성-자구', r'(?:learned prior|representation) (?:is |does )[^.]{0,120}(?:zero-field|measured record)',
     [r'decision threshold', r'pre-registered (?:decision )?threshold'],
     'T-5: "without skill" 금지 -> "유의/결정 문턱에 도달하지 못한다"'),
    # ⚠발동을 **절차를 총괄 서술하는 문장**에만 건다.
    # ⚠[2026-09-02] 트리거가 **너무 좁았다** --- 이 문서에서 한 번도 발동하지 않아
    #   ★항진 검사가 「미검증」으로 보고했고, 확인해 보니 **새로 쓴 전칭 문장이 거짓**이었다
    #   (*"Each row is one pre-registered batch … frozen by checksum"*; B-1 은 사후 작성).
    #   ⟹ 사전등록을 **총괄해 서술하는 문형 전반**으로 넓힌다.
    # ⚠[2026-09-02] 검증 감사 중 이 트리거가 **부정문에 오발동**했다: §5.6 이
    #   *"**not** from a pre-registered contrast"* 라 적었는데(= 규칙이 원하는 정직함
    #   그 자체) 병기 누락으로 잡혔다. ★규칙이 늑대를 외치면 사람이 무시한다.
    #   ⟹ 트리거를 **전칭 한정사**가 붙은 문형으로 좁힌다. 두 번째·세 번째 대안은 그대로.
    ('사전등록-전칭금지',
     r'(?:[Ee]very|[Ee]ach|[Aa]ll|[Bb]oth)\s+(?:comparison|batch|row|experiment)s?\s'
     r'.{0,60}pre-?registered|'
     r'pre-?registration before the runs|analysis code was frozen',
     [r'written after the (?:runs|results)', r'were written after',
      r'[Tt]he exception is', r'two limits of that procedure'],
     '"전부 사전등록됐다" 전칭 금지 — B-1 은 게이트·분석코드가 결과 뒤에 작성됐다'),
    ('MTen-blind', r'\\MTenRel|\\MTenBetter',
     [r'[Nn]ot blind'],
     'M-10 은 blind 아님 — 병기 의무'),

    # ═════════════════════════════════════════════════════════════════════════
    # ★[2026-09-02 P9] 규칙 확장. 표.그림 처분으로 논문이 인용하는 배치가 늘었고,
    #   ⛔그중 `B-26` 은 **동반 의무 2건이 하나도 없는 채로** 인쇄되고 있었다.
    #   (금지목록 08-28 오후 묶음: ①F2 loc 고정 교락 + M2 종결 ②비교자 5% 기술 바닥 미달)
    #   ⚠규칙을 추가할 때마다 **항진 검사**를 하라 --- 위조 문서가 통과하면 규칙이 아니다.
    # ═════════════════════════════════════════════════════════════════════════
    ('B26-F2교락', r'\\BTwentySixRel',
     [r'localisation radius', r'localization radius'],
     'B-26 승수는 F2(loc_radius 4코너 고정) 이력 + M2 종결 병기 없이 인용 금지'),
    ('B26-기술바닥', r'\\BTwentySixRel',
     [r'skill floor'],
     'B-26 승수는 "비교자가 5% 기술 바닥 미달" 병기 없이 인용 금지'),
    ('B26-선형GT', r'\\BTwentySixRel|\\BTwentySixBetter',
     [r'linear[- ]ground[- ]truth', r'linear ground truth'],
     'B-26 "헤드라인이 산다" 는 선형 GT 단서 병기 의무'),

    ('TFive-팔안', r'\\TFiveRel|\\TFiveBetter',
     [r'within-arm', r'remains undetected'],
     'T-5 는 팔 안 대조 — "제로필드를 이겼다" 가 아니다. 끝점 미검출 병기 의무'),

    ('SEns-층한정', r'\\SEnsRel',
     [r'\\SEnsSparseBetter', r'sparse stratum'],
     'S-ENS 혼잡 55/64 는 희소 UNDECIDED 병기 의무'),


    # ★[2026-09-02] 유일하게 살아있는 시간 계수 --- 병기 의무 2건
    ('예산-경로단서', r'\\BudgetUsPerPoint',
     [r'grid-based path', r'not been re-measured'],
     '렌더 단가는 **격자 기반 실측 경로**의 계측 — 학습 렌더러 미측정 병기 의무'),
    ('예산-범위단서', r'\\BudgetUsPerPoint',
     [r'steady state after initialisation', r'proposal stage .{0,30}not in it'],
     '렌더 단가는 **초기화 이후**만 포괄 — 제안 단계 불포함 병기 의무'),

    # ★[2026-09-02] `T6-BUDGET` --- 시간 수치는 쉽게 오독된다. 세 가지를 강제한다.
    # ⛔[2026-09-02] 6차 감사 N16: 이 규칙은 *"most conservative cell"* 이라는 **자구가
    #   있는지**만 봤고 그것이 **참인지**는 보지 않았다. 실측하면 그 칸은 여덟 중
    #   **가장 빠르다**(사전등록의 *사전* 라벨이었고 결과가 지지하지 않았다).
    #   ★검사기가 **거짓을 강제하고 있었다** --- 공시 검사기의 최악의 실패 모드다.
    #   ⟹ 요구를 「사전 지정 칸임 + 최악이 아님 + 조건 명시」로 바꾼다.
    ('예산-보고칸', r'\\FrameMsLearn|\\FramePctLearn',
     [r'reporting cell', r'fixed in advance'],
     '프레임 예산 칸은 **사전등록이 지정한 보고 칸**임을 병기해야 한다'),
    ('예산-최악아님', r'\\FrameMsLearn|\\FramePctLearn',
     [r'fastest of the eight', r'not[^.]{0,40}worst case'],
     '⛔그 칸은 **가장 느린 칸이 아니다** — 보수적 칸으로 읽히지 않게 병기해야 한다'),
    ('예산-실행조건', r'\\FrameMsLearn|\\FramePctLearn',
     [r'CPU-only'],
     # ⚠[2026-09-02] 사유가 코드보다 넓었다(패턴은 `CPU-only` 하나뿐이다).
     #   1스레드·1프로세스는 표 캡션이 지고, 이 규칙은 **본문 절**만 본다.
     '실행 조건 중 CPU 전용을 본문 절에 병기해야 한다(스레드·프로세스는 표 캡션)'),
    ('예산-초기화별도', r'\\FrameMsLearn',
     [r'one-off', r'not part of the per-frame'],
     '초기화는 프레임 수치에 포함되지 않는다 — 병기 의무'),
    ('예산-비교금지', r'\\InitSAna|\\FrameMsLearn',
     [r'separate processes', r'did not run a baseline arm',
      r'no comparative timing claim'],
     '⛔철회된 프로세스간 수치·기준선 지연과 비교 금지 — 병기 의무'),

    # ★[2026-09-02] `T7b` --- 셋 다 사전등록이 요구한 병기다.
    ('스펙트럼-비교자얇음', r'\\SpecBetter|\\SpecP',
     [r'only \$?21\$? of', r'comparator beats a zero field on only',
      r"comparator's weakness"],
     'T7b 는 비교자가 21/32 에서만 제로장을 이긴다 — 병기 의무(F1 부류)'),
    ('스펙트럼-탐색적', r'\\SpecBetter|\\SpecP',
     [r'exploratory and not confirmatory', r'grade this exploratory'],
     'T7b 는 노출 때문에 EXPLORATORY 고정 — 병기 의무'),
    ('스펙트럼-위상없음', r'\\SpecBetter|\\SpecP',
     [r'does not distinguish propagation direction',
      r'symmetric under \$?\\mathbf\{k\}'],
     'directional_r2 는 위상을 보지 않는다 — "스펙트럼 정확도" 일반화 금지 병기'),

    ('실측음성-팔간아님', r'\\BSeventeenBeats|\\BEighteenBeats|\\BFourteenBBeats',
     [r'[Nn]ot a comparison between arms', r'whether the arm has any skill'],
     'B-17/B-18/B-14b 는 제로필드 대비 승수 — 팔간 비교로 읽히면 안 된다'),
    # ★★[2026-09-03] 오늘 새로 들어온 세 배치. 셋 다 **공시 없이는 오독되는 부류**다.
    #   `H3` 는 이 원고의 **가장 큰 자기위협**이라 규칙을 둘 건다(레짐 + 타당도 구분).
    ('H3-레짐', r'\\HThreeMRE|\\HThreeShuffled',
     [r'surf zone', r'exploratory'],
     '⛔`H3` 는 **쇄파대 레짐 한정 · EXPLORATORY** --- 심해로 옮기지 못한다'),
    ('H3-타당도구분', r'\\HThreeMRE',
     [r'same observation model', r'internal comparison', r'external validity'],
     '★`H3` 를 인용하면 **내적(절제는 무손상) / 외적(전이 불가)** 구분을 같은 절에'),
    ('R8-층갈림', r'\\REightBetter|\\REightRatioOut',
     [r'\\REightBigW', r'fails to reproduce', r'stratum'],
     '⛔`R8` 은 **검정력 층(박스 ≥ 40 m)에서 3/5 로 재현되지 않는다** --- 병기 의무'),
    ('R7-교락대조', r'\\RSevenRho',
     [r'\\RSevenCtrlRho', r'uncorrelated with the no-skill'],
     '★`R7` 의 하중은 상관이 아니라 **C-1 대조**(무기술 기준선과 무상관)다'),
    # ★★[2026-09-03] `E1-EXTBASE` --- 라벨 칸은 저들이 관측을 **두 배** 받는다.
    #   예산 규약 없이 인쇄되면 독자가 "우리가 졌다/이겼다" 를 조건 없이 읽는다.
    # ★★[2026-09-03] `E2` — 두 오독 경로가 있다: ①"우리가 낫다" ②"관측만 늘리면 된다".
    ('E2-우열아님', r'\\ETwoDeployBetter|\\ETwoDeployMed',
     [r'does not run our own arm'],
     '⛔`E2` 는 SHIP 을 돌리지 않았다 --- "어느 쪽이 낫다" 로 읽히지 않게 병기'),
    ('E2-참값표본', r'\\ETwoRungTwoHundred',
     [r'sample the ground-truth field rather than a sensor'],
     '⛔사다리 상위 칸은 **참값 표본**이다 --- 얻을 수 있는 관측이라는 주장이 아니다'),
    ('E1-예산규약', r'\\EOneShipWin|\\EOneRung',
     [r'twice our observations'],
     '⛔`E1` 라벨 칸은 제3자가 **관측을 두 배** 받는다 --- 같은 절에 명시'),
    ('E1-동일예산', r'\\EOneShipWin',
     [r'\\EOneEqualShipWin'],
     '★`E1` 라벨을 적는 자리에는 **동일 예산 정면대결(6/12)** 을 같이 적는다'),
    ('E1-비인과', r'\\EOneShipWin|\\EOneSelfFourHundred',
     [r'sees a future frame'],
     '⛔제3자 입력은 t-1,t,t+1 이라 미래 프레임을 본다 = 저들에게 유리 --- 병기 의무'),
    # ★★[2026-09-03] `C1` --- 거리를 충분성으로 읽는 것이 이 수의 유일한 오독 경로다.
    ('C1-충분성아님', r'\\COneCrowdedMed|\\COneCrowdedMax',
     [r'a distance and not a sufficiency'],
     '⛔`C1` 은 **거리**를 잰다 --- "충분히 크다" 로 읽히지 않게 병기'),
    # ★★[2026-09-03] `C1c` 가 `C1` 의 읽기를 좁혔다 — max 노름만 인쇄하면 과대독해된다.
    # ⛔[2026-09-03 정정] 초판은 `\COneCRmsMed`(=**재적합** 잔차)를 요구했다 --- 그것은
    #   **다른 팔**의 값이라, 규칙이 **틀린 짝을 강제**하고 원고가 그대로 인쇄했다.
    #   같은 재구성의 RMS 는 `\COneCStoredMed` 다.
    ('C1-두노름', r'\\COneCrowdedMed|\\COneCrowdedMax',
     [r'\\COneCStoredMed'],
     '⛔`C1` 의 max 노름을 적으면 **같은 재구성의 RMS**(\COneCStoredMed)를 같은 절에'),
    ('C1-재적합구분', r'\\COneCRmsMed',
     [r'\\COneCStoredMed'],
     '⛔재적합 잔차를 적으면 **재적합 전** 값을 같은 절에 --- 둘은 다른 팔이다'),
    ('C1-선형유지', r'\\COneCrowdedMed',
     [r'\\COneCtrlN'],
     '★`C1` 을 인용하면 **선형 코퍼스의 대조(정확히 0)** 를 같은 절에 --- BLOCKER 는 유지된다'),
    ('R7-범위', r'\\RSevenRho',
     [r'upper bound', r'one site', r'exploratory'],
     '⛔`R7` = 상한 · 사이트 1개 · EXPLORATORY --- 셋 다 같은 절에'),
]



# ★[2026-09-02] **결합 위험 규칙** --- 트리거가 "매크로 하나" 가 아니라
#   "서로 다른 크기 매크로 N 개 이상이 같은 절에" 다.
#   사유: 금지목록이 막는 것은 **크기를 말하는 것**이 아니라 **크기를 섞는 것**이다.
#   ⟹ 하나만 나오는 절에까지 병기를 요구하면 오탐이고, 사람이 검사기를 무시하게 된다.
MULTI_RULES = [
    ('크기-결합금지', [r'\\BOneRel', r'\\BTwelveRel', r'\\BSixteenRel', r'\\BNineteenRel',
                  r'\\BTwentyTwoRel', r'\\BFifteenRel', r'\\BTwentySixRel',
                  r'\\MTenRel', r'\\TFourRel'], 2,
     [r'not additive', r'should not add or subtract', r'no defined meaning',
      r'should not be combined arithmetically'],
     '서로 다른 배치의 크기가 같은 절에 둘 이상 있으면 결합 금지 병기 의무'),
    # ★★[2026-09-02] 검증 감사 19: **코퍼스 동일성**을 검사하는 규칙이 하나도 없었고,
    #   그 사각지대가 표 주 (i) 의 거짓("세 행")과 §5.4 의 거짓("각자 자기 코퍼스")을
    #   동시에 통과시켰다. ⚠단 **한 크기만 말하는 절**은 오해할 것이 없으므로
    #   `MULTI_RULES`(둘 이상 동거) 로 둔다 --- 초록까지 요구하면 늑대가 된다.
    ('코퍼스-동일성', [r'\\BTwelveRel', r'\\BSixteenRel', r'\\BTwentyTwoRel',
                 r'\\MTenRel', r'\\TFourRel'], 2,
     [r'\\HosSameRows', r'\\AblationEps\{\} episodes', r're-scorings of one corpus',
      r'run the same', r'different comparison arms'],
     '⛔다섯 행이 **같은 48 에피소드**의 재채점이다 — 독립 실험처럼 읽히지 않게 병기'),
]


# ⚠**입자 공시(2026-09-03 2차 감사)**: 아래 `sections()` 는 `\section`·`\subsection`·
#   초록·하이라이트만 쪼갠다. `\paragraph{}`(48곳)·`\subsubsection{}`·`\section*{}` 은
#   **경계가 아니다** ⟹ 규칙이 말하는 *"같은 절"* 은 실효적으로 *"같은 `\subsection` 안
#   어딘가"* 이고, 가장 큰 절은 문서의 약 16%다. 감사 실증: 근거를 같은 절 안에서
#   **319줄(약 8쪽)** 떨어뜨려도 통과한다.
#   ⛔**입자를 지금 좁히지 않는다** --- `\paragraph` 를 경계로 만들면 지금 통과하는
#   규칙 다수가 깨지고, 그 깨짐이 진짜 결함인지 규약 변경인지 구별하려면 별도 검토가
#   필요하다. ★대신 **이 한계를 여기 적어** *"같은 절"* 을 과대해석하지 않게 한다.
def sections(text):
    """(제목, 본문) 목록.

    절 경계는 `\\section` / `\\subsection`, **그리고 `abstract`·`highlights` 환경**이다.
    ⛔초판은 첫 `\\section` 이전을 한 덩어리(`<preamble>`)로 묶었고, 그래서
    **초록의 공시가 하이라이트의 무공시 인용을 면허**했다(2차 감사 결함 i:
    "highlights 에 비교자붕괴 0인데 rc=0").
    """
    parts, cur, title = [], [], '<frontmatter>'
    for line in text.split('\n'):
        env = re.match(r'\\begin\{(abstract|highlights)\}', line)
        endenv = re.match(r'\\end\{(abstract|highlights)\}', line)
        if re.match(r'\\(sub)?section\{', line) or env:
            parts.append((title, '\n'.join(cur)))
            title = line.strip() if not env else f'<{env.group(1)}>'
            cur = []
        elif endenv:
            parts.append((title, '\n'.join(cur)))
            title, cur = '<frontmatter>', []
        else:
            cur.append(line)
    parts.append((title, '\n'.join(cur)))
    return parts


def _full_body():
    """`waveprior.tex` + 그것이 `\\input` 하는 파일 전부. ⛔조판되는 것이 검사 대상이다.

    ⛔[2026-09-03 2차 감사] 초판은 본문 한 파일만 읽어 `\\input{...}` 안을 전혀 보지
    않았다 --- 그리고 그 안에 이미 `\\CTwoRel` 의 값이 리터럴로 있었다.
    """
    root = open(TEX, encoding='utf-8').read()
    out = [root]
    for m in re.finditer(r'\\input\{([^}]+)\}', root):
        nm = m.group(1)
        for cand in (nm, nm + '.tex'):
            q = os.path.join(os.path.dirname(TEX), cand)
            if not os.path.exists(q):
                continue
            t = open(q, encoding='utf-8').read()
            # ⛔**자동 생성 파일은 제외**한다. 그것들은 수치를 인쇄하는 것이 일이고,
            #   그 수치는 이미 동결 verdict 에서 생성된다 ⟹ 「손타이핑 우회」가 아니다.
            #   ⚠1차 확장이 이것을 빼놓아 **25건이 한꺼번에 떴다**(늑대 외치기).
            #   ★검사 대상은 **손으로 쓰는 산문**이다.
            if '자동 생성' not in t.split('\n', 4)[0] + t.split('\n', 4)[1]:
                out.append(t)
            break
    return '\n'.join(out)


def strip_comments(t):
    """주석을 빼고 **공백을 정규화**한다.

    ⛔초판은 줄바꿈을 그대로 뒀고, 그래서 "the comparator\ndegrades" 같은 정상 문장을
    **공시 누락으로 오판**했다(첫 실행에서 9건 중 여럿이 그것이었다).
    ★검사기가 「있는 것을 없다고」 하는 것도 「없는 것을 있다고」 하는 것만큼 나쁘다.
    """
    out = []
    for line in t.split('\n'):
        if line.lstrip().startswith('%'):
            continue
        out.append(re.sub(r'(?<!\\)%.*$', '', line))
    return re.sub(r'\s+', ' ', ' '.join(out))


def literals_outside_macros():
    """★**매크로 값이 tex 에 리터럴로 나타나는가.**

    3차 감사(2026-09-01)의 최대 구조적 지적:

    > *"매크로를 리터럴 수치로 치환 + 공시 문장 삭제 → 검사기가 `발동 없음` ×10 ·
    > rc=0 · '공시 확인' 을 찍는다. 이것은 가설이 아니다 --- 팔간 주장의 수치가
    > **이미 매크로 밖 리터럴**이라 BLOCKER 전건이 검사기 사각지대에 있었다."*

    ⟹ 공시 규칙은 **매크로 사용**에 걸려 있으므로, 누군가 값을 손으로 적으면 규칙이
    **조용히 침묵**한다. 이 검사가 그 우회를 막는다: `paper_constants.tex` 가 정의하는
    값이 tex 본문에 **그대로** 나타나면 실패한다(그 자리는 매크로를 써야 한다).
    """
    pc = os.path.join(ROOT, 'paper', 'paper_constants.tex')
    if not os.path.exists(pc):
        return True, '상수 파일 없음 — 건너뜀'
    vals = {}
    for line in open(pc, encoding='utf-8'):
        m = re.match(r'\\newcommand\{\\(\w+)\}\{(.+?)\}\s*(?:%|$)', line)
        if m:
            v = m.group(2).strip()
            # ⚠**오탐을 내면 안 된다.** 첫 판은 `10^{-4}`(평활 계수 lambda_s)와
            #   `0.005`(ETKF 관측잡음 sigma)를 잡았다 --- 매크로 값과 **우연히 같은
            #   문자열**인 다른 양이다. ★검사기가 늑대를 외치면 사람이 무시하기 시작한다.
            #   ⟹ **결과 크기**(퍼센트)만 본다: 그것이 공시 의무를 지는 값이고,
            #   손으로 적히면 규칙이 침묵하는 바로 그 값이다.
            if '\\%' in v:
                vals[m.group(1)] = v
    body = strip_comments(_full_body())   # ⛔`\input` 안까지 본다

    # ⚠**우연 일치 면제** --- 매크로 값과 글자가 같지만 **다른 양**인 자리.
    #   ⛔면제는 (매크로, 그 자리의 앞뒤 문맥) 짝으로만 준다. 문맥이 사라지면 다시 잡힌다
    #   ⟹ 새 리터럴 우회는 여전히 실패한다(면제가 구멍이 되지 않는다).
    COINCIDENCE = {
        # `\TOneSkewPct`=77\% 는 T1 의 왜도 비. 아래 자리의 -77% 는 **nRMSE 개선**이다.
        'TOneSkewPct': 'up to $-77\\%$ nRMSE',
    }
    hits = []
    for k, v in vals.items():
        if v not in body:
            continue
        ctx = COINCIDENCE.get(k)
        # ⛔[2026-09-03 2차 감사] 초판 가드는 `count(v) == 1` 이었다. 그것이 안전한 것은
        #   **우연**이었다 --- 현 항목의 문맥 문자열이 값 자신을 포함해서 우회를 넣으면
        #   count 가 2가 됐다. 문맥이 값을 **포함하지 않는** 항목을 넣으면 우회가 통과한다
        #   (감사가 실증). ⟹ ★가드가 검사할 것은 개수가 아니라
        #   ***"그 등장이 문서화된 자리인가"*** 다 ⟹ 문맥이 값을 포함하도록 **요구**하고,
        #   그때 「문맥 안의 그 한 번」을 뺀 나머지가 0 이어야 통과한다.
        if ctx and ctx in body and v in ctx:
            if body.count(v) - ctx.count(v) * body.count(ctx) <= 0:
                continue      # 문서화된 자리 말고는 등장하지 않는다
        hits.append((k, v))
    if hits:
        return False, f'매크로 값이 리터럴로 인쇄됨 {len(hits)}건: ' + \
            ', '.join(f'\\{k}={v}' for k, v in hits[:3])
    return True, f'{len(vals)} 값 검사 · 리터럴 0'


def unused_macros():
    """★생성했는데 **한 번도 쓰이지 않는** 매크로.

    4차 감사(2026-09-02)가 `\\DepLearnedEnsK` 로 이 부류를 지적했다: 생성기가 배포
    상수를 내는데 본문이 그것을 **한 번도 인용하지 않으면**, 그 사실은 보통
    *"적어야 할 것을 안 적었다"* 는 신호다(그때가 정확히 그랬다 --- 배포 앙상블 값이
    본문에 없어서 S-ENS 팔이 배포 구성인지 아닌지 독자가 알 수 없었다).
    ⚠**하드 실패로 만들지 않는다** --- 정당하게 쓰지 않을 값도 있다. 이름을 나열해
    사람이 판단하게 한다. ⛔단 **침묵시키지는 않는다**(그것이 이 저장소의 반복 결함).
    """
    pc = os.path.join(ROOT, 'paper', 'paper_constants.tex')
    if not os.path.exists(pc):
        return []
    names = re.findall(r'\\newcommand\{\\(\w+)\}', open(pc, encoding='utf-8').read())
    body = strip_comments(_full_body())   # ⛔`\input` 안까지 본다
    return [n for n in names
            if not re.search(r'\\' + n + r'(?![A-Za-z])', body)]


def constants_up_to_date():
    """★커밋된 `paper_constants.tex` 가 생성물과 같은가.

    2차 감사: *"생성기를 안 돌리고 tex 만 손으로 고치면 무탐지"* --- CI·훅이 0개다.
    ⟹ 공시 검사기가 그 확인도 함께 한다(한 명령으로 둘 다).
    """
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, 'gen_paper_constants.py'),
                        '--check'], capture_output=True, text=True)
    tail = (r.stdout.strip().split('\n') or [''])[-1]
    return r.returncode == 0, tail


def _eval(name, text, forge_needs=False, forge_trigger=False):
    """(발동했는가, 공시가 있었는가).

    ⚠**위조는 `strip_comments` 이후에 한다.** 초판은 원문에 `re.sub` 했는데, 구절이
    **줄바꿈을 가로지르면** 원문에서는 안 잡히고 정규화 후에는 잡혀서 --- 위조가
    실패하고 항진 검사가 *"이 규칙은 아무것도 막지 않는다"* 로 **오판**했다.
    ★이것은 2026-09-01 에 검사기 본체가 이미 저지른 「줄바꿈 맹목」과 **같은 결함**이다.
    """
    def prep(b):
        if forge_needs:
            for pat in needs:
                b = re.sub(pat, 'XXFORGEDXX', b)
        if forge_trigger:
            b = re.sub(trig, 'XXNOTRIGXX', b)
        return b

    for n, trig, needs, _ in RULES:
        if n != name:
            continue
        fired = False
        for _, body in sections(text):
            b = prep(strip_comments(body))
            if re.search(trig, b):
                fired = True
                if not any(re.search(p, b) for p in needs):
                    return True, False
        return fired, True
    for n, macros, k, needs, _ in MULTI_RULES:
        if n != name:
            continue
        trig = '|'.join(macros)
        fired = False
        for _, body in sections(text):
            b = prep(strip_comments(body))
            if sum(1 for mm in macros if re.search(mm, b)) < k:
                continue
            fired = True
            if not any(re.search(p, b) for p in needs):
                return True, False
        return fired, True
    raise KeyError(name)


def selftest():
    """★**항진 검사** --- 규칙이 규칙인지 확인한다.

    이 저장소가 반복해 배운 것: *"게이트는 통과 가능해야 하고, 위조는 실패해야 한다."*
    (2026-08-30 `T2-STAT` 의 `G-7b` 는 문자열 검색이라 **금지문 자신에 걸려 구조적으로
    항상 실패**했다 --- 어떤 결과로도 라벨이 불가능했다.)

    세 가지를 본다:
      A. **위조**: 근거 표현을 지우면 그 규칙은 **반드시 FAIL** 해야 한다.
         ⛔통과하면 그 규칙은 아무것도 막지 않는다.
      B. **침묵**: 트리거를 지우면 **발동 없음**이 돼야 한다.
         ⛔여전히 발동하면 트리거가 매크로와 무관한 것을 잡고 있다.
      C. **미검증**: 이 문서에서 아예 발동하지 않는 규칙은 A/B 를 시험할 수 없다.
         ⛔이것을 **통과로 세지 마라** --- 세면 죽은 규칙이 조용히 쌓인다.
    """
    raw = open(TEX, encoding='utf-8').read()
    bad, untested = [], []
    names = [r[0] for r in RULES] + [r[0] for r in MULTI_RULES]
    for name in names:
        if not _eval(name, raw)[0]:
            untested.append(name)
            continue
        if not _eval(name, raw, forge_needs=True)[0] or _eval(name, raw, forge_needs=True)[1]:
            bad.append((name, 'A 위조: 근거를 지웠는데 통과했다'))
        if _eval(name, raw, forge_trigger=True)[0]:
            bad.append((name, 'B 침묵: 트리거를 지웠는데 여전히 발동한다'))
    for name, why in bad:
        print(f'  [BAD ] {name:22s} {why}')
    for name in untested:
        print(f'  [--  ] {name:22s} 이 문서에서 발동하지 않는다 (A/B 시험 불가)')
    if bad:
        print(f'\n⛔항진 검사 실패 {len(bad)}건 — 그 규칙들은 아무것도 막지 않는다.')
        return 1
    print(f'✅항진 검사: {len(names) - len(untested)}/{len(names)} 규칙이 위조에 실패하고 '
          f'트리거 없이 침묵한다 (미검증 {len(untested)})')
    return 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    if not os.path.exists(TEX):
        raise SystemExit(f'없다: {TEX}')
    ok_const, msg = constants_up_to_date()
    print(f'  [{"OK  " if ok_const else "FAIL"}] {"상수파일-최신":22s} {msg}')
    ok_lit, msg_lit = literals_outside_macros()
    print(f'  [{"OK  " if ok_lit else "FAIL"}] {"리터럴-우회":22s} {msg_lit}')
    un = unused_macros()
    print(f'  [{"OK  " if not un else "note"}] {"미사용 매크로":22s} '
          + (f'{len(un)}건: ' + ', '.join(un[:5]) + '  ⚠하드 실패 아님 — 사람이 판단하라'
             if un else '0건'))
    raw = open(TEX, encoding='utf-8').read()
    fails = []
    print('── 동반 공시 검사 ─────────────────────────────────────────────')
    for name, trig, needs, why in RULES:
        hit_sections, ok_sections = [], []
        for title, body in sections(raw):
            b = strip_comments(body)
            if not re.search(trig, b):
                continue
            hit_sections.append(title[:52])
            if any(re.search(p, b) for p in needs):
                ok_sections.append(title[:52])
        if not hit_sections:
            print(f'  [--  ] {name:22s} (발동 없음)')
            continue
        missing = [t for t in hit_sections if t not in ok_sections]
        if missing:
            fails.append((name, why, missing))
            print(f'  [FAIL] {name:22s} 공시 누락 절 {len(missing)}: {missing[:2]}')
        else:
            print(f'  [OK  ] {name:22s} 발동 {len(hit_sections)}절 전건 공시')
    for name, macros, k, needs, why in MULTI_RULES:
        hit, missing = [], []
        for title, body in sections(raw):
            b = strip_comments(body)
            n_distinct = sum(1 for m in macros if re.search(m, b))
            if n_distinct < k:
                continue
            hit.append(title[:52])
            if not any(re.search(pat, b) for pat in needs):
                missing.append(f'{title[:44]} (매크로 {n_distinct}종)')
        if not hit:
            print(f'  [--  ] {name:22s} (발동 없음)')
        elif missing:
            fails.append((name, why, missing))
            print(f'  [FAIL] {name:22s} 공시 누락 절 {len(missing)}: {missing[:2]}')
        else:
            print(f'  [OK  ] {name:22s} 발동 {len(hit)}절 전건 공시')
    print('──────────────────────────────────────────────────────────────')
    if not ok_const:
        fails.append(('상수파일-최신', 'paper_constants.tex 가 생성물과 다르다', ['<repo>']))
    if not ok_lit:
        fails.append(('리터럴-우회', '매크로 값을 손으로 적으면 공시 규칙이 침묵한다',
                      ['<paper/waveprior.tex>']))
    if fails:
        print(f'\n⛔실패 {len(fails)}건:')
        for n, why, ms in fails:
            print(f'  · {n}: {why}')
            for m in ms:
                print(f'      절: {m}')
        print('\n⚠수치가 옳아도 병기가 없으면 그 인용은 금지목록 위반이다.')
        return 1
    print('✅발동한 모든 절에서 동반 공시 확인')
    return 0


if __name__ == '__main__':
    sys.exit(main())
