# 사전등록 체크리스트 — 새 배치가 결과 0행 시점에 만족해야 할 것 (2026-09-04)

> **왜 이 문서가 있나.** `T11-EXTACC`(`18216fc`·`7f6debc`)에서 **같은 부류의 결함이 하루에 네 번** 났고, 셋 다
> 사람이 아니라 기계·감사 에이전트가 잡았다(`tmp/T11_EXTACC_2026-09-03/AMENDMENT_2_PREDATA.md` = **A2**,
> `AMENDMENT_3_POSTRESULT.md` = **A3**, `AMENDMENT_1_PREDATA.md` = **A1**). 이 문서는 그 사건들과 `CLAUDE.md`
> 「측정 규율」·`memory/` 의 반복 교훈을 **한 배치의 사전등록이 통과해야 할 항목**으로 접는다.
> 기계로 볼 수 있는 항목은 **`tmp/_audit/check_prereg.py`** 가 검사하고(항목 ID `C1`–`C19` 이 그 스크립트의 출력 ID),
> 못 보는 항목은 **`M-`** 로 표시해 사람이 본다.
>
> **열람 범위 = 주장 범위.** 이 문서가 인용하는 근거는 작성자가 실제로 연 것뿐이다: `CLAUDE.md`(전문) ·
> `memory/MEMORY.md`(색인 머리 + 꼬리 파일의 1–61·152–245행) · T11 의 `PREREG.md`·A1·A2·A3·`gates_t11.py`·
> `analyze_t11.py`·`run_t11.py`(현재판 + `18216fc^`·`18216fc` 판) · `tmp/_audit/` 의 `check_batch_artifacts.py`(전문)
> · `check_gate_coverage.py`(전문) · `check_gate_refs.py`·`check_gate_blindspots.py`·`check_gen_vs_gates.py`·
> `check_verdict_hygiene.py`(헤더) · `run_all.sh` · `DISCIPLINE_gen_gates`·`DISCIPLINE_verdict_labels`·
> `DISCIPLINE_seed_axes`(앞 80행) · feedback 메모 12편(`prereg_needs_power_under_its_own_prediction` ·
> `checker_must_pass_tautology_test` · `gate_must_be_passable` · `verify_the_fix_by_mutation` ·
> `power_table_needs_treatment_null` · `prespecify_aggregation_and_direction` · `never_mutate_tree_during_a_batch` ·
> `a_repair_creates_a_new_defect` · `detection_rate_needs_null` · `relative_margin_fails_near_zero` ·
> `control_stratifying_variable` · `smoke_before_long_runs` · `comments_count_as_N`).
> `NOTE.md` 는 `G-ZERO` 언급 여부만 grep 했다 — 그 밖의 내용은 인용하지 않는다.

## 0. 언제 무엇을 돌리나

| 시점 | 명령 | 이 문서의 항목 |
|---|---|---|
| **결과 0행** (PREREG·gates·analyze 를 쓴 직후, 러너 **전**) | `check_prereg.py <배치>` · `check_gate_refs.py <배치>` · `check_gate_blindspots.py <배치>`(합성 행 훅이 있으면) | A·B·D 전부 |
| **스모크 직후** (2–4 케이스) | 같은 셋 + `check_batch_artifacts.py <배치>` | C · A10 · A11 |
| **결과 이후** | `check_batch_artifacts.py` · `check_gate_coverage.py` · `check_verdict_hygiene.py` · `check_prereg.py`(C9·C10 이 N−1 을 잡는다) | E |

⛔실행은 항상 `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- python3 …` (A2 §0: 09-04 에 OOM 으로 배포판이 두 번 죽었다).
⛔`check_prereg.py` 의 `N/A` 는 **미검증**이다 — 통과로 세지 마라(feedback_checker_must_pass_tautology_test **C**).

---

## A. 사전등록 문서

### A1 · PRIMARY 통계량 · 방향 · 집계 · n 을 이름으로 못 박는다 — `C15`
- **요구**: `PRIMARY` 라는 낱말로 통계량 하나를 지정하고, **열화 방향**(`<`/`>`), **집계 함수**(짝별 차의 중앙값인가 중앙값의 차인가 — 둘은 다른 양), **n** 을 같은 절에 적는다. 부차 통계량은 `SEC-A`… 로 이름 붙인다.
- **왜**: feedback_prespecify_aggregation_and_direction — 2026-08-05 확산 교차설계에서 방향·집계·검정력·층 넷이 **전부** 물렸다(방향 규약 하나로 p 0.037↔0.284). `CLAUDE.md` 「집계·방향·MDE 를 결과 전에 동결」.
- **확인**: `C15`(키워드 부재 = FAIL, 방향/n 부재 = WARN). **집계 함수의 종류**는 기계가 못 본다 → 수동.

### A2 · MDE(또는 q80·등가한계)를 **수치로** 적고 gates 상수로 올린다 — `C1`
- **요구**: PRIMARY 의 검정에서 검정력 0.80 의 최소 검출 효과(`q80=0.760` 꼴)와 **기각 불가 대역**(예: `w=13..16`)을 적고, `gates_*.py` 에 `MDE_Q80` 같은 상수로 두어 verdict 에 실린다.
- **왜**: A2 §3 — T11 배치 전체에 `MDE`·`등가`·`q80` 문자열이 **0회**였다. `CLAUDE.md` `M8b`: MDE 두 값(가족 0.931 / PRIMARY 0.869)을 병기하되 섞지 말 것. `T5-ENS` MDE 0.742 · `S-ENS` q80 0.682 가 본보기.
- **확인**: `C1`. 층별 MDE(`n=12` 면 10/12 = 사실상 검정력 없음, A2 §3)는 수동으로 한 줄 적는다.
- ★★**[2026-09-05 신설] 계산은 `tmp/_audit/sign_test_mde.py` 하나로만 한다.** ⛔직접 이분법을 쓰거나
  다른 배치의 수를 옮겨 적지 마라 — 09-05 수치 정합 감사가 **미달 22건**을 찾았고 부류가 둘이었다:
  (a) `round()` 내림이 검정력을 문턱 **아래**로 떨궜다(`T14 primary_q80 0.7630` → 실제 0.79986,
  `T12 sec_q80 0.8332` → 0.79995), (b) **같은 양이 배치마다 다르게 인쇄**됐다
  (`crit 19 @ n=28` = `T12`·`T14` 0.7304 vs `T13` 0.7305). ★후자는 이 저장소의 이름 붙은 결함
  **「손복사 기준 상수」**이고, 하필 *모든 음성 결과가 병기해야 하는 상수* 안에서 재발했다.
  ⚠최악은 `T7b` — `analyze_t7b.py:38` 이 `q80` **만** 리터럴로 적었는데 그 값이 **`T7`(n=48)의 것**이라
  실제 검정력이 **0.5495** 였다(같은 줄의 나머지 넷은 전부 계산됐다).
  - **규약**: ***MDE 를 참값보다 낮게 보고하지 않는다*** — 효과크기 `|q−0.5|` 를 0.5 에서 **멀어지는**
    쪽으로 반올림한다(이김 다리 올림 · **해침 다리 내림**). 반환값 자신이 목표 검정력을 달성한다.
  - **쓰는 법**: `from sign_test_mde import mde_q, k_one_sided, k_two_sided` 후
    `Q80 = mde_q(N, THRESH, 0.80)` (해침 다리는 `side='lower'`). ⛔반환값을 다시 반올림하지 마라.
  - ★**`verdict` 의 `mde` 블록에 `n`·`thresh`·`power` 를 같이 실어라** — 그러면
    `tmp/_audit/check_mde_constants.py` 가 **자기기술**로 자동 검사하고 등록이 필요 없다.
    싣지 않으면 그 배치는 `UNREGISTERED` 로 래칫에 걸려 `check_all` 을 빨간불로 만든다.

### A3 · 결과 전 예측을 **라벨별 확률 표**로 적고, 확률은 **코드가** 계산한다 — `C2`·`C2b`
- **요구**: 예측 `q`(팔별)를 명시하고, `gates.label()` 이 낼 수 있는 **모든** 라벨에 대해 그 `q` 하의 확률과 귀무(0.5) 하의 오라벨률을 표로 싣는다. 계산 함수(`label_probs`/`power_at`)를 `gates_*.py` 에 두고 verdict 의 `prediction` 에 기록한다. 합이 1 이어야 하고(`UNDECIDED` 포함), 라벨 둘을 하나로 뭉개지 않는다.
- **왜**: A2 §2 — v2 PREREG 는 한 팔 검정력 표(폐기된 v1 표와 축자 동일)만 있었고 선언값(0.65/0.20/0.10/0.05)은 **단일 q 이항모형으로 동시에 성립 불가**였다. feedback_prereg_needs_power — `T5-ENS` v1 은 자기 예측 하에서 **87.5% 확률로 라벨을 못 내는 설계**였고 문서 안에 그것을 알려주는 수가 없었다. A3 §5 — `NEITHER-DETECTED 0.2136` 이 실제로는 음성 라벨 둘의 **합**이었다.
- **확인**: `C2`(라벨마다 확률 존재) · `C2b`(계산 함수 존재). **산술 정합**(단일 q 로 동시 성립?)은 기계가 못 본다 → 수동: 함수를 돌려 표를 재생산한다.

### A4 · 처치 코퍼스의 **귀무 폭**을 적는다 — `M-A4`
- **요구**: 대조 코퍼스에서 잰 자를 처치에 그대로 옮기지 말고, 처치 자료의 귀무 폭을 **처치와 무관한 축**에서 추정해 적는다(모르면 「모른다」).
- **왜**: feedback_power_table_needs_treatment_null — `T1` 에서 제3자 귀무가 3.0배 넓어 검출 문턱이 **1.7배** 모자랐다(빗나감의 상당 부분이 사전확률이 아니라 표본 크기). `T1b` 가 처음 지켰다(24→502 프레임 0.30배).
- **확인**: 수동.

### A5 · 「기각 실패」 다리에 **양성 이름**의 라벨을 두지 않는다 · 이김 문턱이 α 를 지킨다 — `C4`
- **요구**: 라벨 사다리에서 검정은 **이김 다리**(`w ≥ k`, `P(W≥k|.5) ≤ α`)뿐이다. 그 밖의 칸은 이름 자체가 「미검출」이어야 한다(`NEITHER-DETECTED`·`NO-INFORMATION-IN-OBS`). `≤k` 로 도달하는 라벨에 *"…이 아니다/도달 불가"* 같은 양성 주장을 붙이지 않는다. 대칭화(`w≤7`)로 「질문을 바꾸는」 처방도 금지.
- **왜**: A2 §1 — `THRESH_LOSE=13` 은 귀무 아래 **0.5806** 으로 발화하는데 PREREG v2 §2 가 그것을 `BEYOND-REACH`(*"대역 말고 다른 것"*)이라는 **양성 주장**으로 정의했다. 대칭화 처방은 `BAND-MISMATCH` 의 도달 확률을 0.0245 로 만들어 기각. `DISCIPLINE_verdict_labels` — `M1` 의 접기 `stt()` 가 `BASELINE-NO-SKILL` 을 `'break'` 로 접어 비교 불능이 최강 귀속으로 둔갑할 수 있었다. `T5-ENS` 개정문 — *"「검출 불가」이지 폐쇄가 아니다"*.
- **확인**: `C4`(a) 사전등록 라벨 표의 `≤k`/`≥k` 를 이항 귀무로 계산(**부호검정을 선언한 배치에만**) · `C4`(b) `gates.label()` 의 각 반환 경로에 `≥` 항이 있는가(AST). 접기 함수(`stt()`류)의 정확 일치는 수동 + 「무기술 팔」 픽스처(`DISCIPLINE_verdict_labels` P2).

### A6 · 문서의 라벨 집합 = 코드의 라벨 집합 — `C3`
- **요구**: `gates.label()` 이 반환하는 모든 라벨이 사전등록 문서(개정문 포함)에 있고, 표의 라벨을 코드가 낼 수 있다. 개정으로 라벨을 바꾸면 옛 표는 두되 개정문이 그 이름을 언급한다.
- **왜**: `18216fc^` 상태 — PREREG 는 v2(`BAND-MISMATCH`…)인데 `gates_t11.py` 는 v1(`EXTACC-*`)이었다. `PREREG.md` 는 지금도 옛 표를 싣고 A2 가 덮어쓴다(*"철회는 삭제가 아니라 이동"*).
- **확인**: `C3`.

### A7 · 다중비교를 **보정하거나 무보정임을 공시**한다 — `C13`
- **요구**: 부차 통계량이 둘 이상이면 Holm/Bonferroni 를 적용하든지, *"무보정 · 층과 클러스터는 기술통계이고 라벨을 만들지 않는다"* 를 `disclosures` 에 기계로 넣는다. 클러스터 바닥(n=6 만장일치 P=0.0156)을 적는다.
- **왜**: A2 §4 — 통계량 7개인데 Holm/Bonferroni 가 코드·문서에 **0 hits**. `CLAUDE.md` 08-12 감사 — 100ep 코퍼스 클러스터 바닥 p=0.031.
- **확인**: `C13`.

### A8 · blind/노출 상태와 **노출이 결정하는 양**을 적는다 — `C14`
- **요구**: 완전 blind 인가, 어느 파일·셀이 설계에 영향을 줬는가, 그것들을 표본에서 배제했는가(게이트로), 그리고 노출된 양이 확증 대상과 얼마나 상관인가를 **동결 자료에서 계산해** 적는다. 스모크 후 게이트를 바꿨으면 결과 0행에서 **무엇을 왜** 바꿨는지 개정문에 박는다.
- **왜**: PREREG v2 §6(*"완전 blind 가 아니다"*), feedback_prereg_needs_power 3항 — `T5-ENS` lead0↔NEAR **r=0.864** 를 초판이 공시하지 않았다. A2 §9 — 스모크 후 `G-CLAMP`·`G-TORIGIN` 변경 공시(선례 `M4 G-15`).
- **확인**: `C14`(언급 존재만). 상관 계산·배제 게이트 존재는 수동(`G-SAMPLE` 부류).

### A9 · 시드 **축**을 이름으로 적고 추정기 시드 수를 밝힌다 — `C16`
- **요구**: 학습 시드 / 추정 시드 / 생성 시드 중 어느 것을 흔드는지, 추정기 시드가 몇 개인지(median-of-k 인지) 적는다. 체크포인트를 쓰는 팔은 `ckpt_sha256` 열을 행에 싣는다.
- **왜**: `DISCIPLINE_seed_axes` P0·P1 — 세 축을 「시드」 한 낱말로 뭉치지 마라; 노출 판별은 *"두 팔이 다른 `.pt` 를 실었는가"* 하나. `CLAUDE.md` — S13 은 시드 1개(42)로 돌았고 PREREG 에 언급이 없었다; S13b 가 부차 주장 하나를 죽였다(시드 간 CV 혼잡 14.4%).
- **확인**: `C16`(언급만) · `C7s`(지문 줄). 축 이름·개수는 수동.

### A10 · 스모크 계획: **가장 비싼 팔** + **게이트가 발화할 셀** — `C11`
- **요구**: 긴 배치 전에 몇 케이스를 돌리되, 노출 표본만 쓰고(PRIMARY 표본 미접촉), 가장 비싼 팔과 각 게이트가 발화할 수 있는 셀을 포함한다. 단가 문턱(*"추정의 2배 이내"*)을 미리 적는다.
- **왜**: feedback_smoke_before_long_runs — 하루에 같은 부류 세 번(8샤드·Phase A 건너뜀·스레드 미고정, 18시간). `M3` 스모크 1셀이라 `G-17` 이 288런 전에 구조적으로 발화 불가. A2 §9 — 스모크가 §6c 예측을 **반증**하고 `t_origin` 회귀를 잡았다.
- **확인**: `C11`(흔적만). 셀 선택은 수동.

### A11 · 개입·오류의 효과는 **판정에서** 잰다, 그것이 들어가는 양에서 재지 않는다 — `M-A11`
- **요구**: *"X 가 2배니까 Y 게이트가 걸린다"* 류 예측을 적었으면 스모크에서 **그 게이트의 발화**로 확인한다. 값이 바뀌면 그 값을 쓰는 판정을 재실행해 verdict 을 diff 한다.
- **왜**: A2 §9.1 — *"경사가 2배니 RESCALE 이 클램프에 걸린다"* 가 실측에서 **정반대**(SHIP 발화). `CLAUDE.md` `B-2` 수심 17→30 — 분산관계에서는 무해(Δω 0.0012%)였는데 하류 Fisher p 0.0265→0.1591 로 널이 됐다.
- **확인**: 수동(본보기 `tmp/B2_REAL_2026-08-23/depth_recheck.py`).

### A12 · 여유(margin)는 그 층의 **절대 오차 규모**에서 정한다 — `M-A12`
- **요구**: 비열등성·무해 여유를 상대 % 로 등록하기 전에 동결 자료에서 층의 전형적 절대 오차를 보고, 오차가 10⁻² 규모면 절대 여유(또는 둘 중 느슨한 쪽)로 등록한다. 결과를 본 뒤 바꾸지 않는다.
- **왜**: feedback_relative_margin_fails_near_zero — `T8` 희소 층 nRMSE 0.0078→0.0116 이 +49% 로 「해쳤다」가 됐다.
- **확인**: 수동.

### A13 · 판정 통계량의 **부호 퇴화**를 결과 전에 점검한다 — `M-A13`
- **요구**: *"두 팔이 같은 방향으로 실패하면 이 통계량은 무엇을 뜻하는가"* 에 답을 적는다. 팔끼리 비교하면 구동 크기를 정합시킨다. 에피소드 독립성을 생성기 파라미터 수준에서 선언한다.
- **왜**: feedback_prespecify 5–7항 — `Rx-4` 의 `|Δ|` 비교가 *"더 크게 망친 팔이 이기는 검정"* 이 됐다; 위상 벡터 8개·진폭 4개뿐인 16 에피소드.
- **확인**: 수동.

### A14 · 층·셸의 **접두어**를 적고 pooled 상관을 기제로 승격하지 않는다 — `M-A14`
- **요구**: 셸(`obs:`)과 배-거리 고리(`ship:`)는 다른 양이다 — 어느 쪽을 쟀는지 접두어까지 적는다. 층을 정의하는 변수를 통제하지 않은 상관을 기제로 쓰지 않는다. 팔별 필드 존재를 PREREG 게이트로 검사한다.
- **왜**: `CLAUDE.md` `SUPPORT-TRADEOFF`(S13 이 접두어에서 사전등록을 이탈, 같은 표본에서 1.8–4.1배 갈린다) · feedback_control_stratifying_variable(누적 5회).
- **확인**: 수동(본보기 `S15 G-8/G-8b`).

### A15 · 팔은 빈 dict 로 정의하지 않고, 정체는 **되읽기**로 기록한다 — `C12` + `check_batch_artifacts` J
- **요구**: `'ARM': {}` 금지(런타임 기본값 스냅샷). 러너는 `upd.arm_provenance()` 를 행에 넣고 게이트가 `resolved_init_method`·`real_domain_gate_fired`·`ensemble_k` 를 단언한다(`requested` 와 `resolved` 가 다를 수 있다는 것이 요점).
- **왜**: `CLAUDE.md` — `gates_b7.py:46` 의 `{}` 가 당일 철회된 값으로 돌았다; `E1-EXTBASE` 가 실측에서 빈 팔이 `bank_mmse` 로 라우팅되는 것을 모르고 두 원고 6곳에 인쇄했다. A1 §5 — `G-ARM` 이 출하 팔과 「제3의 팔」을 구별 못 했다(`is_ensemble_wrapper`·`ensemble_k` 동반 단언 필요).
- **확인**: `C12` · `check_batch_artifacts` J 열.

---

## B. 게이트 · 분석기

### B1 · 사전등록 표의 게이트가 **전부** 구현돼 있다 — `C8` (+ `check_gate_coverage.py`)
- **왜**: `check_gate_coverage.py` 헤더 — `R1b` 가 `G-FREEZE`·`G-BLOCK` 을 구현 없이 `13/13` 을 인쇄했고, 그 검사기를 만든 한 시간 뒤 `R3b` 가 재발. `18216fc^` 에서 v2 표의 게이트 6개(`G-ATOMS`·`G-CAP`·`G-WRAPPER`·`G-TORIGIN`·`G-SAMPLE`·`G-RESCALE`)가 미구현.
- **확인**: `C8`(소스, 결과 0행) · `check_gate_coverage.py`(verdict, 결과 이후).

### B2 · **장식 게이트** 0 — 모든 게이트 이름이 `blocked_by()` 에 잡힌다 — `C5`, `C18`
- **요구**: 분석기가 만드는 게이트 이름 각각을 실제 `blocked_by([(name, False, '')])` 에 넣어 비지 않음을 확인한다(긴 이름 우선 접두 규약). 라벨 인쇄는 `blocked` **와** `all(ok)` 둘에 매단다.
- **왜**: A2 §7 — `G-COV` 가 차단 목록에 없어 장식. MEMORY 09-01 — `T5-ENS` 의 `blocked_by()` 가 이름 전체를 비교해 **항상 빈 리스트** = 차단 게이트 14개 전부 장식 직전. feedback_a_repair — 같은 버그가 `T7` 에서 재발. A3 §6 — V3 가 `blocked_by` 를 무력화하니 `G-BLOCK` FAIL 인데 라벨이 인쇄됐다.
- **확인**: `C5`(live 호출, 실패 시 AST 폴백) · `C18`(`blocked` 매달림만; `all(ok)` 벨트는 수동).

### B3 · 게이트 조건이 러너 **기록값의 존재/불리언** 그대로이면 항진이다 — `C6b`
- **왜**: A1 §4 — `G-SIZE`(`meta['size_ok']` 러너 리터럴) · `G-DISJOINT`(`not meta.get('disjoint')` = 키 없으면 통과) · `G-ENV`(`bool(meta.get('code_md5'))` = 아무 문자열이나 통과). 처방 = 분석기가 **직접 계산**(`os.path.getsize`, namelist 직접 개봉, md5 항목별 재계산).
- **확인**: `C6b`.

### B4 · 러너가 `k=G.C` 를 쓰고 분석기가 `k == G.C` 를 비교하면 항진이다 — `C6a`
- **왜**: A2 §7 — `G-ENV` 3/3 항진(`threads=1` 리터럴 왕복 · `seed=G.SEED` 상수 왕복 · `bool()`); A3 §2 — 개정문 2 가 **셋 중 둘만** 고쳐 `seed` 왕복이 남았다. 처방 = **행의 시드 집합을 되읽어** 대조(러너가 블록마다 다르게 쓴 값).
- **확인**: `C6a`(문법적 동일성만 — `LI.M_OBS_CAP` ↔ `G.deployed_cap()` 같은 다른 경로는 못 본다).

### B4b · **라벨 이름에 `G-` 를 넣지 않는다** — `C19` (2026-09-05 신설)
- **요구**: `gates_*.py` 의 `label()` 이 내는 문자열과 사전등록 라벨 표의 라벨에 `G-` **부분문자열**이 없어야 한다. 자연스러운 이름이 걸리면 개명한다(`…-LANDING-HURTS` → `…-WORSENS-LANDING`, `SUBADDITIVE-CEILING-CONFOUNDED` → `SUBADDITIVE-AT-CEILING`).
- **왜**: 게이트 이름 파서(`check_prereg.GATE_RE` · `check_gate_coverage._NAME`)가 `G-` 를 찾는데, 라벨 안쪽에서도 물어 **없는 게이트**를 만든다 — `LANDIN**G-H**URTS` → `G-HURTS`. 유령이 「등록됐는데 미구현」 목록에 앉으면 ***진짜 검사가 가짜 사유로 죽거나, 진짜 누락이 유령에 묻힌다***. 2026-09-05 하루에 **세 번**(T16 `G-HURTS` · T17 `G-CHANGE-DETECTED`/`G-WORSE` · T18 `G-CONFOUNDED`), 전부 사람·에이전트가 읽어서 잡았다 ⟹ 주의력은 처방이 아니다.
- ⚠**파서도 함께 고쳤다**(경계 `(?<![0-9A-Za-z_])`) — 그러나 그것은 **읽는 쪽**의 방어다. 라벨은 산문·다른 도구·사람에게도 그대로 보이므로 **저작 자리**에서 막는다(N곳 전부).
- **동결 배치의 위반은 소견이지 수정 대상이 아니다**(라벨은 증거다): `B18`(`…-BINDING-CONSTRAINT`) · `M2`(`TUNING-HURTS`·`TUNING-NULL`·`TUNING-SHRINKS-MARGIN`) · `T13`(`DEPTH-WAS-MASKING-SKILL`).
- **확인**: `C19`. ⚠사각지대 = `G-` 로 **시작**하는 라벨(게이트 참조와 구별 불가) · 라벨 표가 없는 배치는 `N/A`.

### B5 · 게이트가 보는 양은 **만들어진 객체에서 되읽는다** — `M-B5`
- **요구**: `gt_std`·`t_origin`·`disp_relerr`·`n_consumed` 처럼 러너가 리터럴/산식으로 적을 수 있는 값은 프레임·모델·환경에서 되읽고(`readback_frames`, `initialize()` **뒤**에 한 번 더), 산식이면 이름에 `_formula` 를 박아 게이트에서 뺀다. 「변화 탐지기」(소스를 안 고치면 실패 불가)는 그렇게 **이름에 표기**한다.
- **왜**: `run_t11.py` 수리 1 — 초판이 네 값을 리터럴로 적어 게이트 넷이 항진. A2 §7 `G-CAP` — `n_consumed=min(제공, 캡)` 이 분석기와 같은 식이라 발화 불가. A3 §9·§10 — `G-WRAPPER` 는 배포 상수상 구조적으로 참(이름에 표기), `G-TORIGIN` 은 키 부재 탐지로 실제 값을 했다.
- **확인**: 코드 리뷰(수동). `C6a`·`C6b`·`C6c` 가 부분 대리.

### B6 · 게이트가 **통과할 수 있는가** — `M-B6`
- **요구**: 금지 토큰은 문자열 검색이 아니라 AST 로(독스트링·주석 통과), 게이트마다 「현 상태 통과 + 위조 입력 실패」 항진 검사를 배치 selftest 에 넣는다. 러너 쪽 사전 확인은 게이트 함수를 호출한다(2중 구현 금지).
- **왜**: feedback_gate_must_be_passable — `T2 G-7b` 가 금지문 자신에 걸려 **어떤 결과로도 라벨을 낼 수 없었다**. feedback_checker_must_pass_tautology_test A/B/C/D.
- **확인**: 배치 `selftest()`(수동 확인). `C4`(b)·`C5` 가 부분 대리.

### B7 · 게이트가 **자료를 볼 수 있는가** — 외부 키 존재 · 팔 필터 항진 — `M-B7`
- **왜**: `check_gate_refs.py` — `M2 G-13` 이 `B-26` 동결 행에 없는 키를 읽어 자료에 대고는 통과 불가능. `check_gate_blindspots.py` — `M4 G-13` 은 키가 있는데 러너 스탬프 `pc=True` 가 필터를 항진으로 만들어 `0/12` 항등 실패. feedback_prereg_needs_power 4항 — `T5-ENS G-SCENE` 이 `T3` 행에 없는 `depth` 를 읽어 `M8` 재연 직전.
- **확인**: 두 검사기(결과 0행에 돌려야 값이 크다).

### B8 · 개입 기대값·문턱·기준 상수는 **단일출처** — `M-B8`
- **왜**: `CLAUDE.md` — `W4` 에서 개입 기대값 320/820 이 네 곳(러너 G-2·G-3 · `arms_config` · `watchdog`)에 중복돼 하루 네 번 재현; λ 3중 기재. A2 §7 — 러너 `9.80665` 손복사 vs 배포 `dispersion.G=9.81`, `CAP_PROPOSAL=400` 손복사 → `deployed_cap()` 임포트. feedback_verify 4항 — 시험 수 방벽은 일치만 봐서 사고 삭제가 다시 통과 ⟹ 래칫도.
- **확인**: `find_handcopied_refs.py` + 코드 리뷰.

### B9 · 접기 함수와 읽기 표의 **의미론 정합** — `M-B9`
- **왜**: `DISCIPLINE_verdict_labels` — `M1 stt()` 가 `BASELINE-NO-SKILL` 을 `'break'` 로 접어 `('hold','break')` = *"비선형성이 매개한다"* 최강 귀속이 출력될 수 있었다(M1 에서는 미발화). 본보기 `M3 st_()` 정확 일치 + 폴스루 미귀속.
- **확인**: 수동 + 「무기술 팔」 픽스처 1건(P2).

### B10 · 성분+창의 순함수인 술어는 `screen()` 에 — `M-B10`
- **왜**: `DISCIPLINE_gen_gates` P1 — `M3 G-8` 위반이 로그에 찍힌 뒤 46분을 더 돌았다; 술어가 생성 전에 **정확히, 공짜로** 계산되는 것이었다. 사후 백스톱 = `check_gen_vs_gates.py`.
- **확인**: 수동(생성 배치에만).

### B11 · 자기진단 통과 ≠ 판정 불변 — `M-B11`
- **왜**: `CLAUDE.md` — `amplitude_refit` 위상 부호(자기진단 `residual_after` 가 올바른 계수로 계산돼 두 규약에서 비트 동일 = *"진단은 통과하고 결과만 틀린다"*) · `B-2` 수심. `DISCIPLINE_verdict_labels` — `M1 pc1` 은 발화만 검사하고 읽기 의미론은 검사하지 않아 결함이 통과.
- **확인**: 수동 — 값이 바뀌면 판정을 diff.

---

## C. 러너

### C-a · 분석기가 읽는 키를 러너가 **팔마다** 쓴다 — `C17`
- **왜**: A2 §9.2 — 개정 러너가 `ZERO`·`ORACLE` 행에서 `t_origin` 을 떨어뜨려 4/8 행 `None`. **게이트(`G-TORIGIN`)가 잡았다.** `C17` 은 합집합만 보므로 팔 단위 누락은 못 본다 — 그래서 배치 게이트가 키 **부재**를 위반으로 세야 한다.
- **확인**: `C17` + 배치 게이트(부재 = 위반).

### C-b · 같은 시점 같은 식을 두 키로 기록하지 않는다 — `C6c`
- **왜**: A3 §0 — `threads`·`threads_readback` 이 같은 호출이고 주석 *"런 종료 후 재측정"* 은 거짓. 항진은 아니나 **하나를 둘로 센 것**.
- **확인**: `C6c`.

### C-c · `arm_provenance()` 기록 — `C12` (A15 참조)

### C-d · config 재사용 금지 · device 되읽기 · BLAS 스레드 — `M-C-d`
- **요구**: 런마다 `cfg = dict(DEFAULT_CONFIG)` 를 새로 만들고 `assert 'learned_atoms' not in cfg`; 제안 단계가 실제로 쓴 device 를 되읽어 행에(`config` 가 정하지 않는다); `OMP/OPENBLAS/MKL_NUM_THREADS` 를 numpy 임포트 전에 고정하고 meta 에 되읽어 기록(`torch.set_num_threads` 는 BLAS 를 제약하지 않는다 ⟹ `wall_s` 팔 간 비교 금지).
- **왜**: A1 §2 — `_auto_select_init` 이 호출자 dict 에 원자를 캐시해 **한 config 를 96런에 재사용하면 파일 #1 의 원자가 전 런에 재사용**(`G-ARM` 은 여전히 `learned` 라 보고). A2 §7 — 행의 `device='cpu'` 가 거짓(실제 cuda) · `ORACLE` `lstsq` 멀티스레드.
- **확인**: 수동.

### C-e · 축 규약(축 벡터 vs 평탄 격자)은 **스모크로** 확인, 실행은 메모리 캡 아래 — `M-C-e`
- **왜**: A2 §0 — `predict_field` 에 평탄화 배열을 넘겨 16384² = 2.68억 점 ⟹ WSL OOM 두 번. 같은 부류가 `S9 probe_real_path.py` 에서 `.ravel()` 로 **조용히 완주**(가우시안 경로는 즉시 죽고 bank 경로는 삼킨다).
- **확인**: 수동 + `systemd-run` 의무.

### C-f · 재개 가능 · 배치 중 저장소 **어떤 파일도** 고치지 않는다 — `M-C-f`
- **왜**: feedback_never_mutate_tree_during_a_batch — `T9` 384런 중 `git stash/checkout` 네 번으로 꼬리 17행 소실; 같은 날 `sequential_update.py` 주석 편집으로 행 121개가 다른 해시 ⟹ 전량 재실행. 살아남은 것은 **행별 `code_md5` 열**이었다.
- **확인**: 수동(러너는 `done` 집합으로 재개; 수리는 대기열에).

---

## D. 동결 · 환경

### D1 · `FROZEN_md5.txt` 를 **결과 0행 시점**에, 임포트 경로와 체크포인트까지 — `C7`·`C7s`·`C7t`·`C7m`
- **요구**: 배치 파일(PREREG·개정문·gates·analyze·run·표본) + 러너·게이트가 임포트하는 패키지 파일(값을 실제로 움직이는 것 — `varpro`·`spectral_gaussian`·`dispersion`·조건화 망까지) + 확산 체크포인트 `sha256`. 러너는 md5 를 **시작 전과 끝** 둘 다 재고 `meta['code_md5']` 에 기록한다. 표본 파일(`sample_*.json`)은 git 에 넣고 md5 를 개정문에 박는다.
- **왜**: A2 §7·§8 — 초판 동결 목록이 `spectral_gaussian`·`varpro`·`dispersion`·`d3_beamform_net`·**체크포인트**를 빠뜨렸고 md5 를 런 끝에만 쟀다(도중 편집이 「그 런의 상태」로 기록). `CLAUDE.md` — *"동결한 적이 없다"* 는 *"동결 뒤에 움직였다"* 보다 나쁘다(기준선 없으면 md5 열이 영원히 OK). `DISCIPLINE_seed_axes` P1 — `ckpt_sha256` 열이 없으면 *"같은 체크포인트였다"* 는 추정이다.
- **확인**: `C7`(직접 임포트 = FAIL) · `C7t`(깊이-2 = WARN, 사람이 판정) · `C7s`(학습 팔 지문) · `C7m`(mtime 순서) + `check_batch_artifacts` I·`★자기동결`·`⚠미동결n`.

### D2 · 측정면 / 분석면 분리 — `M-D2`
- **요구**: 결과 이후 분석기·게이트를 고치면 **분석면**만 새 `FROZEN_md5_A*.txt` 로 재동결하고, 측정면(러너·임포트·ckpt·표본·PREREG·사전 개정문)은 원래 동결과 `meta['code_md5']` 에 **3중 일치**해야 한다. 원래 동결 파일은 수정하지 않는다.
- **왜**: A3 §2 — *"개정문 3 은 측정을 한 비트도 안 건드렸다"* 의 기계 증명.
- **확인**: 배치 `G-ENV` 가 직접(수동 확인). `check_batch_artifacts` 의 `재동결` 열이 이력을 드러낸다.

### D3 · 표본이 사전등록 모집단 안이고 사전노출·선행 배치와 서로소 — `M-D3`
- **왜**: A3 §4 — `G-SAMPLE` 은 서로소만 보므로 어느 디렉터리에서 뽑았든 통과(`NOTE §8` 의 v1 잔재 `36–59` vs PREREG `mixed_data2`). 처방 = `set(sample()) ⊆ 모집단` 게이트 + 참조 파일 부재 시 FAIL(초판은 부재 시 통과).
- **확인**: 배치 게이트(수동 확인).

---

## E. 결과 이후 · 개정 · 요약

### E1 · 게이트가 사라지면 **원장에 적는다** — `C10`
- **왜**: A3 §3 — `G-DENSITY` 가 A1(→`G-CAP` 대체)·A2(항진이라 제거) 두 단계에 걸쳐 **대체 없이 소실**; 기록만 하고 검증은 안 했다. `C10` 은 현재 T11 에서 `G-ZERO`(v1 의 「같은 프레임·같은 σ」)가 PREREG/개정문 어디에도 언급 없이 사라졌음을 낸다 — 러너 구조상 참이지만 **원장에 없다**.
- **확인**: `C10`(git 이력 대조).

### E2 · 사전등록 §금지/범위 의 ⛔ 의무 공시는 재작성에서 살아남아야 한다 — `C9`
- **왜**: A3 §1 — PREREG §6 의 `E2`·저들 논문·solver 공시 3건 + 예산 공시가 개정문 2 의 분석기 재작성에서 **삭제**됐다(= 사전등록 위반, 그날 세 번째 N−1). `T10` 도 `solver` 공시가 분석기에 0회다(census).
- **확인**: `C9`(토큰 부분문자열 — 긴 토큰 누락 FAIL, 2자 코드만 누락 WARN).

### E3 · 결과 후 추가 게이트는 `post_result_checks` 로 **따로** 표시하고, **막을 수만** 있어야 한다 — `M-E3`
- **왜**: A3 §0 — *"라벨을 새로 만들어내는 경로는 없다"*; 사전등록 차단 게이트로 세지 않는다. 사후 완화는 `M-4` 를 EXPLORATORY 로 강등시킨 죄(feedback_gate_must_be_passable).
- **확인**: 수동(verdict 의 `post_result_checks` 키).

### E4 · 요약은 사전등록보다 넓게 나가지 않는다 · 단위를 바꿔치기하지 않는다 — `M-E4`
- **왜**: `CLAUDE.md` — S13 의 PREREG 는 정확했는데 RESULT/메모리 요약이 부풀었다. MEMORY 09-01 `T5-ENS` 개정문 ① — *"척추가 살아남았다"* 는 셀 36 vs 레코드 12 의 **단위 바꿔치기**. 음성 라벨 인용 시 MDE 병기 의무(A2 §3).
- **확인**: 수동(`check_paper_disclosures`·`audit_thesis_*` 가 원고 쪽을 본다).

### E5 · 수리는 **변이로 확인**하고, 「N곳 전부」의 N 에 **주석**을 넣는다 — `M-E5`
- **왜**: feedback_verify_the_fix_by_mutation — 진단문을 정확히 쓰고 결함을 그대로 남긴 사례 3건(09-03). feedback_comments_count_as_N — 낡은 지시 주석이 정정을 되감는 재발 공장. feedback_a_repair_creates_a_new_defect — 09-02 하루에 수리가 새 결함 10건+.
- **확인**: 수동 — 보고에 *"X 를 지우면 실패하고 되돌리면 통과한다"* 를 실제로 돌려 적는다. (`check_prereg.py --selftest` 가 검사기 자신에 대해 이것을 한다: 정상 픽스처 무경고 + 변이 16종.)

### E6 · 사후 완화 금지 · 게이트 변경은 결과 0행 개정문에 공시 — `M-E6`
- **왜**: A2 §9(스모크 후 `G-CLAMP` 비차단 전환 공시, 선례 `M4 G-15`) · feedback_relative_margin(결과를 본 뒤 여유를 바꾸지 마라 — 다음 사전등록에서).
- **확인**: 수동.

---

## F. `check_prereg.py` 가 **못 보는 것** (정직한 한계)

- 효과를 판정에서 쟀는가(A11) · 접기 의미론(B9) · 순함수 술어의 위치(B10) · 자기진단≠판정불변(B11) — 전부 의미론.
- **팔 단위 키 누락**(C-a): 정적 키 집합은 합집합이다. `t_origin` 부류는 배치 게이트가 잡아야 한다.
- 라벨 확률의 **산술 정합**(A3): 표의 존재와 계산 함수의 존재만 본다.
- 항진 (a) 는 **문법적 동일성**만: 같은 값을 다른 이름으로 들고 오면(`LI.M_OBS_CAP` ↔ `G.deployed_cap()`) 침묵한다. 문자열 선택자(`arm='SHIP'`)는 일부러 뺐다.
- 동결 전이 임포트는 **깊이 2**, 그리고 어느 파일이 실제로 값을 움직이는지는 판정하지 않는다(WARN 목록을 사람이 읽는다).
- C9 는 토큰 **부분문자열**이다 — `E2` 같은 2자 코드는 우연히 맞을 수 있다(그래서 2자 누락은 WARN).
- A3 §4(모집단 핀) · §6 의 `all(ok)` 벨트 · §7(`G-ATOMS` 가 SHIP 만 봄) 은 이 검사기가 잡지 못했다 — 게이트의 **범위**는 `check_gate_blindspots` 부류의 일이다.
- 부호검정을 선언하지 않은 배치에서는 표의 귀무 확률을 검사하지 않는다(T10 의 물리 기준 계수에 이항 귀무를 붙였던 오탐을 막기 위해).

## G. 검증 기록 (2026-09-04, 전부 `systemd-run … MemoryMax=4G`)

- `--selftest`: 정상 픽스처 FAIL/WARN 0 · 변이 16종 전부 기대 등급 검출(다른 항목의 거짓 FAIL 0).
- `T11_EXTACC_2026-09-03` 현재판(`7f6debc`+작업트리): **FAIL 0 · WARN 6** — 전부 개정문이 스스로 공시한 잔여(`NO-INFORMATION-IN-OBS` 확률 미등록 A3 §5 · `threads` 중복 A3 §0 · `BEYOND-REACH` 옛 표 잔존 · 전이 임포트 6파일 미동결 · `G-ZERO` 소실 미언급).
- `--rev 18216fc^`(개정문 2 이전): **FAIL 9** — C1(MDE 0회 = A2 §3) · C2(라벨 확률 0 = A2 §2) · C3(PREREG v2 ↔ gates v1) · C4(`BEYOND-REACH` 0.5806 = A2 §1) · C5(`G-COV` 장식 = A2 §7) · C8(게이트 6개 미구현) · C6b(`size_ok`·`disjoint`·`code_md5` = A1 §4) · C7(동결 없음 = A2 §8) · C9(v2 공시 미구현). WARN: C2b(손계산) · C13(Holm 0회 = A2 §4).
- `--rev 18216fc`(개정문 2 이후·3 이전): **FAIL 2** — C6a(`seed=G.SEED` 왕복 = A3 §2) · C9(`E2`·`solver` 삭제 = A3 §1). WARN: C10(`G-DENSITY` 소실 = A3 §3) 외.
- 오탐 census(다른 배치 6개): 검사기 결함 4종을 잡아 고쳤다 — 게이트 튜플이 `gates_*.py` 의 **호출형**(`A('G-…', …)`)에 있는 것 · 공시 토큰의 파일 글롭/경로 조각/닫히지 않은 백틱 · `≥ 10/12` 의 역추적(`≥1`) · 부호검정이 아닌 설계의 이항 귀무 · PRIMARY 표기 규약. 남은 FAIL 은 규약 위반 목록이지 판정이 아니다(과거 배치는 이 체크리스트 이전에 쓰였고, `C7s`·`C12` 규칙은 08-30·09-03 에 생겼다). ⚠하나는 검사기가 **읽을 수 없는 단서**다: `S_ENS` 의 `G-DEGEN` 은 PREREG 자신이 *"패키지가 멤버별 값을 안 실으면 범위축소로 내린다"* 고 조건부로 적었다 — `C8` 은 그 단서를 못 읽는다.
