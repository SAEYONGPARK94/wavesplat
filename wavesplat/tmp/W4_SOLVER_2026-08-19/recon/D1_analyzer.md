# D1 — `analyze_w4.py` : 판정 분석기, **자료를 보기 전에 동결**

**작성 시각**: 2026-08-19 KST, `results.jsonl` 이 아직 쓰이는 중.
**★핵심 규율**: S17 은 **행을 본 뒤에** 분석기를 썼다. 이번엔 순서를 뒤집었다.
이 파일과 `analyze_w4.py` 는 `PREREG.md` + `AMENDMENT_1.md` + `AMENDMENT_2.md` 만 읽고 작성됐고,
**`results.jsonl` 은 단 한 번도 열지 않았다.** 개발·시험은 전부 `smoke_results.jsonl`(42행) 과
스크래치패드의 합성 행으로 했다.

**건드리지 않은 것**: `run_w4.py` · `PREREG.md` · `AMENDMENT_1/2.md` · `arms_config.py` · `varpro.py` ·
`watchdog.py` · `corpus_w4.json` · `wavesplat/**` · `wavesim/**` · `configs/**`. 추정기 실행 0회.

**신규 파일 2개** (+ 스모크 산출 1개):
`analyze_w4.py` · `recon/D1_analyzer.md` · `verdict_w4_smoke_results.json`(스모크 실행 증거).

---

## 0. ★런 안전성 확인 — 새 파일이 `code_md5` 를 바꾸지 않는다

이것을 **먼저** 확인했다. `run_w4.code_fingerprint()` 가 배치 디렉터리를 glob 했다면 내가 파일 하나만
만들어도 그 뒤에 쓰이는 모든 행의 `code_md5` 가 달라지고 **G-8("1종 digest")이 FAIL** 하면서
1,344런이 통째로 죽는다.

실측(`run_w4.code_fingerprint()` 를 읽기 전용으로 호출):
- 파일 집합 = `wavesplat/**/*.py` + `wavesim/**/*.py` + `configs/` 3개 + **명시된 5개**
  (`run_w4.py`, `PREREG.md`, `arms_config.py`, `varpro.py`, `watchdog.py`) + `run_w3.py` = **91개**.
- `analyze_w4.py` · `D1_analyzer.md` · `verdict_w4_*.json` **전부 미포함**(hits = `[]`).
⟹ **런에 영향 0.**

⚠**부수 발견(수정하지 않음, 기록만)**: 행에 찍히는 `code_md5` 의 지문 집합에 **`AMENDMENT_1.md` 와
`AMENDMENT_2.md` 가 없다.** `PREREG.md` 는 핀으로 박혀 있는데 **판정 규칙을 실제로 지배하는 개정문
2개는 `code_md5` 밖**이다(A-1 이 다중성 보정을 통째로 갈아치웠고 A2-1 이 P-2 의 의미를 바꿨다).
✅**다만 `FROZEN_md5.txt` 에는 둘 다 들어 있다**(실측: `12e7c9ee…` / `96ce1e0e…`, 라이브 md5 와 일치)
⟹ 동결 자체는 온전하고, **행 수준 지문에서만 빠져 있다**. 즉 개정문이 배치 중에 바뀌어도
`G-8`("1종 digest")은 못 잡는다. **이번 배치의 결론에는 영향 없음** — 개정은 판정 행 0개일 때
이뤄졌고 git 커밋(`196050e` 계열)됐다. 다음 배치의 지문 집합에 개정문을 넣는 것을 권한다.

**보호 파일 무손상 확인**(작업 종료 시점 실측): `run_w4.py`·`PREREG.md`·`AMENDMENT_1/2.md`·
`arms_config.py`·`varpro.py`·`watchdog.py`·`corpus_w4.json`·`smoke_results.jsonl` **9개 전건이
`FROZEN_md5.txt` 값과 md5 일치**. 내가 만든 것은 신규 3파일뿐이다.

---

## 1. 규칙 ↔ 구현 대응표

| 출처 | 규칙 | `analyze_w4.py` 구현 | 확인 |
|---|---|---|---|
| PREREG §6 | 짝 단위 = 에피소드 | `pair()` 가 `episodes` 위에서만 돈다 | 스모크 2ep |
| PREREG §6 | **med3 는 에피소드 수준**(시드 median 먼저, 짝 나중) | `med3()` → `pair()` **2단계 분리**. `med3` 가 `(arm, episode)` 로 묶고 시드 median 을 먼저 낸다 | 집계 표에 `seeds/episode=[3]` 인쇄 |
| PREREG §6 | 상대차 `(처치−대조)/대조` | `pair()` 의 `rel` | — |
| PREREG §6 | 결측 리스트와이즈 제외 | `pair()` 의 `drop_missing`, 인쇄됨 | `fail_n` 합성으로 n=40 확인 |
| PREREG §6 | **동점(부동소수 정확 일치) 제외 + n 감소** | `if av == bv: drop_tie` (정확 `==`, 허용오차 없음) | `fail_tie` 합성 = 64동점 → `n=0` |
| PREREG §6 | 셸 접두어 `obs:` | `shell_getter()` 가 `obs:` 우선, 무접두어 폴백. `shell_convention_audit()` 가 팔별 규약을 인쇄하고 **혼재 시 BLOCKER** | 스모크 9팔 전부 `['obs:']` |
| **A-1** | **Holm-2 가 유일 유효 규칙**(고정 문턱 문장 삭제) | `holm2()`. p 오름차순 → 첫째 α/2=0.025, 둘째 α=0.05, **step-down**(첫째 기각 실패 시 둘째도 자동 실패) | `fake_earns` 에서 P-2 가 rank1 기각 → P-1 이 rank2 @0.05 로 검정됨 |
| **A-1** | P-1 단측 ≥41(0.025) / ≥40(0.05) | `region_one(64, α)` **정확 이항으로 재도출** | `>=41` / `>=40` **일치** |
| **A-1** | P-2 양측 ≥42∨≤22(0.025) / ≥41∨≤23(0.05) | `region_two(64, α)` | `>=42 or <=22` / `>=41 or <=23` **일치** |
| **A-2** | `p_one(42|64)=0.008429`, `p_two(42|64)=0.016858` | `self_test()` 가 매 실행마다 검산 | **7/7 PASS** (+ PREREG §1 의 `p_one(41)=0.01638` 도) |
| PREREG §1 | IUT `L-SHIP < ZERO` ≥41/64, 실패 시 `FAIL` | 우선순위 3순위 | `fail_iut` 합성 → `FAIL` |
| §1.1 | P-2 세 라벨 | `OPTIMIZER-HARMS` / `-INERT` / `-EARNS` | 합성 3종으로 **세 분기 전부 실행** |
| **A2-1** | 의미는 *"결합 적합 단계"* 이지 *"모든 하강"* 이 아니다 | 라벨 문구가 **`결합 적합(joint fit) 단계`** 를 쓰고, `A2-1 step check` 줄이 `820 vs 320 = 500/820 = 61%` 를 인쇄 | 스모크 실측 `[820]`/`[320]` |
| **A-5** | `clamp_hits_p1` 판독 규칙 | `L-SHIP` 전 행이 0 → *"결합 적합 단계"*, 하나라도 >0 → *"Phase-1 루프 전체(결합 적합 + clamp 투영)"*. 라벨 문구에 **자동 반영** | 스모크 `0/6` |
| 사용자 지시 | `INERT` 인쇄 시 **MDE 동반** | 기각역의 검정력 0.80 최소 승률 + **짝 상대차 중앙값의 분포무관 CI**(부호검정 순서통계량, 정확 이항 커버리지) | `fake_inert`: π-MDE **0.682(=43.6/64)**, CI 인쇄됨 |
| **A-8** | 전순서: 중단게이트 → DCC → IUT → n<56 → Holm-2 | `analyse()` 의 단일 if-elif 사슬. `HALTING = (G-1,G-2,G-3,G-5,G-7,G-8)` | 4분기 전부 합성으로 실행 |
| **A-8** | `G-4`·`G-9` 는 PRIMARY 를 못 죽인다 | `eval_gates()` 가 게이트마다 `halting` 플래그를 들고, `G-4` 는 `RECORDED`/`NO-INTERVENTION`, `G-9` 는 `OUT-OF-SCOPE` | 인쇄 확인 |
| **A-8** | `P-1 통과 ∧ P-2 HARMS` → **`REMEDY-BY-SUBTRACTION`** | 그대로 | `fake_harms` → 그 어휘 출력 |
| PREREG §8(A-8 개정) | P-1 실패 **∧ P-2 ≠ HARMS** 이면 솔버 축 종결 | `NO-REMEDY` 분기가 정지 규칙 문장을 함께 인쇄 | `fake_earns` |
| PREREG §3 | `S-2` `L-VP<L-ITER0` (단측 α=0.025, ≥41/64) | `one_sided()` | — |
| **A-4** | `S-2b` `L-VP<L-STEP400` | 동일. **A2-4** 의 예산 동수(1220=1220)를 `G-2c` 로 확인하고 라벨에 *"★예산 동수, 매개화 단독 대조"* 명기 | 스모크 `G-2c PASS` |
| **A-4** | `S-2c` `L-VP` vs `L-VP-NW` **양측** | `two_sided()` | — |
| **A-4** | 귀속 금지: S-2·S-2b·S-2c **함께** 통과할 때만 "매개화 덕분" | `attribution_allowed = S-2 PASS ∧ S-2b PASS ∧ **S-2c 널**` (아래 §2.4 참조) | 인쇄 + json |
| PREREG §3 | `S-3` 셸 9개, **`obs:`**, 단측 ≥41/64 | 셸별 `med3`→`pair`→`one_sided` | 9행 인쇄 |
| PREREG §3 | D-1/D-2/D-3 | 창잔차 대 필드(+ `L-VP`·`L-ITER0` 의 창↔필드 전이 상관 pearson/spearman) · 팔별 벽시계 median/IQR · `merge_removed`/원자수 | — |
| **A-7** | `D-4` = 기술통계(문턱·주장 없음) | **p 값 자체를 계산하지 않는다.** 승수·동점수·중앙값만. 2026-07-26 선행(혼잡 20셀 0승)을 **결과 전에 인쇄** | — |
| **A-11** | 무료 필드 | `RMSE_near`·`rho`·`VR`·`decorr_mean`·`shrink_mean`·`init_nrmse`·`vp_diag` 전/후 창잔차 등을 **행에 있으면** 요약 | 스모크: `RMSE_near` **부재** → *"부재는 음성 증거가 아니다"* 인쇄 |
| PREREG §7 + A-4 | 금지 목록 | `PROHIBITIONS` 14줄, **판정 아래에 항상** 인쇄 + json 저장 | — |
| 상시지시 | 이중검증 | 내 게이트 구현 **과** `run_w4.posthoc()` 를 둘 다 돌리고 불일치 시 BLOCKER | 스모크 `AGREE` |

---

## 2. ★내가 **해석해야 했던** 모호한 지점 — 전건 신고

동결 문서가 답을 주지 않아 내가 규칙을 **골라야 했던** 곳이다. 전부 코드 주석에도 남겼다.

### 2.1 결측 시드 — "리스트와이즈 제외"의 단위가 명시돼 있지 않다
PREREG §6 은 *"med3 는 에피소드 수준"* + *"결측은 리스트와이즈 제외"* 라고만 쓴다. 그런데
**어떤 에피소드에서 시드 3개 중 1개만 에러 행이면** ①그 에피소드를 통째로 버리는가 ②남은 2개의 median
으로 med3 를 대신하는가가 안 적혀 있다.
**내 선택**: ② — 있는 시드의 median 을 쓰고, 그 짝은 살린다. 에피소드는 **어느 한 팔에 값이 0개일 때만**
버린다. **완화책**: 시드가 3개 미만인 에피소드 수를 P-1/P-2 줄 바로 아래에 `!! N episodes with <3 seeds`
로 **강제 인쇄**한다. 배치가 정상 완주하면 0이므로 실무상 무해하고, 0이 아니면 눈에 띈다.
⚠**만약 이 값이 0이 아니게 나오면, "시드 3개 완비 에피소드만" 판정을 다시 돌려 민감도를 보고할 것.**

### 2.2 `n < 56` 의 `n` 이 어느 짝인지 안 적혀 있다
P-1 과 P-2 는 동점·결측이 달라 **n 이 서로 다를 수 있다**.
**내 선택**: `n_eff = min(n_P1, n_P2)`. 보수적이고, 둘 다 인쇄한다.

### 2.3 IUT 문턱의 n≠64 보간
IUT 는 *"≥41/64"* 라는 **고정 분수**로 적혀 있다. n=64 면 41 로 그냥 쓴다. 동점·결측으로 n<64 가 되면
문서에 답이 없다. **내 선택**: `ceil(41/64·n)`. 인쇄줄에 *"n≠64, 보간은 분석기 해석이다"* 를 **명시**한다.
(대안 = "n<64 면 IUT 를 UNDECIDED 로" 였는데, IUT 는 우리에게 **불리한** 방향의 게이트라 느슨하게
만들면 안 되므로 비례 문턱을 택했다.)

### 2.4 `A-4` 귀속 조건에서 `S-2c` 의 방향
A-4 는 *"`S-2`·`S-2b`·`S-2c` 가 함께 통과할 때만"* 이라 쓰는데, `S-2c` 는 **양측**이고 그 역할은
**가중 요인을 분리**하는 것이다. "S-2c 통과 = 유의하게 다르다" 면 **가중이 요인이라는 뜻**이므로
"매개화 덕분" 귀속은 오히려 **깨진다**.
**내 해석**: 귀속 허용 = `S-2 기각` ∧ `S-2b 기각` ∧ **`S-2c 널`**(= 가중이 요인이 아님).
⚠**이건 문언 그대로가 아니라 의도 해석이다.** 인쇄줄이 세 항목의 상태를 각각 보여주므로 사람이
다르게 읽을 여지를 남겨 뒀다.

### 2.5 `OPTIMIZER-INERT` 의 MDE 정의가 PREREG 에 없다
사용자 지시로 인쇄하지만 **어떤 MDE 인지**는 어디에도 없다. **내 선택 2종 병기**:
① 부호검정 기각역이 검정력 0.80 으로 잡는 **최소 승률 π**, ② 짝 상대차 중앙값의 **분포무관 CI**
(순서통계량, 정확 이항 커버리지 ≈97%). ②가 "널이 배제하지 못하는 **효과 크기**" 에 더 가깝다.
`target=0.80` 은 관례이고 사전등록에 없다 — **내가 고른 상수**다.

### 2.6 `P-2` 방향 라벨은 `L-ITER0` 이 **낮은** 쪽을 센다
`wins` 를 *"처치 < 대조"* 로 통일했으므로 P-2 의 `wins` = `L-ITER0 < L-SHIP` 인 에피소드 수.
`wins ≥ 상한` → **HARMS**(하강 빼는 게 낫다), `wins ≤ 하한` → **EARNS**. §1.1 표와 부호가 일치함을
합성 3종으로 확인했다. (P-2 는 양측이라 인쇄줄이 `<` 대신 `vs` 를 쓴다 — 표기 혼동 방지.)

### 2.7 `G-6`(DCC)의 위치
PREREG §4 는 G-6 을 *"판정 인쇄 없이 종료"*, §5 우선순위는 *"G-* 실패 → G-6 DCC 종료"* 로 **분리**해
적었고, A-8 은 중단 게이트 목록에서 **G-6 을 뺐다**. 사용자 지시도 `중단게이트 → DCC` 순이다.
**내 구현**: G-6 은 중단 게이트가 **아니고**, 별도 2순위 단계 `NO-VERDICT (DCC)` 다.
⚠`run_w4.posthoc()` 는 G-6/G-6b/G-6c 를 `ok` 에 섞어 넣는다 — **단일 bool 이라 순서를 표현할 수 없기
때문**이지 규칙이 달라서가 아니다. 그래서 교차검증은 "둘 다 통과/둘 다 실패" 수준에서만 비교한다.

### 2.8 `G-6b`/`G-6c`(A-12)의 등급이 안 적혀 있다
A-12 는 *"무결성만, 가설·문턱 불변"* 이라 하고 중단 여부를 안 쓴다.
**내 선택**: 비중단(A-8 의 중단 목록에 없으므로). 실패는 크게 인쇄된다. `b1_full_comparison.json` 의
100/100 프레임 불일치가 이 검사 부재로 46일간 안 잡혔으므로 **실패 시 반드시 사람이 판단할 것**.

### 2.9 `G-7`·`G-8` 을 행에서 어떻게 보는가
`G-7`(위상 규약)은 행에 없다 — **`run_w4._g7_phase_identity()` 를 라이브로 재호출**한다(런 0회, 난수
항등식). 호출 자체가 실패하면 `gates.log` 의 `G-7` 줄로 폴백하고, 그것도 없으면 `PENDING` = 중단.
`G-8` 은 행의 `code_md5` **distinct 수**로 본다(+ `git_head` distinct 도 인쇄, 판정에는 안 씀).

### 2.10 `n_eff` 와 셸 판정의 문턱
`S-3` 은 *"≥41/64"* 로 고정 표기돼 있어 **n=64 기준 문턱**을 헤더에 인쇄하고, 실제 검정은
**그 셸의 실제 n 에서 α=0.025 기각역**으로 한다(동점·결측이 셸마다 다를 수 있으므로).

---

## 3. 셸 키 함정 처리 (저장소가 반복해서 물린 것)

- `run_w4.py:92` 규약 = **`SHELL_KEYS = ['obs:' + k for k in BARE_KEYS]`**, 9개, **모든 팔에 동일**
  (`ZERO`·`ORA-K-CVX` 포함). 스모크 42행 전부 `obs:` 였다.
- 분석기는 `obs:<bare>` 를 **먼저** 찾고 없으면 무접두어로 폴백한다. **`ship:` 은 절대 읽지 않는다**
  (같은 표본에서 1.8–4.1배 갈리는 다른 양이다).
- **팔 하나가 두 규약을 섞으면 BLOCKER 로 인쇄**하고 S-3 을 해석 불가로 표시한다.
- **9셸 전부 0승** 이면 `!! 셸 키 함정을 의심하라` 를, **9셸 전부 n=0** 이면 `BLOCKER` 를 인쇄한다
  (지시받은 안전장치).

---

## 4. 실행 증거

```
python3 -m py_compile analyze_w4.py                       -> COMPILE OK
python3 analyze_w4.py smoke_results.jsonl                 -> exit 0, 159줄, 끝까지 동작
   self-test        : 7/7 frozen thresholds reproduced from exact binomial
   cross-check      : runner_ok=True  analyser_ok=True  -> AGREE
   G-1..G-9         : 전부 인쇄 (halting/soft 구분 포함)
   VERDICT          : UNDECIDED (n)   [n=2 <56 — 스모크라 당연하고 판정 의미 없음]
   wrote verdict_w4_smoke_results.json
```

스모크는 팔당 **2 에피소드**뿐이라 **어떤 방향성도 증거가 아니다.** 여기서 나온 수치는 판정이 아니라
**코드 경로가 도는지**의 확인이다.

합성 행(스크래치패드, 64 에피소드 × 9팔 = 1,472행)으로 **판정 분기 전수 실행**:

| 합성 케이스 | 나온 판정 | 확인한 것 |
|---|---|---|
| `fake_harms` | **`REMEDY-BY-SUBTRACTION`** | A-8 신설 어휘, P-1 rank1 기각 → P-2 rank2 기각 |
| `fake_inert` | `REMEDY-CONFIRMED / OPTIMIZER-INERT` | **MDE π=0.682 + CI 인쇄** |
| `fake_earns` | `NO-REMEDY / OPTIMIZER-EARNS` | Holm **step-down**(P-2 가 rank1) + 정지 규칙 문장 |
| `fail_iut` | `FAIL` | IUT 0/64 |
| `fail_n` | `UNDECIDED (n)` | n=40 <56, P-1/P-2 n 둘 다 인쇄 |
| `fail_dcc2` | `NO-VERDICT (DCC)` | 필수 필드 1개 제거 → G-6 FAIL, **중단게이트와 분리** |
| `fail_gate` | `HALTED (G-3,G-8)` | `vp_steps=399` + 이질 `code_md5` |
| `fail_tie` | D-4 `0/0, 64 동점 제외` | 동점 규칙 |

---

## 5. 본런에서 이 분석기를 돌릴 때

```bash
cd /home/vexit/wavesplat/wavesplat
python3 tmp/W4_SOLVER_2026-08-19/analyze_w4.py            # results.jsonl -> verdict_w4.json
```

- 인자 없이 돌리면 `results.jsonl` → `verdict_w4.json`. 다른 파일을 주면
  `verdict_w4_<stem>.json` 으로 나가므로 **스모크가 본 판정 파일을 덮어쓰지 않는다.**
- `--no-posthoc` 는 교차검증만 끈다. **본런에서는 쓰지 마라**(이중검증이 상시지시다).
- ⚠**출력의 첫 줄 세 개를 먼저 볼 것**: `self-test 7/7` · `shell convention` 혼재 여부 ·
  `cross-check AGREE`. 셋 중 하나라도 어긋나면 그 아래 판정은 읽지 마라.
