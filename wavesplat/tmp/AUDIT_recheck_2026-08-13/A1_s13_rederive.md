# A1 — S13 헤드라인 수치 독립 재도출 (2026-08-13)

**대상 배치**: `wavesplat/tmp/S13_oos_2026-08-12/`
**방법**: `results.jsonl` 704행에서 **새로 작성한 독립 스크립트**로 전량 재집계.
`analyze_s13.py` 는 **참고용으로 읽기만** 했고 실행하지 않았다. `RESULT.md`·`verdict_*.json` 은
**대조 대상**으로만 썼다.

## 열람 범위 (= 주장 범위)

**연 파일 (전문)**
- `tmp/S13_oos_2026-08-12/`: `PREREG.md`, `run_s13.py`, `analyze_s13.py`, `RESULT.md`,
  `GATES.log`, `PRIMARY.log`, `verdict_primary.json`, `verdict_secondary.json`, `results.jsonl`(704행 전량)
- `wavesplat/run.py` (DEFAULT_CONFIG 부·프레임 선택부), `wavesplat/evaluation/metrics.py`(nrmse 정의),
  `wavesplat/baselines/letkf_grid.py:48-66`, `wavesplat/optimizer/sequential_update.py:83`
- `tmp/mixed_retrain_2026-07-26/gen_mixed.py:55-71`
- `tmp/unified_2026-07-26/analyze_switch.py`(임계 스윕부), `arm_runs_extra.jsonl`, `atoms_cache_extra.jsonl`
- `tmp/S8_deploy_confirm_2026-08-10/results.jsonl`, `tmp/S9_lr_x_l1_2026-08-11/results.jsonl` (에피소드 목록만)
- `tmp/DEBATE_blankslate_2026-08-12/r2_letkf_{round2,round3,strength}.json`
- `tmp/DEBATE_blankslate_2026-08-12/PREREG_DRAFT.md` (고리 지정부)

**안 본 것 (주장하지 않음)**
- `tmp/BLANKSLATE_2026-08-12/` 및 `tmp/DEBATE_blankslate_2026-08-12/` 의 나머지 산출물
  (특히 85/100 원판정 원자료 — S13 자체 검증에 불필요)
- S8/S9 의 `PREREG.md`·`RESULT.md` 본문 (에피소드 목록만 `results.jsonl` 에서 추출)
- 학습 체크포인트 가중치 파일 자체
- `results/` 하위 어떤 파일도 안 열었다

---

# 1. MEASURED — 원자료에서 직접 재도출한 값

## 1.1 파일 무결성 (F 항목)

| 점검 | 재도출값 | 판정 |
|---|---|---|
| 총 행수 | **704** | 일치 |
| `error` 키 보유 행 | **0** | 일치 |
| `(episode, arm)` 중복 | **0** | 문제 없음 |
| 비유한(NaN/inf) `avg_nRMSE` | **0** | 문제 없음 |
| `git_head` 고유값 | **1** — `5f0846cc435bc62ff31af6beff72a6a629bc50a4` (704/704) | 단일 코드 |
| 고유 에피소드 | **128** (sparse 64 / crowded 64) | 일치 |

**팔별 행수 (E 항목, n 산술)**
`SHIP` 128 + `LETKF_PLUS` 128 + `LETKF` 128 + `SHIP_NOCONF` 128 + `SHIP_NODISP` 128 + `SHIP_L1E3` 64
= **704**. `SHIP_L1E3` 는 `run_s13.py:65` `CROWDED_ONLY` 로 혼잡 64만 — `results.jsonl` 에서
`Counter(stratum)` = `{crowded: 64}` 로 확인. **누수 없음.**

**셀 구성** — 8셀 × 16 = 128:
`(crowded,BF4,15)16 (crowded,BF4,30)16 (crowded,BF5,15)16 (crowded,BF5,30)16`
`(sparse,BF4,2)16 (sparse,BF4,5)16 (sparse,BF5,2)16 (sparse,BF5,5)16`

**시드**: sparse `48–63`, crowded `32–39 ∪ 48–55`. **32 미만 시드 0개.**

**프레임 정합**: `SHIP.n_match ≠ LETKF_PLUS.n_match` 인 에피소드 **0/128**.
`n_match ∈ {120,160}`, `n_all = 301` (128/128). `run.py:198`
`loader.get_tracking_frames(tp_factor=2.0)` 와 `run_s13.py:157` `t > 2.0*Tp_true` 가 같은 규칙.
`nrmse` 는 양 팔 모두 `RMSE/Hs_true` (`metrics.py:32-33`, `run_s13.py:151-155`) — **동일 정규화**.

## 1.2 PRIMARY

```
crowded  SHIP<LETKF+  56/64  p=2.781e-10   med(rel)=-28.82%  rel(med)=-30.81%  rel(mean)=-28.81%
sparse   SHIP<LETKF+  63/64  p=3.524e-18   med(rel)=-93.24%  rel(med)=-93.28%  rel(mean)=-86.30%
Holm-2   p_adj: crowded=2.781e-10   sparse=7.047e-18
ZERO     crowded 64/64  p=5.421e-20 · sparse 64/64  p=5.421e-20
문턱     41/64 (PREREG.md:85)
```
**주장 대조** — "혼잡 56/64 (p_adj 2.8e-10) · 희소 63/64 · 제로필드 양 층 64/64 · 문턱 41/64":
**전건 일치.**

Holm 방향도 재검: 오름차순 정렬 후 sparse(3.524e-18)×2 = 7.047e-18, crowded(2.781e-10)×1 = 2.781e-10.
`analyze_s13.py:51-58` 의 holm 구현과 내 독립 구현이 동일 결과. **일치.**

**셀 분해 (참고)** — crowded `BF4/15` 14/16 · `BF4/30` 13/16 · `BF5/15` 15/16 · `BF5/30` 14/16;
sparse `BF4/2` 15/16 · `BF4/5` 16/16 · `BF5/2` 16/16 · `BF5/5` 16/16. **어느 셀도 뒤집히지 않는다.**

혼잡 패배 8건 전량:
```
BF4_15comp_seed33  SHIP 0.2188 > LETKF+ 0.2146  route=v2_konly
BF4_15comp_seed55  SHIP 0.1667 > LETKF+ 0.1631  route=v2_konly
BF4_30comp_seed35  SHIP 0.2329 > LETKF+ 0.2215  route=learned
BF4_30comp_seed38  SHIP 0.2162 > LETKF+ 0.2091  route=learned
BF4_30comp_seed49  SHIP 0.2100 > LETKF+ 0.2051  route=learned
BF5_15comp_seed32  SHIP 0.2523 > LETKF+ 0.2145  route=learned
BF5_30comp_seed48  SHIP 0.2327 > LETKF+ 0.2198  route=learned
BF5_30comp_seed50  SHIP 0.1873 > LETKF+ 0.1854  route=learned
```
경로×결과 (혼잡): `learned` 55승/6패, `v2_konly` 1승/2패.

## 1.3 SECONDARY

**S-2e 사전노출 24 제외**
```
crowded 미노출 44/52  p=2.020e-07     (노출 12/12  p=2.44e-04)
sparse  미노출 51/52  p=1.177e-14     (노출 12/12  p=2.44e-04)
```
`pre_exposed=True` 행 수 = **24** (`SHIP` 팔 기준). **"44/52 · 51/52" 일치.**

**성분수 기울기**
```
ncomp= 2  31/32  p=7.683e-09  med(rel)= -97.9%
ncomp= 5  32/32  p=2.328e-10  med(rel)= -88.2%
ncomp=15  29/32  p=1.278e-06  med(rel)= -41.0%
ncomp=30  27/32  p=5.654e-05  med(rel)= -19.8%
```
**"15comp 29/32 · 30comp 27/32" 일치.** ⟹ "≥15 결정적 패" 는 이 표본에서 재현되지 않는다. 확인.

**기준선 강화 효과**
```
LETKF+ < LETKF   crowded 64/64  med(rel)=-4.59%   sparse 62/64  med(rel)=-4.29%
SHIP   < LETKF   crowded 63/64  med(rel)=-32.32%  rel(med)=-34.48%  rel(mean)=-32.14%
SHIP   < LETKF+  crowded 56/64  med(rel)=-28.82%
```
**"공개 LETKF 였다면 63/64·−32.3%, 튜닝 후 56/64·−28.8%" 일치.**

**S-2c λ 지분** (혼잡 n=64)
```
SHIP      vs LETKF+   : 56/64  med(rel)=-28.82%
SHIP_L1E3 vs LETKF+   : 49/64  med(rel)=-18.10%
SHIP      vs SHIP_L1E3: 54/64  단측 p=9.982e-09 (양측 1.996e-08)  med(rel)=-10.47%
```
지분을 **6가지 규약**으로 전부 계산 (PREREG 에 지분 계산 규약이 없으므로 강건성 확인):

| 규약 | 지분 |
|---|---|
| 마진차 / med(rel) | **37.2%** |
| 마진차 / rel(med) | 28.9% |
| 마진차 / rel(mean) | 36.9% |
| 직접비 / med(rel) | 36.3% |
| 직접비 / rel(med) | 37.0% |
| 직접비 / rel(mean) | 45.1% |

**전 규약 28.9–45.1%, 6/6 이 50% 미만.** ⟹ PREREG §8-4 사망조건 미저촉은 **규약 의존이 아니다**.
"λ 지분 37%" 는 med(rel) 마진차 규약에서 **37.2%**, 일치. "54/64" 일치.

**S-2d 라이더** (양수 = 끄면 나빠진다)
```
SHIP_NOCONF crowded  끄면나빠짐 10/64  양측 p=2.00e-08  med(rel of ABL vs SHIP)=-15.6%
SHIP_NOCONF sparse   끄면나빠짐 16/64  양측 p=7.73e-05  med(rel)=-20.4%
SHIP_NODISP crowded  끄면나빠짐 62/64  양측 p=2.26e-16  med(rel)=+70.2%
SHIP_NODISP sparse   끄면나빠짐 62/64  양측 p=2.26e-16  med(rel)=+777.8%
```
`verdict_secondary.json` 과 **전건 일치**. 즉 기여 #3 DEAD(끄는 쪽이 혼잡 54/64·희소 48/64 우세),
기여 #2b ALIVE. **재도출 성공.**

**S-2a 근거리 고리** — 아래 F2 참조 (사전등록 위반).

**라우터 경로 (G-1)**: crowded `learned` 61 / `v2_konly` 3, sparse `learned` 51 / `v2_konly` 13.
일치. `SHIP_NOCONF`·`SHIP_NODISP`·`SHIP_L1E3` 도 **동일 경로 분포** (라우터가 절제에 영향받지 않음).

## 1.4 생성 0 · 시드 소모 0

- 사용 h5 128개 전량 mtime: `data_s8_crowded_2026-08-10` = **08-10 15:30~**,
  `data_s9_{sparse,crowded}_2026-08-11` = **08-11 08:30~**. 배치 시작(08-12 20:20)보다 **앞선다**.
- 결과행 어디에도 생성 기록 없음, 러너(`run_s13.py:38-48`)는 `glob` 으로 기존 파일만 읽는다.
⟹ **"생성 0회 · 시드 소모 0" MEASURED 확인.**

## 1.5 비용

`sum(wall_s)` = 17,863 s = **4.96 h**. 팔별 중앙 벽시계:
`LETKF` 7.0s · `SHIP` 18.9s · `SHIP_L1E3` 19.9s · `SHIP_NOCONF` 19.5s · `SHIP_NODISP` 16.7s · `LETKF_PLUS` 39.9s.

---

# 2. 점검 A — 기준선 오염: **걸리지 않았다**

`SHIP` 팔은 `run_s13.py:53-55` 에서 `DEFAULT_CONFIG` 를 **전 키 복사**해 넘긴다. 결과행에 기록된
`effective_config` 는 **128/128 행 모두 동일**하고 다음을 포함한다:

```
lambda_l1 = 0.0001   init_method = 'auto'   use_confidence = True   use_dispersion = True
tracking_mode='w_phi'  analytic_ensemble_k=3  ... (21 키)
```

- **PRIMARY 대조는 `LETKF+`** — `lambda_l1` 개념이 없는 별개 추정기(`letkf_grid.py`)다.
  ⟹ 08-11 λ 배포로 인한 "기준선이 처치값을 받는" 오염 경로가 **구조적으로 존재하지 않는다.**
- `SHIP_L1E3` 는 `WS_ARMS`(`run_s13.py:60`)에서 `lambda_l1=1e-3` 을 **명시**하고,
  기록된 config 차분 = `{'lambda_l1': 0.001}` 하나뿐.
- **실효성 실증**: `SHIP_L1E3` 0/64, `SHIP_NOCONF` 0/128, `SHIP_NODISP` 0/128 이
  `SHIP` 과 `avg_nRMSE` 가 동일 — 즉 **320/320 행 전부에서 override 가 실제로 작동**했다.
  (E1 이 잡은 "폴백에 맡겨 짝Δ≡0" 상태였다면 여기서 동일값이 쏟아졌을 것이다.)

**판정: A 오염 없음.** 다만 F9(아래) 의 단서를 붙인다.

# 3. 점검 D — 시드 독립성: **성립한다**

| 항목 | 시드 | 출처 |
|---|---|---|
| 라우터 임계 {6,8,10} 스윕 코퍼스 | **{0,1,2,3,4}** | `tmp/unified_2026-07-26/analyze_switch.py:122` + `arm_runs_extra.jsonl`·`atoms_cache_extra.jsonl` 전량 파싱 |
| 체크포인트 기하 도너 | **{0,2,3,4}** | `gen_mixed.py:66-71` (BF6/BF7 seed0 + BF3/BF4 5comp seed2–4) |
| `LETKF+` ne600 튜닝 파일럿 | **{0}** | `r2_letkf_{round2,round3,strength}.json` — 4 에피소드 전부 `seed0` |
| **S13 평가** | **32–63** | `results.jsonl` |

**세 튜닝 코퍼스 모두 32–63 과 교집합 0.** ⟹ "라우터가 안 본 시드" **참, MEASURED.**
파일럿 config 수도 확인: round2 14 + round3 4 + strength 6 = **24** (PREREG §2 "24 config" 일치).
G-4 재도출: ne600 vs published **4/4**, `rel(mean) = −4.05%` — PREREG §2 수치와 **정확히 일치**.
ne1200 은 440.6 ms/frame (ne600 은 135.4 ms) — 비용 기각 근거도 확인.

⚠단 `LETKF+` 파일럿이 seed0 에서만 돌았다는 것은 **기준선 튜닝이 out-of-sample** 이라는 뜻이고,
이는 우리에게 **불리한 쪽**이다 (in-sample 로 튜닝했다면 기준선이 더 강했을 수 있다).
RESULT.md §"기준선 강화가 실제로 판정을 깎았다" 는 **정직하다.**

# 4. 점검 B — 사전등록 동결 순서

```
PREREG.md              08-12 20:16:54
run_s13.py             08-12 20:18:04
GATES.log              08-12 20:20:21
phase1.log (마지막)     08-12 22:44:45   ← SHIP+LETKF_PLUS wall 합 2.41 h ⟹ 시작 ≈20:19
analyze_s13.py         08-12 22:45:31   ← PRIMARY 자료 존재 이후 1분
phase2.log             08-12 23:28:16
phase3.log / results   08-13 01:19:45
PRIMARY.log / verdict  08-13 01:20:25
RESULT.md              08-13 01:21:12
```
**PREREG 는 런보다 확실히 앞선다** (게이트 통과 20:20, phase1 시작 ≈20:19–20:20 역산 일치).
집계 규약·방향·문턱·MDE 는 `PREREG.md:76-99` 에 결과 전 동결돼 있고,
`analyze_s13.py` 는 그 규약(부호검정·단측·α=0.025·Holm-2·≥41/64·3규약 병기·ZERO 는 족 밖)을
**그대로 구현**한다 — 내가 조항별로 대조했다. **프로토콜 표류 없음.**
단서는 F8·F2 참조.

# 5. 점검 C — 다중비교

PREREG §4:84-86 = PRIMARY 2개에 Holm-2 · 단측 α=0.025 · ZERO 는 IUT 로 족 밖 ·
SECONDARY 는 Holm-3 양측 α=0.05/3.
- PRIMARY 개수 = **2** (혼잡·희소), RESULT.md 에서 늘거나 줄지 않았다.
- Holm-2 적용 정확 (§1.2 재검).
- ZERO 를 족 밖에 둔 것은 IUT 이므로 **통계적으로 타당**(두 조건의 교집합이 귀무를 기각하려면
  둘 다 기각돼야 하므로 α 보정 불필요).
- SECONDARY Holm-3 미적용 → F7.
**PRIMARY 다중비교는 사전등록대로다.**

---

# 6. 발견 목록

## [MAJOR] F1 — "λ 자체는 신선 시드에서 재확인됐다(54/64)" 는 **거짓**

- **주장**: `MEMORY.md` RESUME *"λ 자체는 신선 시드 재확인(54/64)"*,
  `RESULT.md:45` *"★동시에 λ 자체가 신선 시드에서 재확인됐다(54/64, p=1.0e-08) — S8/S9 에서 고른 값인데도 살아남았다."*
- **원자료가 말하는 것**: S13 의 **128/128 에피소드 전부가 S8·S9 코퍼스의 재사용**이다.
  집합 교집합으로 확인 (`S13 episodes ALSO in S8/S9 : 128/128`, crowded 64/64, sparse 64/64).
  - S8 (`tmp/S8_deploy_confirm_2026-08-10/results.jsonl`): 에피소드 시드 **32–39**,
    셀 `AUTO · AUTO-L1-1e4 · AUTO-L1-3e4 · AUTO-L1-0 · AUTO-L1-1e4-P1ONLY · AUTO-L1-1e4-TRKONLY`
    ⟹ λ ∈ {1e-3(기본), 1e-4, 3e-4, 0} 을 **바로 이 에피소드들에서 비교**했다.
  - S9 (`tmp/S9_lr_x_l1_2026-08-11/results.jsonl`): 에피소드 시드 **48–63**,
    셀 `AUTO · AUTO-L1 · AUTO-QTR-L1 · AUTO-QTR · AUTO-NT · AUTO-SEED-NULL`.
  - S13 혼잡 = S8 시드 32–39(32ep) + S9 시드 48–55(32ep) ⟹ **λ 선택 코퍼스와 100% 동일**.
  ⟹ `SHIP` vs `SHIP_L1E3` 54/64 는 **신선 시드가 아니라 λ 를 고른 바로 그 시드**에서 나온 값이다.
- **증거**: `tmp/S13_oos_2026-08-12/results.jsonl`(에피소드 키 128개) ·
  `tmp/S8_deploy_confirm_2026-08-10/results.jsonl` · `tmp/S9_lr_x_l1_2026-08-11/results.jsonl` ·
  `PREREG.md:68` (*"⛔ λ=1e-4 에 대해서는 신선하지 않다 — λ 는 S8/S9 에서 선택됐다"* — **사전등록은 맞게 썼다**)
- **제안 수정**: `MEMORY.md` RESUME 및 `RESULT.md:45` 에서 **"신선 시드 재확인" 삭제**.
  정확한 문장은 *"λ 지분은 in-sample 상한으로 37% (PREREG §6 이 '지분 **상한**' 이라 부른 그대로)"*.
  ★단 **판정 자체는 무사하다** — PREREG §3:68 이 이 비신선성을 사전에 선언하고 `SHIP_L1E3` 로
  지분을 재도록 설계했으며, 지분은 6규약 전부 50% 미만이다.

## [MAJOR] F2 — 사전등록된 고리 `ship:10_20` 이 보고에서 `0_5` 로 **바뀌었고**, "사전 지정"이라 라벨됐다

- **주장**: `RESULT.md:65` *"### S-2a 근거리 고리 `0_5` (사전 지정 1개)"*
- **원자료가 말하는 것**: 사전등록은 **`10_20` 하나만** 지정했다.
  `PREREG.md:88` *"⛔고리 족(ring family) 검정 금지 — 사전에 `ship:10_20` **하나만** 지정"*,
  `PREREG.md:102` *"S-2a 근거리 고리 `ship:10_20`"*. 초안도 동일(`PREREG_DRAFT.md:81,97`).
  그런데 `analyze_s13.py:125,128` 은 `'0_5'` 를 집계하고 `RESULT.md:65` 가 그것을 "사전 지정" 이라 부른다.
  **`10_20` 자료는 존재한다** — `SHIP` 128/128 행에 `shell_rmse` 9개 고리 전부 기록돼 있다
  (`0_5, 5_10, 10_20, 20_30, 30_50, 50_75, 75_100, 100_150, 150_inf`).
  내가 사전등록된 고리로 재도출한 값:
  | 고리 | 혼잡 중앙 | 희소 중앙 |
  |---|---|---|
  | `0_5` (보고된 것) | 0.0527 m | 0.0058 m |
  | **`10_20` (사전등록된 것)** | **0.1152 m** | **0.0096 m** |
- **증거**: `PREREG.md:88,102` · `analyze_s13.py:125,128` · `RESULT.md:65` ·
  `results.jsonl` (`shell_rmse` 키 9종 × 128행)
- **제안 수정**: `RESULT.md` §S-2a 를 **`10_20` 으로 교체**하고 `0_5` 는 탐색적으로 강등 표기.
  "(사전 지정 1개)" 라벨은 `0_5` 에서 **제거**. 이 저장소가 S6/S7 에서 대가를 치른 바로 그
  고리-족 문제이고, PREREG §4:88 이 명시적으로 막으려 한 것이다.

## [MAJOR] F3 — 사전등록된 사망조건 §8-3(S-2a)이 **러너 설계상 평가 불가**였고, 헤드라인은 강등되지 않았다

- **주장**: `RESULT.md:9` **BOTH-PASS**, 그리고 `RESULT.md:22` *"### 사전 등록된 민감도 — 전부 통과"*
- **원자료가 말하는 것**: PREREG §8-3 은
  *"S-2a 실패(근거리 고리에서 `LETKF+` 에 진다) ⟹ ... 전역 승리를 배포·응용 문장으로 쓰는 것 금지,
  헤드라인은 '먼바다 우세'로 강등"* 이라는 **사망조건**을 걸었다.
  그런데 `run_s13.py:158-162` 의 `run_letkf()` 반환 dict 에는 `shell_rmse` 가 **없다** —
  `LETKF`/`LETKF_PLUS` 행 128×2개 전부 `shell_rmse` 키 부재(재도출 확인).
  ⟹ `SHIP` vs `LETKF+` 의 고리 대조는 **처음부터 계산 불가**였다.
  `RESULT.md:67` 이 이를 인정한다(*"LETKF+ 는 셸을 기록하지 않으므로 ... 미측정"*).
  **게이트 G-1~G-5 중 어느 것도 이 사망조건의 계산 가능성을 점검하지 않는다**
  (`run_s13.py:72-119` 전문 확인).
- **증거**: `PREREG.md:126` (사망조건) · `run_s13.py:158-162` (반환 dict) · `run_s13.py:72-119` (게이트) ·
  `RESULT.md:67` · `results.jsonl` (LETKF 계열 256행에 `shell_rmse` 0건)
- **제안 수정**: "전부 통과" 를 **"S-2a 는 평가 불가로 미실행, 사망조건 §8-3 미검정"** 으로 정정.
  `SUPPORT-TRADEOFF` 가 7회 재현된 저장소에서 **근거리 증거 없는 전역 승리**를 배포·응용 문장으로
  쓰지 말 것. 다음 배치에서 `run_letkf()` 에 셸 기록 3줄 추가(비용 0, 재실행만 필요).

## [MAJOR] F4 — "사전노출 24" 는 **노출을 과소계상**한다 (실제 노출 에피소드 128/128)

- **주장**: `PREREG.md:69` *"사전 노출 24/128 (19%)"*, `MEMORY.md` RESUME *"사전노출 24 제외해도 유지"*
- **원자료가 말하는 것**: 노출은 BS-3 의 24개가 전부가 아니다. **128개 전량**이 S8·S9 에서
  이미 `AUTO-L1-1e4`(S8) / `AUTO-L1`(S9) 팔로 돌아 집계·보고됐다(F1 의 집합 증거).
  게다가 S9 커밋(`40e22f6`)과 S13 의 HEAD(`5f0846c`) 사이 커밋은 **단 1개**이고,
  그 커밋의 동작 변경은 커밋 메시지 자신의 표현으로 *"동작 변경은 λ 단일출처화뿐"* 이다
  ⟹ S13 의 `SHIP` 팔은 S8/S9 의 λ=1e-4 팔과 **사실상 같은 추정기**다.
  ⟹ S-2e("노출 24 제외") 는 실제 노출의 19% 만 제거한다.
- **증거**: `git log --oneline 40e22f6..HEAD` = 1건(`5f0846c`) · `git show HEAD` 커밋 본문 ·
  S8/S9 `results.jsonl` 셀 목록 · S13 `results.jsonl` 에피소드 집합
- **제안 수정**: PREREG §3 의 노출 문구를 *"`SHIP` 계열 절대값은 128/128 이 S8·S9 에서 이미 산출됨;
  **신선한 것은 `LETKF+` 대조뿐**"* 으로 정정.
  ★**핵심 방어는 유지된다** — `LETKF+` 는 이 128 에피소드에서 **한 번도 돌지 않았고**(PREREG:71),
  판정 통계량은 그 대조의 부호다. 그리고 **라우터**는 진짜로 out-of-sample(§3). 즉 배치의
  중심 주장은 살아 있고, 정정 대상은 **"노출 24" 라는 숫자와 그것이 준 안심**이다.

## [MINOR] F5 — G-3 이 선언한 "층 간 파랑 시드 중복 0" 은 **점검되지 않았고 사실도 아니다**

- **주장**: `PREREG.md:115` *"G-3 | 128 에피소드 전부 ra/rf/ds ON, **층 간 파랑 시드 중복 0** | 파일명 검사"*, `GATES.log` **G-3 PASS**
- **원자료가 말하는 것**: `run_s13.py:80-83` 의 `dup` 은 `(stratum,bf,ncomp,seed)` 튜플 중복만 본다 —
  **층 간 시드 교집합을 전혀 검사하지 않는다.** 그리고 실제로 교집합이 있다:
  sparse 시드 `{48..63}` ∩ crowded 시드 `{32..39,48..55}` = **`{48,49,50,51,52,53,54,55}` 8개**,
  해당 에피소드 **sparse 32개 + crowded 32개**.
- **증거**: `run_s13.py:80-83` · `GATES.log` (G-3 PASS) · `results.jsonl` 시드 집합
- **제안 수정**: 판정은 무사하다 — **Holm 은 임의 종속 하에서도 타당**하므로 두 층의 p 가
  상관돼도 FWER 통제는 유지된다. 다만 **게이트가 검사하지 않은 조건에 PASS 를 찍었다**는 사실을
  기록하고, 게이트 코드에 실제 교집합 assert 를 추가할 것.

## [MINOR] F6 — `verdict_secondary.json` 을 만든 스크립트가 **저장소에 없다**

- **주장**: RESULT.md 의 SECONDARY 표 전부(기여 #3 DEAD, #2b ALIVE, λ 지분).
- **원자료가 말하는 것**: 저장소 전역 `grep -rl "verdict_secondary" .` = **0건**.
  `analyze_s13.py:149` 는 `verdict_primary.json` 만 쓴다. ⟹ SECONDARY 수치는
  **커밋된 코드로 재현 불가**(대화형/인라인 실행으로 추정).
- **증거**: 저장소 전역 grep · `analyze_s13.py:149`
- **제안 수정**: **수치 자체는 내가 독립 재도출로 7/7 전건 확인했다**(§1.3) — 값은 옳다.
  집계 스크립트를 파일로 남길 것. (이 배치 디렉터리는 git 미추적이기도 하다 — F8.)

## [MINOR] F7 — SECONDARY 에 사전등록된 Holm-3 이 **적용되지 않았다** (판정 불변)

- **주장**: `PREREG.md:86` *"SECONDARY | Holm-3, 양측 α=0.05/3"*
- **원자료가 말하는 것**: `verdict_secondary.json` 의 p 는 전부 **미보정 양측 p** 다
  (내가 확인: 각 값이 정확히 단측 p 의 2배). Holm 보정 흔적 없음.
  내가 직접 Holm-3 을 걸어보면: 최악 원p = `SHIP_NOCONF|sparse` **7.73e-05**, ×3 = **2.32e-04**
  ≪ α=0.05/3=0.0167 ⟹ **어떤 SECONDARY 판정도 바뀌지 않는다.**
- **증거**: `verdict_secondary.json` (7개 항목) · `PREREG.md:86`
- **제안 수정**: 보고에 보정 p 를 병기. 실질 영향 없음.

## [MINOR] F8 — 동결의 증거가 **mtime 뿐**이고, 분석기는 PRIMARY 자료를 본 뒤 작성됐다

- **주장**: `RESULT.md:3` *"사전등록: PREREG.md (런 시작 전 동결)"*
- **원자료가 말하는 것**: `git ls-files wavesplat/tmp/S13_oos_2026-08-12/` = **빈 결과** —
  배치 디렉터리 전체가 **git 미추적**이다. 동결을 입증하는 것은 파일 mtime 하나뿐이며
  mtime 은 위조 가능하다. 또 `analyze_s13.py` mtime **22:45:31** 은 `phase1.log` 마지막 기록
  **22:44:45** 보다 **46초 뒤** — 분석 코드는 PRIMARY 자료가 존재하는 상태에서 작성됐다.
- **증거**: `git ls-files` · `stat` mtime 목록(§4)
- **제안 수정**: **프로토콜 표류는 없다** — 나는 `analyze_s13.py` 를 `PREREG.md:76-99` 조항과
  1:1 대조했고 부호검정·단측·α=0.025·Holm-2·문턱 41/64·3규약 병기·ZERO 족 밖이 전부 일치한다
  (예외는 F2 의 고리 하나). 다만 **배치 디렉터리를 커밋해 동결을 암호학적으로 못박을 것.**
  (S8 은 `FROZEN_md5.txt` 를 남겼다 — S13 에는 없다.)

## [MINOR] F9 — G-1 의 "effective-config" 는 **실효값이 아니라 의도표**다 (E1 이 08-11 에 잡은 그 실패모드)

- **주장**: `PREREG.md:113` *"G-1 | 모든 결과 행에 ... **effective-config** 기록"*
- **원자료가 말하는 것**: `run_s13.py:181-183` 은 `cfg` **입력 dict 를 그대로** 직렬화한다 —
  구성된 추정기에서 되읽은 값이 아니다. 이는 `5f0846c` 커밋 본문이 스스로 경고한
  *"게이트가 실효값이 아니라 **의도표**를 인쇄"* 와 같은 형태다.
  **완화 사실 2가지**: (a) 게이트가 `SequentialUpdater(cfg).lambda_l1 == cfg['lambda_l1']` 로
  **실제 되읽기를 1회** 한다(`run_s13.py:112`); (b) 내가 실증했듯 override 3종이
  **320/320 행에서 결과를 바꿨다** ⟹ 배선은 실제로 살아 있다.
  **잔여 사각**: `TRACK_PHASE_PREDICT`(`sequential_update.py:83`, 모듈 상수) 는 config 에 없어
  **어떤 결과 행에도 기록되지 않는다** — `MEMORY.md` 가 헤드라인을 움직인 결정적 레버라고
  지목한 바로 그 불리언이다.
- **증거**: `run_s13.py:181-183`, `:112` · `sequential_update.py:83` · `run.py` DEFAULT_CONFIG(21키, `TRACK_PHASE_PREDICT` 부재)
- **제안 수정**: 결과 행에 모듈 상수 스냅샷(`TRACK_PHASE_PREDICT` 포함)을 추가 기록.
  이번 배치는 단일 HEAD(704/704 동일)라 **실질 위험은 없다.**

## [MINOR] F10 — `LETKF+` 의 `obs_noise`·`max_obs` 는 명시되지 않고 **기본값 폴백**이다 (값은 일치)

- **주장**: `PREREG.md:31` *"`LETKFGrid(40×30, n_ensemble=600, mult 1.0, add 0.005, **obs_noise 0.05**, loc 50, **max_obs 40**)`"*
- **원자료가 말하는 것**: `run_s13.py:136-137` 은 `obs_noise`·`max_obs` 를 **넘기지 않는다**.
  `letkf_grid.py:48-50` 의 기본값이 `obs_noise=0.05, max_obs=40` 이라 **결과적으로 값은 같다**(내가 확인).
  PREREG §2:38 이 *"기준선 팔은 ... 명시한다 — 폴백 오염 재발 방지"* 라 선언한 것과 형식적으로 어긋난다.
- **증거**: `run_s13.py:136-137` · `letkf_grid.py:48-50` · `PREREG.md:31,38`
- **제안 수정**: 값이 우연히 일치하므로 **수치 영향 0**. 다음 러너에서 명시할 것.

## [MINOR] F11 — 비용 수치 미세 불일치

- **주장**: `RESULT.md:5` *"총 ≈5.5 h"*
- **원자료가 말하는 것**: `sum(wall_s)` = **17,863 s = 4.96 h**. 차이는 행 간 오버헤드/유휴로 설명 가능하고
  `RESULT.md` 도 "≈" 를 붙였다.
- **증거**: `results.jsonl` `wall_s` 704행
- **제안 수정**: *"계산 시간 4.96 h(행 내 합), 벽시계 ≈5.5 h"* 로 분리 표기.

---

# 7. 일치 확인 목록 (숫자 병기)

| 주장 | 재도출값 | 판정 |
|---|---|---|
| 704런 | 704행 | **일치** |
| 에러 0 | `error` 행 0 | **일치** |
| 생성 0 · 시드 소모 0 | h5 128개 전량 08-10~11 mtime, 배치는 08-12 20:20 시작 | **일치** |
| 라우터가 안 본 시드 32–63 | 스윕 코퍼스 시드 {0,1,2,3,4}, 도너 {0,2,3,4} — 교집합 0 | **일치** |
| 혼잡 56/64, p_adj 2.8e-10 | 56/64, p=2.781e-10, Holm p_adj=2.781e-10 | **일치** |
| 희소 63/64 | 63/64, p=3.524e-18, p_adj=7.047e-18 | **일치** |
| 제로필드 양 층 64/64 | crowded 64/64 · sparse 64/64, p=5.421e-20 | **일치** |
| 문턱 41/64 | `PREREG.md:85` 결과 전 동결 | **일치** |
| 사전노출 제외 44/52 · 51/52 | 44/52 p=2.020e-07 · 51/52 p=1.177e-14 | **일치** (단 F4) |
| 15comp 29/32 · 30comp 27/32 | 29/32 p=1.278e-06 · 27/32 p=5.654e-05 | **일치** |
| 공개 LETKF 였다면 63/64 · −32.3% | 63/64, med(rel)=−32.32% | **일치** |
| 튜닝 후 56/64 · −28.8% | 56/64, med(rel)=−28.82% | **일치** |
| λ 지분 37%, 문턱 50% 미저촉 | 37.2%(med(rel) 마진차); 6규약 28.9–45.1%, 6/6 <50% | **일치·강건** |
| λ 재확인 54/64 | 54/64, 단측 p=9.982e-09 | 수치 **일치**, "신선 시드" 라벨 **거짓 → F1** |
| 기여 #3 DEAD | NOCONF 끄면나빠짐 혼잡 10/64 · 희소 16/64 | **일치** |
| 기여 #2b ALIVE | NODISP 끄면나빠짐 62/64 양 층, med(rel) +70.2% / +777.8% | **일치** |
| 희소 −93% 는 표현 아티팩트 | 희소 `SHIP` 중앙 nRMSE **0.0147**(1.5%), `LETKF+` 0.2193 | 수치 **일치**; 아티팩트 해석은 **ARGUED**(X1/X2 인용, 이 배치는 표현 바닥을 재측정하지 않았다) |

---

# 8. 총평

**PRIMARY 판정(BOTH-PASS)은 원자료에서 그대로 재현된다.** 704행 전량이 깨끗하고(에러·중복·NaN 0),
프레임 정합이 128/128 완벽하며, 기준선 오염(점검 A)은 구조적으로 발생할 수 없고,
라우터 out-of-sample(점검 D)은 세 튜닝 코퍼스를 전부 파싱해 확인했다. 다중비교(점검 C)는 사전등록대로다.
**기준선을 스스로 강화해 마진을 −32.3% → −28.8% 로 깎은 것은 실제로 그렇고, 정직한 설계다.**

정정이 필요한 것은 **판정이 아니라 판정을 둘러싼 서술 4건**이다:
1. **λ 는 신선 시드에서 확인되지 않았다** — 128/128 이 λ 선택 코퍼스다 (F1).
2. **사전등록된 고리는 `10_20` 인데 `0_5` 가 "사전 지정" 이라며 보고됐다** (F2).
3. **근거리 사망조건 §8-3 은 러너 설계 때문에 평가조차 되지 않았는데 "민감도 전부 통과" 로 적혔다** (F3).
4. **노출은 24 가 아니라 128 이다** — 신선한 것은 `LETKF+` 대조와 라우터뿐 (F4).

셋(F1·F3·F4)은 모두 같은 방향을 가리킨다: **이 배치가 산 것은 "라우터가 in-sample 이 아니었다"
하나이고, PREREG §0 이 정확히 그렇게 썼다**(*"이 배치는 결과의 신뢰만 산다"*).
서술이 그보다 넓게 나간 지점만 좁히면 된다.
