# M7 AMENDMENT 1 — 사용자 결정으로 `DATA` 팔을 되살린다 (2026-08-29)

> **런 0회 · 학습 0회 · 추정 0런.** `PREREG.md` · `gates_m7.py` · `premise_m7.*` **무수정**
> (md5 대조 §5.3 에 첨부 — 다섯 줄 전부 `OK`).
> 비용 smoke 3건은 **저장소 밖**(scratchpad)에서 동결 스크립트를 그대로 호출해 돌렸고
> 산출물은 삭제했다. 저장소 파일 변경 0건(`git status` 확인, `results/direction3/*.pt` mtime 7월 그대로).
> ⛔**이 개정은 어떤 문턱도 낮추지 않는다.** 유일한 변경은 **주장 가능 범위의 축소**(§2)이고,
> 나머지는 전부 **절차의 명문화**(§5–§7)다.
> 열람 범위 = §9.

---

## 0. 결정

**결정 일시**: 2026-08-29 (이 문서 작성 시각 13:43 KST). **전달 경로**: 오케스트레이터 경유 사용자 축자.

> ***"포함해서 돌려줘 가파른 데이터가 들어갈 때 방향성을 찾을 수도 있으니까"***

**결정 내용**: 동결 2×2 **전체**(`BASE`·`LOSS`·`DATA`·`BOTH`) + 음성대조 `BASE-B` 를 **설계 그대로**
실행한다. ⛔**축소 실행이 아니다**(`DATA_ARM_REVIEW` 권고 2 축자: *"축소는 설계 변경"*).

⚠**이 결정이 승인하지 않은 것**: **E1(실측 12 레코드)**. E1 은 *"실측에 learned 강제"* = 비승인
정책이고 `W12`·`B-2` 때 각각 **별도 사용자 승인**으로 풀렸다(`PREREG §3 E1` 축자).
위 축자는 **팔 구성**에 대한 결정이지 E1 실행 승인이 아니다 ⟹ **§7 Step 5 는 승인 단계로 분리**한다.

---

## 1. ★분기표를 넘어선다 — 감추지 않고 공시한다

### 1.1 결과 전에 동결된 분기표가 무엇을 지시했나 (축자)

`DATA_ARM_REVIEW.md §2`, **`M1` 결과 전에** 작성·동결:

> **(hold, hold)** — 비선형성 비매개 · **"인과 사유("실패를 고친다")는 남지 않는다** — 고칠 측정된
> 결손이 소거표 어디에도 없다 … 남는 것 둘: ①**S-1** … ②널일 때의 소거표 한 줄 + 기여② 강화 …
> **우선순위 강등.**"

같은 문서 §2 말미:

> "⛔**"남는 게 없다" 를 물었으므로 명시한다**: (hold, hold) 와 (break, break) 에서 DATA 팔의
> **고유 사유는 남지 않는다**."

### 1.2 그 분기가 실제로 나왔다 (동결 `verdict_m1.json` 에서 직접 읽음)

| | 라벨 | `SHIP<SPEC25` | rel median | p | 음성대조 `SPEC25<ZERO` |
|---|---|---|---|---|---|
| `LIN` | `HOLDS (CONFIRMATORY)` | **22/24** | −43.756% | 1.794e-05 | **24/24** |
| `HOS` | `HOLDS (CONFIRMATORY)` | **21/24** | −33.654% | 1.386e-04 | **24/24** |

`pair_key = ["hold", "hold"]`. (md5 `31c2d6e2…`, §5.3 에 동결.)

### 1.3 ★공시 문언 — 이 배치를 인용할 때 **동반 의무**

> ***"이 배치는 결과 전에 동결된 분기표를 넘어서 실행됐다. `DATA_ARM_REVIEW.md §2` 는*
> *`M1 (hold, hold)` 에서 `DATA` 팔의 우선순위 강등을 지시했고, `M1` 은 실제로 `(hold, hold)` 로*
> *나왔다. 실행 사유는 그 분기표가 다룬 「고장 수리(인과 사유)」가 아니라 사용자가 지시한*
> *「탐색적 방향 탐지」이며, 그러므로 `DATA` 축의 결과는 확증으로 승격되지 않는다."***

⛔이 문장은 **결과가 무엇이든** 붙는다. 양성이면 *"분기표를 어겼는데 맞았다"* 가 아니라
*"탐색적으로 방향이 잡혔고 확증은 새 사전등록이 필요하다"* 이고, 널이면 *"분기표가 옳았다"* 다.

### 1.4 `REVIEW_2026-08-29.md` 의 지위

그 문서의 결론 **(나) 보류**는 사용자 결정으로 **대체**된다. ⛔**그 문서를 수정하거나 지우지 않는다** —
이제 *"미집행 사유의 기록"* 이 아니라 ***"집행 전에 적힌 반대 논거의 기록"*** 으로 남고, 그것이
결과를 읽을 때의 사전 기대(§3)를 지탱한다. ⚠그 문서가 적어 둔 착수 트리거 4개 중 **어느 것도
발화하지 않았다** — 사용자의 사유는 트리거 목록 밖의 새 사유다(§2). 그렇게 적는다.

---

## 2. ★사유의 재작성 — 바뀐 것은 **판정이 아니라 사유**다

### 2.1 두 질문은 다르다

| | 물은 것 | 답 |
|---|---|---|
| **`M1` 이 물은 것** | *"가파른 비선형 바다에서 배포 사전이 **깨지는가**"* | **안 깨진다**(22/24 · 21/24, 확증) |
| **사용자가 묻는 것** | *"그 바다에 **맞춰 학습하면 나아지는가**"* | ★**한 번도 물은 적이 없다** |

⟹ ***"깨지지 않는다" 가 "개선 여지가 없다" 를 함의하지 않는다.***
`M1` 은 **평가측**에서 참값을 바꿨고, `DATA` 팔은 **학습측**에서 분포를 바꾼다.
`DATA_ARM_REVIEW §1.4` 가 이미 *"학습측에서 가파름×비선형 결합 칸은 시도된 적이 없다"* 를
코드 근거(`build_dataset.py:39,65,66` · `gen_mixed.py:344` · `d3_beamform_cond_diffusion.py:86,144`)
로 확정해 뒀고, **그 사실은 `M1` 결과와 무관하게 참이다.**

⚠**그러나 분기표는 무효가 아니다.** 분기표가 다룬 것은 *"고칠 **측정된** 결손이 있는가"* 였고,
`(hold, hold)` 에서 그 답은 여전히 **없다**이다. ⟹ ★***사유가 바뀐 것이지 판정이 바뀐 것이 아니다.***

### 2.2 `DATA` 팔 근거의 교체

| | 이전(사전등록 §1.4-2) | **이 개정 이후** |
|---|---|---|
| 근거 | 전방모델의 선형성 = **측정된 결손을 고친다**(인과) | **탐색적 방향 탐지** — 측정된 결손은 없고, 시도된 적 없는 학습측 칸을 **한 번 열어 본다** |
| 결과 등급 | 사전등록 §6 라벨 그대로 | ★**`EXPLORATORY` 로만 읽는다**(아래 2.3) |

### 2.3 ⛔결과 등급 강등 (결과 전 선언, **축소 방향 전용**)

`gates_m7.label()` 이 내는 라벨 문자열은 **코드가 권위**이고 이 개정은 그것을 건드리지 않는다.
바뀌는 것은 **보고할 때의 읽기 등급**이며, 다음 셋을 결과 전에 못 박는다:

1. **`T-2_data`(분포 주효과)와 `T-3_interaction`, 그리고 라벨 `DISTRIBUTION-EARNS` ·
   `BOTH-ADDITIVE` · `INTERACTION` 의 분포 성분** = **`EXPLORATORY`**.
   ⛔확증(CONFIRMATORY) 승격 금지 · ⛔논문 기여 목록 승격 금지 ·
   ⛔후속 배치의 **우선순위 근거**로 인용 금지(`B-23` 금지 항목과 같은 규율).
   양성이면 허용되는 최대치 = *"탐색적으로 방향이 잡혔다. 확증은 신선 시드·새 사전등록으로만."*
2. **`T-1_loss`(손실 주효과)·음성대조 `BASE-B`·중단 규칙·`S-1`·E2 하드 바** = **원 사전등록 지위 그대로.**
   (사용자 결정은 `DATA` 팔의 **포함 여부**에 대한 것이고 `LOSS` 축 사유는 `M1` 과 형식상 독립이다 —
   `REVIEW_2026-08-29 §1.4` 축자 확인.)
3. ★**`DATA` 팔이 `EXPLORATORY` 라도 `S-1`(`DISTRIBUTION-SHIFT-SURVIVES`)은 강등되지 않는다.**
   `S-1` 의 주어는 *"HARD 로 학습한 넷이 **동결 B-12 코퍼스에서 헤드라인을 유지하는가**"* 이고
   그 문턱은 `B-12` 에서 상속된 동결값(31/34/31, 패리티 검사 `True` — §5.7)이다.
   `S-1` 은 실측 이득과 무관하며 심사 단서 ③을 **학습측에서** 겨눈다.

⟹ ★**이 개정의 순효과는 "주장할 수 있는 것이 줄어드는 것" 하나다.** 문턱·표본·검정은 불변이다.

---

## 3. ★결과 전 기대치 — 다시, 그리고 더 낮게 못 박는다

`PREREG §1.4` 축자: ***"E1(실측)의 가장 가능성 높은 결과는 널이다."*** 그 문장을 이 개정이
**철회하지 않는다.** 근거를 여기 다시 모아 둔다(전부 동결 자료, 오늘 재확인한 것은 ✅표시):

| 선행 | 무엇을 했나 | 결과 |
|---|---|---|
| `wp_e2e_2026-07-25` | 장 오차 보조손실(λ_field) 미세조정 = **`LOSS` 팔의 직계 선례** | ✅**PRIMARY FAIL** — 21/40 · median Δ −0.0007 · **p=0.30**(원문 :26,:35,:148 재확인) |
| 07-03 `ds` vs `dslin` | HOS 비선형 관측 **혼합 학습** | ✅**HOS 평가에서도 이득 0** — `b1_ds` 0.24622 vs `b1_dslin` 0.24032, 짝 **4/15**(`hos_eval_new.json` 재확인) |
| fan-geometry(07-24) | 실측-유사 **기하**로 재학습 | 합성 PRIMARY FAIL + 실측 무개선 ⟹ **축 폐쇄** |
| `cd_task`(07-22) | task-loss 미세조정 | HOS 재검에서 **사망** |
| `mixed_retrain` PREREG 축자 | — | *"loss / architecture … **That card is spent** (three independent FAILs)"* |
| ★`B-8b` | 실측에서 **학습을 통째로 제거** | `V2K` 29/72 = 무기술이고 **학습 경로 21/72 = 무작위보다 유의하게 나쁨**(p=2.7e-04) ⟹ *"체크포인트만 더 잘 만들면 실측이 열린다"* 에 **이미 반대 증거** |
| ★`B-26`(탐색적) | (Hs,Tp) 지지집합 소속과 성능 | ***"지지집합 소속이 성능을 예측하지 못한다"***(안쪽 `SS-LOWH` 가 최약) ⟹ **`DATA` 팔 전제에 직접 불리** |
| ★`W12`·`B-2c` | 실측 과적합의 **발생 지점** | **시험 시점 VARPRO**(관측 목적함수 −63.2% 뒤 장 오차 악화) · 실측에서 λ 벌점을 전면 제거해도 7–8/12 불변 ⟹ **학습 손실은 거기 닿지 않는다** |

⟹ ★**결과 전 기대: 가장 가능성 높은 결과는 널이다.** 그리고 그것은 이 배치의 실패가 아니다 —

* **널의 값(사전등록 §6-4)**: 소거표에 **"학습 레시피(분포·감독)"** 한 줄이 붙고, 논문 기여 ②
  (*제안 표현의 실측 음성*)가 *"쉬운 학습 분포나 대리 손실의 산물이 아니다"* 로 강해진다.
  ⚠**단 E2 바를 통과했을 때만** — 못 통과하면 `UNINTERPRETABLE-NULL` 이고 **아무것도 못 쓴다.**
* ★**사용자의 사유에서는 널도 방향이다**: *"방향성을 찾을 수도 있으니까"* 에 대한 답이
  *"이 방향에는 없다"* 로 확정된다. 그 답을 얻으려면 **MDE 를 함께 적어야 한다**(§4 = 12개 중 11개).

⛔**금지(양방향, 이미 명문화된 것 재확인)**: *"재학습이 무익"* 확장 금지(`B-8b` 는 **학습 없는
대안**을 쟀다) · *"완전-HOS 학습 무익"* 확장 금지(M7 HARD 의 비선형은 **Stokes 자기조화 한정**
= `M1` HOS M=3 의 부분집합, `DATA_ARM_REVIEW §1.5`) · `B-23` 우선순위 주장 금지.

---

## 4. ⛔이 개정이 **바꾸지 않는** 것 (전수 확인)

| 항목 | 동결 출처 | 상태 |
|---|---|---|
| 팔 정의 (5칸 `CELLS`, 사전학습 3 `PRETRAINS`, 참조 `SHIPPED`·`BANK`) | `gates_m7.py:26-44` | **불변** |
| 학습 상수 (40k/4k, lr 2e-4/5e-5, bsz 32, λ_field∈{0,1}, T_LO 0.25, clip 5.0, NGRID 20) | `gates_m7.py:47-57` | **불변** |
| 평가 3종 (E1 실측 12·win 3·확산시드 3 / E2 `B12` 48·추정시드 3 / E3 `M1` 48 보고전용) | `gates_m7.py:60-73` | **불변** |
| 검정·가족·MDE (부호검정, Holm 3, α=0.05, **11/12**) | `gates_m7.mde()` = `{min_wins: 11, p 0.003174}` (오늘 재실행) | **불변** |
| 중단 규칙 `NEG_CONTROL_ABORT = 10` (`BASE` vs `BASE-B` 10/12) | `gates_m7.py:79` | **불변** |
| E2 하드 바 `R1_MARGIN = 0.010` · `S-1` 문턱 31/34/31 | `gates_m7.py:181,213` (+`B-12` 패리티 `True`) | **불변** |
| 결과 전 해석 규칙 5갈래 + `label()` · `label_s1()` | `gates_m7.py:317-367` | **불변** |
| 게이트 16개 | `gates_m7.gates()` | **불변**(⚠`G-9` 공허 PASS 는 §5.5 에서 **러너 의무**로 막는다 — 게이트는 안 고친다) |
| 앵커 물리 스크린(`k·Hs>0.44` ⟹ 학습 앵커에서만 `LJ18a` 제외, **평가에서는 미제외**) | `gates_m7.py:52-54` · `G-8` | **불변** |
| 시드 구간 분리 (학습 ≥2e9) | `gates_m7.py:55` | **불변** |
| ★**E1 사용자 승인 요건** | `PREREG §3 E1` | **불변** — §0 참조 |
| ⛔어떤 가지도 배포 기본값을 바꾸지 않는다 | `PREREG §6` 말미 | **불변** |

**md5 대조(오늘)**: `PREREG.md` `ad53ecbf…` · `gates_m7.py` `7c4cc145…` · `premise_m7.py` `c0f19f48…` ·
`premise_m7.json` `874b1393…` · `PRIOR_WORK.md` `bbd42472…` — **5/5 `OK`, 드리프트 0**.

---

## 5. 실행 가능성 점검 (런 0회 · 배치 0행)

### 5.1 ⛔★[BLOCKER-1] **하네스 4개가 아직 존재하지 않는다**

`PREREG §9` 축자: `gen_m7.py` · `train_m7.py` · `run_m7.py` · `analyze_m7.py` = **(미작성)**,
그리고 *"미작성 4개는 착수 시 작성하고, 착수 전에 `FROZEN_md5.txt` 를 갱신·동결한다."*
디렉터리 실측(오늘): 존재하는 것은 `PREREG.md`·`PRIOR_WORK.md`·`DATA_ARM_REVIEW.md`·
`REVIEW_2026-08-29.md`·`gates_m7.py`·`premise_m7.{py,json}`·`FROZEN_md5.txt` 뿐이다.

⟹ ★**지금 붙여넣어 돌릴 수 있는 명령은 없다.** §7 Step 0 이 그 작성 단계이고, **작성 요건은 §6
에서 결과 전에 동결**한다. 이것이 이 개정의 [BLOCKER] 이고 유일한 것이다.

### 5.2 ★디바이스 판정 — ⛔**`CUDA_VISIBLE_DEVICES` 를 세우지 마라**

* **`M7` 사전등록은 device 를 한 번도 지정하지 않는다**(grep 실측: 배치 `*.py`·`PREREG.md` 에
  `cuda`·`device`·`CUDA_VISIBLE_DEVICES` **0회**, `GPU` 는 비용 문맥 4회뿐).
* **코드 사실**(`M1 RUNBOOK §1` 과 같은 근거, 오늘 재확인): `learned_init.py:133-134` 가
  `device = 'cuda' if torch.cuda.is_available() else 'cpu'` 로 **스스로 고르고**,
  `config['device']`(실측 하네스가 `'cpu'` 로 넣는 값)는 **읽지도 않는다**.
  ⟹ `CUDA_VISIBLE_DEVICES=''` 를 세우면 **제안이 조용히 CPU 로 내려가 측정된 적 없는 팔**이 된다.
  E1 기준선 계보(`B-2` PROP)·E2 계보(`B-12`)가 전부 GPU 가시 상태에서 측정됐다.
* **그리고 이 배치는 학습 배치다** — Phase A/B 는 GPU 없이는 예산이 성립하지 않는다(§5.4).
* ⚠**대조**: `M4`·`M5b`·`M8` 은 `CUDA_VISIBLE_DEVICES=''` 를 **torch 임포트 전에** 박는 것이
  사전등록 요건이었다(그 배치들은 GPU 금지 설계). **`M7` 은 그 부류가 아니다.**
* ✅**GPU 상태(오늘 13:31)**: RTX 4080 SUPER 16,376 MiB 중 **453 MiB 사용, 프로세스 0개** ⟹
  `M1` 점유 해제 확인(`PREREG` 머리의 실행 시점 조건 소진).
* ⚠**운영 제약은 변수가 아니라 순서다**: Phase A 3회를 **동시에 띄우지 마라**(단일 GPU).
  E2/E3 샤딩은 8분할까지 선례가 있다(`M1` `est_0..est_7`).

### 5.3 ★동결 드리프트 — 배치 자신은 **0건**, 그러나 **임포트 경로는 동결된 적이 없다**

* `md5sum -c FROZEN_md5.txt` = **5/5 `OK`**. `M1 RUNBOOK` 이 08-28 에 적은 *"`M7` 은 깨끗하다"* 가
  오늘도 참이다. ⟹ **덮어쓸 해시가 없다**(줄 덧붙이기 규칙은 §7 Step 0c 에서 처음 발동한다).
* ⛔**그러나 `FROZEN_md5.txt` 에는 배치 자신의 5파일밖에 없다** — `PREREG §9` 가 요구한
  *"배치-내 임포트의 md5"* 는 **아직 한 줄도 없다**. 이것은 저장소가 이름 붙인 결함
  (*"동결한 적이 없다"* — 탐지 자체가 불가능한 상태)이다. ⟹ **Step 0c 에서 아래 기준선을 덧붙인다.**
  ★**오늘 측정한 값을 여기 적어 둔다**(Step 0c 가 이 값과 다르면 그 사이에 파일이 움직인 것이다):

```
a5d71fafa08f50fa1b6fda9650863392  tmp/B2_REAL_2026-08-23/gates_b2.py
01f7667f9762f18f8293a331a4a6219c  tmp/B2_REAL_2026-08-23/run_b2.py
68dcd2073c4ec60897645faa29b16172  tmp/B2_REAL_2026-08-23/scene_station.py
9d3dada84a71722dc81089bcd7fe3f80  tmp/B2_REAL_2026-08-23/zero_ref.py
97f8d2cbe3f355251eb8afe146176d65  tmp/bankmmse_port_2026-07-29/make_gif_bank_deployed.py
09e7b03b1b5ac098cfce2fddacffd482  tmp/B12_HOS_EXTEND_2026-08-23/run_b12.py
2186b01913559f9205653c86950ecdc8  tmp/B12_HOS_EXTEND_2026-08-23/gates_b12.py
37d0ddd3aca03118ae41d52c8324e0b5  tmp/B11_HOS_BASELINE_2026-08-23/gates_b11.py
ae3e2c21321760608108205342a442b9  tmp/S16_BASELINE_2026-08-14/run_s16.py
7991d24a8a4e31580d888ddf766ec1ca  tmp/S16_BASELINE_2026-08-14/letkf_spec.py
31c2d6e23a8bfdf5beac69369c34073f  tmp/M1_NONLIN_REAL_2026-08-28/verdict_m1.json
f72c811e1f85a9df910f78ee6ff16b90  tmp/M1_NONLIN_REAL_2026-08-28/gen_manifest_m1.json
a64be4101242f500f4d6bcf491ba2af4  tmp/mixed_retrain_2026-07-26/gen_mixed.py
5f7a51a2e40a184fff0d967359bb6dc3  tmp/mixed_retrain_2026-07-26/train_mixed.py
2ea7fc4caa7445c9e3af718d5b5f4786  tmp/wp_e2e_2026-07-25/e2e_finetune.py
4926acd6a625f24adfd6cd5b38c5b589  scripts/d3_beamform_cond_diffusion.py
dfcdf182d62c6c91d3b2e59e83fc0428  scripts/d3_setdiff_prior.py
683928cd6d8e532e51c4c635b36c2a23  scripts/nonlinear_forward.py
5f9e1117609b98879d042fef98cad526  wavesplat/model/tnse.py
3e4a6fe1b34b84313a33cba443cf7bfe  wavesplat/model/spectral_gaussian.py
10e407348291e0438a94f459f4757c27  wavesplat/run.py
1b473e066cb02e43f39a9887a090c7d4  wavesplat/optimizer/sequential_update.py
5ebcf5a3d6c11d44f41eec920e38594f  wavesplat/optimizer/learned_init.py
039e2273264893629c0288de92b8903b  wavesplat/optimizer/varpro.py
44d83b26cd61f11f5e59a9ee01808e22  results/direction3/setdiff_prior.pt
d81d91152ec7e916c687a40272dccb7a  results/direction3/beamform_cond_diffusion_mixed_v1.pt
0b2917fc97f8589a88cf860a0e83d534  tmp/M7_SUPERVISION_2026-08-28/DATA_ARM_REVIEW.md
0c36723bacf6e3ca2841c45e65637c78  tmp/M7_SUPERVISION_2026-08-28/REVIEW_2026-08-29.md
```

  ⚠`git_head = c428d47834cfeab67aeb86a1ba0894d65ccb3569`, 작업트리 **clean**(추적 파일 수정 0).
  ⚠`sequential_update.py` 는 08-28→08-29 에 실제로 움직였고(`M3` 가 공시), 위 값은 **이동 후**의
  현재값이다 — `M3` 의 현재 동결 행(`1b473e06…`)과 **일치**한다.
  ⛔**배포 체크포인트 sha256** = `636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9`
  (오늘 `sha256sum` 재계산 = `learned_init.py:62 DEFAULT_CKPT_SHA256` 축자와 일치 ⟹ `G-3` 사전 통과).

### 5.4 ★비용 실측 (`PREREG §7` 의 착수 전 의무 이행)

**200 스텝 smoke, 동결 스크립트 그대로, 산출물 삭제 완료.** GPU = `cuda`(RTX 4080 SUPER).

| 항목 | 사전등록 §7 | **오늘 실측** | 비 | 판정 |
|---|---|---|---|---|
| Phase A (사전학습, `gen_mixed` BASE 스트림) | 117.79 ms/step ⟹ 1.31 h | **109 ms/step ⟹ 1.21 h** | 0.93× | ✅ |
| Phase B (미세조정, `make_batch` 장면, λ_field=1) | 43.5 ms/step ⟹ 2.9 min | **46 ms/step ⟹ 3.0 min** | 1.06× | ✅ |
| Phase B (같은 장면, λ_field=0 대조) | — | **33 ms/step ⟹ 2.2 min** | — | 기록 |
| Phase B **on `gen_m7` 장면**(보수 추정) | ≈120 ms/step ⟹ 8 min | **≈109 ms/step ⟹ 7.3 min** | 0.91× | ✅ |

* 추정의 근거(장면 생성 단가 분리 측정): `gen_mixed` 32-장면 배치 **중앙 83 ms**(n=60, p10 65 · p90 110) ·
  `e2e_finetune.make_scene_batch(32)` **20 ms** ⟹ 이식 시 +63 ms/step.
* ⟹ **전부 2배 이내** ⟹ `PREREG §7` 축자 *"2배 이상 어긋나면 착수하지 않고 사전등록을 고친다"* 의
  **착수 조건 충족. 사전등록 수정 사유 없음.**
* ⚠**미측정 하나**: HARD 스트림의 **2차 Stokes 전방모델 오버헤드**. `gen_m7.py` 가 없어 오늘 잴 수
  없다 ⟹ ★**Step 1a 에서 같은 200스텝 규약으로 재측정**하고, 표와 2배 이상 어긋나면 **거기서 멈춘다**.

**평가 비용(동결 로그에서 읽음, 런 0회)** — ⚠사전등록 §7 은 E2 를 *"8팔 × 11.9 s"* 단일 단가로
적었는데 **팔마다 단가가 다르다**:

| | 단가(동결 실측) | 계산 | 소계 |
|---|---|---|---|
| E1 | `B2_REAL/results.jsonl` 중앙 **4.16 s** | 6 학습팔×12×3 + `BANK` 12 = **228 런** | **≈16 min** |
| E2 학습팔 | `B12/results_b12.jsonl` `SHIP-NOW` 중앙 **8.1 s**(n=144) | 6팔×48×3 = 864 런 | ≈1.94 h |
| E2 `SPEC25` | 같은 파일 **37.8 s**(n=144) | 48×3 = 144 런 | ≈1.51 h |
| E2 `ZERO` | **런 0회** — `B-12` 규약에서 `zero_nRMSE` 는 각 행에 동봉된다 | — | 0 |
| E3 | E2 와 같은 구성 | 48 에피소드(LIN 24+HOS 24) | ≈3.4 h |

⟹ **학습 ≈4.3–4.7 h** + **평가 ≈7 h 단일 / 8샤드 시 ≈1.2 h** ⟹ `PREREG §7` 의 *"총 하룻밤"* 과 정합.

### 5.5 ★게이트 자가진단 + **필터 항진 검사**(`M4 G-13` 부류) — 결함 4건, 전부 **러너 의무**로 막는다

`gates_m7.py` 를 결과 0행에서 실행·탐침했다(스크래치, 배치 파일 무수정):

| # | 발견 | 부류 | 처방(⛔게이트는 고치지 않는다) |
|---|---|---|---|
| **F-1** | ⛔**`G-9` 가 빈 매니페스트에서 PASS** — `all(... for s in man.get('train_scene_seeds_sample', []))` 는 **빈 리스트에서 공허하게 참**. 16개 중 이것 하나만 그렇다(나머지 15는 빈 매니페스트에서 전부 FAIL) | `M4 G-13` 부류(항등적으로 참) | `run_m7`/`train_m7` 이 매니페스트에 **≥1개 시드 표본을 반드시 채우고**, `analyze_m7` 이 `len(...)>0` 을 **선행 assert** 로 검사한다 |
| **F-2** | ⛔**`contrasts()` 가 결측 칸을 조용히 건너뛴다**(`if None in (b,l,d,o): continue`) ⟹ 한 칸이라도 빠지면 **n 이 12→11 로 줄어드는데 `label()` 은 그대로 돈다**. MDE 11/12 는 **n=12 에서 동결**된 값이다 | 조용한 표본 축소 | `analyze_m7` 은 **세 대비 전부 `n == 12`** 일 때만 라벨을 낸다. 아니면 `INCOMPLETE` 로 멈추고 라벨을 내지 않는다 |
| **F-3** | ⛔**E2 행에 `cell=='SPEC25'`·`'ZERO'` 가 없으면 `S-1` 이 실패가 아니라 `BASELINE-NO-SKILL-HERE` 라는 *무효 라벨*로 조용히 착지**한다(`neg_control n=0 ⟹ ok=False`) | 침묵 오착지 | `run_m7` 이 `SPEC25` 행을 실제로 기록하고, **`ZERO` 행은 각 행의 `zero_nRMSE` 열에서 합성**한다(`B-12` 규약; 런 0회). `analyze_m7` 이 `neg_control['n'] == 48` 을 선행 검사 |
| **F-4** | ⚠**`mechanism()` 이 `varpro_obj_before` 없는 행에서 조용히 `{}`** 를 반환 | 바 없음(기술 전용)이라 무해 | 행에 두 값을 기록하고, 없으면 **RESULT 에 "기제 부차 미측정" 을 공시**한다 |

✅**항진 아닌 것**: `G-2`·`G-4`·`G-5`·`G-5b`·`G-10`·`G-11`·`G-12`·`G-13`·`G-14`·`G-15` 는 키가
없으면 **FAIL** 로 떨어진다(`None == 0` 이 거짓). `G-6`·`G-7` 은 존재 가드를 갖고 있다.
✅`_b12_threshold_parity()` = **`True`**(손복사 아님, `gates_b12.py` 에서 되읽음).
✅`mde()` = `{n: 12, holm_first_alpha: 0.016667, min_wins: 11, p 0.003174, p(10/12) 0.019287}` — 문서와 일치.

★**이 넷은 전부 「계산되는데 검사되지 않는다」 부류의 변종**이다(저장소 누적 5회). 하네스가 아직
없으므로 **게이트를 고치지 않고 러너 계약으로 막는 것**이 가능하다 — 그것이 §6 이다.

### 5.6 검사기 (오늘 실행, `M7` 관련 줄만)

| 검사기 | `M7` 결과 |
|---|---|
| 0. 자가진단 4종(`check_batch_artifacts`·`check_gate_refs`·`check_gate_blindspots`·`check_gen_vs_gates`) | ✅전부 통과 |
| 1. 배치 산출물 | `M7_SUPERVISION_2026-08-28  OK OK -- ⚠미실행 -- -- OK -- -- OK OK OK` = **결과 0행**(예상된 상태). `PREREG`·`gates`·자기동결 열 `OK` |
| 2. 판정 문턱 | ⚠*"표본 크기를 못 찾아 검사하지 못했다 (문턱 `S1_*` 은 검사되지 않음)"* — **자문**. `S-1` 문턱은 `B-12` 패리티(§5.5)가 대신 지킨다 |
| 3. 손복사 기준 상수 | `M7` 히트 **0건** |
| 5. 게이트 외부 참조 키 | `M7 … 0 0 0 -- 외부참조 없음` = 공허 참조 0 |
| 5b. 필터 항진 | *"행 0개 — 검사 불가(⛔`OK` 아님)"* ⟹ **기계가 못 보는 자리**를 §5.5 의 수동 탐침이 채웠다 |
| 5c. 생성-게이트 지연 | 대상 아님(자기 생성 없음). ⟹ `gen_m7` 이 생기면 **대상이 된다** — §6-5 |

### 5.7 선행 조건 존재 확인 (전부 실측)

* ✅`M1` **판정 동결**: `verdict_m1.json` 존재 · `pair_key ["hold","hold"]` ⟹ **`G-14` 사전 충족**.
* ✅E3 자료: `data_m1/LIN` **24**개 · `data_m1/HOS` **24**개 `.h5`(533 MB) — **재생성 금지 대상**.
* ✅E2 코퍼스: `gates_b12.HOS` = `tmp/HOSGATE_2026-08-13/data_hos` **64개 `.h5`**(혼잡 필터 후 48).
* ✅E1 하네스: `scene_station.py`·`zero_ref.py`·`run_b2.py`·`gates_b2.py` 존재.
  `gates_b2.RECORDS` == `gates_m7.REAL_RECORDS` **전건 일치**, `WIN` 3 == `REAL_WIN_PRIMARY` 3.
* ✅`learned_ckpt` 경로: `learned_init.py:351` `config.get('learned_ckpt', DEFAULT_CKPT)` ⟹
  **팔 정의가 한 키로 끝난다**는 `PREREG §1.1` 전제 재확인. 모델 캐시 키는 `(ckpt 절대경로, device)`
  (`:103`) ⟹ **5개 체크포인트가 한 프로세스에서 섞이지 않는다**.
* ✅`learned_diffusion_seed` 키 존재(`:352-354`, 기본값 = `seed`) ⟹ **확산 시드 3개는 이 키로 준다**.

---

## 6. ★하네스 작성 요건 — **결과 전에 계약을 동결한다**

⛔아래는 **사전등록의 구현**이지 변경이 아니다. 전부 **더 엄격한 방향**이고, 문턱·팔·라벨을 건드리지 않는다.

1. **`gen_m7.py`** = `gen_mixed.py` 의 **복제**(⛔원본 수정 금지) + ①HARD 앵커(실측 12 레코드의
   (Hs, Tp_disp) 중앙값, `k·Hs>0.44` 인 `LJ18a` 만 학습 앵커에서 제외) ②`nonlinear_forward.elevation(nonlinear=True)`
   관측(라벨은 자유파) ③장 손실용 **obs·참 성분·유보 격자 참 장** 반환(`e2e_finetune.make_scene_batch` 규약).
   ★**`G-10`**: `data='BASE'` 모드에서 원본 `gen_mixed` 와 **비트 동일**(`max|Δ| = 0.0`)임을 스스로 검사한다.
2. **`train_m7.py`** = Phase A(`train_mixed.py` 레시피 축자) + Phase B(`e2e_finetune.finetune` 의
   장 손실을 `gen_m7` 장면 위로 **이식**). ⛔`--out` 은 반드시 `results/direction3/m7_*.pt`
   (기존 파일 덮어쓰기 금지 — 원본들이 이미 `assert not os.path.exists` 를 갖고 있다).
   ⛔**`tmp/wp_e2e_2026-07-25/train_curve_*.json` 에 쓰지 마라** — `e2e_finetune.py` 의 `__main__` 은
   그 경로에 **하드코딩**돼 있어 다른 배치의 동결 산출물을 덮는다(오늘 발견, 그래서 비용 smoke 는
   `finetune()` 을 **라이브러리로** 불렀다). 곡선은 `tmp/M7_SUPERVISION_2026-08-28/curve_*.json` 으로.
   ★`λ_field=0` 팔도 **4,000 스텝을 돈다**(`G-11`).
3. **행 스키마 계약(E1)**: `cell` · `record` · `win` · `diffusion_seed` · `nowcast` · `leadmean` ·
   `curve` · `n_atoms` · `add_calls` · `varpro_obj_before` · `varpro_obj_after` · `varpro_fired` ·
   `route`(=`real_gate`) · `ckpt_path` · `ckpt_sha256`(**행마다 `learned_init.ckpt_sha256()` 로 되읽기** = `G-4`) ·
   `wall_s`. **E2/E3**: `cell` · `episode` · `est_seed` · `avg_nRMSE` · `zero_nRMSE` · `wall_s`
   (+ `SPEC25` 는 `loc_radius`·`n_ensemble`·`background`).
4. **판정 선행 assert (§5.5 의 F-1~F-4)**: `analyze_m7.py` 는 라벨을 내기 **전에**
   ①`train_scene_seeds_sample` 비지 않음 ②세 대비 `n == 12` ③`s1.neg_control['n'] == 48` 을 검사하고,
   하나라도 어긋나면 **`INCOMPLETE` 로 멈춘다**(라벨 없음).
5. **생성-게이트 규율**(`tmp/_audit/DISCIPLINE_gen_gates_2026-08-29.md` P1): 성분 draw 의 순함수인
   술어(`repr_frac ≥ 0.5`, `k·Hs` 스크린)는 `gen_m7` 의 **`screen()`** 에 두고, 위반이면 **즉시 중단**한다.
   `G-6`·`G-7`·`G-8` 은 **같은 심볼**을 되읽어 검사한다(손복사 0).
6. **RNG 위생**: 런마다 격리 프로세스 + 추정 직전 `torch.manual_seed`, 행마다 `add_calls`·`final_n` 기록
   (`PREREG §5` 완화책 ①②③).

---

## 7. ★실행 명령 (붙여넣기 가능, 순서 그대로)

⛔**전제**: 모든 명령은 `cd /home/vexit/wavesplat/wavesplat` 에서 돈다. ⛔`git commit` 금지.
⛔**`CUDA_VISIBLE_DEVICES` 를 세우지 마라**(§5.2). ⛔**`nohup … &` 로 떼지 마라** — 08-22 에 그렇게
띄웠다가 완료 알림을 못 받고 3시간을 잃었다. **추적되는 백그라운드 실행 + 완료 대기**를 쓴다.

### Step 0 — 작성·동결 (GPU 0회, 결과 0행)

```bash
cd /home/vexit/wavesplat/wavesplat
# 0a. §6 요건대로 네 파일을 작성한다 (gen_m7.py · train_m7.py · run_m7.py · analyze_m7.py)
# 0b. G-10 패리티 (BASE 모드에서 원본 생성기와 비트 동일)
python3 tmp/M7_SUPERVISION_2026-08-28/gen_m7.py --parity      # 기대: max|Δ| = 0.0
# 0c. 동결 — ⛔기존 줄을 고치지 말고 **덧붙인다**
#     ⚠`AMENDMENT_1.md` 자신은 **이 문서 작성 직후 이미 동결**됐다(FROZEN_md5.txt 의 개정 블록) ⟹ 목록에서 제외.
{ echo; echo "# ── 착수 동결 (2026-08-29, AMENDMENT_1 §5.3/§7 Step 0c). ⛔결과 0행."; \
  md5sum tmp/M7_SUPERVISION_2026-08-28/{gen_m7.py,train_m7.py,run_m7.py,analyze_m7.py} \
         tmp/M7_SUPERVISION_2026-08-28/{DATA_ARM_REVIEW.md,REVIEW_2026-08-29.md} \
         tmp/B2_REAL_2026-08-23/{gates_b2.py,run_b2.py,scene_station.py,zero_ref.py} \
         tmp/bankmmse_port_2026-07-29/make_gif_bank_deployed.py \
         tmp/B12_HOS_EXTEND_2026-08-23/{run_b12.py,gates_b12.py} \
         tmp/B11_HOS_BASELINE_2026-08-23/gates_b11.py \
         tmp/S16_BASELINE_2026-08-14/{run_s16.py,letkf_spec.py} \
         tmp/M1_NONLIN_REAL_2026-08-28/{verdict_m1.json,gen_manifest_m1.json} \
         tmp/mixed_retrain_2026-07-26/{gen_mixed.py,train_mixed.py} \
         tmp/wp_e2e_2026-07-25/e2e_finetune.py \
         scripts/{d3_beamform_cond_diffusion.py,d3_setdiff_prior.py,nonlinear_forward.py} \
         wavesplat/model/{tnse.py,spectral_gaussian.py} wavesplat/run.py \
         wavesplat/optimizer/{sequential_update.py,learned_init.py,varpro.py} \
         results/direction3/{setdiff_prior.pt,beamform_cond_diffusion_mixed_v1.pt} | sed 's#  #  #'; \
  echo "# git_head $(cd /home/vexit/wavesplat && git rev-parse HEAD)  (⛔commit 금지 상태로 실행)"; } \
  >> tmp/M7_SUPERVISION_2026-08-28/FROZEN_md5.txt
md5sum -c tmp/M7_SUPERVISION_2026-08-28/FROZEN_md5.txt 2>&1 | grep -v ': OK$' ; echo "(빈 출력 = 무드리프트)"
# 0d. 검사기
python3 tmp/M7_SUPERVISION_2026-08-28/gates_m7.py                       # 자가진단 (MDE·전제)
python3 tmp/_audit/check_batch_artifacts.py tmp/M7_SUPERVISION_2026-08-28
python3 tmp/_audit/check_gate_refs.py       | grep -i m7
python3 tmp/_audit/check_gate_blindspots.py | grep -i m7
```

### Step 1 — Phase A 사전학습 **3회** (GPU, ⛔순차 — 단일 GPU)

```bash
# 1a. ⚠HARD 스트림 비용 재측정 (사전등록 §7 의무). 표와 2배 이상 어긋나면 여기서 멈춘다.
python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase A --data HARD --smoke 200 \
    --out /tmp/claude-1000/-home-vexit-wavesplat/scratch_m7_smoke.pt      # 기대 ≈110–160 ms/step
rm -f /tmp/claude-1000/-home-vexit-wavesplat/scratch_m7_smoke.pt

# 1b. 본 실행 — 세 번을 **차례로** (각 ≈1.2–1.6 h)
python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase A --data BASE --seed 20260828 \
    --steps 40000 --out results/direction3/m7_PRE-BASE-1.pt \
    --curve tmp/M7_SUPERVISION_2026-08-28/curve_PRE-BASE-1.json \
    2>&1 | tee tmp/M7_SUPERVISION_2026-08-28/train_PRE-BASE-1.log
python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase A --data BASE --seed 20260829 \
    --steps 40000 --out results/direction3/m7_PRE-BASE-2.pt \
    --curve tmp/M7_SUPERVISION_2026-08-28/curve_PRE-BASE-2.json \
    2>&1 | tee tmp/M7_SUPERVISION_2026-08-28/train_PRE-BASE-2.log
python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase A --data HARD --seed 20260828 \
    --steps 40000 --out results/direction3/m7_PRE-HARD-1.pt \
    --curve tmp/M7_SUPERVISION_2026-08-28/curve_PRE-HARD-1.json \
    2>&1 | tee tmp/M7_SUPERVISION_2026-08-28/train_PRE-HARD-1.log
```

### Step 2 — Phase B 미세조정 **5회** (칸마다 1회, 각 ≈8 min)

```bash
for cell in BASE:PRE-BASE-1:0.0  LOSS:PRE-BASE-1:1.0  DATA:PRE-HARD-1:0.0 \
            BOTH:PRE-HARD-1:1.0  BASE-B:PRE-BASE-2:0.0 ; do
  C=${cell%%:*}; rest=${cell#*:}; P=${rest%%:*}; L=${rest##*:}
  python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase B --cell "$C" \
      --init results/direction3/m7_${P}.pt --lambda_field "$L" --steps 4000 \
      --out results/direction3/m7_${C}.pt \
      --curve tmp/M7_SUPERVISION_2026-08-28/curve_${C}.json \
      2>&1 | tee tmp/M7_SUPERVISION_2026-08-28/ft_${C}.log
done
sha256sum results/direction3/m7_*.pt        # G-2: 칸 5개가 서로 달라야 한다
sha256sum results/direction3/beamform_cond_diffusion_mixed_v1.pt   # G-3: 636c72a7… 무손상
```

### Step 3 — E2 (합성 회귀 + `S-1`), 8샤드 ≈30 min

```bash
for i in 0 1 2 3 4 5 6 7; do
  python3 tmp/M7_SUPERVISION_2026-08-28/run_m7.py e2 $i 8 \
      > tmp/M7_SUPERVISION_2026-08-28/e2_$i.log 2>&1 &
done; wait
```
⛔`8` 을 바꾸지 마라(샤딩이 `idx % nshard`). ⚠재개는 `results_m7_e2_*.jsonl` 의 `done` 집합으로.

### Step 4 — E3 (보고 전용, `M1` 자료 재사용) — ⛔**생성 0회**

```bash
python3 -c "import json;print(json.load(open('tmp/M1_NONLIN_REAL_2026-08-28/verdict_m1.json'))['pair_key'])"
# 기대: ['hold','hold']  ⟹ G-14(판정 동결) 충족 확인 후에만 아래를 돈다
for i in 0 1 2 3 4 5 6 7; do
  python3 tmp/M7_SUPERVISION_2026-08-28/run_m7.py e3 $i 8 \
      > tmp/M7_SUPERVISION_2026-08-28/e3_$i.log 2>&1 &
done; wait
```

### Step 5 — ★E1 (실측 12 레코드) — ⛔**사용자 승인 없이는 돌리지 않는다**

> 승인 요청 문안(그대로 물을 것):
> *"M7 의 E1 은 실측 12 레코드에 **learned 를 강제**합니다(`real_domain_gate=False`).
> 이는 비승인 정책이고 `W12`·`B-2` 때 각각 승인으로 풀렸습니다. 사전 기대는 **널**이며
> (`B-8b`), MDE 는 12개 중 11개 일치입니다. 돌릴까요?"*

```bash
# 승인 후에만
python3 tmp/M7_SUPERVISION_2026-08-28/run_m7.py e1 \
    2>&1 | tee tmp/M7_SUPERVISION_2026-08-28/e1.log        # ≈16 min, 단일 프로세스
```

### Step 6 — 분석·판정·검사기

```bash
python3 tmp/M7_SUPERVISION_2026-08-28/analyze_m7.py        # verdict_m7.json 생성 (§6-4 선행 assert 포함)
md5sum -c tmp/M7_SUPERVISION_2026-08-28/FROZEN_md5.txt | grep -v ': OK$'   # G-15
bash tmp/_audit/run_all.sh 2>&1 | tail -40
```

⚠**중단 조건(결과 전 고정)**: ①Step 1a 비용이 2배 이탈 ②`G-10` 패리티 ≠ 0 ③`G-3` sha 변화
④`BASE` vs `BASE-B` 가 **10/12 이상** 갈림 ⟹ `SEED-NOISE-DOMINATES`(E1 전부 DESCRIPTIVE, 요인 귀속 금지).

---

## 8. 문서 갱신 의무 (이 개정이 만든 것)

* `tmp/STATUS_2026-08-28/STATUS.md:381` = `**M7** | ⛔대기 | … M1 완료 후에만 돈다` ⟹
  **"집행 중 — 사용자 결정으로 분기표 초과(AMENDMENT_1)"** 로 갱신할 것.
* 커밋 `c428d478` 의 메시지가 *"M7 **보류**"* 라 적었다 — **그 판단은 사용자 결정으로 대체됐다.**
  ⛔커밋을 고치지 말고 **원장(RESUME)에 한 줄**로 정정할 것.
* `REVIEW_2026-08-29.md` 는 **무수정**으로 남긴다(§1.4).

---

## 9. 열람 범위 (주장 범위)

`tmp/M7_SUPERVISION_2026-08-28/{PREREG.md, PRIOR_WORK.md, DATA_ARM_REVIEW.md, REVIEW_2026-08-29.md,
gates_m7.py, FROZEN_md5.txt}`(전문) · `tmp/M1_NONLIN_REAL_2026-08-28/{verdict_m1.json,
RUNBOOK.md §1, data_m1(ls)}` · `tmp/M3_SUPPORT_2026-08-28/{FROZEN_md5.txt, RUNBOOK.md §1}` ·
`tmp/B2_REAL_2026-08-23/{run_b2.py, gates_b2.py, zero_ref.py 머리, results.jsonl(집계)}` ·
`tmp/B12_HOS_EXTEND_2026-08-23/{run_b12.py, gates_b12.py, results_b12.jsonl(집계)}` ·
`tmp/mixed_retrain_2026-07-26/train_mixed.py`(전문) · `tmp/wp_e2e_2026-07-25/{e2e_finetune.py,
WP_E2E_2026-07-25.md grep}` · `results/direction3/hos_eval_new.json` ·
`wavesplat/optimizer/learned_init.py`(로더·캐시) · `tmp/_audit/{run_all.sh,
DISCIPLINE_gen_gates_2026-08-29.md, 검사기 6종 실행 출력}` · `tmp/M4_CAPACITY_2026-08-28/{FROZEN_md5.txt,
AMENDMENT_1.md 머리}` · `nvidia-smi` · `git rev-parse/status/log`.

**이 문서가 실행한 계산**: 200스텝 비용 smoke 3건(Phase A / Phase B λ=1 / λ=0) + 장면 생성 단가 2건 +
게이트 탐침 1건 — 전부 스크래치, 산출물 삭제 완료. **저장소 파일 변경 0건, 배치 결과 0행.**
