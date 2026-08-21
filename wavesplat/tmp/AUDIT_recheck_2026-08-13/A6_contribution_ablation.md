# A6 — 기여 절제(ablation) 재검증: "2026-04 파일 부호 반전" 주장

**감사관**: A6 (서브에이전트). **2026-08-13.** **런 0회** — 전부 기존 원자료 재도출 + 코드 열람.
**READ-ONLY** — 이 파일 외 수정·생성 0.

**검증 대상 주장** (`MEMORY.md` RESUME · `project_s13_oos_2026-08-12.md` ·
`tmp/S13_oos_2026-08-12/RESULT.md:60-63`):

> ★★★**#2b 는 2026-04 절제 파일이 틀렸다** — 그 파일은 "끄는 쪽이 37/50 로 낫다" 였는데
> 현재 코드에선 **완전한 부호 반전**. ⟹ ★**대칭 감사가 값을 했다** … 적용하지 않았다면
> **살아 있는 기여를 잘못 죽였을 것.**

---

## 0. 최종 판정 한 줄

| 해석 | 판정 |
|---|---|
| "2026-04 파일의 **팔 라벨이 뒤바뀌어 있었다**" | **REFUTED** — 두 러너 모두 라벨링이 정확하다 |
| "2026-04 와 2026-08 의 **측정된 효과 방향이 반대다**" | **CONFIRMED** — 양쪽 수치 전건 재도출됨 |
| "따라서 **낡은 파일이 틀렸고 #2b 는 살아 있다**" | **UNDECIDED (귀속 불가)** — 처치 팔이 ω 초기화를 파괴하는 교락을 안고 있다 |

---

## 1. 기여 7개 × 현재 판정 × 근거파일

원문: `thesis/thesis_main.tex:484-528` (`\begin{enumerate}` 안 `\item` **7개**).
아래 "현재 판정" 은 `tmp/DEBATE_blankslate_2026-08-12/CONTRIB_AUDIT.md:17-27` 의 census 에
S13 (`tmp/S13_oos_2026-08-12/RESULT.md:47-63`) 를 반영하고, **내가 재도출한 것만 MEASURED** 로 표시.

| # | 기여 (thesis 줄) | 08-12 census | S13 이후 | 근거 파일 | 내 재도출 |
|---|---|---|---|---|---|
| 1 | 격자 비의존 스펙트럼 프레임워크 (`:485-497`) | ALIVE (탐색적) | **ALIVE (확증)** | `S13/results.jsonl` 혼잡 56/64 · 희소 63/64 | ✅ 재도출 |
| 2a | 자유도 5N→4N (`:500-505`) | ALIVE (산술) | ALIVE | `spectral_gaussian.py:63-64,99-103` | ✅ 코드 |
| 2b | 분산제약의 **정확도** 기여 | **UNMEASURED** | **"ALIVE, 복권"** | `S13/verdict_secondary.json` 62/64 | ⚠ **교락** (§4) |
| 2c | "프레임당 200→10 스텝" | UNMEASURED | UNMEASURED | 200-스텝 추적 팔 **부재** | 미검증(범위 밖) |
| 3 | 기하 신뢰 가중치 (`:506-510`) | DEAD (정확도) | **DEAD 확정** | `S13/verdict_secondary.json` 10/64·16/64 | ✅ 재도출 (§4.2 교락 소) |
| 4 | 적응형 N (`:511-515`) | 정확도 DEAD · 기제 WEAKENED | 불변 | `paper/wavesplat.tex:1457-1533` | 미재도출 |
| 5 | 사전 지정 6-지표 패널 (`:516-520`) | ALIVE | 불변 | `wavesplat/evaluation/metrics.py:1-20` | 미재도출 |
| 6 | 공정한 ETKF 기준선 (`:521-524`) | WEAKENED | **더 약화** | S13 `LETKF+`(ne600) 가 공개 LETKF 를 64/64 로 이김 | ✅ 재도출 (63/64, +47.7%) |
| 7 | 하이드로포일 FF 제어 (`:525-527`) | DEAD / UNANSWERABLE | 불변 | `run.py:322-327`·`configs/default.yaml:6,10,30` | 미재도출 |

⚠**"#2b" 는 학위논문의 라벨이 아니다.** 논문 기여 #2 는 하나이고, `2a/2b/2c` 는
`CONTRIB_AUDIT.md:84-122` 가 **감사 편의를 위해 쪼갠 하위 라벨**이다. 그리고 그 감사의 #2b 판정은
`DEAD` 가 아니라 **`UNMEASURED`** 였다 (`CONTRIB_AUDIT.md:21,88`) ⟹ **"복권(revival)" 은
어휘가 정확하지 않다.** 정확한 형태는 **"UNMEASURED 였던 항목이 처음으로 측정됐고, 그 측정은
편향돼 있다"**.

---

## 2. "2026-04 절제 파일" 특정 및 라벨링 검증 — **라벨 반전 없음**

**파일**: `results/ablation_bf45.json` (커밋 `fb5b96f` 2026-04-10, 작업트리 = 커밋본, `git diff` 공백).
**러너**: `scripts/run_ablation_bf45.py:31-41`.

```python
CONFIGS = {
    'Full':       {'use_dispersion': True,  'use_confidence': True,  'init_method': 'v2_konly', ...},
    'A1_noDisp':  {'use_dispersion': False, 'use_confidence': True,  'init_method': 'v2_konly', ...},
    'A2_noConf':  {'use_dispersion': True,  'use_confidence': False, 'init_method': 'v2_konly', ...},
}
```

라벨 → config override 매핑이 **한 줄 안에서 직접적**이고, 결과 행은 `results.append({'config': cfg_name, ...})`
(`:83-87`) 로 그 키를 그대로 쓴다. **중간 재매핑·정렬·zip 이 없다** ⟹ 라벨 교환 여지 0.

S13 쪽도 동일: `tmp/S13_oos_2026-08-12/run_s13.py:58-63`
`'SHIP_NOCONF': {'use_confidence': False}` · `'SHIP_NODISP': {'use_dispersion': False}`,
그리고 각 행이 `effective_config` 를 통째로 기록한다 (`run_s13.py:181-183`).
**내가 `results.jsonl` 의 `effective_config` 를 직접 확인**:

```
SHIP        use_dispersion=True  use_confidence=True   lambda_l1=1e-4  init_method=auto
SHIP_NOCONF use_dispersion=True  use_confidence=False  lambda_l1=1e-4  init_method=auto
SHIP_NODISP use_dispersion=False use_confidence=True   lambda_l1=1e-4  init_method=auto
```

⟹ **`[REFUTED]` 라벨 반전은 없었다.** 다른 절제 러너도 같은 패턴으로 확인:
`scripts/run_ablation_seeded.py:26-31` · `scripts/run_gpr_confidence_ablation.py:52-77`
(함수명 `run_wavesplat_with_conf` / `run_wavesplat_no_conf` 안에 `use_confidence` 리터럴이 있다).
**저장소의 절제 라벨링에서 부호 반전 위험은 발견되지 않았다.**

---

## 3. 원자료 재도출 — 양쪽 수치 전건 재현

### 3.1 2026-04 파일 (`results/ablation_bf45.json`, 50 에피소드 × 3 시드)

시드 집계 규약에 따라 카운트가 하나씩 움직인다:

| 집계 | `A1_noDisp` 가 Full 보다 좋음 | `A2_noConf` 가 Full 보다 좋음 |
|---|---|---|
| 시드 **중앙값** | **37/50** | **48/50** |
| 시드 평균 | 38/50 | 47/50 |
| 짝 단위(에피소드×시드) | 102/150 | 134/150 |

⟹ PREREG §"라이더 R-1 의 근거" 가 인용한 **"48/50 · 37/50" 은 시드-중앙값 집계에서 정확히 재현**된다
(= `CONTRIB_AUDIT.md:94,139` 의 13/50·2/50 의 여집합). **MEASURED.**

구성별 평균 nRMSE: `Full 0.2662` · `A1_noDisp 0.2412` · `A2_noConf 0.2425` (n=50).
⟹ 학위논문 `thesis_main.tex:2159` 의 "(A1=0.241 vs. Full=0.266)" **정확히 재현**.

### 3.2 S13 (`tmp/S13_oos_2026-08-12/results.jsonl`, 704행, 에러 0, 전행 `git_head=5f0846cc`)

`verdict_secondary.json` 을 믿지 않고 jsonl 에서 직접:

| 팔 | 층 | 끄면 나빠짐 | 양측 p | med(rel) | rel(med) | rel(mean) |
|---|---|---|---|---|---|---|
| `SHIP_NOCONF` | 혼잡 | **10/64** | 2.00e-08 | −15.6% | −13.2% | −17.9% |
| `SHIP_NOCONF` | 희소 | **16/64** | 7.73e-05 | −20.4% | −26.9% | −27.2% |
| `SHIP_NODISP` | 혼잡 | **62/64** | 2.26e-16 | **+70.2%** | +90.2% | +76.2% |
| `SHIP_NODISP` | 희소 | **62/64** | 2.26e-16 | **+777.8%** | **+1075.2%** | **+495.3%** |
| `SHIP_L1E3` | 혼잡 | 54/64 | 2.00e-08 | +11.7% | +12.9% | +14.9% |

⟹ RESULT.md §S-2d 의 **10/64 · 16/64 · 62/64 · 62/64 · −15.6% · −20.4% · +70.2% · +777.8% 전건 일치**.
PRIMARY(56/64 · 63/64 · 64/64 · 44/52 · 51/52 · 셀별 31/32·32/32·29/32·27/32)도 재도출 일치.
**계산은 깨끗하다.**

### 3.3 ⟹ 부호 반전 자체는 **CONFIRMED (사실로서)**

2026-04: 분산제약을 **끄는 쪽이 37/50 로 좋다**. 2026-08: **끄면 62/64 로 나쁘다**.
두 수치 다 원자료에서 재현된다. **방향이 반대인 것은 사실이다.**

---

## 4. 그러나 — 처치 팔이 정상이 아니다 (BLOCKER)

### 4.1 `use_dispersion=False` 는 "제약 제거" 가 아니라 "제약 제거 + ω 초기화 파괴"

`wavesplat/model/spectral_gaussian.py:62-64`:
```python
# A1 ablation: ω as free parameter (no dispersion constraint)
if not use_dispersion:
    self.omega_free = nn.Parameter(torch.randn(n_gaussians) * 0.5)
```
`omega_free` 는 **초기화기로부터 값을 받는 코드 경로가 저장소 전체에 존재하지 않는다.**
초기화 결과가 모델로 복사되는 지점은 두 곳뿐이고 둘 다 ω 를 건드리지 않는다:

- `wavesplat/optimizer/sequential_update.py:550-553` — `kx, ky, w, phi` 만 `copy_`
- `wavesplat/optimizer/sequential_update.py:635-638` — 동일

`omega_free` 를 다루는 전체 grep 결과(출하 코드): `spectral_gaussian.py:64,103,197,215` ·
`sequential_update.py:268,306,1031,1048`. **어디에도 `dispersion_omega(k)` 로 초기화하는 줄이 없다.**
성장 시에도 `torch.randn(n_new) * 0.5` (`spectral_gaussian.py:196-198`).

⟹ **`SHIP_NODISP` 는 빔포밍/학습 초기화기가 배치한 (kx,ky) 와 완전히 무상관인
|N(0, 0.5)| rad/s 난수에서 ω 를 출발시킨다.** (BF4–BF5 의 실제 ω ≈ 0.7–1.0 rad/s.)

**데이터가 이것을 확증한다** — `results.jsonl` 의 `init_nrmse` (= Phase-1 초기화 **직후**,
추적 **이전**, `wavesplat/run.py:186-191`):

| 팔 | 혼잡 init_nrmse (중앙) | 희소 init_nrmse (중앙) |
|---|---|---|
| `SHIP` | 0.1504 | **0.0081** |
| `SHIP_NODISP` | 0.2641 (1.76×) | **0.2046 (25.3×)** |
| `SHIP_NOCONF` | 0.1144 | 0.0060 |

⟹ **손상은 추적이 시작되기 전에 이미 전부 들어와 있다.** 희소에서 25배.
학위논문 기여 #2 가 실제로 주장하는 기제는 *"추적 단계에서 위상 예측의 warm start 를 제공"*
(`thesis_main.tex:503-505`) 인데, S13 이 잰 것은 **그 기제가 아니라 "ω 를 초기화할 수 없다"** 다.

**⟹ `+777.8%` 는 분산관계 하드 제약의 가치가 아니라, 대조 팔에 ω 초기화를 주지 않은 설계의
가치다.** 공정한 대조는 `omega_free ← √(g|k|tanh(|k|d))` 로 warm-start 한 뒤 자유화하는 팔이며,
그 팔은 저장소에 **존재한 적이 없다**.

### 4.2 `use_confidence=False` 도 순수하지 않다 (크기는 작음)

`sequential_update.py:322-336` 의 출하 형태는 **가중 합을 비가중 개수로 나눈다**:
`total_loss = Σ(c_j·r_j²) / n_obs`. 코드 자신이 `sequential_update.py:107-110` 에
*"the data term is scaled by mean(confidence) (~0.86 measured) while the L1 and smoothness
penalties are not -- i.e. the penalties are relatively amplified"* 라고 적어 놨다.

⟹ `use_confidence=False` 는 가중치를 1 로 바꾸면서 **데이터항을 1/0.86 ≈ 1.16배 키우고,
그만큼 L1·smooth 벌점의 상대 강도를 ~14% 낮춘다.** 이 저장소에서 벌점 완화는
**성능을 올리는 방향**이다(S5 "범인은 정규화항" · S9 λ 1e-3→1e-4 배포).
즉 **`NOCONF` 팔은 처치와 무관하게 유리한 쪽으로 살짝 밀려 있다.**

**크기 상한**: 같은 배치의 `SHIP_L1E3`(λ 10배) 가 med(rel) +11.7% 다.
로그-λ 선형을 가정한 조잡한 보간으로 1.16배 ⟹ 약 **−0.7%** ⟹ 관측된 −15.6% 의 **5% 미만**.
⟹ **#3 DEAD 판정은 이 교락으로 뒤집히지 않는다. `MINOR`.**

### 4.3 두 교락이 **각자의 판정 방향으로** 민다

- `NODISP` 는 **불리하게** 교락 ⟹ "#2b ALIVE" 를 부풀린다. (크기 미상, 잠재적으로 전부)
- `NOCONF` 는 **유리하게** 교락 ⟹ "#3 DEAD" 를 부풀린다. (크기 ≲5%)

**두 교락 다 보고된 결론을 강화하는 쪽이다.** 이것이 이 배치의 대칭성 문제의 실체다 —
통계 프로토콜은 대칭인데 **처치의 순도가 대칭이 아니다.**

---

## 5. 부호 반전의 대안 설명 (교락 없는 귀속은 불가)

S13 은 반전을 "낡은 파일이 틀렸다" 로 귀속했다. **원자료는 다른 그림을 보여준다.**

| | 분산 **ON** 팔 | 분산 **OFF** 팔 |
|---|---|---|
| 2026-04, `v2_konly`, 구 코퍼스, uniform 15/30comp (n=20) | Full **0.2950** | A1 **0.2912** |
| 2026-08 S13, `auto`, 신 코퍼스, 15/30comp (n=64) | SHIP **0.1484** | NODISP **0.2823** |

⟹ **OFF 팔은 사실상 안 움직였다(0.291 → 0.282). 움직인 것은 ON 팔뿐이다(0.295 → 0.148, 2배).**
그리고 §4.1 이 그 이유를 준다 — 46일간의 개선(라우터 `auto`, `TRACK_PHASE_PREDICT` 위상수정,
λ=1e-4, 생성기 독립성 수정)은 **전부 (kx,ky,w,φ) 를 놓는 초기화·추적 경로를 통해 작동**하는데,
**OFF 팔은 ω 가 그 경로에 연결돼 있지 않아 개선을 하나도 못 받는다.**

⚠**이것은 ARGUED 다** — 두 코퍼스가 다르고(구 `data/` vs `data_s8/s9_*_uniform_ra_rf_ds`),
init_method 도 다르다. 통제된 분해가 아니다. 그러나 **"낡은 파일이 틀렸다" 보다 자료에 더 잘 맞는다.**

### 5.1 ★반전은 S13 이 처음 본 것이 아니다 — 9일 전에 이미 기록돼 있다

`tmp/rerun_2026-08-03/ablation_seeded_rerun.json` (교정 트래커, BF3, n=6) 을 내가 재도출:

```
A1_noDisp: Full 보다 나쁨 4/6,  median(A1 − Full) = +0.0913
A2_noConf: Full 보다 나쁨 1/6,  median(A2 − Full) = −0.0021
```

= `paper/wavesplat.tex:1576-1580` 의 수치와 일치. 그리고 **논문은 그 반전을 이미 적었고,
S13 이 한 귀속을 명시적으로 거부했다** (`paper/wavesplat.tex:1618-1622`):

> "this reverses the direction reported in the earlier pre-correction run … and since
> **tracker, mesh and seed protocol all changed between the runs, the reversal cannot be
> attributed to any single one of them.**"

⟹ **S13 RESULT.md:60-63 의 "낡은 파일이 틀렸다" 는 논문이 8월 3일에 근거를 대고 거부한 문장이다.**
S13 이 새로 산 것은 **반전의 발견**이 아니라 **검정력**(n=6 → n=64×2) 이다. 그것은 실질적 기여지만,
RESULT.md 와 `MEMORY.md` RESUME 의 서술("대칭 감사가 살아 있는 기여를 구했다")은
**08-03 측정과 그 측정의 자기제한을 지운다.**

### 5.2 학위논문 절제 절은 별개로 이미 낡았다

`thesis/thesis_main.tex:2121-2183` 은 **교정 전** 값이다(Full 0.188 / A1 0.158 / A2 0.215, BF3 n=6).
논문 `wavesplat.tex` 표는 같은 6 에피소드에서 Full 0.044 / A1 0.122 / A2 0.043 — **레벨이 4배 다르다.**

★그리고 **학위논문은 BF4–BF5 확장 파일을 A1 에만 인용한다**(`:2158-2160`, "A1=0.241 vs Full=0.266").
**같은 파일의 A2 = 0.2425 (Full 0.2662 보다 좋음, 48/50) 는 인용하지 않는다** —
바로 그 절이 "A2 제거는 평균 14.5% 성능 저하"(`:2177`) 라고 쓰는 동안.
⟹ **기여 #3 을 지지하는 유일한 표본은 n=6 BF3 (교정 전) 이고, 같은 페이지가 인용한
n=50 파일은 반대 방향인데 그 열만 빠져 있다.** S13 과 독립인 `MAJOR` 다.

---

## 6. 대칭 감사가 됐는가

| 항목 | #3 (NOCONF) | #2b (NODISP) | 대칭? |
|---|---|---|---|
| 러너·에피소드·프레임 집합 | 동일 (`run_s13.py`, 128 ep, `t>2Tp`) | 동일 | ✅ |
| n | 64 + 64 | 64 + 64 | ✅ |
| 검정 | 짝 부호검정, 양측 | 동일 | ✅ |
| 사전등록 여부 | PREREG §2·§6 S-2d 에 명시 | 동일 조항 | ✅ |
| **처치의 순도** | 손실 가중만 변경 (+ ~14% 벌점 완화) | **+ ω 초기화 파괴 (25×)** | ❌ **비대칭** |
| **결론 규칙의 사전등록** | §8-5 가 사망 분기만 명시 | **부활 분기 미명시** | ❌ **비대칭** |

### 6.1 사전등록 여부 — 지표는 O, 결론 규칙은 △

- `PREREG.md:36,105` 가 두 팔과 두 대조를 **런 전에 동결**했다 ⟹ **지표는 사후 발견이 아니다.**
- 그러나 `PREREG.md:129-130` (§8-5) 는 **한쪽 분기만** 적었다:
  > "`NOCONF`·`NODISP` 가 `SHIP` 을 이긴다 ⟹ 기여 #3·#2b 는 오늘 코드에서도 **사망**"

  **반대 분기(=지면 어떻게 할 것인가)는 동결돼 있지 않다.** 검정 자체는 양측이므로 결과는
  사전등록 안이지만, **"ALIVE / 복권" 이라는 결론 라벨은 결과를 본 뒤 붙은 것**이다.
  ⟹ **`#2b 복권` 에는 "탐색적(결론 규칙 사후)" 라벨이 붙어야 한다.** `MAJOR`.

### 6.2 자기 규칙 위반 2건

1. `PREREG.md:83` — *"효과크기 보고 | med(rel) · rel(med) · rel(mean) **세 값 전부 의무 병기.
   하나만 쓰면 무효**"*. **PRIMARY 는 지켰고**(`PRIMARY.log`), **SECONDARY 는 `med(rel)` 하나만**
   인쇄한다 (`verdict_secondary.json` · `RESULT.md:50-55`). 희소 NODISP 는 규약에 따라
   **+495.3% ~ +1075.2%** 로 **2.2배** 움직인다 — 규칙이 존재하는 바로 그 이유. `MINOR`(판정 불변).
2. `PREREG.md:86` — SECONDARY 는 **Holm-3**. `verdict_secondary.json` 에 `p_adj` 가 **없다**.
   **내가 적용해 봤다**: 모두 통과(p_adj = 6.77e-16 / 3.99e-08 / 7.73e-05). `MINOR`(판정 불변).

### 6.3 재현 경로의 구멍

`analyze_s13.py` 는 PRIMARY·S-2a·S-2e 만 계산한다. **`verdict_secondary.json`(= S-2c·S-2d,
즉 기여 생사 판정의 근거)을 생성하는 스크립트가 배치 디렉터리에도 저장소 어디에도 없다**
(`grep -rn verdict_secondary tmp/ scripts/` = 0건). 값은 내가 독립 재도출로 확인했으므로
**수치는 맞다**. 그러나 **기여 목록을 바꾸는 판정의 산출 코드가 봉인돼 있지 않다.** `MINOR`.

### 6.4 나머지 기여 5개의 라벨 반전 위험

`run_ablation_seeded.py:26-31` (`v1` init) · `run_gpr_confidence_ablation.py:52-77` ·
`run_ablation_bf45.py:31-41` · `run_s13.py:58-63` 을 전수 열람.
**전부 라벨 문자열 옆에 config 리터럴이 붙어 있는 형태**로, 재매핑 단계가 없다.
⟹ **라벨 반전 위험은 발견되지 않았다.** 대신 발견된 실제 위험은 두 종류다:

- **개입 지점의 비순수성** (§4) — `use_dispersion` 은 ω 초기화와 묶여 있다.
- **구성 표류** — 같은 라벨의 팔이 시대마다 다른 시스템을 잰다:
  `run_ablation_seeded.py` = `init_method:'v1'` · `run_ablation_bf45.py` = `'v2_konly'` ·
  `run_s13.py` = `'auto'`. **세 파일의 "Full" 은 서로 다른 세 추정기다.**
  (`CLAUDE.md` 의 *"저장소에 'WaveSplat' 이라 불리는 서로 다른 추정기가 최소 4개 있다"* 의 절제판.)

---

## 7. 발견 목록

### `[BLOCKER]`
- **B-1. `SHIP_NODISP` 는 분산제약의 절제가 아니다.** `omega_free` 를 초기화기로부터 채우는
  코드 경로가 저장소에 **존재하지 않는다**(`spectral_gaussian.py:64`; `sequential_update.py:550-553,635-638`
  는 kx/ky/w/phi 만 복사). 손상은 **추적 전 초기화 시점에 이미 전부** 들어와 있다
  (희소 `init_nrmse` 0.0081 → 0.2046, **25.3배**). ⟹ **"+777.8% / +70.2%" 를 분산관계 하드
  제약의 정확도 기여로 인용 금지.** 공정한 대조(ω warm-start 후 자유화)는 미실행.
  ⟹ **#2b 는 `ALIVE` 가 아니라 여전히 `UNMEASURED`(교락된 상한만 존재)** 로 되돌려야 한다.

### `[MAJOR]`
- **M-1. "낡은 파일이 틀렸다" 는 귀속이 근거를 넘어선다.** 두 측정 사이에 최소 5가지가 동시에
  바뀌었다: `init_method` v2_konly→auto · `TRACK_PHASE_PREDICT` 위상수정(07-26) ·
  `lambda_l1` 1e-3→1e-4(08-11) · 생성기 독립성 수정(08-07) · 코퍼스/시드 전면 교체.
  `paper/wavesplat.tex:1618-1622` 가 **이미 같은 반전을 보고 "어느 하나에도 귀속할 수 없다"** 고
  적었다. RESULT.md:60-63 과 `MEMORY.md` RESUME 이 그 자기제한을 지운다.
- **M-2. 반전은 S13 이 발견한 것이 아니다.** `tmp/rerun_2026-08-03/ablation_seeded_rerun.json`
  (교정 트래커, n=6, 4/6, +0.0913) 이 **9일 먼저** 같은 부호를 냈고 논문에 실려 있다.
  S13 이 산 것은 **검정력**(n=6→128)이지 발견이 아니다. "대칭 감사가 살아 있는 기여를 구했다" 는
  서사는 08-03 측정을 지운 위에 서 있다.
- **M-3. "#2b 복권" 은 결론 규칙이 사후다.** `PREREG.md:129-130` 은 **사망 분기만** 동결했다.
  ⟹ **"탐색적(결론 규칙 사후)" 라벨 필수.** 또 `CONTRIB_AUDIT.md:21` 의 원 판정은 `DEAD` 가
  아니라 **`UNMEASURED`** 였으므로 **"복권" 이라는 단어 자체가 부정확**하다.
- **M-4. (S13 과 독립) 학위논문 절제 절이 자기 근거 파일의 반대 열을 인용하지 않는다.**
  `thesis_main.tex:2158-2160` 은 `ablation_bf45.json` 을 **A1 에만** 인용하고
  (A1=0.241 vs Full=0.266 — 내가 재현), **같은 파일의 A2=0.2425 (Full 보다 좋음, 48/50)** 는
  인용하지 않은 채 `:2177` 에서 "A2 제거 = 평균 14.5% 저하" 라고 쓴다.
  ⟹ 기여 #3 의 유일한 양성 근거는 **BF3 n=6 교정 전** 표본뿐이고, 같은 페이지가 이미
  더 큰 반대 표본을 손에 쥐고 있었다. 게다가 `thesis_main.tex:2121-2183` 전체가
  `paper/wavesplat.tex` 의 교정 후 표(레벨 4배 차이)로 **대체돼야 할 낡은 수치**다.

### `[MINOR]`
- **m-1.** `SHIP_NOCONF` 는 벌점의 상대 강도를 ~14% 낮춘다(`sequential_update.py:107-110`,
  출하 손실이 가중합÷비가중개수). 방향은 처치에 **유리**. 크기 상한 ≲ 관측 효과의 5%
  (`SHIP_L1E3` λ10배 = +11.7% 로부터 로그 보간) ⟹ **#3 DEAD 는 뒤집히지 않는다.**
- **m-2.** PREREG §4 의 **"3규약 의무 병기, 하나만 쓰면 무효"** 를 SECONDARY 가 위반
  (`verdict_secondary.json`·`RESULT.md:50-55` 는 `med(rel)` 만). 희소 NODISP 는 규약에 따라
  +495.3% ~ +1075.2% 로 **2.2배** 움직인다. (본 보고서 §3.2 에 세 값 전부 인쇄.)
- **m-3.** PREREG §4 의 SECONDARY **Holm-3 미적용**(`p_adj` 부재). 내가 적용: 전부 통과.
- **m-4.** `verdict_secondary.json` 생성 스크립트가 **저장소에 없다**(`analyze_s13.py` 는 PRIMARY 만).
  기여 생사 판정의 산출 경로가 봉인돼 있지 않다. (값 자체는 내가 독립 재도출로 확인.)
- **m-5.** PREREG 가 인용한 "48/50 · 37/50" 은 **시드-중앙값 집계에서만** 나온다(평균이면 47/38).
  집계 규약이 명시되지 않았다.

---

## 8. "부호 반전" 주장에 대한 명시적 판정

> **CONFIRMED (사실) / REFUTED (라벨) / UNDECIDED (함의)**
>
> - **라벨이 뒤바뀌어 있었다 = `REFUTED`** — `run_ablation_bf45.py:31-41` 과 `run_s13.py:58-63`
>   둘 다 라벨 옆에 config 리터럴이 있고 재매핑 단계가 없다.
> - **측정된 효과 방향이 반대다 = `CONFIRMED`** — 37/50(끄면 좋음) vs 62/64(끄면 나쁨),
>   양쪽 원자료에서 전건 재도출.
> - **그러므로 낡은 파일이 틀렸고 #2b 가 살아났다 = `UNDECIDED`, 실질적으로 `기각 권고`** —
>   신 측정의 처치 팔이 ω 초기화를 파괴하고(희소 init 25배 손상, 추적 이전) 그 교락이
>   판정 방향과 같으며, 반전 자체는 논문이 08-03 에 이미 보고하고 **귀속을 거부**한 것이다.

**한 줄**: 부호는 진짜로 뒤집혔지만, 뒤집힌 이유가 "제약이 도움이 된다" 인지
**"대조 팔만 ω 를 못 받는다"** 인지를 이 배치는 가르지 못한다 — 그리고 `init_nrmse` 가
후자를 가리킨다.

---

## 9. 권고 (런 0회로 가능한 순서)

1. **`MEMORY.md` RESUME 과 `S13/RESULT.md:60-63` 의 "#2b 복권 / 낡은 파일이 틀렸다" 를
   `UNMEASURED (교락)` 으로 정정**하고, `+777.8%` 를 **인용금지 목록에 등재**.
2. **공정한 대조 팔 1개** (`SHIP_NODISP_WARM`: `omega_free ← √(g|k|tanh(|k|d))` 로 초기화 후 자유화)
   — 코드 1줄, `sequential_update.py:553` 뒤. 이것 없이는 기여 #2 의 정확도 문장을 쓸 수 없다.
3. **기여 #2 의 진짜 기제(`200→10` warm start)는 여전히 `UNMEASURED`** — S13 은 그것을 재지 않았다
   (손상이 추적 전에 다 들어온다).
4. **M-4 는 S13 과 무관하게 지금 고칠 것** — 학위논문 절제 절이 자기 근거 파일의 반대 열을 빠뜨렸다.

---

## 10. 감사 범위 (열람 목록 — 열지 않은 것은 주장하지 않았다)

`tmp/S13_oos_2026-08-12/{PREREG.md, RESULT.md, GATES.log, PRIMARY.log, run_s13.py, analyze_s13.py,
results.jsonl, verdict_primary.json, verdict_secondary.json, phase3.log(말미)}` ·
`results/{ablation_bf45.json, ablation_seeded.json, three_contributions.json(구조만)}` ·
`tmp/rerun_2026-08-03/ablation_seeded_rerun.json` ·
`scripts/{run_ablation_bf45.py, run_ablation_seeded.py(1-60), run_gpr_confidence_ablation.py(45-80)}` ·
`wavesplat/{run.py(1-120,160-200), model/spectral_gaussian.py(40-230), model/confidence.py(1-45),
optimizer/sequential_update.py(99-120,195-345,990-1075), optimizer/initializer.py(grep only),
data_loader.py(grep only)}` ·
`thesis/thesis_main.tex(470-570, 2110-2183)` · `paper/wavesplat.tex(1555-1650)` ·
`tmp/DEBATE_blankslate_2026-08-12/CONTRIB_AUDIT.md` ·
memory `{MEMORY.md, project_s13_oos_2026-08-12.md, data_corrupt_slope_files.md,
reference_citation_bans.md(grep only)}` · git log/diff of `results/ablation_bf45.json`,
`wavesplat/model/spectral_gaussian.py`.

**열지 않은 것**: 기여 #4·#5·#7 의 원자료(census 인용만), `results/direction*` 계열,
`etkf_tuning_sweep.json`, S7/S8/S9 원자료, `hydrofoil/`, `configs/estimation.yaml` 본문.
