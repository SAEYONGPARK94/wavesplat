# E3 — 역방향 탐색 (문헌 → 자산), 2026-08-15

**역할**: E1·E2 는 자산에서 출발해 문헌을 확인한다. E3 는 반대로 간다 —
*"이 분야가 지금 스스로 '아직 안 됐다'고 말하는 것은 무엇이고, 우리가 그걸 할 수 있는가."*

**READ-ONLY.** 이 파일 외 저장소 수정 0. 추정기 런 0. 하위 에이전트 0.

---

## 0. 열람 범위 = 주장 범위

**웹 검색 12회 · 원문 fetch 9회**(성공 6 / 403 차단 4: MDPI ×2, APS ×1, ScienceDirect ×2).
**저장소 열람**: `memory/reference_citation_bans.md`(전문) · `memory/MEMORY.md`(전문) ·
`memory/project_persona_debate_novelty_2026-08-14.md` · `memory/project_method_debate_2026-08-15.md` ·
`memory/project_hosgate_result_2026-08-13.md` · `memory/project_s16_baseline_2026-08-15.md` ·
`wavesim/pipeline.py:43-62`(`run_episode` 시그니처) ·
`tmp/W3_SOLVER_2026-08-15/{PREREG.md 일부, NOTES.md, run_w3.py:66-94, run.log}`.

⚠**이 보고서가 열지 않은 것**: `paper/`·`thesis/`·`results/`·추정기 소스 본체.
따라서 여기의 "우리 자산" 서술은 **auto-memory 의 판정을 신뢰한 2차 서술**이고,
새로 측정한 것은 `run_episode` 시그니처와 W3 러너의 팔 목록 **두 개뿐**이다.

**인용 검증 등급** — 각 문헌 항목에 표기한다:
- `[FETCH]` = 원문(arXiv abs/HTML)을 직접 fetch 해서 읽은 문장
- `[SUMM]` = 검색엔진 요약 경유. **원문 미확인** ⟹ 논문에 쓰기 전 반드시 원문 확인
- 인용은 전부 URL 또는 arXiv ID 를 단다. ⛔검색 없이 "문헌에 없다"고 쓴 문장은 없다.

---

## 1. 문헌이 스스로 말한 공백 (5개)

### L-1 ★★★ "위상분해 실시간 예측의 **자료동화(초기화)** 가 미해결" — 분야가 직접 그렇게 쓴다

> `[FETCH]` **Ehlers et al., arXiv:2508.03315** (Phys. Fluids 37, 107119 (2025)):
> *"Accurate real-time prediction of phase-resolved ocean wave fields remains a critical yet
> largely unsolved problem, primarily due to the absence of practical data assimilation methods."*
> 그리고: *"the data assimilation problem for initializing nonlinear, phase-resolved, real-time
> wave prediction methods **remains largely unresolved**."*
> 선행 연구 평가도 축자로 적는다 — Desmars 계열: *"its real-time capability is **not demonstrated**"* ·
> Köllisch 계열: *"demands **unrealistically high** spatial and temporal resolution of measurement data."*

보강 `[FETCH]` **arXiv:2501.08430**(PINN 위상분해 동화, 2025-01):
*"fully nonlinear PFT solvers often fail to meet the speed requirements of engineering applications.
This computational inefficiency also **hinders the development of effective data assimilation techniques**,
which are required to reconstruct spatial wave information from sparse measurements."*

보강 `[SUMM]` X-band 레이더 계열: *"The reconstruction step represents **the weakest part** in wave
prediction processes, as optimization procedures decrease prediction horizons and hinder real-time
capability."* (출처 미확정 — 검색 요약, 원문 확인 필요)

★**중요**: 이 공백은 **"더 좋은 아키텍처가 없다"** 가 아니라 **"희소·간접 관측에서 초기장을 만드는
실용적 방법이 없다"** 로 서술돼 있다. 그런데 **아무도 왜 안 되는지는 분해하지 않는다** —
전부 새 아키텍처(PINO/PINN/FNO/U-Net/adjoint/EnKF)를 제안하고 자기 수치를 보고할 뿐이다.

### L-2 ★★ "실해역에는 정답이 없어서 전부 합성으로 검증한다"

> `[FETCH]` **arXiv:2508.03315**: *"Since real-world measurements **lack ground truth** wave elevation
> data for direct comparison, this study validates the PINO approach using **synthetic data**."*
> 또한 *"they require large labelled datasets of ground truth wave data, which are **infeasible to
> obtain** in real-world scenarios."*

> `[SUMM]` **Cao et al., J. Mar. Sci. Eng. 13(11):2196 (2025)**, "Evaluation of Different AI-Based Wave
> Phase-Resolved Prediction Methods" (https://doi.org/10.3390/jmse13112196):
> *"All conclusions are drawn from a **single wave-basin dataset**, whereas real seas exhibit far greater
> variability … the proposed selection guidelines should be viewed as an **initial rather than universal**
> reference, with future work needing to **validate these findings with field measurements**."*
> (MDPI 403 ⟹ 원문 미확인)

### L-3 ★★ "센서가 해면에 대해 고정돼 있다는 가정을 벗어나야 한다"(= 이동 플랫폼)

> `[SUMM]` **Pistellato et al., Remote Sens. 13(18):3780 (2021)** 계열(WASS/WASSfast):
> *"The first research direction is driven by the need to reconstruct wave fields **without assuming
> that the cameras are fixed with respect to the sea surface**, which would allow the installation of
> such systems on **oceanographic or general purpose vessels**."*
> (MDPI 403 ⟹ 원문 미확인. 이 문장이 2021 년 것이면 **5년째 미해결로 선언된 공백**이다.)

> `[SUMM]` X-band 레이더 계열: *"Radar position **fixed** scenarios are the focus of current research,
> while images from a radar on a ship **moving towards waves require adaptations and reduce the
> prediction horizon**."*

> `[SUMM]` LiDAR 기하 서술: *"Single sensors such as LIDAR mounted on **fixed offshore structures** …
> Due to the geometric decrease in density of observation points with distance from the sensor and
> wave shadowing effects, observations of surface elevation are **sparsely distributed**."*
> ⟹ **우리 센서 기하를 문헌이 그대로 적어 놨고, 그 문헌의 마운트는 "고정 구조물"이다.**

### L-4 ★★★ "약한 기준선 + 음성결과 미보고"가 이 분야 전체의 결함으로 **측정**돼 있다

> `[FETCH]` **McGreivy & Hakim, Nat. Mach. Intell. 6, 1256–1269 (2024)** / arXiv:2407.07218, 초록 축자:
> *"Of articles that use ML to solve a fluid-related PDE and claim to outperform a standard numerical
> method, we determine that **79% (60/76) compare to a weak baseline**. Second, we find evidence that
> reporting biases, especially outcome reporting bias and publication bias, are widespread. We conclude
> that ML-for-PDE solving research is **overoptimistic** … We call for **bottom-up cultural changes** …
> as well as **top-down structural reforms**."*

> `[SUMM]` 후속 관점 논문 존재: **Brandstetter, "Envisioning better benchmarks for machine learning PDE
> solvers", Nat. Mach. Intell. (2025)** · **"Machine learning solutions looking for PDE problems",
> Nat. Mach. Intell. (2025)** (https://www.nature.com/articles/s42256-025-00989-w).
> ⟹ **분야가 지금 이 문제를 계속 이야기하고 있다** = 지면이 열려 있다.

★**이것이 "분야가 원한다고 명시했는데 아무도 공급하지 않는" 유일한 항목**이다.
McGreivy & Hakim 은 **질병을 측정**했지 **치료를 시연**하지 않았다.
그리고 "structural reforms" 를 요구했다.

### L-5 ★ 격자없는 초해상(atomic norm) 문헌이 **자기 한계를 명시**한다

> `[SUMM]` (Yang & Xie, arXiv:1407.2490 및 후속 계열 요약):
> *"due to the convex relaxation, these methods suffer from a **serious resolution limit** where
> frequencies are required to be **well separated** for recovery."* ·
> *"alternating projections-based gridless methods directly solve the atomic ℓ₀ … however the
> **convergence performance is not guaranteed** due to unclosed or nonconvex sets."* ·
> *"The state-of-the-art 2-D ANM approach vectorizes the 2-D measurements to their 1-D equivalence,
> which incurs **huge computational cost**."*

⟹ 이 세 문장은 **우리 혼잡 레짐의 증상과 1:1로 대응한다**:
성분이 붙을수록(=주파수 미분리) 나빠짐 · shrinkage 편향 · 2-D 파수 문제의 비용.
★**단 이것은 우리에게 novelty 를 주지 않는다** — 이미 알려진 한계다.
주는 것은 **이름과 좌표**다(§4 참조).

### (보조) L-6 "희소→조밀"이 병목이라고 명시

> `[FETCH]` **arXiv:2509.19384 §4.1**: *"Errors are **lowest near buoy locations and increase away from
> anchors**, reflecting the **inherent limits of sparse sensing**."* — 그리고 §4.5 에서
> *"the framework imposes **no hard physical constraints**, so non-physical oscillations may occur."*
> `[SUMM]` WASSfast: 희소 특징점이 원하는 균일 격자점의 **5–20% 뿐**이라 sparse-to-dense 가 필요.

⟹ 문헌은 이것을 **"희소 센싱의 내재적 한계"** 라고 **선언**한다. **분해하지 않는다.**
우리는 같은 현상을 셸 단위 n=64 로 분해해 놨다(S16 §3).

---

## 2. 우리 자산과 대조 — 채울 수 있는 것 / 없는 것

| 문헌 공백 | 우리가 채울 수 있나 | 이유 (MEASURED / ARGUED) |
|---|---|---|
| **L-1** 동화·초기화가 미해결 | **부분 가능 — "왜 안 되는지"만** | MEASURED: S4 오라클 출발서 순수 오적합 **26.9×**(16/16) · S3 잔여의 **99% decorrelation** · S5 정규화항이 오차의 84% · M1 오라클 k 위 `lstsq` 1회로 전역 ρ=**0.99996**. ⟹ **"동화 기계가 없다"가 아니라 "지지집합+shrinkage 가 벽"** 이라고 말할 자료가 있다. ⛔단 **더 나은 동화기를 내놓을 수는 없다**(그건 PINO/adjoint 경쟁) |
| **L-2** 실측 정답 부재 | **반쪽만** | MEASURED: HOSGATE 가 **GT≡모델식 지름길을 깨는 개입**을 짝 설계로 처음 쟀다(희소 +181.5% 14/16, 혼잡 널 7/16). ⟹ **"합성 전용 주장이 얼마나 표현일치 덕인지 재는 절차"** 는 공급 가능. ⛔**"우리 방법이 실해역에서 맞는다"는 불가** — 실측 정확도 양성 **0건**, B-3 은 개구 부족으로 `UNANSWERABLE`(필요 ~195 m vs 보유 최대 162×183 m) |
| **L-3** 이동 플랫폼 | ⛔**불가, 그리고 지금 선점당하는 중** | MEASURED: `wavesim/pipeline.py:43-62` `run_episode()` 에 **선속 인자가 없다**(직접 확인). 재건축 실비 25–35 h + 메시-Nyquist 재개방(B-1 감사). B-1 축은 **세 번 죽었다**. 실측 관측기하 CRB 는 **이동이 정보를 줄인다**(정지 0.00986 vs cross 0.01615, 0/12). 게다가 §5 의 선점 목록 참조 |
| **L-4** 약한 기준선·음성 미보고 | ★★★**가장 잘 맞는다** | MEASURED: 16,327 런 · 죽은 축 ~103개 · **S16 = 기준선을 수리했더니 우리 PRIMARY 가 스스로 무효화된 사전등록 게이트**(R-2 28/64 ⟹ PRIMARY 59/64 무효). 이건 McGreivy & Hakim 이 요구한 행동의 **실물 시연**이다. ⛔단 M4 판정 = **이전 가능한 인공물 0개**(`sign_one_sided` 4파일 복붙) · M2 감사 = `PREREG.md` 50개 중 `FROZEN_md5` **7개** ⟹ ⛔**"preregistered" 대신 "self-attested freeze"**. **작업이 남아 있다** |
| **L-5** 격자없는 초해상의 자기 한계 | **가능하나 novelty 0** | ARGUED: 처방 이름(VARPRO / Sliding Frank–Wolfe / post-lasso)이 전부 오래됐다(M1). ⟹ **방법 novelty 로 팔면 즉사**. **해양 쪽 진단 문장**으로만 값이 있다 |
| **L-6** 희소→조밀 내재 한계 | **분해는 가능** | MEASURED: S16 §3 셸표 n=64 — 근거리 대등(`0_5` 46/64) · 이득은 중거리 · 먼바다는 **양 팔 다 무기술**(`SPEC25<ZERO` **22/64**) · `SPEC100`(loc 200 m)도 실패 ⟹ **원인은 국소화 반경이 아니라 관측 발자국** |

### 2.1 ⛔후보에서 뺀 것과 이유

- **"이동 플랫폼 위상분해 재구성"** — 코드 부재 + 세 번 사망 + CRB 역방향 + **분야가 12개월 안에 닫는다**(§5).
- **"비선형 포함 재학습"** — HOSGATE 가 이 근거를 **제거**했다(혼잡이 비선형에서 그대로 버팀).
  ⛔인용금지 등재.
- **"더 나은 동화 아키텍처"** — 우리 정직한 초기화 비용은 **중앙 24.0 s**, Ehlers 는 **0.014 s/sample**.
  ⟹ 지연 축으로는 못 선다(08-14 P1 이 이미 이 문을 닫았다).
- **"실측에서 이긴다"** — 08-01 이후 양성 0건, B-3 `UNANSWERABLE`. **사망조건을 붙일 수 없다** ⟹ 제외.
- **`SUPPORT-TRADEOFF` 를 방법-대-기준선 명제로 파는 것** — 08-15 감사가 재현 후보 10건 전수에서
  외부 기준선 행 **0개**임을 확인. ⛔등재됨.

---

## 3. ★1순위 후보 — `R1` : **"동화가 아니라 지지집합이다"** (문헌 L-1 × L-5 를 동시에 친다)

### 3.1 한 문장

> 문헌은 *"희소 관측에서 위상분해 초기장을 만드는 실용적 방법이 없다"*(L-1) 고 선언한다.
> 우리는 **그 문장이 부정확함을 측정으로 보인다** — 지지집합(파수)을 주면 남은 동화는
> **프레임당 50점으로 닫힌형 최소제곱 한 번에 전역 ρ=0.99996** 이다.
> 미해결인 것은 **동화가 아니라 shrinkage 편향이 걸린 지지집합 추정**이고,
> 그 실패가 **① 다봉성(진짜 비볼록) · ② 벌점 편향 · ③ 매개화** 중 무엇인지 **가른다.**

### 3.2 왜 지금인가 — 결정적 런이 **이미 돌고 있다**

`tmp/W3_SOLVER_2026-08-15/` 가 지금(20:59 KST) **152/912** 진행 중이고, 팔 목록에 이미 있다
(`run_w3.py:76-94` 직접 확인):

```
ZERO · ORA-K-CVX(닫힌형) · ORA-K@L3(λ=1e-3) · ORA-K-NOREG(λ=0) · ORA-K@L4(λ=1e-4)
ORA-K@L3-DUP · SHIP · SHIP-FITPHI · SHIP-DUP · KONLY · KONLY-FITPHI
```

⟹ **② 벌점 편향과 ③ 매개화(위상 난수 폐기)는 오늘 밤 답이 나온다.**
**없는 것은 딱 하나 — ① 다봉성을 가르는 다중 재시작 팔**이다.
그리고 `reference_citation_bans` 가 이미 그 결손을 적어 놨다:
*"'비볼록성' 이라는 이름은 아직 못 샀다(**다중 재시작 팔 부재**)."*

### 3.3 최소 실험 (`R1-MR`, W3 종료 직후 부착)

**설계**: W3 와 **같은 에피소드·같은 시드·같은 채점기**, 팔 2개 추가.

| 팔 | 정의 | 목적 |
|---|---|---|
| `SHIP-MR8` | 출하 구성 그대로, **위상 초기화만 8회 독립 재시작**, `avg_obs_resid_rms` **최소** 선택 | 다봉성 가르기 |
| `ORA-K-MR8` | `ORA-K-NOREG` + 위상 8회 재시작 | λ 를 뺀 상태의 순수 다봉성 |

- **선택 기준은 GT-free** (`avg_obs_resid_rms`, 벌점 없는 순수 오적합). ⛔`avg_obs_loss` 금지
  (08-14 감사: λ 벌점 지분 69.3–95.8%).
- **널 팔 필수** — `SHIP-DUP8`(**재시작 없이 시드만 다른 8회 중 최소**).
  08-12 S10 이 *"두 번 돌려 운 좋은 쪽 고르기"* 로 물린 그 함정을 미리 막는다.
  ⟹ **판정량 = `MR8 − DUP8`**, `MR8 − SHIP` 단독 금지.
- **층화**: 혼잡(15/30comp) · 희소(2comp) 각 n=32, median-of-3 없이 단일시드
  (W3 규약과 정합, 짝 부호검정 단측 α=0.025, 문턱 21/32).
- **셸 의무 기록**: 팔마다 `obs:*` **와** `ship:*` 고리를 **둘 다** 기록(S-2a 재발 방지, DCC 적용).
  ⟹ 사망조건이 읽는 필드가 한 팔이라도 없으면 **판정 인쇄 없이 종료**.

**비용** (MEASURED 기반): W3 est 팔 실측 **33.5 s/런**, 워커 8.
`(2 처치 + 1 널) × 8 재시작 × 64 에피소드 = 1,536 최적화` ⟹ 이상적 병렬 **1.8 h**,
CPU 경합 관측치(S13b 에서 18 s→182 s 사례 존재) 감안 **2–5 h**. **생성 0 · 시드 소모 0**
(W3 코퍼스 재사용). ⚠단 그래서 **λ·라우터에 신선하지 않다** — PREREG 에 명시할 것.

### 3.4 사전등록 가능한 사망조건 (3분기 전부 정보를 산다)

| 결과 | 판정 | 척추에 대한 처분 |
|---|---|---|
| `MR8 − DUP8` 이 혼잡에서 **≥21/32**, 크기가 `ORA-K-CVX` 격차의 **≥30%** 를 메움 | **다봉성 확인** | ★**"비볼록성"이라는 이름을 산다.** 처방 = 재시작/전역법. 문헌과 모순되는 축이 처음 생김 |
| `MR8 − DUP8` **널**(≤20/32) 이고 `ORA-K-NOREG < ORA-K@L3` 가 통과 | **벌점 편향** | ⛔**"비볼록"이라는 단어 영구 금지.** 척추 개명 = **"shrinkage 편향 지지집합 추정"** |
| **둘 다 널** | **매개화/지지집합** | 척추 개명 = M3 안(*"지지집합 안=최적화 실패, 밖=정보"*)에서 **"최적화 실패"를 "지지집합 식별 실패"로 좁힌다.** 처방 = VARPRO / SFW |
| `MR8` 이 **희소에서만** 발화 | **레짐 비대칭 재확인** | 혼잡 주장 금지 — S4 의 레짐 비대칭과 정합하는지만 기록 |

⛔**PRIMARY 통과만으로 "우리가 문헌보다 낫다"고 쓰지 않는다.** 이 배치가 사는 것은
**진단 문장 하나**이지 우수성이 아니다(M4 §4-3: *"크기는 있으나 우수성을 지지하는 크기가 0개"*).

### 3.5 이게 문헌의 어느 문장을 정확히 반박/보완하는가

- **반박 대상 (L-1)**: *"largely unsolved … due to the absence of practical **data assimilation** methods"*
  ⟹ 우리 반례: 지지집합을 고정하면 동화는 **볼록·닫힌형·1회**. 병목의 이름이 틀렸다.
- **보완 대상 (L-5)**: 격자없는 문헌은 *"well separated 가 필요하다"* 를 **이론 조건**으로 안다.
  우리는 **실제 선박 LiDAR 관측기하 위에서 그 조건이 깨지는 지점을 성분수 축으로 측정**해 놨다
  (λ=0 에서도 30성분 잔여 0.0485 = 15성분의 **3.2배**, M4 §3-1).
- **보완 대상 (L-6)**: *"inherent limits of sparse sensing"* 을 **셸 분해**로 바꾼다
  (S16: 근거리 대등 · 중거리 이득 · 먼바다 양 팔 무기술).

---

## 4. 2순위 — `R2` : **L-4 를 판다** (가장 값싸고 지면이 가장 확실하다)

McGreivy & Hakim 은 **질병을 76편에서 측정**했고 **"structural reforms"** 를 요구했다.
**치료를 한 연구 프로그램 전체에 적용해 시연한 사례가 이 검색에서 나오지 않았다**
(⚠**미확인** — 검색 4회로는 부재를 증명하지 못한다. `preregistration + PDE/ocean ML benchmark` 는
`arXiv:2311.18807`(Pre-registration for Predictive Modeling)·NeurIPS pre-registration workshop·
PDEBench 정도가 나왔고 **전부 "치료의 시연"이 아니다**).

**우리 자산이 이례적인 이유** (MEASURED):
- **S16**: 기준선을 수리하는 **사전등록 게이트**가 자기 PRIMARY 를 **무효화**했다(R-2 28/64).
  게다가 감사 두 건이 *"수리 성공"* 을 **두 번 걷어냈다**(방향 오라클 · loc 정합).
  ⟹ McGreivy & Hakim 의 "weak baseline" 을 **자기 자신에게 집행한 실물**.
- **HOSGATE**: 표현일치 지름길을 **개입으로 깬** 짝 설계.
- **죽은 축 ~103개 · 16,327런 · 인용금지 174건**: 음성결과 미보고의 정반대.

**남은 작업(정직하게)**: M4 판정 = **이전 가능한 인공물 0개**. 그래서 R2 는 "글쓰기"가 아니라
**BRV/DCC 를 라이브러리 1개 + 스펙 1개로 만드는 엔지니어링**이다. 지면 = **GMD**(M2 권고), ⛔OCEANS 아님.

**사망조건**: (i) 문헌검증에서 동등한 프로토콜 논문이 발견되면 폐기.
(ii) 우리 50개 PREREG 중 **`FROZEN_md5` 가 7개뿐**이므로, 소급 주장은 그 7개로만 한다 —
7개로 "프로그램 규모 시연"이 안 되면 **다음 배치부터 커밋+push 로 쌓는 3개월 대기**를 감수하거나 폐기.

⚠**R1 과 R2 는 경쟁하지 않는다.** R1 은 척추(기술)를, R2 는 방법론(메타)을 산다.
R1 이 3.4 의 어느 분기로 가든 R2 의 값은 안 변한다.

---

## 5. ★★★ "이 분야가 곧 할 것 같은 것" — 늦으면 선점당한다

**전부 최근 12개월 내 출판이거나 저자가 명시한 다음 단계다.**

### 5.1 ⛔**이동 플랫폼 재구성 — 이미 닫히는 중** (우리가 절대 못 이긴다)

| 연도 | 문헌 | 무엇을 했나 |
|---|---|---|
| 2025-09 | `[SUMM]` **Phys. Rev. Fluids 10, 094801** — *Nonlinear wave reconstruction and prediction by a shipborne radar with a dynamic averaging algorithm* | **선박 레이더 + 이동하는 배**에서 HOS 3차 비선형 재구성·예측. 레이더 사각지대로 들어오는 파를 점진 재구성 |
| 2026 | `[SUMM]` **DSGP-Net**, Ocean Eng. (S0029801826009200) — *dual-space guided propagation network … from **sparse stereo** measurements* | 고정 플랫폼서 SOTA 대비 RMSE **14–23%** 개선, 그리고 *"uniquely enable accurate reconstruction on **moving vessels**"*, **실시간** |
| 2026 | `[SUMM]` **Sensors 26(3):993** — *Research on Ship-Borne Wave Observation Experiment Based on Stereoscopic Vision* | 선박 탑재 스테레오 실험 |
| 2025 | `[SUMM]` Ocean Eng. S0029801825006717 — *Wave-field reconstruction using stereo cameras on a **floating platform*** | 부유 플랫폼 |

⟹ ★**우리의 "선박 탑재(이동)" 차별점은 2026 년 안에 소멸한다.**
그리고 우리는 **선속 인자조차 없다**(`pipeline.py:43-62`). **이 축으로 경주하면 진다.**

### 5.2 ⛔**"희소 관측 → 전체 파도장" 프레이밍 자체가 식민화되는 중**

- `[FETCH]` **arXiv:2512.06024** (2025-12-04, Liu·Benetazzo·Li 등) — *Neural reconstruction of 3D ocean
  wave hydrodynamics from camera sensing*: 자유표면 **+ 비선형 속도장**까지, 폐색 조건서 강건, 실해역.
- `[SUMM]` Ocean Eng. S002980182503848X — *Sparse-to-full reconstruction of ocean wave height fields
  using lightweight recurrent neural models*.
- `[SUMM]` Ocean Eng. 353:124786 (2026) — *Fusing SWAN priors with Tucker decomposition: a
  **training-free** framework for sparse wave field reconstruction*.
  ⚠단 이건 **bulk 파라미터**(Hsig/Tm01/Dir)라 위상분해가 아니다 ⟹ **직접 경쟁 아님**.
- `[SUMM]` **JMSE 12(7):1082 (2024)** — *Task-Driven Learning **Downsampling** Network … Phase-Resolved
  Wave Fields Reconstruction with Remote Optical Observations*.

### 5.3 ⛔**PINO/PINN 동화가 2D+t 로 간다고 저자가 명시했다** — 우리 문제 정의와 정면 충돌

> `[FETCH]` **arXiv:2508.03315 결론**: *"Currently, the proposed PINO approach has been demonstrated on
> long-crested, unidirectional (**1D+t**) wave surfaces. Extending this framework to realistic
> **two-dimensional (2D+t)** wave fields … is **the future perspective for this work**."*

⟹ 이 그룹이 2D+t 를 내놓는 순간 **"희소 관측에서 실시간 위상분해 동화"** 는 그들의 것이 된다
(합성 검증만이라도). **우리 척추(L-1 반박)의 유효기간은 그 논문까지다.**
★그래서 R1 의 값은 **속도에 달려 있다** — 진단은 그들이 안 하지만, 문제 자체가 "풀렸다"고 선언되면
진단의 수요가 준다.

### 5.4 ✅**아직 비어 보이는 곳** (⚠검색 4회 근거 — 부재 증명 아님)

- **격자없는 sparse spectral estimation(atomic norm/off-grid line spectral)을 위상분해 해양 파도장
  재구성에 적용한 논문**을 이번 검색에서 **찾지 못했다**. 해양 쪽은 FFT/HOS/adjoint/NN 이고,
  격자없는 쪽은 DOA/레이더다. ⟹ **우리가 서 있는 교차점은 아직 비어 있을 가능성**.
  ⛔단 08-14 토론이 **Katende 2606.07153 · Sprunck 2510.27663 · Barthelme&Utschick TSP 2021** 을
  이미 선점으로 등재했으므로, **"최초"라고 쓰기 전에 그 3건과 대조 필수**.
- **재구성 실패의 기제 분해**(정보 vs 최적화 vs 표현)를 한 해양 논문을 못 찾았다.

---

## 6. 내 오류·미확인 신고

1. ⚠**MDPI·APS·ScienceDirect 4건이 403** 이라 §1 의 L-2·L-3 인용문과 §5.1 표는 **검색엔진 요약 경유**다.
   **논문에 쓰기 전 원문 확인 필수.** 특히 WASS 인용문의 **연도**(2017 논문인지 2021 논문인지)가
   미확정이고, 그 연도가 "몇 년째 미해결인가"라는 서술을 좌우한다.
2. ⚠검색 요약에 나온 *"generalizing to an unknown sea state … error would surge about **6 to 11 times**"*
   는 **출처를 특정하지 못했다** ⟹ ⛔**인용하지 마라.**
3. ⚠*"LiDAR 는 약 100 m 로 제한돼 예보 지평이 몇 특성주기로 묶인다"* 도 **출처 특정 실패**
   (arXiv:2508.03315 원문 재확인 결과 그런 문장 없음) ⟹ ⛔**인용 금지**.
4. ⚠§5.4 의 "비어 있다"는 **검색 4회 범위의 부재 주장**이다. `feedback_scope_of_search_equals_scope_of_claim`
   에 따라 **"우리가 확인한 것 중"** 으로만 쓸 것.
5. ⚠나는 `paper/`·`thesis/`·`results/` 를 **한 파일도 열지 않았다**. 08-12 감사가 A1 을 잡은 그 실패와
   같은 범위 제한이므로 명시한다.

---

## 7. 한 줄 결론

> **분야는 "동화가 미해결"이라고 쓰고 새 아키텍처를 낸다. 우리는 새 아키텍처를 낼 수 없지만,
> 그 문장의 이름이 틀렸다는 것은 잴 수 있다** — 그리고 그 결정적 런은 **오늘 밤 이미 돌고 있으며,
> 빠진 팔은 다중 재시작 하나뿐이다.**
> ⛔**이동 플랫폼으로 경주하지 마라 — 2026 년에 스테레오와 레이더가 그 문을 닫는다.**

---

### 참고문헌 (URL / arXiv ID)

- Ehlers et al., *Bridging ocean wave physics and deep learning: PINOs for nonlinear wavefield
  reconstruction in real-time* — https://arxiv.org/abs/2508.03315 (Phys. Fluids 37, 107119, 2025) `[FETCH]`
- *PINNs for phase-resolved data assimilation and prediction of nonlinear ocean waves* —
  https://arxiv.org/abs/2501.08430 `[FETCH]`
- Ehlers et al., *ML for phase-resolved reconstruction … from sparse remote sensing data* —
  https://arxiv.org/abs/2305.11913 (Ocean Eng., 2023) `[FETCH]`
- McGreivy & Hakim, *Weak baselines and reporting biases …* — https://arxiv.org/abs/2407.07218 ·
  https://www.nature.com/articles/s42256-024-00897-5 (Nat. Mach. Intell. 6, 1256–1269, 2024) `[FETCH]`
- *AUWave / Data-Driven Reconstruction of Significant Wave Heights from Sparse Observations* —
  https://arxiv.org/html/2509.19384 `[FETCH]`
- Liu, Zhou, Yan, Guo, Benetazzo, Li, *Neural reconstruction of 3D ocean wave hydrodynamics from camera
  sensing* — https://arxiv.org/abs/2512.06024 (2025-12-04) `[FETCH]`
- Cao et al., *Evaluation of Different AI-Based Wave Phase-Resolved Prediction Methods* —
  https://doi.org/10.3390/jmse13112196 (JMSE 13(11):2196, 2025) `[SUMM]`
- Pistellato et al., *A Physics-Driven CNN Model for Real-Time Sea Waves 3D Reconstruction* —
  https://doi.org/10.3390/rs13183780 (Remote Sens. 13(18):3780, 2021) `[SUMM]`
- *Nonlinear wave reconstruction and prediction by a shipborne radar with a dynamic averaging algorithm* —
  https://link.aps.org/doi/10.1103/jsd9-4kx1 (Phys. Rev. Fluids 10, 094801, 2025-09-18) `[SUMM]`
- *DSGP-Net: dual-space guided propagation network … from sparse stereo measurements* —
  https://www.sciencedirect.com/science/article/abs/pii/S0029801826009200 (Ocean Eng., 2026) `[SUMM]`
- *Research on Ship-Borne Wave Observation Experiment Based on Stereoscopic Vision* —
  https://www.mdpi.com/1424-8220/26/3/993 (Sensors 26(3):993, 2026) `[SUMM]`
- *Wave-field reconstruction using stereo cameras on a floating platform* —
  https://www.sciencedirect.com/science/article/abs/pii/S0029801825006717 (Ocean Eng., 2025) `[SUMM]`
- *Sparse-to-full reconstruction of ocean wave height fields using lightweight recurrent neural models* —
  https://www.sciencedirect.com/science/article/abs/pii/S002980182503848X (Ocean Eng., 2025) `[SUMM]`
- *Fusing SWAN priors with Tucker decomposition: training-free framework for sparse wave field
  reconstruction* — https://www.sciencedirect.com/science/article/abs/pii/S0029801826006207
  (Ocean Eng. 353:124786, 2026) `[SUMM]`
- *Task-Driven Learning Downsampling Network … Phase-Resolved Wave Fields Reconstruction* —
  https://doi.org/10.3390/jmse12071082 (JMSE 12(7):1082, 2024) `[SUMM]`
- Yang & Xie, *On Gridless Sparse Methods for Line Spectral Estimation …* —
  https://arxiv.org/abs/1407.2490 `[SUMM]`
- *Pre-registration for Predictive Modeling* — https://arxiv.org/abs/2311.18807 `[SUMM]`
- *Machine learning solutions looking for PDE problems* —
  https://www.nature.com/articles/s42256-025-00989-w (Nat. Mach. Intell., 2025) `[SUMM]`
