# W3_SOLVER — AUDIT_PRO (돌릴 이유를 최대한 강하게)

**역할**: 이 배치를 **변호**한다. ⛔거짓 변호 금지 — 원자료·PREREG·코드로 뒷받침되는 것만 쓴다.
**판정**: ★**GO** (조건 3개, §7)

**열람 범위 = 주장 범위.** 내가 실제로 연 것:
`PREREG.md`(md5 재검증 `9f4e9c7e…` ✅) · `AMENDMENT_1.md`(md5 `db6d3203…` ✅) · `FROZEN_md5.txt` ·
`SMOKE.md` · `NOTES.md` · `run_w3.py`(전문 698줄) · `analyze_w3.py`(발췌 + 함수 목록) ·
`smoke_results.jsonl`(49행 **재도출**) · `dryrun_smoke.log` ·
`memory/project_method_debate_2026-08-15.md` · `memory/reference_citation_bans.md`(관련 절) ·
`wavesplat/data_loader.py:106-131` · `wavesplat/optimizer/coarse_to_fine.py:150-200` ·
`wavesplat/optimizer/initializer.py:417-442` · `tmp/**/*.jsonl` 의 `ORACLE-*` 행 전수 스캔.

---

## 0. 내가 원자료에서 **독립 재도출**한 것 (MEASURED)

`smoke_results.jsonl` 을 직접 파싱해 SMOKE.md 의 수치를 재계산했다. **전건 일치**.

| 항목 | 재도출 값 | SMOKE.md 주장 | 일치 |
|---|---|---|---|
| 행 수 / 에러 | 49 / 1 (`AcceleratorError` CUDA OOM, `SHIP-FITPHI\|BF4_2comp_seed20\|1593`) | 49행, OOM 1건 | ✅ |
| `ORA-K-CVX` 혼잡 / 희소 | **0.00201 / 0.00144** | 동일 | ✅ |
| `ORA-K-NOREG` 혼잡 중앙값 | **0.14521** | 동일 | ✅ |
| 상한 대비 격차 | **72.2×** (0.14521 ÷ 0.00201) | "72배" | ✅ |
| `ZERO` 대비 CVX | 119.4× (0.24005 ÷ 0.00201) | "×119" | ✅ |
| `ORA-K@L3` vs `ZERO` (혼잡) | **0.25337 > 0.24005 = 대조가 퇴화** | 동일 | ✅ |
| `SHIP` vs `SHIP-FITPHI` 혼잡 | 3시드 **완전 동일** (0.22566 / 0.23790 / 0.25489) | "한 비트도 안 닿음" | ✅ |
| `ORA-K@L3` vs `-DUP` | 3시드 **완전 동일** ⟹ G-5 \|Δ\|=0 | 동일 | ✅ |
| λ 객체 되읽기 | 0.001 / 0.0001 / **0.0** 팔별 정확 | 불일치 0 | ✅ |
| `fitphi_atoms` | `KONLY-FITPHI` 12(양 층) · `KONLY` 0(randphi 16/12) · OFF 8팔 전부 0 | 동일 | ✅ |
| 표본 파일 | 혼잡 32 / 희소 16, **결측 0** | 48 에피소드 | ✅ |
| 총 잡 수 | **1,200** (러너 `build_jobs()` 직접 실행) | 1,200 | ✅ |

**추가로 내가 코드에서 확인한 것** (SMOKE.md 에 없던 것):

1. ★**`ORA-K-CVX` 는 홀드아웃에서 채점된다.** `data_loader.py:106-131` — `get_init_window` 은
   `t ≤ 2·Tp`, `get_tracking_frames` 는 `t > 2·Tp` 로 **완전 배타**다. CVX 는 초기화 창에서
   `lstsq` 한 번을 풀고, **그 창에 없는 160 프레임**에서 **미관측 영역 포함 전역 격자**로 채점된다.
   ⟹ 0.00201 은 in-sample 적합값이 **아니다**. 시간축으로도 공간축으로도 외삽이다.
2. ★**FITPHI 두 사이트 모두 프로브에 덮인다.** `coarse_to_fine.py:172` 와
   `sequential_update.py:1187` 이 **둘 다 호출 시점에** `from ...initializer import _rx2_gate` 를
   하므로, 러너가 `I._rx2_gate` 속성을 갈아끼우면 양쪽이 잡힌다.
3. ★**원자 카운터의 dtype 가정이 옳다.** `initializer._linear_projection` 은 성분 **하나당 스칼라**
   `(amplitude, phase)` 를 반환하고(`:417-442`), `coarse_to_fine.py:173-175` 는 그 값을 변형 없이
   `torch.tensor(..., dtype=torch.float32)` 로 넣는다 ⟹ float32 집합 대조가 **위음성을 만들지 않는다**.
4. ★★★**저장소의 어떤 `ORACLE-*` 행도 λ 를 기록하지 않는다.** `tmp/**/*.jsonl` 전수 스캔:
   `ORACLE-K` 288행(S3 96 · K6 48 · B2 36 · rx2 12 · S1 96) · `ORACLE-NOTRACK` 96 · `ORACLE` 60 —
   **`lambda` 를 담은 키가 0개**, `lambda_l1==0` 인 행도 **0개**. (§6-B 에서 다시 쓴다.)
5. ★**M1 의 닫힌형 측정에는 아카이브가 없다.** `tmp/DEBATE_method_2026-08-15/` 는 `.md` **4개뿐** —
   스크립트도 `.jsonl` 도 없다. 저장소가 가진 "정보는 충분하다" 근거는 **산문 한 문단**이 전부다.

---

## 1. 이 배치가 사는 것 — 한 문장

> **저장소는 "벽은 위상 하위공간의 최적화 실패다" 라고 척추에 써 놓고도, 그 벽의 *바닥*(오라클 k 에서
> 경사하강을 완전히 제거한 닫힌형 해)과 그 벽의 *유력 용의자*(shrinkage 벌점)를 **출하 채점기 위에서
> 단 한 번도 같이 잰 적이 없다** — W3 는 48 에피소드에서 그 두 칸을 동시에 채운다.**

### 지금 저장소에 **없는** 것 (전부 §0 에서 실측)

| 빈칸 | 저장소 현황 (MEASURED) | W3 가 채우는 것 |
|---|---|---|
| **`ORA-K` × λ=0 셀** | `tmp/**/*.jsonl` 전수 스캔에서 **0행** | 32 혼잡 × 3시드 = 96행 |
| **λ 가 기록된 오라클 행** | 444 `ORACLE-*` 행 중 λ 키 **0개** | G-3 객체 되읽기로 전 행 기록 |
| **출하 채점기 위의 닫힌형 상한** | M1 산문 1문단, 코드·데이터 **0개** | 48 에피소드 × `compute_all_metrics` |
| **위상 수리의 실집행** | 러너 8개 셀 dict 에 실렸으나 **결과 행 0회** (method-debate §5) | `KONLY-FITPHI` 96행, 개입 3/3 실측 |
| **개입 지점의 런 전 실측** | 저장소가 같은 결함에 **5번** 물림 (§5) | 원자 카운터가 스모크에서 이미 1건 적발 |

⚠**정직한 한계**: 이 배치는 **개선축을 사지 않는다.** `ORA-K*` 는 전부 GT 를 쓰는 연구 팔이고
(PREREG §9.1), 유일한 배포 후보 `SHIP-FITPHI` 는 스모크가 이미 **사망 예고**했다(§5). 사는 것은
**진단**이지 **처방**이 아니다. 그래도 사는 이유는 §3(실패해도 남는 것)에 있다.

---

## 2. 통과 시나리오의 논문 문장 (축자 초안)

⛔**아래 숫자칸은 전량 결과로 채운다. 스모크(n=1/층) 값을 논문에 넣으면 무효.**
⛔전량에서도 `ORA-K@L3` 이 `ZERO` 를 못 이기면 `AMENDMENT_1 A-3` 의 표현 규율이 발동한다.

### 2.1 본문 (영문, 판정 = `LAMBDA-ARTIFACT` 또는 `LAMBDA-HURTS-ONLY`)

> On 32 held-out synthetic episodes in the crowded regime (15 and 30 components, BF4–BF5,
> drawn from two independently generated corpora), we separate the *solver* from the
> *estimation problem* by fixing the wavenumbers to their ground-truth values and solving
> only for the per-component amplitude and phase. Written in the Cartesian pair `(a,b)`
> via `w·cos(θ+φ) = a·cosθ + b·sinθ`, that residual problem is linear, and a single
> unregularised least-squares solve on the same initialisation-window LiDAR returns a
> median nRMSE of **[CVX_med]**. It is scored on the held-out tracking frames that the
> fit never saw, over the full domain including unobserved regions, with the identical
> metric applied to every other arm. Handing the same oracle wavenumbers to our
> gradient-based solver leaves **[NOREG_med]** at `λ = 0` — **[ratio]×** the closed-form
> bound — and **[L3_med]** at `λ = 10⁻³`, the penalty setting under which four earlier
> oracle-wavenumber experiments returned null. Removing the ℓ1 penalty recovers
> **[wins]/32** episodes (one-sided sign test on per-episode medians of three seeds,
> exact `p = [p]`, pre-specified threshold 23/32, pre-run power 0.910 at `p_win = 0.80`).

**핵심 문장 (초록용)**

> The information needed to reconstruct the field is present in the observations; what is
> missing is a solver that can reach it. With the support fixed, a closed-form solve is
> **[ratio]×** more accurate than our gradient descent on the identical data and the
> identical metric.

### 2.2 범위 문장 (⚠**다섯 개 전부 필수** — 하나라도 빠지면 이 절 인용 무효)

> **Scope.**
> **(i)** All measurements are synthetic; no real-sea claim is made.
> **(ii)** The oracle arms are not deployable. They receive the ground-truth wavenumbers,
> the ground-truth component count, and a single 900-step stage; this is a **three-factor
> bundle** and is labelled as such wherever it is compared to the shipped configuration.
> **(iii)** ★**The synthetic surface is generated as a sum of plane waves and our model is a
> sum of plane waves, so the model class contains the truth exactly.** The least-squares
> arm therefore bounds the gap attributable to the **solver and the `(w,φ)` polar
> parameterisation**, and says **nothing** about the gap attributable to model class on a
> real sea. It is a solver bound, not a representation bound.
> **(iv)** The least-squares arm is **unweighted** and carries **no tracking update**, while
> the gradient arms use confidence weighting and ten tracking steps per frame and are fit
> on strictly more data. Both asymmetries are designed to favour the gradient arms, so the
> reported ratio is a **lower bound** on the solver gap.
> **(v)** The design, arms, thresholds, direction and aggregation order were frozen with a
> self-attested md5 record **before any run, the smoke run included**. This is a
> **self-attested freeze**, not third-party preregistration.

### 2.3 함께 인쇄해야 하는 것 (PREREG §5)

- 크기 3규약 **A/B/C 전부**(절대 중앙값 차 · 에피소드별 상대차 중앙값 · 풀링 상대차).
- **셸 분해 9키** — `SUPPORT-TRADEOFF` 7회 재현 이력. ⚠단 `reference_citation_bans`(08-12 BLANKSLATE)
  가 ⛔**"어떤 셸도 '배로부터의 거리' 로 인용 금지"** 를 걸었다 ⟹ `obs:` 접두어가 붙은 이 배치의
  키는 반드시 **"관측점까지의 최소거리"** 로만 서술한다. (러너가 접두어를 붙인 것은 옳은 설계다.)
- 혼잡 32 의 **코퍼스 16/16 분해**(`indep` vs `s9`) — 풀링 판정은 사전 동결이나 분해는 서술 의무.
- ⛔**지연(ms) 인용 금지**(PREREG §9.4).

---

## 3. 실패해도 남는 것 — ★**이 배치의 가장 강한 변호**

### 3.1 헤드라인은 PRIMARY 에 **의존하지 않는다**

`ORA-K-CVX` 는 가설검정 대상이 아니라 **상한 서술**이다(PREREG §4.2). P-1 이 23/32 를 못 넘어도
48 에피소드의 상한표는 그대로 인쇄된다. ⟹ ★**이 배치는 방향에 건 도박이 아니다.**
CVX 팔만 따로 떼면 **48잡 · 추정기 런 0회 · 약 3분**이다(§4).

### 3.2 `NOT-A-LAMBDA-ARTIFACT` 가 나오면 확정되는 것

1. **4연속 널은 λ=1e-3 의 인공물이 아니다** — n=32, 사전 검정력 0.910(p_win=0.80).
   이 저장소가 축을 죽여 온 통상 검정력보다 **높다**.
2. **M4 의 잔여-벽 가설이 생존**한다(성분 수와 함께 자라는 벽).
3. `ORA-K-CVX` 가 **벽의 위치를 지정한다** — 상한이 ~0 인데 λ 로 설명이 안 되면, 남는 후보는
   **솔버/매개화**이고 **정보도 조건수도 아니다**(cond 32.5, rank full — 스모크 실측).

### 3.3 ★★★그리고 그것이 **척추에 더 유리하다**

M3 의 척추 개정안은 *"지지집합 **안**에서는 위상 하위공간의 **최적화 실패**가 벽"* 이다.

| P-1 결과 | 진단 이름 | 척추에 대한 효과 |
|---|---|---|
| **통과** | 벌점이 범인 | ⚠**약화**. lasso 편향은 **이미 이름이 있다**(M1: post-lasso 재적합 · VARPRO). 처방이 교과서에 있으니 novelty 는 더 멀어지고, 서사는 "우리가 정규화를 잘못 걸었다"로 축소된다. |
| **미달** | 솔버/매개화가 범인 | ★**강화**. 벌점이라는 평범한 설명이 **배제**되고, CVX≈0 이 정보·조건수를 동시에 배제한다 ⟹ 척추 문장이 **소거법으로 좁혀진다**. |

⟹ ★**PRIMARY 실패가 척추에는 더 좋은 결과다.** 두 갈래 모두 값이 나오는 배치는 이 저장소에 드물다.

⚠**정직한 반대 항목**: 미달이면 P0(값싼 수리)는 죽고 **개선축은 하나도 안 산다**. 다만 MEMORY 가
기록한 이 저장소의 자산 1위는 **"음성-기제"** 이고, 이 미달은 **검정력이 있는 음성**이다
(저장소 전례 다수는 검정력 부족으로 `UNDECIDED` 였다 — S6 13/16, S8 22/32).

### 3.4 P-1 과 **무관하게** 남는 3건

1. **G-4 문서화**: 배포 경로에 위상 수리의 **개입 지점이 없다**는 사실을 32 에피소드에서 확정.
2. **S-4** (`KONLY-FITPHI < KONLY`): P-1 과 통계적으로 독립. Rx-2 의 −0.0337(4/4) 재시험.
3. **원자 카운터 + 객체 되읽기 장치**(§5) — 결과와 무관하게 **다음 배치로 이월되는 산출물**.

---

## 4. 더 싸게 같은 답을 얻는 길 — **있다. 그리고 이미 사전등록돼 있다.**

PREREG §10 이 축약 순서를 **동결**해 두었다: *"PRIMARY 4팔(#1,#2,#9,#10)은 절대 줄이지 않는다"*.
그 4팔이 정확히 축약본이다. 러너 `build_jobs()` 를 직접 실행해 잡 수를 셌고, 스모크의 **팔별
최소 `wall_s`**(경합 없는 조건에 가장 가까운 값)로 비용을 재계산했다.

| 계획 | 팔 | 잡 | 순차 상한 | 8워커 (×7.62) | GPU | 답하는 것 |
|---|---|---|---|---|---|---|
| **T0 상한만** | `ORA-K-CVX` | **48** | 0.04 h | **≈3 분** | 아니오 | 상한표 48 에피소드 (헤드라인 절반) |
| **T1 PRIMARY** | +`ZERO`,`ORA-K@L3`,`ORA-K-NOREG` | **384** | 2.65 h | **≈0.35 h** | 아니오 | **P-1 전 검정력** + 퇴화 필요조건 + 상한 |
| **T2 오라클 완본** | +`ORA-K@L4`,`ORA-K@L3-DUP` | 624 | 4.84 h | **≈0.64 h** | 아니오 | +S-1 +**G-5 누출 게이트** |
| **T3 Phase A** | +`KONLY`,`KONLY-FITPHI` | 912 | 7.94 h | **≈1.04 h** | 아니오 | +**S-4 = 위상 수리의 유일한 실시험** |
| **T4 전량** | +`SHIP`,`SHIP-FITPHI` | 1,200 | 9.82 h | +Phase B ≈0.6–0.7 h | **예** | +S-2(F-3) +S-3(사망 예상) |

### 4.1 "PRIMARY 만" 은 성립하는가 — **성립한다. 단 대조 팔은 못 뺀다.**

- **`ORA-K@L3` 은 뺄 수 없다** — 그게 PRIMARY 의 **대조 그 자체**다.
- **`ZERO` 는 뺄 수 없다** — `AMENDMENT_1 A-3` 의 `(DEGENERATE-CONTROL)` 판정이 `ZERO` 없이는
  불가능하고, 스모크가 이미 그 조건이 **발동할 것**이라고 예고했다. 비용은 48잡 · 0.02 h 다.
- **`ORA-K@L3-DUP`(96잡 · 0.12 h) 은 빼지 말 것.** 이건 통계가 아니라 **런 수준 누출 탐지기**다.
  MEMORY 08-12: *"S7·S8 을 지금 재실행하면 기준선 팔이 처치값을 받는다"* — 이 저장소는
  **대조가 조용히 처치를 받는 사고를 이미 겪었다**. 0.12 h 로 그 사고를 배제한다.
- **`ORA-K@L4`(144잡 · 0.17 h)** 는 **현 출하 λ** 다. 가장 외부 관련성이 높은 λ 를 0.17 h 에 사는 것은
  싸다. S-1 = M4 의 "6/6 나쁨" 재시험.

### 4.2 ⛔**시드는 줄이지 말 것** (가장 커 보이는 절감이 함정)

median-of-3 이 추정기 팔을 3배로 만든다(1,104 잡). 줄이면 368 잡이 되지만:
- G-6 실측 시드 퍼짐이 **0이 아니다**: `ORA-K@L3` max 8.80e-3 · `ORA-K-NOREG` 2.39e-2 ·
  `KONLY` 5.93e-2 (dryrun 로그). ⟹ **median-of-3 은 가짜복제가 아니라 실제로 일하고 있다.**
- MEMORY 2026-07-23: **추정기 분산 CV 53%, 처방 = median-of-k**. 저장소 자신의 처방이다.
- 1시드로 줄이면 `ORA-K-NOREG` 의 2.4e-2 퍼짐이 부호검정 문턱(23/32)을 직접 흔든다.

### 4.3 권고 실행

★**T3(Phase A, ≈1.04 h) 를 먼저 완주하고, Phase B(`SHIP` 짝)를 별도로 돌린다.**
근거: Phase A 는 **GPU 를 안 쓰므로 OOM 위험이 0** 이고(스모크의 유일한 실패는 `SHIP-FITPHI`),
T3 까지가 P-1 · S-1 · S-4 · 상한 · 게이트 전부를 담는다. Phase B 는 S-2 하나(F-3, 진짜 과학적 질문)와
S-3(사망 예상)만 남으므로, 시간이 부족하면 **여기서 자르는 것이 사전등록된 순서와 일치**한다.
⚠단 자르면 `analyze_w3.py` 가 해당 SECONDARY 를 `NOT-RUN` 으로 인쇄해야 하고, **`NOT-RUN` 을
"측정했으나 널" 로 서술하는 것은 금지**다.

---

## 5. 설계에서 승계할 것 — **원자 카운터** (Q5)

### 5.1 이 장치가 실제로 한 일 (MEASURED)

`run_w3.py:170-242` `install_probe()`. **3층 계측**이다.

| 층 | 필드 | 무엇을 재나 | `SHIP-FITPHI` 혼잡에서 |
|---|---|---|---|
| ① 환경 | `rx2_fitphi_env` | 워커 프로세스가 **실제로 본** env | `'1'` — **통과했을 것** |
| ② 호출 지점 | `fitphi_site_hits` | `_rx2_gate` 호출을 **`파일:행\|T/F`** 로 계수 | **빈 dict** (사이트 미도달) |
| ③ **원자** | `fitphi_atoms` / `randphi_atoms` | 새로 심긴 **각 원자의 `init_phi`** 가 적합된 값인가 난수인가 | **0 / 0** |

★**③층이 왜 ②층보다 강한가**: `coarse_to_fine.py:173` 의 분기는
`if _rx2('RX2_GROW_FITPHI') **and phi_new is not None and len(phi_new) >= n_new**` 다.
게이트가 True 를 반환해도 **뒤의 두 조건이 깨지면 여전히 `torch.rand(n_new)*2π`** 로 간다.
②층만 있으면 그 경우 `…:173|T` 히트가 찍혀 **"개입했다"고 오판**한다. ③층은 오판할 수 없다.

★**독립 확증이 있다**: 혼잡에서 `SHIP` 과 `SHIP-FITPHI` 의 세 시드 nRMSE 가
**완전히 동일**하다(0.22566 / 0.23790 / 0.25489, §0 재도출). 계측(③층)과 결과(비트 동일)가
**서로 독립한 두 경로로 같은 결론**을 낸다 = `feedback_dual_verification_default` 충족.

### 5.2 ★장치를 장치로 만드는 것은 카운터가 아니라 **사전등록된 판정 이름**이다

저장소가 다섯 번 물린 결함의 정체는 *"측정할 수 없었다"* 가 **아니다**. 매번 사후에는 알아낼 수
있었다. 없었던 것은 **개입 부재를 판정으로 승격시키는 규칙**이다. PREREG §2.2 가 그것을 만들었다:

> 개입 부재 비율 **≥50%** ⟹ 해당 SECONDARY 는 판정 없이 **`UNDECIDED-NO-INTERVENTION`** 으로 인쇄.

`analyze_w3.py:310` 이 이 이름을 실제로 인쇄하고 `:318` 이 Holm 통과를 강제로 막는다(코드 확인).

### 5.3 저장소가 물린 **5건**에 소급 적용하면

| # | 사건 | 결함의 형태 | 잡았을 층 |
|---|---|---|---|
| 1 | **S1** — 출하 팔이 `beamform` **0회 호출**(G1P1 배포 불가) | 코드경로 | ②③ |
| 2 | **method-debate** — `RX2_GROW_FITPHI` 가 러너 8개 셀 dict 에 실렸으나 **결과 행 0회 등장** | 코드경로 | ②③ |
| 3 | **S11** — `learned_init_iters` 축이 **런 전에 사망**(닿는 단계에 격차 0) | 코드경로/용량 | ③ |
| 4 | **P1** — 개입 지점 **4건 부재** | 코드경로 | ②③ |
| 5 | **S7·S8** — **기준선 팔이 처치값 λ 를 받음**(대조 오염) | 설정값 | **③이 아니라 G-3 객체 되읽기** |

★**결론: 템플릿은 하나가 아니라 두 개이고, 둘이 짝을 이뤄야 5건을 전부 덮는다.**

> **T-A (설정값형 처치)** — 의도 dict 가 아니라 **살아 있는 객체**에서 되읽는다.
> `run_w3.py:249-298` (`install_updater_capture` + `_readback`). 앙상블 멤버까지 각각 읽어
> **멤버 하나가 다른 λ 를 숨길 수 없다**(`member_lambda_l1`).
> **T-B (코드경로형 처치)** — 처치의 **산출물이 기록되는 슬롯**을 지목하고, 생산자를 감싸 산출물을
> 기록하고 소비자를 감싸 **슬롯마다 생산자 유래인지 기본값 유래인지 귀속**한다.

### 5.4 T-B 를 다른 처치로 옮기는 4단계 (일반화 가능한 부분)

1. 처치의 **payload** 를 "식별 가능한 슬롯에 쓰이는 값" 으로 정의한다.
2. **생산자**를 감싸 payload 를 기록, **소비자**를 감싸 슬롯마다 귀속한다.
3. 귀속 결과를 **DCC 필수 필드**로 매 행에 기록한다(빠지면 `SCHEMA-FAIL`).
4. **문턱과 판정 이름**을 사전등록한다 — 이것이 없으면 그냥 진단이지 장치가 아니다.

### 5.5 ⚠이 장치의 한계 (변호하지 않는다)

- **동일 프로세스 몽키패치**다. 처치가 서브프로세스나 다른 임포트 경로로 도달하면 놓친다.
  (W3 는 두 사이트가 **호출 시점 임포트**라서 안전하다 — §0-2 에서 코드로 확인.)
- **float32 집합 대조**는 위양성 확률이 기록 집합 크기에 비례한다. 원자당 약 `|set|·1e-8` 수준으로
  무시 가능하나, **가정이지 측정이 아니다**. 소비자가 payload 를 변형해 저장하면 **위음성**이 난다
  (W3 에서는 변형 없음을 코드로 확인 — `:173-175`).
- **`≥50%` 문턱은 검정력 논증 없이 정해졌다.** 설계 선택이지 유도된 값이 아니다.
- **범위**: T-B 는 "생산자–소비자" 형태가 없는 처치(스칼라 설정, 손실 항 가중 등)에는 안 붙는다.
  거기서는 T-A 를 써야 한다.

### 5.6 ★승계 권고 (비용 0)

M4 실측: `def sign_one_sided` 가 **4개 파일에 복붙**, PREREG 50개 · 러너 85개에 **공유 라이브러리 0**.
`install_probe` / `install_updater_capture` / `_readback` / `missing_fields`(DCC) 는 **이미 동작하는
코드**다. `tmp/_lib/manipulation_check.py` 로 추출하는 것이 이 배치의 **가장 이전 가능한 산출물**이고,
결과와 무관하게 회수된다. ⚠단 **이 배치 도중에는 하지 말 것** — `run_w3.py` 가 G-7 코드 지문에
포함돼 있어 러너를 건드리면 게이트가 즉시 깨진다(의도된 설계, `run_w3.py:151-154`).

---

## 6. 변호하지 못한 것 (PRO 라도 적어야 하는 것)

**A. PREREG §1-1 의 동기 문장이 인용금지 프레이밍을 그대로 되쓴다.**
*"`ORA-K` 계열 4연속 널 — K6 (p=0.79)"*. `reference_citation_bans`(08-12 B-1)는
⛔*"K6: 완벽한 k 를 줘도 0(p=0.79)"* 를 금지하며 **같은 파일의 `ORACLE−AUTO` = −0.0640, 14/14,
p=1.22e-4 ⟹ "k 통로는 열려 있다"** 를 이미 확정했다. PREREG 는 §9.5 에서 3요인 묶음을 라벨하고
§1.1 F-3 이 결론을 제한하므로 **규율 위반은 아니다**. 그러나 **RESULT.md 가 이 동기 문장을 그대로
복사하면 금지 프레이밍이 재전파된다.** ⟹ **GO 조건 ②**.

**B. "네 번 모두 λ=1e-3 에서 측정됐다" 는 MEASURED 가 아니라 ARGUED 다.**
§0-4 실측: `ORACLE-*` 444행 중 **λ 를 담은 키가 0개**. 전제는 배포 날짜(08-11)로부터의 **추론**이다.
⟹ ★역설적으로 이것이 **배치를 더 강하게 변호한다**: W3 는 저장소 최초로 **λ 가 기록된 오라클 측정**이
된다. 다만 PREREG 문구는 "기록됨" 이 아니라 **"배포 시점으로부터 추론됨"** 으로 고쳐 적어야 한다.

**C. `AMENDMENT_1 A-3` 는 스모크 방향을 본 뒤에 쓰였다.** 스스로 그렇게 적었고, 문턱·표본·방향·통계량을
건드리지 않으며 **주장을 약화시키는 방향으로만** 작동한다. 수용 가능하나 **동결이 완전 맹검은 아니다**.

**D. `SHIP-FITPHI` 는 설계상 사망이 예고됐다.** 144잡을 써서 사는 것은 **시험이 아니라 문서화**다.
정직한 이름은 "S-3 는 배포 경로에 개입 지점이 없음을 32 에피소드에서 확정한다" 이지
"배포 위 수리를 시험한다" 가 아니다.

**E. `tmp/W3_SOLVER_2026-08-15/` 는 git 미추적이다**(NOTES 마무리 점검, 이 경로는 gitignore 대상).
M2 감사가 *"다음 배치부터 커밋+push"* 를 권고했고 **이 배치가 그 다음 배치**다. 커밋 없이 돌리면
동결의 유일한 증인이 **자기가 쓴 텍스트 파일**뿐이다. ⟹ **GO 조건 ①**.

**F. 혼잡 32 는 두 코퍼스 풀링**(`indep` 16 + `s9` 16). PREREG 가 풀링을 사전 동결하고 16/16 분해
병기를 의무화했으므로 규율 위반은 아니나, **두 코퍼스가 갈리면 그것이 결과의 일부**다.

---

## 7. 판정

### ★ **GO** — 조건 3개

1. ★**전량 실행 전에 `git add -f tmp/W3_SOLVER_2026-08-15/ && git commit`** (사용자 승인 필요).
   `PREREG.md` · `AMENDMENT_1.md` · `FROZEN_md5.txt` · `run_w3.py` · `analyze_w3.py` ·
   `smoke_results.jsonl` · `code_fingerprint.json`. 이유 = §6-E. **런 1초 뒤에는 소급 불가.**
2. ★`RESULT.md` 에 **§6-A(K6 프레이밍) · §6-B(λ 는 추론)** 를 인용금지 준수 문구로 명시.
3. ★`AMENDMENT_1 A-2` 의 **2위상 실행**(Phase A 8워커 → Phase B 3워커). 근거 = 스모크의
   유일한 실패가 GPU 팔이고, 이 저장소에서 **재발**하는 결함이다.

### 이유 3줄

1. **저장소의 척추가 주장하는 벽의 바닥과 그 벽의 유력 용의자가, 출하 채점기 위에서 한 번도 같이
   측정된 적이 없다** — `ORA-K × λ=0` 셀 **0행**, λ 를 기록한 오라클 행 **0개**, 닫힌형 상한의
   아카이브 **0개**(전부 §0 에서 전수 스캔으로 실측).
2. **헤드라인이 판정 방향에 걸려 있지 않다** — `ORA-K-CVX` 는 가설검정 대상이 아니라 상한 서술이고,
   그것만 따로 돌리면 **48잡·3분·추정기 런 0회**다. 게다가 ★**PRIMARY 실패 쪽이 척추에 더 유리**하다
   (벌점이라는 교과서적 설명이 배제되므로).
3. **비용은 사전등록된 축약 순서로 이미 통제돼 있다** — PRIMARY 만 384잡 **≈0.35 h**, GPU 안 쓰는
   Phase A 전체가 **≈1.04 h**. 게다가 스모크가 **런 전에** 배포 팔의 개입 부재를 원자 단위로 적발해
   저장소가 다섯 번 물린 결함을 여섯 번째로 되풀이하는 것을 이미 막았다 — **그 장치 하나가
   결과와 무관하게 이월된다.**
