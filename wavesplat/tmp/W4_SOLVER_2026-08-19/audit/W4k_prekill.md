# W4k — 적대적 사전 킬 (런 0회, 2026-08-19)

대상: `W4 · SOLVER-REMEDY` (post-lasso · VARPRO · Sliding Frank–Wolfe · `merge_atoms`).
근거: `tmp/W3_SOLVER_2026-08-15/results.jsonl` **1,212행 재집계** + 배포 코드 열람. 신규 런 0회.
표기: **MEASURED** = 원자료/코드에서 직접 확인 · **ARGUED** = 추론.

## 판정 요약

| # | 공격 | 판정 |
|---|---|---|
| 4 | post-lasso 가 이미 되어 있는가 | **KILL** |
| 2 | 상한 ×112.8 이 도달 가능한 목표인가 | **KILL** |
| 3 | SFW 가 배포 경로에서 도는가 | **KILL** |
| 5 | novelty — 1973년 방법 + "진단이 정당화한다" 논지 | **KILL** |
| 1 | W3 `KONLY-FITPHI` 의 재탕인가 | **SURVIVES** |
| 6 | `merge_atoms` 기록이 옳은가 | **SURVIVES** (기록 정확) |
| 7 | 재발 검사 | **WOUND** (기존 기록 정정 1건) |

---

## 1. 재탕 여부 — **SURVIVES**

**MEASURED.** post-lasso 와 `KONLY-FITPHI` 는 **다른 조작**이다.

`RX2_GROW_FITPHI` 개입 지점은 두 곳뿐이고(`coarse_to_fine.py:173`, `sequential_update.py:1188`)
둘 다 **새로 추가되는 원자의 `init_phi`** 만 바꾼다. 그 위상의 출처는
`sequential_update.py:1144 _linear_projection(all_x, all_y, z_work, ...)` = **잔차에 대한 원자 1개짜리
rank-1 그리디 사영**이다(`initializer.py:417-450`, 스칼라 `cc/ss/cs` 2×2 해). 진폭 `amp` 는
원래도 그 사영에서 왔고 **위상만 버려지고 있었다**.

post-lasso 는 **지지집합 전체에 대한 결합 2N 열 무벌점 해**다. 혼잡처럼 원자가 상관될 때
그리디 rank-1 사영과 결합 해는 일치하지 않는다.
⟹ **"온전히 개입해도 안 산다"(`KONLY-FITPHI` 20/32 FAIL)는 post-lasso 를 직접 반증하지 않는다.**

⚠단 이 생존은 4번에서 다른 경로로 죽는다.

---

## 2. 이미 되어 있는가 — **KILL**

**MEASURED.** 결합 2N 무벌점 LS 는 **이미 배포 혼잡 경로에 들어 있다.**

`sequential_update._initialize_learned` → `learned_init.learned_init_params:386` →
`ls_fit_wphi` → `model/tnse.py:176 fit_amplitudes_phases`. 그 함수는

```
psi  = kx·x + ky·y − ω·t ,  ω = √(g|k|tanh(|k|d))
Phi  = [cos ψ | sin ψ]                      # (M, 2N)
coef = solve(Phiᵀ Phi + λ_reg·I, Phiᵀ z)
A    = √(a²+b²) ,  φ = atan2(−b, a)
```

이것은 `PREREG §2.1` 의 `ORA-K-CVX` 정의 **4–5단계와 식이 동일**하다(CVX 는 `lstsq`, 즉 λ_reg=0).
⟹ **`ORA-K-CVX` 와 배포 초기화의 차이는 정확히 둘뿐: (i) 오라클 k vs 학습 k, (ii) ridge 0.5 vs 0.**

### 2.1 ridge 0.5 의 크기 — 중앙에서 무시 가능

**MEASURED** (`results.jsonl`, `ORA-K-CVX` 48행):

| 량 | min | med | max |
|---|---|---|---|
| `cvx_n_obs` (M) | 6,975 | **8,026** | 9,179 |
| `cvx_cond` = sv[0]/sv[-1] of Φ | 1.02 | **11.80** | 275.7 |
| `cvx_rank` | 4 | 30 | 60 (= 2·n_gt 전건) |

`Phiᵀ Phi` 대각 ≈ M/2 ≈ **4,013**. ridge = 0.5 ⟹ 상대 크기 **≈1.2e-4**.
⟹ **ARGUED**: 중앙 조건수(11.8)에서 ridge 0.5 는 최소 고유값 대비 ~1% 미만 ⟹ **수치적으로 no-op**.
⟹ **post-lasso 의 "ridge 를 끄는" 델타는 중앙에서 사실상 0 이다.**

⚠**반대 방향 위험 (ARGUED)**: `cvx_cond` 최대 275.7 은 **오라클 k** 기준이다. 배포는 학습 k 이고
분리 문턱 `learned_dedup_dk=0.03` 이 측정 분해능(반경 0.042–0.074)보다 작아 **한 주엽 안 중복 원자**가
남는다 ⟹ 그 셀에서 Φ 는 훨씬 나쁘게 조건화되고 **ridge 0.5 가 유일한 안정자**다.
⟹ **ridge 제거는 개선이 아니라 퇴행일 수 있다.** ridge 제거는 `merge_atoms` 와 **묶어야만** 안전하다.

### 2.2 남은 델타 = **배치 위치**뿐

배포 순서는 `LS 적합(초기화) → GD → 추적`이다. post-lasso 의 고전적 정의는
*"벌점 해가 고른 지지집합 위에서 무벌점 재적합"* ⟹ **GD 이후**여야 한다. 그건 아직 미측정이다.

⛔**그러나 그 델타를 지금 주장할 수 없다** — 저장소에 **GD 직전 LS 해의 점수가 없다.**
`init_nrmse`(`run.py:185-189`)는 `updater.predict_field` 를 **Phase-1 최적화가 끝난 뒤** 부르므로
**GD 이후 값**이다(초기 시드 품질이 아니다).

**MEASURED** (혼잡, median-of-3 아님·행 중앙값):

| 팔 | `init_nrmse` (Phase-1 출력) | `avg_nRMSE` (추적 평균) |
|---|---|---|
| `SHIP` | 0.1447 | 0.1596 |
| `ORA-K-NOREG` | 0.1293 | 0.1327 |
| `ORA-K@L4` | 0.1332 | 0.1368 |

⟹ 손상의 대부분은 **Phase-1 안에서** 이미 발생해 있다.

> **최소 변경**: post-lasso 팔을 **초기화에 넣지 마라(이미 배포됨)**. 유일하게 새로운 것은
> *"Phase-1 GD 이후의 최종 지지집합 위 결합 무벌점 재적합"* 이고, 그 델타를 귀속하려면
> **이미 구현된 `learned_init.raw_field_nrmse`(GD 0회 LS 해의 nRMSE)를 혼잡 32 에피소드에서
> 먼저 기록**해야 한다. 그 값이 0.15 근처면 학습 k 의 천장이 이미 거기이고 **W4 는 값어치가 없다.**

---

## 3. 상한 ×112.8 이 목표인가 — **KILL** (두 겹)

### 3.1 오라클 팔은 k 를 고정하지 않는다 ⟹ ×112.8 은 "솔버"로 식별되지 않는다

**MEASURED** (`k_drift_rel_median`, `k_drift_valid` 행만):

| 팔 | 층 | n | med | p90 | max |
|---|---|---|---|---|---|
| `ORA-K@L3` | **혼잡** | 94/96 | **0.2495** | 0.3606 | 0.5111 |
| `ORA-K-NOREG` | **혼잡** | 94/96 | **0.2274** | 0.3649 | 0.5133 |
| `ORA-K@L4` | **혼잡** | 94/96 | **0.2249** | 0.3779 | 0.5215 |
| `ORA-K@L3` | 희소 | 44/48 | 0.0007 | 0.0539 | 0.4307 |
| `ORA-K-NOREG` | 희소 | 44/48 | 0.0004 | 0.0456 | 0.3499 |
| `ORA-K@L4` | 희소 | 44/48 | 0.0003 | 0.0463 | 0.3497 |

`ORA-K-CVX` 는 **k 를 완전히 동결**한다(추정기 미실행). 나머지 오라클 팔은 혼잡에서
**중앙 22.5–25.0% k 를 흘린다**. ⟹ ×95~×178 은 **"경사하강 (w,φ) 솔버"의 크기가 아니라
"k 표류 + (w,φ) 솔버 + 무추적 + 무가중"의 4요인 묶음**이다. W3 PREREG §9.7 이 이미 그렇게 라벨했다.

**MEASURED — 층별 대응이 이 해석을 지지한다**(per-episode median-of-3, CVX 대비 비):

| 층 | k 표류 중앙 | `SHIP`/CVX | `ORA-K@L4`/CVX |
|---|---|---|---|
| 혼잡 | **22.5%** | **112.8×** | 103.7× |
| 희소 | **0.03%** | 23.3× | **2.8×** |

⟹ **ARGUED (n=2 층, 상관)**: k 표류가 0 인 층에서 오라클 팔은 CVX 의 **2.8배 안**까지 간다.
표류가 22.5% 인 층에서만 100배가 열린다. **솔버를 갈아도 k 를 못 잡으면 아무것도 안 바뀐다**는
가설과 정합한다.

### 3.2 CVX ≈ 0 은 부분적으로 **구성상 참**이다

**MEASURED**: W3 세 코퍼스(`data_indep_crowded_2026-08-08`, `data_s9_crowded_2026-08-11`,
`data_indep_sparse_2026-08-08`)의 `_manifest.json` 에 HOS/비선형 키가 **0건** ⟹ 선형 GT.
`CLAUDE.md` BLOCKER: 저장된 `gt_wave_field` 를 `Σ A cos(kx·x+ky·y−ωt+φ)` 로 재구성하면
**max|Δ| = 0.000e+00** — GT 가 추정기 모델식과 **100% 동일**하고 ω 도 같은 분산관계다.

⟹ **ARGUED**: 오라클 k + 동일 ω + M≈8,000 관측 ⟹ Φ 는 **정확히 정합**한 선형계이고
정적 해는 시간 외삽까지 정확하다. `cvx_window_resid_rms`(med **0.00549**)가
`cvx_truth_window_resid_rms`(med **0.00571**)**보다 작다** = 진실보다 창을 잘 맞춘다 = **LiDAR 잡음 바닥**.
⟹ **CVX 0.00139 는 "솔버가 완벽하면 도달 가능한 값"이 아니라 "합성 우주가 정확히 표현 가능함"의
재확인**이다. HOSGATE 가 이 표현 가정이 **하중을 받는다**고 이미 개입 측정했다(`SHIP` 희소 3.6배).

> **최소 변경**: (a) **`ORA-K-FROZEN` 팔을 PRIMARY 대조로 추가**하라 —
> Phase-1 내내 `model.kx/ky.requires_grad_(False)` (기계는 이미 있다:
> `sequential_update.py:281-296` 의 `w_phi` 분기) + LS 시드 (w,φ) + λ=0.
> 이 팔이 없으면 W4 는 **k 표류와 (w,φ) 솔버를 분리할 수 없고** 어떤 귀속도 못 한다.
> (b) 헤드룸을 인용하려면 **CVX 를 HOS M=3 코퍼스에서 재측정**하라. 선형 코퍼스의 ×112.8 은
> 인용 금지 후보다.

---

## 4. SFW 가 배포 경로에서 도는가 — **KILL**

**MEASURED** (`SHIP` 144행 전수):

| 량 | 값 |
|---|---|
| `init_method_resolved` | `learned` **130/144** · `v2_konly` 14/144 |
| 혼잡만 | `learned` **87/96 = 90.6%** (메모리 기록 재현 ✅) |
| `add_calls > 0` | **23/144 = 16.0%** (혼잡 17/96 = 17.7%) |
| `lp_calls == 0` | **121/144 = 84.0%** |
| `fitphi_atoms` | **0/144 전건** (OFF 팔 누출 0 ✅) |

교차표: `(learned, add>0)` = **9/130 = 6.9%** · `(v2_konly, add>0)` = **14/14 = 100%**.

⟹ **배포 학습 경로에서 원자 추가 분기는 93.1% 의 런에서 한 번도 실행되지 않는다.**
SFW 는 **원자 추가(그리디 conditional-gradient) + sliding 을 교대**하는 알고리즘이다.
그 추가 사이트가 죽어 있는 경로에 SFW 를 "끼워 넣는" 설계는 **S1 의 "출하 팔은 beamform 0회 호출"
5번째 재발**이 된다.

> **최소 변경**: SFW 를 **성장 사이트 drop-in 으로 넣지 마라.** 유지하려면
> `_initialize_learned` **전체를 대체**하는 팔이어야 하고, PREREG 는 W3 의 `G-4` 형식 그대로
> **원자 단위 카운터로 개입률 ≥50% 를 런 전에 게이트**해야 한다.

---

## 5. novelty — **KILL**

**MEASURED (문헌)**:

1. **Sliding Frank–Wolfe** = Denoyelle, Duval, Peyré, Soubies, *Inverse Problems* 35(11) 2019
   (arXiv:1811.06416). BLASSO(off-the-grid ℓ¹)의 표준 해법으로, 논문 자신이
   *"interleaving convex optimization updates (adding new spikes) with non-convex optimization
   steps (moving the spikes)"* 라고 적는다 — **본 배치가 하려는 것과 문제 구조가 동일**하다.
2. **VARPRO** = Golub & Pereyra 1973; separable NLLS 의 교과서 방법.
3. ★★★**"진단이 낡은 방법을 옳은 방법으로 만든다"는 논지는 이미 정리로 출판됐다** —
   Gan, Chen, Yi, Yang, *A Saddle Point Remedy: Power of Variable Elimination in Non-convex
   Optimization*, **arXiv:2511.01234 (2025-11-03)**. Hessian inertia / Schur complement 로
   **변수 소거가 임계점 구조를 바꾸어 원 문제의 안장점이 축소 지형의 국소 최대가 됨**을 증명한다.
   ⟹ **W4 의 척추 논지(= "비볼록 위상 지형이 문제이므로 VARPRO 가 맞다")가 9개월 전에 선점됐다.**
4. 해양 쪽도 비어 있지 않다 — HOR 계열이 이미 *"optimizes the amplitude and phase of free wave
   components"* 를 최소제곱으로 푼다.

⟹ **ARGUED**: 성공해도 방법 novelty 0. 남는 기여는 **"어느 요인이 벽인지에 대한 통제된 측정"**
이고 그건 알고리즘 논문이 아니라 **음성 결과 논문**이다. 저장소의 기존 판정
(*"novelty 옹호 0표", 누적 21축 사망*)과 정합한다.

> **최소 변경**: W4 를 "새 솔버 제안"으로 프레이밍하지 마라. 인용 가능한 산출은
> **분해 측정 + 정직한 음성**뿐이고, arXiv:2511.01234 을 **선행연구로 반드시 인용**해야 한다.

---

## 6. `merge_atoms` — **SURVIVES** (기록이 옳다)

**MEASURED** (`optimizer/learned_init.py`):

- 구현 존재 ✅ (단일연결 응집, 반경 `r = merge_frac · kp`, 에너지 가중 중심, 에너지 합).
- **GT-free** ✅ — 입력이 `(kx, ky, energy)` 뿐. GT 참조 0건.
- **끄면 비트 동일** ✅ — `if merge_frac is None or merge_frac <= 0 or n <= 1: return kx, ky, en`
  (입력 배열 그대로 반환) 이고 호출부 `:367-369` 는 `if merge_frac:` 로 감싸 **falsy 면 호출 자체 없음**.
- **기본값 `None`** ✅ (`config.get('learned_merge_frac')`, 기본 미지정).
- **미노출** ✅ — `learned_merge_frac` 는 저장소 전체에서 `learned_init.py` **2회**(주석 + `get`)에만
  등장. `configs/`·`run.py` 에 **0건**. `learned_peak_anchor`·`learned_dedup_dk` 도 동일하게 0건.

⟹ 4개 처방 중 **유일하게 "싸고·GT-free·되돌릴 수 있고·실제로 배포 경로에서 도는"** 것이다
(`learned_init_params` 는 혼잡 90.6% 에서 실행된다 — 4번 표 참조).
⚠단 `merge_atoms` 는 **k 를 병합할 뿐 k 를 정확하게 만들지 않는다.** 3.1 이 옳다면 상한 기여는 작다.
⚠O(n²) 이중 루프지만 `max_n=30` 이라 비용 무시 가능.

---

## 7. 재발 검사 — **WOUND** (기록 정정 1건)

### ★정정: `k_drift` 의 "중앙 15%" 는 **층을 뭉갠 값**이다

`project_w3_solver_2026-08-16.md` §3 은 *"중앙 상대 표류 **14.9–16.2%**, 최대 34%"* 라 적었다.
**MEASURED 재집계**: 그 값은 **혼잡+희소 풀링**이다.

| 집계 | `ORA-K@L3` | `ORA-K-NOREG` | `ORA-K@L4` |
|---|---|---|---|
| **풀링**(n=138) | 0.1529 | 0.1618 | 0.1494 |
| **혼잡만**(n=94) | **0.2495** | **0.2274** | **0.2249** |
| 희소만(n=44) | 0.0007 | 0.0004 | 0.0003 |

희소가 사실상 0 이라 풀링 중앙값이 혼잡의 하위 사분위로 끌려 내려갔다.
⟹ ⛔**"오라클 팔이 k 를 중앙 15% 흘린다" 를 혼잡 문장으로 인용 금지.** 혼잡은 **22.5–25.0%** 다.
⟹ "최대 34%" 도 재현되지 않는다 — `k_drift_rel_median` 의 혼잡 최대는 **0.51–0.52**,
`k_drift_max`(절대) 혼잡 중앙은 **0.15**. **두 필드가 섞였을 가능성**이 높다.
⟹ 저장소 상시 규율 **"셸/층 접두어까지 적을 것"의 재발**이다(S13 이후 반복).

### 기타 (전부 W4 PREREG 가 런 전에 처리해야 함)

| 항목 | 상태 |
|---|---|
| 팔별 필드 존재 게이트 | W3 의 `DCC`/`G-2` 를 **그대로 승계할 것**. CVX/ZERO 는 `N_final`·`fitphi_atoms` 등이 **구조적으로 없다**(스키마 확인함) ⟹ 팔별 필수키 목록을 분리 유지. |
| 개입률 | SFW·post-lasso 팔에 **원자 카운터 필수**(4번). |
| 셸 접두어 | `obs:` 유지. 7번 정정이 같은 부류. |
| 가짜복제 | 혼잡 오라클 팔은 시드 퍼짐이 0 이 아니지만(`torch.rand` (w,φ)), **n 은 에피소드**. W3 `G-6` 승계. |
| in-sample | 학습 체크포인트는 평가와 같은 분포 ⟹ **"out-of-sample" 은 시드 축에서만**. |
| λ 재탕 | ⛔**λ 축은 W3 로 종결**(S-1 18/32 널, 출하 λ=1e-4 ≈ λ=0). W4 는 λ 를 **처치로 쓰지 마라**. 단 **전 팔이 λ 를 명시**해야 한다(F-1 함정). |
| 요약 > 사전등록 | 3.2 때문에 **어떤 "격차를 N% 닫았다" 문장도 선형 코퍼스 한정**으로 라벨. |

---

## 8. 종합 — 런 전에 반드시 바꿔야 할 것

1. **post-lasso 를 초기화 팔로 넣지 마라** — 이미 배포돼 있다(`ls_fit_wphi`, 혼잡 90.6%).
   새로운 것은 *GD 이후 재적합* 하나뿐이고, 그전에 **`raw_field_nrmse`(GD 0회)를 먼저 기록**하라.
2. **`ORA-K-FROZEN`(Phase-1 k 동결) 을 PRIMARY 대조로 추가하라** — 없으면 ×112.8 을
   k 표류와 솔버로 분해할 수 없다. 혼잡 k 표류는 **22.5%**(15% 아님).
3. **SFW 를 성장 사이트에 끼우지 마라** — 그 분기는 배포 학습 경로 **93.1%** 에서 0회 실행된다.
   유지하려면 초기화 전체 대체 + 개입률 게이트.
4. **novelty 를 주장하지 마라** — arXiv:2511.01234 이 논지를 선점했고 SFW/VARPRO 는 표준이다.
5. **헤드룸을 인용하려면 CVX 를 HOS M=3 에서 재측정하라** — 선형 GT 는 모델식과 max|Δ|=0 이다.
6. **살릴 것 하나**: `merge_atoms` 노출. 기록이 코드로 전건 확인됐고 배포 경로에서 실제로 돈다.
   단 ridge 제거와 **묶어서만** 안전하다(2.1).

**EOF**
