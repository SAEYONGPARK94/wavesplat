> ⛔**2026-08-13 감사 — 이 로그의 수치를 출하 시스템의 성능·배포 근거로 인용하지 말 것.**
> ⚠**2026-08-13 3라운드 정정** — 이 자리에 처음 붙었던 *"`wavesim` 임포트 0회 / LiDAR 0회 호출 /
> 관측을 상자 균일난수로 만든다"* 는 **이 로그의 두 Step 에 대해 거짓**이었다(회귀검증 R3 §B-2).
>
> - **Step 1** (`scripts/d1_crb.py`) — ✅**실제 에피소드 LiDAR 점군을 쓴다**:
>   `:39` `from wavesim.data_writer import load_episode` · `:149` 호출 ·
>   `:154` 가 `lidar_points`·`lidar_valid_mask` 를 관측 좌표로 사용.
> - **Step 2** (`scripts/d1_predictable_zone.py`) — ✅**동상**(`:29` 가 `d1_crb` 의 `load_ep`/`collect_obs` 임포트).
> - ⚠**예외**: `fig_aperture_law.png` 를 만드는 `d1_crb.validate_aperture_law` 만 **상자 균일난수**
>   (M=300, L=50 m, 단일 성분).
>
> **출하 증거가 아닌 진짜 이유**(이쪽은 유효): 이 로그는 **CRB(하한) 계산**이고 출하 추정기를
> **한 번도 돌리지 않았다**. REPORT §4 의 "estimator" 열은 `results/spacetime_gp_30ep.json` 의
> **ST-DF-GP** 값을 인용한 것이지 출하 `SequentialUpdater` 가 아니다. CRB 는 참 성분·정합 차수
> 가정의 최선 경계이고, 깊이 `d = 50 m` · 잡음 `σ = 0.02 m` iid 가정 위에 서 있다.
>
> ⚠`19.97×19.96 m` 개구는 **`d74_closed_loop_trajectory.py`**(`L_FOOT = 20.0`)의 값이지 이 로그의 값이
> 아니다 — 여기 개구는 `crb_results.json` 의 **31.6×59.7 … 71.9×53.2 m**(실 LiDAR 발자국).
> 근거: `tmp/AUDIT_recheck_2026-08-13/{A3_citation_bans.md, R3_regression.md §B-2}`

# Direction I — 검증 로그 (CRB + synthetic aperture 이론)

리포트: [REPORT.md](REPORT.md) · 코드: [scripts/d1_crb.py](../../scripts/d1_crb.py) · 전환 배경: [direction2/REVIEW_LOG.md](../direction2/REVIEW_LOG.md) 판정 F

---

## Step 1 — FIM 유도 + 유효개구 + 정보의 벽 (핵심 GO)

**날짜:** 2026-06-27 · **GPU:** RTX 4080 Super (float64)

### A. 무엇을 했나 (사용자 요청 1~4 전부)
1. **FIM:** 다성분·분산제약·시공간 관측모델의 Fisher 정보행렬 F=(1/σ²)JᵀJ (4N×4N, 성분 교차항 포함).
   ω=√(gk·tanh(kd)) 분산제약이 핵심 도함수 `∂ψ/∂kx = x − c_g(kx/k)t` 로 들어감.
   → **해석적 Jacobian vs autograd 오차 5.3e-15 (기계정밀도). 검증 통과.**
2. **유효개구:** `D_eff = L_footprint + c_g·T_obs` 유도. 시간변위가 전파방향 k̂ 으로만 작용 →
   점군이 k̂ 방향으로 c_g·T_obs 만큼 늘어남. **통제실험(M·footprint 고정, T_obs만 변화)에서
   σ_|k| ∝ 1/D_eff, log-log 기울기 −1.09 (이상 −1.0) 확인.**
3. **정보의 벽:** 에피소드별 field-nRMSE CRB = √(tr(F⁻¹G))/Hs vs T_obs 곡선. T_wall = CRB가
   0.25(zero-predictor) 아래로 내려가는 T_obs.
4. **추정기 비교:** ST-DF-GP(1s 윈도우) 실측오차를 CRB 위에 overlay.

### B. 결과
**정보의 벽 (T_wall = 비편향 추정 가능 최소 관측시간):**
| 에피소드 | N | T_wall(CRB<0.25) | T(CRB<0.10) | cond(F)@1s |
|---|---|---|---|---|
| BF4 5c  |  5 | **0.1s** | 0.2s | 7.7e4 (양호) |
| BF4 15c | 15 | **4.3s** | 5.9s | 8.6e12 |
| BF4 30c | 30 | **9.7s** | 13.4s | 9.0e17 (수치적 특이) |

→ 30성분 해상은 **~10s ≈ 1.4 Tp** 관측 전엔 field가 식별 불가. cond 수가 진짜 rank-deficiency 확인(버그 아님).

**추정기 vs CRB (near-optimal 판정):**
| 에피소드 | ST-DF-GP fused | CRB@1s | 판정 |
|---|---|---|---|
| 5c  | 0.070 | 0.026 | **near-optimal** (한계의 2.7배) |
| 15c | 0.272 | 3.04 | CRB **무의미**(발산) → 추정기가 한계보다 좋음 |
| 30c | 0.308 | 195 | CRB **무의미**(발산) |

### C. 핵심 결론 (두 체제, 둘 다 결정적)
1. **식별가능 체제(5c):** 기존 추정기가 이미 CRB의 ~3배 이내 → **point추정기 개선 여지 거의 없음.**
   "point추정기 그만, UQ로" 전환을 이론으로 정당화.
2. **부족결정 체제(15/30c @1s):** 비편향 CRB가 발산 → 추정기는 **편향(amplitude reg + GP fusion)**
   덕분에만 작동. 비편향 추정은 zero보다 나쁨. → **편향/학습prior가 휴리스틱이 아니라 필수**임을
   엄밀 증명. Direction II(prior가 돕는다)와 Direction I(한계)가 한 점에서 만남.

### D. 미세방향의 비대칭 (Step1 30°-flat의 이론적 근거)
- synthetic aperture(c_g·T_obs)는 **전파방향(|k|, ω)만** 날카롭게 함. **방향(cross-range)엔 무효** —
  단일성분 방향 CRB는 T_obs에 평평(~0.25°). 방향은 물리 footprint(cross-range)에만 의존.
- 즉 Direction II의 "30° 근방 likelihood flat" = footprint cross-range 부족 = Fisher 정보 낮음.
  시간은 다성분 분리(c_g·T_obs ≳ L_foot 필요)를 통해서만 간접적으로 방향을 돕는다.
- 실전 1s 윈도우: c_g·1s≈5m ≪ footprint 60m → synthetic aperture 거의 0. **단일윈도우 한계의 정량 근거.**

### E. 박사 novelty / contribution
- **양성 기여 (도메인 최초):** ship-LiDAR 파랑추정의 **분산제약 시공간 CRB** 정식화 + 유효개구
  D_eff=L+c_g·T_obs 유도/검증 + "정보의 벽" T_wall(N) 정량화. 이건 추정기 논문이 아니라
  **문제의 근본한계 논문** — 더 강한 박사 기여.
- **통합서사 완성:** (I)벽 존재 → (II)prior 필수 → (winning) space-time 융합이 벽을 낮추는 유일한 레버.
  부정결과(T-NSE, off-grid)가 "왜 영리한 추정기로 안 되는가"의 *증명*으로 승격.
- **정직:** depth=50 가정(분산 잔차<0.3% 검증), true-component matched-order CRB(best-case),
  단일 seed 곡선. 다seed 밴드는 다음 단계 cheap.

### F. 판정
✅ **GO — CRB 백본 성립.** FIM 검증·유효개구 검증·정보의 벽 정량화·추정기 near-optimality 판정 모두 완료.

---

## Step 2 — predictable-zone CRB (식별성 기반, 기하학 아님)

**날짜:** 2026-06-27 · 코드: [scripts/d1_predictable_zone.py](../../scripts/d1_predictable_zone.py)

### A. 무엇을 했나
관측창 [0,t_now]의 F로부터 예보시각 t*=t_now+Δt 에서 점별 field CRB σ_z(x,y,t*)/Hs=√(gᵀF⁻¹g)
공간맵 계산. predictable zone = {σ_z/Hs < 0.10}. 기하학적 c_g-cone(footprint를 성분별 군속도로
이류) overlay로 대조.

### B. 결과 (두 체제)
| 체제 | 설정 | Δt=0 | Δt=4s | Δt=8s |
|---|---|---|---|---|
| **식별됨** | 5c, t_now=2s (벽 통과) | 100% 격자 | 100% | 100% |
| **부분식별** | 15c, t_now=5s (벽 근처) | 77% | 87% | 90% |

### C. 핵심 발견 — predictability는 식별성 속성이지 전파 속성이 아니다
- **식별된 모델(5c)**: 한번 파라미터가 잡히면 field는 **모든 공간·모든 미래시각에서 예측가능** →
  기하학적 predictable zone 자체가 없음 (parametric 전역모델이라 당연).
- **부분식별(15c 벽 근처)**: zone이 **유계+구조적**(σ/Hs 최대 0.45, 국소적으로 zero보다 나쁜 패치).
  경계는 **FIM이 어떤 스펙트럼 성분을 고정하는가**로 결정 — **c_g-cone과 불일치**(cyan은 footprint
  근처, predictable 영역은 훨씬 멀리). → 고전 DSWP(기하학적 cone)와 **본질적으로 다른** novel 결과.
- → (C) 연결: CRB 맵 = 물리기반 predictability mask. 보정UQ/hydrofoil feedforward의 신뢰영역.

### D. 판정
✅ **GO — novel predictable-zone 정식화.** "predictability = 정보의 벽을 field로 전파한 식별성"이
한 줄 결론. 다음: ST-DF-GP UQ를 이 CRB mask에 보정 (C 작업).
