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
다음: (1) 다seed CRB 밴드, (2) **predictable-zone CRB**(목표 예보시각에 CRB<thr 공간영역) →
(C) 보정UQ/predictable-zone 작업과 연결, (3) ST-DF-GP 효율 vs T_obs 전구간 추적.
