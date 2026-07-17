# E-46 — 실측(REAL-DATA) leg: buoy-grade external regime supply

> **Deliverable**: e45 감사에서 유일하게 살아남은 헤드라인(합성 cross-Tp 아레나에서
> buoy-grade 외부 regime 공급이 deployment functional로 전이: buoy−cur +0.0349
> CI[+0.0084,+0.0615], WW3-grade는 DEAD)의 **실측 검증** — Guimarães 2020 스테레오
> 12레코드, 110창, 10 sea-event 클러스터.
> Script: `tmp/e46_real_supply.py` / JSON: `results/direction3/e46_real_supply.json`

## 0. 한 줄 요약

**buoy-grade spec은 실측 바다에서 생존을 입증하지 못했고, deliverable은 synthetic-scoped로
남는다 — 단 이유가 셋으로 갈라진다**: ① 타깃 regime(방향-분리 cross-Tp)이 데이터에
거의 없음(클린 1/12), ② pre-registered ANCHOR 실패 — 깨끗한 dense-유래 θ 공급조차
incumbent를 못 이김(fld orc−cur clustered −0.006 CI[−0.039,+0.027]); **regime이 실재하는
유일한 클린 레코드(BS11d)에서 supply는 오히려 −0.10~−0.12 해악**, ③ 합성에서 사전
입증된 supply-grade 한계 — dense patch 파티션은 어느 개구에서도 buoy급(5°/5%)에 못
미치고 WW3급(DEAD 등급)에 머묾. PRIMARY bar는 PILOT 검정력(클러스터 MDE 0.106 ≫
+0.03 bar)이라 형식적 KILL은 아니나(CI 상한 +0.126 > +0.02), **입증 실패 + regime 희소 +
구현 불가능한 supply 등급**이 결합해 실측 스코프 획득에 실패했다.

---

## 1. 전제조건 deliverable: 타깃 regime census + 스크린 검증

코디네이터의 스냅샷-FFT 스크린은 unimodal 컨트롤에서 **FPR 6/6 = 100%로 폐기**되었다
(180° 방향 접힘 + 마스크 경계 아티팩트 = e37 검출기/PERM과 동일 실패양식: "abstain하지
않는 검출기는 아무것도 재지 않는다"). E-46 스크린은 요구사항 3개를 전부 충족:

1. **시간축**: 70 s dense cube의 3D FFT + dispersion shell `ω²=g|k|tanh(|k|d)` 게이팅,
   **ω>0 반공간으로 180° 해소**(rfft 시간축, Hermitian 대칭, 방향=(−kx,−ky)).
2. **사이트별 수심**: BS 30 / AA 17 / YS 50 / LJ 45 m.
3. **컨트롤 선행 보고** (동일 파이프라인, 실측 투입 전):

| 컨트롤 | n | 결과 | 게이트 | 판정 |
|---|---|---|---|---|
| plane-wave micro(방향해소) | 1 | dir err 0.5°, Tp err 0.5% | — | OK |
| cross-Tp TPR (신선 si=5000–5005) | 6 | **det 5/6**, derr 12.4°, tperr 10.0% | ≥4/6, ≤20°, ≤12% | **PASS** |
| unimodal FPR (unimodal=True, 5100–5105) | 6 | **1/6 = 17%** | ≤ 1/6 | **PASS** |
| ★harness anchor(TRUE-θ oracle, A8) | 32 | **fld orcT−cur = +0.0481±0.0180** | ∈(0,+0.20) | **PASS** |

구조적 abstention: peak2는 smoothed hist의 **진짜 2D local max**(θ-wrap 8-이웃) + 배제
Δθ<55°(전 k) + soft-mass ≥0.12. 스크린 범위는 **방향-분리 cross-Tp**(아레나 regime,
d-sep∈[60°,150°])로 한정 — 같은 방향 swell+windsea는 범위 밖, λ₂≳box/1.5 swell쪽은
개구 실명(명시적 한계).

### 1.1 ★Census: 타깃 regime은 이 데이터셋에서 희소하다

| record | 판정 | supply(θ̂) | Tp 교차검증(spec/Welch) | 비고 |
|---|---|---|---|---|
| BS11a | unimodal | Tp2.6@−176° | 0.95 | wind sea |
| BS11b | unimodal | Tp2.4@−162° | 0.81 | |
| BS11c | unimodal | Tp2.9@−176° | 0.97 | |
| **BS11d** | **CROSS-Tp** | Tp2.8@−153°×0.84 + Tp3.3@124°×0.16 | 0.77 | **유일한 클린 타깃-regime** (d-sep 83°) |
| BS13a | CROSS-Tp | Tp1.8@91°×0.62 + Tp4.1@−143°×0.38 | **0.38 ⚠** | radial 판독 신뢰불가 |
| BS13b | CROSS-Tp | Tp2.9@−2°×0.73 + Tp3.7@−52°×0.27 | **0.76 ⚠** | 〃 |
| BS13c | CROSS-Tp | Tp2.9@−140°×0.80 + Tp4.0@167°×0.20 | **0.71 ⚠** | 〃 |
| AA14 | unimodal | Tp4.7@−28° | 0.97 | |
| AA15a | unimodal | Tp5.7@−28° | 1.01 | |
| AA15b | unimodal | Tp5.2@−80° | 0.96 | |
| YS17 | unimodal | Tp6.0@−122° | 1.02 | 최대 개구(3.8λ)도 단봉 |
| LJ18g | unimodal | Tp11.3@−28° | 0.80 | 폭풍 swell |

**발화 4/12, 클린(교차검증 통과) 1/12.** LOWO 창-수준 발화율은 census와 정합
(BS11d/BS13a/b 100%, BS13c 90%, BS11b 38%, 나머지 0%) — 판정이 안정적이다.
⚠3건(BS13 전부)은 주-lobe Tp가 Welch 대비 계통적으로 짧아 radial 캘리브레이션
신뢰불가로 표기. ⟹ **아크의 타깃 regime은 가용 실측에서 드물다** — 이것만으로도
실측 검증의 표본이 구조적으로 부족하다(코디네이터가 예고한 가능성 그대로).

---

## 2. 설계 (e44 미러, 실측 이식; pre-registered bars 동결)

- **Truth** = dense 스테레오 필드. **관측** = E-9 희소: valid-box 내접 ping-pong 대각,
  U~U(0.3,1.5)·cp_ref(cap 12 m/s), jitter r_fp=fp_ref/2, **400점/70 s**, NaN-aware
  bilinear(최근접 프레임), σ₀=0.05·Hs_hat 추가, solver 바닥 (3σ₀)²(SMULT=3).
- **공급 θ** = dispersion-gated (θ, ln k) 파티션의 **레코드 LOWO 평균**(채점창과
  realization 비공유). family 상수 γ=3.3, spread=30° **동결** = 검증 대상 misspecification.
  정직 공지: dense-θ는 "co-located/co-temporal" 의미의 상한일 뿐, **방향 정확도의
  상한이 아니다**(실제 계류 부이 20–30분 스펙트럼이 더 정확할 수 있음 — §3 A8b).
- **Arms**: `cur`(incumbent net) / `m2`(정직 all-marginal, **필수 게이트**) /
  `ens1`(1-lobe smooth baseline) / `orc`(LOWO θ̂) / `buoy`(θ̂ + d±5°·Tp±5%(양 lobe)·
  r2±0.05, R=5 평균 후 pairing).
- **Score**: enc10/enc05(궤적 연장 40점, deployment functional) + fld(corridor-밖
  ≤1200픽셀, 아레나 비교용).
- **통계**: t-CI 전용, naive + **sea-event 클러스터(10)** 이중 보고, 판정은 클러스터.
  결정론 시드(`hash()` 금지), 기존 스크립트 무수정.
- **Bars(동결)**: ANCHOR fld orc−cur>0 & <+0.30 / PASS enc10 buoy−cur ≥+0.03 &
  클러스터 CI 0 제외 & buoy−m2 CI 0 제외 / KILL CI 상한 <+0.02 / DNH(unimodal 창)
  ≥−0.01 / **MDE>0.03이면 PILOT 자동 강등**(e44 창당 sd 0.372로 사전 예고).

---

## 3. Amendment 로그 (전부 실측 스코어 확정 전; 근거는 기하·유효성·합성 컨트롤만)

| # | 내용 | 근거 |
|---|---|---|
| A1 | hist radial을 ω-dispersion 역산으로(buoy 원리) | BS13 patch 19 m ≈ 1λ — 공간 k축이 피크를 원리적으로 못 봄; census 플래그 발화(Tp 2.0 vs 3.3) |
| A2 | kp̂를 record 메타데이터로 clip [0.4,2.5]·kp_ref | two_stage_kp가 동일 바다에서 0.14→1.17 폭주(전 arm 공통) |
| A3 | 궤적을 marginal-valid box 내접으로 | YS17 valid 영역 offset sub-box → 전창 드롭 |
| A4/A8b | 1λ/2λ/3.8λ aperture-grade 진단 | BS/LJ~1λ, AA~1.5–2λ, YS~3.8λ 해석용 |
| A5 | off-shell 노이즈 바닥 차감(ω-bin median) | gated 셀 수 ∝k² → 노이즈 적분 고ω tilt |
| A6 | 파티션 v2: 배제 Δθ<55°(전 k)+mass≥0.12+local-max | unimodal 광폭 lobe가 realization lump로 41° 분열(질량 0.44/0.56) + ln k escape 루트 — 컨트롤이 재발화 |
| A7 | selftest = 4-세그먼트 LOWO 평균(실경로 미러) | 실측 supply는 레코드 평균(5–13창) |
| A8 | anchor 게이트 = TRUE-θ n=32 | n=6 추정-θ anchor 실패(−0.081)를 bisect: 아레나 네이티브 +0.097 재현·TRUE-θ+아레나U +0.056·보간/노이즈/쿼리 무죄·**창당 프라이즈는 궤적 추첨 sd~0.12** ⟹ 기존 anchor는 harness/detector/추첨 혼합 |
| A9 | u-hist를 ω-bin 수로 정규화(per-ω density=buoy/Welch 규약)+bank-밖 supply NaN | 1차 실측 런 중 census 플래그(BS13a 1.8 vs 4.8, 0.37)가 포착: du=2dω/ω → 광대역 고주파 tilt; BS13a 교란 ring이 bank 밖 → zero-grid 크래시 |

**정직 공지**: 첫 `--smoke`와 1차 본 런에서 BS13a·YS17 일부 창 점수가 노출되었다.
A1–A9의 어떤 결정도 "어느 arm이 이겼는가"에 의존하지 않는다(전부 기하/유효성/합성
컨트롤/교차검증 플래그 근거). Bars는 한 번도 움직이지 않았다.

### ★A8b — "dense patch = buoy급 supply"는 성립하지 않는다 (합성 사전 입증)

| box | det2 | derr | tperr |
|---|---|---|---|
| 1.0λ | 75% | 21.9° | 14.1% |
| 2.0λ (screen) | 83% | 12.4° | 10.0% |
| 3.8λ | 100% | 19.5° | 12.5% |

어느 개구에서도 buoy급(5°/5%)에 못 미치고 **아크의 WW3급(15°/10%, e43-44에서 DEAD
판정) 언저리**. 같은 합성 바다에서 grade penalty orc−orcT = **−0.120**(추정 supply가
TRUE-θ 프라이즈 +0.048을 전부 상쇄). ⟹ 실측 `orc`는 깨끗한 ceiling이 아니라
**detector-limited supply**이며, 실측 anchor 실패는 family misspecification과 supply
등급이 겹친 과잉결정이다.

---

## 4. 실측 결과 (n=110창, 10 클러스터)

### 4.1 Pre-registered 판정

| 항목 | 값 | 판정 |
|---|---|---|
| ANCHOR fld orc−cur (clustered) | **−0.0059 CI[−0.0390,+0.0272]** | **FAIL** |
| PRIMARY enc10 buoy−cur (clustered) | +0.0199 CI[−0.0857,+0.1255] | PASS 아님 |
| GATE enc10 buoy−m2 (clustered) | +0.0376 CI[−0.0628,+0.1380] | CI 0 포함 |
| KILL (CI 상한 <+0.02) | 상한 +0.1255 | KILL 아님 |
| 클러스터 MDE | **0.1056** ≫ 0.03 | **PILOT 강등** (사전 예고대로) |
| DNH (unimodal 창 buoy−cur) | +0.0732 (n=68) | PASS (≥−0.01) |

### 4.2 주요 대비 (naive / clustered, * = 클러스터 CI 0 제외)

| 대비 | fld | enc10 |
|---|---|---|
| orc−cur | −0.010±0.016 / **−0.006 [−0.039,+0.027]** | +0.010±0.026 / +0.027 [−0.076,+0.131] |
| buoy−cur | −0.009±0.016 / −0.005 [−0.035,+0.025] | +0.003±0.027 / +0.020 [−0.086,+0.126] |
| orc−m2 | +0.044±0.021 / **+0.042 [+0.020,+0.064]*** | +0.032±0.030 / +0.045 [−0.048,+0.138] |
| buoy−m2 | +0.045±0.021 / **+0.043 [+0.019,+0.067]*** | +0.026±0.030 / +0.038 [−0.063,+0.138] |
| m2−cur | −0.054±0.019 / **−0.048 [−0.082,−0.015]*** | −0.023±0.024 / −0.018 [−0.087,+0.051] |
| ens1−cur | −0.051±0.018 / **−0.042 [−0.078,−0.007]*** | −0.026±0.019 / −0.032 [−0.065,+0.002] |

읽기: ① **공급 θ는 정보를 담는다** — orc/buoy가 hand marginal(m2)을 fld에서 유의하게
이김(+0.042*/+0.043*). ② 그러나 **incumbent net이 실측 최강 baseline** — m2−cur
−0.048*, ens1−cur −0.042* (아레나에서 cur≈m2였던 것과 달리, 실측의 cur는 이 아레나-형
hand marginal을 유의하게 이긴다). ③ 결과적으로 supply는 net 대비 ≈0 — **net이 희소
관측만으로 이미 뽑아내는 regime 정보를 넘어서지 못함**.

### 4.3 ★결정적 세부: supply가 발화한 곳에서 정확히 해악

| 창 부분집합 | n | enc10 buoy−cur | fld buoy−cur |
|---|---|---|---|
| 1-lobe supply (unimodal 레코드) | 68 | **+0.073** | +0.008 |
| **2-lobe supply (BS11d+BS13)** | 42 | **−0.110** | −0.036 |

유일한 클린 타깃-regime 레코드 **BS11d(n=10)**: enc10 orc−cur **−0.104**, buoy−cur
**−0.116**, buoy−m2 −0.062 — cur 자체는 강함(enc10 +0.53). 플래그된 BS13a는 −0.29
(mis-ring된 Tp1.8 supply의 active harm — WW3-실패의 실측판 재현). ⟹ **cross-Tp
2-lobe frozen family는 실측 bimodal 바다에서 값을 만들지 못하고 해악을 만든다.**

### 4.4 탐색적 관찰 (post-hoc, 가설 생성 전용 — pre-registered 아님)

unimodal 5개 클러스터(AA14/AA15a/AA15b/YS17/LJ18g)에서 **1-lobe LOWO supply**의
enc10 buoy−cur 클러스터 평균 +0.130, t-CI [+0.045,+0.216] (n_cl=5). 외부 공급의
가치가 "cross-Tp 2-lobe 구조"가 아니라 **ring/방향 캘리브레이션**(two_stage kp̂·
beamform mu_hat보다 정확한 dense 추정)에서 나올 가능성. **사전등록 안 된 부분집합**
이므로 주장 아님 — 후속 실험 가설로만 기록.

---

## 5. Verdict — 질문에 대한 답

**"buoy-grade spec이 실측 바다에서 생존하는가?" → 생존을 입증하지 못했고,
deliverable은 synthetic JONSWAP 스코프에 남는다.** 구체적으로:

1. **Regime 희소**: 방향-분리 cross-Tp는 클린 기준 1/12 레코드 — 실측 검증의 표본
   자체가 구조적으로 부족(그 자체가 독립 발견: 아크의 타깃 regime은 이 해역
   데이터셋에서 드물다).
2. **ANCHOR FAIL**: 깨끗한 dense-θ 공급조차 incumbent를 못 이김(−0.006, CI 0 포함),
   그리고 regime이 실재하는 BS11d에서 supply가 **음(−0.10~−0.12)**.
3. **원인은 과잉결정**(정직하게 분리 불가): (a) frozen 2-lobe JONSWAP family의 실측
   misspecification (BS11d/BS13에서 발화-창 해악 −0.11), (b) supply 등급이 buoy급이
   아니라 WW3급(합성 사전 입증, grade penalty −0.120 — WW3는 아크에서 이미 DEAD),
   (c) 실측에서 incumbent가 아레나보다 상대적으로 강함(m2−cur −0.048*).
4. **형식적 KILL은 아님**: PRIMARY의 클러스터 CI 상한 +0.126 > +0.02, MDE 0.106 —
   pre-registration의 자동 PILOT 강등 조항 적용. "진짜 buoy(계류, 20–30분 스펙트럼,
   5°/5%)를 물리적으로 공급"하는 실험은 이 데이터셋으로 불가능하므로 그 축은
   **미검증으로 열려 있음** — 단 그 실험을 하려면 buoy 병설 레코드가 필요하다.

### 한계
- 스크린은 방향-분리 cross-Tp 전용(같은 방향 swell+windsea 실명, swell쪽 λ₂≳box/1.5
  개구 실명) — census의 "unimodal"은 "이 스크린 기준 unimodal".
- BS13 3건 radial ⚠ 미해결(고ω tilt 잔여 또는 실제 chop 지배 — 미분리).
- 창간 준독립 가정은 클러스터링으로 방어했으나 레코드 내 sea-state 공유는 남음.
- enc corr의 유효 dof ~5–10 (e44의 caveat 그대로).
- 실물 LiDAR는 여전히 미관측(스테레오 유래 truth) — 아크 전체의 상수 한계.

### 산출물
- `tmp/e46_real_supply.py` (pre-registration + amendment 로그 포함, 게이트 자동 중단)
- `results/direction3/e46_real_supply.json` (selftest·census·110창 원자료·대비·판정)
- 기존 스크립트/결과 무수정 (git status로 확인, 신규 파일만 추가)
