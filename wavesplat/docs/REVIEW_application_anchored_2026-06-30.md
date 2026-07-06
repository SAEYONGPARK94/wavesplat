# 전체 검토 — 원래 목적(하이드로포일 FF 제어)에 정렬한 파 추정 (2026-06-30)

> 계기: d72/d74(platform-aperture / 침로제어) 라인이 "파를 잘 보려고 배를 움직인다"로 가며
> **本末전도**(목적함수=추정품질, 결정변수=선박궤적)됨을 사용자(psy9412)가 지적.
> 원래 本: **계측된 파 → 하이드로포일 자세제어를 안전·효율적으로.** 이 문서는 그 목적에
> 정렬해 "선형·비선형 포함 현실 파를 어떻게 잘 추정할 것인가"를 전체 검토한다.

## 0. 원래 목적이 실제로 요구하는 것
`wavesplat/hydrofoil/` = wave-preview feedforward 제어 (1700 kg·22 kn 텐덤포일, 2-DOF
heave/pitch). FF는 **각 foil 앞 preview point의 η·∂η/∂x를 도달 몇 초 전**에 알아야 성립.
22 kn ≈ 11 m/s → preview 20–50 m 앞 = lead ~2–5 s. 따라서 풀어야 할 것은:

- **(A) Nowcast 정확도** — 현실 파(연속 스펙트럼 + 비선형) 위에서.
- **(B) 단기 예보** — ~2–5 s preview horizon.
- **(C) 실시간성 + UQ** — ms 추론 + 신뢰도(FF 게이팅용).

파 = 외란/입력, 선박 자세제어 = 목적. 추정의 품질은 *제어가 필요로 하는 만큼*이 기준.

## A. Nowcast — 현실 파 정확도 (HOS 검증)
| 방법 | HOS 연속스펙트럼 nRMSE | 추론 | UQ | 제어 deployable |
|---|---|---|---|---|
| RELAX (고전 이산원자, 32 atom) | 0.273 | 1054 ms | ✗ | 약함 |
| DPS (학습 확산 prior) | 0.246 (−10%) | 9.8 s | ✓ | ✗ (느림) |
| F1 (연속, 비학습) | 0.235 | 404 ms | ✗ | △ (UQ 없음) |
| **amortized posterior (d56)** | **0.219** | **8.3 ms** | **✓ (보정됨)** | **✓** |

- **이산-성분 벤치(nRMSE 0.10–0.16)는 현실을 과대평가.** 그 벤치는 15–30개 *이산* 성분이라
  이산원자 추정기(RELAX)에 구조적으로 유리. *연속* 스펙트럼(HOS, JONSWAP 64² 모드)에선
  학습 연속표현이 이긴다 (`results/direction3/hos_nonlinear_VERDICT.md`).
- 제어용으론 **amortized posterior만이 (속도+단기윈도 정확도+보정 UQ) 셋을 동시에** 만족.
  → deployable nowcast의 정답.

## B. Forecast — preview (HOS 검증)
- **전장(full-field) GP space-time 예보**: nRMSE ~0.09를 0.5–6 s에서 평탄 유지, persistence
  대비 skill 0.5–0.8, **유효 horizon ~6 s** (`results/spacetime_gp_forecast.json`). 제어가
  필요한 2–5 s lead를 충분히 커버.
- **이산성분 전파식 예보는 실패** (nRMSE 0.7–0.9): HOS에선 성분 추정 후 propagate가 안 됨.
  → 예보는 *전장 기반*만 유효.
- 비선형: **2차 bound-wave(Stokes-2) 보정이 5 s 예보 gap의 86–90%를 메움**(60 s에선 7–17%로
  감소) (`results/hos_prediction_gap.json`, `results/stokes2_gap.json`).

## C. 선형 vs 비선형 — 정리
1. **Nowcast 재구성: 비선형 gap ≈ 0.** RELAX가 비선형(M=3)을 선형(M=1)만큼 복원 (O(ε²)≈0.25%).
   진짜 gap은 비선형이 아니라 **연속 스펙트럼 표현력**.
2. **Forecast: 비선형이 중요.** 짧은 horizon에서 2차 bound-wave 보정이 실질 이득.
→ *재구성 = 연속표현이 관건, 예보 = 2차 비선형 보정이 관건.*

## D. amortized posterior — 무엇이고 왜 정답인가 (`scripts/d56_amortized_posterior.py`)
**구조 (PosteriorNet, conditional neural field):**
- 관측 인코더: 각 LiDAR 관측 (x,y,t,z) → 스케일정규화 → MLP(d=160) → TransformerEncoder(4층,
  4헤드) → 관측집합 context (관측 순서에 불변한 set encoding).
- 쿼리 인코더: 임의 질의점 (x,y,t) → MLP(d=160).
- 디코더: TransformerDecoder(3층)가 쿼리→관측 context를 cross-attend → 질의별 hidden.
- 헤드: → (mean η, log_var). **관측집합 + 질의점 → (평균, 분산)을 단 1회 forward로.**

**학습 (Gaussian NLL, heteroscedastic):**
`nll = 0.5*((mean−z)²/exp(logv) + logv)`, Hs 정규화. mean항=정확도(정보한계로), var항=보정된
불확실성(어디서 더 불확실한지 σ를 학습). 합성 파윈도(gen_batch) 대량으로 *오프라인 1회* 학습.

**"amortized"의 핵심:** RELAX(윈도마다 성분 최적화)·DPS(윈도마다 확산 샘플링)는 배포 시점에
*매번* 최적화/샘플링을 풀어 느림. amortized는 추론비용을 **학습 때 가중치에 미리 상환** →
배포 시 추론 = forward 1회. = amortized posterior inference.

**왜 제어 정답인가:**
- 8.3 ms (F1 50×, DPS 1180× 빠름) → 제어 루프 안에서 돈다.
- 질의별 보정된 σ 제공 → FF를 σ로 게이팅/블렌딩 가능 (믿을 때만 강하게). RELAX/F1은 점추정뿐.
- 짧은 윈도(≤200 obs, ~2.2·Tp) 친화 → 저지연 + 정상성 안전.
- UQ 보정: 온도 s≈1.07 1파라미터로 ±1σ 커버리지 0.68(이상 .68)·±2σ 0.95(이상 .95);
  corr(logσ, log|err|)=0.40 → 예측 σ가 실제 오차의 공간분포를 추적.

## E. 검증 안 된 것 (정직)
- amortized 정확도 0.219는 **HOS 6 window(3조건×2seed)뿐** — F1(0.235) 대비 순위는 노이즈 가능,
  robust한 건 **속도·UQ**. seed 확대 필요.
- **정보한계를 깨지 않음** — 빠르게·UQ와 함께 *도달*할 뿐 (전 thesis와 일관).
- **실측/수조 데이터 0** (보유 110 ep 전부 시뮬·정지센서; ship/platform/pose 데이터셋 없음).
  SNU 예인수조 plan(`docs/snu_towing_tank_experiment_plan.md`) 미실행.
- **end-to-end 미연결**: 추정→FF→실제 ride 개선은 hydrofoil sim에서만. HOS+실측 사슬 미연결.
- ∂η/∂x(slope): 현재 헤드는 η만 출력. preview slope는 질의 x에 대한 autodiff 또는 이웃질의
  차분으로 얻어야 함 (확장 항목).

## F. 수렴 권고 — 단일 라인 (이상한 데로 분기 금지)
> **★ PhD 프레이밍 (2026-07-01 사용자 정정): 추정과 제어를 분리한다.** thesis의 *증명가능·
> 이식가능한* 기여는 **파 추정기**다. 제어를 핵심에 두면 *하드웨어 제어 입증*이 필요해지고
> (sim-only 제어는 박사논문감이 아님), 그건 비현실적. 따라서 하이드로포일 FF는 *"추정기를
> 여러 곳에 붙여 쓸 수 있고, 그 중 하나로 적용해봤다"*는 **(sim) 응용 시연**일 뿐, 증명 부담을
> 지지 않는다. 수조 실험도 *제어가 아니라 추정기*의 sim-to-real 검증(실측 파, wave-probe GT)이다.

**헤드라인(thesis core): "배포가능한 실시간 phase-resolved 파 추정기 (amortized neural
posterior) — 현실 비선형·연속 스펙트럼에서 정확 + ms + 보정 UQ."** 3기둥:
1. **Nowcast** = amortized neural posterior (HOS 연속, ms, 보정 UQ). seed 늘려 0.219 robust화.
2. **Forecast** = 전장 GP/학습 예보 + 2차 bound-wave 보정 (추정기의 *시공간 예측 능력*으로 독립
   정당화; preview-제어가 이를 소비하는 한 응용일 뿐).
3. **검증** = HOS에서 헤드라인 재현(상시지시) → SNU 수조 1세트로 **추정기**의 sim-to-real.

**본선에서 분리:** ① d74 침로제어(별도 USV 매핑 문제), 격자 prior, buoy. ② **제어(하이드로포일
FF)도 증명 핵심에서 분리** — sim 응용 시연으로만. d72는 "운용 운동이 관측성을 돕는다"는 *종속*
해석으로만 유지. CRB(d1)는 'preview 정확도·예보 horizon의 근본한계'로 종속.

## G. 실행 로드맵 & 교수님 승인 Go-bar
> ★ 정정(2026-07-01): 수조 실험은 **추정기**의 sim-to-real 검증이다(제어 하드웨어 X).
승인 논리: **"sim에서 추정기가 현실 비선형 해상에서 정량적으로 검증됐고(정확도+보정 UQ), sim이
못 푸는 단 하나(실측 파에 대한 sim-to-real gap)를 수조가 해소한다"** 를 보이면 승인을 받는다.
제어이득(end-to-end)은 *증명 핵심이 아니라* 추정 완성 후의 (sim) 응용 시연으로 곁들인다.

### G.1 검증 (Validation)
- **V1 (최우선)** amortized nowcast 통계 robust화: HOS 6 → **≥30 window**(BF3/4/5 × ≥5 seed).
  N=6이 현재 최약점.
- **V2** 헤드라인 HOS 재현(상시지시): nowcast·forecast 전부 연속 비선형 HOS에서. 이산벤치 금지.
- **V3** HOS forecast: 예보 skill·유효 horizon을 HOS + 2차보정에서 측정(현 skill 수치는 대부분
  이산/선형 ep).
- **V4 (가장 설득력)** end-to-end: amortized 추정 → FF → ride 지표(heave/pitch RMS). "추정이
  좋아지면 ride가 좋아지나"의 연결이 현재 비어 있음.

### G.2 작업 (Build)
- **W1** slope ∂η/∂x 출력: FF가 쓰는 건 slope인데 헤드는 η만 → autodiff/이웃차분 + slope 정확도 검증.
- **W2** amortized를 hydrofoil FF 루프에 결합(현 WaveSplat-FF와 비교) + **σ-게이팅**(UQ→제어; novel+안전).
- **W3** 단기예보를 FF preview로 주입(foil은 공간·시간상 앞 → 현 FF는 순간장 평가).
- **W4** 2차 bound-wave 보정 모듈(예보용).

### G.3 정량화 (Quantify)
- **Q1** nRMSE(mean±SE) 해상태별, **η와 slope 둘 다.**
- **Q2** 지연: 추론 ms + 제어루프 latency budget.
- **Q3** UQ: 커버리지(±1σ/±2σ), corr(logσ,|err|), reliability diagram.
- **Q4** 예보: skill vs persistence vs horizon, 유효 horizon(skill>0), nRMSE-horizon.
- **Q5 (헤드라인)** 제어이득: heave/pitch RMS(또는 RMS 가속도/foil AoA 변동) **%감소** —
  vs (a) FF 없음, (b) 단일점 센서 FF, (c) 완벽 preview oracle(상한).

### G.4 통계 (Statistics)
- **S1** 표본: ≥5 seed × 3 해상태 × 다수 window → mean±SE·유의성 가능(현재 총 N=6).
- **S2** 유의성: amortized vs F1(0.219 vs 0.235는 "노이즈 가능") → matched window **paired test**
  (paired-t/Wilcoxon). 유의하면 정확도 주장, 아니면 정직하게 "동급 + 유일 ms + 보정 UQ".
- **S3** 부트스트랩 CI: nRMSE·skill·제어이득%.
- **S4** 보정 통계: coverage CI, reliability diagram, (가능하면) CRPS(proper score).
- **S5** ablation+통계: σ-게이팅 on/off, 예보 on/off, slope source — 각 effect size+CI.

### G.5 승인 Go-bar (추정 중심 — 이 정도면 설명·승인 가능)
1. **robust HOS nowcast(핵심)**: HOS N≥30, nRMSE mean±SE — baseline 대비 유의 우위 *또는*
   동급 + 유일 ms + 보정 UQ. 커버리지 .68/.95 + reliability/CRPS로 보정 입증.
2. **선형 vs 비선형 근거(V2)**: "이산벤치에선 고전기법이 이기지만, 현실 비선형·연속 스펙트럼으론
   학습 연속표현 posterior가 이긴다" — *왜 이렇게 설계했는지*의 과학적 정당화. (연구 비교에서만
   비선형 사용)
3. **forecast 능력 정량화(V3)**: HOS+2차보정에서 skill>0(persistence 대비), 유효 horizon, CI.
   — 추정기의 시공간 예측 능력으로서.
4. **수조 프로토콜**: **추정기**를 실측 파로 검증(LiDAR + wave-probe GT). 해상태·계측·성공기준·
   "수조가 해소하는 단 하나 = 추정기의 sim2real gap"·비용/일정 (`docs/snu_towing_tank_experiment_plan.md`).

*제어이득(Q5/V4)은 승인 핵심이 아님 — 추정 완성 후 sim 응용 시연으로 보강.*

### G.6 Critical path (순서)
① V1 + S1/S2 + W1 (싸고 최약점 제거) → ② V2 (선형vs비선형 근거) + V3/Q4 (HOS forecast) +
S3/S4 (CI·보정 통계) → ③ 수조 프로토콜 패키지 → ④ (추정 완성 후) V4/W2/W3/Q5 제어 응용 시연.
**승인 타이밍 = ②까지의 robust 추정 결과 + 선형/비선형 근거 + 수조 프로토콜이 갖춰지는 순간.**
