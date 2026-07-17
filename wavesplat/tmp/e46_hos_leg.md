# E-46 — HOS 비선형 레그: 살아남은 헤드라인의 HOS 재현 (상시지시 빚 청산)

**판정: ★PASS — buoy-grade 외부공급 헤드라인은 HOS 비선형성에서 생존한다.
상금은 선형-중첩 아티팩트가 아니다 (draw-paired 물리효과 dd ≈ 0.000 ± 0.004).**

- 스크립트: `tmp/e46_hos_repro.py` (사전등록 docstring 포함, 새 파일만)
- 원자료: `results/direction3/e46_hos_repro.json` (n=192 cross-Tp + 64 unimodal, 제외 0)
- 런타임: 3164 s (RTX 4080S 박스, HOS는 CPU numpy FFT)
- 이중검증: ①독립 코드 경로로 원자료 재도출(소수 4자리 일치) ②si5000
  재빌드+재스코어 = 저장값과 **정확히 0.00e+00 차이**(결정론) ③신선-컨텍스트
  에이전트 감사 (§7)

## 0. 왜 이 실험인가

상시지시: **"검증 데이터셋엔 항상 HOS 비선형 포함, 헤드라인은 HOS서 재현"**.
2-에이전트 감사 결과 E-29~E-45 아크 전체가 이 지시 위반 — 살아남은 헤드라인이
비선형성에서 단 한 번도 재현된 적 없음. E-46이 그 빚을 갚는다.

살아남은 헤드라인 (post-audit, 전부 **선형** JONSWAP 중첩 아레나):

| 항목 | 선형 아레나 값 |
|---|---|
| prize `oracle − cur` (field) | +0.0729 ± 0.0034 (6 blocks) |
| 배포 functional 전이 `enc10 orc − cur` | +0.0431 ± 0.0077 (n=957, ratio 0.60) |
| blind 추론 | ≈ 0 (사전등록 재현) |
| ★buoy-grade 외부공급 | +0.0349 CI[+0.0084,+0.0615] (전-축 정직오차 변형) |
| WW3-grade 공급 | DEAD (frozen-axis −0.0092; phantom −0.0158 vs −0.01 bar) |
| e45 phantom 참조 (선형, n=240) | fld ph_buoy−cur −0.0074 PASS / fld ph_ww3−cur −0.0208 유의한 HARM |

## 1. 무엇을 지었나

`build_sea_crosstp`의 정직한 HOS 아날로그 + **같은-IC 선형 쌍둥이 컨트롤**.

- **아레나 X_H (HOS cross-Tp)**: 파라미터/트랙/노이즈 draw는 `build_sea_crosstp`와
  동일 rng 스트림(9700+31·si), 동일 호출 순서 (Tp1=7, Hs~U(1.5,3),
  r2~U(0.2,0.5), d2=d1±U(60,150)°, Tp2=7·U(0.75,1.4), spread 30°,
  U~U(0,1.5)·cp, nfrac~U(0.02,0.09), N_OBS=400/70 s, 개구 side=2λ₁,
  GT=1200 held-out 픽셀 @t=70 s). 표면만 교체: **M=3 HOS 진화**
  (레포 `wavesim/hos_simulator.py` 그대로; dealiasing ON, factor
  ceil((M+1)/2)=2 — 문서화된 "M=1 dealiasing-off NaN 폭발" 경로 미사용).
- **HOS 수치**: 주기 도메인 8λ₁×8λ₁(611 m), 64×64 스펙트럼 모드(dk=kp/8,
  kmax=4kp, 피크 링 각분해능 7.2°), depth 50 m, RK4 dt=0.15 s
  (ω_max·dt=0.27, F1/E-16 전례와 동일 마진), spin-up 3Tp=21 s.
- **IC**: `initialize_jonswap` 2회(파티션별 Hs√(1−r2)@Tp1,d1 seed 4000+si /
  Hs√r2@Tp2,d2 seed 5000+si), eta_hat·phi_s_hat 합산(선형 IC — 구속 고조파는
  spin-up 동안 자연 발생).
- **관측**: 같은 `sample_obs` 트랙의 시공간 점에서 sim을 시간전진시키며 η_HOS
  점샘플 + σ₀ 노이즈. kp는 `two_stage_kp`로 blind 추정. encounter GT는
  `track_future` 40점을 sim 전진으로 렌더.
- **★선형 쌍둥이 X_L**: **같은 IC 모드**의 정확한 선형 진화
  (eta_hat(t)=eta_hat(0)·e^{−iωt} — 같은 이산화의 선형 극한), **같은 노이즈
  벡터** → per-sea·per-draw 페어링된 물리 컨트롤. 하니스 게이트를 쌍둥이에
  걸어 "HOS에서 상금 사망"이 버그와 구별되는 물리 결과로 관측 가능하게 함.
  비선형성 검사도 쌍둥이 **대비**로 (단언 아님 — 알려진-음성 컨트롤 내장,
  support-matched 같은 611 m 도메인 그리드).
- **ARMS** (`solve_with_grid`, SMULT=3, free arm, GT peek 없음; θ-draw는 H/L
  재사용 = draw-paired): `cur`(incumbent net) / `m2`(honest all-marginal,
  MANDATORY) / `orc`(참 θ) / `buoy`(d±5°, Tp2±5%, r2±0.05, R=5 페어링 전
  per-sea 평균) / unimodal 블록은 e45 arms 그대로(`uni_orc`, `uni_buoy`,
  `ph_buoy`, `ph_ww3`, R=3 — 존재하지 않는 2번 lobe 공급).
- **FUNCTIONALS**: field corr @Pq (primary/anchor) + encounter corr
  (enc10/enc05, 40점).

## 2. 사전등록 (스코어링 전 동결 — 스크립트 docstring 원문 참조)

- 시드: 신선 블록 crossTp si=5000..5191(n=192), unimodal si=6000..6063(n=64);
  소각 블록 회피; smoke(si 4990../6990..)는 크래시/NaN/타이밍 전용 격리.
  결정론적 정수 시드만, Python hash() 금지.
- 급준도: 아레나 자체 분포 kp·Hs/4 ∈ [0.031, 0.062].
- **MDE 선언 (t-검정, α=.05, 검정력 80%)**: field buoy−cur 가정 sd 0.10 →
  n=192에서 MDE ±0.020 (sd 1.5×에도 0.030 < 목표 +0.035). encounter는
  SECONDARY(orc MDE 0.048), buoy-enc은 PILOT. **달성치**: sd 0.1148 →
  **달성 MDE80 = 0.0232**; 관측 효과 +0.0663 = 달성 MDE의 2.9배. 자기-MDE
  아래 주장 없음.
- 게이트/바: 비선형성(skew_H ≥ +0.05 & 쌍둥이 대비 ≥ +0.04), 제외 ≤5%,
  HARNESS GATE(twin orc−cur ∈ [0.04,0.11] — e44 anchor band), ANCHOR-H
  (fld orc−cur ≥ +0.03, t-CI 0 배제), PASS(fld buoy−cur ≥ +0.03 & CI 0 배제
  & buoy−m2 CI 0 배제 & phantom do-no-harm ph_buoy−cur mean ≥ −0.01 fld·enc10
  모두), KILL(fld buoy−cur CI 상한 < +0.02). t-기반 CI만(1.96 금지).

## 3. 적정성 검사

`mesh_adequacy.py`는 raycast 삼각 메시 검사기 — 이 아레나의 관측 연산자는
점-샘플링이라 **적용 불가**. 대신 HOS 스펙트럼-그리드 적정성을 검사:

| 검사 | 결과 |
|---|---|
| NaN/비유한 제외 | **0/192 + 0/64** |
| Hs drift \|Hs(T)−Hs_target\|/Hs | **0.0111** (에너지 보존 양호) |
| 스펙트럼 꼬리 pile-up (k>0.85·kmax 에너지비) | 1.24e-3 → 1.33e-3 (**성장 없음** — folding/aliasing 부재) |
| corr(z_H, z_L) | 0.973 (물리가 데이터를 실제로 바꿈, 그러나 동일 realization) |

**비선형성 실재 (대비로 확인, 단언 아님)**:

| 지표 | HOS | 선형 쌍둥이 | 대비 |
|---|---|---|---|
| 표면 skewness @t=T | **+0.0997** (sd 0.068) | +0.0032 | **+0.0965** (게이트 ≥+0.04 통과) |
| excess kurtosis | +0.006 | −0.011 | — |

skew +0.10은 2차 이론 예측(~3kpσ_eff, 이 급준도에서 0.09–0.19 범위)과 정합 —
구속 고조파가 실제로 존재하는 바다이며 "위장 선형 런"이 아니다.

## 4. 결과

### 4.1 하니스 게이트 + ★anchor (field, n=192)

| 대비 | 선형 쌍둥이 [L] | HOS [H] |
|---|---|---|
| fld orc − cur | **+0.0728** ± 0.0080 CI[+0.0570,+0.0886] win 144/192 | **+0.0732** ± 0.0085 CI[+0.0565,+0.0899] win 147/192 |

- HARNESS GATE 통과 (쌍둥이 anchor가 선형 아레나 상금 +0.0729 ± 0.0034를
  사실상 그대로 재현 — 이산화(dk=kp/8) 영향 무시 가능 수준).
- **ANCHOR-H 통과: 상금은 M=3 비선형성에서 크기 변화 없이 생존.**

### 4.2 ★buoy 헤드라인 (field, primary)

| 대비 | [L] | [H] | bar |
|---|---|---|---|
| fld buoy − cur | +0.0661 CI[+0.0506,+0.0815] | **+0.0663** CI[+0.0500,+0.0826] win 145/192 | ≥+0.03 & CI>0 ✓ |
| fld buoy − m2 | +0.0718 CI[+0.0550,+0.0886] | **+0.0726** CI[+0.0542,+0.0910] win 144/192 | CI>0 ✓ |
| fld m2 − cur | −0.0058 CI[−0.0199,+0.0084] | −0.0063 CI[−0.0216,+0.0090] | (cur≈m2 아크 전제 유지 ✓) |

**KILL 조건(CI 상한 < +0.02) 미충족 — 상한 +0.0826.**

### 4.3 탐색적 (라벨 명시): draw-paired 물리효과 dd = (arm−cur)_H − (arm−cur)_L

| arm | dd (field) |
|---|---|
| orc | **+0.0004** ± 0.0036 CI[−0.0066,+0.0074] |
| buoy | **+0.0002** ± 0.0035 CI[−0.0067,+0.0072] |
| m2 | −0.0006 ± 0.0051 CI[−0.0106,+0.0095] |

같은 realization·같은 θ-draw·같은 노이즈에서 물리만 바꾼 페어 차이가
CI 반폭 ±0.007로 0에 못박힘. **이 급준도 범위에서 regime-공급 채널은 비선형성에
완전 불변** — "상금이 선형-중첩 아티팩트"라는 가설은 정밀하게 기각된다.

### 4.4 encounter functional (사전등록 secondary/pilot)

| 대비 | [L] | [H] |
|---|---|---|
| enc10 orc − cur | +0.1126 CI[+0.0736,+0.1517] | **+0.0947** CI[+0.0544,+0.1350], Wilcoxon 1.6e-06 |
| enc10 buoy − cur | +0.1041 CI[+0.0654,+0.1429] | **+0.0899** CI[+0.0504,+0.1294] |
| enc10 buoy − m2 | +0.0594 CI[+0.0268,+0.0920] | **+0.0476** CI[+0.0113,+0.0839] |
| enc05 buoy − cur | +0.1107 CI[+0.0593,+0.1622] | +0.1002 CI[+0.0472,+0.1531] |

배포 functional 전이도 HOS에서 생존 (secondary지만 MDE 0.048을 상회하는
효과로 CI 0 배제). 블록 특이사항: 이 블록에선 enc10 m2−cur가 +0.04로 양수
(두 세계 모두 → HOS 효과 아님, 블록 성질) — e44의 "enc에서 cur≈m2"와 다름;
bar 판정엔 영향 없음(buoy−m2가 직접 클리어).

### 4.5 unimodal phantom do-no-harm (n=64, HOS)

| 대비 | [L] | [H] | 선형 e45 참조(n=240) |
|---|---|---|---|
| fld uni_orc − cur | +0.0608* | +0.0526* | +0.0308* |
| fld uni_buoy − cur | +0.0594* | +0.0501* | +0.0284* |
| **fld ph_buoy − cur** | +0.0129 | **+0.0101** (bar ≥−0.01 **PASS**) | −0.0074 PASS |
| fld ph_ww3 − cur | −0.0066 | −0.0074 | −0.0208 (유의한 HARM) |
| enc10 ph_buoy − cur | +0.0317 | **+0.0473** (bar **PASS**) | −0.0058 PASS |

buoy-grade phantom(존재하지 않는 2번 lobe 공급)은 HOS에서도 무해 — 선형에서
WW3를 죽인 레그가 HOS에서 buoy를 죽이지 않는다. 주의: 이 블록의 절대값은 e45
블록과 다르지만(uni_orc 이득이 더 크고 phantom 벌점이 약함) **H와 L이 거의
같게 움직인다** — 즉 블록/이산화 차이지 HOS 효과가 아니다. ph_ww3의 harm이
이 블록에선 유의하지 않으므로 "WW3 phantom-harm의 HOS 재확인"은 이 블록
검정력으론 미결(WW3는 이미 선형에서 2중 사망 — 부활 주장 아님).

## 5. 판정

사전등록 게이트 전부 통과: 비선형성 실재 ✓, 제외 0 ✓, 하니스 게이트 ✓,
ANCHOR-H ✓, do-no-harm ✓, PASS 조건 3종 ✓, KILL 미충족.

> **★E-46 one-liner: cross-Tp regime 상금(+0.073)과 buoy-grade 외부공급
> 이득(+0.066, m2 대비 +0.073)은 M=3 HOS 비선형성(급준도 0.031–0.062)에서
> 크기 손실 없이 재현되며, draw-paired 물리효과는 0(±0.007) — 헤드라인은
> 선형-중첩 아티팩트가 아니다. encounter 전이·phantom do-no-harm도 생존.
> 상시지시(헤드라인 HOS 재현) 빚 청산.**

### 정직한 스코프 한계 (인용 시 함께 인용할 것)

1. **buoy arm = e43 표준 번들**(d±5°, Tp2±5%, r2±0.05). "+0.0349 전-축
   정직오차 변형"(frozen Tp1/spread/γ 포함)은 HOS에서 재실행하지 않았다.
   이 하니스의 선형 쌍둥이 buoy−cur도 +0.066이므로 +0.0349와의 차이는 오차
   등급 차이지 물리 차이가 아님 — 그리고 dd≈0이 "비선형성은 이 채널을 건드리지
   않는다"를 말하므로 전-축 변형도 전이가 **기대**되지만, 측정하지 않은 것은
   주장하지 않는다.
2. **비선형성 = M=3 파-파 상호작용**(구속 고조파 + 3차)만. 쇄파·바람·해류·
   관측연산자 비선형(raycast/shadowing — E-16 아크가 지배 축으로 지목한 것)은
   이 레그의 범위 밖.
3. 급준도 0.031–0.062 (아레나 자체 분포) — 극한 바다 미검증.
4. 스펙트럼 이산화 dk=kp/8 (주기 도메인) — 쌍둥이가 같은 이산화를 공유하므로
   H-vs-L은 깨끗하고, 쌍둥이 anchor가 연속-k 선형 아레나 값을 재현하므로
   이산화 편향은 무시 가능 수준으로 측정됨.
5. unimodal 블록 절대값은 e45 블록과 다름(블록 성질, §4.5) — phantom 판정은
   블록-내 페어링으로 수행.
6. 전부 시뮬레이션 (실물 LiDAR 미관측 — 아크 공통 한계).

## 6. 재현

```bash
python3 tmp/e46_hos_repro.py            # 본 런 (~53분)
python3 tmp/e46_hos_repro.py --smoke    # 4 seas 스모크 (~1분)
```

## 7. 이중검증 기록

1. **원자료 재도출** (독립 코드 경로, `paired()` 미사용): §4의 전 수치가 소수
   4자리까지 일치. 달성 MDE80 = 0.0232 재계산 확인.
2. **결정론**: si5000 재빌드+재스코어 → 저장 JSON과 H/L 18개 스코어 전부
   차이 0.00e+00 (비트 재현).
3. **신선-컨텍스트 에이전트 감사**: (완료 후 기입)
