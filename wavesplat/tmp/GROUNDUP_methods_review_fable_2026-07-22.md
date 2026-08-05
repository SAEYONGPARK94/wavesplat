# Ground-up 학습방법·네트워크 지형 리뷰 (fable, 2026-07-22)

> coordinator 분담: Opus=kp-robust B-2 구체설계, fable(본 문서)=**더 넓은 질문** — "이 프로젝트(희소 이동
> 표고 관측→위상분해 파 추정)에 정말 쓸만한 다른 네트워크/학습법이 있나?"를 ground-up으로. 실험 미실행 —
> 순수 문헌·메모리 리뷰. 근거 = memory index 전체 + 원 topic 파일 20개 직접 열람(직접 인용, 요약 아님)
> + 오늘자 `tmp/EXECUTION_SYNTHESIS_2026-07-22.md` + WebSearch 4회(신규 후보 novelty 1차 스캔).
> 규율: MEASURED(이미 실행·검증된 프로젝트 결과) / ARGUED(본 문서의 종합·추론) 분리, 이미 폐기된 것 재제안 금지.

---

## 0. 왜 이 리뷰가 가능한가 — 이 프로젝트는 이미 ~40개 학습 아키텍처를 시도했다
`direction3_gridfree_prior.md` 한 파일에만 diffusion/DPS/amortized-posterior/active-sensing/synthetic-aperture
학습 계열 실험이 **50개 이상** 기록돼 있다(2026-06-27~30, 3일 자율세션). 여기에 `project_setdiff_component`,
`project_beamform_offgrid`, `project_tnse_status`, `project_selfsup_novelty`, `project_gpu_day`,
`project_h1h2_learning_closure`, `project_door1_single_channel`, `project_e28_gennet` 등을 더하면 이 코드베이스는
아마 개인 연구 프로젝트치고 **학습 아키텍처 다양성이 이례적으로 넓다** — raw-좌표 회귀부터 diffusion·SBI에
근접한 것·active sensing·강화학습형 궤적정책까지 다 있다. **본 리뷰의 1차 가치는 새 아이디어 나열이 아니라
"뭘 다시 시도하면 시간 낭비인가"의 정확한 지도다.**

---

## 1. 시도-이력 지도

### 1.1 SUCCESS (현재도 유효하거나 명확한 승리로 종결)

| 방법 | 요지 | 이긴 지점 | 근거 |
|---|---|---|---|
| **RELAX (greedy off-grid matching pursuit)** | 성분 순차 검출+refine, 학습 無 | 밀집 관측·well-determined 레짐서 CRB 근접(5comp 3배 이내) | `direction1_crb` |
| **net-MMSE / amortized empirical-Bayes γ(k) prior (현 production, B-2 계열)** | net은 정규화 k-grid 위 **prior gain만** 예측, 실제 추론은 닫힌형 Woodbury MMSE | 실측 8레코드 +34~36%, 이동(synthetic aperture) 레짐 net-MMSE 최초 학습 실측 양성전이(BS11a/AA14 net>hand) | `project_synthetic_aperture` 07-05 대반전, `CLAUDE.md` 현재 파이프라인 |
| **beamform-power-map + CNN 검출 + 잔차 반복** (E-2X 돌파) | 입력을 raw점→beamform k-공간 파워맵으로 바꾼 것 자체가 돌파 | 단순해서 RELAX급 도달(BF3_5c 0.076≈0.075) | `project_setdiff_component` "★★돌파" |
| **set-diffusion 성분 prior (연속 atom DDPM + grid-free GP-marginal guidance)** | 성분(atom) 공간에서 생성, 물리 likelihood로 guide | ⚠**부분 정정 필요(아래 참조)** — 초기 0.208 vs 0.258~0.261은 **최종 검증 전 수치**, 보정 UQ(+0.28~0.40)만 확정 생존 | `project_setdiff_component`, `direction3_gridfree_prior` |
| **빠른 amortized 신경 posterior (obs→query별 (mean,σ), 직접 필드 회귀, d56~d61)** | set-diffusion과 **다른 아키텍처**(atom-DDPM 아님) — 위 표 성공 레시피(저차원·물리정규화·닫힌형 solve)와 살짝 다른 예외: 필드를 직접 회귀하나 **Gaussian NLL이 목표라 point-estimate 강제 없음**(MSE 아님) | **F1(비학습 최선 baseline) 0.234·RELAX 0.274를 유의하게 격파**(0.222±0.024, paired Δ+0.013±0.004 SE, p<0.01, 11/12윈도우) + **8.3ms**(F1 대비 50배·DPS 대비 1180배) + 보정 UQ(s=1.02) — **이 프로젝트에서 "학습이 최선 비학습 baseline을 정확도로도 유의하게 이긴" 유일하게 명확한 사례** | `direction3_gridfree_prior` d56/d57 |
| **학습된 비근시 궤적 정책 (CEM 모방, platform-motion 합성개구)** | 배 침로 제어로 CRB 최소화, 정책이 CEM 근접·greedy 격파 | **유일하게 "새 정보 자체를 만든" 학습**(플랫폼 운동 cross-track ×2.5~4.4 방향CRB 개선) 위에서 정책이 greedy 이김, 7.8ms | `direction3_gridfree_prior` d72~d74 |
| **적응 대역폭(대역 하한) 선택** (무학습이지만 "학습 레버"로 오인되기 쉬움) | 관측된 스펙트럼에 맞춰 bank 하한 적응 | 합성·HOS·실측 3중 검증(적대적 바다 3.9→0.27) | `project_gpu_day` |
| **비학습 배포 레시피 (고정 raydrop 문턱 + range-stratified 샘플링 + 고정 잡음바닥 하이퍼파라미터)** | **학습 0** — 이게 SUCCESS 표에 있는 이유 자체가 진단(§2.2 아래 참조) | 조우예보 corr **0.10→0.90~0.99**(8/8 바다), <100ms, 학습 없이. "학습이 필요해 보였던 문제의 상당수는 검증 프로토콜 버그였다"는 이 프로젝트 최대 메타교훈의 근거 | `project_h1h2_learning_closure` (H4/H5) |

**★set-diffusion "RELAX 전승" 수치 정정 (제가 원문 direction3_gridfree_prior.md을 직접 재확인, 위 표의 ⚠ 해설)**:
"DPS 0.208 vs RELAX 0.258~0.261"(d43 근방)은 **RELAX가 관측-기아 상태였을 때의 초기 비교**였다. 같은 파일의
**후속·더 엄격한 검증(d51/d52, "근본적 해결 F1-F4 검토")**이 이를 뒤집는다: 비학습 dense-grid+ridge(**F1**)가
**0.233**, 학습 DPS가 **0.234** — **동률(tie)**, F4(neural operator)는 0.250(더 나쁨). 결론(원문 그대로):
**"~0.233=연속스펙트럼 정보한계, 아무도 못 넘음... 학습이 유일승자 성립 안 함."** 더 나아가 d54(합성개구
capstone)는 짧은 창에서의 "연속표현 승" 자체가 **창-길이 아티팩트**임을 보임(창을 늘리면 적응형 RELAX가
다시 추월). **이 계보의 최종·정직한 결론(파일 자체의 2026-06-29 논문정리 절)은 "C3: 어떤 방법도 정보한계를
못 넘는다(엄밀 negative), 학습의 유일 지속가치=보정 UQ"**이지 "DPS가 RELAX를 이겼다"가 아니다. **실제로
정확도까지 유의하게 이긴 건 set-diffusion(atom-DDPM)이 아니라 완전히 다른 후속 아키텍처(빠른 amortized
posterior, 위 표 별도 행)**다. **이 정정은 아래 제안 4(flow-matching distillation)의 전제에 직접 영향** —
"이미 정확도로 이기는 방법이 속도 때문에 창고행"이 아니라 "비학습과 동률이던 방법을 속도 문제로 대체한
자리에 이미 다른(더 빠르고 더 정확한) 아키텍처가 들어와 있다"가 정확한 서술이다. §3 제안 4를 이 정정에 맞춰
재작성함.

### 1.2 FAIL / COLLAPSE (반복 확증된 dead end — **재제안 금지**)

| 방법 | 왜 죽었나 | 반복 횟수 |
|---|---|---|
| **Raw 좌표/점군 직접 회귀** (필드 또는 파수로) | under-determined obs서 MSE 최적해=평균 → mean-collapse(0.25=zero-predictor) | ≥6회 독립(포인트 자기지도, ConvCNP류 4변형, amortized field망, 판별 성분망 DETR) |
| **T-NSE (feedforward 방향/파수 amortization, cross-attn+Hungarian)** | aperture-limited 약신호 → 입력-불변 붕괴(kx_std≈0.0001) | 4개 변형 전부, 견고 negative |
| **DETR/set-prediction (A2, 좌표 직접 예측)** | downsample+bipartite가 정밀 국소화 실패, grid+offset보다 나쁨 | 1회, grid-free-by-construction이 오히려 나쁨 |
| **점군-native attention/perceiver (M3-v1, 최근 07-12)** | raw점 attention이 beamform-CNN의 물리 귀납편향을 데이터만으로 재발견 못함(8k 샘플 예산) | 1회, 실패창 +0.279 vs CNN +0.372 — **표현 축 폐쇄 4번째 사례로 명기됨** |
| **연속 (kx,ky) gradient 회귀 (라벨 없이)** | 손실지형이 비볼록 → gradient가 k~0에 갇힘(RELAX가 grid search 쓰는 이유 그 자체) | 1회, 원리 확인 |
| **Neural operator (FNO류, F4)** | near-mean-collapse, 비학습 dense-ridge(F1)보다도 나쁨 | 1회, "근본적 F1~F4 검토"의 일부로 확정 |
| **자기지도 인코더 사전학습 (W7, 해상상태 비정규성 인코딩 주장)** | **레코드 ID 지문조회였음** — LORO R² −0.603 | 1회, 심각한 사전점검 누락 사례(#7 신설 계기) |
| **모드-시킹/posterior 평균 readout (medoid, MAP-select)** | 다봉 posterior에서 mean/mode readout이 BMA보다 나쁨 — greedy의 sharp commitment를 못 흉내냄 | 3변형 전부 |
| **track-level 순차/critic/결정-헤드 네트워크** (M2×M7, critic v1/v2, L4) | LORO 전이 불가(레코드 하나가 전체를 깨뜨림), 선택상금 +0.14 미회수 | ≥6개 사전등록 실험 전부 FAIL |
| **해류(Uc) 블라인드 추정 (4개 목적함수)** | 단일 sparse 트랙은 (파수,조우주파수)만 관측 → Uc 비식별(편향, 평탄 아님) | 4개 전부, oracle 대조는 통과(정보는 있으나 식별 불가) |

### 1.3 PARTIAL / CONDITIONAL (스코프 한정에서만 참 — 인용 시 스코프 필수 병기)

| 방법 | 조건 | 캐비엇 |
|---|---|---|
| **DPS field-level diffusion prior (무조건)** | 메커니즘 작동(obs-fit 0.98+) | 300 obs가 2×4096 field DOF 과소결정 → seed별 0.09~0.84 들쭉, 성분 prior로 대체됨 |
| **amortized "fast" posterior (d56 계열, 직접 필드 회귀+Gaussian NLL)** | **실시간 사이클 제약 안에서만** 최선(0.222, 8ms, UQ) | 사이클 제약 풀면(긴 창 허용) RELAX가 다시 추월(0.240) — "무조건 최고" 인용 금지 |
| **SmultHead (장면-적응 잡음바닥)** | 실패창 rescue 재현(+0.21 t4.6) | **saliency 0 — 상수 3.3으로 완전 환원**. "학습이 장면-적응"은 거짓, 진짜 기여=상수 자동발견 |
| **E-28 FiLM/게이트 조건화** | 원 "정보벽 확증" 결론 자체가 **기각됨**(아레나 퇴화 아티팩트) | 열린 레짐서 +0.04~0.08 진짜지만 incumbent(l1s2bw)를 못 이김, B는 자기 사전등록 위반 |
| **D-optimal 능동 센싱 학습 정책 (자유 조준)** | linear/이산 벤치마크서 real, +16~25% | **continuous/HOS서 증발**(+1%, wash) — 이산 벤치마크 아티팩트로 판명 |

### 1.4 메타 교훈 — "결론이 뒤집힌 사례"에 특히 주의
- E-28의 "정보의 벽 확증"은 2026-07-17에 **완전 기각**됨(아레나가 조건화할 변수가 μ 하나뿐인 퇴화 레짐이었음).
  **이 문서를 읽는 사람은 project_e28_gennet 원문이 아니라 반드시 정정 참조(project_e29_e39)를 봐야 함.**
- "이산 벤치마크"가 만든 가짜 레버 최소 3종(FIM predictable-zone novelty, active-sensing 승리, discrete-linear
  gain +23.5%)이 **연속 스펙트럼으로 가면 전부 증발**(연속성-사다리 p=2.6e-3으로 유의하게 확인). 새 방법을
  이산·저성분 벤치마크에서만 검증하고 "이겼다"고 결론 내리면 **거의 확실히 재현 안 됨** — 이 지뢰를 다시
  밟지 말 것.

---

## 2. 핵심 진단 — 왜 성공/실패가 이렇게 깔끔하게 갈리나

### 2.1 실패의 공통 메커니즘 (MEASURED, 반복 재현)
모든 실패 사례의 공통점은 **"희소·과소결정 관측에서, raw 좌표/점을 표적(필드값 또는 파라미터)에 직접
매핑하는 손실(MSE/NLL)로 학습"**이다. 사후분포가 다봉/평탄하면 MSE의 베이즈-최적해는 **평균**이고, 평균은
운용상 의미 없는 답(0.25=zero-predictor)이다. 이건 **"아직 안 풀린 표현학습 문제"가 아니라 point-estimation
자체가 그 레짐에서 잘못된 목표라는 구조적 사실**이다 — 데이터 3배, 용량 1.5배를 부어도 손실 바닥이
안 움직이는 것으로 반복 확인(diffusion prior 재학습, set-diffusion mixed 150k+d320). `feedback_train_enough_
before_judging`이 늘 재확인시키듯 "학습 안 됨"의 상당수는 원래 **버그**(좌표 spectral bias, FF 스케일)였지만,
버그 다 잡은 뒤에도 살아남는 진짜 벽이 바로 이것이다.

### 2.2 성공의 공통 레시피 (MEASURED)
성공 사례 전부가 예외 없이 다음 패턴을 공유한다:
> **네트워크는 저차원·물리-정규화된 양(스펙트럼 gain, 잡음배수, 궤적 정책)만 예측하고, 실제 필드 재구성은
> 닫힌형·볼록(convex)한 물리 solve(Woodbury MMSE, 분산관계-제약 최소자승)에 넘긴다.**
- net-MMSE: net은 k/kp 정규화 격자 위 gain γ(k) 하나만 (51×51, record-agnostic) → MMSE가 나머지.
- set-diffusion: 생성 대상이 필드가 아니라 **성분 atom**(kx,ky,logE) — RELAX와 같은 도메인.
- 궤적 정책: 예측 대상이 필드가 아니라 **다음 침로**(스칼라/저차원 행동) — CRB가 보상.
- beamform-detect: CNN이 파수를 회귀하지 않고 **검출**(heatmap)한다 — 회귀가 아니라서 안 뭉갬.
**공통점: 네트워크가 한 번도 "불확실한 상황에서 하나의 값을 커밋"하도록 강요받지 않는다.** 물리가 그 역할을
맡고, 네트워크는 물리가 쓸 재료(soft prior)만 만든다.

### 2.2b ★세 번째 범주 (MEASURED, 진단 질문에 대한 필수 보완) — "학습 필요"로 보였던 문제의 상당수는 검증 프로토콜 버그였다
§2.1/2.2의 "정보 대 표현" 이분법 밖에 **제3의 범주가 실재한다**: 배포 레시피 재검토(H4/H5, `project_h1h2_
learning_closure`)가 밝힌 것은, 초기에 "학습이 필요하다"는 동기가 됐던 실패의 **상당 부분이 정보/표현 문제가
아니라 순수 검증-프로토콜 버그**였다는 것이다 — 잡음바닥 하이퍼파라미터(pexp, smult)를 **창-안 held-out
잔차로 검증**하면, 관측점 수 n이 늘수록 그 잔차에 과적합해 잡음바닥이 계속 수축(n-독약)하고, 배포 시점의
**창-밖(예보) 질의**에서는 오히려 나빠진다. 이 버그를 고치자(비적응·고정 하이퍼파라미터 + range-stratified
샘플링) 조우예보 corr이 **0.10→0.90~0.99로, 학습 없이** 뛰었다. **이건 §2.1의 "표현학습 미해결"도 §2.3의
"정보의 벽"도 아니다 — 애초에 비교 기준(비학습 baseline)이 부러져 있었을 뿐**이다. **실무적 함의**: 새
네트워크를 제안하기 전에 항상 먼저 물을 것 — "이 실패가 진짜 정보/표현 한계인가, 아니면 비학습 baseline의
검증 프로토콜이 깨져 있어서 그렇게 보이는 것뿐인가?" 이 프로젝트에서 후자였던 사례가 최소 1건(그것도 가장
파급력 컸던 사례) 확인됐다 — **재확인 없이 "정보의 벽"을 전제하는 것 자체가 위험**하다는 대칭적 경고.

### 2.3 정보의 벽 vs 표현학습 — 레짐별로 답이 다르다 (핵심 질문에 대한 직접 답)
**둘 다 맞다, 단 레짐이 완전히 분리된다:**
- **Well-determined/CRB-포화 레짐(5comp, 밀집 관측)**: 진짜 정보의 벽. RELAX·dense-MMSE가 이미 CRB의 3배
  이내(near-optimal). **어떤 네트워크도 여기서 못 이긴다** — 5중 이상 독립 확증(추정 CRB근접·자유센싱균일·
  연속표현 F1동률·비선형 무력·용량증가 무력). **표현학습 문제가 아니다.**
- **Under-determined/이동-센서 식별 레짐(15~30comp, sparse+synthetic aperture)**: CRB가 예측하는 상한과
  실제 추정기(RELAX greedy, dense ridge) 사이에 **정량화된, 아직 안 닫힌 갭**이 실재한다(FIM bound corr
  0.71~0.93 vs RELAX 0.27~0.54 — `project_synthetic_aperture` achievability 분해). **이건 정보의 벽이
  아니라 알고리즘의 벽**이고, 바로 이 갭에서만 set-diffusion/amortized-prior가 RELAX를 실제로 이긴다.
  **여기가 "더 나은 네트워크"가 진짜 움직일 수 있는 유일한 순수-추정 축이다.**
- **센서 기하 자체(개구/궤적)**: CRB의 "벽" 위치 자체를 유일하게 옮긴 레버는 네트워크가 아니라 **플랫폼
  운동**(cross-track 이동이 방향 CRB를 2.5~4.4배 조인다). 여기서 네트워크(비근시 정책)가 greedy를 이긴 건
  추정이 아니라 **순차 의사결정** 문제였기 때문 — 이 구분이 왜 여기서만 학습이 "새 것"을 만들어내는지 설명한다.

### 2.4 한 줄 종합 (ARGUED, 위 MEASURED 발견들의 종합)
> "더 나은 네트워크"가 이 프로젝트에서 여전히 움직일 수 있는 축은 정확히 두 개뿐이다 — **①이미 알려진
> 좁은 형식(저차원 물리-정규화 prior + 닫힌형 물리 solve)의 더 나은 버전**과 **②센싱/궤적을 정하는 순차
> 의사결정**. 그 형식을 벗어난(raw 표현·end-to-end·점추정 discriminative) 모든 새 아키텍처는, 아무리
> 최신이어도, 같은 mean-collapse 벽에 부딪힐 개연이 매우 높다 — 이건 비관이 아니라 ~15개 독립 실험이
> 가리키는 방향이다.

---

## 3. Ground-up 제안 (5개; 문서상 배치는 1→5, **실제 권고 우선순위는 1→2→3→5→4** — 제안 4는 §1.1 정정으로 하향됨)

### ★제안 1 — 궤적 정책의 미완성 고리를 닫는다 (완결형, 신규 아키텍처는 아님)
**(a) 무엇**: `d74`(learned non-myopic trajectory policy)를 (i) 이산·선형·16 에피소드가 아니라 **연속
다방향 HOS 비선형 바다**로, (ii) oracle 파수 achievability가 아니라 **실제 production 추정기(net-MMSE)를
루프 안에** 넣어서, (iii) 실측 센서 제약(mesh Nyquist, 방사모델 범주오류 — `project_observation_operator_
audit`)을 반영해서 재실행.
**(b) 왜 병목에 맞나**: 이 프로젝트에서 유일하게 "학습이 정보 자체를 새로 만든" 검증된 사례. 순수 추정
개선이 아니라 순차 의사결정이라 CRB-포화 논변이 안 막는다.
**(c) 기존 실패와 차이**: 새 아키텍처가 아니라 **기존 성공을 이 프로젝트의 가장 흔한 함정(이산→연속
붕괴 패턴, E-16~E-27·d65 연속성-사다리)에 노출시켜 살아남는지 확인**하는 것 — 재검증이지 재시도가 아님.
**(d) novelty/선점**: platform-motion 합성개구 자체는 SAR/sonar 기지이나, phase-resolved 파랑 방향식별
+ CRB 궤적최적화 조합은 project litcheck상 미발견(`direction3_gridfree_prior` d72). **연속/HOS로 확장은
검증되면 이 novelty를 실질화**(현재는 이산·oracle이라 achievability 미입증 상태).
**(e) 최소 검증**: 기존 `d74` 하니스에 HOS 바다 1개 클래스 + net-MMSE achievability 대체만 추가 — 새 코드
대부분 재사용 가능, 소규모 실험.
**리스크**: 이산→연속 붕괴 패턴이 여기도 적용될 수 있음(가장 유력한 반증 시나리오) — 확인 자체가 가치.

### 제안 2 — Simulation-Based Inference (NPE)를 regime-shape θ(keystone③ CRB 파라미터)에
**(a) 무엇**: E-51/52가 이미 분석적 Gaussian-근사 CRB로 특성화한 **저차원(5-dim) 스펙트럼-형태 파라미터**
θ=(d1,Tp1,d2,Tp2,w)에 대해, normalizing-flow 기반 **Neural Posterior Estimator**(SBI 표준 기법)를 시뮬레이션
쌍(θ,관측)으로 학습 — 전체 필드가 아니라 **이미 CRB로 벽을 아는 바로 그 파라미터**의 완전한 posterior를 냄.
**(b) 왜 맞나**: §2.2 성공 레시피와 정확히 같은 형태(저차원·물리-정규화 표적) + 기존 JointEngine
Gaussian-근사 likelihood의 **경험적 대조군**을 제공(현재 keystone③은 순수 해석적 CRB, achievability를
"θ̂~N(θ,CRB) MC"로 근사) — NPE의 실제 학습된 posterior 폭을 CRB와 대조하면 "Gaussian-근사가 진짜 posterior
모양을 대표하는가"(GUARD2가 미납한 부분)를 **경험적으로 직접 검증**할 수 있음. §5 회계 프로토콜(CRB 판별기)의
독립 교차검증 도구가 됨 — 새 헤드라인이 아니라 **기존 keystone③의 신뢰도를 올리는 보강 도구**로 포지셔닝.
**(c) 기존과 차이**: d56 계열 amortized posterior는 **필드 query별 (mean,σ)**(factorized, Gaussian) — NPE는
**5-dim θ의 결합(joint) posterior**(비-factorized, 임의 형태). E-51/52의 물리 CRB조차 못 잡는 비-Gaussian
posterior 형태(다봉·비선형 상관)를 NPE는 원리상 표현 가능 — 이건 "더 정확한 추정"이 아니라 **"CRB의
Gaussian-근사가 어디서 깨지는지 측정"**이 목적이라 §2.3의 "표현학습으론 못 넘음" 결론과 안 부딪힘.
**(d) novelty/선점**: WebSearch 확인 — buoy/vessel-motion 기반 amortized NN 방향스펙트럼 추정은 **이미
선행 존재**(JMSE 2025 FPSO 계열, PSV encoder-decoder). **바로 그 형식("NN이 파 스펙트럼 파라미터 추정")은
선점** — 좁은 차별점 = "이미 해석적 CRB로 특성화된 저차원 파라미터의 SBI-경험적 posterior를 그 CRB와 직접
대조해 Gaussian-근사 오차를 정량화"뿐. 헤드라인 방법론 주장 금지, keystone③ 보강 도구로만.
**(e) 최소 검증**: arena X(이미 존재하는 build_sea_crosstp)에서 (θ,z) 시뮬 쌍 5k개 생성 → 기성 SBI 라이브러리
(sbi 패키지 등)로 NPE 학습 → n=64 held-out서 NPE posterior std vs 해석적 √CRB 산점도 하나로 1차 판별 가능
(수 시간급 소규모 실험).
**리스크**: E-51 자체가 이미 "GUARD2 achievability 미납"을 큰 리스크로 남겨뒀음 — NPE가 그 미납을 메우려다
오히려 "Gaussian-근사가 틀렸다"는 새 불편한 결과를 낼 수도 있음(그 자체가 정직한 기여이나 서사 관리 필요).

### 제안 3 — k-grid gain-map CNN의 방향 정준화(rotation canonicalization)
**(a) 무엇**: 현재 net-MMSE의 CNN은 `kxn*kp_e, kyn*kp_e`로 스케일만 정규화된 **고정 Cartesian 격자**에서
동작(방향 정렬 없음) — 입력을 **beamform-검출 우세방향으로 회전정렬**한 뒤 CNN에 먹이고(canonicalization),
출력을 다시 역회전 — 방향에 대해 등변(equivariant)이 되도록 강제.
**(b) 왜 맞나**: 훈련은 `mdir~U(0,360°)`로 방향을 무작위화해 **경험적** 방향-불변만 확보 중(데이터 증강
의존) — E-28의 "지지집합 암기"(L1 prior_gain이 고-k 대역에서 붕괴, 학습분포 밖 지지 못 함) 패턴이 방향
축에도 잠재할 수 있음. 정준화는 **구조로 보장**해 표본효율·OOD 강건성을 개선할 후보.
**(c) 기존과 차이**: N5(door1 채널)에서 위상-등변성 뼈대는 검증됐으나(4/4 테스트) **주 추정 파이프라인엔
미적용**(별도 인프라 검증용). 이건 그걸 **production CNN에 실제로 적용**하는 첫 시도.
**(d) novelty/선점**: WebSearch 확인 — 등변 DOA 추정(icosahedral CNN, group-equivariant CNN)은 **일반
신호처리 문헌에 이미 풍부**. "등변 네트워크로 방향 추정"은 novelty 전무. 좁은 차별점 = **이 특정 파이프라인
(물리-정규화 gain-map + closed-form MMSE) 안에 정준화를 끼워넣는 것**뿐 — 방법론 novelty 주장 절대 금지,
순수 엔지니어링 개선으로만 제시.
**(e) 최소 검증**: 기존 bwtask 학습 파이프라인에서 입력 회전+출력 역회전 한 겹만 추가(모델 구조 변경
최소) → 기존 held-out val 세트에서 baseline과 A/B, 저비용.
**리스크**: 현재도 do-no-harm(무해)인 방향 데이터 증강이 사실상 이미 잘 작동 중일 수 있어(측정 안 됨)
개선폭이 작을 수 있음 — 우선순위가 제안 1·2보다 낮은 이유.

### 제안 4 [★하향 조정·전제 정정됨] — set-diffusion 성분 prior의 다봉 UQ만 distill (정확도 재도전 아님)
> **원안 전제 오류 정정(제가 원문 재확인, §1.1 참조)**: "이미 정확도로 이기는 방법이 속도 때문에 창고행됐다"는
> **부정확**하다 — set-diffusion(DPS)은 최종·엄격 검증(F1-F4)에서 **비학습 baseline과 동률**(0.234 vs
> 0.233)이었고, 속도 문제가 나오자 이미 **다른(더 빠르고 더 정확한) 아키텍처**(amortized 직접-필드
> posterior, d56-d61)가 그 자리를 채워 F1도 유의하게 이겼다(§1.1). ⟹ "distill해서 부활"시킬 **정확도
> 우위 자체가 없다** — 이 제안은 정확도 회복이 아니라 **다른 이유**로만 값이 있다(아래 (b) 재정의).
**(a) 무엇**: set-diffusion 성분 prior의 **완전한 다봉(multi-modal) 생성 posterior**(atom 공간 DDPM)를
flow-matching/consistency-distillation으로 few-step화 — 단 목표는 nRMSE 개선이 아니라 **UQ의 형태**.
**(b) 왜 맞나(재정의)**: 이미 배포된 amortized posterior(d56)는 질의점별 **독립 factorized Gaussian**
(mean,σ)만 낸다 — 방향 모호성(180° 대칭)이나 성분간 상관 같은 **진짜 다봉/상관 구조는 원리상 표현
불가**. set-diffusion은 이걸 표현할 수 있다(생성 표본이 다봉일 수 있음). **즉 이 제안의 유일하게 방어
가능한 가치 = "기존 빠른 posterior가 구조적으로 못 주는 상관·다봉 UQ를, 그것도 빠르게" — 정확도 경쟁이
아니라 UQ-형태 경쟁.**
**(c) 기존과 차이**: 아키텍처 재시도가 아니라 이미 **동률로 끝난** 아키텍처의 **UQ-형태만** distill.
**(d) novelty/선점**: WebSearch 확인 — flow-matching/consistency distillation은 2026년 현재 매우 활발한
범용 ML 주제(F2D2, STMD 등 다수), 방법 자체는 완전 기성. 응용/엔지니어링 기여로만 제시 가능.
**(e) 최소 검증**: 먼저(실험 아님, 저비용 확인) — 기존 amortized posterior의 factorized-Gaussian
가정이 실제로 깨지는 질의점 비율을 기존 set-diffusion 표본에서 역산(다봉성 진단). **이게 무시할 수준이면
이 제안 전체를 폐기하는 게 맞다** — distillation 작업 전에 반드시 이 진단부터.
**리스크(상향)**: 원안보다 **가치 자체가 훨씬 좁아짐** — 우선순위를 제안 5 아래로 낮춤. 다봉성 진단에서
음성이 나오면 **제안 폐기가 옳은 결론**(이 리뷰의 규율 "재제안 금지"와 동일 정신으로, 여기서도 값싼
반증을 먼저 하라).

### 제안 5 — coarse-beamform 조건부 국소 정밀화 attention (off-grid 정밀도 장벽 겨냥)
**(a) 무엇**: RELAX/beamform이 찾은 **성긴 후보 파수(atom)를 query**로, 원본 관측점을 attention으로
cross-attend해 **국소 정밀 보정**(off-grid refine)을 배우는 작은 모듈 — attention을 **주 표현이 아니라
국소 정밀화 연산자**로만 사용.
**(b) 왜 맞나**: `direction3_gridfree_prior`가 명시적으로 **"off-grid ID 정밀도 장벽은 amortize 안 됨"**을
3중 확증(d30/d33/d34)했으나, 그건 **feedforward가 정밀 위상 배치를 처음부터 예측**하려 했을 때 얘기다 —
이미 성긴 후보가 주어진 상태에서 **국소** 보정만 배우는 건 미시도(untried) 좁은 영역.
**(c) 기존과 차이**: M3-v1(점군-native attention, 07-12)은 attention을 **1차 표현**으로 썼다가 실패(§1.2) —
이건 attention을 beamform의 **후처리기**로만 국한. E1~E3(direction3)의 오차예산 분해에서 "mode-geometry
58%"가 남은 회수가능 성분이라 특정된 지점과도 직결.
**(d) novelty/선점**: WebSearch 확인 — "coarse beamforming + attention refinement" 조합은 레이더/오디오
초해상서 근접 사례 존재하나 **해양파 필드 재구성 맥락에서는 특정 못 함**(비교적 열린 틈). 단 일반 기법
자체(coarse-to-fine attention)는 컴퓨터비전에 풍부 — "grid-free 정밀도 장벽 해소"라는 **문제 특정**이
novelty지 attention 자체가 아님.
**(e) 최소 검증**: `d34_learned_offgrid.py` 계열 인프라(이미 존재, 이전 시도의 실패 지점) 재사용 —
feedforward 전체 배치 대신 "이미 준 성긴 배치 + 국소 attention 보정"만 검증하는 작은 ablation.
**리스크**: 이전 R1(direction2) 감사가 "d_geom 58%는 검출이 아니라 sparse-obs 유효해상도(정보) 한계"로
**재프레임**됐음(§1.4-유사 반전 패턴) — 이게 진짜 정보한계라면 이 제안도 §2.3의 "well-determined 벽"에
해당해 실패할 수 있음. **가장 반증 위험이 높은 제안** — 그래도 미시도라 값은 있음.

### 재고하지 않는 것 (명시적으로, 이미 폐기됨 — §1.2 참조)
- **Vanilla FNO/DeepONet을 1차 추정기로**: F4로 이미 **내부에서** 사망(near-mean) — **그리고 별도로 제가
  이번 리뷰에서 직접 WebSearch 확인한 바, 외부에서도 이중으로 죽어 있다**: Ehlers et al. **OE2023**
  (이 프로젝트 novelty litcheck의 기존 선점 앵커, arXiv 2305.11913)가 이미 **정확히 이 문제**(희소
  레이더 관측→phase-resolved 파랑재구성)에서 **U-Net vs FNO를 직접 head-to-head 비교해 FNO 승리를
  보고**했다("FNO demonstrates superior performance ... due to its global approach ... in Fourier
  space"). 즉 "FNO를 써보자"는 **내부 실패 + 동일 그룹의 외부 선행연구가 이미 벤치마킹까지 끝낸 이중
  선점** — 재고 가치 0.
- **물리-잔차(PDE-residual) 자기지도 학습(라벨-free physics-informed 신경망)**: 이 프로젝트가 시도한
  자기지도(obs-fit 직접 회귀, `project_selfsup_novelty`)와는 **다른 메커니즘**(자유표면 경계조건
  잔차를 손실에 넣는 PINN류)이라 내부적으로 정면 시도된 적은 없어 보이나, **같은 Ehlers/Stender/Hoffmann
  그룹이 2025-08 이 정확한 메커니즘으로 이미 발표**(arXiv 2508.03315, "PINO" — 라벨 없이 자유표면 경계조건
  잔차를 신경 연산자 손실에 넣어 희소 관측→비선형 phase-resolved 파장 재구성). **novelty 선점 정보 신규
  추가**(기존 memory엔 "PINO2025"란 이름만 있고 이게 정확히 무슨 메커니즘인지 명시 안 돼 있었음 — 이번에
  확인) — 이 축도 사실상 봉인된 것으로 취급 권고.
- **일반 좌표-MLP implicit representation (SIREN류)**: 이미 시도된 raw-좌표 회귀 계열과 본질적으로 동일한
  실패 패턴 예상(spectral bias 버그를 고쳐도 mean-collapse 축은 안 풀림) — 프로젝트가 이미 발견한 "올바른
  implicit representation은 사실 물리 분산관계 기저 그 자체"라는 결론과 충돌.
- **점군-native transformer를 1차 표현으로**: M3-v1이 최근(07-12) 정면으로 닫음.
- **Track-level 순차/critic 네트워크**: 6개 사전등록 실험 전패, LORO 근본 취약.
- **희소점군 관측점 능동-선택 네트워크(어디를 관측할지 학습)**: WebSearch 확인 — "관측점 선택+파장재구성"
  조합도 외부에 근접 선행 존재(2025 ScienceDirect "observation point selection" 프레임워크, U-Net+GAN
  Actor-Critic 관측점 선택). 이 프로젝트 내부에서도 이미 능동센싱 정책(자유조준)이 이산서만 이기고
  연속/HOS서 증발함을 확인(§1.3) — 안팎으로 이중 약함, 제안 목록에서 제외.

---

## 4. 정직한 메타판정 (핵심 질문 4에 대한 답)

memory의 결론(정보의 벽·추정기-급 폐쇄·양성 사망)을 감안할 때 **"더 나은 네트워크"가 움직일 수 있는 축은
좁지만 실재한다** — 무조건 "네트워크는 끝났다"도, "네트워크만 더 좋으면 뚫린다"도 둘 다 틀렸다.

- **못 움직이는 것(강한 확신, MEASURED, 재시도 무의미)**: well-determined 레짐의 point-estimation 정확도.
  raw-표현 end-to-end 학습 전반(≥15 독립 확증). 이건 "아직 못 찾은 좋은 아키텍처"가 없어서가 아니라, 그
  레짐에서 point-estimation 자체가 통계적으로 잘못된 질문이기 때문(다봉 posterior의 평균은 무의미) —
  트랜스포머든 SBI든 확산모델이든 raw 표현으로 점추정을 시도하는 한 같은 벽에 부딪힌다.
- **움직일 수 있는 것(ARGUED, 이 리뷰의 실질 기여)**: **①** 이미 검증된 좁은 형식(물리-정규화 저차원 prior
  + 닫힌형 물리 solve)의 개선(제안 2,3,4,5는 전부 이 형식 안) — 여기선 새 아키텍처가 유효할 수 있음, 단
  "더 똑똑한 네트워크"가 아니라 "이 형식 안에서 더 나은 부품"이라는 겸손한 스코프에서만. **②** 센싱/궤적
  설계라는, 아예 다른 문제 클래스(추정이 아니라 순차 의사결정) — 여기가 학습이 유일하게 "이겼다"가 아니라
  "새 것을 만들었다"고 부를 수 있는 곳(제안 1).
- **왜 이게 "정보한계 진단학위"와 모순 안 되나**: 이 프로젝트의 학위 척추(A: CRB-앵커 운용성 envelope)는
  "정보한계가 있다"는 게 아니라 "**어디가 정보한계고 어디가 추정기 효율 문제인지 판별한다**"는 것 — 제안
  2~5는 바로 이 판별선(under-determined·achievability-gap 레짐)의 **효율 쪽**을 공략하는 것이라 척추와
  정합적이다. 제안 1은 벽 자체를 옮기는 센싱-설계 축이라 별개 정합.

---

## 실행 요약 (3-5줄)
이 프로젝트는 이미 ~40개의 학습 아키텍처를 시도했고, 실패 패턴(raw-표현 point-estimation → mean-collapse)과
성공 패턴(물리-정규화 저차원 prior + 닫힌형 물리 solve)이 예외 없이 갈린다 — 새 아키텍처를 제안하기 전에
반드시 이 지도를 봐야 한다. Well-determined 레짐의 점추정 정확도는 정보의 벽이 진짜라 어떤 네트워크로도
안 움직이지만(≥15 독립 확증), under-determined 레짐엔 CRB-대비 정량화된 알고리즘 갭이 남아있고, 플랫폼
궤적 설계는 학습이 유일하게 벽 자체를 옮긴 축이다. 최우선 제안은 새 아키텍처가 아니라 **이미 검증된 유일한
"진짜 신규" 성공(학습 궤적 정책)을 이 프로젝트의 가장 흔한 함정(이산→연속 붕괴)에 노출시켜 완결**하는 것이고,
다음으로 SBI/등변화/증류/국소-attention은 전부 "물리-정규화 저차원 prior + 물리 solve" 틀 안의 겸손한 부품
교체로만 값이 있다 — 전부 novelty 주장 아니라 응용/보강으로 스코프해야 한다.
