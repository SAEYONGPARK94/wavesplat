# E-44 설계 N2 — "family 밖"의 정직한 지도: field 과제에선 공집합, 살아있는 질문은 과제 전환 하나 (2026-07-17)

> **결론 먼저**: 나는 새 아키텍처를 제안하지 않는다. field-corr 과제에서 금지된 family 밖은
> **구조적으로 비어 있다**(아래 §1, Bayes 분해 + fact 9). 이것은 회피가 아니라 증명 시도다.
> 밖에 사는 유일한 것은 **추정기가 아니라 과제 함수형(task functional)의 전환** — 배포 목표인
> encounter forecast에서 상금이 존재하는지의 감사이며, 프로브를 이미 돌렸고(0.15 s/sea)
> n=960 사전등록이 ~10분에 끝난다(§4).
> fact 7("우도 평평/점수 가파름")은 오케스트레이터 정정대로 **가설로 강등**하고 어디에도
> 기대지 않았다 — 아래 논증은 fact 7 없이 성립한다.

---

## 1. 왜 "밖"이 없는가 — Bayes 분해 논증

생성 모델은 정확히 인수분해된다:
`p(field, z) = Σ_θ p(θ) · p(realization|θ) · p(z|field)` (arena X: θ=(d1,d2,Tp2,r2,…),
realization = 위상 200개 + 방향 draw 200개, `wave_generator.py:205-222`).

임의의 blind 추정기는 z의 가측 함수 f̂(z)다. 기대 점수(corr 포함 — corr의 Bayes 최적해는
정규화된 posterior mean `Σ_θ p(θ|z)·E[F/‖F‖ | z,θ]`)는 Bayes predictor가 지배하고, 그
**형태 자체가 "θ 사후 → 혼합"** 이다. 즉 금지된 family는 여러 가설류 중 하나가 아니라
**이 생성 모델의 Bayes 형태 그 자체**다. 따라서 "family 밖"은 논리적으로 세 가지뿐:

| 채널 | 내용 | 판정 |
|---|---|---|
| (i) 같은 함수의 다른 근사 | 아키텍처만 바꿈 (unrolled solver, direct regression, CV 선택, 새 입력 채널) | 천장 동일 — family의 Bayes ceiling(≈+0.03, fact 6)을 못 넘음 |
| (ii) within-θ 조건부의 비-Gaussian 보정 | E[f\|z,θ]를 random-phase 정확 조건부로 교체 = **realization 구조 활용** | **fact 9가 폐쇄**: RL deconv @무한 SNR + GT-peek이 −0.0003 tie; +0.05엔 각도 200개 중 ~90개 필요 |
| (iii) z 밖의 정보 | 외부 regime 공급 | fact 12 — 이미 이 아크의 납품물 |

### fact 11 감사 — fact 9는 family 밖의 fact 11에 치명적이다. 명시적으로 말한다.

- **주파수 격자**: lobe당 100성분, Δω = 2.5ωp/100 = 0.0224 rad/s, **Δω·T = 1.57 rad ≪ 2π**
  ⟹ 70 s 창에서 격자는 Rayleigh 분해능의 **4배 아래** — 이산 격자 구조는 관측상 연속체와
  구별 불가. 격자를 "보는" 추정기는 존재할 수 없다(창 연장은 fact 10이 폐쇄).
- **결정적 진폭 (constant-modulus 4차 구조)**: Rayleigh 셀당 ~4 tone, 셀당 4차 큐뮬런트
  결손 ~1/4, 셀 ~25개 ⟹ 수 nats 규모의 비-Gaussian 정보가 **원리상** 존재한다. 그러나
  현금화 경로는 둘뿐: (a) E[f|z,θ] 보정 = 채널 (ii) = **fact 9로 죽음**; (b) p(θ|z) 선명화 =
  **N1 family 내부**. 밖에서는 0이다.
- **N1 인계(가격표 포함)**: ensemble-Gaussian evidence를 exact random-phase evidence로
  교체하는 것이 fact 11의 유일한 생존 경로다. 규모 일치가 시사적: 비-Gaussian 정보 ~수 nats,
  그리고 argmax−true-node 갭 3 nats가 한 블록에서 +0.037의 점수 값(정정된 생존 조각).
  단, 그 +0.037 환산은 단일 블록·해석 미검증 — N1이 열거의 evidence만 바꿔 재실행하면 판정된다.

### 기각 후보들 (프롬프트의 방향 각각, 한 줄 사형선고)

- **unrolled/amortized solver**: 통계 내용 0, 순수 공학(36 s→100 ms). 그리고 가속할 가치가
  있는 건 N1의 family인데 현재 그 가치가 +0.03·unimodal 유해 −0.044 ⟹ 비용-편익 탈락.
  N1이 천장을 올리면 그때 증류(distill)하면 된다 — 지금 지을 이유 없음.
- **γ 버리고 field/forecast 직접 회귀**: proper loss 하의 수렴점 = conditional mean =
  정확히 지금의 smear/BMA. **mean-collapse 역사(fact 13)는 이 정리의 경험적 얼굴**이지
  엔지니어링 실패가 아니다. 재시도는 같은 벽을 학습으로 재발견하는 것.
- **CV/transductive 선택, self-supervised 재랭킹**: 전부 z의 함수 ⟹ 같은 바운드. 특히
  held-out 관측점은 같은 궤적 위라 우도가 못 가르는 가설쌍을 CV도 못 가른다.
- **4차-통계 입력 채널 추가 CNN**: 출력이 γ면 incumbent 재건축(mandate 기각 조건), 효과
  경로는 암묵적 θ 추론 = N1 천장 이하, 게다가 학습 필요(금지). 유일한 정직한 자리는 N1의
  evidence 항.

---

## 2. 밖에 사는 유일한 것 — 과제 함수형 전환 (그리고 프로브 결과)

배포 목표는 domain-wide field corr가 아니라 **제어용 단기 encounter forecast**다. 상금
+0.078이 그 함수형으로 **투영되는지는 지금까지 아무도 재지 않았다**. 둘 중 뭐가 나와도
결정적이다:
- **전이 안 됨** ⟹ regime 상금은 field-metric 전용 아티팩트, fact 12는 "mapping 납품물"로
  강등, STOP에 독립적 두 번째 다리.
- **전이 됨** ⟹ fact 12(외부 θ 공급 사양)가 control-관련으로 승격 — 여전히 새 아키텍처는
  불필요, 사양의 가치만 올라감.

**프로브 실행됨** (`tmp/e44_probe_N2_encounter_transfer.py`, n=16, si=900..915 — 프로브
전용으로 소각; 결과 `results/direction3/e44_probe_N2_encounter_transfer.json`):
궤적을 t∈(70,80] s로 결정론적 연장(ping-pong 법칙 그대로, jitter 제거), 참-모드 렌더로
GT, arms = cur(incumbent γ) vs orc(supplied_grid 참 θ, e43 dose-0과 동일 구성).

```
oracle-θ − cur   field(앵커) : +0.0556 ± 0.0411  (sd 0.164, win 10/16)  ← +0.07 상금과 정합
                 enc 0-10 s  : +0.0206 ± 0.0930  (sd 0.372, win 11/16)  ← 미결
                 enc 0-5 s   : +0.0556 ± 0.0867
                 enc 5-10 s  : −0.0700 ± 0.1462  (sd 0.585, 꼬리 무거움: si901 d=−1.16)
                 transfer ratio 0.37 (CI가 모든 것을 포함)
```

n=16으로는 아무것도 판정 못 한다 — 그러라고 돌린 게 아니다. 프로브의 산출물은 **분산**이다:
per-sea sd 0.372 ⟹ n=960이면 SE 0.012, +0.03 바에 MDE ±0.024. 그리고 **0.15 s/sea**
(JointEngine 없음) ⟹ 사전등록 본실험이 **~10분**. 이 아크에서 처음으로 "결정 실험이 공짜"다.

---

## 3. (B) 가설류 — 한 문장

**"새 추정기 없음: 동일 입력 z → 기존 γ-MMSE family의 고정 arms(cur/m2/supplied-θ/dose) →
배포 함수형(플랫폼 궤적 0–10 s encounter corr)으로 재채점하는 과제-전이 감사; 학습 신호 없음."**

이것은 mandate의 두 기각 조건(incumbent 재건축 / θ-추론 family) 어느 쪽도 아니다 —
추정기 설계 자체를 §1의 논증으로 거부하고, 밖에 실재하는 유일한 자유도(점수 함수형)를 잰다.

---

## 4. (C) 사전등록 — E-44-N2 "ENCOUNTER TRANSFER" (전부 무학습, ~10분, 4080S 여유)

- **아레나**: `build_sea_crosstp`, **신선 si=1000..1959 (n=960)** + unimodal 컨트롤 240
  (`build_sea_bimodal(si, unimodal=True)` 유형, 신선 시드). si=900..915는 소각 완료.
- **쿼리**: 궤적 결정론적 연장 t∈(70,80] s 40점(프로브와 동일 코드 경로); 부-지표 (0,5]/(5,10].
- **arms**: `cur_grid` / **`m2`(필수 컨트롤, e37 mean-arm 레시피)** / `orc`(supplied_grid 참 θ)
  / `buoy`(5°,5%,0.05) / `ww3`(15°,10%,0.10) — dose는 e43과 동일 섭동 코드, BUNDLE_SEED 방식.
- **1차 판정 (TRANSFER)**: enc10에서 `orc − cur ≥ +0.03` AND 95% CI 0 제외 AND win ≥55%
  AND **`orc − m2`가 같은 바 통과**(cur만 이기고 m2를 못 이기면 회수 아님 — 필수).
  꼬리가 무거우므로(프로브 실측) **Wilcoxon을 공동 1차 검정**으로 사전등록: p<0.01.
- **2차**: `buoy − cur`가 orc 효과의 ≥70% 유지 ⟹ fact 12를 "control-관련 사양"으로 승격.
  horizon 분해 (0,5] vs (5,10] 보고(사전등록된 탐색 지표, 판정엔 불사용).
- **do-no-harm**: unimodal 240에서 supplied 1-lobe θ arm ≥ −0.01.
- **절대치 공동 보고**: 각 arm의 절대 corr + zero-γ 베이스라인 — "전이 없음"이 천장 포화인지
  prior-불감인지 구별하기 위함(내 실패 모드 #2의 완화).
- **★내 kill 조건**: enc10 `orc − cur`의 **CI 상한 < +0.02** ⟹ regime 상금은 배포 과제에
  존재하지 않음 ⟹ **이 아크 전체 STOP-and-write**, fact 12는 mapping 납품물로 강등, 그리고
  N1의 evidence 개선도 control 관점에선 무의미해짐(field-metric 전용). 반대로 TRANSFER면
  나는 "새 아키텍처 불필요, 외부-θ 사양이 control 납품물" 로 서명한다 — 어느 쪽도 내 설계를
  살리지 않는다. 이 사전등록에 나의 생존 이익은 없다.

---

## 5. 실패 모드 (내 것, 선제 고백)

1. **지표의 dof**: 40점 smooth 시계열의 corr은 유효 dof ~5–10 — per-sea 추정 자체가 잡음
   덩어리고 꼬리가 무겁다(프로브 실측 d=−1.16 존재). 평균이 아웃라이어에 휘둘리는 걸
   Wilcoxon 공동 검정으로 막지만, **corr-on-track이 제어 효용을 대변 못 할 위험**은 남는다
   (제어가 원하는 건 위상·파고 타이밍 — corr은 그 대리일 뿐).
2. **논증의 조건부성**: §1의 Bayes 바운드는 (a) 아레나 = 배포 분포, (b) evidence 근사 오차와
   lattice 양자화가 작다는 가정 위에 있다. (b)의 갭(수 nats)은 정확히 N1의 영토라 내 결론
   "밖은 공집합"은 유지되지만, **"blind 천장 ≈ +0.03"이라는 수치는 N1이 evidence를 고치면
   +0.05~0.07까지 올라갈 수 있다** — 그 경우에도 필요한 건 새 아키텍처가 아니라 더 나은
   우도(family 내부)라는 점은 불변.

---

## 6. Build / Stop call

- **아키텍처: STOP.** family 밖에 추정기는 없다 — 부재를 주장이 아니라 분해(§1 표)로 보였다.
  fact 9는 fact 11을 family 밖에서 죽인다; 안에서는 N1의 것이고 가격표는 3 nats/+0.037.
- **실험: BUILD 하나** — §4의 transfer 감사(10분, 무학습). 이 아크가 3주간 갚지 못한 질문
  "이 상금이 배포 과제에 존재하는가"를 처음으로 직접 잰다.
- 산출물: `tmp/e44_probe_N2_encounter_transfer.py`,
  `results/direction3/e44_probe_N2_encounter_transfer.json`, 이 문서.
