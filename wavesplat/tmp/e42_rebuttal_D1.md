# E-42 Rebuttal D1 — 기각은 unsound: DEAD 수치는 fallback 아티팩트, "식별성 벽"은 비-순차 추론, "exact likelihood"는 거짓 전제
2026-07-17. 원자료: `results/direction3/e42_blind_recovery.json` 재분석 + 코드 판독 + noisy-oracle 파일럿 (`tmp/e42_rebuttal_probe_D1.py`, `results/direction3/e42_rebuttal_probe_D1.json`).

## 헤드라인
v2 실험은 좋은 실험이고 수치는 믿는다. 그러나 **그 수치가 지지하는 결론은 "이 파이프라인 구성이 실패했다"이지 "식별성 벽 / 학습 설계 3종 사전 기각"이 아니다.** 근거 4개, 전부 원자료에서 재도출:

## 1. DEAD 수치(−0.0166)는 통째로 fallback 아티팩트다 (JSON 재분석)
```
fired(43/64):  fit_bmaF − cur = +0.0001     not-fired(21): −0.0509
HYBRID(fired→fit_bmaF, else cur) − cur = +0.0001 ± 0.0136
fit1 − cur = −0.0587,  ens1 − cur = −0.0596
unimodal harm: fired율 4.2%, 손해는 not-fired에서 −0.0186 (fired 1건은 +0.0676)
```
풀드 손실 전체가 **커밋이 아니라 포기의 착지점**에서 나온다: not-fired 시 mixture가 fit1(−0.059짜리 1-lobe 손 prior)로 지배되는 구성. **unimodal do-no-harm 위반(−0.0150)도 같은 아티팩트다** — 거짓 발화(4%)가 아니라 fallback이 낸 손해. 내 설계 문서(§2.2 backstop, Q4 수정: "포기 = incumbent 자신의 γ")가 명시한 바로 그 지점이며, v2는 그 반대(포기 = fit1)로 지어졌다. **파이프라인의 β→0 극한이 m2도 cur도 아닌 fit1인 실험으로 "우리 설계의 사망"을 선고할 수 없다.**

## 2. "fit_bmaF ≈ U_BMA ⟹ likelihood가 식별 불가"는 비-순차다 (코드 판독)
`fit_joint`의 mix 집합 = **evidence 가중 상위 48개(KEEP_HYP)로 이미 선별된 집합**이고, U_BMA는 그 위의 균일이다(`e42_detector_v2.py:223,238`). ~50k 가설 중 48개를 고르는 행위 자체가 likelihood의 식별이다 — 남은 48개가 score-균질하다는 것은 "argmax 근방이 평평하다"이지 "사후가 평평하다"가 아니다. **진짜 β→0 극한은 prior 위의 균일 = m2(−0.026)이고, β bracket [U_BMA, argmax]는 argmax의 반경 안만 왕복한다.** 온도 논증(Q2)에 대한 내 답: **concede하되 이유가 다르다** — 손실이 β에 있지 않음이 §1로 판명됐으므로 fit_bma_temp도 답이 아니다. bracket이 죽인 게 아니라, 애초에 병소가 가중치가 아니었다.

## 3. "exact marginal likelihood → 어떤 학습망도 못 이김"은 전제가 거짓 (Q1)
- **진폭은 결정적이다**: `wave_generator.py` — `amplitudes = √(2·S·dω)` 후 Hs 정확 정규화; rng는 위상·방향만 만짐(closure doc §3와 일치). 따라서 z|θ는 Gaussian이 아니고, 참 marginal은 100+100 방향 draw에 대한 **Gaussian이 아닌 혼합**이다.
- **family도 오지정**: 51-뱅크 k-양자화 — Δk=0.1kp, Tp1=7·T=70s에서 셀 경계 Δω·T ≈ **1.6 rad**(이 아크 자신이 280s에서 51→201 정제로 0.049→0.378을 봤다). smooth ensemble γ는 realization 공분산의 기대값일 뿐.
- 즉 이 evidence는 **composite likelihood**이고 그 사후는 p(θ|z)가 아니다. "학습망은 그것의 근사일 뿐"은 성립하지 않는다 — NLL 학습망의 표적은 (speckle·양자화 포함된) 생성분포의 참 사후다.
- **정직한 한정**: 이 공격은 "사전 기각" 논리를 죽일 뿐, "그 차이가 +0.03어치"라는 증거는 아니다. 실제로 v2 표면은 이제 기만적이지 않고(corr(d2_err,dlogE) = +0.077, e41의 +0.419 소멸) d2 median 17.5°로 꽤 좋다. **likelihood는 적이 아니었다. 결정층이 적이다.**

## 4. ★수익 임계 파일럿: v2가 달성한 오차 수준의 점-커밋은 이미 ≈+0.03을 번다
noisy-oracle dose-response (n=23 finite, si=100..123, 오차를 fresh-run 경험 분포에서 재추첨, reps=2):
```
s=0.00 (exact)        +0.0861 ± 0.0255
s=0.25                +0.0837 ± 0.0243
s=0.50                +0.0413 ± 0.0279
s=1.00 (v2 달성 오차)  +0.0306 ± 0.0283   (각도만 s=1: +0.0227 / 진폭·w만 s=1: +0.0428)
```
**"달성 추정 품질 < 수익 임계"가 아니다.** v2 품질의 θ̂을 점-커밋하면 사전등록 bar(+0.03) 수준을 버는데, 실제 파이프라인은 fired에서도 +0.0001에 앉았다 → 가치는 θ̂과 γ_out 사이(48-grid 혼합 + p2 모델평균 + fallback)에서 파괴됐다. **캐비엇 병기**: 파일럿(n=23, reps=2, CI가 0 걸침; 오차 부호 랜덤·파라미터 간 독립 재추첨 = 실제 오차 상관 무시) — 벽 주장을 깨는 용도이지 +0.03 확증이 아니다.

## 5. 프로토콜 위반 두 건
(a) v2는 스스로 **UNQUALIFIED → "detector-limited" 라벨**을 달았다 — 그 라벨의 의미가 정확히 "이 결과로 정보 벽을 선언하지 말라"다. (b) closure doc §6.2의 사전 요구: "지면 **CRB/식별성을 실제로 계산**해 검출기 한계가 아님을 보일 것" — 미이행. 벽 선언의 등록 요건이 충족되지 않았다.

## 정직한 피해 평가 — D1에서 실제로 죽은 것
- `fit_bma_temp`(evidence-온도 채널): 무용 판명(병소가 온도가 아님). 철회.
- "m2 fallback ≈ do-no-harm": fresh에서 m2−cur = −0.026 — 내 등록 문서의 안전 주장은 과장이었다. 수정: backstop = **incumbent γ 그 자체**(그러면 HYBRID가 보여주듯 바닥이 정확히 0).
- **"학습이 필요한가" 자체가 불확실해졌다**: 가장 싼 생존 경로는 무학습 파이프라인 수리다. D1의 학습(JPN)은 그 다음 단계 — fired-subset 회수(+0.0001 → +0.03~0.07)와 TPR 58%의 나머지 33% seas 전환이 남은 상금이고, 그건 정확히 내 q-채널+교정의 표적이다. G-JOINT 게이트(Q3)는 여전히 checkable(이 실험은 map-조건부 사후에 대해 아무것도 측정하지 않았다)이나, **v3 결과 전까지 훈련 보류가 옳다.**

## 제안: e43 "v3-pipeline" — 무학습, ~1 GPU-h, 사전등록
θ̂ = v2 `fit_joint` 그대로. 바꾸는 것 셋뿐: ① fired → **점-커밋**(또는 top-48 대신 top-4 좁은 혼합) ② not-fired → **cur_grid γ 그대로**(fit1 폐기) ③ unimodal 안전은 ①+②로 구조 보장. **평가는 si=200..263 (n=64) — si=100..163은 이 반박 분석으로 오염됐음(fired/subgroup을 봤다), 재사용 금지.** PASS: mean ≥ +0.03, CI 0 배제, unimodal harm ≥ −0.01. FAIL 시: (i) true-θ lattice node의 evidence rank 진단(surface가 flat인지 deceived인지 직접 분리), (ii) §6.2가 요구한 CRB 계산 → 그 둘이 flat+CRB>임계를 보이면 **그때** 벽을 선언하고 나도 D1 사망에 서명한다.
