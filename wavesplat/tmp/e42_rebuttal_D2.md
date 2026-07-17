# E-42 D2 반박 — kill은 스코프 초과다: 죽은 것은 A5(evidence 채널)이지 D2의 클래스가 아니다
**Designer 2. 2026-07-17. 프로브: `tmp/e42_rebuttal_probe_D2.py` → `tmp/e42_rebuttal_probe_D2.json`
(fit 재적합 n=16 si=100..115 + value-curve n=64 si=100..163, 전부 no-training, ~80 s).**

## 0. 판정 요약
- **당신의 두 수정(v2 global d1, mixture-of-fields arm)은 옳았고 감사하다.** 그리고 당신의
  attack #1 제안이 정답 경로였다 — 다만 반대 극에서: ESS≈1이 아니라 **ESS≈667의 단일 blob**.
- **죽은 것**: 이 v2 lattice 파이프라인, 그리고 내 설계의 **β·L(evidence 가중) 채널** — 이건
  concede한다(아래 §4). **안 죽은 것**: θ-posterior + **calibrated commitment** 클래스. 그
  클래스는 이 런에서 **한 번도 instantiate되지 않았다**.

## 1. 클래스 판별식(fit_bmaF − fit_bma = −0.0007)은 degeneracy에 의한 null이다
n=16 재적합: 2-lobe lattice posterior의 ESS **median 667**(27–1747), 그러나 retained top-48
(KEEP_HYP) 질량 중 argmax d2에서 30° 이상 떨어진 비율 **far median 0.00**. 즉 ~13k 가설 중
48개 truncation이 **argmax 이웃 하나의 unimodal blob**만 남긴다. 단일 blob 위에서
mixture-of-fields ≡ grid-BMA는 항등에 가깝다(국소 선형) — 내 프로브에서도 commit_pt − bma
= −0.0014 ± 0.0082. **"mixture-of-fields는 실증적으로 무가치" 결론은 클래스를 시험한 적이
없는 비교다.** 클래스가 갈리는 조건(서로 다른 joint mode 간 데이터-의존 경쟁)은 KEEP_HYP
truncation이 구조적으로 제거했다.

## 2. "exact likelihood ⟹ 어떤 학습망도 못 이김" — 전제가 구현에서 거짓
- 엔진은 **smult=3**: 참 노이즈 분산의 **9배**로 evidence를 계산한다(nfrac 중앙값에서 z 분산의
  ~30–40%를 백색잡음으로 오귀속). 이것은 THE posterior가 아니라 **tempered surrogate**다.
- 그럼 "정확한" smult=1로 하면? **더 나빠진다**: d2e median 16.1→36.7°, d1e 9.7→24.1°
  (n=16) — realization speckle을 fitting하는 +0.419 병리의 정량 재현. 노이즈 스케일 노브
  하나에 답이 median 20° 요동하는 객체는 posterior가 아니다. 남은 부정확성: bank k-양자화
  (70 s에서 δω·T≈1.2 rad decoherence — E-29가 확립한 아티팩트 축), deposit quadrature,
  10°/0.1 lattice measure, Gaussian-amplitude surrogate.
- 따라서 "net은 그것을 근사할 수 있을 뿐"은 성립하지 않는다: NLL 학습은 **참 generator의
  joint p(θ,z)에서 직접 표본**을 얻으며 위 surrogate 중 어느 것도 필요로 하지 않는다. 참고로
  amplitude가 결정적(비-Gaussian)인 점도 형식적으로 exactness를 깨지만 CLT로 약하다 — 주장의
  하중은 smult 요동에 있다.

## 3. −0.017의 원인 분해 — 정보 부족도, 정밀도 부족도 아니다
**Value curve (n=64, oracle 구조에 방향 오차만 주입):**
```
j0_15 (d2 15°)          : +0.0658 ± 0.0143
j20_30 (d1 20°, d2 30°) : +0.0435 ± 0.0170   ← blind fit의 달성 정밀도보다 나쁜 오차
blind fit 달성 정밀도    : d1e median 16.1°, d2e median 17.5° (n=64 scored rows)
commit_pt (미채점이던 arm): −0.0123 ± 0.0204 (8/16)
```
**"PERM at the family level"이 아니다** — 전형 정밀도의 commitment는 +0.04를 번다. 손실은
두 개의 **꼬리**다:
1. d2 blunder 꼬리: e_d2_err **>30°가 24/64, >60°가 9/64** (mean 27.9 vs median 17.5);
   corr(d2e, commit-gain) = **−0.40**.
2. family-상속 꼬리: **14/64 바다에서 family-oracle 자체가 cur에 짐**(그 바다 평균 −0.077) —
   fit 품질과 무관하게 commit이 상속.
**perfect-gate commit(해로운 바다만 기권) = +0.0289 ± 0.0093 (n=16).** 즉 net-양수 상금이
blind 파라미터 안에 이미 들어 있고, 그것을 전부 되뱉게 만든 것은 **abstention 메커니즘의
부재**다. 파이프라인의 유일한 게이트 dlogE는 commit-gate로서 무가치하다: fired-gate commit
= −0.0022, corr(fired, gain) = +0.12 — dlogE는 "2시스템인가"를 재지 "commit이 incumbent를
이기는가"를 재지 않는다. 네 수치(commit −0.012 / j20_30 +0.044 / perfect-gate +0.029 /
oracle +0.097)를 동시에 설명하는 유일한 독법 = **calibration이 상금의 병목** — D2 설계
문서의 F1이 예언한 바로 그 구조이고, 그 축(q-calibrated commitment + anchor fallback)은
이 런에 존재하지 않았다.

## 4. K3는 발화하지 않는다 + 정직한 concession
K3 = **AMP(학습됨) vs A5** 비교인데 이 런은 A5를 두 번(fit_bmaF, U_BMA) instantiate했고
AMP를 0번 했다. fit_bmaF − U_BMA = −0.0009가 실제로 보여준 것: **A5 내부에서 evidence
가중이 evidence 선별 위에 더하는 것이 없다.** Concede: 내 스펙 §3도 smult=3 evidence를
그대로 이식했으므로, 내 설계의 β·L 채널은 이 결과로 **사망이 맞다**(β→0로 붕괴했을 것).
살아남는 것은 q^α 채널 하나로 좁아진 설계다: **q(θ|maps, NLL)로 제안·commit하고, 보정된
게이트로 기권하는 것.** 상한도 정직하게: perfect-gate +0.029 + lattice(10°/0.1) 초과 정밀도
헤드룸(j-curve상 +0.04~0.066)으로 **현실적 기대 +0.03~0.05** — +0.098 전액이 아니다.
사전등록 +0.04 bar 기준으로 생존은 **한계선상**이다.

## 5. 생존을 판정할 단일 저비용 측정 (no training, ~1 h, n=64 동일 바다)
**`commit_cal`**: commit_pt에 **blind value-comparison 게이트** 하나만 얹는다 — obs 400점을
leg-split(기존 `split_masks`)하여 300점으로 committed γ와 cur γ 각각 MMSE를 풀고, held-out
100점 예측오차가 committed 쪽이 좋을 때만 commit, 아니면 cur 유지. (dlogE와 달리 이 게이트는
정확히 "commit이 이기는가"를 잰다; GT 불사용; 바다당 solve 2회 추가.)
- **commit_cal − cur ≥ +0.02, 95% CI 0 배제 (n=64)** → calibration 논제가 무학습으로 증명,
  D2의 잔여 클레임(q가 제안·정밀도·게이트를 개선)이 학습 라이선스를 얻는다.
- **≈ 0 또는 CI가 0 포함** → 기권이 blind로 학습 불가능하다는 뜻이고, 그때는 D2가 진짜로
  죽었다 — 내가 서명한다.
