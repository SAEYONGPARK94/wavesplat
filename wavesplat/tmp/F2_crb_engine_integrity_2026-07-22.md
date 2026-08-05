# F-2 — "하나의 CRB 엔진" 무결성 점검 (code inspection + spot-check, 2026-07-22)

> 담당: fable (coordinator 지시, 분석 트랙). 대상: `scripts/d1_crb.py`(성분급 백본) / `tmp/e51_crb.py`
> (파티션 θ-CRB, keystone③/GUARD2) / `tmp/e58_envelope_validation.py`(§8 envelope). 방법 = 코드 대조
> (import 그래프, FIM 구성식, noise 모델) + 1개 에피소드 스팟체크. 코드 수정 없음, read-only.

## 결론 (요약)
**"하나의 엔진"은 §4(envelope)와 백본(`d1_crb.py`) 사이에서는 코드-급으로 참이다**(e58이 d1_crb.py 함수를
직접 import·재사용, import 무공유 아님). **그러나 §5(파티션 θ-CRB, keystone③·GUARD2·"결과 B")는 완전히
별개 구현이다** — 다른 파라미터화, 다른 FIM 공식, 다른 noise 모델, **import 교집합 0**. 논문의 "하나의 엔진이
진단과 envelope 둘 다 구동" 서술은 **부분적으로만 참** — 정정 권고를 §"권고"에 명기.

## 1. 세 경로의 실제 계산물

| | `scripts/d1_crb.py` (백본) | `tmp/e58_envelope_validation.py` (§4 envelope) | `tmp/e51_crb.py` (§5 파티션 θ-CRB) |
|---|---|---|---|
| **무엇을 묻나** | 개별 성분(진폭·kx·ky·위상)을 잡음 관측점에서 얼마나 잘 복원하나 | 위와 동일(재사용) + 필드값 재구성 오차 | 레짐/스펙트럼-형태 파라미터(d1,Tp1,d2,Tp2,w)를 얼마나 식별하나 |
| **모델** | z=μ(θ)+ε, ε~N(0,σ²) i.i.d. (평균-파라미터 회귀) | **동일**(d1_crb 재사용) | z~N(0,C(θ)) (공분산-파라미터, zero-mean Gaussian process) |
| **FIM 공식** | F=(1/σ²)JᵀJ | **동일**(d1_crb의 `fisher()` 직접 호출) | F_kl = ½tr(C⁻¹∂_kC·C⁻¹∂_lC) (완전히 다른 공식족) |
| **noise 항** | σ=0.02 (물리 상수, LiDAR 2cm spec) | **σ=0.02 동일 상수**(하드코딩 재확인) | (smult·σ₀)², smult=3.0 (JointEngine **모델 하이퍼파라미터**, 물리 spec 아님); σ₀=nfrac·Hs (**Hs에 비례하는 잡음**, 고정 상수 아님) |
| **θ의 의미** | 결정론적 개별 파동성분의 (w,kx,ky,φ) | 동일 | 스펙트럼 **혼합family**의 형태 파라미터(2-lobe 방향·주기·가중치) |
| **import** | (독립) | `from scripts.d1_crb import jacobian, fisher, predict, omega_of_k` | `from e29_gennet_C / e37_bimodal_ladder / e42_detector_v2 / e41_regime_prize_ladder / e49_open_arena_blind / d3_sa_achievability_gap` — **d1_crb.py 미참조** |

## 2. 스팟체크 (BF4_15comp_seed0_slope.h5, d1_crb.py `run_episode` 직접 재실행)
```
N comps 15, Hs 1.2, Tp 7.0
  t_obs= 0.5s n_obs=  326  dir_crb(우세성분, median)= 0.323 deg  field_nrmse_crb=0.049
  t_obs=30.0s n_obs=15015  dir_crb(우세성분, median)= 0.032 deg  field_nrmse_crb=0.002
```
d1_crb 백본의 "방향 CRB"는 **개별 성분**(결정론적 사인파 하나)의 전파각을 수천 개 잡음 관측점에서 역산하는
초-과결정 문제라 CRB가 **0.03~0.3도** 수준(사실상 완전식별). 반면 e51의 d1(레짐 우세방향) CRB는 **~11도**
(THESIS_FRAMING·e50 인용). **두 수치는 자릿수가 다르고 비교 불가능** — 전자는 "이미 아는 개수의 사인파를
정확히 어디에 놓을까"이고 후자는 "어떤 스펙트럼-형태 family에서 이 관측이 나왔을까"이다. 스팟체크가 이
질적 차이를 수치로 재확인한다: **엄밀 동일 CRB로 서술하면 오해를 부른다.**

## 3. 판정
- **MEASURED**: §4(envelope)와 Direction-I 백본은 코드-공유로 진짜 하나의 엔진(동일 FIM, 동일 σ=0.02, 동일
  `jacobian/fisher/predict` 함수 재사용 — import 문으로 확인).
- **MEASURED**: §5(파티션 θ-CRB, E-51/52, "결과 B" 판별 근거)는 **다른 엔진**이다 — 다른 FIM 공식(공분산
  파라미터화 vs 평균 파라미터화), 다른 noise 모델(smult·Hs-비례 vs 고정 물리 σ), import 교집합 0, 스팟체크로
  수치 자릿수도 무관함을 확인.
- **관찰(참고, 인과 주장 아님)**: e51의 noise-floor 하이퍼파라미터 이름이 `smult`이고 값=3.0(고정) — 이는
  E60K/B-2 실험이 손실 함수별로 학습해낸 바로 그 `smult`(T≈3.24 / L≈1.81 / N≈1.89)와 **이름이 같은 양**이다.
  e51_crb.py는 이를 상수로 고정해 사용하며 손실-종속 민감도를 테스트하지 않는다. 두 결과(B-2, keystone③)가
  우연히 같은 모델 하이퍼파라미터를 다른 목적으로 다루고 있다는 점만 기록 — 추가 주장 없음.

## 4. 권고 (문서 수정)
PAPER_SKELETON §4/§5 서술 "같은 결과수치 전부 재사용, headline만 진단→방법 flip"은 **§4 자체**(envelope)에는
정확하나, §4↔§5 사이를 "하나의 엔진"으로 뭉뚱그리면 부정확하다. 권장 재서술:
> "두 개의 개별적인 Cramér–Rao 계측기(공통 논리·서로 다른 파라미터화)를 쓴다: (i) **field-reconstruction
> CRB**(성분급, §4 envelope의 엔진, Direction-I 백본과 코드 공유) — 알려진 성분 개수의 관측-잡음 한계.
> (ii) **regime-partition CRB**(§5 채널 판별의 엔진, E-51/52) — 스펙트럼-형태 identifiability 한계, 별도
> 구현. 두 계측기 모두 CRB 논리를 쓰지만 수치는 직접 비교 불가하며 서로 다른 질문에 답한다."
**리스크가 크지 않은 이유**: 두 CRB 모두 정직하게 유도·감사됐고(E-51/52 GUARD1/2, d1_crb gradcheck), 각자
스코프 안에서 결론은 유효하다. **리스크가 있는 부분**: "하나의 엔진"이라는 통합 서사가 리뷰어에게 "같은
계산을 재사용해 파슬리하게 확장했다"는 인상을 주면, 실제로는 "두 개의 독립 CRB 유도"임이 드러날 때 통합
주장 자체보다 더 큰 신뢰 손상(불필요한 리스크)이 될 수 있음 — 재서술로 선제 방어 권장.

## 캐비엇
- 스팟체크는 정성적 일관성 확인이지 정량적 등가성 증명이 아님(애초에 등가일 이유가 없는 두 다른 CRB).
- e51의 noise 모델(smult·σ₀, σ₀∝Hs)이 물리적으로 정당한지 자체는 본 점검 범위 밖(E-51/52가 이미 별도로
  검증·감사됨, 재소송 아님).
- 완전한 수학적 통합(예: 두 CRB를 같은 상위-모델의 다른 marginal/projection으로 유도)은 시도하지 않음 —
  가능할 수 있으나 새 유도가 필요해 본 점검(코드 무결성) 범위를 벗어남.
