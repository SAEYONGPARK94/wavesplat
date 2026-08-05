# ① under-det 알고리즘 갭 = **파수 식별(wavenumber ID)로 국소화 · 학습 헤드룸 실재** (2026-07-22)

> 스크립트 `tmp/gap_ladder.py`, 원자료 `results/direction3/gap_ladder.json`. under-det 바다(15/30comp),
> sparse obs(250점), synthetic(predict(θ)+noise). [[project_methods_landscape_2026-07-22]] fable 리뷰 검증.

## 결과 (achieved 재구성 corr median, n=16, higher=better)
| 추정기 | corr | 해석 |
|---|---|---|
| ridge (dense 선형, sparse ID 없음) | 0.124 | 최악 = sparse ID가 관건 |
| RELAX (greedy 파수탐색) | 0.390 | 고전 |
| **net-MMSE (학습 prior, beamform kp)** | **0.552** | 현 최선 |
| **oracle-LS (참 파수)** | **0.999** | 완벽-ID 천장 |
| field-CRB 천장 | 0.998 | 정보 천장 ≈ 완벽 |

- **net-MMSE − RELAX = +0.162** (학습 이득, 재확인 — 학습이 RELAX 이김).
- **oracle-LS − net-MMSE = +0.448** (완벽-ID까지 헤드룸).
- **CRB 천장 ≈ oracle-LS ≈ 1.0**: **참 파수를 주면 재구성 완벽** ⟹ **갭 전체가 파수 식별(ID)**, 필드-결정성 아님.

## ★판정 = 갭은 wavenumber ID · 학습 헤드룸 실재하나 realizability 미해결
1. **국소화**: under-det "알고리즘 갭"은 **파수 식별**(비볼록 탐색)이다 — 참 파수면 완벽, sparse aperture서 15개
   파수를 찾는 게 병목. (ridge<RELAX<net 순서가 "sparse ID 능력"과 정합.)
2. **학습이 이기는 곳 확증**: net-MMSE가 RELAX보다 +0.16, 완벽-ID까지 +0.45 헤드룸 ⟹ **더 나은 학습 식별기가
   더 먹을 여지 측정됨.** B-2/궤적과 **달리 oracle-의존 아님**(참 파수는 천장정의용, net-MMSE는 beamform=비oracle).
3. **⚠ realizability 미해결(핵심 캐비엇)**: field-CRB(θ_true)이 작음(천장~1) = **국소적으론 파수 식별가능(정보 있음)** ⟹
   헤드룸은 **알고리즘적(전역 탐색)**으로 보이나, sparse aperture가 **다봉 우도(여러 파수집합이 동등 적합=aliasing)**
   면 전역-ID가 **정보-한계**일 수 있음(SBI multimodal 힌트 있었으나 미보정이라 미확정). ⟹ **파수-ID CRB(전역/random-phase)**
   가 있어야 "realizable vs info-limited" 판별 — 다음 관문.
4. **⚠ synthetic 낙관**: 완벽-ID=완벽재구성은 obs=model이라서. 실측/HOS(model mismatch)면 천장<1 ⟹ 실현 헤드룸 축소.

## ⟹ 이 세션 종합서의 위치
- **오늘 유일하게 살아있는 "학습이 이기는" 축 = under-det 파수 ID.** oracle-의존 아님(B-2·궤적과 대조). net-MMSE가
  이미 일부 먹고 +0.45 여지. **단 realizability = 파수-ID CRB 미납**(다음). = fable 리뷰 "under-det 알고리즘 갭"을
  **국소화+정량화+캐비엇**으로 정련.
- **정직 권고 불변**: 학습 더 밀면 **여기(더 나은 학습 파수식별기)** 가 유일 승산축, 단 realizability 먼저 판별.
  안 밀면 spine A 논문화. (다음 값싼 관문 = 파수-ID CRB로 headroom 실현가능성 시험.)
