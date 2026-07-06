# 개선 계획 — 무엇을 하면 결과가 나아지는가 (2026-07-02 세션 종합)

> 이번 세션의 모든 실증에서 도출한 **우선순위 개선 레버**. 핵심 통찰: **nowcast 정확도는 정보한계라
> 알고리즘으로 못 넘는다(RELAX=ceiling, 모든 방법이 거기 수렴). 개선 레버는 정보량(개구/센서)과, 가치가
> 실제로 있는 곳(forecast·UQ·operability)이다.**

## 하면 안 되는 것 (개선 안 됨 — 이번 세션서 확인)
- ❌ black-box 지도-회귀(pixel-MSE) 더 시도 → mean 붕괴 (ConvCNP·NP·transformer·내 4변형 전부)
- ❌ 생성 prior 더/크게 재학습 → 개선 안 됨 (260ep로 충분, 600ep·3배모델 오히려 약간 나빠짐)
- ❌ 같은 sparse 데이터에 더 영리한 nowcast 알고리즘 → 정보한계 불변

## Tier 1 — 정보 ceiling을 올린다 (nowcast 정확도의 유일한 진짜 레버)
CRB 증명: 정보 = 유효개구. RELAX가 ceiling이고 모두 거기 수렴하므로, **정확도를 올리려면 관측 정보를 늘려야.**
1. **다중센서/다중개구 융합**: LiDAR + X-band radar + stereo, 또는 다중 LiDAR 유닛, 또는 이동플랫폼 합성개구.
   - 검증실험: 2~3 센서 융합 시 corr 0.2→0.5+ 오르는가? (개구비-corr 곡선)
   - repo 관련: 이미 fusion/multi-aperture 언급(direction3 doctoral review §2).
2. **능동/적응 센싱**: predictable-zone로 LiDAR 조준해 정보 최대화 (d40/d41 기반). 한계적이나 실재.

## Tier 2 — 가치가 있는 곳을 강화한다 (방어가능한 기여)
3. **forecast·외삽을 다레코드 실측으로 확정 + operability 곡선**: A 다운로드분(BS·AA·YS 다레코드)으로 B 확장 →
   **개구비 vs forecast-skill 곡선**(설계법칙). BS서 이미 positive(skill 0.76@1s, 외삽 +0.078) — 일반화 확정.
4. **물리 + 생성 하이브리드**: DPS 구조 샘플 + dispersion 물리 forecast + 보정 UQ. 학습(구조/UQ)과 물리(forecast/
   정확도)의 best-of-both. + guidance 튜닝(0.1 최적) 반영.
5. **보정 UQ로 "식별 불가(개구비 부족)" 케이스 flag**: 실무 배치 가치(언제 믿으면 안 되는지). La Jument형 검출.

## Tier 3 — 데이터 현실성 (결과 전이성)
6. **시뮬레이터 방향 다양성**: 고정 ~19° spread → sea-state별 sampling·주파수의존 cos-2s·bimodal(swell+windsea).
   현재 최대 약점(비선형 아님). 이래야 합성 결과가 실측으로 전이.
7. **다사이트 실측 검증**: A 다운로드분(BS·AA·YS·LJ)로 sea-state 다양성 확보.

## 학습법 개선 (지도/비지도 각각)
- **지도**: pixel-MSE 버리고 **구조보존 목적**(성분예측+순열불변/스펙트럼 loss, adversarial) → 붕괴 회피. 또는
  **deep-unroll 고정λ**(=learned F1, 구조 보장). 미검증 후보.
- **비지도**: diffusion(검증됨) + guidance 튜닝. 대안 VAE/flow. **정확도 아닌 구조·UQ가 목적**임을 명시.

## 권장 실행 순서
1. **Tier 1 검증실험(다중개구 융합)** — 정보 ceiling이 실제로 오르는지 (이게 되면 nowcast도 개선, 판 바뀜).
2. **Tier 2-3(forecast operability 곡선 + 실측 다레코드)** — 방어선 확정 (A 다운로드 완료 시 즉시).
3. 시뮬레이터 방향 다양성(전이성) 병행.
→ **한 줄 요약: 알고리즘이 아니라 (a) 센싱 정보(개구/융합)를 늘리고 (b) 가치가 있는 forecast·UQ·operability를
   강화하라. nowcast 정확도 경쟁은 정보한계라 소모전.**
