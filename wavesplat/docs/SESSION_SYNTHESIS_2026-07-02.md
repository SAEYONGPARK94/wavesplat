# 세션 종합 (2026-07-02) — 굳히기 문서

> 하루치 집중 세션의 확정 결론. 여러 개별 문서(SIMULATOR_REALISM·METHODS_PARAMS·REALDATA_FORECAST·
> AMORTIZED_NET_BUG·METHODOLOGY_REVIEW·IMPROVEMENT_PLAN·DIRECTION_DECISION)의 상위 요약.
> ⚠️ 이 세션에서 성급한 "안 된다" 결론이 다수 있었고 검증으로 교정됨 — 아래는 **교정된 최종 결론.**

## 0. 출발 질문
"HOS 비선형이 주도적이지 않은데 추정이 안 맞는 게 데이터/방법 탓인가? 학습이 결과를 내는가?"

## 1. 시뮬레이터 데이터 현실성
- HOS bound harmonic(자잘한 파)은 오차의 3~5%뿐 — **비선형은 주범 아님.**
- **진짜 약점 = 방향 구조**: 고정 ~19° 단봉 Gaussian spread(전 에피소드 동일, 주파수독립, swell+windsea 혼합 없음).
  실측(BS 27°, 일반 wind-sea 30~50°, 극단 short-crested 109°)과 갭. → sea-state별·주파수의존 cos-2s·bimodal 필요.

## 2. 실측 데이터 (Guimarães/Bergamasco stereo, 42GB·13레코드 전사이트, `data/real_bergamasco/`)
- **개구비(aperture-per-wavelength)가 binding 변수**(CRB 일관): La Jument 0.11λ→물리 이점 소멸, BS ~1λ→작동.
- **nowcast 단일프레임 = 물리·naive·학습 모두 정보한계**(RELAX corr episode별 0.09~0.89, sparse선 ~0.2).
- **positive = forecast·외삽**: BS서 물리 forecast가 persistence를 0.5~3s 압도(skill 0.76@1s), unobserved 외삽
  +0.078 (naive 원리적 불가). = **방어가능 헤드라인.**

## 3. 코드·추정기
- **면밀 검토: 결론 바꿀 버그 없음** (RELAX·fit/render 부호·forecast 누수·GP·metrics·data_loader·DPS 전부 clean).
- 하드닝: `scripts/estimator_improved.py`(적응 klim + CV 모델차수), `relax_step`에 klim 파라미터. 합성 헤드라인 회귀 없음.

## 4. 학습 파트 — 대교정 (세션 최대 수확)
- **"학습 collapse/안 됨"은 제 오류였음.** 원인 = 좌표 인코딩 버그 2개:
  (a) 원본 d52~d68 **raw 선형좌표** → 진동 파동 표현 불가(spectral bias),
  (b) 내 FourierFeatures **주파수 스케일 0.3** → 좌표[10,250]m서 유효파장 3m≪obs간격 9m → **aliasing**.
- **수정**(`scripts/amortized_ff.py`, FF를 파동대역 k∈[0.02,0.6] log-spaced 정합): 단일시간 보간 0.004→0.39,
  **전체 파동문제(보간+dispersion) corr 0.004→0.60 학습됨.**
- **Step 2**: learnable FF는 오히려 악화(0.49<0.60, 과적합) → FF튜닝은 레버 아님. **net ~0.6 plateau, RELAX(0.98)
  격차는 아키텍처 문제.**
- **위치**: determined-저성분서 물리(0.98)>학습(0.6). 학습 값어치 = 물리 약한 고성분/복잡해(DPS>RELAX 사례) + 속도·UQ.
- **함의**: d52~d68 amortized 전체(전부 0.25)가 버그 오염 → **fixed-FF로 재실행 필요.**

## 5. 방법론 지도 (prior 유무로 갈림)
- **구조 냄**: 물리solver(RELAX/F1/GP), 생성prior(diffusion DPS — 구조샘플+UQ), 물리내장 학습(fixed-FF net 0.6).
- **mean 붕괴**: black-box 지도-회귀(pixel-MSE) — 라벨+MSE가 under-determined서 mean 최적. (단 버그도 겹쳐 있었음.)
- **pixel-MSE는 틀린 목적**(mean 선호) → obs-fit / 생성 / 구조 목적 써야.

## 6. 개선 레버 (우선순위)
- **Tier 1 (정보 ceiling)**: 다중센서/다중개구 융합 — nowcast 정확도의 유일한 진짜 레버(CRB: 정보=유효개구).
- **Tier 2 (가치)**: forecast·외삽 operability 곡선 + 물리·생성 하이브리드 + 보정 UQ.
- **Tier 3 (현실성)**: 시뮬레이터 방향 다양성.
- ❌ 하면 안 됨: black-box 지도-회귀 더, 생성 prior 더 학습(guidance 튜닝이 레버), 같은 sparse에 영리한 nowcast.

## 7. 메타 교훈 (중대)
**"안 된다" 결론 전에 (a) 격리실험(trivial 하위문제가 되는가) + (b) 하이퍼파라미터·task별 ceiling 점검 + (c)
multi-frame×seed·공정채점 필수.** 이 세션 성급한 단정 다수(단일프레임·잘못된 ceiling·짧은 학습·FF스케일)가 매번
사용자 지적으로 교정됨. 검증 규율이 최대 자산.

## 8. 다음
- **Step 3**: forecast operability 곡선(개구비 vs skill, 다레코드 실측).
- **개선된 학습법 설계**(off-the-shelf 아닌, 경험 기반): **physics-informed amortized** — 파동대역 FF + dispersion-제약
  디코더(성분 예측) + obs-fit/구조 목적 + (선택)불확실성 head. RELAX의 물리 prior를 학습에 내장해 0.6→0.9 목표.

## 9. Step 3 결과 — operability 법칙 (실측 5레코드·4사이트, 확정)
| 레코드 | 개구비 | 분산 | forecast skill | 외삽margin |
|---|---|---|---|---|
| LJ_2018 | 0.19λ | 37° | **−0.02 (실패)** | +0.003 |
| BS_2013 | 0.67λ | 53° | **+0.87** | +0.12 |
| BS_2011 | 1.03λ | 27° | +0.72 | +0.03 |
| AA_2014 | 1.60λ | 37° | +0.79 | +0.05 |
| YS_2017 | 3.44λ | 32° | +0.84 | +0.06 |
- **설계법칙: 개구비 ≥~0.7λ → 물리 forecast가 persistence 압도(skill 0.7~0.9), 4사이트 일관. <0.2λ→실패.**
- **방향분산 무관**(BS_2013 53°서 skill 최고 0.87) → **개구비가 유일 binding 변수**(CRB 일관). 외삽도 물리>naive 전레코드.
- 원래 역설 해소: forecast는 개구비≥0.7λ면 실측서 확실(skill 0.7-0.9), nowcast 단일프레임만 정보한계.
- `results/operability_curve.json`, `results/gifs/operability_curve.png`.

## 10. 다음 — 개선된 학습법 설계 (경험 기반, off-the-shelf 아님)
경험이 준 설계 원칙: (1) **파동대역 FF 필수**(스케일버그 교훈), (2) **dispersion 물리 내장**(RELAX가 이기는 이유),
(3) **pixel-MSE 금지**(mean붕괴)→obs-fit/구조목적, (4) 제대로된 aggregation.
→ **physics-informed amortized**: 파동대역 FF 인코더 + **dispersion-제약 출력**(성분/기저계수, 파동 manifold 위)
+ obs-fit 훈련 → 구조 강제(붕괴 불가) + 물리 solver를 ms로 amortize. FF버그 고친 지금 재시도(이전 amortized_spectral은
buggy FF서 실패). 목표: net 0.6 → RELAX 0.9+ 격차를 물리내장으로 좁힘.
