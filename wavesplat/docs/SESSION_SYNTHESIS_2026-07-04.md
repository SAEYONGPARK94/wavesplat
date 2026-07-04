# 세션 종합 2026-07-04 — Multi-record collapse → 연속-k 초분해 문제 특정

하루 동안 10커밋. thesis pivot의 최종 GO/NO-GO(multi-record D_eff/λ collapse)에서
출발해, 이동 플랫폼 문제의 정체를 "연속-k 초분해 sparse 추정"으로 특정하고 탈출구
후보 2개를 확보한 세션. 아래 순서 = 실제 발견 순서 (각 단계가 앞 단계를 교정).

## 발견의 사슬 (커밋 순)

1. **8224213 — 보편 D_eff/λ 법칙 기각.** 실측 10 record(56윈도우, 292점), 무차원
   정합 프로토콜. BS_2011 앵커는 3번째 독립 재현(+0.31), LJ negative control 적중,
   그러나 supra-0.7λ 6개 record가 유의 음수. 프로토콜 수정 3건이 결론 보호:
   valid-aware track(예산붕괴), robust mask(LJ dead px 26%), per-window demean
   (정적 stereo bias). 지배구조 = corr(gain, A_abs) −0.56: "이동은 고정이
   정보기아일 때만 돈이 된다."
2. **9e1e2fe — starvation predictor 실패 + 식별붕괴 발견.** 9개 물리피처로 LORO
   sign acc 0.67 < 항상음수 0.78 (얕은학습 불가). AA14 드릴다운: RELAX가 이동
   track서 N~4 붕괴 → min_comp 강제성장 반증실험 → **오히려 악화**(+0.13→−0.55)
   = CV는 정직한 방어, 병목은 이동 샘플링 하 성분 식별.
3. **4d52922 — track-aware 재학습 부분회복(+150% 상대), RELAX 미달.** 연산자
   분포 주입이 방향은 맞으나(2.2Tp 레짐) 절대수준 미달.
4. **c10026a — 학습전이 블록별 리뷰.** 9블록 중 7개 학습 대응물 존재, 가치는
   선별적(검출·prior·UQ만), 두 게이트(정보한계·sim2real)는 물리가 지킴.
5. **859f066 — FIM/CRB 1차: "이론은 해로움을 예측 안 함".** 56윈도우 참모드 추출
   (corner-origin 위상버그 수정+회귀게이트) → Bayesian FIM → 이동이득 양수 예측
   vs 관측 RELAX 음수 = 알고리즘 갭. RIDGE는 0 부근.
6. **d83a44c — achievability 3층 분해 (이산 합성).** oracle(참k+LS) 연산자-무관
   0.99 / bound 0.62~0.93 / RELAX 0.27~0.54. 10Tp·2λ 합성이 실측 이동해로움 첫
   재현(스케일 매칭이 열쇠). 이산-모드 상금 ~0.35 corr.
7. **9788bdb — 연속스펙트럼 교정 + colored FIM.** 이산 상금은 well-posed 클래스
   인플레. 올바른 reference = bin-클래스 linear-MMSE ≈ **oracle_ls2(참 top-32 k +
   ridge λ=1.0; smoke의 "oracle 붕괴"는 λ0.1 정규화부족)**. ceiling
   0.28→0.41→0.46(속도별) vs RELAX ~0.2 평탄 → **상금 +0.08/+0.18/+0.28, 속도
   비례 성장**. 실측 colored-residual GLS FIM: 3개 잡음모델 전부 "이론은 해로움
   예측 안 함" 견고.
8. **67a074b — 레짐 net + dense-GLS 통제.** 레짐타깃 net: 단발 실패, +잔차반복
   = 고정 0.19≈RELAX(돌파 재현)하나 이동서 하락. dense-bin GLS(무선택) 실패
   → 당시 결론 "support가 상금".
9. **b17507a — ★support 가설 기각: 상금 = 연속-k 정밀도.** oracle_bins(참
   support + lattice 양자화 k) 0.04/0.12/0.15 붕괴. 물리: 10Tp 장창서
   δω=c_g·δk(반-bin)의 시간 레버암 ~8rad = decoherence. lattice 방법 전부
   (dense·λ-CV·OMP-CV·ARD-SBL·heatmap-격자 학습망) 구조적 ceiling. RELAX가
   lattice를 이기는 유일 이유 = off-grid refine. SBL+refine 하이브리드도 RELAX
   이하(lattice init 국소최소). **문제의 최종 이름: "chirped 이동샘플링 하
   연속-k 초분해 sparse 추정" (NOMP 계열).**
   **DPS field-prior 이동검증: 유일한 이동-강건 양성** — B2(frozen+polish)
   native domain서 이동 열화 없음, sparse 예산서 RELAX 압도, track043·30comp
   unobs corr +0.51 (단 1윈도우).

## 현재 상태 요약 (내일의 출발점)

- **thesis 서사 완성형**: 정보의 벽(개구) → 알고리즘의 벽(식별) → **연속-k
  초분해 문제** → 탈출구 ⓐNOMP류 연속-k 솔버 ⓑDPS field-prior×이동.
- **핵심 수치**: 연속바다 ceiling(oracle_ls2) 0.28/0.41/0.46 (고정/0.43cp/1.0cp),
  RELAX ~0.2 평탄, lattice ceiling(oracle_bins) 0.04~0.15, DPS×이동 nrmse
  0.28~0.45 (RELAX >1.0), 이동 양성 +0.51 (1건).

## 내일 할 일 (소요 추정 포함, 디스코드 합의)

1. **DPS×이동 확정실험** (~2-3h): `d3_sa_dps_moving.py`에 multi-window(3ep×5win)
   루프 추가 → +0.51이 진짜인지. 리스크 낮음, 가치 높음.
2. **NOMP식 연속-k 솔버** (~2-3h): greedy + 성분별 Newton + cyclic re-refine.
   성공기준: oracle_ls2 방향으로 RELAX(0.2) 초과.
3. **승자로 실측 BS/AA 재도전** (~2h): multirecord 하니스 estimator 교체.
   ①②를 병행하면 전체 ~4-5h.

## 주의사항 (내일의 나에게)

- oracle_ls2의 λ=1.0은 PhiᵀPhi 대각(~n/2=200) 대비 0.5% 상대 shrinkage — λ의
  절대/상대 스케일 혼동 금지 (dense_cv에서 λ 그리드 1~3000 이미 검증).
- 합성 achievability 하니스의 seed/조건/픽셀은 `sa_achievability_gap_cont.json`과
  정확히 공유됨 (`d3_sa_regime_net.py`의 evaluate가 재현 예시).
- FIM류 판정은 정성(부호·순서) 수준까지만 신뢰 — 정량 bound는 [colored, nominal]
  브래킷. LJ18n의 window-level λp/cp는 패치 아티팩트(분석 시 lam_ref 재정규화).
- DPS×이동의 RELAX 비교치(nrmse>1.0)는 150obs sparse 예산 탓 — SA 프로토콜
  (400obs)과 직접 비교 금지.

## 산출물 위치

- 실험: `scripts/d3_sa_{multirecord,collapse_analysis,starvation_probe,
  moving_ident_probe,fim_identifiability,achievability_gap,regime_net,
  dps_moving}.py`
- 결과: `results/direction3/sa_*.json` (+`_cont`, `_colored`, `_eff` 변형),
  그림 `sa_multirecord_collapse.png`
- 리뷰: `docs/LEARNING_TRANSFER_REVIEW_2026-07-04.md`
- 메모리: `project_synthetic_aperture_2026-07-03.md` (전 블록 상세)
