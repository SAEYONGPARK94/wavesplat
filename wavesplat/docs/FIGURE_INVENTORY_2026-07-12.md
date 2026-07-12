# 그림·GIF 인벤토리 — 논문/발표 자산 목록 (2026-07-12, 큐레이션)

> 용도별 추천 세트. 전 그림 수치는 커밋된 JSON 재도출분(VERIFICATION_
> 2026-07-12.md 감사 통과). 경로는 `results/direction3/` 기준(별도 표기
> 제외).

## 1. 논문 본문 후보 (순서 = 서사 순)

| # | 파일 | 내용 | 배치 |
|---|---|---|---|
| F1 | **fig_gradient_2026-07-12.png** ★신규 | 오지정 기울기 3무대(−0.041/+0.070/+0.033) — thesis 한 장 | Ch.C 도입 (중심 그림) |
| F2 | fig_ext1_track_match_2026-07-12.png | 실측 3사이트 조우 시계열(ρ 0.97/0.94/0.93) | 서론/결과 헤드라인 |
| F3 | fig_field_verdict_2026-07-12.png | 전장 리드별 곡선+60창 산점(군집 CI) | 전장 판정 절 |
| F4 | fig_ext2_field_vs_track_2026-07-12.png | "자신있게 틀림 vs 겸손하게 맞음" 사례(AA14 w10, 창-특이 명시판) | C.2b |
| F5 | fig_attribution_2026-07-12.png | γ·바닥 가법 기여 분해(4팔×전체/실패창) | C.2 |
| F6 | fig_ext3_aggregate_2026-07-12.png | 8/8 레코드 막대+60창 트랙 산점 | 학습 판정 절 |
| F7 | **fig_d20_control_2026-07-12.png** ★신규 | 실측 리플레이 제어(3레코드×3팔×2τ, t−4.02) | 응용(D-19) 절 |
| F8 | figs_appendix/appendix_synthetic.png | 합성 전장 시트(BF2–BF5, 부호 반전 가시화) | 벤치마크 절 or 부록 |
| F9 | fig_unified_uq.png / uq_calibration_*.png | UQ 보정 | UQ 절 |
| 표 | DRAFT_chAB_tables(10종)·B-10 커버리지 매트릭스 | 수치 표 | Ch.A/B |

## 2. 박사 부록 (전체 공개 세트)

- **appendix_{YS17,AA15a,AA14,AA15b,LJ18g}.png** (v1: footprint 마스킹판)
- **appendix_{...}_v2.png** (v2: 사각형 전장+footprint 윤곽선 — **부록
  기본 추천**, 채점 범위 부제 명시)
- appendix_synthetic.png + appendix_synth_eval.json
- VERIFICATION_2026-07-12.md (재현 감사 원장)

## 3. 발표용 GIF (경로 `results/gifs/`, `gifs_rescue/`)

| 용도 | 파일 | 내용 |
|---|---|---|
| **오프닝 데모** | falcon_stream_demo_22kt.gif (+BF2/BF4s3/BF5 변형) | 설계 작동점 22kt 스트리밍 추정 데모 |
| **실측 예보** | falcon_real_YS17_forecast.gif, falcon_real_AA15b_site4_forecast.gif, falcon_real_LJ18g_swell_forecast.gif | 사이트별 실측 조우예보 |
| **★기제 서사** | gifs_rescue/rescue_AA14_w10.gif | hand 역위상 vs learned 회복(3패널+★조우점) |
| 기제 보조 | gifs_rescue/rescue_LJ18g_w10.gif·w5, rescue_YS17_w11, rescue_LJ18g_w8(동률 대조) | 실패-창 구조 사례들 |
| 나쁜/좋은 창 대비 | falcon_real_YS17_{badwin,goodwin}_forecast.gif | 창 편차 정직 공개 |
| UQ 3제품 | posterior_sample_sea4.gif, masked_forecast_sea4.gif, ladder_track100_sea4.gif, advect_error_sea4.gif | 평균/분산/샘플 체계 |
| HOS/실측 비교 | compare_{linear,hos,real}.gif, static_real_{YS17,LJ18g}.gif | 3무대 시각 대응 |
| (참고) 구세대 | ceiling_BF*.gif, net_vs_RELAX*, DvsB_* | 과거 아크 — 발표 비추천 |

## 4. 개념도·사진 (docs/)

- hydrofoil_controller_flow.png — 제어 절 다이어그램
- sea_trial_lidar.png / sea_trial_cam.png — 실물 센서 사진(동기 슬라이드)
- lidar_input_3d.png (results/direction3/) — 관측 점군 3D
- wavesim_BF4_5c_wave_lidar.png 등 — 시뮬 개요

## 5. 부족분 (제작 대기 — 필요 시)

- 파이프라인 블록 다이어그램(관측→beamform→γ-CNN→은행→MMSE→s2 수축)
  — 발표자료 에이전트가 슬라이드용 텍스트 도식으로 우선 대체, 벡터
  그림은 집필 시.
- 커버리지 매트릭스의 그림판(현재 표) — 선택.
- 속도/방향 불변 패널(a3/a2 JSON에서 1장 통합) — 선택(표로 충분).

## 6. 인용 주의 (그림 캡션 규율)

- F1 기울기의 깨끗한-합성 표본은 35에피소드(부록 6 중 비-BF2 5 + dir180
  30) — 이전 언급 "−0.015"는 5에피소드 시트 기준, **통합 −0.041**로 인용.
- F4는 창-특이 사례임을 캡션에 유지(60창 평균은 learned 전 리드 우위).
- "실측 검증" 용어는 B-7 정의 문단과 함께만.
