# 재현성 원장 — 2026-07-06 주어진-센서 검증 사이클
(논문 부록 초안. 모든 헤드라인 수치 ↔ 결과 JSON ↔ 생성 스크립트 매핑.
전 수치는 세션 중 JSON 재도출로 팩트체크됨. 원장 서사 =
`docs/SESSION_SYNTHESIS_2026-07-06.md`, 논문 구조 =
`docs/PAPER_OUTLINE_2026-07-06.md`.)

| # | 수치 (본문 주장) | 결과 파일 (results/direction3/) | 스크립트 (scripts/) |
|---|---|---|---|
| 1 | 네이티브 벤치: 2/4s 예보=nowcast, hand .534/net .495 | `falcon_native_bench.json` | `d3_falcon_native_bench.py` |
| 2 | 실측 조우예보 0-10s: YS17 0.63-0.71 / AA15a 0.60-0.71 | `falcon_real_operator_{YS17,AA15a}.json` | `d3_falcon_real_operator.py` |
| 3 | U-스윕(실측): 0.62/0.70/0.66 | `falcon_real_operator_YS17_U{1,3}.json` | 〃 `--U` |
| 4 | HOS 재현 0.94-0.96 (U=3) | `falcon_hos_probe.json` | `d3_falcon_hos_probe.py` |
| 5 | 설계점(22kt BF4) FF대역 0.94-0.95 | `falcon_hos_probe_design22kt.json` (n12 확장판 `_design22kt_n12`) | 〃 |
| 6 | 강건성: 포즈 .96-.92/차폐 .96/사선 .99-.95/BF2 .90/BF5 .95 | `falcon_hos_probe_{pose*,shadow3m,head*,BF2,BF5}.json` | 〃 |
| 7 | 스트리밍 등가 7e-14, 배치-600 상회 | `stream_seqmmse.json` | `d3_stream_seqmmse.py` (+ `wavesplat/tests/test_stream_seqmmse.py`) |
| 8 | 지연 통일: 설계점 18.5/30.8ms, 실측 17.6/37.1ms (단독·48pt) | `stream_design_pipeline_{hos,ys17}_m48solo.json` | `d3_stream_design_pipeline.py` |
| 9 | 스트리밍 파이프라인: 설계점 0.90-0.93 / 실측 0.75-0.78 | `stream_design_pipeline_{hos,ys17}[_m*].json` | 〃 |
| 10 | 10분 무드리프트 | `stream_longrun.json` | `d3_stream_longrun.py` |
| 11 | 나쁜창 마커: bimod −0.84(p=.001), 선상 −0.78(p=.003) | `badwin_forensics.json`, `onboard_bimod_gate.json` | `d3_badwin_forensics.py` |
| 12 | bimod 홀드아웃 실패 +0.13 / 이봉 재현 실패 | `onboard_bimod_gate_AA15a.json`, `bimodal_probe.json` | `d3_bimodal_probe.py` |
| 13 | 게이트 4종 비교(σ/val-res/NIS/bimod) | `uq_gating_real.json`, `val_gate_probe.json`, `net_uq_gate.json` | `d3_uq_gating.py`, `d3_val_gate_probe.py`, `d3_net_uq_gate.py` |
| 14 | ★사이트-적응 게이트: 3레코드 −0.38(p=.021), 0.68/0.30 | `val_gate_probe{,_AA14}.json` + 인라인 계산(세션 ⑬) | `d3_val_gate_probe.py --records AA14` |
| 15 | 게이트 운용곡선: 0.49→0.68@50%/0.73@30% | `gate_roc.json`, `figs/gate_roc.png` | (세션 인라인; 곡선 재생성 `d3_paper_figs.py`) |
| 16 | 게이트 인라인 실증(0.8s 슬로우루프, 15ms 고속 불변) | `stream_design_pipeline_ys17_gateinline.json` | `d3_stream_design_pipeline.py --gate` |
| 17 | E2E: GT-FF 6-16%→튜닝 2배(preview 0.2s), Stream ~20% | `e2e_stream_ff.json`, `e2e_velff.json`, `e2e_ctrl_tune.json` | `d3_e2e_stream_ff.py`, `d3_e2e_velff.py`, `d3_e2e_ctrl_tune.py` |
| 18 | 기울기 진단 .68/oracle 갭 +.06/틸트 무효 | `slope_skill_probe*.json`, `slope_oracle_gap*.json` | `d3_slope_skill_probe.py`, `d3_slope_oracle_gap.py` |
| 19 | 구조화 샘플링 +0.13(합성) / 실제-경향 정책 역효과 / 예산 +0.04/4× | `slope_oracle_gap_lines.json`, `adjacency_real_test.json`, `budget_slope_test.json` | `d3_slope_oracle_gap.py --sampling lines`, `d3_adjacency_real_test.py`, `d3_budget_slope_test.py` |
| 20 | bound-wave 무효 | `stokes2_probe.json` | `d3_stokes2_probe.py` |
| 21 | net: v3 parity → v6fan 0.609 → v7fan 0.595(나쁜셀 0.214) | `v{5,6,7}fan_real_eval.json`, `prior_net_v{5,6,7}*.pt` | `d3_prior_net_data.py`(fan/bimodal 증강), `d3_prior_net_taskft.py`, `d3_netfan_eval.py` |
| 22 | 선체융합 기각(나쁜창 +0.11, 불안정) | `hull_value_test.json` | `d3_hull_value_test.py` |
| 23 | 해류/노이즈/스프레드 기각 | `gap_dissect_{current,noisefloor}.json` | `d3_gap_dissect.py` |
| 24 | 그림 3종 + GIF 9종 | `results/figs/*.{png,pdf}`, `results/gifs/*.gif` | `d3_paper_figs.py`, `d3_*_gif.py` |

주: 실측 레코드 특성/윈도우 시드는 각 스크립트에 하드코딩된 규약
(`rng(700+wi)`, i0 = linspace 12분할, W_max 40 s)으로 완전 결정적.
