# 철회 작업 진행 로그 (2026-08-14)

권한: 편집정지 **부분 해제** — 철회/정정/⛔표시/범위문장만. 새 주장·새 수치 금지.

## 사전 재확인 (직접 재도출, 런 0회)
- `wavesim/gt_generator.py:38` = `_, _, Z, NX, NY, NZ = single_gerstner(...)` → 수평 변위 2성분 폐기. **직접 확인.**
- `wavesim/pipeline.py:157,165` 가 GT 필드와 레이캐스트 메시 **둘 다** `evaluate_field` 로 만든다. **직접 확인.**
- `wavesim/pipeline.py:44 run_episode(...)` 에 **선속 인자 없음**, `direction_deg: float = 0.0`. **직접 확인.**
- `scripts/d3_h6_v2_e2e.py:14-16` = *"direction_deg = 0 (waves travel +x, AWAY from the sensor) — the anti-causal configuration for encounter"*. **직접 확인.**
- `results/b1_full_comparison.json` 재집계: WS `avg_nRMSE` 0.2229 / ETKF 0.3189 / GPR-1 0.2610,
  `avg_inference_ms` 13.376 / 132.639 / 284.457, `n_frames` 평균 WS **167** vs ETKF·GPR **301**. **직접 재도출.**
  ⟹ 학위논문 표 `tab:main_results`·`tab:bybf` 와 지연 표가 **전부 이 파일 산출**임을 확인.
- `results/ablation_bf45.json` 재집계 (150 짝): Full 0.2662 · A1 0.2412 · A2 **0.2425**,
  **A2 가 Full 보다 좋은 짝이 134/150**. ⟹ `:2158` 의 A1 단독 인용은 선택 인용. **직접 재도출.**

## 진행
- [x] `thesis/thesis_main.tex`
- [x] `paper/wavesplat.tex`
- [x] `paper/cover_letter.tex`
- [x] `paper/submission/cover_letter.tex`
- [x] `paper/submission/graphical_abstract.md`
- [x] `hydrofoil/paper/hydrofoil_feedforward.tex`

## 완료 (2026-08-14)
6개 파일 · 정정 지점 **64곳**. 상세 표 = `RETRACTION_2026-08-14.md`.
LaTeX 정적 점검 5/5 통과(환경 짝·`$` 짝·중괄호 순증감 0·빈 리스트 0).
⚠**컴파일 불가** — pdflatex/xelatex/lualatex/latexmk 가 이 환경에 없다.
⚠`.tex` 에는 이모지 0개(kotex 폰트 리스크) — 마커는 `[철회]`/`[RETRACTED]`/`[범위]`/`[SCOPE]`.
