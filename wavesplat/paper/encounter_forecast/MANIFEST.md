# Submission package manifest — encounter forecasting paper (OE)
(2026-07-06 조립. 이 폴더가 투고에 필요한 전부. 수치 원본은 레포의
results/direction3/, 실행형 검증은 supplementary/d3_final_verification.py)

## 원고
- `encounter_forecast.tex` — OE elsarticle 골격 (그림 배선: figures/)
- `supplementary/docs/DRAFT_main_paper_v0.md` — **본문 원고 (v2 산문,
  §1-8 완문)** — tex 이식의 원천. 최신 정정(ka 0.027-0.035, 리턴 48-51,
  추파, Al-Ani 원문 수치) 반영본
- `cover_letter_draft.md` — 커버레터 (WaveSplat 방법논문 구분 단락 포함)
- `highlights.txt` — OE 필수 하이라이트 5개

## 그림 (figures/, PDF=투고용 / PNG=검토용)
1. `graphical_abstract` — OE graphical abstract (thread 질문 + 핵심 4수치)
2. `fig_concept` — 기하·정보지평 개념도 (§1-2)
3. `fig_horizon` — 설계점 skill vs lead + ℓ* (n=12) (§2)
4. `fig_robust` — FF-대역 강건성 바 (§4) ※추파 180° 추가는 이식 시
5. `fig_sites` — 실측 5사이트 + 창 이질성 (§4)
6. `fig_gate_roc` — 게이트 운용곡선 (§5, 'exploratory' 라벨)
7. `fig_e2e` — 제어 폐쇄 막대 (§6)

## 보충자료 (supplementary/)
- `gifs/` 11종 — 목록·용도는 `REPRODUCIBILITY_README.md` 참조
- `docs/REPRODUCIBILITY_2026-07-06.md` — 수치↔JSON↔스크립트 24항 원장
- `docs/GATE_CONFIRMATORY{,2}_prereg_2026-07-06.md` — 사전등록 원장
  (커밋 타임스탬프: c76994e, 23d45e5 — 실행 전 고정 증빙)
- `d3_final_verification.py` — 한 줄 전체 검증 (31/31)
- `REPRODUCIBILITY_README.md` — 패키지 안내

## 이식 시 잔여 작업 (tex)
- [ ] markdown §1-8 → tex 본문 이식 (+§5 재구성: 법의학 주, 게이트 종절)
- [ ] references.bib 정식화 (초안 인용 목록 12종)
- [ ] fig_robust에 추파(180°, FF 0.95) 막대 추가 재렌더
- [ ] 저자/소속/사사 기입 (Saeyong Park, Seoul National University)
