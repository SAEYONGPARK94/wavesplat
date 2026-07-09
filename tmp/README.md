# tmp/ — 레포 내 scratch 작업 폴더

시스템 `/tmp`(세션 종료 시 삭제) 대신 여기서 임시/분석 작업을 진행한다.
(이전 세션에서 scratchpad에 둔 실측 파일럿 데이터·스크립트가 소실된 교훈)

## 현재 파일
- `review_simulator_realism.py` — 합성 에피소드의 방향분산·steepness·aperture 측정.
  → `simulator_realism.json` 생성. 결론은 `docs/SIMULATOR_REALISM_REVIEW_2026-07-02.md`.
- `hos_residual_decomp.py` — HOS(M=3) 필드에서 추정기 잔차의 에너지/밴드 분해.
  ⚠️ 밴드분할(k_split) 결함 있음 — bound harmonic 격리는 `results/direction3/steepness_nonlinearity.json` 사용.
- `simulator_realism.json` — 측정 결과.
