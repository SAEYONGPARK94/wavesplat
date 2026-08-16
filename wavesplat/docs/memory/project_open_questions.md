---
name: WaveSplat remaining open questions (post-P2)
description: 남은 미해결 항목 — 대부분 해결, 실험적 확장 항목만 잔존
type: project
---
> ⛔**ARCHIVED / 낡은 사본 (2026-08-13 감사).** 현행 연구 기록의 권위는 auto-memory(`~/.claude/projects/-home-vexit-wavesplat/memory/MEMORY.md`)와 `tmp/<배치>/RESULT.md` 다. 이 디렉터리의 수치는 다수가 이후 측정에서 무효화됐다 — 인용 금지.


## 코드/문서 수정 — ✅ 모두 해결 완료

## 구현 시 실험적 결정 — ✅ 대부분 해결
1. ~~Coarse-to-fine 전환 residual threshold~~ → P0에서 결정 (auto schedule)
2. ~~Initialization 2Tp 충분성~~ → Beamforming v2_konly로 대체
3. ~~정규화 λ₁, λ₂~~ → 최종 구현에서 L1/smoothness 미사용 (MSE loss만으로 충분)
4. ~~ETKF-grid inflation factor~~ → n_ensemble=50으로 구현 완료
5. ~~GPR window 프레임 수~~ → GPR baseline 미포함 (ETKF만 비교)

## 논문 작성 시 반영 사항 — ✅ 대부분 반영
- ~~Linear vs Gerstner mismatch~~ → Discussion에 명시
- ~~Confidence = observation noise weight~~ → Method Section에 명시
- ~~Raydrop model 비교~~ → Table 4 + Discussion 단락 추가 (2026-04-07)

## 향후 연구 기회
1. Real LiDAR 데이터 (수조 실험 또는 NEUBOAT 실선 데이터)
2. 해류 Doppler shift 실제 영향 크기
3. Multi-seed 통계 검증 (현재 seed0 위주, 5-seed로 확장 가능)
4. BF5 low-comp, BF2 high-comp 추가 시나리오
5. 2nd order bound harmonic correction (C4)
6. Learned initialization (CNN/transformer for faster convergence)
7. Ship motion 효과 (underway scenarios)

**Status:** 핵심 실험/논문 작성 완료. 남은 항목은 extension/future work.
