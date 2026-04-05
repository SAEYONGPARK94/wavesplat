---
name: WaveSplat paper and timeline plan
description: 논문 구조, 타겟 학회, 타임라인 — 논문1(추정 알고리즘), 논문2(MPC), 학위논문
type: project
---

## 논문 1 (핵심, 12개월 목표)
**제목**: "WaveSplat: Grid-free Wave Field Estimation from Sparse Ship-mounted LiDAR via Spectral Gaussian Mixture"
**타겟**: Ocean Engineering (IF~5) 또는 IROS 2027
**구조**: Intro → Problem Formulation → Algorithm → WaveSim → Experiments(6개 subsection) → Discussion → Conclusion
**실험**: progressive complexity(1→30성분), sparsity sweep(N_pts), baseline비교(GPR, ETKF-grid), raydrop 모델 비교, ablation, NEUBOAT anchor

## 논문 2 (확장, 4개월)
**제목**: "Wave-Preview MPC for Hydrofoil Vessels Using LiDAR-based Wave Field Estimation"
**타겟**: IROS 또는 Control Engineering Practice

## 타임라인
- M1~M2: NEUBOAT 분석 + WaveSim 완성 ← **현재 WaveSim Phase 1-B 완료**
- M3~M4: WaveSim 검증 + 데이터셋 생성 (75 episodes)
- M5~M8: WaveSplat 알고리즘 + 실험
- M9~M11: 논문 1 작성/투고
- M12: (선택) 수조/실선, revision
- M14~M17: 논문 2 (MPC)
- M18~M24: 학위 논문

**Why:** 논문 1이 나오면 나머지가 따라오는 구조. 논문 1에 집중.
**How to apply:** Phase 3(추정 알고리즘)이 가장 중요한 다음 단계. 시뮬레이터 개선에 시간 낭비 금지.
