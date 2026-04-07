---
name: WaveSplat paper and timeline plan (updated)
description: 논문 구조, 타겟 학회, 현재 상태 — 논문1 초안 완료, 논문2(MPC) 미착수
type: project
---

## 논문 1 — 초안 완료 (2026-04-07)
**제목**: "WaveSplat: Grid-free Spectral Gaussian Wave Field Estimation from Sparse LiDAR"
**타겟**: Ocean Engineering (IF~5) 또는 IROS 2027
**저자**: Saeyong Park (Avikus / HD Hyundai)
**파일**: `paper/wavesplat.tex` (IEEE conference format)

### 구조 (완료)
- Abstract, Introduction, Related Work
- Method: Forward Model, Observation Model, Confidence, Two-Phase Estimation, Adaptive N
- Experimental Setup: Simulation, Metrics, Baselines
- Results: Main Comparison (Table 1), Adaptive N (Table 2), Ablation (Table 3), Raydrop (Table 4)
- Discussion: Grid-free advantages, Limitations, Dispersion constraint, Raydrop analysis
- Conclusion

### 핵심 수치
- WaveSplat avg nRMSE: **0.211** vs ETKF **0.436** (51.5% better)
- Speed: **13.4 ms/frame** (2.4× faster)
- 12/12 wins, 20/20 raydrop robustness

### 남은 작업
- [ ] Literature references 정리 (.bib 파일)
- [ ] 공저자 추가/확인
- [ ] Abstract/Introduction 최종 다듬기
- [ ] (Optional) Real data experiment 추가

## 논문 2 (확장, 미착수)
**제목**: "Wave-Preview MPC for Hydrofoil Vessels Using LiDAR-based Wave Field Estimation"
**타겟**: IROS 또는 Control Engineering Practice
**상태**: Hydrofoil model 작성 중 (별도 진행)

## 타임라인
- M1~M2: WaveSim 완성 ← ✅ 완료
- M3~M4: 데이터셋 생성 ← ✅ 완료 (100 episodes)
- M5~M8: WaveSplat 알고리즘 + 실험 ← ✅ P0-P1 완료
- M9~M11: 논문 1 작성/투고 ← **현재 단계** (초안 완료, revision 필요)
- M12: (선택) 수조/실선, revision
- M14~M17: 논문 2 (MPC + hydrofoil)
- M18~M24: 학위 논문
