---
name: WaveSplat core research decisions (revised)
description: 핵심 연구 결정 — Contributions 통일, 시뮬레이션 전략, 물리적 범위
type: project
---
> ⛔**ARCHIVED / 낡은 사본 (2026-08-13 감사).** 현행 연구 기록의 권위는 auto-memory(`~/.claude/projects/-home-vexit-wavesplat/memory/MEMORY.md`)와 `tmp/<배치>/RESULT.md` 다. 이 디렉터리의 수치는 다수가 이후 측정에서 무효화됐다 — 인용 금지.


## 확정된 Contributions (방법론 중심, 통일)
- **C1**: Grid-free spectral estimation framework (ETKF의 grid 보간 한계 극복)
- **C2**: Dispersion-constrained spectral Gaussian parameterization (물리를 구조에 내장)
- **C3**: Geometry-aware confidence model (LiDAR 수면 관측 특화; Lambertian × inverse range, 정확한 Fresnel은 raydrop 시뮬에서만)
- **C4**: 2nd order bound harmonic correction (선택적, 후속 연구 가능)

⚠️ 개발가이드의 C1~C4(논문구성 중심)와 다름 → 개발가이드 업데이트 필요

## 방법론 전환: 3DGS → Spectral Gaussian Estimation
- 3DGS 핵심 메커니즘 (photometric loss, tile rasterization) 불필요
- FluidGS 차별화 불가, "From Points to Waves" grid 한계 극복이 진짜 novelty
- "Gaussian Splatting" 용어 사용 금지

## 시뮬레이션 전략
- Pure Python WaveSim (Isaac Sim 불사용)
- Calibration-free: N_pts=5~500 parametric sweep
- NEUBOAT = reality anchor (1~2문장)
- 수조/실선은 논문1 투고 지연시키지 않음

## 물리적 범위
- BF3~5, non-breaking waves
- Full dispersion(tanh) 항상 사용
- Q 제약 필수 (ΣQᵢωᵢAᵢ ≤ 1)
- 추정 알고리즘은 linear sinusoidal model (GT Gerstner와 O(ε²) mismatch — 의도됨)
- 비선형(bound harmonics, 쇄파) = 후속 연구

## LiDAR 스펙 (확정)
- Seyond Falcon K1, **1550nm** (905nm 아님), 152채널
- 120°×25° FOV, range 1.5~250m

**Why:** 반복 검토에서 Contribution 번호 불일치, LiDAR 파장 불일치, linear/Gerstner mismatch 발견 후 통일.
**How to apply:** 개발가이드 §1.3, §3.2의 파장/Contribution 수정. 논문 작성 시 이 파일의 C1~C4 사용.
