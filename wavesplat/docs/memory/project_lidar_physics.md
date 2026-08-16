---
name: LiDAR surface return physics decisions
description: 수면 LiDAR 반사 메커니즘 2모델(diffuse vs specular), 방향 의존성, return probability
type: project
---
> ⛔**ARCHIVED / 낡은 사본 (2026-08-13 감사).** 현행 연구 기록의 권위는 auto-memory(`~/.claude/projects/-home-vexit-wavesplat/memory/MEMORY.md`)와 `tmp/<배치>/RESULT.md` 다. 이 디렉터리의 수치는 다수가 이후 측정에서 무효화됐다 — 인용 금지.


## 두 가지 Return 모델 (병렬 유지)

### Model A — Diffuse (Fresnel/capillary)
- P_return = η × R_fresnel(θ_i) × cos(θ_i) / distance²
- Capillary wave에 의한 미세 산란
- 정수면에서도 약간의 return 가능

### Model B — Specular backscatter (gravity wave slope)
- 파봉 경사면에서 법선이 빔 방향과 정렬될 때만 return
- 파 없으면 return 불가, head sea에서 최대
- BF4 R=100m에서 필요 slope=0.05 → 파봉에서 조건 성립

### 핵심 발견
- 방향 의존성: head sea → return 최대, beam sea → return 최소 (논문에서 분석 가치)
- Model B에서 파봉 편향 분포 → 오히려 wave field 복원에 유리할 수 있음
- Return probability 정확한 값은 이론만으로 불가 → parametric sweep으로 해소

**Why:** 어느 메커니즘인지는 NEUBOAT 분석 또는 수조 실험에서 확인. 그 전까지 두 모델 유지.
**How to apply:** 두 모델 비교 자체가 ablation study의 한 축. 코드에서 항상 양쪽 모두 지원.
