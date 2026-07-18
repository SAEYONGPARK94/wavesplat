# E-59 v2 사전등록 — §8 실측 anchor (조건③, 옵션 A) + fable red-team 반영 2026-07-18

> v1(`E59_real_anchor_prereg.md`)을 fable 적대 심사가 **"실행 불가(FATAL 3)"** 판정 → C1–C7 반영한 v2.
> 내가 fable의 핵심 산술 독립검증(감쇠 rel₁=0.289, R12/R96 예측 0.297/0.322 ≈ 실측 0.287/0.321). 규율 유지.

## fable FATAL + 반영
- **F1 kill 발동불가**(리드-풀링이 t² σ_CRB로 양성 보장·거리가드 dynamic range 0) → **C2 within-lead 공간
  Spearman 주판정**, FAIL = within-lead median ≤0.
- **F2 공유-기저 tautology**(θ̂가 예보와 같은 k-그리드 → σ_CRB≈σ_emp → self-calibration 시험) → **C1
  decorrelated null(위상-교란 truth, spectrum+geometry 보존·coherent 파괴)**; Δρ=ρ_real−ρ_null 주판정,
  SUPPORT는 blocked-CI Δρ>0 요구. **estimator-독립 헤드라인은 이 시험이 못 뒷받침**(테스트 추정기=CRB의 ML twin) →
  C8 cross-estimator는 별건 캐비엇.
- **F3 단일실현 감쇠**(rel₁=0.289 → 완벽 envelope도 R=1서 ρ≈0.175=PARTIAL밴드) → **공간 국소-RMS σ_actual**
  (sub-λ 위상진동 평균 = 위상평균의 실측 대응물; 유효신뢰도 spatial split-half로 **측정**·disattenuate) +
  **C3 matched twin**(합성 R=1·fitted-θ̂·strip·실측 NaN마스크·matched kp/spread)을 **먼저** 돌려 바 설정.

## 설계 (frozen, C1–C7)
- rec = {AA15a(aper/λ2.0), AA14(2.6), YS17(4.7)} primary. LJ18g exploratory.
- **C4 strip-obs(최고가치)**: 관측 = 패치 x축 앞 **f_strip=0.40** 구간의 valid 픽셀만(정보 gradient 생성·거리가드
  부활); 질의 = **전체 패치**(strip 바깥 ≥1λ = **국소 under-determined = 강-레짐 도달 경로**).
- **C5 operator-detrend + ring-clip**: k-그리드 하단 = max(0.35·kp, 2π/(0.9·patch_min))/kp (super-patch λ 제거);
  detrend 평면은 obs·truth 동일 적용.
- **C7 freeze**: W=6.0s, N_obs=1200, θ̂ top-K=24, query stride→~600점, leads={0,2,4,6}s(rec 상한서 잘리면 그
  lead 제외 사전명기), spatial-RMS 반경 r=λ/4, f_strip=0.40, null=위상교란(seed별), R=1.
- σ_CRB(q)=√(J_q Finv J_qᵀ)/Hs (nonlinear θ̂ [A,kx,ky,φ], dispersion ω). σ_actual = √(box_λ/4⟨(예보−truth)²⟩)/Hs.
  σ_emp=√(Φ_q Ainv Φ_qᵀ)σ/Hs. σ_actual_null = 위상교란 truth 동일. dist=최근접 obs.

## 판정 (frozen, C2·C1·C6)
per (rec, window, lead): **within-lead 공간 Spearman(σ_CRB,σ_actual)** [주] · null · emp · dist · reliability(split-half).
- **SUPPORT**: within-lead median ρ**>0** AND **Δρ(real−null) blocked-CI 하한>0** AND real ρ가 **matched-twin 바
  구간과 일관**(twin이 완벽재현 기대치) AND 3 rec 부호 일치 ⟹ 실측 재현(스테레오·비이동·strip-기하 캐비엇).
- **FAIL**: within-lead median ≤0 또는 Δρ CI가 0 포함 ⟹ 실측서 envelope이 기하/스펙트럼 이상 못 잡음 ⟹ 헤드라인
  합성-only 후퇴(옵션 C). **이제 실제로 발동 가능**(F1 해소).
- **PARTIAL**: Δρ>0이나 twin 바 미달 or rec 이질 ⟹ CI 보고, 라벨 강제 금지.
- **C6**: record-blocked bootstrap CI + n_eff(≈patch/λ² per frame) 명시; ~400점 CI 금지(가짜 정밀).

## 대칭 감사 (사전, 규율4)
- 과대: 공유기저(C1 null 필수)·리드풀링(C2)·거리(strip이 dynamic range 부여+partial)·detrend edge(C5)·twin 없는
  matched비교(C3).
- 과소: 단일실현 감쇠(spatial-RMS+측정), n_eff 소(blocked CI), MNAR 구멍(급준부 CRB 낙관·국소 부호반전 가능),
  strip이 이동센서 아님(정지 기하), fitted-θ̂ 불안정.
- **최대 캐비엇 불변**: 실측=스테레오 유래, **실물 이동 LiDAR 미관측** → "strip anchor"는 정지-스테레오-기하 재현.
- **estimator-독립 = 이 시험 무증거**(F2): 별도 cross-estimator(C8) 필요, 본 실험 스코프 밖.

## 실행 순서
1. `tmp/e59v2_real_anchor.py`: strip+operator-detrend+spatial-RMS+within-lead+null+twin+blocked-boot.
2. smoke → **matched twin(바)** → 실측 primary(3 rec) → **fable 결과 감사(같은 에이전트 SendMessage)** → 화해.
