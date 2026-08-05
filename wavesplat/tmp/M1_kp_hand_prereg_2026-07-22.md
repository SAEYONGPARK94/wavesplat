# M-1 prereg — kp↔hand A↔B 통합 확인 (2026-07-22, 결과 보기 전 동결)

> 질문: E-B2K의 실패창(hand<0.5, task-loss 이득 집중) 축이 E-58의 under-determination(kp/개구비) 축과
> **같은 물리축인가?** 순수 재분석(새 시뮬 0), `results/direction3/e60k_hos_arena.json`(160행=32바다×5draw).
> [[project_brainstorm_next_2026-07-22]] fable 아이디어. [[feedback_direction_flip_discipline]] 적용.

## 데이터 (행별)
hand, T_s0..2/L_s0..2/N_s0..2(3팔×3시드), n_obs, Tp, Hs, **lam_p**(peak 파장), steepness, draw, sea.
파생: **kp = 2π/lam_p** (peak 파수). gain_TL = mean(T)−mean(L), gain_TN = mean(T)−mean(N).

## 방법 (동결)
- 160행을 32바다로 집계(draw 5개 평균) — 바다가 군집 단위. 상관은 바다-수준(n=32).
- **주 통계**: Spearman(각 쌍), **바다 재표집 부트스트랩 CI 2000회**.
- 검정력 게이트: n=32 바다, 소표본이므로 **방향 라벨/★ 금지**. 결과=상관계수+CI만, 해석은 ARGUED.

## ★사전 예측 (E-58/개구비 법칙에서 도출 — 결과 보기 전 기입)
E-58: envelope 강함 at **낮은 kp / 긴 λp**(Spearman(kp,ρ)=−0.874). 개구비 법칙: "관측이 파장 대비 좁을수록
(=긴 λ) 학습 이득 큼". E-B2K 최난셀 Tp13_Hs5 = 긴 Tp→긴 λp→낮은 kp. ⟹ **통합 가설이 참이면**:
1. **Spearman(gain_TL, lam_p) > 0** (긴 λp → 이득 큼), Spearman(gain_TL, kp) < 0.
2. **Spearman(hand, lam_p) < 0** (긴 λp → hand 낮음=실패), Spearman(hand, kp) > 0.
3. gain_TL과 hand는 음의 상관(이득이 실패창에 집중 — E-B2K 기지, 확인용).

## 판정 프레임 (사후 라벨 금지)
- 세 예측 방향 일치 + CI가 0 제외 ⟹ **"방향-일관, ARGUED 통합 가설 지지"**(★헤드라인 아님, 별도 아레나·사후매칭 캐비엇).
- 방향 불일치/CI가 0 포함 ⟹ **"두 축 분리 or 미확립"** — B는 별도챕터 유지 근거.
- ⚠캐비엇: E-B2K=HOS broadband(kp=참 스펙트럼 피크), E-58=합성 다성분 — cross-arena. 개구(관측 공간범위)가
  바다간 일정한지 미확인(n_obs=600 고정이나 공간범위는 미기록) ⟹ lam_p 단독이 개구비 대리(불완전).
