# SBI/NPE keystone③ CRB 검증 결과 = **INCONCLUSIVE (NPE 미보정)** (2026-07-22)

> fable 소유 트랙(제안2). 스크립트 `tmp/sbi_npe_crb_verify.py`, 원자료 `results/direction3/sbi_npe_crb_verify.json`,
> 로그 `tmp/sbi_rerun.log`. 목적 = E-51/52 CRB Gaussian-근사(GUARD2 미납)가 진짜 posterior 대표하나 경험검증.
> Opus 독립 cross-check + calibration heads-up 반영. (fable 최종 md 보고 별도.)

## 방법
5-dim regime-shape θ → 4-dim 비교(d1,d2,Tp2,w). arena 2개(si=700/750), N_TRAIN=4000, MDN(K_MIX=4) 3000스텝,
held-out n=64. NPE posterior std vs 해석적 √CRB 비교 + **PIT calibration 체크**(내 confound 경고 반영).

## ★판정 = INCONCLUSIVE — NPE가 이 예산선 미보정
- **calibration 실패(결정적)**: PIT KS-stat 0.22~0.42 **p 1e-5~1e-11 전 차원**, ±1σ cover **0.16~0.50** (Gaussian이면
  ~0.68). ⟹ **MDN posterior가 미보정** = θ를 명목 커버율로 못 덮음.
- ⟹ **넓은 width 비율(d1/d2 NPE-std 44~73 = √CRB의 160~170×)은 CRB 반증 아님** — NPE 품질 아티팩트(+ d1/d2는
  방향이라 각도 wraparound로 linear-std 부풀 가능; 로그 periodicity 체크가 시사).
- Spearman(√CRB, NPE-std) ~0(전 차원 p>0.16)·multimodal_frac 0.53(top weight 0.73)는 **관찰이나 미보정 MDN이라
  신뢰불가**(spurious 모드 가능).

## ⟹ 결론 (정직)
- **CRB Gaussian-근사를 반증도 확증도 못 함. GUARD2 여전히 미납.** 소규모 예산 MDN이 신뢰할 reference가 못 됨.
- **진짜 검증 = 제대로 보정된 NPE 필요**(더 큰 예산·각도-인지 파라미터화·`sbi` 패키지 normalizing-flow, SBC 통과) =
  별도 실질 작업.
- **⛔인용금지**: "SBI가 CRB 반증"·"posterior 다봉 확인"(둘 다 미보정 confound). **인용가능**: "소규모 NPE는 이
  검증에 부족(calibration 실패), GUARD2엔 보정된 NPE 필요" + 프로세스(calibration 체크가 false-positive 방지).
- **가치**: 값싼 probe가 (a) GUARD2 검증 경로를 구체화 (b) calibration 규율이 "CRB 반증" 오주장 차단 — 정직한 음성.

## ⓪ 후속 — fable의 "e51_crb 단위버그" = **FALSE ALARM (경험검증)**
fable가 SBI 자체검증 중 "d1/d2가 도인데 라디안 취급"이라 플래그 → 코드추적상 그럴듯(angle_mix "mu_rad", build_sea_crosstp
로컬 d1=uniform(0,360) 도). **그러나 경험 before/after가 반증**: `build_sea_crosstp` 반환문(e37:475)이 `d1=deg2rad(d1)`로
**이미 라디안 저장**(로컬 도-변수와 별개). 현 코드(라디안)=키스톤③ 헤드라인 재현(d1 11°/d2 16°/w 0.146), deg2rad "수정"=
깨짐(w 642·d2 1888°). **⟹ 버그 없음, keystone③ CRB 정상, spine A 무결성 OK. "체크리스트 단위버그" 철회.**
**교훈**: 코드추적만으론 fable도 Opus도 false-positive → **경험 before/after 검증이 정상코드 파괴를 막음.**

## 이 라운드 spine A 영향
- keystone③(CRB Gaussian-근사)은 **미보강도 미반증** — 현 상태 유지(E-51/52 "achievability 미납" 리스크 그대로).
- #1 궤적정책 FAIL과 합쳐, **이 라운드는 spine A를 흔들지 않음**(오히려 학습 positive들의 oracle-의존 재확인이
  CRB-판별 척추의 상대적 견고성 부각). [[project_methods_landscape_2026-07-22]].
