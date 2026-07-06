# 게이트 확증 재검증 — 사전 등록 (2026-07-06, 실행 전 커밋)

목적: 적대 검토(R2 공격 1, garden of forking paths)에 대한 확증 시험.
아래 프로토콜은 **실행 전에 고정·커밋**되며 이후 변경하지 않는다.
결과가 어느 쪽이든 그대로 보고한다.

## 고정 가설
사이트-적응 val-residual 게이트: 각 레코드 내에서 hand-MMSE의 held-out
잔차 vr_rel을 그 레코드의 median/MAD로 정규화한 값이 낮은 창일수록
조우예보 실현 skill(리드 0-10 s corr)이 높다.

## 고정 프로토콜
- 신규 데이터(개발에 미사용): BS_2011 2레코드 + BS_2013 2레코드
  (BS11a/BS11b/BS13a/BS13b). 패치 14-19 m — **스케일 프로토콜**:
  r_fan = (0.15, 0.7)·λ_p(창별 characterize), U = 0.13·c_p,
  W = 10 s(=고정), lead_max = 6 s, 리드 스텝 1 s, 12윈도우/레코드,
  시드 규약 동일(rng 700+wi), m_scan 12, m_cap 600, 노이즈 2 cm.
- 신호: `mmse_bank_estimate` diag_out['vr_rel'] (정의 고정, 기존 코드).
- 정규화: 레코드-내 median/MAD (기존 정의 그대로).
- 판정 기준 (사전 고정):
  P1(주판정): 4레코드 pooled Spearman(vr_norm, realized) < 0 이고
  단측 p < 0.05.
  P2(보조): median-split 신뢰/위험 평균 corr 차 > +0.10.
- 성공 = P1 충족. P1 실패·P2만 충족 = "방향성 재현, 미확증".
  둘 다 실패 = 게이트 확증 실패로 보고.

## 참고 (탐색 단계 결과 — 비교용으로만)
개발 3레코드(YS17/AA15a/AA14) pooled −0.38(p=0.021), 분리 0.68/0.30.

(실행 스크립트: d3_gate_confirmatory.py — 이 문서 커밋 후 작성·실행)
