# 파수-ID CRB 결과 — ① 헤드룸 = **REALIZABLE (알고리즘 탐색갭, info-limited 아님)** (2026-07-22)

> 사용자 "파수 ID CRB를 봐보자". ① 헤드룸(+0.45)이 realizable(알고리즘)냐 info-limited(aperture 해상불가/다봉)냐 판별.
> 스크립트 `tmp/wavenumber_id_crb.py`, 원자료 `results/direction3/wavenumber_id_crb.json`. under-det(15comp) n=16.

## 판정 = **REALIZABLE** (info는 있음, 현 방법이 탐색 실패)
### (1) 국소 식별성 — 파수는 식별가능
- **각도 CRB median 0.5°** vs 성분 각도간격 **2.6°** ⟹ **ratio 0.21 (≪1) = aperture가 방향을 해상 가능**, 정보 존재.
- 반경 CRB/kp 0.044 (tight). ⚠성분별 편차: 일부 바다 ratio 0.9(간격 근처=marginal), median은 양호.
- FIM cond 6.7e8(높음) — **비-방향 파라미터**(약성분 진폭/위상)의 ill-conditioning, 각도 CRB는 tight 유지
  (gap_ladder서 field-CRB~0과 정합). 방향 ID엔 무영향.

### (2) 전역 모호성 — 참 파수집합이 유일 최적 (다봉 아님)
- **참 파수(N개) obs-fit 잔차 0.018 ≈ 잡음바닥(0.02)** = 참 집합이 obs를 완벽 적합.
- **RELAX 추정(N개) 잔차 0.094 = 5× 나쁨** ⟹ **틀린 집합이 참만큼 적합하지 못함 = 우도 단봉(다봉 아님).**
  RELAX는 참 basin을 **못 찾은** 것(탐색 실패)이지, 동등한 대안 basin이 있어서가 아님.

## ⟹ 결론 (사용자 질문 답)
**under-det +0.45 헤드룸은 info-limited가 아니라 알고리즘 탐색갭이다:**
- 파수는 **식별가능**(각도 CRB 0.5°≪2.6° 간격) + 참 집합이 **유일 최적**(잔차 잡음바닥, RELAX 5× 열위).
- ⟹ 갭(net-MMSE 0.55 → oracle 1.0)은 **비볼록 파수탐색의 알고리즘적 실패**, **원리상 더 나은 식별기가 실현 가능.**
- = **오늘 세션 유일하게 "학습이 원리상 뚫을 수 있다"고 CRB이 지지하는 forward path.** (B-2·궤적은 oracle-washout였음.)

## ⚠캐비엇 (over-claim 방지)
- **"원리상 realizable(info 존재)" ≠ "닫는 방법을 시연함".** 특정 학습법이 +0.45를 실제 회수하는지는 **다음 단계**
  (본 결과는 "천장 도달 가능"을 CRB로 확인한 것). 
- **비-uniform**: 일부 바다 각도 ratio 0.9(marginal 해상) — 전 레짐 균일 아님.
- **synthetic 낙관**: obs=model이라 참 잔차=잡음바닥. 실측/HOS(model mismatch)면 천장<1, 실현 헤드룸 축소.
- RELAX 잔차엔 **격자 이산화**(NK=25)도 일부 기여 ⟹ off-grid 정밀 식별기가 유리(realizability 방향 지지).

## ⟹ 세션 종합 갱신 (이 결과의 위치)
- ① under-det 파수-ID 갭 = **정체 규명(파수 ID) + 헤드룸 실재(+0.45) + realizable(algorithmic, info-there) 확인.**
  = **유일 살아있는 학습축이 "죽은 축(궤적) 아니라 진짜 forward path"임을 CRB이 승인.**
- **다음 자연 단계(하면)**: 더 나은 파수 식별기(off-grid·비볼록-강건·학습) 시연해 +0.45 중 얼마 회수하나 = 직접 realizability.
- **정직 권고 불변**: 이건 학습을 더 밀 값어치 있는 **유일** 축(realizable 확인). 병행/대안 = spine A 집필.
