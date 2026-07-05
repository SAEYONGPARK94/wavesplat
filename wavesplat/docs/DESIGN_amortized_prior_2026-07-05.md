# 설계: Amortized Neural Empirical Bayes for Moving-Platform Wave MMSE
(2026-07-05, 사용자 방향 지시: "학습 쪽 정확도 있는 결과 + 분석"이 논문/학위의 골격)

## 왜 지금 학습인가 (이전 실패와의 차이)

07-02~04의 학습 시도들이 죽은 이유는 타깃 미정의였다. 오늘(07-05) 병목 해부가
학습 타깃을 처음으로 정량 특정했다:

- 상금: 이동 ceiling(oracle_ls2) 0.41/0.46 vs hand-crafted MMSE 0.264/0.234
  → **+0.15~0.2 corr, oracle로 존재 증명된 갭**
- 갭의 원인 2개가 모두 학습 가능한 함수 형태:
  (A) **prior 형상 추정** — 현재 beamform-power^p + val-EM 휴리스틱
  (B) **차수-48 성분 선별의 커버리지** — bank-IRLS는 k 정확도(0.012)는 충분한데
      성능 0.26 (참 k 48개 ridge는 0.47~0.60) = 선별 품질이 전부

## 모듈 1 — 학습 prior (헤드라인 후보)

```
(P, z) --무차원 정규화--> net --> gamma(k) : bank prior 분산장 (B~1900)
                                     |
                       물리 MMSE 선형해 (dispersion 구속, Woodbury, 해석적)
```

- **net은 prior만, 추정은 물리가**: mean-collapse 면역(field 직접회귀 아님),
  sim2real 안전판(net 오류 시에도 최종 추정기는 z-선형 + λ-CV 방어),
  "net 제안-물리 처분" = 유일하게 RELAX를 이겼던 neural-guided RELAX 패턴.
- **입력 표현**: bank 좌표를 kp로 정규화(k/kp), obs를 (x/λp, y/λp, t/Tp,
  z/(Hs/4))로 무차원화 — component-distill net의 sim2real 사망 원인
  (스케일/broadband gap)을 입력 설계에서 제거. 후보 아키텍처:
  ① beamform-power map(정규화 k-grid) 입력 CNN → γ-map 출력 (07-03 돌파 형식)
  ② obs set-transformer → γ-map (형식 자유도 높음, 무거움)
  1차는 ①: 입력 = [half1 파워맵, half2 파워맵, min맵] 3채널 (유령 정보 내장),
  출력 = log-γ 맵 (bank 격자로 보간).
- **라벨/손실**: 합성 참 스펙트럼에서 bin별 참 γ* = Σ A_m²/2 (bank cell 귀속)
  → log-γ 회귀 + (중요) **task loss**: MMSE 통과 후 field MSE를 미분가능하게
  (Woodbury 해는 미분가능; batch당 400×400 solve, GPU OK). 라벨 회귀만으로
  시작, task loss는 2차.
- **학습 분포**: 연속스펙트럼(nc=200, Tp/Hs/γ_J/방향 spread 랜덤), 연산자
  {fixed, track U/cp 0.1~1.2, 방향 랜덤}, 노이즈 2~10%, **HOS M=3 비율 ≥ 30%**
  (상시지시), 400obs/10Tp/2λ 프로토콜.
- **게이트(사전 등록)**: G1 합성 이동조건 hand-MMSE 대비 +0.05↑,
  G2 HOS에서 유지, G3 실측 record 무-fine-tune 전이(≥ hand-MMSE).

## 모듈 2 — 이동-연산자 차수-48 set-검출기 (독립 기여)

- 타깃: top-48 성분의 (kx,ky) set — 커버리지 loss(Chamfer/헝가리안) +
  진폭가중. 07-03 beamform-CNN 돌파의 moving-operator 버전.
- 추론: net 제안 48개 → per-comp Newton refine → λ-CV ridge (nomp 인프라 재사용).
- 성공 범위가 이미 브래킷됨: bank-IRLS 0.26 (하한) ~ 참-k-48 0.47-0.60 (상한).

## 평가/논문 구도

- 이론(완성): CRB/achievability 3층 + 차수·정밀도·유령 해부 = 학습의 정당화
- 알고리즘(신규): 모듈 1(+2)이 ceiling 갭 회수 — gap-to-bound 회수율 figure
- 검증(인프라 완성): matched-budget 실측 10 record + HOS + UQ calibration
  (MMSE posterior 분산 해석적 → 학습 prior의 calibration 평가 공짜)
- 서사: "정보-구조 분석이 설계한 학습 추정기" — 분석이 학습을 정당화하는 구조

## 실행 계획 (1차 프로토타입)

1. 데이터 생성기: 하니스 재사용, (파워맵 3ch, γ* 라벨, MMSE-ready obs) 샘플
   ~20k (합성 linear) + HOS 청크 — CPU 여유 시 백그라운드.
2. CNN(U-Net 소형) log-γ 회귀 → 하니스 10-sea 평가 (mmse_bank_estimate의
   γ 주입 스위치 추가).
3. G1 통과 시 task-loss 미세조정 + 모듈 2 착수.

리스크: γ* 라벨의 bank-귀속 모호(연속스펙트럼 → cell 적분으로 정의),
파워맵 해상도 vs bank 해상도 불일치(보간), EM과의 이중정제 충돌(net-γ에는
EM 끄고 비교). 전부 ablation 슬롯.
