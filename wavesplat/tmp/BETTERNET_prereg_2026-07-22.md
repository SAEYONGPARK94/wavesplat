# 사전등록 — Better net (comp-diverse γ 예측기) 2026-07-22

> 사용자 "Better net으로 가보자 + fable과 같이". ①→realizable→도전(quick FAIL)→bank-ceiling 진단.
> **핵심 진단**: +0.33 학습가능 room은 **sparse(low-rank) regime 전용**(broadband선 net이 이미 최적, room −0.05).
> 원인 = 현 net(prior_net_v3)이 **200comp broadband로만 훈련** ⟹ **broadband-특화, sparse-swell서 suboptimal.**
> ⟹ **better net = comp-수 다양 분포로 재훈련해 sparse 능력 추가(broadband 유지).**

## 설계
- **데이터**: comp-diverse(nc∈{12,20,35,60,100,160,200} 균등, n=4000), supervised γ(Y=참모드 bank라벨).
  `scripts/d3_prior_net_data.py --nc_diverse 1`(backward-compat 추가). `tmp/prior_train/compdiverse`.
- **훈련**: PriorNet(기존 아키텍처) supervised MSE on Y, `d3_prior_net_train`. → `prior_net_compdiverse.pt`.
- **baseline**: prior_net_v3(현 production, broadband-훈련).

## 평가 (multi-regime, held-out)
- **Sparse(15comp episodes, gap_ladder harness)**: new net이 prior_net_v3(0.57)를 이기나? (achieved 재구성 corr, n=16.)
- **Broadband(200comp bwtask held-out)**: new net이 prior_net_v3(0.45) 유지하나(안 나빠지나)?
- (선택) 30comp·5comp regime도 병기.

## ★사전 판정 (동결)
- **PASS**: sparse서 Δ(new−v3)≥+0.05 (CI 0제외) **AND** broadband서 Δ≥−0.02(regression 없음). ⟹ better net 성공
  = "comp-diverse 훈련이 sparse-swell 능력 추가, broadband 무해" = 배포-관련 개선(realizable 시연).
- **FAIL**: sparse gain 없음 OR broadband regression. ⟹ comp-diverse 훈련만으론 부족(더 큰 재설계/off-grid 필요).
- **검정력**: n=16 regime당, held-out.

## ⚠검증/캐비엇 (over-claim 방지)
- **validity(핵심 ARGUED)**: low-rank(sparse-sampling) regime의 **배포-관련성**(swell=narrow spectrum=적은 유효성분)은
  논증이지 실증 아님. 평가는 **synthetic JONSWAP-sampled** 바다 — **HOS/실측 미검**(실측 천장<1 예상). 
  ⟹ PASS여도 "swell 배포 개선"은 synthetic-스코프, 실측 재검 필수.
- **이중검증**: fable 독립 eval(다른 seed/regime) + 설계 red-team(leakage·overfit·regime 정의).
- fable 담당: validity red-team + 독립 eval + (선택)대안 변형(task-loss vs supervised-γ / 다른 nc 분포).
- Opus 담당: 데이터 생성(진행중)·훈련·multi-regime eval.
