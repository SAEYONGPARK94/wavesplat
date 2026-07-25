# G1 WavePrior Power — 2026-07-24 저녁 세션 보고서 (사후 재구성)

> **작성 경위**: 2026-07-24 저녁 세션(~17:00–22:12)이 마지막 실험 완료 직후 비정상 종료됨.
> 이 보고서는 2026-07-25 아침에 로그·JSON에서 재구성한 것. 실험 유실 없음 —
> 유실된 것은 (a) `unify_complexity`의 stdout 요약(JSON에서 재계산해 복원, 아래 §3),
> (b) 세션 보고서/메모리/커밋, (c) 사전등록 SECONDARY의 실측 재검(07-25 아침 실행).

## 맥락
07-24 낮 대장정 verdict = "WP niche 열림, 그러나 해석 prior 대비 우월성 반증"
([[project_waveprior_promotion_2026-07-24]]). 저녁 세션 = "WP 다른 각도" 후속:
복잡도 승리 정밀화 + Doppler 교란 통제 + fan-geometry 재학습(마지막 아키텍처 카드).

## 1. 복잡도 스윕 (item1.log, 18:07)
합성 in-dist(gen_sample), hand vs learned(v3), gt_q nRMSE, n=20/nc:
- nc=5: d +0.045 (15/20, p=0.0042)
- nc=15: d +0.055 (17/20, p=0.0023)
- nc=30: d +0.104 (18/20, p=0.0004)
**격차가 혼잡도와 단조 증가.** UQ 캘리브(spread-vs-error corr, nc=30): analytic +0.145 vs
learned +0.192 — 둘 다 약함, UQ 각도는 카드 아님.
그림: `results/gifs/item1_complexity_win.png` (median 장면은 d +0.003 = 시각적 구분 없음 —
그림은 분포 승리이지 장면 극적 차이가 아님을 유의).

## 2. Doppler A/B (doppler_ab.log, 21:06)
사전 우려 = "복잡도 승리가 실은 해류-오지정 강건성 승리 아닌가".
- Doppler=False: d +0.060, 7/8, p=0.023 (클린 조건에서 승리 존재)
- Doppler=True: d +0.019, 5/8, p=0.547
**⟹ 우려 기각.** 승리는 무-Doppler에서 더 크고 유의 — 복잡도 승리는 해류 아티팩트가 아님.

## 3. 실측 창별 복잡도-델타 상관 (unify_complexity.json, 19:12; stdout 유실→재계산)
n=30 창(5 record×6), Spearman:
- POOLED entropy vs delta: +0.318 (p=0.087) / ang-entropy +0.305 (p=0.102)
- **WITHIN-RECORD(demeaned): +0.035 (p=0.855)** / non-degen +0.050
- confound(entropy vs obs_std): −0.123 (독립)
- record별: YS17 +0.014 / AA15a +0.079 / AA14 +0.152 / AA15b −0.003 / LJ18g +0.363
**⟹ 합성의 복잡도 승리가 실측 창-내 예측자로 이전되지 않음.** pooled 경향은 record-간
차이(AA14·LJ18g)가 견인하고, LJ18g는 붕괴창(w3: WS 2.925, delta +1.54) 포함 —
[[project_waveprior_promotion_2026-07-24]]의 "실측 승리=붕괴창 아티팩트" 경고와 일관.
⛔ "복잡할수록 WP가 이긴다"를 실측 근거로 인용 금지 (합성 한정 MEASURED).

## 4. 실측 조우예보 head-to-head (enc_forecast.log, 18:15)
n=6, nRMSE vs lead (1.0=zero-pred): predictable horizon(<0.7) analytic 11.0s vs learned 12.0s.
lead별 우열 혼재 — 의미있는 차이 아님.

## 5. fan-geometry 재학습 (21:42 데이터 → 21:49 fan.pt / 22:10 fanTF.pt)
- `tmp/prior_train/fan_mix_s000–199.npz` 200시드 (fan_frac 혼합, bimodal 0.3, bw 0.15)
- 코드 diff(미커밋): `scripts/d3_prior_net_data.py` (+12/−4), `scripts/d3_prior_net_eval.py` (+20/−4)
- 산출: `results/direction3/prior_net_fan.pt`, `prior_net_fanTF.pt`

## 6. ★사전등록 평가 (eval_fan_prereg.py 21:25 동결 → 22:12 완료)
- **PRIMARY FAIL**: pure-fan 신선시드 n=30, fanTF vs hand: d +0.0159, 18/30,
  Wilcoxon 단측 p=0.1027 ≥ 0.05.
- **SECONDARY PASS**: fanTF vs old footprint-TF: d +0.0649, 24/30, p<0.0001.
  old vs hand: d −0.049 (11/30) — 옛 net은 fan 기하에서 hand보다 나빴음.

**해석**: fan-geometry 재학습은 기하 미스매치를 실제로 고쳤으나(SECONDARY),
그래도 해석적 hand prior를 못 이김(PRIMARY). 낮 세션 "novelty≠우월성" 판정이
아키텍처-기하 정합 후에도 유지됨. **fan-geometry는 마지막 미점검 아키텍처 축이었고,
이제 닫힘.**

## 7. 사전등록 SECONDARY 실측 재검 (07-25 아침 보완 실행)
`run_wp_realdata --ckpt prior_net_fanTF.pt --tag fanTF` — 결과는 `wp_realdata_fanTF.log`
/ `wp_realdata_fanTF.json` 및 본 파일 아래 추기 참조.

**결과 (07-25 완료, WS팔은 두 런 bit-identical 확인)**:
- ALL 30창: fanTF d +0.107 (t p=0.063, 18/30) vs v3 d +0.121 (p=0.043, 21/30).
  **paired fanTF−v3 = −0.014 (fanTF 우세 16/30) = 사실상 무승부, 방향은 소폭 악화.**
- 정상 27창(WS<1.0 붕괴창 3개 제외: AA14_w0 2.31, AA15a_w3 1.12, LJ18g_w3 2.92):
  fanTF d +0.023 (p=0.144, **n.s.**, 15/27) vs v3 +0.035 (p=0.034).
  ※ v3 정상창 수치가 낮 세션의 "+0.026 n.s."와 약간 다름 — 붕괴창 판정 기준 차이로 추정,
  fanTF 판정에는 영향 없음.

**⟹ 종합 판정: fan-geometry 재학습은 실측에서도 v3 대비 개선 없음(무승부~소폭 악화),
정상창 vs WaveSplat 우위도 n.s.** 합성 PRIMARY FAIL과 합치면 **fan-geometry 축 폐쇄** —
어제 낮 재학습 실패(40k, AA15b −0.070)와 같은 결론이 기하-정합 아키텍처에서도 재현됨.
"novelty≠우월성" 판정 3중 확인(v3 재학습·40k 재학습·fan-geometry 재학습).

## 미커밋 자산 목록 (커밋 대기)
- 코드: `scripts/d3_prior_net_data.py`, `scripts/d3_prior_net_eval.py`
- 모델: `results/direction3/prior_net_fan.pt`, `prior_net_fanTF.pt`
- 그림: `results/gifs/item1_complexity_win.png`
- 로그/JSON: `tmp/G1_waveprior_power/` 저녁분 전체 + 본 보고서
