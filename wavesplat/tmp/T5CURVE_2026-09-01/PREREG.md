# `T5-CURVE` 사전등록 (2026-09-01) — ⛔**등급 `EXPLORATORY` 고정**

> **런 0회 · 추정기 0회 · 새 시드 0.** 동결 `T5_ENS_2026-09-01/results_t5_*.jsonl`(72행) 만 읽는다.
> 선례 = `B-20b`(동결 자료 재분석) · `Stage A`(`T5-ENS §4.7`).

## §0 무엇을 하는가
`T5-ENS` 행에 이미 저장된 `per_lead_med`·`per_lead_m0`·`zero_per_lead`(15 lead × 36 셀)를
**lead 별로 분해**한다.

## §1 ⛔등급은 결과와 무관하게 `EXPLORATORY` 다
- `T5-ENS` 의 확증 가족은 **PRIMARY(NEAR)·SEC-A(FAR) 둘뿐**이고 lead 별 분해는 그 안에 없다.
- **사후**이고 **노출**돼 있다(내가 `T5` 결과를 봤다).
- ⛔**사후 방향 선택 금지**(`T4 §4` 규율) — 이 배치는 **검정을 하지 않는다**. 산출은 **기술**이다.
- ⛔`T5-ENS` 판정·계수 갱신 없음.

## §2 왜 만드는가
`T5 RESULT §1` 축자: **FAR 에서 결과 전 예측이 실패했다**(예측 q 0.65–0.75 vs 실측 0.972).
***예측이 빗나간 지점의 모양을 보는 것***이 다음 설계의 입력이고, 자료는 이미 행에 있다.

## §3 ★그림은 **두 패널**이어야 한다
한 패널만 그리면 `F1` 부류의 오독이 생긴다(*"lead 가 길수록 우리가 낫다"*).
- (위) 앙상블 이득 곡선
- (아래) ⚠**같은 구간에서 참 제로필드를 이기는 셀 수** — BANK · PROP(med3) · PROP(단일)

## §4 산출물
`analyze_curve.py` → `curve.json` · `verdict_curve.json`(`source_results` 선언, 결과 파일 없음)
`make_fig_curve.py` → `thesis/figures_ws/fig11_ensemble_leadcurve.pdf`

## §5 ⛔주장하지 않는 것
⛔확증 · ⛔`T5-ENS` 갱신 · ⛔*"lead 가 길수록 제안 표현이 낫다"*(비교자가 무너진다) ·
⛔배포 권고 · ⛔사이트 일반화
