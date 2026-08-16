# 토론 중 정정 2건 — 그리고 **정정 자체의 정정**

## 최종 판정: P4 의 §F-0 은 **유지된다**. 내(주 세션) 1차 정정이 **과잉정정이었다.**

### 사건 경과
1. **P5 반박**: *"P4 의 `SUPPORT-TRADEOFF 부재`는 원자료로 반증된다 — `results.jsonl` 에
   `shell_rmse` 는 `SHIP` 64행뿐, `LETKF_PLUS`·`ZERO` 는 0행. 셸 대조는 계산 불가."*
2. **내 1차 정정**: 원자료를 확인하니 P5 의 **저장 데이터 진술은 참**이었다
   (`SHIP` 64/64 · `LETKF_PLUS` 0/64 · `ZERO` 0/64). 나는 여기서 **"⟹ 근거 없음, REFUTED"** 로 갔다.
3. **P4 의 R3**: 수치는 저장 파일이 아니라 **자기가 `LETKF_PLUS` 를 셸 계측 붙여 재실행**한 것.
4. **내 재검증 — P4 가 옳다.** 산출물 실재:
   `tmp/DEBATE_novelty_2026-08-14/p4_shell_rerun/{p4_letkf_shells.py, p4_shells.out}`
   (원본은 스크래치패드, 이번에 저장소로 보존). 스크립트 docstring 이 명시한다 —
   *"Re-runs LETKF_PLUS exactly as run_hosgate.py:run_letkf does (same seed, same config)
   and additionally records obs-distance shell RMSE … matching wavesplat/run.py:358 edges,
   so it is directly comparable with the SHIP shell_rmse already stored"*.
   **재현 게이트: 재실행 `avg_nRMSE` 가 저장값과 |Δ| = 0.000e+00, 16/16** (`P4_skeptic.md:265`).
   셸 표 전체가 `P4_skeptic.md:275-283`.

### 그래서 무엇이 참인가

| 주장 | 판정 |
|---|---|
| P4 §F-0: HOSGATE `LIN` 혼잡에서 `SHIP` 이 9셸 전부 승, `0_5` **15/16 −41.6% p=0.0005** | **유지**(재실행 근거, 재현 게이트 통과) |
| P5: 저장 `results.jsonl` 에 기준선 셸이 **0행**이다 | **참** — 그리고 이것은 별개의 진짜 결함이다(아래) |
| P5: ⟹ 따라서 P4 는 근거가 없다 | **거짓** |
| 내 1차 정정 "REFUTED" | ⛔**철회** |

### ★내 오류의 정체 (이게 기록할 값어치다)
나는 **P4 에게는 "처방도 검증하라"를 적용하고 P5 에게는 안 했다.** P5 의 반박을 받고 원자료만
확인한 뒤 **P4 보고서 §4(`:262-279`)를 열지 않았다** — 거기에 재실행 설계와 셸 표가 전부 있었다.
이것은 이 저장소가 다섯 번 물린 `scope_of_search_equals_scope_of_claim` 이고, 이번엔 **반박을
검증할 때** 발생했다. ⟹ **규율 추가: 반박도 원 보고서 전문을 열고 검증하라. 반박은 무료가 아니다.**

### ★그럼에도 P5 가 드러낸 진짜 결함 — S-2a 가 재발했다
2026-08-13 감사가 S13 에 대해 *"`run_letkf()` 가 셸 미반환 ⟹ 사망조건 S-2a 미검정"* 을 BLOCKER 로
잡았는데, **HOSGATE 러너(주 세션 설계)가 같은 결함을 반복했다** — 감사 4시간 뒤였다.
P4 가 별도 재실행으로 메웠으나 **배치 자체는 여전히 기준선 셸을 기록하지 않는다.**
⟹ 처방: **PREREG 에 "각 팔이 기록해야 할 필드"를 명시하고 게이트로 검사하라.**
HOSGATE 게이트 G-1~G-7 중 팔별 필드 존재를 검사하는 것이 **없었다.**

### ★P4 재실행이 부수적으로 드러낸 것 — 기여 ⑥ 에 대한 네 번째 타격
스크립트 docstring: **Gaspari-Cohn 국소화 `loc_radius=50 m` 는 `2c=100 m` 밖에서 항등적으로 0**
⟹ 모든 관측점에서 100 m 넘게 떨어진 격자점에서 **LETKF 해석 증분이 정확히 0**이다.
즉 기준선은 먼바다를 **구조적으로 갱신할 수 없다**. 이것은 P2(백색 배경공분산) · P3(Q3) ·
P4(스윕 클래스 불일치 `ETKFGrid` vs 배포 `LETKFGrid(loc50)`) 와 **독립된 네 번째 경로**로
같은 결론에 닿는다 — **기여 ⑥ "공정한 ETKF 기준선" 은 방어 불가.**

### 아직 열려 있는 것
**P4 주장 A** — *"기존 `SUPPORT-TRADEOFF` 7회 재현은 전부 WaveSplat 내부 처치 대조였고
method-vs-baseline 은 한 번도 없었다"* → P4 스스로 **`ARGUED`** 로 남겼다(8개 배치의 팔 정의를
원자료로 확인하지 않음). ⟹ ⛔**규율 개명을 지금 승인하지 마라.** 처방 = 각 배치 러너의 팔 dict 를
읽는 **런 0회 감사 1건**.
