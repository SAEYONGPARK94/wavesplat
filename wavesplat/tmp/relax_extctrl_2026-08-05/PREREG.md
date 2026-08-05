# PRE-REGISTRATION — 구성요소 수준 **외부** 대조: RELAX 초기화기 vs 우리 beamforming

작성 2026-08-05, **런 실행 전**. 근거: `tmp/blank_slate_2026-08-04/R5_IMPROVEMENT_LEDGER.md`
발견 #1 — *A1·A2·적응형-N·v1/v2·raydrop 전부 내부-내부이고, 외부 대조 도구는 완비인데
(`initializer.py:569` `init_method='relax'` + `scripts/compare_relax_init_e2e.py`)
`results/relax_init_e2e.json` 이 존재하지 않는다.*

## 0. 이 측정이 **아닌** 것 (선언)
우리 하네스 안에서 **초기화기 슬롯만 교체**한다. 따라서 측정 대상은
**"우리 트래커 하에서 우리 초기화기 vs 외부 초기화기"** 이지 **"우리 방법 vs RELAX 방법"이 아니다.**
⛔ 결과를 "WaveSplat vs Li&Stoica(1996)" 로 인용 금지.

## 1. 팔 (전부 `init_method` 명시 — W9 오염 방지)
공통 config = 2026-08-03 전파 2단계의 **WS-P1 그대로**:
`use_dispersion=True, use_confidence=True, adaptive_n=True, tracking_window=3`.

| 팔 | init_method | 계보 | 역할 |
|---|---|---|---|
| OURS | `v2_konly` | 내부 beamforming (k만, w/φ 무작위) | 대조군 = 배포 기본값 |
| **EXT-M** | **`relax_konly`** | **RELAX (Li & Stoica 1996)**, k만 — **w/φ 처리 일치** | ★**PRIMARY** |
| EXT-F | `relax` | RELAX, k/w/φ 전부 | 서술용 (처리 불일치) |

PRIMARY 를 `relax_konly` 로 두는 이유: `v2_konly` 는 w/φ 를 버리고 무작위로 뽑는다
(`initializer.py:583`). `relax`(full) 와 붙이면 **초기화기 차이 + w/φ 처리 차이**가 교락한다.
`relax_konly` 만이 **한 요인만** 바꾼다.

## 2. 데이터·시드 (08-03 프로토콜 복제, 재선택 없음)
- 데이터: `data_finemesh_2026-07-23/` (구 `data/` 는 메시-Nyquist 위반)
- 에피소드: **P1 12개 세트 그대로** (적응형-N 과 동일 ⟹ 비교 가능)
  `BF2_1comp_u, BF3_1comp_u, BF3_2comp_u, BF3_2comp_s, BF3_5comp_u, BF3_5comp_s,
   BF4_1comp_u, BF4_5comp_u, BF4_15comp_u, BF4_30comp_u, BF5_15comp_u, BF5_30comp_u`
- 시드: **[42, 123, 456] — median-of-3 + 산포** (WS 시드분산 bimodal CV~53%)
- 총 12 × 3팔 × 3시드 = **108 런** (스모크 실측 ~26 s/런 ⟹ ~47분)

## 3. 사전 지정 결과변수
- **PRIMARY**: e2e `avg_nRMSE`, 에피소드별 **3시드 중앙값**, EXT-M vs OURS **짝지음**
- SECONDARY-a: `init_nrmse` — 초기화기가 **직접** 내놓는 것 (구성요소 수준 본체)
- SECONDARY-b: EXT-F 팔 (서술)
- SECONDARY-c: **동결 층화** — on-support(≥5comp, n=7) / off-support(<5comp, n=5).
  임계는 08-03 에 CDIP 로 **동결**됨(≥12.18° ⟺ n_comp≥5). **재선택 금지.**
- 보고 의무: `avg_inference_ms`, `final_n`, 시드 std

## 4. 검정과 검정력 (**먼저** 적는다)
양측 부호검정, n=12: **12/12 → p=0.00049 · 11/12 → p=0.0063 · 10/12 → p=0.039**
⟹ α=0.05 는 **≥10/12** 에서만 도달. 보조로 Wilcoxon signed-rank 병기(검정명 명기 — ⛔#16).
- on-support n=7: 양측 바닥 **0.0156**(7/7) ⟹ α=0.05 도달 가능
- off-support n=5: 양측 바닥 **0.0625** ⟹ ★**원리적으로 α=0.05 불가, 서술 전용으로 선언**

## 5. 판정 규칙 (사전 확정)
**PASS(외부 대조 성립)** = PRIMARY 부호검정 p<0.05 **그리고**
|중앙 효과| > **에피소드별 시드 std 의 중앙값** (효과가 잡음보다 큰가).
두 조건 중 하나라도 미달 = **INCONCLUSIVE(검정력 부족)**, ⛔"null" 이라 쓰지 않는다(⛔#2).

## 6. 방향에 무관한 사전 공약
★**RELAX 가 이기면 그것은 우리 초기화기에 대한 음성 결과이고, 그대로 보고한다.**
특히 혼잡(≥15comp)에서 외부가 이기면, **"혼잡 벽"의 일부는 표현의 한계가 아니라
우리 초기화기의 결함**이라는 뜻이며 이는 R4 의 novelty 후보 (B)(표현-불일치 식별가능성)를
**약화시키는 방향**이다. 이 해석 경로를 **지금** 적어 둔다.

## 7. 스모크(참고, 시드 1개 — 판정 아님)
`BF3_1comp_u`: OURS 0.0082 / EXT-M 0.0126 / EXT-F 0.0159
`BF4_30comp_u`: OURS 0.3015 / **EXT-M 0.2593** / EXT-F 0.3248
⟹ 방향이 레짐에 따라 갈릴 수 있다는 **힌트**. 단일 시드이므로 ⛔인용 금지.
