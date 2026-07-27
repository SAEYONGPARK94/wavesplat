# PREREG — v1/v3 배포 확증 실험 (fresh reserve-seed 40)

**동결 시각: 2026-07-27 09:05 KST. 이 문서 저장 후 데이터 생성 시작. 게이트 수치를 본 뒤 어떤
바·문구·판정 규칙도 변경하지 않는다. (mixed_retrain PREREG와 동일 규율)**

## 0. 목적과 배경

혼합 재학습(`tmp/mixed_retrain_2026-07-26/`)이 v3 FULL PASS / v1 G2a-FAIL을 냈으나, 07-27
Opus 3각 검토(`tmp/v1v3_compare_2026-07-27/OPUS_REVIEW_SYNTHESIS_2026-07-27.md`)가 판정을
"확증 실험 필요"로 갱신:
- v3의 FULL PASS는 knife-edge 2건(G2a=bar 동값, lane 재추첨 P(PASS)≈0.30)
- v1 옹호 근거였던 "v3 paired 유의 악화 p=0.029"는 다중성·jackknife 불통과로 인용 불가
- 기존 40은 게이트에 28/40 소비 + 체크포인트 선택이 그 위에서 이뤄져 held-out 지위 소멸
- 신규 발견 2comp 회귀는 기존 게이트(G2b=1comp만)의 사각지대였음 → P5로 명시 겨냥
- 라우터 은퇴는 반증됨(v1+라우터 0.0768, v3+라우터 0.0814 — 둘 다 learned 단독보다 우위)
  → 후보는 "체크포인트+라우터(th=8)+guard 은퇴" 조합으로 정의

## 1. 데이터 (이 실험에서 처음 소비)

- **시드: 999, 1001** — heldout40 PREREG §3.1의 동결 예비 사다리 첫 2개.
  사전 감사(2026-07-27, 아래 §6): 파일 0건·코드/결과 참조 0건·07-26 사다리 walk 0건 = 미소비 확인.
- 구성: heldout40과 동일 20셀(S01-S10 × uniform/slope) × 2시드 = **40 에피소드**,
  mesh 200×167, adequacy 게이트 + 잔여 예비 사다리 [1111, 1212, 1313, 1414] 순서 소비, walk 전량 로깅.
- 생성기: `gen_confirm40.py` = 동결된 `gen_heldout40.py`에서 시드·출력경로·레이캐스트 경로만 변경
  (GPU 레이캐스트 = 07-24 finemesh 재생성의 원 인증 경로; CPU 경로와 bit-identical 인증 완료였음).
- 출력: `data_confirm40_2026-07-27/` (신규 디렉토리, 기존 데이터와 격리).

## 2. 팔과 프로토콜

순수 팔 3개, 각 에피소드 × 5 lane `(est_seed, diff_seed) ∈ {(42,0),(1,1),(2,2),(3,3),(4,4)}`,
에피소드 대표값 = **median-of-5** (프로젝트 규약):

| 팔 | init_method | 체크포인트 |
|---|---|---|
| `v1` | `learned` (명시) | `results/direction3/beamform_cond_diffusion_mixed_v1.pt` |
| `v3` | `learned` (명시) | `results/direction3/beamform_cond_diffusion_mixed_v3.pt` |
| `v2` | `v2_konly` (명시) | — |

- `TRACK_PHASE_PREDICT=False` 고정. `init_method` 전 팔 명시(⛔auto-default 금지, W9 규율).
- 실행 하네스 = `tmp/mixed_retrain_2026-07-26/run_arms_new.py`·`precompute_new.py`·
  heldout40 `v2_arm.py`를 에피소드 목록만 바꿔 사용(하네스 동일성은 mixed_retrain RESULT §2에서
  구 기록 대비 max|Δ|=0으로 검증된 그 코드).

**조립 팔(추가 GPU 없음, lane 단위 후처리)**: `auto_v1`, `auto_v3` =
lane별로 `n_selected ≥ 8`(auto_switch_thresh, 출하값)이면 learned lane, 아니면 v2 lane.
**guard 없음**(후보 정의가 "guard 은퇴"). 에피소드 대표값 = 조립 lane 5개의 median.

## 3. 하드 바 (후보 C ∈ {auto_v1, auto_v3} 각각 판정)

| 바 | 정의 | 근거 |
|---|---|---|
| **P1 안전** | 희소(1·2comp, 16ep)에서 에피소드 median nRMSE > 0.10 **0건** AND 희소 단일-lane 최댓값 ≤ 0.10 | 구 net 파국 0.329의 재발 차단, 마진 2배 |
| **P2 비열등** | 전체 40 mean ≤ v2 mean **+0.005** AND paired medianΔ(C−v2) ≤ **+0.002** | n=40 우월 주장은 사전 금지, 비열등만 |
| **P3 혼잡 가치** | 혼잡(15·30comp, 16ep) mean ≤ v2 mean **−0.03** AND 개선 에피소드 ≥ **12/16** | 학습팔의 존재 이유 (표본내 −0.041~−0.058) |
| **P4 5comp 무해** | 5comp(8ep) mean ≤ v2 mean **+0.005** | 표본내에서 v1↔v3를 가른 바 (v1+라우터 +0.0115 FAIL, v3+라우터 +0.0027 PASS) |
| **P5 2comp 무해** | 2comp(4ep) mean ≤ v2 mean **+0.005** | 07-27 신발견 회귀(구 게이트 사각지대) 겨냥 |

## 4. 판정 규칙 (동결)

1. **한 후보만 P1–P5 전부 PASS** → 그 후보가 배포 후보.
2. **둘 다 전부 PASS** → `auto_v1` 채택 **iff** 혼잡 재현 바 충족: 혼잡 16ep paired에서
   auto_v1이 auto_v3보다 ≥ **11/16** 우위(표본내 유일 견고 효과 13/16의 재현 판정).
   미충족이면 `auto_v3`(게이트 인가 기본).
3. **둘 다 FAIL** → 배포 없음, 현행 유지, 바별 실패 보고.
   부분 분기: P4/P5**만** 실패 시 "혼잡 전용 스코프(th=12) 축소안"을 **별도 사전등록 후** 검토
   — 오늘 배포로 이어지지 않음.
4. 사후 바 변경·재실행·lane 추가 없음. 이 40ep은 이 판정 1회로 소비됨을 기록.

## 5. 보고 전용 (판정에 불사용)

- R1: 순수 팔 3개의 슬라이스별 mean/median 표 + 에피소드별 paired 목록.
- R2: pure_v1 vs pure_v2 전체 40 paired median — 07-27 검토 ④-4("v1이 분석팔을 페어드로 이긴
  최초 팔" 주장)의 신선 재판정. **이 항목이 유의해도 배포 판정에는 불개입.**
- R3: 단일-draw(비median) 최악값·lane 스프레드(>0.02 에피소드 수) — 1시드 배포 현실 반영.
- R4: n_selected 분포(라우팅 census, 시드별) — G2a knife-edge 후속(20시드 재측정은 별건).
- R5: 사다리 walk 발생 시 그 목록.

## 6. 데이터 분리 사전 감사 (실행 전 완료, 2026-07-27)

| 항목 | 결과 |
|---|---|
| 훈련 장면 시드 대역 | `gen_mixed.py:280` `rng.integers(2e9, 2^31)` — 코드 확인 |
| 평가 시드 | 전부 < 2000 (777/888/999/1001/…): 구간 서로소 |
| 훈련-중 val 스트림 | master 987654321(양 대역 밖), **보고 전용** — `train_mixed.py:138` 저장은 무조건 최종 스텝(`torch.save` 1회), best-val 선택 없음. 라운드 선택은 게이트로만 |
| 예약 시드 999/1001 | 파일 0건(`find`)·참조 0건(`grep` tmp/scripts/results)·07-26 사다리 walk 0건(`seed_ladder_log.json`: 20셀 전부 base seed ACCEPT) |
| 기하 donor 14 | BF6/BF7 seed0 uniform + BF3/BF4 5comp seed2/3/4 slope — 확증셀(BF2–BF5, seed999/1001)과 시드·셀 모두 불겹침; (x,y,t)만 판독 |
| heldout40 구성 | 정확히 40ep(20×777, 20×888) + manifest — 오염 파일 없음 |
| 캐비엇(정직 기록) | val 스트림 장면 시드가 훈련과 같은 대역에서 독립 추첨 → 기대 ~2/256 장면 시드 충돌 가능. val은 보고 전용이므로 게이트 누수 경로 아님 |

## 7. 사전공약 문구

- **채택 후보 전부 PASS**: "혼합 재학습 체크포인트 {C}는 예약-시드 신선 표본에서 안전(P1)·
  비열등(P2)·혼잡 가치(P3)·중간/2성분 무해(P4/P5)를 재현했다 — 배포 인가. 단 실측 경로는
  별도 게이트(실측 n_selected 측정) 통과 전까지 `auto` 비활성."
- **P1 FAIL**: "재학습의 존재 이유(파국 제거)가 신선 표본에서 재현되지 않았다 — 배포 금지,
  현행 유지, 재학습 프로그램 재개."
- **P3 FAIL**: "학습팔의 혼잡 가치가 신선 표본에서 재현되지 않았다 — 학습팔 배포 철회,
  분석팔 전용 회귀. 혼잡 니치 주장(−23% 표본외)은 이 표본을 포함해 재판정."
- **P4/P5만 FAIL**: "수정은 부분적 — 혼잡 전용 축소 스코프를 별도 사전등록으로."
- 어떤 결과든 **"학습이 더 정확하다" 무조건형 문장 금지**(기존 ⛔ 유지), pooled 단일 수치
  헤드라인 금지(슬라이스별 보고만).
