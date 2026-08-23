# 배포 영향 census — `bank_refit_every_s: None → 5.0` (2026-08-23)

저장소 규율: **배포 기본값을 바꿀 때는 어떤 배치가 재실행에서 무효가 되는지 먼저 세라**
(`lambda_l1` 08-11 배포가 이 census 없이 나가서 S1–S8 을 "재실행 시 무효" 로 만들었다).

## ✅기록된 결과는 **전부 무손상**
이 키는 **실측 경로에서만** 의미가 있고(`_bank` 는 실측 게이트 안에서만 채워진다),
**이미 저장된 숫자는 그때의 값으로 계산됐다.** 바뀌는 것은 **재실행**뿐이다.

## 1. 재실행해도 **불변**(키를 명시한다)
`B-3` · `B-3b` · `B-3c` · `B-4` · `B-6` — 팔마다 `bank_refit_every_s` 를
**명시적으로** 준다(`FROZEN`/`PERSIST` 는 `None`, 나머지는 각자의 τ).
⟹ ★**오늘의 정확도 사슬 전체가 재실행에 안전하다.**

## ⚠2. 재실행하면 **달라지는 것** (키 미명시 + bank 경로 도달)
| 배치 | 어느 팔 | 왜 |
|---|---|---|
| `B-2` | **`SHIP`** | 배포 라우터로 bank 에 가고 키를 안 준다 ⟹ 이제 5 s 마다 재적합 |
| `B-2b`/`c`/`d`/`e`/`f` | (없음) | 전부 `PROP`(`real_domain_gate=False`) 이라 **bank 에 안 간다** ✅ |
| `B-6b` | 렌더 측정 | `DEFAULT_CONFIG` 를 쓰고 키 미명시 ⟹ 루프 중 재적합이 끼면 타이머가 오염된다 |
| `B-6c` | 프레임 측정 | 같은 사유 |
| `W12` | `DEPLOYED` | 실측 배포 팔, 키 미명시 |
| `A2`·`G1`·`blind_auto`·`bankmmse_port`·`mixed_retrain` | 실측 팔 | 전부 배포 전 배치, 키 미명시 |

★**처방(즉시 적용)**: 위 배치를 **재실행할 일이 생기면 `pre_2026_08_23_config()` 를 먼저 넣어라.**
`B-2` 의 `SHIP` 숫자를 다시 쓸 일이 있으면 **반드시** 그렇게 해야 한다.

## 3. 합성은 **원리적으로 불변**
`_bank_update` 는 `self._bank is not None` 일 때만 불리고 `_bank` 는 실측 게이트 안에서만
채워진다. 합성 프레임은 `gt_field` 를 갖고 있어 게이트를 통과하지 못한다.
✅**코드 불변량 시험으로 고정**: `test_bank_refit_cannot_affect_synthetic_runs`.

## 4. 되돌림
`sequential_update.pre_2026_08_23_config()` → `{'bank_refit_every_s': None}`.
2026-08-22 의 세 스위치까지 되돌리려면 `pre_2026_08_22_config()` 를 **함께** 쓴다.
✅양방향 거동을 `test_bank_refit_default_refits_and_revert_freezes` 가 고정한다.
