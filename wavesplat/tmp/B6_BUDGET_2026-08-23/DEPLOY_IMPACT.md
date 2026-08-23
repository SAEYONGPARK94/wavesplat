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

## ⛔2. 나의 첫 census 가 **과대**였다 — 정정 (같은 날, 런 0회)

처음 나는 *"`B-2` 의 `SHIP` 이 재실행에서 달라진다"* 고 적었다. **틀렸다.**
재적합은 **`update()` 안에서만** 발화한다(`sequential_update.py:1043`, `_bank_update` 내부).
⟹ **`update()` 를 부르지 않는 배치는 이 키에 영향을 받지 않는다.**

**저장소 전수 grep**(`upd.update(`) 결과 — 실측 추정기의 `update()` 를 부르는 러너는 **여섯 개뿐**:
`run_b3.py` · `run_b3b.py` · `run_b3c.py` · `run_b4.py` · `run_b6.py` · `run_b6c.py`.
그리고 **앞의 다섯은 키를 명시한다.**

| 배치 | `update()` 호출 | 키 명시 | 재실행 영향 |
|---|---|---|---|
| `B-2` ~ `B-2f` | ❌ **0회**(`cfg.update()` 는 dict 메서드다) | — | ✅**없음** |
| `W12` · `B-6b` | ❌ 0회 | — | ✅**없음** |
| `A2`·`G1`·`blind_auto`·`bankmmse_port`·`mixed_retrain` | ❌ 0회(확인) | — | ✅**없음** |
| `B-3`·`B-3b`·`B-3c`·`B-4`·`B-6` | ✅ | ✅ | ✅**없음** |
| ⚠**`B-6c`** | ✅ | ❌ | ⚠**달라진다** — 루프 중 재적합이 끼어 `t_upd` 타이머가 오염된다 |

⟹ ★***영향을 받는 배치는 저장소 전체에서 `B-6c` 하나뿐이다.***
⚠**내 grep 이 처음에 `cfg.update(...)`(dict 메서드)를 추정기 호출로 오인했다** — 오늘 하루
*"같은 이름이 두 가지를 뜻한다"* 부류에 또 물린 것이다.

★**처방**: `B-6c` 를 재실행할 일이 생기면 `pre_2026_08_23_config()` 를 먼저 넣어라.
그 외 배치는 이 키와 무관하다.

## 3. 합성은 **원리적으로 불변**
`_bank_update` 는 `self._bank is not None` 일 때만 불리고 `_bank` 는 실측 게이트 안에서만
채워진다. 합성 프레임은 `gt_field` 를 갖고 있어 게이트를 통과하지 못한다.
✅**코드 불변량 시험으로 고정**: `test_bank_refit_cannot_affect_synthetic_runs`.

## 4. 되돌림
`sequential_update.pre_2026_08_23_config()` → `{'bank_refit_every_s': None}`.
2026-08-22 의 세 스위치까지 되돌리려면 `pre_2026_08_22_config()` 를 **함께** 쓴다.
✅양방향 거동을 `test_bank_refit_default_refits_and_revert_freezes` 가 고정한다.


## 5. 동결 이동(md5) 중앙 기록 — 2026-08-23 밤

검사기(`tmp/_audit/check_batch_artifacts.py` I열)가 오늘 배치들에서 md5 불일치를 보고한다.
**전건 사유가 이것이다**: 배포(5.0)와 **같은 날 철회**(None)가 다음 파일들을 바꿨다.

| 파일 | 무엇이 바뀌었나 | 결과 무손상? |
|---|---|---|
| `wavesplat/optimizer/sequential_update.py` | `BANK_REFIT_EVERY_S_DEFAULT` 신설 → 5.0 → None · `recommended_real_refit_config()` 추가 · 폴백을 상수로 · 주석 다수 | ✅**팔마다 키를 명시한 배치는 불변** |
| `wavesplat/run.py` | `DEFAULT_CONFIG` 항목 · yaml 로더가 키를 읽게 · CLI `'none'` 지원 | ✅동일 |
| `tmp/B3_REFIT_2026-08-23/run_b3.py` | 런 **전** `FrameData` 인자 수리(옛 md5 줄을 안 지웠다) | ✅실행은 새 판으로 |

⟹ ★**md5 witness 가 `--` 인 것은 *정상 동작*이다** — *"HEAD 에서 비트 재현 불가"* 를 말하는 것이
그 witness 의 일이다. ⛔**재동결로 지우지 않는다.** 사유를 여기에 남긴다.
⚠**한 가지는 진짜 결함이었다**: `run_b3.py` 는 옛 줄을 **지웠어야** 했다(append 가 거짓 경보를 낸다).
★**처방(재확인)**: `FROZEN_md5` 는 **append 하지 말고 갱신하되 사유를 남겨라.**
