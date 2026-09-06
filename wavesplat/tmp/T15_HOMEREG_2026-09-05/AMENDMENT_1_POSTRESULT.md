# `T15-HOMEREG` **개정문 1 (결과 후)** — ⛔등가 판정이 **사전등록을 어기고 인쇄됐다** (2026-09-06)

> ⛔**판정(`label = HOME-REGRESSION`)도 계수도 바꾸지 않는다.** 동결 파일(`gates_t15.py` ·
> `analyze_t15.py` · `results_t15_*.jsonl` · `verdict_t15.json`) **무수정**.
> 이 문서는 **하나의 부차 출력(`equivalence`)을 철회**한다.
> 발견 = 2026-09-06 실험설계 감사(외부 렌즈) · **확인 = 내가 직접 되읽었다**.

## 1. 사전등록이 요구한 것 (`PREREG.md:219-220`, 축자)

> ⚠**`FLOOR-VPLR` 은 `T12` 에서 측정된 널**이다 — 이 코퍼스에서 다시 널이라는 보장은 없다.
> **널이 아니면 `verdict['floor_moved']=True` 로 공시하고 등가 판정을 인쇄하지 않는다.**

## 2. 실제로 일어난 것 (되읽기)

| 되읽은 것 | 값 |
|---|---|
| `verdict_t15.json → floor_moved` | ★`{"crowded": true, "sparse": true}` — **둘 다 True** |
| `verdict_t15.json → equivalence.sparse` | ⛔`{"floor_median_abs_rel": 0.1654, "effect_median_rel": 0.0, "within_floor": true}` — **인쇄됐다** |
| `verdict_t15.json → equivalence.crowded` | `{"floor_median_abs_rel": 0.1058, …, "within_floor": false}` — 역시 인쇄됐다 |
| `verdict_t15.json → curve.sparse.FLOOR-VPLR` | ★`w = 41`, `n = 64`, `median_rel = −0.0296` |
| `consts_t15.py:26` | `THRESH_HELP = 41` |

⟹ ★★***희소 층에서 「무해 노브」 바닥 팔이 **이김 문턱에 정확히 닿았다**(41/64).***
그것은 **널이 아니다** — 사전등록의 조건이 발동했는데 **등가 판정이 그대로 인쇄됐다.**

## 3. 왜 막히지 않았나 — ★**가드가 구조적으로 항진이다**

`analyze_t15.py:419`:
```python
'floor_moved': {L: (floor[L] is not None and floor[L] > 0.0) for L in LAYERS},
```
`floor[L]` 은 **절대 상대편차의 중앙값**이므로 **0 이 될 수 없다** ⟹ `floor_moved` 는 **항상 True** 다.
⟹ 그 값이 True 인 것은 *"바닥이 움직였다"* 를 뜻하지 않고 **아무것도 뜻하지 않는다**.
그리고 `:416-418` 은 `floor_moved` 와 **무관하게** `equivalence` 를 인쇄한다 —
사전등록이 요구한 **「인쇄하지 않는다」가 코드에 존재하지 않는다.**

★**부류**: 이 저장소가 이름 붙인 ***"자기진단이 통과하고 결과만 틀린다"*** 와
***"항상 통과하는 검사는 항상 막는 검사만큼 장식이다"*** 의 교집합이다.
`T5-ENS` 의 `blocked_by()` 항진(차단 게이트 14개가 장식이 될 뻔했다)과 같은 부류.

## 4. ⟹ 처분

1. ⛔**`verdict_t15.equivalence` 전건 인용 금지** — `crowded.floor_median_abs_rel = 0.1058` ·
   `sparse.floor_median_abs_rel = 0.1654` · `sparse.within_floor = true` **셋 다**.
   ★특히 그 두 여백을 **다른 배치의 등가한계(margin)로 쓰지 마라** — 바닥이 아니라 **처치**다.
2. ✅**나머지는 무손상**: `label = HOME-REGRESSION` · `control`(`SHIP<ZERO` 63/64 두 층) ·
   `curve` 계수 · `medians` · 게이트. 이 개정문은 **하나의 부차 출력만** 철회한다.
3. ★**후속 배치가 「개입 바닥」 팔을 쓸 때 `vp_lr` 을 쓰지 마라** — 이 코퍼스에서 널이 아니었다.
   새 바닥 팔은 **그 코퍼스에서 널임을 먼저 보여야** 한다.
4. ⚠**`T12` 에서 `D-VPLR` 이 널이었다는 것은 여전히 참**이다(10/24). 무너진 것은
   *"그러므로 홈에서도 널이다"* 라는 **이전**이다. ⟹ ★**바닥은 코퍼스마다 다시 재야 한다.**

## 5. ⛔고치지 않는 것

`analyze_t15.py` 를 수정하지 않는다(동결물이다). 대신 **후속 배치의 사전등록**이
`floor_moved` 를 **거동으로** 정의해야 한다 — 예: *"바닥 팔의 `w` 가 `[THRESH_HURT, THRESH_HELP]`
밖이면 `floor_moved=True`"*. ★그리고 그 가드에 **변이 시험**을 붙여라(오늘 `learned_polish_iters`
에서 한 것처럼) — 항진 가드는 시험 없이는 다시 태어난다.
