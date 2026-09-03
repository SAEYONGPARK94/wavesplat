# `H3b-LAW` 사전등록 — **틀린 것이 「모형」인가 「검출 법칙의 부류」인가**

> **동결 시점: 결과 0행.** ⛔**추정기 런 0회 · 신규 계측 0회** —
> `H3-FRF` 의 동결 셀(`cells_h3.npz`, 236,423 셀)을 **다시 읽기만** 한다.
> 선례 = `B-20b`·`R7`(`REANALYSIS-OF-FROZEN-DATA`).

---

## §0 왜 — `H3` 는 *"이 모형이 안 맞는다"* 만 말했다

어제 `H3` 가 배포 raydrop 모형(`raydrop_uniform`, **Model A**)을 실물 Velodyne 반환율에
대고 **MRE 0.3479 > 문턱 0.20** 으로 기각했고, 치환 대조가 *"거의 신호가 없다"* 를 보였다.

⚠**그런데 그것은 두 가지를 구분하지 않는다**:
1. **복사량 `A` 가 틀렸다**(θ·d 의존성 자체가 반대) — `raydrop_uniform.py:20-31` 이 스스로 적은 문제
2. **검출 법칙의 부류가 틀렸다** — `A` 는 맞는데 *"반환율 ∝ A"* 라는 **곱셈-이득 베르누이**가 틀렸다

★**저장소에 두 번째 후보가 이미 있다**: `wavesim/raydrop_threshold.py`(**Model C**).
같은 `A` 를 쓰되 **스페클 + 문턱 검출**이라 `M=1` 에서 **반환율 = exp(−T/A)** 다.
그 파일의 독스트링이 두 부류가 다르다고 축자로 적는다 —
*"raydrop_uniform is Bernoulli-with-gain (P = eta*A, **multiplicative**)"*.

⚠**Model B(`raydrop_slope`)는 이 검정에서 제외한다** — 같은 독스트링이
*"raydrop_slope's specular gate **never fires at this geometry** … and falls back to
constant-P"* 라 적는다. 상수 모형은 이 자로 잴 것이 없다. **결과 전 배제**다.

---

## §1 자 — `H3` 규약을 **한 글자도 바꾸지 않는다**

| | Model A (배포) | Model C (문턱) |
|---|---|---|
| 예측 반환율 | `s · A` | `exp(−T / A)` |
| 자유 모수 | **1개** (`s`) | **1개** (`T`) |
| 적합 규칙 | **중앙값 정합** — `median(pred) = median(rate)` | **같다** |
| 자 | `MRE = median(|pred − rate| / rate)` | **같다** |

★**적합 규칙을 같게 두는 것이 이 배치의 공정성 전부다.** 최적화로 `T` 를 고르면
Model C 에 자유도를 하나 더 주는 셈이다 ⟹ ⛔**최소화 금지, 중앙값 정합만**(이분법으로 푼다).

⛔`A` 는 `H3` 가 저장한 `P` 열을 **그대로** 쓴다(재계산하지 않는다).

---

## §2 PRIMARY — 결과 전 고정

**문턱은 `H3` 의 것을 상속한다: `MRE ≤ 0.20`**(출처 = `thesis_main.tex` 제5장 자신의 수).

`has_signal(모형)` = 그 모형의 **자기 치환 귀무 5백분위보다 작다**
(⛔`H3` 에서 Model A 는 여유가 **1.3%** 뿐이었다).

### 결과 전 동결 판정표

| 조건 | 라벨 |
|---|---|
| `MRE_C ≤ 0.20` **그리고** `has_signal(C)` | **`THRESHOLD-LAW-FAITHFUL`** — ★틀린 것은 **부류**였고 옳은 것이 **이미 저장소에 있다** |
| `MRE_C > 0.20` · `has_signal(C)` · `MRE_C < MRE_A` | **`LAW-FAMILY-MATTERS-NOT-ENOUGH`** — 부류가 문제의 일부이나 충실하지 않다 |
| 그 밖 | **`LAW-FAMILY-NOT-THE-ISSUE`** — ⛔*"부류를 바꾸면 된다"* 반증. 복사량 `A` 쪽을 봐야 한다 |

---

## §3 ★결과 전 예측 (⛔결과를 보지 않고 적는다)

**`LAW-FAMILY-MATTERS-NOT-ENOUGH`** 를 예측한다. **확률 0.60**
(`THRESHOLD-LAW-FAITHFUL` 0.20 · `LAW-FAMILY-NOT-THE-ISSUE` 0.20).

**사유**: 관측 반환율은 **1 에서 포화**하는데 `s·A` 는 위로 열려 있다(`A` 의 실측 범위가
9.3e−05 … 52.1 로 **다섯 자릿수**다) ⟹ 곱셈-이득은 구조적으로 큰 `A` 에서 과대예측한다.
`exp(−T/A)` 는 포화를 갖는다. 그래서 **더 나을 것**이라 본다. 그러나 `H3` 의 음성대조가
`A` 자체에 신호가 거의 없다고 했으므로 **부류만 바꿔서 0.20 아래로 내려가진 않을 것**이다.

---

## §4 차단 게이트 (⛔사후 완화 금지)

| # | 게이트 |
|---|---|
| ★`G-FROZEN` | `cells_h3.npz` · `verdict_h3.json` 의 md5 가 `FROZEN_md5.txt` 와 일치 |
| ★`G-REPRO` | **Model A 의 `MRE`·`scale` 이 `verdict_h3.json` 을 재현**(상대오차 < 1e−9) — 자가 재현 |
| ★`G-FIT` | 두 모형의 적합이 **같은 규칙**(중앙값 정합)이고, 적합 후 `median(pred)/median(rate)` 가 두 모형 모두 1 ± 1e−6 |
| ★`G-IMPORT` | Model C 를 `wavesim.raydrop_threshold` 에서 **임포트**해 `received_power` 로 `A` 를 재계산했을 때 `H3` 의 `P` 열과 **상대오차 < 1e−9** (⛔재구현 아님을 증명) |
| `G-NEG` | 두 모형 각각의 치환 귀무를 **같은 반복수·같은 시드**로 돌린다 |
| `G-EXCL` | Model B 배제 사유가 **코드 독스트링에서 인용**된 것임을 행에 기록 |
| `G-CELLS` | 셀 수 = **236,423**(`H3` 와 동일) |
| `G-BLOCK` | `blocked_by()` **자가진단** — 합성 실패행으로 이름을 잡는지 확인 |

---

## §5 ⛔범위와 금지

- ⛔**`H3` 판정 갱신 아님** — `OBSMODEL-NOT-FAITHFUL` 은 그대로다.
- ⛔**배포 기본값 변경 아님.** 어떤 결과가 나와도 `wavesim/**` 는 **무수정**이고,
  코퍼스 재생성은 **별도 사전등록**이다.
- ⛔**정확도 주장 아님** — 추정기 런 0회.
- ⛔**쇄파대 레짐 한정**(FRF Duck) · 심해 미검 · **역방향 전이 교락**은 `H3` 에서 상속.
- ⛔`M` 은 **1 로 고정**한다(`raydrop_threshold` 독스트링이 *"M = 1 is fully-developed …
  the standard model"* 이라 적는다). ⚠`M` 을 흔드는 것은 **모수를 늘리는 일**이므로
  **보고 전용 SECONDARY** 이고 ⛔판정에 쓰지 않는다.
- ⛔**`A` 가 옳다고 주장하지 않는다** — 이 배치는 `A` 를 고정한 채 **법칙만** 바꾼다.
