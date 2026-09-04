# `T12-ISOLATE` 개정문 2 — ⚠**결과 이후** · 분석기 전용 · 새 런 0회

작성: 2026-09-05. 계기: **소급 구현 감사**(다중 에이전트).
⛔**판정 불변** — `MULTIPLE-AXES-CARRY` 그대로. 게이트를 완화하지 않았다.

## 1. ⛔`G-SHIPPIN` 의 「손복사 0」이 **16종 중 3종에서 거짓**이었다

`consts_t12.py:230-232` 의 `deployed()` 가 이렇게 낸다:
```python
'learned_dedup_dk': float(DEFAULT_CONFIG.get('learned_dedup_dk', 0.03)),
'max_steepness':    float(DEFAULT_CONFIG.get('max_steepness', 0.15)),
'grow_krange': 4.0,     # ⚠하드코딩 리터럴이라고 주석까지 달아놓고 핀했다
```
**실측: 세 키 모두 `DEFAULT_CONFIG` 에 없다** ⟹ 반환값은 **폴백 리터럴**이고, 그것을
`SHIP_PIN`(`:96-97`)의 **같은 리터럴**과 비교한다 = ***자기 자신과의 대조***.
실제 정의는 `sequential_update.py:525`(0.15) · `learned_init.py:344`(0.03) ·
`sequential_update.py:1692-1693`(4.0) 이고 **어느 것도 임포트되지 않는다**.
⟹ 저 셋 중 무엇이 바뀌어도 `G-SHIPPIN` 은 영원히 *"불일치 없음"* 이다.
동반: `run_t12.py:58` 의 `grow_krange` 되읽기 폴백도 같은 `4.0` 이라 **양쪽이 동시에 눈이 먼다**.
★저장소가 이름 붙인 **「손복사 기준 상수」** 부류의 재발이다.

**해소**: ⛔측정면(`consts_t12.py`)을 고치지 않는다. 분석기에 ★**`G-SHIPPIN-EXT`** 를 신설해
**소스 리터럴을 정규식으로 직접 뽑아** 핀한다(`T14 G-SRC-PIN` 규약).
→ **판정 불변**: 라이브 값이 핀과 실제로 같음을 확인했다. 깨진 것은 **탐지기**다.

## 2. SEC-R 112행이 **측정면 검증 밖**에 있었다

`run_t12.py:312-314` 는 SEC-R 을 `results_t12_SECR.jsonl` + `meta_t12_SECR.json` 에 쓰는데
`analyze_t12.py:23` 은 `results_t12.jsonl`/`meta_t12.json` **만** 읽는다.
⟹ `results_t12.jsonl` 의 뒤 112행은 **내가 손으로 이어붙인 것**이다(러너는 `open(out,'w')` 다).
`PREREG.md` 에 `_SECR` 파일명은 **0회** 등장한다.
결과: `G-FREEZE`·`G-ENV` 가 대조하는 `code_md5`·`threads_readback` 은 **PRIMARY 프로세스 것뿐**이고,
SEC-R 프로세스의 meta 는 **어느 코드도 열지 않았다**.

**해소**: ★**`G-SECR`** 신설 — SEC-R meta 가 존재하고 `code_md5` 가 PRIMARY 와 **동일**하며
`threads_readback == 1` 이고, `results_t12.jsonl` 의 꼬리 112행이 원본 `_SECR` 파일과
**행 단위로 일치**함을 확인한다.
→ **판정 불변**: 두 meta 의 `code_md5` 차이 키가 0 임을 확인했다. 그러나 **게이트가 확인한 것이
아니라 사람이 확인한 것**이었다 — 이제 게이트가 한다.

## 3. ⚠공시 두 건 추가 (읽기를 좁힌다)

- ⛔⛔**`C-MINDK` 의 「널」은 75% no-op 위에 있다**: `verdict['effect_vs_ship']['C-MINDK']` =
  **"6/24 다름"** ⟹ **18/24 블록이 `SHIP` 과 비트 동일**이고, 그래서 그 9/24 는 사실상
  `SHIP` 앵커(8/24)의 **재인쇄**다. `G-EFFECT` 는 **전건 no-op** 만 막고, 성장이 24/24
  발화했으므로 `not_exercised` 도 비어 있다. ⟹ **6/24 병기 없이 이 축의 널을 인용하지 마라** —
  「효과 없음」이 아니라 **「지렛대 부족」 후보**다(이 배치 자신의 사전등록 규율).
- ⚠**라벨이 한 블록 차이로 뒤집힌다**: `A-MAPCELL` 의 `w = 19` 이고 `THRESH_SEC = 19` 다.
  `label(24, 18)` → `PROPOSAL-SCALE-CARRIES` · `label(24, 19)` → `MULTIPLE-AXES-CARRY`.
  ⟹ **강건성 진술 없이 라벨을 인용하지 마라.** (현 자료로 라벨은 맞다 — 경계에 있을 뿐이다.)

## 4. ⚠고치지 않고 기록만 하는 것
- `G-RNGSTREAM`·`G-GROWFIRE` 는 통과조건이 *"값이 존재한다"* 뿐이라 **FAIL 불가**다.
  사전등록 §6 이 둘을 **「기록」 게이트**로 적었으므로 사전등록 위반은 아니다 — 이름과 코드의
  드리프트다. ⚠특히 `G-RNGSTREAM` 은 상세에 *"진행한 행 **0/424**"* 를 찍는다 —
  이름이 지목한 조건이 전건 성립하는데 그것을 판정하지는 않는다.
- 인쇄된 개수 2건이 코드와 다르다: `G-SHIPPIN` 이름이 *"17종"* 인데 `SHIP_PIN` 은 **16키** ·
  `G-ENV` 이름이 *"ckpt sha256 2개"* 인데 실제 핀은 **하나**(`setdiff_prior.pt` 는 리터럴 대조 없음).
- 측정면 동결이 **파일 삭제**를 못 본다(존재하는 것만 담고 `run_keys` 도 거기서 파생).
  분석면은 합집합이라 삭제를 잡는다 — **같은 파일 안에서 두 규약이 다르다**.
- `(x or 기본값)` 자구가 `analyze_t12.py:163`·`run_t12.py:278` 에 잔존한다(오늘은 불활성:
  `GT_CONST=0.5≠0` · `kp_true>0`). ⚠`T14 AMENDMENT_1 §1` 의 일소가 **N−1 에서 멈췄다**.
