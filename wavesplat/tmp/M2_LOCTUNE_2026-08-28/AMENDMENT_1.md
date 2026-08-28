# AMENDMENT 1 — M2_LOCTUNE (2026-08-28)

★★**시점: 결과 0행.** `results_m2*.jsonl` 0개, `verdict_m2*.json` 0개(§7 스모크 산출물은
동결 전에 삭제됨). ⟹ **사후 수정이 아니다** — 어떤 결과도 이 수정 전에 존재하지 않았다.

⛔**판정 규칙 무변**: `gates_m2.py`(문턱·팔·표본·라벨·해석 규칙·집계)에 손대지 않았다.
수정 파일은 `run_m2.py` **한 줄의 출력 파일명**뿐. 수정 후 `gates_m2.py --selftest` PASS ·
런0회 게이트 **12/12** 재확인.

## [MAJOR] 양성대조(PC) 출력 파일명 `results_m2_pc.jsonl` → `results_m2_9.jsonl`

- **결함**: 검사기 `tmp/_audit/check_batch_artifacts.py` 의 `_sfx()`(:149, `SHARD = r'_\d+$'`)
  는 끝의 `_<숫자>` 만 벗긴다 — `results_m2_pc.jsonl` 은 접미사 **`m2_pc`** 가 되고,
  접미사마다 `verdict_<sfx>.json` 을 요구하므로 `verdict_m2_pc.json` 부재로
  **`paired=False` = 인용 불가**. 적대적 관문이 가짜 배치로 **실제 재현**했다.
  (같은 날 M3 는 옳게 했다 — PC 를 jsonl 이 아니라 `pc2_m3.json` 에 쓴다.)
- **수정**: `run_m2.py:134` 부근 파일명 dict 의 `'pc'` 값만 교체. **9 는 본배치 샤드
  0–7 밖의 예약 숫자**다. 검사기 실측: `_sfx('results_m2_9.jsonl','results') == 'm2'`
  ⟹ `verdict_m2.json` 과 짝지어진다.
- **직접 확인한 두 사실**(수정의 전제):
  1. 분석기 glob — `analyze_m2.py:15` `results_m2*.jsonl` 이 새 이름을 그대로 집는다.
  2. PC 행 제외 — 행 식별은 파일명이 아니라 러너가 심는 `pc` 필드다
     (`run_m2.py` `pc=(which=='pc')`); `gates_m2.aggregate()`(:605) 가
     `not r.get('pc')` 로 본집계에서 제외하고, PC 게이트는 `r.get('pc')` 행만 집는다(:563).
- ⛔**가지 않은 길**: `gates_m2.py:368` 의 미사용 `PC_PATH` 경유 — `gates_run` 서명이
  바뀌어 판정 코드 수정이 된다. 배제.

## 변경 파일

- `run_m2.py` — `'pc'` 출력 파일명 1건 (+ 사유 주석)
- `FROZEN_md5.txt` — `run_m2.py` md5 1건 갱신 + 헤더에 재기록 사유 명시
  (`gates_m2.py`·`analyze_m2.py`·`PREREG.md`·임포트 대상 md5 는 전부 원본과 동일)
