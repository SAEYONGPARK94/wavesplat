# B-3b 동결물 재동결 기록 (2026-08-23, RV-5 지적)

`gates_b3b.py` 의 현재 md5 가 `FROZEN_md5.txt` 의 값과 다르다.
- 동결값 `034ad719cc7b0dccda7301c131275ed7` → 현재 `23fabca2b443fec59b74f3f7716b1ac3`
- mtime 14:03:13 이 `results_b3b.jsonl`(14:02:28)보다 **45초 늦다.**

**사유**: 결과 파일이 완성된 뒤, **분석기를 돌리기 전에** 사전등록의 두 번째 목적
(누수 vs 유효수명 판별자)을 **철회**했다(커밋 `b574e46`). 변경은 **독스트링뿐**이다.

**확인**: `ARMS`·`ARM_ORDER`·`REUSED`·`REF`·`NOMINAL`·`FLAT_PP`·`gates()`·`label()` —
**판정에 쓰이는 것은 전부 불변**이다. 따라서 `verdict_b3b.json` 은 무손상.

⛔**그러나 md5 witness 가 지금 거짓 경보 상태였다**(재동결도 이 기록도 없었다).
✅**여기서 재동결한다** — 아래 값이 철회 이후의 정본이다.
23fabca2b443fec59b74f3f7716b1ac3  tmp/B3_REFIT_2026-08-23/gates_b3b.py
