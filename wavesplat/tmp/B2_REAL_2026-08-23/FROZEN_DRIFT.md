# 동결물 이동 기록 (RV-5 지적, 2026-08-23)

검사기(`tmp/_audit/check_batch_artifacts.py` I열)가 잡은 md5 불일치 **전건의 사유**를 적는다.
⛔**재동결하지 않는다** — witness 가 *"HEAD 에서 비트 재현 불가"* 를 말하는 것이 **정상 동작**이다.

| 동결 파일 | 대상 | 사유 | 결과 무손상? |
|---|---|---|---|
| `FROZEN_md5_b2f.txt` | `wavesplat/optimizer/sequential_update.py` | **B-2f 실행 후** `SCALE_INVARIANT_GROW` 와 `:1220` 에 **실측 측정 주석 추가**(코드 무변경) | ✅ 주석뿐 |
| `FROZEN_md5_b2f.txt` | `tmp/B2_REAL_2026-08-23/scene_station.py` | **B-2f 실행 후** ①BS 7개 `depth` 17→30 수정(`AMENDMENT_ZEROREF §5`, 측정된 효과 Δω **0.0000–0.0012%**) ②`with_future_obs` 옵션 추가(**기본 False, 비트동일 확인**) ③사전등록 문언 정정 | ✅ **두 변경 모두 기본 경로 비트동일** |
| `B3/FROZEN_md5.txt` | `tmp/B3_REFIT_2026-08-23/run_b3.py` | **B-3 실행 전** `FrameData` 인자 수리(스모크가 잡음). **수정 후 md5 를 append 했고 옛 줄을 안 지웠다** | ✅ 실행은 새 판으로 |
| `B3/FROZEN_md5.txt` | `gates_b3b.py` | 결과 후·분석기 전 **판별자 철회**(독스트링만). **재동결함** = `AMENDMENT_B3B_MD5.md` | ✅ 판정 규칙 불변 |

★**교훈**: `FROZEN_md5` 는 **append 하지 말고 갱신**하되, **갱신했다는 사실과 사유를 반드시 남겨라.**
옛 줄을 남겨 두면 witness 가 **거짓 경보**를 낸다(위 3행이 그 사례다).
