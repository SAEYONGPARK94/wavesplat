# ⚠동결 불일치 3건 — `wavesplat/optimizer/sequential_update.py` (2026-09-03 심야)

**세 배치의 `FROZEN_md5.txt` 가 이 파일의 md5 를 핀하는데 오늘 그 파일을 고쳤다**:
`E1_EXTBASE_2026-09-03` · `E1B_EXTBASE_2026-09-03` · `R8_HELDOUT_2026-09-03`.
`tmp/_audit/check_batch_artifacts.py` 의 `md5` 열이 그 셋에서 `--` 다. **지적은 옳다.**

⛔**숨기지 않는다.** 아래는 *"판정이 바뀔 수 있는가"* 에 대한 증거다.

## §1 무엇을 고쳤나 — 변경 훅 전부 (`git diff 108219ed HEAD`)

1. 생성자에 `self._init_method_requested = self.init_method` **추가**
   ⟹ 값을 복사해 보관할 뿐, 동작 없음.
2. 게이트 진입 조건 `if self.init_method == 'auto':` → `if self._init_method_requested == 'auto':`
   와 그 안의 `self.init_method = 'auto'` **추가**.
3. **읽기 전용** 추가: `resolved_init_method`(property) · `arm_provenance()` ·
   `MedianEnsembleUpdater.arm_provenance()`.

## §2 ★단일 `initialize()` 에서는 **동작이 동일하다**(논증이 아니라 코드다)

**첫 호출 시점에 `_init_method_requested == self.init_method` 다**(생성자가 같은 값을 넣고
그 사이에 아무도 바꾸지 않는다) ⟹
- 2번의 조건식은 **같은 값을 비교**한다 = 분기 동일.
- `self.init_method = 'auto'` 는 **이미 `'auto'` 인 값의 재대입** = no-op.
- 1·3번은 동작 경로에 없다(3번은 이 배치들이 **호출하지 않는다** — 전수 grep: 시험 파일 밖
  호출 지점 0개).

⟹ ***차이는 두 번째 `initialize()` 에서만 나타난다.***

## §3 세 배치는 `initialize()` 를 **한 번만** 부른다 (실측)

| 배치 | 호출 지점 |
|---|---|
| `E1` | `run_e1.py:128` (1곳) |
| `E1b` | `run_e1b.py:129` (1곳) |
| `R8` | `run_r8.py:78` (1곳) |
| (공통 bank 경로) | `make_gif_bank_deployed.py:144` (1곳) |

각 러너가 **셀마다 새 updater 를 만들고 한 번 초기화**한다.
⟹ **오늘의 수정이 닿는 경로를 이 배치들은 지나지 않는다.**

## §4 ⛔그래도 남는 것 — 정직한 잔여 위험

- **§2·§3 은 코드 읽기와 호출지점 세기다.** 저장소 자신의 규율은 더 강한 것을 요구한다:
  ***"개입의 효과를 그것이 들어가는 양에서 재지 말고 **판정에서** 재라"***(`B-2` 수심 선례).
  ⟹ **가장 강한 증거는 세 배치를 재실행해 verdict 을 diff 하는 것**이고, **나는 그것을 하지
  않았다**(동결 결과를 덮어쓸 수 없고, 별도 사본 실행은 시간이 필요했다).
- ⟹ ★**미결 작업으로 남긴다**: `E1`·`E1b`·`R8` 을 **사본 디렉터리**에서 재실행해
  `verdict` 를 키 단위로 diff 하고 그 결과를 여기 붙일 것. 그때까지 이 문서가
  **불일치의 설명**이지 **불일치의 해소**는 아니다.
- ⚠그리고 이 사건이 규율 하나를 가리킨다: **배포 코드 파일을 동결 목록에 넣은 배치가
  있으면, 그 파일을 고치는 날 그 배치들의 판정을 재확인해야 한다.** 오늘은 감사가
  찾아 준 결함을 고치는 것이 우선이었고, 그 대가가 이 불일치다.
