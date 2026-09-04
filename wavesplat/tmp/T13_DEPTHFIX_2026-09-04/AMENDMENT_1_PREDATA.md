# `T13-DEPTHFIX` 개정문 1 — **결과 0행** (프리즌 표본 52 미접촉)

작성: 2026-09-05. `results_t13.jsonl` **없음** · `verdict_t13.json` **없음**.
스모크(노출된 감사 파일 2개)가 **BLOCKER 1건**을 잡았다.

## 1. ⛔⛔BLOCKER — 「임포트 부작용 정리」가 배치 전체를 **다른 팔**로 돌렸다

초판 러너가 이렇게 썼다:
```python
_cwd = os.getcwd()
import d3_beamform_cond_diffusion as CD
os.chdir(_cwd)          # ← "임포트 부작용 되돌리기"
```
`d3_beamform_cond_diffusion.py:25` 가 `os.chdir(repo_root)` 를 하는 것을 **부작용으로 오해**했다.
실측: 그것은 **의존성**이다 —
```
d3_beamform_cond_diffusion.py:40   PRIOR_CKPT = 'results/direction3/setdiff_prior.pt'   ← 상대 경로
d3_beamform_cond_diffusion.py:222  stats = torch.load(PRIOR_CKPT, ...)                  ← cwd 로 해소
```
cwd 를 되돌리면 `torch.load` 가 `FileNotFoundError` 를 내고,
`sequential_update.py:1111` 의 `except (FileNotFoundError, OSError, ImportError)` 가
**조용히 `v2_konly` 로 폴백**한다 ⟹ **배치 전체가 학습 팔이 아니라 해석적 팔로 돈다.**

**증상 (스모크 실측)**
| | 깨진 판본 | 수리 후 | T11 동결값 |
|---|---|---|---|
| `SHIP` `angle10h0.5` | 0.2218 | **0.2282** | **0.2282** |
| `SHIP` `angle10h2.0` | 0.1764 | **0.2333** | **0.2333** |
| `RESCALE` `angle10h0.5` | 0.1678 | **0.1339** | **0.1339** |
| 블록당 시간 | 240–279 s | **36–46 s** | — |
| `G-REPEAT` | **False** | **True** | — |

★**경고는 났다**(`init_method='auto': learned proposal unavailable`) **그러나 치명적이지 않았다.**
숫자만 봤다면 그럴듯했다 — `SHIP` 0.2218 은 `ZERO` 0.2287 보다 낮아 *"기술이 있다"* 로 읽혔을 것이다.
***드러난 것은 앵커가 T11 과 어긋났기 때문이다.***
⟹ ★`G-ANCHOR-SHIP`(비트 등식)·`G-ARM`(`resolved_init_method=='learned'`)·
`G-BMAPCALLS`(==1)·`G-MAP` 넷 중 **어느 하나라도** 이것을 잡았을 것이다.
그리고 **스모크가 본실행 전에 잡았다** — 상시지시 *"긴 배치 전에 몇 케이스 먼저 돌려라"* 의 값이다.

★**교훈**: ***임포트 부작용처럼 보이는 것이 의존성일 수 있다.*** `T11`·`T12` 러너는 cwd 를
건드리지 않았고, 그래서 이 결함이 없었다. 「정리」가 회귀를 만든 것이다 —
`T11 AMENDMENT_3 §1` 의 *"재작성은 기능을 조용히 떨어뜨린다"* 와 같은 부류.

## 2. ✅`PREREG.md §10.1` 의 `G-REPEAT` 분기 — **판정: 앵커를 비트로 유지**

수리 후 두 블록 모두 같은 파일 `SHIP` 2회가 **비트 동일**(`nrmse` 등식 ∧ `proposal_md5` 등식):
```
[{'block': '…angle10h0.5.mat', 'bit': True}, {'block': '…angle10h2.0.mat', 'bit': True}]
```
⟹ `G-ANCHOR-SHIP`·`G-ANCHOR-RESCALE` 을 **허용오차 없는 비트 등식**으로 유지한다.
사전등록의 대체 처방(*"앵커를 BLOCKED 로 기록하고 T13 내부 짝으로만 산다"*)은 **발동하지 않는다**.

## 3. 스모크가 기록한 나머지 (⛔노출 파일이므로 결과로 인용하지 않는다)
- `FIX` 가 `SHIP` 대비 0.2333 vs 0.2282(악화) · 0.2330 vs 0.2333(거의 동일)
  — `T12` 의 `E-MAPDEPTH` **8/24 널**과 방향이 정합한다. ⛔n=2 로 아무것도 주장하지 않는다.
- 음성대조 `RESCALE-FIX` 가 `RESCALE` 대비 0.1453 vs 0.1339 · 0.1713 vs 0.1620 로 **움직였다**
  ⟹ **섭동 잡음 바닥이 0 이 아니다.** `PREREG §4` 의 `(WITHIN-PERTURBATION-FLOOR)` 접미
  규칙이 실제로 발동할 수 있다 — 결과 전에 고정해 둔 것이 값을 한다.
