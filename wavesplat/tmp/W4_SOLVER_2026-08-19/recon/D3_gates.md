# D3 — 사후 게이트 무결성 검사기 (`gatecheck_w4.py`)

**작성**: 2026-08-19 KST · **런 0회**(추정기 실행 없음) · **수정한 파일 0개**(신규 2개만:
`gatecheck_w4.py`, 이 문서).
**읽은 것**: `PREREG.md` · `AMENDMENT_1.md` · `AMENDMENT_2.md` · `FROZEN_md5.txt` ·
`corpus_w4.json` · `arms_config.py` · `run_w4.py`(열람만) · `smoke_results.jsonl`(42행) ·
`../W3_SOLVER_2026-08-15/results.jsonl`(비트 앵커 참조).
⛔**`results.jsonl` 은 열지 않았다** — 본런이 쓰는 중이고 열면 사전노출이다.

---

## 0. 왜 또 만드는가

`run_w4.py::posthoc()` 안에 이미 사후 게이트가 있다. 그러나 그것은 **러너가 자기 답안을 자기가 채점하는
구조**다. 저장소는 이 형태로 이미 한 번 크게 물렸다 — `results/b1_full_comparison.json` 의
**100/100 에피소드 프레임 불일치**가 몇 주를 살아남은 이유는, 그 파일을 검사할 코드가 그 파일을 만든
코드밖에 없었기 때문이다.

⟹ `gatecheck_w4.py` 는 **`run_w4.py`·`arms_config.py`·`varpro.py` 를 하나도 import 하지 않는다.**
모든 상수를 **동결된 산문에서 다시 옮겨 적고**, 옆줄에 근거 절 번호를 달았다.
**두 구현이 일치하면 그 일치가 증거이고, 어긋나면 그 어긋남이 발견이다.**

⛔**이 스크립트는 판정을 만들지 않는다.** 효과크기·승수·p 를 단 하나도 계산하지 않는다.
답하는 질문은 정확히 하나 — **"판정을 인쇄해도 되는가, 된다면 어느 범위까지인가."**
(판정은 `analyze_w4.py`, 독립검증은 `verify_w4.py` 가 만든다. 이 파일은 그 둘을 게이트한다.)

### 사전노출 위생 (설계에 박아 넣음)
- **행 파일 경로가 필수 위치인자다. 기본값이 없다** ⟹ 배치가 도는 중에 `results.jsonl` 을 **실수로 열 수
  없다**.
- **`--descriptives` 는 옵트인**이다. D 족 수치는 결과 인접이므로 일상 무결성 점검에서 새어나오면 안 된다.

```bash
python3 gatecheck_w4.py smoke_results.jsonl              # 개발/스모크
python3 gatecheck_w4.py results.jsonl                    # 본런 (배치 종료 후)
python3 gatecheck_w4.py results.jsonl --descriptives --json gate_w4.json
# exit 0 = 인쇄 가능 · 1 = 아직(PENDING) · 2 = 인쇄 금지(HALT FAIL) · 3 = 사용법 오류
```

---

## 1. ★중단 / 비중단 구분표 — **본런에서 실패하면 어디까지 못 쓰는가**

근거: `PREREG §4`(게이트 표) · `§5`(판정 우선순위 전순서) · `AMENDMENT_1 A-8`
(*"1순위 `G-* 실패 → 중단` 은 **중단 게이트(G-1·G-2·G-3·G-5·G-7·G-8)에만** 적용한다.
`G-4`(기록)·`G-9`(감시)는 PRIMARY 를 죽이지 않는다"*).

### 1.1 ⛔실패하면 **판정을 인쇄하면 안 되는** 게이트 (HALT)

| 게이트 | 근거 | 검사 내용 | 실패 시 못 쓰는 것 |
|---|---|---|---|
| `G-0` arm vocabulary | — (전제) | 알 수 없는 팔 이름 0개 | **전부** (그 행 집합이 이 배치가 아니다) |
| `G-0b` malformed lines | — (전제) | jsonl 파싱 실패 0줄 | **전부** (잘린 write = 부분 행) |
| `G-0c` duplicate cells | `§6` 집계 규약 | `(arm, episode, est_seed)` 중복 0 | **모든 집계** (중복 셀이 med3 과 짝검정 가중치를 왜곡) |
| `G-1` route==learned | `§4 G-1` | est 전 행 `route=='learned'` **∧** `learned_init_calls==1` | **전부** (A족은 라우터 고정이 구성 전제) |
| `G-2` phase1 budget | `A2-2` | `L-ITER0` **정확히 320** · `L-SHIP` **정확히 820** | **전부** (개입이 사전등록된 그 개입이 아니다) |
| `G-2d` arm-delta read-back | `CLAUDE.md`+`A-4` | `effective_config`(**객체 되읽기**)에서 `learned_init_iters`·`learned_merge_frac`·`lambda_l1==1e-4`·`init_method=='learned'` | **전부** (델타가 객체에 도달 못 한 팔 = 사전등록된 팔이 아님) |
| `G-3` vp_steps | `§4 G-3`+`A2-3` | `L-VP`·`L-VP-NW` **400**, 나머지 전부 **0** | **전부** |
| `G-3b/G-7` vp provenance | `§2`+`§4 G-7` | VARPRO 행의 `vp_describe.code_md5 == FROZEN varpro.py` · `vp_entries≥1` · 가중 설정 일치 | **P-1 과 모든 VARPRO 주장** (다른 varpro.py = 다른 처치) |
| `G-5` dup determinism | `§4 G-5`, `§5 D-3` | `L-SHIP-DUP` vs `L-SHIP@1595` **64쌍 전부 `\|Δ\|=0`** | **전부** (비결정성 = 중단) |
| `G-6` DCC (팔 종류별) | `§5`+`A-6.2` | est/cvx/zero **3계약**으로 필수 필드 non-null + 셸 9키 + **`obs:` 접두어** | **전부** (`§4 G-6`: "판정 인쇄 없이 종료") |
| `G-6b` frame parity | `A-12` | 에피소드별 팔 간 `n_match` **및** `n_all` 동수 | **모든 팔 간 비교 = 판정 전부** |
| `G-6c` zero-field identity | `A-12` | 에피소드별 `zero_nRMSE` 동일(tol 1e-12) | **IUT(`L-SHIP<ZERO`) ⟹ 판정 전부** |
| `G-8` one code digest | `§4 G-8` | `code_md5` **1종** | **전부** (두 지문 = 한 표에 두 시스템) |
| `G-8b` one git_head | `§4 G-8` | `git_head` **1종** | **전부** |
| `G-8d` cell census | `A-13` | 팔별 행수 · 총 **1,344** · 시드 집합 | **초과 채워진 셀이 있으면 전부**. ⚠**부족은 `PENDING`이지 `FAIL` 이 아니다** |
| `W3 bit anchor` | `§2` | W3 와 겹치는 에피소드에서 `ZERO`·`ORA-K-CVX` **\|Δ\|=0** | **IUT 와 모든 `ORA-K-CVX` 상한 주장 ⟹ IUT 가 PRIMARY 를 게이트하므로 판정 전부** |

### 1.2 ⚠실패해도 **판정을 인쇄해도 되는** 게이트 (SCOPED / INFO)

| 게이트 | 근거 | 실패 시 **못 쓰게 되는 것만** | 판정 전체는? |
|---|---|---|---|
| `G-2c` budget parity | `A2-4` | **`S-2b` 하나.** `L-VP`≠`L-STEP400` 예산이면 S-2b 가 "매개화 단독"이 아니라 **예산 대조**가 된다 ⟹ 해석이 바뀐다 | ★**살아있다** (아래 §4 불일치 참고) |
| `G-4` merge_removed | `§4 G-4`+**`A-7`** | **`L-MERGE` 기술통계 `D-4`** 하나 | ★**살아있다** — A-7 이 `S-1` 을 가설에서 강등했고 A-8 이 G-4 를 우선순위에서 뺐다 |
| `G-8f` pair availability | `§5 D-4` | 해당 대조가 **`UNDECIDED (n)`** 이 된다 | 살아있다 — `UNDECIDED` 는 `FAIL` 이 아니다 |
| `G-8g` freeze verification | `A-8`·`FROZEN_md5.txt` | **"self-attested freeze + git-anchored" 문구**와 "사전등록 코드로 돌았다" 진술 | 살아있다(이미 쓰인 행 자체는 `G-8`/`G-8c` 가 본다) |
| `G-2*` phase1 (advisory) | `A2-1` 표 | 아무것도. 단 이탈은 **A2-1 표가 틀렸다**는 뜻이므로 조사 | 살아있다 |
| `G-2b` unattributed steps | **`A-6.1`** (원문이 *"경보"*) | 아무것도 직접적으로는. 단 `adam_steps_other>0` 은 **G-2 카운터 자체를 반증**하므로 G-2 를 믿기 **전에** 조사 | 살아있다 |
| `G-0d` error rows | `§5 D-4` | 아무것도. n 을 줄일 뿐 | 살아있다 |
| `G-0e` stale row-gate | 아래 §4.1 | 아무것도. **단 `gates_pass` 를 행 필터로 쓰면 안 된다는 경고** | 살아있다 |
| `G-8c` fingerprint match | `§4 G-8` | 아무것도. **스모크 행에서는 불일치가 정상** | 살아있다 |
| `G-8e` dirty tree | `CLAUDE.md`(S13/S13b) | 아무것도. **범위 문구 의무** | 살아있다 |
| `A-5` P-2 wording | **`A-5` 판독 규칙** | 게이트가 아니라 **P-2 라벨의 문장을 결정**한다 | — |

### 1.3 이 스크립트가 **재도출할 수 없는** 게이트
- **`G-7`(위상 규약 항등식 ≤1e-12)** — 실행이 필요하고 나는 런 0회다. 대신 **행 수준 대리검사**를
  `G-3b` 로 넣었다(FROZEN `varpro.py` md5 일치 + 실제 진입). 항등식 본체는 `run_w4.gates()` 안의
  런프리 검사로 남는다.
- **`G-9`(진행 감시)** — 행 파일이 아니라 mtime·행수의 문제다. `watchdog.py` 소관.

---

## 2. 스모크 실행 결과 (`smoke_results.jsonl`, 42행, 에러 0)

```
$ python3 gatecheck_w4.py smoke_results.jsonl        →  exit 1  (= NOT-YET / PENDING)
```

| 게이트 | 결과 | 관측값 |
|---|---|---|
| G-0 arm vocabulary | **PASS** | 9팔, 미지 0 |
| G-0b malformed / G-0c dup / G-0d error | **PASS** | 0 / 0 / 0 |
| G-0e stale row-gate | **NOTE** | ★2패턴 발견 — §4.1 |
| G-1 route==learned | **PASS** | 38/38 est 행 |
| G-2 phase1 (mandated) | **PASS** | `L-ITER0 [320]` · `L-SHIP [820]`, 12행 |
| G-2* phase1 (advisory) | **PASS** | DUP 820 · VP 1220 · VP-NW 1220 · STEP400 1220 · MERGE 820 |
| G-2b unattributed | **PASS** | `adam_steps_other>0` **0행** |
| G-2c budget parity | **PASS** | `L-VP [1220]` = `L-STEP400 [1220]` ⟹ S-2b 는 **완전 대조** |
| G-2d arm-delta read-back | **PASS** | 38행 전건 객체 되읽기 일치, λ 전건 1e-4 |
| G-3 vp_steps | **PASS** | VP·VP-NW `[400]`, 나머지 7팔 `[0]` |
| G-3b/G-7 vp provenance | **PASS** | 12행, `varpro.py` md5 `5e920c91…` 일치 |
| G-4 merge_removed | **PASS** | `L-MERGE` 중앙 **3.5**, 다른 est 팔 전부 `[0]`, **2/6 무동작**(A-7 이 예고한 no-op) |
| **G-5 dup determinism** | **PENDING** | **2/64 쌍**, `max\|Δ\| = 0.000e+00`, 불일치 0 — ★값은 완벽, **쌍 수가 미달**이라 PENDING |
| G-6 DCC | **PASS** | 42/42 |
| G-6b frame parity | **PASS** | `n_match` 분열 0 · **`n_all` 분열 0** |
| G-6c zero identity | **PASS** | 0 에피소드 |
| G-8 / G-8b | **PASS** | `code_md5` 1종 · `git_head` 1종 |
| G-8c fingerprint | **INVESTIGATE** | 행 `92e1330a…` vs 파일 `a29340ce…` — ★§4.3, **스모크에서는 정상** |
| G-8g freeze verification | **PASS** | FROZEN 10개 파일 디스크 재해시, **이탈 0** |
| **G-8d cell census** | **PENDING** | **42/1344** (팔별 6/192 · 2/64) |
| G-8f pair availability | **PENDING** | 각 대조 med3 완전쌍 **2** (필요 56) |
| **W3 bit anchor** | **PASS** | 겹침 **2행 검사, 불일치 0**, 2행 not-applicable(s8 은 W3 에 없음) |
| A-5 P-2 wording | **NOTE** | `L-SHIP` 6행 전부 `clamp_hits_p1==0` ⟹ ★**P-2 는 "500스텝 결합 적합" 으로 읽는다** |

★**PENDING 3건은 전부 "42행짜리 스모크라서" 이지 결함이 아니다.** 스크립트는 이것을 **`FAIL` 로
오인쇄하지 않는다** — 규율상 *"실행되지 않은 검사는 통과한 검사가 아니다"* 이므로 `PENDING` 이고,
종료코드도 `2`(금지)가 아니라 `1`(아직)이다.

### 본런 겹침 범위 (실측)
`corpus_w4.json` 64 에피소드 중 **W3 의 `ZERO`/`ORA-K-CVX` 표본과 겹치는 것은 16개**(s9 계열).
⟹ 본런에서 비트 앵커는 **`ZERO` 16행 + `ORA-K-CVX` 16행 = 32행**에서 검사되고, 나머지 48은
`not-applicable` 로 계수된다(**결함이 아님**).

---

## 3. 음성 대조 — **게이트가 실제로 발화하는가** (23건 전건 발화)

★저장소는 *"게이트가 위음성이었다"* 로 이미 한 번 대가를 치렀다(`SUPPORT-TRADEOFF` 게이트는 λ=0 도
통과시켰다). 그래서 스모크 행을 **한 필드씩 변조한 사본 23개**를 만들어 각 게이트가 발화하는지 확인했다.
(사본은 스크래치패드에만 만들었고 저장소에는 남기지 않았다.)

| 변조 | 발화한 게이트 | exit |
|---|---|---|
| `L-ITER0.adam_steps_phase1 320→321` | G-2 | **2** |
| `L-STEP400.phase1 1220→1620` | G-2c(+G-2* 조사) | 1(**SCOPED**) |
| `adam_steps_other 0→7` | G-2b(조사) | 1 |
| `effective_config.lambda_l1 1e-4→1e-3` | G-2d | **2** |
| `config_is_input_dict False→True`(의도표 위장) | G-2d | **2** |
| `L-VP.vp_steps 400→0` | G-3 | **2** |
| `vp_describe.code_md5` 변조 | G-3b | **2** |
| `route learned→v2_konly` | G-1 | **2** |
| `L-SHIP-DUP.avg_nRMSE += 1e-15` | G-5 | **2** |
| `n_all` 키 삭제 | G-6 **＋** G-6b | **2** |
| `cvx_rank → null` | G-6 | **2** |
| 셸 키에 `ship:` 삽입 | G-6 (WRONG-PREFIX) | **2** |
| `n_match 160→159` | G-6b | **2** |
| `n_all 301→300` | G-6b | **2** |
| `zero_nRMSE` 한 팔만 변조 | G-6c | **2** |
| `code_md5` 2종 / `git_head` 2종 | G-8 / G-8b | **2** |
| 행 1개 복제 | G-0c | **2** |
| 팔 이름 `L-MYSTERY` | G-0 | **2** |
| `ORA-K-CVX.avg_nRMSE ×1.000001` | W3 bit anchor | **2** |
| `clamp_hits_p1 0→3` | A-5 판독 규칙이 **문장을 뒤집음** | 1 |
| `L-VP.merge_removed 0→2`(누출) | G-4 | 1(**SCOPED**, 배치 안 죽임 ✅) |
| 에러 행 추가 | G-0d(조사) | 1 |

★**중요**: `G-4` 누출과 `G-2c` 파열은 **배치를 죽이지 않았다**(exit 1, SCOPED). 이것이
`AMENDMENT_1 A-7/A-8` 이 요구한 정확한 거동이고, S17 이 죽을 뻔한 **"게이트 자살"의 반대**다.

⚠**이 음성 대조가 스크립트 버그를 하나 실제로 잡았다** — `n_all` 삭제 변조에서 `sorted()` 가
`int` 와 `None` 을 비교하다 **TypeError 로 죽었다**. 검사기가 자기가 잡아야 할 바로 그 손상에서
크래시하면 검사기가 없느니만 못하다. `_ssorted()` 로 전 정렬 지점(15곳)을 경화했다.

---

## 4. ★런 전 발견 3건 (전부 코드 수정 없이 보고만)

### 4.1 ⛔`gates_pass` 를 행 필터로 쓰면 **P-2 의 처치 팔 192행이 통째로 사라진다**
`run_w4.py` 는 행마다 `arms_config.check_gates()` 결과를 `gates_pass`/`gate_fails` 로 찍는다.
그런데 **`arms_config.ARM_GATES` 는 `AMENDMENT_2` 로 갱신되지 않았다** — 갱신된 것은
`run_w4.posthoc()` 쪽뿐이다. 그래서 그 표는 여전히

- `L-ITER0: adam_steps_phase1 == 0` 을 요구한다. **A2-1 이 실측으로 불가능함을 확정한 바로 그 조건**이다.
- `L-MERGE: merge_removed > 0` 을 요구한다. **A-7 이 명시적으로 기술통계로 강등한 조건**이다.

스모크 42행 실측:

```
L-ITER0  gates_pass=False  ['G-2/G-4 adam_steps_phase1: got 320, need == 0']  x6/6
L-MERGE  gates_pass=False  ['G-2/G-4 merge_removed: got 0, need > 0']         x2/6
```

⟹ ★**`analyze_w4.py` 나 `verify_w4.py` 가 `gates_pass==True` 로 행을 거르면 `L-ITER0` 이 100%
탈락하고 `P-2` 가 자료 없이 죽는다.** `L-MERGE` 도 no-op 행(A-7 예측 ~25/64)이 편향 제거된다.
**처방: 두 스크립트에서 `gates_pass` 를 필터로 쓰지 말 것.** 권위 있는 게이트는
`run_w4.posthoc()` 과 이 파일이고, 행 스탬프는 **낡았다**.
(`arms_config.py` 는 동결·수정 금지이므로 코드는 그대로 두고 **소비 규칙**만 고정한다.)

✅**현 시점 실측(2026-08-19)**: 함정은 **아직 밟히지 않았다.** `analyze_w4.py` 는 `gates_pass` 를
**0회** 참조하고, `verify_w4.py` 는 `:395`·`:907` 에서 *"gates recomputed from the raw rows
(never trusting `gates_pass`)"* 라고 명시적으로 거부한다. ⟹ **지금은 정상이고, 이 항목은 앞으로
누가 "게이트 통과 행만 쓰자" 고 최적화할 때를 막는 기록이다.**

### 4.2 ⚠`G-2c` 의 심각도가 **문서와 러너에서 다르다**
- `AMENDMENT_2 A2-4`: *"게이트로 승격해 본런에서도 동수가 유지되는지 검사한다
  (**어긋나면 `S-2b` 의 해석이 바뀌므로**)"* ⟹ **범위는 S-2b**.
- `AMENDMENT_1 A-8`: 중단 게이트 목록은 **`G-1·G-2·G-3·G-5·G-7·G-8`** 뿐 (G-2c 는 A-8 이후 신설).
- `run_w4.py::posthoc()` 은 `ok &= g2c` 로 **배치 전체를 중단**시킨다.

⟹ 이 스크립트는 **문서를 따라 `SCOPED`(S-2b 만 차단)** 로 분류하고, 출력에 러너와 다르다는 사실을
같이 인쇄한다. **본런에서 G-2c 가 실제로 깨지면 두 규약이 서로 다른 답을 준다** ⟹ 그때는
아무것도 인쇄하기 전에 어느 쪽인지 먼저 확정할 것. (스모크에서는 1220=1220 으로 **PASS** 라 지금은
잠재적 충돌이다.)

### 4.3 `code_md5` 가 스모크와 본런에서 **다르다** (정상, 단 기록 필요)
`code_fingerprint()` 는 `wavesplat/**`·`wavesim/**`·`configs/*.yaml` **에 더해 `run_w4.py` 와
`PREREG.md` 까지** 해시한다. `AMENDMENT_2` 가 `run_w4.py` 를 고쳤으므로(커밋 `348d0f2`, 08:19)

- 스모크 행(08:17 작성) `code_md5 = 92e1330aca85ccefd8782ef9f4463216`, `git_head = b0ecafe…`
- 현재 `code_fingerprint.json` `digest = a29340cee6c1de6b3bc08c089ceb8e61`, `git_head = 348d0f2…`

⟹ **스모크 행과 본런 행을 같은 표에 섞으면 `G-8`(지문 1종)이 정당하게 FAIL 한다.** 절대 섞지 말 것.
그래서 `G-8c` 는 `INFO`(비중단)로 두고, 대신 **`G-8g`(FROZEN md5 디스크 재해시)** 를 추가했다 —
`git_head` 는 더러운 트리를 못 잡지만(S13·S13b 가 여기서 재현을 잃었다) 파일 md5 는 잡는다.
실측: **FROZEN 10개 전건 일치**(`run_w4.py 818eb7d9…`·`arms_config.py 5c63a1a2…`·
`varpro.py 5e920c91…`·`PREREG.md 24727f54…` 등).

### 4.4 `RMSE_near` 는 **여전히 저장되지 않는다** (A-11 미이행)
`AMENDMENT_1 A-11` 이 *"`run.py:425-433` 이 이미 버리는데 `frame_results` 에 살아 있다 — 한 줄로 회수"*
라고 적었으나, `run_w4.py` 의 결과 dict(`:398-436`)에 **`RMSE_near` 키가 없다.** 스모크 42행에도 없다.
⟹ 기술통계 추출기는 이 필드를 **`ABSENT (not persisted by run_w4)`** 로 명시 인쇄한다.
⛔**부재를 "근거리 효과 없음" 으로 읽지 마라** — 저장소가 이미 한 번 그렇게 물렸다
(`run_b1_full_comparison.py:51-54` 가 `RMSE_near` 를 버려 근거리가 **설계상 검출 불가**였다).
`rho`/`VR` 은 원시 쌍이 아니라 파생 집계 `decorr_mean`(=1−ρ²)·`shrink_mean`(=(√VR−ρ)²)로만 존재하며,
추출기는 그 둘을 그대로 인쇄한다.

---

## 5. 무료 기술통계(A-11) — ⛔**가설로 승격 금지**

`--descriptives` 로만 인쇄된다. 팔별로:
`wall_s`(중앙·총합, **D-2**) · `window_resid_rms`(**D-1**) · `n_atoms_final` ·
`merge_removed`(**D-3**) · `decorr_mean`/`shrink_mean`(ρ·VR 파생) · `obs_resid_rms_mean` ·
`elapsed_ms_mean` · `init_time_s` · `RMSE_near`(있으면; 현재 ABSENT) ·
VARPRO 팔은 추가로 `vp_win_rms_before/after` · `vp_k_shift` · `vp_clamp_hits`.

출력 최상단에 경고 문자열이 박혀 나온다:
> *D-family descriptives. AMENDMENT_1 A-11: never promoted to a hypothesis. No threshold,
> no direction, no p-value.*

**문턱도 방향도 p 도 계산하지 않는다.** `PREREG §3` 의 D 족이고 그 이상이 아니다.

---

## 6. 본런에서 이 파일을 쓰는 순서

1. 배치 종료(또는 중단) 후 **판정 스크립트보다 먼저** 돌린다.
2. `exit 2` → ⛔**`analyze_w4.py` 를 돌리지 마라.** 출력의 `BLOCKED BY` 줄이 사유다.
3. `exit 1` → 아직 미완(또는 팔 누락). `G-8d`·`G-8f` 를 보고 **`PENDING`인지 `FAIL`인지 구분**할 것.
   ⚠부족한 표는 **측정되지 않은 것**이지 실패한 것이 아니다.
4. `exit 0` → 판정 인쇄 허용. 단 **`SCOPED` 줄에 적힌 대조는 여전히 인쇄 금지**다
   (전형적으로 `G-2c → S-2b`, `G-4 → D-4`).
5. `A-5 P-2 wording` 줄을 **그대로** 판정문에 옮겨 적을 것 — P-2 라벨을
   *"하강"* 이라 쓸지 *"결합 적합"* 이라 쓸지를 이 줄이 결정한다(`A2-1`).
6. `--json gate_w4.json` 을 남겨 게이트 결과 자체를 감사 사슬에 넣을 것.

**범위 문구 의무**(`PREREG §7-8`): 합성 코퍼스 · 선형 GT · `direction_deg=0` · 단일 체크포인트 ·
혼잡 층 한정 · `init_method='learned'` **고정**(배포 라우터가 아님). 게이트가 전건 통과해도
이 여섯 줄은 사라지지 않는다.
