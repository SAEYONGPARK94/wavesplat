# M1 실행 수첩 (RUNBOOK) — **생성 완료 순간 붙여 넣는 명령**

> ⛔**런 0회 · GPU 0회 · 설계 변경 0건**으로 작성했다(2026-08-29 03:1x~03:4x).
> `PREREG.md`·`gates_m1.py`·`gen_m1.py`·`run_m1.py`·`analyze_m1.py` 는 **한 글자도 건드리지 않았다**.
> 이 문서는 **동결 절차의 전사(轉寫)**이고, 어긋나면 `PREREG.md`·`gates_m1.py` 가 권위다.
> 신설 파일은 `verify_corpus_m1.py` **하나**(읽기 전용 검증기, 판정 0건, JUDGE 규약 밖 이름).
>
> 작성 사유 = 오늘 하루에 **두 번** 사전등록의 실행 절차를 안 읽어 사고가 났다
> (`B-26` 8샤드 34배 손실 · `M4` Phase A 건너뜀 396런 사망).

---

## 0. 지금 상태 (2026-08-29 03:15 KST 실측)

| | 값 | 출처 |
|---|---|---|
| 생성 진행 | **39/48 파일** · 짝 **20/24** (LIN 20 · HOS 19) | `ls data_m1/{LIN,HOS}` · `gen_manifest_m1.json` |
| 생성 PID | **254576** (`gen_m1.py all`, 08-28 10:16 기동) | `ps aux` |
| 잔여 | HOS 1개(짝 20) + 짝 21–24 (8개) = **9 파일** | — |
| 최근 실측 단가 | LIN **758–1024 s** · HOS **2164–2431 s** | `gen_m1.log` |
| 러너 자신의 ETA | **264 min** (03:01:47 기록) ⟹ 대략 **07:00–07:30 KST** | `gen_m1.log` 마지막 누적행 |
| GPU | **14,878 / 16,376 MiB 점유** (생성이 잡고 있다) | `nvidia-smi` |
| 디스크 | 여유 **775 G**, 48 파일 ≈ **559 MB** | `df -h` · 파일당 11.63 MB |
| 결과 행 | `results_m1_*.jsonl` **0개** · `verdict_m1.json` **없음** | — |
| 스모크 잔재 | **0** (§2.6 대로 삭제됨) | `verify_corpus_m1.py F-1` |
| 완료된 선행 단계 | 0·1·2 게이트 · **3 PC-1(비트동일 max|Δ|=0.0)** · 4 스모크 | `pc1_m1.json` · PREREG §2.6 |

⟹ ★**남은 것은 `PREREG §7` 의 5 → 8 이다.** 5는 지금 돌고 있다.

---

## 1. ⛔BLOCKER · 공시 — **읽고 나서 명령을 붙여라**

### ★[BLOCKER-1] 판정 코드가 `BASELINE-NO-SKILL` 을 짝 읽기에서 **`break` 로 접는다**

`gates_m1.py:pair_reading.stt()` 축자:

```python
return ('hold' if labels[a].startswith('HOLDS')
        else 'break' if labels[a] in ('BREAKS', 'BASELINE-NO-SKILL') else 'undecided')
```

⟹ HOS 팔이 `BASELINE-NO-SKILL` 로 나오고 LIN 이 `HOLDS` 면 짝 키가 `('hold','break')` 가 되고
`PAIR_READING` 이 ***"★★비선형성이 매개한다 — 가장 깨끗한 귀속"*** 을 출력한다.
**그런데 같은 파일의 `READING['BASELINE-NO-SKILL']` 은 ⛔*"우열을 말하지 않는다"* 라고 적었고**,
`PREREG §6` 도 *"미달 ⟹ 그 팔은 `BASELINE-NO-SKILL` 이고 ⛔우열을 말하지 않는다"* 라고 못박았다.
**한 파일이 자기 자신과 어긋난다** ⟹ *"두 곳이 어긋나면 코드가 권위"* 규칙으로 풀리지 않는다.

⚠**이 위험은 사전등록이 스스로 실재한다고 적었다**: `B-26` 의 **선형** `SS-REAL` 에서
기준선 기술이 **−3.96% (23/24)** 로 간신히 살았다(BF4 는 −16.17%). 비선형에서 무너질 수 있다.

★**결과 0행에서 지금 정하는 해석 규칙**(⛔코드 수정 없음 · 문턱 변경 없음):

> **어느 팔이든 라벨이 `BASELINE-NO-SKILL` 이면, `analyze_m1.py` 가 출력한 `pair_reading` 문장을
> 인용하지 않는다.** 그 팔에는 `READING['BASELINE-NO-SKILL']` 만 적용하고 **귀속을 말하지 않는다**
> (= 더 제한적인 규칙이 이긴다). `verdict_m1.json` 의 `pair_key`·`pair_reading` 필드는
> **그대로 두고**(동결 산출물 훼손 금지) 보고문에서 **금지 사유와 함께** 공시한다.

이 규칙은 `M7/DATA_ARM_REVIEW.md §2` 의 마지막 행(*"`BASELINE-NO-SKILL` … 우열 발화 불가"*)과
**일치**한다 ⟹ 두 문서가 코드의 한 줄을 상대로 같은 편이다.

### ★[BLOCKER-2 → 해소됨, 공시 유지] 동결 파일 **1건이 배치 도중 움직였다**

```
wavesplat/optimizer/sequential_update.py
  동결 dedb685a6735517d21285ddc5d36075d   (커밋 c69683b7, 08-24)
  현재 1b473e066cb02e43f39a9887a090c7d4   (커밋 19cc1ec5, 08-28 18:46 = M1 생성 중)
```

⚠**이것이 `FROZEN_md5.txt` 헤더가 스스로 경고한 바로 그 창**이다(*"S13·S13b 가 배치 중 다른
세션의 임포트 경로 수정으로 물린 창"*). 그리고 **여기서는 더 나쁘다** — 저장소 통념
*"런은 옛 코드로 돌았다"* 가 성립하지 않는다. **추정은 아직 안 돌았고 새 코드로 돌 것이다.**

★**그런데 이 드리프트는 M1 의 팔에 도달하지 못한다. 세 갈래로 확인했다(런 0회):**

1. **진입 자체가 불가** — 변경의 첫 훅은 `__init__` 의 `ValueError` 가드인데
   `bank_dk_frac`/`bank_k_set` 중 하나라도 **`None` 이 아닐 때만** 발화한다.
   M1 의 `SHIP-NOW` 는 `run_b12.run_ship:34` 의 `cfg = dict(DEFAULT_CONFIG)` 이고,
   ★**`wavesplat/run.py` 의 md5 는 동결값과 정확히 일치한다**(`10e40734…`) ⟹ `DEFAULT_CONFIG`
   에 두 키가 **존재할 수 없다**.
2. **둘째 훅은 실측 전용 경로** — `_bank_fit_window`(`:1168`)는 `_initialize_bank`(`:1108`)에서만
   불리고, 그것은 `:715 if frames and real_domain_gate and not _frames_have_gt(frames)` 뒤에 있다.
   M1 은 **합성**이고 `EpisodeDataLoader` 는 항상 `gt_field` 를 붙인다(코드 주석 축자)
   ⟹ 이 배치에서 그 함수는 **0회 실행**된다.
3. **독립 회귀** — 그 변경을 넣은 배치가 자기 회귀를 남겼다:
   `tmp/M8_LADDER_2026-08-28/regress_bank_mmse.out` = **비교 165건 · PASS (0 실패)**,
   기본값(둘 다 `None`)에서 **비트 동일**.

⟹ ★**판정 무효화 없음. 그러나 `check_batch_artifacts.py` 는 이 드리프트를 계속 `동결 이동` 으로
보고할 것이고 그것이 옳다** — ⛔**해시를 다시 적어 지우지 마라**(탐지가능성을 없애는 것이
「동결한 적이 없다」보다 나쁜 상태로 가는 길이다). 보고문에 **이 절을 그대로 인용**하라.
(같은 드리프트가 `M3_SUPPORT_2026-08-28` 에도 있다 — 같은 사유로 같이 무해하다. `M7` 은 깨끗하다.)

### ★[BLOCKER-3 아님 · 그러나 ⛔금지] `CUDA_VISIBLE_DEVICES` — **세우지 마라**

- **사전등록은 device 를 한 번도 언급하지 않는다** — `M1` 배치의 `*.py`·`*.md` 전체에
  `device`·`cuda`·`CUDA_VISIBLE_DEVICES` **0회**(grep 실측).
- **코드 사실**: `sequential_update.py:1026` 이 `propose_atoms(...)` 를 **`device=` 없이** 부르고,
  `learned_init.py:133-134` 가 `device = 'cuda' if torch.cuda.is_available() else 'cpu'` 로
  **스스로 고른다**. `config['device']`(배포 기본값 `'cpu'`)는 **읽지도 않는다**.
- ⟹ 배포 `SHIP-NOW` 의 확산 제안은 **GPU 가 보이면 GPU 에서 돈다**. `B-11`·`B-12`·`B-26` 과
  08-28 M1 스모크가 전부 그 상태에서 측정됐다.
- ⟹ ⛔***`CUDA_VISIBLE_DEVICES=""` 를 세우면 제안이 조용히 CPU 로 내려가 **측정된 적 없는 제4의 팔**이
  된다*** (`run_b12` 를 *"그대로 import"* 한다는 §2.4 의 전제가 깨진다).
  **아무것도 세우지 않는 것이 동결 절차와 계보 양쪽에 맞는 유일한 행동이다.**
- ✅**운영상의 진짜 제약은 device 변수가 아니라 순서다** — §4.2 가 *"여유 3.6%"* 라 적었다.
  **생성 프로세스가 끝나 GPU 가 비기 전에는 추정 샤드를 띄우지 마라**(OOM). §3 의 G0 관문.

### ⚠[공시-1] 생성이 **`NUM_THREADS=1`** 로 돌고 있다 — 사전등록은 **8** 을 적었다

`PREREG §7 환경 고정`: *"생성(5·3·4) = `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8`"*.
실제 기동 명령(`ps aux` 축자)은 **`=1`** 이다. ⟹ **동결 절차와의 이탈 1건**. 평가:

- **비용 주장은 그대로 성립한다** — 실측 LIN 758–1024 s / HOS 2164–2431 s 가 사전등록의
  **979 s / 2,229 s** 를 감싼다(§4.1 이 인용 가능하다고 지정한 값들).
- **판정 무영향** — 스레드 수를 읽는 게이트·팔·문턱이 **0개**다.
- **비트 재현성도 이 형상에서는 무영향으로 실측됐다** — HOS `evaluate_at` 의 지배적 연산
  (`Ey @ a.T @ Ex.T`, 실제 형상 499×384·384×576·576×598)을 `NUM_THREADS=1` 과 `8` 에서
  각각 돌려 **md5 동일**(`8fa9f394…`, 3회 재현). 레이캐스트는 GPU 라 스레드 무관.
  ⚠**범위**: 이 탐침이 덮는 것은 그 곱셈 하나다. 전 파이프라인의 비트 동일성 증명이 아니다.
- ⟹ ★**조치: 재생성하지 않는다**(17 h 재소모 vs 이득 0). **공시로 남긴다.**
- ⚠**추정(6단계)은 `=1` 이 맞다**(사전등록이 그렇게 적었다). 이탈이 아니다.

### ⚠[공시-2] `PREREG §5` 와 `§7` 이 어긋난다 — `gates_data` 가 실행 순서에 **없다**

§5 는 `gates_data` 를 *"생성 후 · 추정 전"* 이라 분류하는데, §7 실행 순서에는 그 단계가 없고
`gates_data` 는 **`analyze_m1.py`(7단계) 안에서만** 불린다(`analyze_m1.py:59`).
⟹ 손상된 코퍼스는 **288 런을 다 태운 뒤에야** 드러난다. **§3 의 5b 단계가 이 창을 닫는다.**

---

## 2. ★검증 스크립트 — `verify_corpus_m1.py` (신설, 읽기 전용)

`/home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/verify_corpus_m1.py`

- ⛔**새 문턱·새 상수 0개.** 게이트는 **동결된** `gates_m1.gates_runfree()` ·
  `gates_data()` · `gates_pc1()` 를 **그대로** 부른다. 나머지는 파일에서 잰 사실뿐이다.
- ⛔이름이 `gates*`·`run*`·`analyze*` 가 **아니다** — `check_batch_artifacts.py:77` 의 `JUDGE`
  정규식에 걸리지 않게 했다(판정을 낳지 않는 사후 검증기. `B2_REAL/depth_recheck.py` 선례).
  ⟹ `FROZEN_md5.txt` 에 **추가하지 않는다**(배치 도중 동결 파일을 다시 쓰지 않는다).

| 검사 | 무엇을 잡는가 |
|---|---|
| `A-1` | 동결 md5 **42줄 전건** (배치기준 → 저장소기준 해석) — 배치 중 드리프트 |
| `B-1` | 48 파일 존재 |
| ★`B-2` | ***부분 파일*** — 헤더만 열지 않고 **마지막 프레임을 실제로 읽는다** (`gt_wave_field[-1]` · `lidar_points[-1]` · `lidar_valid_mask[-1]` · `config.attrs['Hs']`). 러너가 `os.path.exists` 로 skip 하므로 **잘린 h5 는 조용히 코퍼스가 된다** — 여기서 갈린다 |
| ★`B-3` | 파일 크기 산포 ≤ 5% (잘린 파일은 작다) |
| `B-4` | 프레임 수 단일값 |
| `C-1`/★`C-2` | 매니페스트 키 == 기대 48 · `status=='ok'` 48 |
| ★`C-3` | **디스크 파일 ⊆ ok 매니페스트** — *"파일은 있는데 항목이 없다"* = 중단된 쓰기의 서명 |
| `G-*` | 동결 `gates_runfree` 10 + `gates_data` 10 + `gates_pc1` 1 |
| `E-1` | `mesh_verdict` **3분할 재도출**(§1.3 은 손복사 아님 — `screen()` 순함수에서 다시 만든다) |
| `F-1` | 스모크 잔재 0 |

**지금(39/48) 돌린 실측 출력** — 미완 사유의 FAIL 만 뜨고 나머지는 전부 PASS:

```
[⛔FAIL] A-1 동결 md5 : OK 41 · 이동 ['wavesplat/optimizer/sequential_update.py']   ← §1 BLOCKER-2
[⛔FAIL] B-1 48 파일  : 39/48                                                       ← 생성 진행 중
[PASS]  ★B-2 마지막 프레임까지 실제 판독 : 손상 []
[PASS]  ★B-3 크기 산포 : 11626263–11645463 bytes (0.16%)
[PASS]  ★C-3 디스크 파일 ⊆ ok 매니페스트 : 항목 없는 파일 []
[PASS]  ★G-22/G-22b/G-22c/G-23/G-19/G-20  (런 0회 10/10)
[PASS]  ★G-7/G-8/G-8c/G-9/G-10/G-24       (자료 게이트, 39 파일 기준)
[PASS]  ★G-16b PC-1 비트동일 : {"gt_wave_field": 0.0, "lidar_points": 0.0, "lidar_valid_mask": 0.0}
★표본 서술(§1.3 재도출): LIN(M=1 채점) {'ADEQUATE': 24} · HOS(M=3 채점) {'DEGRADED': 23, 'ADEQUATE': 1}
```

★**`§1.3` 의 동결 표가 코드에서 정확히 재현된다**(LIN 24 ADEQ · HOS 23 DEGR + 1 ADEQ · INADEQ 0).
⚠**진행 지표**(판정 아님, 생성 게이트다): `G-26` 왜도 **19/19**, 중앙 LIN −0.0002 vs HOS **+0.1472**;
`G-26b` min max|Δ| = **0.3347 m** ⟹ 비선형성이 실제로 켜져 있다.

---

## 3. ★붙여 넣는 명령 (5 → 8) — `PREREG §7` 축자 + 관문

⚠**전부 절대경로.** ⛔**`pgrep -f` 로 감시하지 마라** — 감시 쉘 자신의 명령줄이 패턴에 걸려
무한 대기한다(08-28 에 이 배치가 실제로 물렸다). **파일 수와 PID 로 본다.**

### 5. 생성 (지금 진행 중 — ⛔새로 띄우지 마라)

```bash
# ⛔이미 PID 254576 이 돌고 있다. 아래는 **죽었을 때만**. 존재 파일은 건너뛰므로 이어서 돈다.
\
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
python3 /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/gen_m1.py all
# ⛔샤딩 금지 — gen_m1.py:186 이 nshard>1 을 exit 1 로 거부한다(B-26 실측: 병렬 소득 0).
```

**진행 확인 (읽기 전용, 언제든):**

```bash
ls /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/data_m1/LIN | wc -l; \
ls /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/data_m1/HOS | wc -l; \
tail -3 /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/gen_m1.log
```

### ★G0 관문 — 추정을 띄우기 전에 **셋 다** 참이어야 한다

```bash
cd /home/vexit/wavesplat/wavesplat && \
echo "파일:  $(ls tmp/M1_NONLIN_REAL_2026-08-28/data_m1/LIN tmp/M1_NONLIN_REAL_2026-08-28/data_m1/HOS | grep -c '\.h5$') / 48" && \
echo "PID:   $(ps -p 254576 -o pid= 2>/dev/null || echo '없음(=생성 종료)')" && \
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

* **파일 48/48** · **PID 없음** · **GPU 사용량이 1 GB 미만으로 떨어짐**.
* ⛔셋 중 하나라도 아니면 **6단계로 가지 마라**(§4.2: 여유 3.6% ⟹ 겹치면 OOM).

### ★5b. 코퍼스 검증 (§5 가 요구하고 §7 이 빠뜨린 단계 — 반드시 여기서)

```bash
cd /home/vexit/wavesplat/wavesplat && \
python3 /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/verify_corpus_m1.py
```

* **기대**: `A-1` 만 FAIL(§1 BLOCKER-2, 사유 확인됨) · 나머지 **전부 PASS** · `48/48`.
* ⛔`B-2`·`B-3`·`C-2`·`C-3` 중 하나라도 FAIL ⟹ **손상된 h5 다**. 해당 파일을 지우고
  `gen_m1.py all` 을 다시 걸어라(그 짝만 재생성된다). ⛔추정으로 가지 마라.
* ⛔`G-25`(짝 성분 md5) · `G-26`/`G-26b`(비선형성) · `G-27` FAIL ⟹ **코퍼스가 설계를 배신했다**.
  ⛔추정 금지, 사용자에게 보고.

### 6. 추정 — **288 런 · 8 샤드 · 스레드 1** (⚠`run_m1.py` 는 스레드를 스스로 안 고정한다)

```bash
cd /home/vexit/wavesplat/wavesplat && \
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
for i in 0 1 2 3 4 5 6 7; do \
  python3 /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/run_m1.py run $i 8 \
    > /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/run_m1_$i.log 2>&1 & \
done; wait; echo "=== 8 샤드 종료 ==="
```

* ⚠**배경으로 떼어 놓고 잊지 마라** — 08-22 에 `nohup &` 로 3시간을 놓친 전례가 있다.
  하네스에서는 이 블록을 **`run_in_background: true`** 로 걸어 종료 시 통보받아라.
* ⛔**`CUDA_VISIBLE_DEVICES` 를 세우지 마라**(§1 BLOCKER-3).
* ⛔**`8` 을 바꾸지 마라.** 샤딩은 `i % nshard` 라 다른 `nshard` 로 재실행하면 같은
  에피소드가 **다른 샤드 파일에 중복 기록**되고 `G-12`(팔당 144행)가 FAIL 한다.
* ✅**재실행은 안전하다** — 러너가 `results_m1_{i}.jsonl` 을 읽어 `(arm, episode, seed)` 를
  건너뛴다. 에러 행은 건너뛰지 않으므로 **실패한 런만** 다시 돈다.
* ⚠러너는 첫 줄에서 `G.selftest()` 를 강제하고 실패하면 **추정을 시작하지 않는다**.
* **예상**: 에피소드당 SHIP 3×≈8 s + SPEC25 3×≈37 s ≈ 135 s, 샤드당 6 에피소드
  ⟹ **≈15–25 분**(§4: 0.25 h).

**진행 확인:**

```bash
wc -l /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/results_m1_*.jsonl | tail -1   # 288 이면 끝
grep -c '"error"' /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/results_m1_*.jsonl  # ⛔0 이어야 한다
```

### 7. 분석 (동결 분석기 — ⛔결과를 보고 고치지 않는다)

```bash
cd /home/vexit/wavesplat/wavesplat && \
python3 /home/vexit/wavesplat/wavesplat/tmp/M1_NONLIN_REAL_2026-08-28/analyze_m1.py
```

* 유효 0행이면 `verdict_m1.json` 을 **쓰지 않고 exit 1** 한다.
* ⛔**출력의 `★짝 읽기` 문장을 그대로 인용하기 전에 §1 BLOCKER-1 을 적용하라.**

### 8. 방벽

```bash
cd /home/vexit/wavesplat/wavesplat && \
python3 tmp/_audit/check_batch_artifacts.py tmp/M1_NONLIN_REAL_2026-08-28 && \
python3 tmp/_audit/check_thresholds.py       tmp/M1_NONLIN_REAL_2026-08-28
# ★권장(사전등록 밖, 12 검사기 전체): bash tmp/_audit/run_all.sh
```

* ⚠`check_batch_artifacts` 는 `sequential_update.py` 드리프트를 **계속 보고할 것이다** — 정상이다(§1).

---

## 4. ★분기표 최종 확정 — **`M1 PREREG §6` 이 이긴다**

`M7/DATA_ARM_REVIEW.md §2` 의 4갈래를 `M1 §6.1`·`§6.2`(단일출처 `gates_m1.READING`/`PAIR_READING`)와
대조했다. **어긋남 = 실질 0건 · 문면 2건**(둘 다 M1 쪽으로 정정한다).

| (LIN, HOS) | ★**M1 §6.2 (권위)** | M7 리뷰 §2 의 후속 행동 | 대조 |
|---|---|---|---|
| **hold, hold** | 비선형성은 **매개하지 않는다**. 덤: `B-26` 의 `SS-REAL` 판정이 **3배 조밀 메시 · 신선 시드**에서 독립 재현(메시 정련 대조) | M7 `DATA` 팔의 **인과 사유 소멸** ⟹ 우선순위 **강등**. 축소 실행 금지(하려면 새 사전등록) | ✅일치 |
| **hold, break** | ★★**비선형성이 매개한다** — 저장소가 얻을 수 있는 **가장 깨끗한 귀속**(같은 성분·메시·광선, 다른 것은 시간전개 하나) | M7 을 **동결대로** 착수. `E3` 가 M1 자료 재사용(보고 전용). ⛔확증 뒤집기는 `M7b` 로만 | ✅일치 |
| **break, hold** | ⛔**귀속 금지**(가설과 반대). 메시 정련 **또는 시드**가 움직였다는 뜻 | 판단 보류, 착수 결정 보류 | ✅일치 |
| **break, break** | ★비선형성은 매개자가 **아니다**. 남는 후보 = **메시 정련(관측의 공간 분해능)** ***또는 시드*** ⟹ 다음 배치는 메시 축, 서사는 관측 모형 쪽 | 사실상 소멸. `DATA` 팔에 GPU 쓰지 마라 | ⚠**문면 1**: M7 이 *"메시/관측 공간 분해능으로 지목"* 이라 적으며 **`또는 시드` 를 떨어뜨렸다** ⟹ 보고문은 **M1 문면(두 후보)** 을 쓴다 |
| 그 밖(UNDECIDED 포함) | ⛔귀속 금지, 두 라벨을 그대로 보고 | 착수 보류 | ✅일치 |
| **`BASELINE-NO-SKILL`** | `READING` = ⛔**우열 없음**. ★강한 관찰: 가파른 비선형 실측 해상상태에서는 **최강 기준선조차** 제로필드를 못 이긴다 ⟹ 실측 음성이 **방법이 아니라 레짐**에 붙는다. `SHIP<ZERO` 를 보고하고 `B-26` 의 −3.96% 를 병기 | 별도 행: 귀속 금지 / 우열 발화 불가 | ⚠**문면 2 = §1 BLOCKER-1**: **코드는 이 라벨을 `break` 로 접는다**. M7 리뷰가 별행으로 둔 것이 **옳고**, 코드 출력이 그것을 배신할 수 있다 ⟹ §1 의 해석 규칙 적용 |

★**팔 단위 라벨 5종과 문턱(재도출 실측, `G-19` PASS)**:

| 라벨 | 조건 (`SHIP-NOW < SPEC25`, n=24, 시드 median-of-3) | 단측 p |
|---|---|---|
| `HOLDS (CONFIRMATORY)` | ≥ **18**/24 | **0.011328** ≤ 가족 α 0.025 (✅`B-26` 의 0.0125 도 통과) |
| `HOLDS (NOMINAL)` | 17/24 | 0.031957 (가족단위 미달 ⟹ `EXPLORATORY`) |
| `UNDECIDED` | 8–16 | — |
| `BREAKS` | ≤ **7**/24 | 0.031957 |
| `BASELINE-NO-SKILL` | 음성대조 `SPEC25 < ZERO` **< 17**/24 | (다른 라벨을 **덮어쓴다**) |

⛔**사전 지정 추가 검정 0개.** LIN↔HOS 짝 대비(`paired_contrast`)는 **탐색적**으로만 보고한다.
⛔`B-1`·`B-11`·`B-12`·`B-22`·`B-26` 과 **크기 직접 비교 금지**. ⛔*"완전한 M=3 에서 유지"* 로 일반화 금지
(§1.1a: 속박 조화가 `λ ≲ 6 m` 에서 절단 = **절단된 M=3**). ⛔표본 서술은 **항상 3분할**(§1.3).
⚠`F2` 교락(`loc_radius=25` 고정, `loc/λ_p` 0.33→1.32)은 **이 배치가 고치지 않는다** — §1.4 대로 공시.

---

## 5. ★GPU 대기열 — M1 추정이 끝난 뒤

| 순위 | 배치 | 선행조건 | 상태 | 비용 |
|---|---|---|---|---|
| **1** | ★**M1 6→8 단계** | 생성 48/48 · GPU 해제 | **대기** | 추정 ≈0.25 h |
| **2** | `M3_SUPPORT_2026-08-28` | **동결 완료 · PC-1/PC-2 완료 · 스모크 완료** ⟹ ★**지금 당장 돌 수 있다** | **준비됨** | 생성 **2.3–5.0 h** + 추정 ≈25 min ⟹ **3.5–6 h** |
| **3** | `M7_SUPERVISION_2026-08-28` | ⛔**`gen_m7.py`·`train_m7.py`·`run_m7.py`·`analyze_m7.py` 가 전부 미작성** + **M1 분기가 `(hold, break)` 여야 착수 값이 있다** | ⛔**착수 불가** | ≈12.6 h |

### M3 — 붙여 넣는 명령 (⚠**M1 추정이 끝난 뒤에만**)

```bash
cd /home/vexit/wavesplat/wavesplat && \
python3 /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/gen_m3.py all
# ⛔직렬 1프로세스. gen_m3.py:107 이 nshard>1 을 exit 1 로 거부한다(--allow-parallel 없이는).
#    ✅확인함 — M1 과 같은 가드가 M3 에도 있다.

cd /home/vexit/wavesplat/wavesplat && \
for i in 0 1 2 3 4 5 6 7; do \
  python3 /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/run_m3.py run $i 8 \
    > /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/run_m3_$i.log 2>&1 & \
done; wait

cd /home/vexit/wavesplat/wavesplat && \
python3 /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/analyze_m3.py && \
python3 tmp/_audit/check_batch_artifacts.py tmp/M3_SUPPORT_2026-08-28 && \
python3 tmp/_audit/check_thresholds.py       tmp/M3_SUPPORT_2026-08-28
```

* ★**M1 과 다른 점 하나**: `run_m3.py:19-21` 이 **스레드를 스스로 `1` 로 고정**한다(import 전).
  ⟹ M3 추정에는 `export …_NUM_THREADS=1` 이 **불필요**하다. (M1 러너에는 그 코드가 **없다**.)
* ⚠`PREREG §5`: PC-1(1단계)·PC-2(3단계)는 **이미 완료**(`pc1_m3.json` · `pc2_m3.json`,
  `SMOKE.md` §3 = 3/3 비트 동일). 다시 돌릴 필요 없다.
* ⚠M3 도 `sequential_update.py` 동결 드리프트 **1건**(§1 BLOCKER-2 와 **같은 건, 같은 사유로 무해**).

### M7 — ⛔지금 착수 불가 (선행조건 2개)

1. **러너 4파일이 존재하지 않는다** — `PREREG §9` 가 스스로 *(미작성)* 이라 적었다:
   `gen_m7.py` · `train_m7.py` · `run_m7.py` · `analyze_m7.py`.
   작성 후 **`FROZEN_md5.txt` 재동결**이 필요하다(현재 5줄, 드리프트 0).
2. **M1 분기 조건** — `DATA_ARM_REVIEW §4-1`: ***"(hold, break) 에서만 동결 설계 전체(2×2)를
   돌 값이 있다"***. 다른 분기에서는 **강등**(hold,hold) 또는 **DATA 팔에 GPU 금지**(break,break).
3. ⚠착수 전 **의무 재측정**(`PREREG §7`): 200 스텝 `--smoke` 로 ms/step 을 재고 예산표와
   **2배 이상 어긋나면 착수하지 않고 사전등록을 고친다**.

⟹ ★**M1 추정 직후 GPU 를 이어받을 후보는 `M3` 하나다.**

---

## 6. 열람 범위 (이 문서의 근거)

`M1/{PREREG.md 전문(468행), gates_m1.py(구조·라벨·문턱·게이트·pair_reading 축자),
gen_m1.py 전문, run_m1.py 전문, analyze_m1.py 전문, FROZEN_md5.txt, gen_m1.log, gen_manifest_m1.json,
pc1_m1.json, data_m1/ 파일목록·크기}` ·
`M7/{PREREG.md 목차·§7·§8·§9, DATA_ARM_REVIEW.md 전문}` · `M3/{PREREG.md §5, SMOKE.md §0–3, gen_m3.py 가드, run_m3.py 스레드고정}` ·
`B26/{디렉터리 목록, run_b26_0.log}` · `B12/run_b12.py:1-80` ·
`wavesplat/optimizer/{learned_init.py:98-155, sequential_update.py:390-441·705-755·1005-1035·1108-1200, bank_mmse.py:121-133}` ·
`wavesplat/run.py(md5)` · `tmp/_audit/{check_batch_artifacts.py:77-111·259-280, check_thresholds.py 헤더, run_all.sh:1-40}` ·
`tmp/M8_LADDER_2026-08-28/regress_bank_mmse.out(꼬리)` · `git log/diff (c69683b7 ↔ 19cc1ec5)`.
⛔**M1 추정 결과는 존재하지 않는다**(`results_m1_*.jsonl` 0 · `verdict_m1.json` 없음) — 열 것이 없었다.
BLAS 스레드 탐침 1건은 **이 수첩이 계산한 도출값**(스크래치 실행, 저장소 밖).
