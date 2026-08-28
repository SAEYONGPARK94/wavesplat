# M3 실행 수첩 (RUNBOOK) — **M1 이 GPU 를 놓는 순간 붙여 넣는 명령**

> ⛔**런 0회 · GPU 0회 · 설계 변경 0건**으로 작성했다(2026-08-29 03:2x–03:5x KST).
> `PREREG.md`·`gates_m3.py`·`gen_m3.py`·`run_m3.py`·`analyze_m3.py`·`FROZEN_md5.txt` 는
> **한 글자도 건드리지 않았다.** 이 문서는 **동결 절차(`PREREG §5` + `AMENDMENT_1`)의 전사**이고,
> 어긋나면 `PREREG.md`·`gates_m3.py` 가 권위다. 신설 파일은 이 수첩 **하나**
> (판정 0건, `check_batch_artifacts.py` 의 `JUDGE` 규약 밖 이름).
> 본보기 = `tmp/M1_NONLIN_REAL_2026-08-28/RUNBOOK.md`(같은 밤 작성, 같은 사유).

---

## 0. 지금 상태 (2026-08-29 03:49 KST 실측)

| | 값 | 출처 |
|---|---|---|
| M3 동결 | 완료 (AMENDMENT_1 반영판, git_head `bdb822da`) | `FROZEN_md5.txt` |
| 선행 단계 | PC-1 **PASS**(3/3 비트동일) · PC-2 **PASS**(4/4 비트동일) · 1셀 스모크 완료·삭제 | `pc1_m3.json` · `pc2_m3.json` · `SMOKE.md` |
| 런-프리 게이트 | **8/8 PASS — 이 수첩 작성 시각에 재실측**(G-0 충돌 `[]`, M1 39파일이 디스크에 있는 상태에서) | `gates_m3.gates_runfree()` 읽기전용 호출 |
| 생성 | **0회** — `data_m3/` 미존재, `gen_manifest_m3.json` 미존재 | `ls` |
| 결과 행 | `results_m3_*.jsonl` **0개** · `verdict_m3.json` **없음** | `ls` |
| ⛔GPU | **M1 생성이 점유 중** — PID 254576(17h+), 39/48, 러너 ETA ≈ **07:00–07:30 KST** | `ps`·`nvidia-smi`(14.6/16.4 GB)·`gen_m1.log` |
| 디스크 | 여유 **775 G** vs M3 소요 ≈ **1.7 GB** (48×~35 MB) | `df -h` |
| 동결 드리프트 | **1건/22줄** — `wavesplat/optimizer/sequential_update.py` (§2. 무해 확인됨, 공시 유지) | `md5sum -c` 실측 |

⟹ ★**M3 는 "준비됨" 상태다. 남은 것은 §3 의 대기 관문 G0 뿐이고, 그 뒤 명령 순서는
`PREREG §5` 의 2 → 4 → 5 다** (1=pc1 · 3=pc2 는 완료; pc2 는 §2 사유로 재실행 권장).

---

## 1. ★디바이스 판정 — ⛔**`CUDA_VISIBLE_DEVICES` 를 세우지 마라** (M1 과 같은 답, M3 자신의 근거로)

★**사전등록이 정한 것을 확인했다: M3 는 device 를 한 번도 언급하지 않는다** —
`PREREG.md`·`AMENDMENT_1.md`·`SMOKE.md`·`gates_m3.py`·`gen_m3.py`·`run_m3.py`·`analyze_m3.py`
전체에 `cuda`·`device`·`CUDA_VISIBLE` **0회**(grep 실측). ⟹ 동결 절차 = **아무것도 세우지 않은 환경**이고,
PC-1·PC-2·스모크가 전부 그 환경에서 통과했다. 세우면 안 되는 이유가 셋이다:

1. **생성**: `gen_m3.py:57` 이 `use_gpu_raycast=True` 를 **명시**로 넘긴다 — GPU 를 가리면
   PC-1 이 증명한 `B-1` 코퍼스 생성 계보(비트동일)가 깨진다.
2. **추정(SHIP-NOW)**: `run_b12.run_ship:33-34` = `dict(DEFAULT_CONFIG)`, 학습 경로의 확산 제안은
   `learned_init.py:133-134` 가 `device = 'cuda' if torch.cuda.is_available() else 'cpu'` 로 **스스로 고른다**
   (M1 RUNBOOK §1 BLOCKER-3 과 동일 근거; `wavesplat/run.py` md5 는 동결값 `10e40734…` 그대로).
   `""` 로 가리면 제안이 조용히 CPU 로 내려가 **측정된 적 없는 제4의 팔**이 된다.
3. **PC-2 의 전제**: 재현 대상인 동결 `B-26` 행이 GPU 가 보이는 환경에서 측정됐다 — 환경을 바꾸면
   "좌표=배포값 ⟹ 동결 행과 비트동일" 이라는 이 배치의 앵커 대체물 자체가 성립하지 않는다.

⟹ ★**M4/M5b/M8 과 다르고 M1 과 같다. 아무것도 세우지 않는다.**
운영상의 진짜 제약은 device 변수가 아니라 **순서**다(§3 G0: M1 이 GPU 를 놓기 전에 띄우지 마라).
스레드 환경변수도 **불필요** — 생성은 사전등록이 스레드를 지정하지 않았고(§5 무언급),
추정은 `run_m3.py:19-21` 이 import 전에 **스스로 `1` 로 고정**한다(M1 러너와 다른 점).

---

## 2. ★동결 드리프트 — **1건, 무해 확인(세 갈래), 공시 유지**

```
wavesplat/optimizer/sequential_update.py
  동결 dedb685a6735517d21285ddc5d36075d   (커밋 c69683b7, 08-24)
  현재 1b473e066cb02e43f39a9887a090c7d4   (커밋 19cc1ec5, 08-28 18:46 = M8/M8b 패치, +17/−1줄)
나머지 21줄(임포트 대상 16 + 배치 5 + 체크포인트 .pt)은 md5sum -c 전건 OK (2026-08-29 03:2x 실측)
```

⚠**시간 순서가 핵심이다**: M3 의 동결·PC-1·PC-2·스모크는 08-28 08:33–10:12, 드리프트는 08-28 18:46.
⟹ **PC 대조는 옛 코드에서 통과했고, 288 런은 새 코드로 돈다** — M1 BLOCKER-2 와 같은 모양.

★**그러나 이 드리프트는 M3 의 팔에 도달하지 못한다. 세 갈래로 확인했다(런 0회, M1 §1 방법 재적용):**

1. **진입 자체가 불가** — 패치의 첫 훅은 `__init__` 의 `ValueError` 가드(`:436-439`)인데
   `bank_dk_frac`/`bank_k_set` 중 하나라도 **`None` 이 아닐 때만** 발화한다. M3 의 `SHIP-NOW` 는
   `run_b12.run_ship:33-34` 의 `cfg = dict(DEFAULT_CONFIG)` 이고 `wavesplat/run.py` md5 가 동결값과
   일치하므로 두 키가 존재할 수 없다. `SPEC25`(`run_s16.run_letkf` LETKF)는 `SequentialUpdater` 를
   아예 인스턴스화하지 않고, `ZERO` 는 기준선 행의 열이다.
2. **둘째 훅은 실측 전용 경로** — 두 키의 소비 지점은 `:1190-1191`(`_initialize_bank:1108` →
   `_bank_fit_window:1140`) 뿐이고, 뱅크 경로 진입은 `:714-715`
   `if frames and real_domain_gate and not _frames_have_gt(frames)` 뒤다. M3 는 **합성**(gt 부착),
   PC-2 대상도 합성 ⟹ 이 배치에서 **0회 실행**. 나머지 호출 지점 `:1234`(비동기 재적합 shim)·
   `:1319`(재적합)도 같은 뱅크 경로 안이다.
3. **독립 회귀** — `tmp/M8_LADDER_2026-08-28/regress_bank_mmse.out` = **비교 165건 · PASS(0 실패)**,
   기본값(둘 다 `None`)에서 **비트 동일**.
   보강: **G-20(배포 상수 4종 되읽기)을 드리프트된 파일 위에서 오늘 재실행 — PASS**(400·0·0·1e-4 불변).

⟹ ★**판정 무효화 없음.** ⛔**`FROZEN_md5.txt` 를 다시 적어 지우지 마라** — 검사기가 이 드리프트를
계속 `동결 이동` 으로 보고할 것이고 **그것이 옳다**(M1 RUNBOOK §1 BLOCKER-2 규칙 그대로.
탐지가능성을 없애는 것이 「동결한 적이 없다」보다 나쁜 상태로 가는 길이다). 보고문에 이 절을 인용하라.

★**권장(선택, ≈5 분): 추정 전에 `pc2` 를 재실행하라** — 위 세 갈래는 정적 논증이고, PC-2 재실행은
드리프트된 임포트 경로에서의 **측정된 비트동일**로 그것을 승격한다. 실패하면 분석기가
`BATCH-VOID` 를 내게 설계돼 있다(오염된 판정 대신). 원본 보존을 위해 먼저 복사해 둔다:

```bash
cp /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/pc2_m3.json \
   /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/pc2_m3.pre_drift.json && \
cd /home/vexit/wavesplat/wavesplat && \
python3 tmp/M3_SUPPORT_2026-08-28/run_m3.py pc2
# 기대: [PASS] ★G-23 … 4행 전건 비트동일.  FAIL ⟹ ⛔추정 금지, 사용자 보고.
# (M1 RUNBOOK §5 는 "다시 돌릴 필요 없다" 였다 — 이 수첩은 드리프트가 PC-2 이후에
#  들어왔다는 시간 순서 때문에 한 단계 더 확인한다. 비용 5분, GPU 필요 ⟹ G0 뒤에.)
```

---

## 3. ★붙여 넣는 명령 — `PREREG §5` 축자 + 관문

⚠**전부 절대경로.** ⛔**`pgrep -f` 로 감시하지 마라**(M1 이 08-28 에 실제로 물린 함정 —
감시 쉘 자신의 명령줄이 패턴에 걸린다). **파일 수·행 수·PID 로 본다.**
⛔**배경으로 떼어 놓고 잊지 마라**(08-22 `nohup &` 3시간 손실 전례) — 하네스에서는
`run_in_background: true` 로 걸어 종료 통보를 받아라.

### ★G0 관문 — M3 를 띄우기 전에 **셋 다** 참이어야 한다 (⛔M1 과의 GPU 겹침 금지)

```bash
cd /home/vexit/wavesplat/wavesplat && \
echo "M1 결과행: $(cat tmp/M1_NONLIN_REAL_2026-08-28/results_m1_*.jsonl 2>/dev/null | wc -l) / 288" && \
echo "M1 에러행: $(cat tmp/M1_NONLIN_REAL_2026-08-28/results_m1_*.jsonl 2>/dev/null | grep -c '"error"')" && \
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

* **M1 결과행 288 · 에러행 0**(= M1 추정 종료) · **GPU 사용량 1 GB 미만**.
* ⛔하나라도 아니면 M3 를 띄우지 마라 — M1 생성(ETA 07:00–07:30 KST) → M1 추정(≈25 분)이 먼저다
  (M1 RUNBOOK §5 GPU 대기열: M1 다음이 M3 다). M1 의 7·8단계(분석·방벽)는 CPU 라 M3 생성과 겹쳐도 된다.

### (권장) PC-2 재실행 — §2 의 명령. 그 다음:

### 1. 생성 — 48 에피소드 · ⛔직렬 1프로세스 · 환경변수 없음

```bash
cd /home/vexit/wavesplat/wavesplat && \
python3 /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/gen_m3.py all
# ⛔샤딩 금지 — gen_m3.py:106-111 이 nshard>1 을 --allow-parallel 없이 exit 1 로 거부한다
#    (★확인함 — AMENDMENT_1 ② 로 이식된 M1 과 같은 기계. 08-27 오케스트레이터 34배 사고의 방벽).
# ⚠스레드 환경변수를 세우지 마라 — M3 사전등록은 생성 스레드를 지정하지 않았다(M1 §7 의 "=8" 은 M1 것).
# ✅재실행 안전 — 존재 파일은 건너뛴다(gen_m3.py:124-126). ⚠단 §3.1b 의 부분 파일 탐침이 그 함정을 덮는다.
```

**진행 확인 (읽기 전용, 언제든):**

```bash
ls /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/data_m3/*/ 2>/dev/null | grep -c '\.h5$'; \
tail -3 /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/gen_m3.log
```

### 1b. 코퍼스 탐침 — **부분 파일**(생성기 skip-on-exists 의 사각지대) · 런 0회

`gates_data`(러너가 추정 전에 강제한다)는 gt 전체·attrs·성분을 읽지만 **lidar 데이터셋은 안 읽는다**.
잘린 h5 는 288 런을 태운 뒤 에러 행으로야 드러난다(G-11) — M1 `verify_corpus_m1.py` B-2 와 같은 창.
파일을 신설하지 않고 여기서 닫는다(읽기 전용, 판정 0건):

```bash
cd /home/vexit/wavesplat/wavesplat && python3 - <<'EOF'
import glob, json, os, h5py
H = '/home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28'
fs = sorted(glob.glob(H + '/data_m3/*/episode_*.h5'))
bad = []
for p in fs:
    try:
        with h5py.File(p, 'r') as f:
            f['frames/gt_wave_field'][-1]; f['frames/lidar_points'][-1]
            f['frames/lidar_valid_mask'][-1]; float(f['config'].attrs['Hs'])
    except Exception as e:
        bad.append((os.path.basename(p), repr(e)[:60]))
m = json.load(open(H + '/gen_manifest_m3.json'))
ok = {k.split('@')[0] for k, v in m.items() if v.get('status') == 'ok'}
disk = {os.path.basename(p)[len('episode_'):-3] for p in fs}
sz = [os.path.getsize(p) for p in fs]
print('파일', len(fs), '/48 · 마지막 프레임까지 판독 손상', bad)
print('manifest ok', len(ok), '/48 · 디스크⊆ok 위반', sorted(disk - ok)[:3])
print('크기 산포 %.2f%%' % ((max(sz) - min(sz)) / min(sz) * 100) if sz else '')
EOF
```

* **기대**: 48/48 · 손상 `[]` · manifest ok 48 · 위반 `[]`. 손상/위반 ⟹ 해당 h5 삭제 후
  `gen_m3.py all` 재개(그 파일만 다시 생성된다). ⛔추정으로 가지 마라.

### 2. 추정 — **288 런 · 8 샤드** (⛔export 불필요 — 러너가 스레드를 스스로 고정한다)

```bash
cd /home/vexit/wavesplat/wavesplat && \
for i in 0 1 2 3 4 5 6 7; do \
  python3 /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/run_m3.py run $i 8 \
    > /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/run_m3_$i.log 2>&1 & \
done; wait; echo "=== 8 샤드 종료 ==="
```

* ⛔**`CUDA_VISIBLE_DEVICES` 세우지 마라**(§1). ⛔**`8` 을 바꾸지 마라** — 샤딩이 `i % nshard`
  (`run_m3.py:202`)라 다른 `nshard` 재실행은 같은 에피소드를 다른 샤드 파일에 중복 기록해
  `G-12`(팔당 144행)가 FAIL 한다. ✅같은 `8` 로의 재실행은 안전 — `(arm, episode, seed)` 완료분을
  건너뛰고 에러 행만 다시 돈다(`run_m3.py:192-201`).
* 러너가 시작 전에 `G.selftest()`(25항)와 `gates_data`(파생측정)를 **스스로 강제**하고 실패 시
  추정을 돌리지 않는다(`run_m3.py:173-189`) — M1 §7 이 빠뜨렸던 5b 단계가 M3 러너에는 내장돼 있다.
* **예상**: 에피소드당 SHIP 3×≈10.4 s + SPEC25 3×≈66 s ⟹ 직렬 2.9 h → 8샤드 ≈ **25 분**(PREREG §5).

**진행 확인:**

```bash
cat /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/results_m3_*.jsonl 2>/dev/null | wc -l   # 288 이면 끝
cat /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/results_m3_*.jsonl 2>/dev/null | grep -c '"error"'  # ⛔0
```

### 3. 분석 (동결 분석기 — ⛔결과를 보고 고치지 않는다)

```bash
cd /home/vexit/wavesplat/wavesplat && \
python3 /home/vexit/wavesplat/wavesplat/tmp/M3_SUPPORT_2026-08-28/analyze_m3.py
```

* 유효 0행이면 `verdict_m3.json` 을 쓰지 않고 exit 1. PC-1/PC-2 FAIL 이면 라벨이 `BATCH-VOID`.

### 4. 방벽

```bash
cd /home/vexit/wavesplat/wavesplat && \
python3 tmp/_audit/check_batch_artifacts.py tmp/M3_SUPPORT_2026-08-28 && \
python3 tmp/_audit/check_thresholds.py       tmp/M3_SUPPORT_2026-08-28
```

* ⚠`check_batch_artifacts` 는 `sequential_update.py` 드리프트를 **계속 보고할 것이다** — 정상이다(§2).
* ⚠검사기 자체가 지금(08-29 새벽) 다른 에이전트의 수리 중이다 — 방벽 출력이 이 수첩 작성 시점과
  다르게 생겼어도 **동결 이동 1건 보고는 유지**돼야 한다. 사라져 있으면 그게 이상 신호다.

---

## 4. ★해석 관문 (결과 전에 다시 적어 둔다 — 전부 동결 규칙의 전사)

- **단일출처 = `gates_m3.READING`/`DECOMPOSITION`.** PRIMARY = `DIR180`. 문턱: NOMINAL 12/16 ·
  CONFIRM 13/16(가족 α=0.05/3) · BREAKS ≤4/16 · NEG_MIN/SZ_MIN 12/16 (G-19 재도출 PASS).
- ✅**M1 §1 BLOCKER-1 의 결함 부류는 M3 에 없다**(실측): `decompose.st_()` 가
  `BASELINE-NO-SKILL/SHIP-EARNS`·`NO-SKILL-CELL` 을 `'undecided'` 로 보내 분해 표 **밖** =
  *"단일 축 귀속 금지, 세 라벨을 그대로 보고"* 기본 문장으로 떨어진다 — 코드와 `READING` 이 일치한다.
- ⚠**분해 키는 NOMINAL 과 CONFIRMATORY 를 구분하지 않는다**(`st_` 가 둘 다 `'hold'`).
  어느 셀이 NOMINAL 이면 분해 문장을 인용할 때 **그 셀의 `EXPLORATORY` 지위를 반드시 병기**하라
  (B-26 의 `SS-LOWH` 선례와 같은 규율 — 새 규칙이 아니라 `READING['HOLDS (NOMINAL)']` 의 의무).
- ⚠의무 병기(어느 결과든): **선형 GT** · **`SS-LOWH` 반례**(지지집합 안 칸이 최약이었다) ·
  **사전 노출 3건**(S15 dir180 역풍 · DEBATE dir180 · 스모크 1셀 SHIP 0.3655 > SPEC 0.2538 — blind 아님).
- ⛔셀 간 강도 순위 · 3셀 합산 · `B-1`/`B-12`/`B-16`/`B-26` 크기 직접 비교 · 결과 보고 n 증축 전부 금지.
- ⚠`DEP12` 의 BREAKS 는 이중독해(조건자 상수 `d3_beamform_net.py:30` 기제) — 가르는 후속은 새 사전등록으로만.

---

## 5. 시드 · 비용 (재확인 실측)

- **시드 충돌 0**: M1 `SEEDS_CONSUMED = 1681–1694`(`gates_m1.py:218`) · M3 `= 1695–1702`
  (`gates_m3.py:283`, AMENDMENT_1 ① 로 1690→1695 이동) — 겹침 0. B-26(1641–1653)·B-12(1581–4·1633–40)
  와도 0. **G-0 디스크 원장 스캔을 M1 39파일이 있는 오늘 상태에서 재실행 — 충돌 `[]` PASS.**
- **비용**: 생성 **2.3–5.0 h**(직렬; B-26 실측 172 s/ep, 스모크 373 s/ep 는 GPU 경합 중 상한 아님 —
  AMENDMENT_1 ③) + 추정 **≈25 분**(8샤드) + 분석 초 단위 ⟹ **벽시계 ≈3–6 h**. 디스크 ≈**1.7 GB**(여유 775 G).
  G0 시점(≈07:30–08:00 KST 예상)부터 세면 완료는 대략 **11:00–14:00 KST**.

## 6. 열람 범위 (이 수첩의 근거 — 전부 이 세션에서 직접 열었다)

`M3/{PREREG.md 전문, AMENDMENT_1.md 전문, SMOKE.md 전문, FROZEN_md5.txt 전문(md5sum -c 실측),
gen_m3.py 전문, run_m3.py 전문, analyze_m3.py 전문, gates_m3.py(:1-64 헤더·:90-107 상수·:116-131 시드/문턱·
:277-283 동결시드·:285-360 되읽기/파생·:370-427 런프리·:428-548 gates_data/run·:549-660 라벨/READING/decompose),
gates_runfree() 읽기전용 재실행 8/8}` · `M1/{RUNBOOK.md 전문, gates_m1.py 시드행, data_m1 파일수, gen_m1.log 꼬리}` ·
`B12/run_b12.py:25-45` · `B11/gates_b11.py:41-42` · `B26/{data_b26 위치, results_b26 8파일}` ·
`wavesplat/optimizer/sequential_update.py(:436-439·:710-720·:1108·:1140·:1168·:1228-1240·:1310-1322 +
git diff c69683b7↔19cc1ec5 전문)` · `wavesplat/run.py·learned_init.py(md5)` ·
`tmp/M8_LADDER_2026-08-28/regress_bank_mmse.out(꼬리)` · `data_s8_crowded_2026-08-10/…seed32(존재)` ·
`nvidia-smi`·`ps`·`df`. ⛔**M3 결과는 존재하지 않는다** — 열 것이 없었다.
