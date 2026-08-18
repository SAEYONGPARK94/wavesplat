# T1-B — 코퍼스·재현 정찰 (S17 `WHITE25` / `WHITEHS50` 사전정찰)

**런 0회.** 추정기·필터·LiDAR 를 한 번도 실행하지 않았다. 읽은 것은 코드·JSONL·메타데이터뿐이다.
**작성** 2026-08-18. **재도출 산출물** `recon/corpus_manifest.json`(이 정찰이 새로 만든 유일한 파일).

## 0. 판정 요약

| 질문 | 판정 | 근거 |
|---|---|---|
| **1. 코퍼스 동일성** | ✅ **재사용 가능** — 128/128 이 지금 디스크에 존재, S16 `results.jsonl` 의 episode 키 집합과 `run_s16.py:episodes()` 의 현재 디스크 스캔이 **완전 일치** | §1 |
| **2. `SHIP` 행 재사용** | ✅ **가능 · 프레임 정합 확인** — `n_match` 가 `SHIP`(3 추정기 시드) ↔ 15개 기준선 셀(5팔×3필터시드) **전부에서 0/128 불일치** | §2 |
| **3. 시드 원장 0 소모** | ✅ **맞다** — `BASELINE_SEEDS=(42,43,44)` 는 numpy 필터 RNG 시드이고 원장 `[1595–1818]` 밖. 신규 `.h5` 생성 0 | §3 |
| **4. G-5 앵커** | ✅ **붙일 수 있다** — 그리고 S16 이 이미 **128/128 `|Δ|=0.0`** 으로 확장 검증했다 | §4 |
| **5. 코드 지문** | ✅ 87 파일 고정집합. **오늘 재계산해도 `5bba1a48…` 로 비트 일치** (git HEAD 는 그 사이 2 커밋 전진) | §5 |
| **6. `effective_config`** | ✅ S16 자기 행은 **객체 되읽기**(`letkf_spec.readback` / `_stock_readback`). ⚠단 수입된 `SHIP@42` 는 **의도표** | §6 |

⚠**한 가지 정정이 필요하다**: RESULT §7 의 *"≈1.5 h"* 는 **단일 필터시드 비용**이다. G-6 이 발동해 배치가 median-of-3 로 올라갔으므로 같은 규약을 지키면 팔당 **≈4.0–4.4 h**, 두 팔 **≈8.2–8.8 h** 다(§7).

---

## 1. 코퍼스 동일성 — **MEASURED**

### 1.1 S16 이 실제로 쓴 128개

`run_s16.py:46-48` 이 소스 디렉터리를 못 박는다(주석 스스로 *"identical to run_s13.py:32-33"*):

```
SPARSE_DIR  = ['data_s9_sparse_2026-08-11']
CROWDED_DIR = ['data_s8_crowded_2026-08-10', 'data_s9_crowded_2026-08-11']
```

`run_s13.py:32-33` 을 열어 대조했다 — **문자 그대로 동일**. 두 배치는 같은 `episodes()` 를 쓴다
(`run_s16.py:104-116` vs `run_s13.py:38-50`, 정규식·정렬·키 생성까지 동일).

**재도출**(오늘 실행):

| 확인 | 결과 |
|---|---|
| `results.jsonl` 의 고유 `episode` | **128** |
| 현재 디스크 `episodes()` 스캔 | **128** |
| 두 집합 동일? | **True** (디스크 결손 0 · 잉여 0) |
| 총 용량 | 1,486,487,973 B ≈ **1.486 GB** |
| md5 중복 | **0** (128개 전부 서로 다른 파일) |
| mtime 범위 | 2026-08-10 ~ 2026-08-11 (= S8/S9 생성 시점, S16 이후 재생성 흔적 없음) |

```bash
# 재도출
cd /home/vexit/wavesplat/wavesplat
python3 - <<'EOF'
import json,glob,os,re
eps={json.loads(l)['episode'] for l in open('tmp/S16_BASELINE_2026-08-14/results.jsonl')}
disk=set()
for d in ('data_s9_sparse_2026-08-11','data_s8_crowded_2026-08-10','data_s9_crowded_2026-08-11'):
    for p in glob.glob(os.path.join(d,'*.h5')):
        disk.add(os.path.basename(p)[len('episode_'):-3])
print(len(eps), len(disk), eps==disk)
EOF
```

### 1.2 층 구분 (MEASURED)

| stratum | 소스 디렉터리 | n | 셀 | 시드 |
|---|---|---|---|---|
| **sparse** | `data_s9_sparse_2026-08-11` | **64** | BF4-2comp 16 · BF4-5comp 16 · BF5-2comp 16 · BF5-5comp 16 | 48–63 |
| **crowded** | `data_s8_crowded_2026-08-10` (32) + `data_s9_crowded_2026-08-11` (32) | **64** | BF4-15comp 16 · BF4-30comp 16 · BF5-15comp 16 · BF5-30comp 16 | 32–55 |

`results.jsonl` 의 `ncomp` 집계로 교차확인: `(sparse,2)` 480행 · `(sparse,5)` 480 · `(crowded,15)` 480 · `(crowded,30)` 480 = 각 셀 32 에피소드 × 5팔 × 3시드. **`ncomp∈{15,30}` = crowded, `{2,5}` = sparse 로 100% 분리**(교차 0행).

접미사는 128/128 전부 `uniform_ra_rf_ds` (`run_s16.py:353` G-1 이 `suff == {'uniform_ra_rf_ds'}` 로 검사, `analyze_final.log` 에 PASS).

### 1.3 ⚠**경고: 이 코퍼스는 git 에 없다**

```bash
cd /home/vexit/wavesplat
git ls-files wavesplat/data_s8_crowded_2026-08-10 | wc -l   # 0
git ls-files wavesplat/data_s9_crowded_2026-08-11 | wc -l   # 0
git ls-files wavesplat/data_s9_sparse_2026-08-11  | wc -l   # 0
```

세 디렉터리 모두 **추적 파일 0개**. `tmp/S16_BASELINE_2026-08-14` · `tmp/S13_oos_2026-08-12` · `tmp/S13b_seed_robust_2026-08-13` 도 **전부 0개**. `.gitignore` 탓이 아니다(11행이 *"h5 tracked via git LFS"* 라 적었으나 실제로 add 된 적이 없다) — CLAUDE.md 가 경고한 그 상태다.

⟹ **동결 증거가 mtime 뿐이다.** 이 정찰이 그 공백을 메우려고 128개 전건 md5 를 떴다:
**`recon/corpus_manifest.json`** (key = episode key, value = `{path, stratum, md5, bytes, mtime}`).
**S17 PREREG 는 이 매니페스트의 md5 를 게이트로 걸어야 한다** — "S13 코퍼스 재사용" 이라는 문장의 유일한 검증 가능한 형태다.

---

## 2. 출하 팔 `SHIP` 행 재사용 — **MEASURED · 프레임 정합 확인**

### 2.1 `SHIP` 은 `results.jsonl` 에 없다 (부재 아님, 설계다)

`tmp/S16_BASELINE_2026-08-14/results.jsonl` 1,920행은 **전부 LETKF 기준선**이다
(키셋 1종 · 32키 · `arm ∈ {SPEC25,SPEC50,SPEC100,WHITER50,WHITE50}` 각 384행).
`SHIP` 열은 **분석 시점에 수입**된다 — `run_s16.py:150-165 import_ship()`:

- **seed 42** ← `tmp/S13_oos_2026-08-12/results.jsonl` 의 `arm=='SHIP'` 행 (128)
- **seed 43·44** ← `tmp/S13b_seed_robust_2026-08-13/results_s13b.jsonl` 의 `arm` 이 `SHIP` 으로 시작하고 `est_seed ∈ {43,44}` 인 행 (256)

**재도출**: 수입 결과 `episodes=128`, `3시드 완비=128/128`, `총 384행`, 에러 0.
라우터 경로 분포 `learned 332 / v2_konly 52` (RESULT §4 G-4 와 일치).

median-of-3 은 `run_s16.py:168-179 ship_med3()` 가 **스칼라와 9개 셸 모두**에 대해 시드축 median 을 취한다.
셸 키 변환도 여기서 처리된다 — `SHIP` 행의 셸 키는 **접두어 없는 `0_5`…`150_inf`** 이고 `run_s16.py:174` 가 `bare = k.split(':',1)[1]` 로 맞춘다.

### 2.2 ★프레임 정합 — 저장소가 헤드라인 하나를 잃은 바로 그 점검

**두 팔의 창은 코드상 같은 정의다**:

| 팔 | 창 | 코드 |
|---|---|---|
| `SHIP` | `t > 2.0 × Tp_true` | `run.py:197` → `data_loader.py:124-131 get_tracking_frames(tp_factor=2.0)` (`:129` `if self.timestamps[i] > t_init`) |
| 기준선 | `t > 2.0 × Tp_true` | `run_s16.py:249` `sel = t > 2.0 * float(loader.Tp_true)` |
| (S13 기준선도 동일) | | `run_s13.py:158` |

`n_match` 의 산출 경로는 **서로 다르다**(`SHIP` 은 `len(frame_results)`, 기준선은 `sel.sum()`), 따라서 이건 동어반복이 아니라 진짜 검정이다.

**재도출 결과 (MEASURED)**:

| 점검 | 결과 |
|---|---|
| 에피소드 키 집합 `SHIP` ↔ `WHITE50@42` | **동일** (결손 0 · 잉여 0) |
| `n_match` 불일치 (SHIP@42 vs WHITE50@42) | **0 / 128** |
| `n_match` 가 추정기 시드 42/43/44 사이에서 흔들리는 에피소드 | **0 / 128** |
| **`SHIP` 의 `n_match` 가 15개 기준선 셀(5팔×3필터시드) 전부와 같지 않은 에피소드** | **0 / 128** |
| `n_match` 관측 범위 | 120–160 (양 팔 동일) |
| 기준선 `n_all` | **301** (128×5×3 전부) |

```bash
cd /home/vexit/wavesplat/wavesplat/tmp && python3 - <<'EOF'
import json,collections
ship={}; 
for r in map(json.loads,open('S13_oos_2026-08-12/results.jsonl')):
    if r.get('arm')=='SHIP': ship[r['episode']]=r['n_match']
base=collections.defaultdict(set)
for r in map(json.loads,open('S16_BASELINE_2026-08-14/results.jsonl')):
    base[r['episode']].add(r['n_match'])
print(sum(1 for k in ship if {ship[k]}!=base[k]), '/', len(ship), 'mismatches')
EOF
```

⟹ ⛔**`b1_full_comparison.json` 을 죽인 결함(WS 120–220 프레임 vs ETKF 301 프레임, 100/100 불일치)은 이 자료에 없다.** `SHIP` 행에 `n_all` 필드는 **없지만**(§2.3), 프레임 정합의 결정적 양은 `n_match` 이고 그것이 15셀 전부와 일치한다.

⚠**INFERRED (측정 아님)**: 위는 *프레임 개수*의 일치다. *같은 프레임 인덱스*임은 두 경로가 같은 `loader.timestamps` 와 같은 부등식을 쓴다는 **코드 동일성**에서 따라오는 추론이지, 프레임 t 값을 행별로 비교한 측정은 아니다(`SHIP` 행에 프레임 타임스탬프가 저장돼 있지 않아 자료만으로는 불가). 개수 일치 + 코드 동일성 + `n_match` 가 120–160 으로 에피소드마다 다른데도 128/128 일치라는 점이 합쳐지면 실질적으로 닫힌다.

### 2.3 필드 목록

**`SHIP@42` (S13, 17키)**: `arm, avg_nRMSE, bf, effective_config, episode, git_head, init_nrmse, n_gaussians, n_match, n_selected, ncomp, pre_exposed, route, seed, shell_rmse, stratum, wall_s`
**`SHIP@43/44` (S13b, 22키)**: 위 + `attempt, est_seed, intent_config, np_state0, torch_initial_seed` (단 `route`·`n_selected` 유지, `effective_config` 는 객체 되읽기)

**없는 것**: `n_all` · `zero_nRMSE` · `zero_shell` · `shell_frac` · `ms_median` · `code_md5`.
- `zero_nRMSE` 부재는 문제가 아니다 — G-9 가 **제로필드가 팔에 무관함을 전량 15행/에피소드에서 spread `0.000e+00`** 으로 확인했으므로 기준선 행의 `zero_nRMSE`/`zero_shell` 을 공통 바닥으로 쓴다(`analyze_s16.py:44-56 base_med`).
- `code_md5` 부재 ⟹ **`SHIP` 열에는 S16 급 지문이 없다.** S13/S13b 행은 `git_head` 만 갖는다(더러운 트리를 못 잡는 바로 그 한계). §5 의 처방이 필요한 이유다.

**필드 완결성 재도출**: `SHIP` 384행 전부 9개 셸 키 존재, **결손 0 · 비유한(NaN/inf) 0**. 기준선 1,920행 × 3필드(`shell_rmse`/`zero_shell`/`shell_frac`) × 9키 = 51,840 슬롯 **결손 0 · 비유한 0**.

### 2.4 판정 재현 (독립 재도출)

`analyze_s16.py` 를 쓰지 않고 원자료에서 직접 median-of-3 → 짝 부호검정을 다시 돌렸다:

| 대조 (혼잡 n=64) | 재도출 | RESULT.md 기재 | 일치 |
|---|---|---|---|
| `SHIP < SPEC25` | 59/64, −24.4%, p=4.5e-13 | 59/64, −24.4%, 4.5e-13 | ✅ |
| `SPEC50 < WHITER50` (R-2) | 28/64, +3.5%, p=0.870 | 28/64, +3.5%, 0.870 | ✅ |
| `SPEC25 < WHITER50` | 59/64, −4.6% | 59/64, −4.6% | ✅ |
| `SPEC100 < WHITER50` | 24/64, +5.4% | 24/64, +5.4% | ✅ |
| `WHITER50 < WHITE50` | 37/64, −0.1% | 37/64, −0.1% | ✅ |

⟹ **`SHIP` 열 수입 경로 전체가 제3자 재도출로 검증됐다.** S17 이 그대로 재사용해도 된다.

---

## 3. 시드 원장 — **MEASURED, S17 은 0 소모가 맞다**

### 3.1 두 종류의 시드를 구별해야 한다

| 시드 | 값 | 출처 | 원장 대상? |
|---|---|---|---|
| **필터 RNG 시드** | 42, 43, 44 | `run_s16.py:69 BASELINE_SEEDS = (42, 43, 44)` → `run_s16.py:222 np.random.seed(filter_seed)` + `letkf_spec.py:139 self._rng = np.random.default_rng(self.filter_seed)` | ❌ 아니다 |
| **추정기 시드** | 42(S13), 43·44(S13b) | S13/S13b 에서 이미 소모, S16 은 **수입만** | ❌ (이미 지출됨, 재지출 없음) |
| **원장 시드** | `[1595–1818]` | 에피소드 생성 / 신선 추정기 시드용 | ✅ — **S16 도 S17 도 건드리지 않는다** |

`run_s16.py:66-69` 의 주석이 `FILTER_SEED = 42 # identical to S13` 라 명시하고, S13 은 `run_s13.py:133 np.random.seed(42)` 하드코딩이었다. 42/43/44 는 **원장 구간 [15xx] 밖의 관례 상수**다.

### 3.2 원장 현재값

- `tmp/W3_SOLVER_2026-08-15/PREREG.md:141` — *"추정기 시드 = 1592, 1593, 1594. 원장 잔여 `[1592–1818]` → 소모 후 `[1595–1818]`"*
- `tmp/W3_SOLVER_2026-08-15/AUDIT_NEUTRAL.md:162` — *"잔여 `[1595–1818]` 원장과 정합 ✅"*
- ⟹ 과제 브리프의 `[1595–1818]` 이 맞다. S16 RESULT §4 G-10 이 적은 **`[1589–1818] 불변`** 은 S16 시점(08-15) 기준이고 지금은 낡았다 — 그 뒤 W3 가 1592–1594 를 썼다.

### 3.3 S17 이 0 소모인 이유 (코드 근거)

1. `run_s16.py:353-357` `episodes()` 는 **기존 디렉터리만 glob** 한다. 생성 코드 경로 없음.
2. `run_s16.py:434-438` **G-10**: `h5 = sorted(glob.glob(os.path.join(HERE, '*.h5'))); g10 = not h5` — 배치 디렉터리에 신규 `.h5` 가 생기면 배치 중단.
3. `WHITE25` / `WHITEHS50` 는 **`ARMS` dict 항목 하나씩**이다(`run_s16.py:70-76`). 필터 인자만 바뀌고 데이터·추정기는 그대로다.
4. `SHIP` 은 재실행하지 않는다 — `import_ship()` 이 JSONL 을 읽을 뿐이다(`run_s16.py:150`, 독스트링 *"ZERO new runs"*).

⟹ **S17 소모 후에도 원장은 `[1595–1818]` 이다.**
⚠단 G-5a 앵커(§4.2)를 켜면 **출하 팔을 2 에피소드 재실행**한다 — 이건 `est_seed=42` 재사용이라 원장 소모는 여전히 0 이지만 "런 0회" 는 아니다.

---

## 4. 재현 앵커 (G-5) — **MEASURED, 그대로 붙는다**

### 4.1 S16 이 쓴 앵커는 두 개다

| | 내용 | 코드 | 실측 |
|---|---|---|---|
| **G-5a** | 출하 팔을 **오늘 코드로 재실행**해 S13 저장값과 대조 | `run_s16.py:281-285 run_ship_rerun()` → `smoke():441-451` | `|Δ| = 0.0` 2/2 (`smoke_report.json` · `smoke2_report.json` 둘 다) |
| **G-5b** | `WHITE50@42` 가 S13 `LETKF_PLUS` 를 비트 재현 | `run_s16.py:489-497` | RESULT §4: **128/128 `|Δ| = 0.000e+00`** |

### 4.2 S17 이 붙일 수 있는 정확한 앵커

✅ **G-5b 는 그대로 유지된다** — S17 이 `WHITE50` 팔을 한 번 더 돌릴 필요조차 없다.
내가 오늘 **전량 재도출로 확인**했다:

| 앵커 | 필드 | 결과 |
|---|---|---|
| S16 `arm='WHITE50', filter_seed=42` ↔ S13 `arm='LETKF_PLUS'` | `avg_nRMSE` | **128/128 정확히 0.0** (max\|Δ\| = 0.0) |
| 〃 | `zero_nRMSE` | **128/128 정확히 0.0** |
| 〃 | `n_match`, `n_all` | **불일치 0/128** |

```bash
cd /home/vexit/wavesplat/wavesplat/tmp && python3 - <<'EOF'
import json
a={json.loads(l)['episode']:json.loads(l) for l in open('S13_oos_2026-08-12/results.jsonl') if json.loads(l)['arm']=='LETKF_PLUS'}
b={r['episode']:r for r in map(json.loads,open('S16_BASELINE_2026-08-14/results.jsonl')) if r['arm']=='WHITE50' and r['filter_seed']==42}
print(max(abs(b[k]['avg_nRMSE']-a[k]['avg_nRMSE']) for k in b), sum(1 for k in b if b[k]['avg_nRMSE']==a[k]['avg_nRMSE']))
EOF
```

**S17 용 앵커 사양 (권고, 3단)**:

| | 팔 | 에피소드 | 필드 | 기대 |
|---|---|---|---|---|
| **A5-1** (신규, 필수) | `WHITE50 @ filter_seed 42` **재실행 2 에피소드** (`BF4_15comp_seed32_uniform_ra_rf_ds`, `BF4_2comp_seed48_uniform_ra_rf_ds`) | 2 | `avg_nRMSE`, `zero_nRMSE` | `|Δ|=0` vs S16 `results.jsonl` 의 같은 셀 (= 오늘 코드가 S16 과 비트 동일함의 증명) |
| **A5-2** (자료만, 런 0회) | S16 `WHITE50@42` ↔ S13 `LETKF_PLUS` | 128 | `avg_nRMSE`·`zero_nRMSE`·`n_match`·`n_all` | 위 표대로 이미 0 — S17 은 **인용**만 하면 된다 |
| **A5-3** (선택) | `SHIP` 재실행 | 2 | `avg_nRMSE` | `|Δ|=0` vs `import_ship()[key][42]` — S16 G-5a 와 같은 형태. 체크포인트 무결성(§5)까지 덮는다 |

⚠**A5-1 이 A5-2 보다 강하다.** A5-2 는 *과거 두 자료가 서로 일치한다*는 사실이고, S17 의 오늘 코드가 그 상태인지는 말하지 않는다. §5 의 지문 재계산이 그것을 대신하지만, 실제 런으로 한 번 찍는 편이 싸다(에피소드당 ~40 s).

---

## 5. 코드 지문 — **MEASURED**

### 5.1 무엇을 해싱하는가

`run_s16.py:124-144 code_fingerprint()`. 독스트링이 자기 이력을 적어뒀다 — 옛 버전은 `sys.modules` 를 순회해 **임포트 순서에 따라 지문이 갈렸고 스모크 10행에 2종이 생겼다**. AMENDMENT_1 A-5 가 **고정 파일집합**으로 바꿨다:

| 그룹 | 패턴 | 개수 |
|---|---|---|
| 패키지 | `wavesplat/**/*.py` (recursive) | 57 |
| 시뮬 | `wavesim/**/*.py` (recursive) | 25 |
| 설정 | `configs/{estimation,default,server}.yaml` | 3 |
| 처치 정의 | `tmp/S16_BASELINE_2026-08-14/{letkf_spec,run_s16}.py` | 2 |
| **합계** | | **87** |

digest = `md5(json.dumps({relpath: md5}, sort_keys=True))`. 결과가 `code_fingerprint.json` 과 **모든 결과 행의 `code_md5`** 에 들어간다.

### 5.2 오늘 재계산 (★핵심 결과)

```
digest now : 5bba1a480cba1ac14c97f237a0b9777d
digest S16 : 5bba1a480cba1ac14c97f237a0b9777d   -> MATCH
added 0 / removed 0 / changed 0   (87 vs 87)
git HEAD: 5f0846c (S16 당시) -> 554dd78 (지금),  dirty 103 -> 62
```

⟹ ★**git HEAD 가 2 커밋 전진하고 더러움이 103→62 로 바뀌었는데 임포트 경로 87 파일은 한 바이트도 안 바뀌었다.** 이것이 정확히 저장소 규율이 요구하는 증거다 — `git_head` 만이었다면 "다르다" 로 읽혔을 상황에서 지문은 "같다" 를 말한다. **S17 은 S16 과 동일한 코드 상태에서 출발한다.**

행 수준 재도출: `results.jsonl` 1,920행의 `code_md5` **1종**(`5bba1a48…`), `(git_head, git_dirty_n)` **1종**(`5f0846c…`, 103). 혼입 0.

### 5.3 ⚠지문이 **덮지 않는** 것 — S17 이 추가해야 할 목록

| 대상 | 현 상태 | 왜 필요한가 |
|---|---|---|
| **`results/direction3/beamform_cond_diffusion_mixed_v1.pt`** | ❌ 미포함 | `SHIP` 행의 **332/384 가 `route='learned'`**. 이 체크포인트가 출하 팔의 실질 파라미터다. 오늘 재도출: sha256 `636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9` — `learned_init.py:52` 의 기록값과 **일치**, 5,856,104 B, **git 추적됨**(16개 `.pt` 중 하나). |
| **에피소드 128개의 md5** | ❌ 미포함 | §1.3 — 코퍼스가 git 밖이다. `recon/corpus_manifest.json` 이 이번에 생성됐다. |
| `tmp/S16_BASELINE_2026-08-14/analyze_s16.py` | ❌ 미포함 | 판정 코드. 처치 정의만 들어 있고 분석기는 빠졌다. |
| S17 자기 스크립트 | (해당 없음) | `run_s17.py` · `letkf_spec.py`(복사/임포트 중 택1) |

**S17 `code_fingerprint()` 권고 파일집합**:

```python
files = {}
for pat in ('wavesplat/**/*.py', 'wavesim/**/*.py'):            # 82 — S16 과 동일
    ...
for extra in ('configs/estimation.yaml','configs/default.yaml','configs/server.yaml'):  # 3
    ...
for own in ('letkf_spec.py','run_s17.py','analyze_s17.py'):     # 처치 + 판정
    ...
files['ckpt:beamform_cond_diffusion_mixed_v1'] = sha256(...)    # ★신규
files['corpus:manifest'] = md5(recon/corpus_manifest.json)      # ★신규 (128 에피소드 md5의 md5)
```

그리고 **G-8 에 "S16 부분집합 87 파일의 digest == `5bba1a48…`" 를 별도 라인으로 인쇄**하면, S17 이 S16 과 같은 코드 위에 있다는 명제가 게이트 로그에 박힌다.

### 5.4 ⚠부수 발견 — `SCRIPTS_md5.txt` 가 낡았다

| 파일 | `SCRIPTS_md5.txt` (AMENDMENT_1 스탬프) | 현재 디스크 | `code_fingerprint.json` |
|---|---|---|---|
| `letkf_spec.py` | `7991d24a…` | `7991d24a…` | `7991d24a…` ✅ |
| `run_s16.py` | `119424cb…` | **`ae3e2c21…`** | **`ae3e2c21…`** |
| `analyze_s16.py` | `e4946a50…` | **`01bc0fdf…`** | (미포함) |

`SCRIPTS_md5.txt` 의 자체 규칙은 *"이 스탬프 이후 스크립트를 수정하면 여기를 갱신하고 사유를 NOTES.md 에 적는다"* 인데, `run_s16.py`·`analyze_s16.py` 두 파일이 **갱신도 NOTES 기재도 없이 바뀌었다**.

**단 자료는 무손상이다** — 1,920행 전부의 `code_md5` 가 **현재 디스크의 `run_s16.py`(`ae3e2c21`)를 포함한 지문**이므로, 실제로 런을 만든 코드가 지금 디스크에 있는 그것이다. 즉 이건 **장부 누락이지 재현성 결함이 아니다**. 그래도 S17 은 같은 구멍을 피해야 한다 — 처방: **`SCRIPTS_md5.txt` 를 손으로 쓰지 말고 `code_fingerprint.json` 의 해당 항목을 그대로 인쇄**하면 두 기록이 갈릴 수 없다.

---

## 6. `effective_config` — **MEASURED**

### 6.1 S16 자기 행 = **객체 되읽기** ✅ (규율 준수)

- **spec 팔** (`SPEC*`, `WHITER50`): `letkf_spec.py:276-283 readback(f)` — 독스트링이 명시적으로 *"read back FROM THE OBJECT (discipline 4), never from intent"*. `getattr(f, k, '<absent>')` 로 **살아 있는 객체 속성**을 읽는다.
- **stock 팔** (`WHITE50`): `run_s16.py:267-277 _stock_readback(f)` — 같은 방식, 16개 속성 + `filter_class=type(f).__name__`.

**S13b 가 이 방식으로 잡아낸 부류의 결함이 실제로 여기서도 잡혔다**: `effective_config['_add_inf'] = 0.005` 이고 `add_inflation = 0.0` 이다. 이건 의도표였다면 안 보였을 값이다 — `letkf_spec.py:154-155` 가 `self._add_inf = float(self.add_inflation); self.add_inflation = 0.0` 로 **가산팽창의 적용 지점을 서브클래스로 옮겼기** 때문이고, 되읽기가 그 이동을 그대로 인쇄한다.

또 `nx/ny/dx/dy/n_grid/n_state/g` 같은 **생성자가 파생시킨 값**들이 기록돼 있다(예: `dx=6.1538…`) — 입력 dict 에는 존재하지 않는 양이다. **되읽기의 결정적 증거.**

### 6.2 ⚠수입된 `SHIP` 열은 반반이다

`run_s16.py:381-389` 이 이 사실을 스스로 기록해뒀다:

| 출처 | `effective_config` 의미 | `init_method` 값 |
|---|---|---|
| **S13 (`SHIP@42`)** | ❌ **의도표** — `run_s13.py:181-183` 이 입력 `cfg` 를 그대로 직렬화 | `'auto'` (미해결) |
| **S13b (`SHIP@43/44`)** | ✅ **객체 되읽기** (+ `intent_config` 별도 보관) | `'learned'` / `'v2_konly'` (해결됨) |

G-4 는 그래서 **분기 검사**를 한다 — `intent_config` 가 있으면 그쪽에서 `init_method=='auto'` 를, 없으면 `route` 필드에서 라우터 결과를 읽는다(`run_s16.py:390-402`). 위반 **0/384**.

⟹ **S17 도 같은 분기 검사를 그대로 복사해야 한다.** `SHIP@42` 를 `SHIP@43/44` 와 같은 스키마로 취급하면 조용히 어긋난다. RESULT §6-3 이 이 항목을 의무기재로 남겨뒀다.

---

## 7. S17 설계 함의 (짧게)

1. ★**PREREG §11-4 가 명령한다**: *"팔을 사후에 추가하지 않는다. `loc` 는 {25,50,100} 셋뿐이다."* ⟹ `WHITE25`·`WHITEHS50` 은 **S16 의 개정으로 붙일 수 없고 새 사전등록이 필수**다. RESULT §7 이월 항목도 같은 말을 한다.
2. **팔 정의는 한 줄씩이다** (`run_s16.py:70-76` `ARMS`):
   - `WHITE25`: `dict(kind='spec', background='white', loc=25.0)` — `WHITER50` 과 **loc 만** 다르다. ⟹ §3.2 의 loc 효과가 스펙트럼 없이도 나오는지 가른다.
   - `WHITEHS50`: `dict(kind='spec', background='white', loc=50.0, white_amp='Hs/4')` — ⚠**`run_letkf` 수정이 필요하다.** 현재 `white_amp` 는 `letkf_spec.py:130` 의 기본값 `0.1` 로 고정이고 `run_s16.py:230-236` 이 전달하지 않는다. `white_amp=float(loader.Hs_true)/4.0` 한 줄 추가. **이건 처치 정의 변경이므로 지문이 바뀐다** — 정상이다.
3. **비용 정정 (MEASURED)**: RESULT §7 의 *"≈1.5 h"* 는 **단일 시드** 값이다.

   | 팔 | min wall_s | 128ep × 1시드 | 128ep × 3시드 |
   |---|---|---|---|
   | `WHITER50` | 38 s | 1.34 h | **4.02 h** |
   | `WHITE50` | 39 s | 1.40 h | 4.20 h |
   | `SPEC*` | 41 s | 1.45 h | 4.36 h |

   G-6 이 발동한 상태(필터시드 rel_sd 8.216% / 13.549% > 2% 문턱)를 승계하면 **두 팔 ≈8.2–8.8 h**. ⛔`wall_s` 는 경쟁 오염 가능(RESULT §6-8 이 최대 11.7배 인용금지) — 위는 **min 기반**이고 단독 점유 가정이다.
   ⚠**G-6 을 승계할지 재측정할지는 S17 PREREG 가 런 전에 동결해야 한다.** S16 이 3배를 지불하고 산 지식은 *"median-of-3 와 시드42 단독이 승수 ±1 안에서 같다"* 이므로, S17 이 **단일 시드로 가고 그 사실을 명시**하는 선택도 사전등록 가능하다(단 S16 열과 짝지을 때는 **같은 집계**여야 한다 — S16 열은 median-of-3 다).
4. **재사용 가능한 열** (런 0회로 얻는 것): `SHIP`(384행) · `SPEC25/50/100`(1,152행) · `WHITER50`·`WHITE50`(768행) 의 **스칼라 + 9셸 + 제로셸 + 면적분율 전량**. S17 이 실제로 돌릴 것은 **신규 2팔뿐**이다.
5. ★**게이트에 넣을 것 (S13·HOSGATE 가 두 번 죽은 지점)**: `analyze_s16.py:84-105 field_gate()` 를 그대로 승계하라 — 팔별 필수 필드 존재를 판정 **전에** 검사하고, 하나라도 없으면 `sys.exit(2)`. S16 에서 위반 0 이었다.

---

## 8. 열람 범위 (= 주장 범위)

**전문 열람**: `run_s16.py`(전 660행 중 1–620 정독) · `letkf_spec.py`(85–283) · `analyze_s16.py`(30–110 + grep) · `RESULT.md`(전문) · `PREREG.md`(grep + §11 전문) · `FROZEN_md5.txt` · `SCRIPTS_md5.txt` · `run_s13.py`(전문) · `wavesplat/run.py`(190–230, 340–480) · `wavesplat/data_loader.py`(124–131) · `wavesplat/optimizer/learned_init.py`(grep).
**집계로만 열람**: `results.jsonl`(1,920행) · `S13/results.jsonl`(704행) · `S13b/results_s13b.jsonl`(256행) · `code_fingerprint.json` · `smoke_report.json` · `smoke2_report.json` · `isotropy_probe.json`.
**미열람**: `NOTES.md`(grep 만) · `AMENDMENT_1.md`(md5 만) · `SMOKE.md` · `verdict_s16.json` · `analyze_final.log` · `phase*_read/` · `provisional/` · `dryrun_analyzer.py` · S13 `PREREG/RESULT`.
**런**: **0회**. HDF5 를 연 적 없다(md5 는 바이트 읽기).
