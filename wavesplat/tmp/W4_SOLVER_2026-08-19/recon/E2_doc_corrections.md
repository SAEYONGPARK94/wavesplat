# E2 — 저장소 문서 정정문 (런 0회, 파일 수정 0건)

**작성**: 2026-08-19 · **역할**: 대체 문장 **작성만**. 적용은 사용자가 한다.
**열람 범위 = 주장 범위**: 아래에 인용한 파일·줄만 근거다. `W4/results.jsonl` 은 **열지 않았다**(배치 실행 중, 사전노출 차단).

**재도출 자료**
`tmp/W3_SOLVER_2026-08-15/results.jsonl` (1,212행) · `tmp/W3_SOLVER_2026-08-15/run_w3.py` ·
`tmp/unified_guard_2026-07-26/{merge_arms.jsonl,task2_results.json,score_task2.py,PREREG.md}` ·
`tmp/wp_integration_2026-07-25/arm_runs.jsonl` · `tmp/unified_2026-07-26/arm_runs_extra.jsonl` ·
`tmp/S1_indep_sparse_2026-08-08/beamform_probe.txt` · `tmp/W4_SOLVER_2026-08-19/{AMENDMENT_1.md,recon/B2_factors.md,recon/B3_pathfacts.md,audit/probe_merge_scale.json}` ·
`wavesplat/optimizer/{sequential_update.py,coarse_to_fine.py,learned_init.py}` · `wavesplat/model/{spectral_gaussian.py,tnse.py}` · `wavesplat/run.py`

**표기**: `[M]` = 이 문서에서 원자료로 재도출 · `[R]` = 타 산출물 재현 확인 · `[A]` = 논증

---

## 0. 한 눈에

| # | 현재 서술 | 판정 | 핵심 수치 |
|---|---|---|---|
| 1 | "빔포밍·`MIN_DK`·성장 분기·난수 위상이 배포 혼잡에서 실행되지 않는다" | ⛔**거짓 (범위 오류)** | 학습 경로 성장 **8/87**, `v2_konly` **9/9**. 우회되는 것은 **`coarse_to_fine.py` 하나뿐** `[M]` |
| 2 | "격차는 곱이다" / "21축이 죽은 이유 = 전부 하나만 고쳐서" | ⛔**철회** / ⛔**`ARGUED` 강등** | 곱 ×1.58 vs 측정 ×110, log 지분 **9.7%**. **단 한 표본 안에서는 곱이 1.8% 오차로 성립** `[M]` |
| 3 | "`merge_atoms` 는 공짜 티켓 (이미 구현·미노출)" | ⛔**거짓** | 2026-07-26 에 **이미 측정·실패**. 혼잡 **0승/20셀**, `0.375` `no_harm` **FAIL** (p=0.03125) `[M]` |
| 4 | "k 표류 중앙 15% / 최대 34%" | ⛔**두 겹 결함** | 층 풀링 + **단위 다른 두 필드**. 정직값 혼잡 **22.5–25.0%**·희소 **0.03–0.07%**, 실제 최대 **52.1%** `[M]` |
| 5 | `amplitude_refit` 부호 결함에 **경고 없음** | ⚠**신규 경고 필요** | `arctan2(b,a)` → `arctan2(-b,a)`. **과거 오염 0건**, 그러나 **자기진단이 못 잡는다** `[M]` |

★**공통 사인**: 1·3·4 는 전부 같은 결함 부류다 — **한 곳만 계측하고 전 범위를 말했다.**
(1 = 중복 구현 중 한쪽만 · 3 = 후속 배치 미확인 · 4 = 층 풀링). λ 3중 기재와 같은 부류다.

---

## 1. ★"빔포밍·성장 분기·난수 위상이 배포 혼잡에서 실행되지 않는다"

### ① 현재 문장 (인용)

**`CLAUDE.md`, 「측정 규율」절**:
> **`sequential_update._initialize_learned` 가 코스투파인 staging 을 우회한다**(독스트링이 스스로
> *"the coarse-to-fine N-staging is bypassed"* 라 적었다). 라우터가 **혼잡 90.6%** 를 그쪽으로 보낸다.
> ⟹ **빔포밍 · `MIN_DK` · 성장 분기 · 난수 위상 · `RX2_GROW_FITPHI` 는 배포 혼잡에서 실행되지 않는다.**

**`MEMORY.md` 2026-08-15 novelty 항목**:
> ★★★**둘의 사인이 같고 그게 왜 축이 계속 죽는지를 설명한다**: `_initialize_learned` 가 코스투파인
> staging 을 **우회**해서 **빔포밍·`MIN_DK`·성장 분기·난수 위상이 배포 혼잡(90.6%)에서 실행되지 않는다**

**`project_novelty_search_2026-08-15.md` K1 행**: "★**출하 혼잡 경로가 `MIN_DK` 를 안 쓴다**"
**같은 문서 K4 행**: "**출하 팔에 재시작할 난수 위상이 없다** … 난수 위상 site 는 staging 우회로 **도달 불가**"

### ② 실측 `[M]`

`W3/results.jsonl` 의 `SHIP` 팔(= 출하 기본값 `init_method: auto`) 144행. 계측기는
`run_w3.py:install_probe()` 가 `SpectralGaussianModel.add_gaussians` 를 감싼 **원자 카운터**다.

**(a) 성장(`add_calls>0`) — 경로별로 완전히 갈린다**

| 층 | 라우트 | n | 성장 발화 | 비율 |
|---|---|---|---|---|
| 혼잡 | `learned` | 87 | **8** | **9.2%** |
| 혼잡 | `v2_konly` | 9 | **9** | **100%** |
| 혼잡 | (합) | 96 | **17** | 17.7% |
| 희소 | `learned` | 43 | **1** | 2.3% |
| 희소 | `v2_konly` | 5 | **5** | **100%** |

라우터 분기: 혼잡 `learned` **87/96 = 90.6%** (문서의 90.6% 는 정확하다).

**(b) 사이트 분해 — 여기가 결정적이다.** 두 사이트는 라우트로 **완전 분리**된다.

| 사이트 | `learned` 행 | `v2_konly` 행 |
|---|---|---|
| `coarse_to_fine.py:173` (코스투파인 staging **안**) | **0 / 130** | **14 / 14** |
| `sequential_update.py:1188` (`_adaptive_prune_grow`, staging **밖**) | **9 / 130** (혼잡 8 · 희소 1) | 7 / 14 |

⟹ ★**우회되는 것은 `coarse_to_fine.py` **하나뿐**이다.** 학습 경로에서 그 사이트는 **0/130 = 진짜 0회**다.
그러나 **같은 기능의 두 번째 구현**이 `sequential_update.py:1065-1200` 에 있고, 그 조건절이
**`'learned'` 를 명시적으로 포함**한다:

```
sequential_update.py:1104:  if self.init_method in ('v2', 'v2_konly', 'learned', 'oracle') and all_t is not None:
sequential_update.py:1111:      from wavesplat.optimizer.initializer import (
                                  _beamform_coarse, _beamform_refine, _linear_projection,
sequential_update.py:1115-1116:  min_dk = (RECOMPUTE_MIN_DK if _rx2_gate('RX2_GROW_TIGHT_SUPPRESS') else MIN_DK)
sequential_update.py:1193:      _init_phi = torch.rand(n_actual, device=device) * 2 * np.pi
```

그리고 `_initialize_learned` 자신이 **그 함수를 부른다**:
`sequential_update.py:660` → `if self.adaptive_n: self._adaptive_prune_grow(frames, x_range, y_range, verbose)`

**(c) 독스트링은 원래 옳았다.** `sequential_update.py:620-626` 축자:
> "…so the coarse-to-fine N-staging is bypassed — there is nothing to grow into. Everything after
> the install is the shipped pipeline: the same joint-fit optimizer / loss, **the same adaptive
> prune+grow**, the same tracking optimizer."

⟹ ⛔**문서가 독스트링을 과대해석했다.** 독스트링은 "staging 우회"라고만 했고 "prune+grow 는 그대로"라고
**명시**했는데, 문서가 그 뒷문장을 떨어뜨렸다.

**(d) 난수 위상 — 발화하면 반드시 난수다** (`SHIP` 전 144행 `fitphi_atoms = 0`)

| 층·라우트 | `randphi_atoms` | `fitphi_atoms` |
|---|---|---|
| 혼잡 `learned` | **28** | 0 |
| 혼잡 `v2_konly` | 402 | 0 |
| 희소 `learned` | 3 | 0 |
| 희소 `v2_konly` | 204 | 0 |

혼잡 `learned` 의 28원자는 **총 원자 1,660개의 1.69%**(N_final 합계 기준, 중앙 N_final = 19).
발화 8행은 전부 `add_calls == 1`, 원자 3–5개씩.

**(e) `RX2_GROW_FITPHI` 는 다른 이유로 무효다.** 게이트는 출하 팔에서 **읽힌다**(혼잡 17행이 읽는다).
다만 env 미설정이라 전부 `F` 로 반환되어 적합 위상이 설치되지 않는다. ⟹ "실행되지 않는다"가 아니라
**"실행되나 기본 OFF"** 다. (W3 가 `SHIP-FITPHI` 를 **0/3 개입**으로 런 전에 잡은 것과 일관.)

**(f) S1 의 "beamform 0회 호출"은 n=1 이다** `[M]`.
`tmp/S1_indep_sparse_2026-08-08/beamform_probe.txt` 축자: 에피소드 **1개**(`BF4_15comp_seed20`), 시드 **1개**(1548)
에서 `AUTO` 0회 · `X-BASE(v2_konly)` 3회. ⟹ 그 관측은 **옳다**(학습 경로 90.8%에서 재현된다).
⛔**틀린 것은 그로부터 뽑은 무조건문** *"출하 팔은 beamforming 을 아예 하지 않는다"* 다.

### ③ 정확한 대체 문장

> ★**어느 슬라이스가 맞는 문장인가**: **둘 다 각자의 범위에서 맞고, 어느 쪽도 현재 문장을 구제하지 못한다.**
> `17/96` 은 **층(혼잡) 전체**의 정답이고 `8/87` 은 **라우트(학습)** 의 정답이다.
> 기제 주장이 라우트별이므로 **써야 할 것은 라우트 조건부 문장**이고, 층 수치는 병기한다.

**`CLAUDE.md` 「측정 규율」절 대체안** (기존 ⟹ 줄을 통째로 교체):

> ★★★**새 축을 제안하기 전에 그 코드가 배포 혼잡 경로에서 실제로 실행되는지 **원자 카운터로** 확인하라.**
> **`sequential_update._initialize_learned` 가 우회하는 것은 `coarse_to_fine.py` 의 코스투파인
> N-staging **하나뿐**이다**(독스트링 `sequential_update.py:620-626` 이 *"the coarse-to-fine N-staging is
> bypassed"* 라 적고, **바로 다음 문장에서 *"the same adaptive prune+grow"* 가 남는다고 명시**한다).
> 라우터는 혼잡의 **90.6%**(87/96)를 학습 경로로 보낸다.
> ⟹ **성장 분기는 배포 혼잡 런의 82%에서 발화하지 않는다**(학습 경로 **8/87 = 9.2%** 발화;
> `v2_konly` 로 빠지는 9런은 **9/9 = 100%** 발화 ⟹ 층 전체로는 **17/96**). `W3 SHIP` 실측.
> ⚠**두 사이트를 구별하라** — 학습 경로에서 `coarse_to_fine.py:173` 은 **0/130 = 진짜 0회**지만,
> **같은 기능의 두 번째 구현** `sequential_update.py:1065-1200`(`_adaptive_prune_grow`)은 조건절
> `:1104` 가 `'learned'` 를 포함해 **살아 있고**, `:1111-1116` 이 `MIN_DK` 를, `:1193` 이 난수 위상을 쓴다.
> 호출 지점은 `_initialize_learned` 자신(`sequential_update.py:660`)이다.
> ⟹ ★**축이 죽은 이유는 "안 돌아가서"가 아니라 "중복 구현 중 한쪽만 계측해서"다**(λ 3중 기재와 같은 결함 부류).
> ⚠**단위를 섞지 마라**: `17/96` 은 층 수치이고 성장의 절반 이상이 **`v2_konly` 현상**이다. 라우트를 적어라.
> ⚠**크기도 함께 적어라**: 혼잡 학습 경로에서 성장이 실제로 넣는 원자는 **28개 / 총 1,660개 = 1.69%** 다
> (8런 × 3–5원자). **"돈다"와 "지렛대가 있다"는 다른 주장이다.**
> ⚠`RX2_GROW_FITPHI` 는 출하 팔에서 **읽히되 기본 OFF** 다(`fitphi_atoms` **0/144**) — "미실행"이 아니다.

**`MEMORY.md` 2026-08-15 항목 대체안** (해당 ★★★ 문장 교체):

> ★★★**둘의 사인이 같다 — 그러나 사인의 이름을 2026-08-19 에 고쳤다**: `_initialize_learned` 가
> 우회하는 것은 **`coarse_to_fine.py` staging 하나뿐**이고, `sequential_update.py:1065-1200` 에
> **같은 기능의 두 번째 구현**이 `'learned'` 를 포함한 채 살아 있다(`:1104`).
> ⟹ ⛔**"빔포밍·`MIN_DK`·성장 분기·난수 위상이 배포 혼잡에서 실행되지 않는다" 는 거짓**(`W3 SHIP` 실측:
> 성장 **혼잡 17/96**, 학습 경로 **8/87 = 9.2%**, `v2_konly` **9/9**).
> ★**정확한 사인 = "중복 구현 중 한쪽만 계측했다"** — 안 돌아간 것이 아니라 **안 보고 있었다**.
> ⚠**K1·K4 의 판정(사망)은 유지하되 사유는 교체**해야 한다(아래 ④).

**`project_novelty_search_2026-08-15.md` K1·K4 행 대체안** — ★**판정을 뒤집지 마라. 사유만 바꿔라.**

> **K1 비등방 분리 셀** — **사망 (사유 2026-08-19 교체)**: ⛔종전 사유 *"출하 혼잡 경로가 `MIN_DK` 를
> 안 쓴다"* 는 **거짓**이다. `sequential_update.py:1111-1116` 이 `MIN_DK` 를 import·사용하고 그 분기는
> 학습 경로에서 **살아 있다**. **살아남는 사유 = 지렛대 부족**: 그 사이트는 배포 혼잡 학습 런의
> **9.2%(8/87)** 에서만 발화하고, 넣는 원자는 **총 원자의 1.69%(28/1,660)** 다 ⟹ 개입해도 90.8%의 런에서
> **비트 동일**이 되어 PRIMARY 가 검정력을 못 낸다. (E1 이 든 5개 site 가 출하 실행 0개인 것은 **여전히 참**.)
>
> **K4 다중 재시작** — **사망 (사유 2026-08-19 축소)**: ⛔종전 사유 *"난수 위상 site 는 도달 불가"* 는
> **거짓**이다 — `sequential_update.py:1193` 이 `torch.rand(n_actual)*2π` 로 난수 위상을 넣고, `W3 SHIP`
> 혼잡 학습에서 **28원자가 실제로 난수 위상으로 설치**됐다(`fitphi_atoms` 0). **살아남는 사유**는 둘:
> (i) 초기 설치 위상은 `learned_konly` 기본 False 라 **LS 적합**이므로(`learned_init.py:386`) 재추첨할
> 난수가 **초기화 시점엔** 없고, (ii) 성장 경로의 난수 위상은 **9.2% 런 · 1.69% 원자**라 `SHIP-MR8` 과
> `SHIP-DUP8` 의 개입 차가 그 안에 묻힌다. ⟹ **PRIMARY 퇴화라는 결론은 유지**.

### ④ 근거

- `[M]` 재도출 명령:
  ```bash
  cd /home/vexit/wavesplat/wavesplat/tmp/W3_SOLVER_2026-08-15 && python3 -c "
  import json,collections
  R=[json.loads(l) for l in open('results.jsonl')]
  S=[r for r in R if r['arm']=='SHIP']
  for st in ('crowded','sparse'):
    for rt in ('learned','v2_konly'):
      s=[r for r in S if r['stratum']==st and r.get('init_method_resolved')==rt]
      c2f=sum(any(k.startswith('coarse_to_fine.py:173') for k in r['fitphi_site_hits']) for r in s)
      seq=sum(any(k.startswith('sequential_update.py:1188') for k in r['fitphi_site_hits']) for r in s)
      print(st,rt,'n',len(s),'add>0',sum(r['add_calls']>0 for r in s),'c2f',c2f,'seq',seq,
            'randphi',sum(r['randphi_atoms'] for r in s),'fitphi',sum(r['fitphi_atoms'] for r in s))"
  ```
- 코드: `wavesplat/optimizer/sequential_update.py:620-626`(독스트링) · `:660`(호출) · `:1104`(조건절) ·
  `:1111-1116`(`MIN_DK`) · `:1188`(FITPHI 게이트) · `:1193`(난수 위상) ·
  `wavesplat/optimizer/coarse_to_fine.py:173`(우회되는 사이트)
- 계측기: `tmp/W3_SOLVER_2026-08-15/run_w3.py:199-256` (`install_probe`)
- 선행 산출물: `tmp/W4_SOLVER_2026-08-19/AMENDMENT_1.md:135-145` (A-14) `[R]`
- n=1 한계: `tmp/S1_indep_sparse_2026-08-08/beamform_probe.txt`

---

## 2. ⛔"격차는 곱이다" / "21축이 죽은 이유는 전부 하나만 고쳐서다"

### ① 현재 문장 (인용)

`tmp/W3_SOLVER_2026-08-15` 계열 §0.2 및 `MEMORY.md` 2026-08-16 W3 항목이 상한 분해를
**"격차는 곱이다"** 로 서술하고, `MEMORY.md` 2026-08-15 novelty 항목이
> ★**축이 죽는 것은 아이디어가 나빠서가 아니라 안 돌아가는 코드를 연구해서다** (누적 **21축**)

로 **전 축에 단일 사인**을 붙인다. `CLAUDE.md` 측정 규율절에도 같은 문장이 복제돼 있다.

### ② 실측 `[M]` — 재검증 결과 **철회가 옳다. 단 이유가 문서에 적힌 것보다 좁다.**

**(a) 산술 재확인 — 인용된 두 요인은 측정 격차를 만들지 못한다**

| 항목 | 값 |
|---|---|
| `(a,b)` 단독 요인 | ×1.00245 |
| `k` 단독 요인 | ×0.63263 |
| **곱** | **×0.63418 ⟹ 개선 ×1.577** |
| 측정 결합 격차 `SHIP`/`ORA-K-CVX` | **×110.00**(중앙값의 몫) / **×112.82**(짝 비의 중앙) |
| **어긋남** | **×69.75** |
| **곱이 설명하는 log 지분** | **ln(1.577)/ln(110) = 9.69%** |

내 독립 산술로 전건 재현 `[M]`. `B2_factors.md:215-224` 와 일치 `[R]`.

**(b) ★그런데 — 한 표본 안에서는 곱이 성립한다.** 이것이 정정의 핵심이고, 문서가 놓친 부분이다.

S4 동일 16에피소드 안에서:

| 요인 | 비 | 승 |
|---|---|---|
| `(a,b)` — 추정기 k 고정, GD `(a,b)` → 닫힌형 `(a,b)` | ×1.00245 | 6/16 = **널** |
| `k` — 닫힌형 `(a,b)` 고정, 추정기 k → 진실 k | **×0.00922** | **16/16** |
| **결합 (측정)** | ×0.00941 | 16/16 |
| **두 요인의 곱 (예측)** | **×0.00924** | — |

⟹ **예측 ×0.00924 vs 측정 ×0.00941 = 오차 1.8%.** ★**곱셈 분해 자체는 멀쩡하다.**

**(c) 그러면 무엇이 깨졌나 — 표본 합성이다** `[M]`

- 교집합: **`S4 ∩ K6 = 0`**, **`W3 ∩ K6 = 0`**.
- 코퍼스·λ 시대(1e-3 vs 1e-4)·추적기 on/off·시드가 전부 다르다
  (`S4/PROBE-A`: `data_indep_crowded_2026-08-08`, 시드 1557–1559, λ=1e-3, 추적기 OFF /
   `W3`: + `data_s9_crowded_2026-08-11`, 시드 1592–1594, λ=1e-4, 추적기 ON).
- **같은 "출하" 팔의 중앙 nRMSE 가 배치마다 갈린다**: `W3 SHIP` **0.15301** · `S4 AUTO-NT` **0.16552** ·
  `K6 AUTO` **0.18399** ⟹ **산포 20.2%**. `(a,b)` 요인(×1.00245 = +0.24%)은 **이 산포 안에 완전히 묻힌다**.

**(d) "21축 단일 사인" — 축별 귀속 자료 0건** `[A]`
`project_novelty_search_2026-08-15.md` 의 사망 사유는 후보 **K1–K4** 에만 표로 적혀 있고, 누적분에 대한
축별 사유 표는 저장소에 **없다**. 그리고 1번에서 확인했듯 **K1·K4 의 사유 자체가 틀렸다** ⟹ 4개 중 2개가
근거를 잃은 일반화를 나머지로 확장할 수 없다.
⚠**추가로, `"21축"` 은 이 저장소가 금지한 총량 표현이다**(`reference_citation_bans.md` 머리말: "누적 40여 건"·
"실측 75개" 인용 금지, 계수 규칙 없이 총량을 쓰지 말 것). **세지 말고 열거하라.**

### ③ 정확한 대체 문장

**§0.2 / `MEMORY.md` W3 항목 대체안**:

> ⛔**"격차는 곱이다" 삭제.** 대체 = ★**"한 표본 안에서 격차는 `k` 다. `(a,b)` 는 널이다."**
> S4 16에피소드 안에서 `(a,b)` ×1.00245(6/16 = 널) · `k` **×0.00922(16/16)** · 결합 ×0.00941
> ⟹ **결합이 사실상 `k` 단독**이다.
> ⚠**곱셈 분해 자체가 틀린 것이 아니다** — 같은 표본 안에서는 곱(×0.00924)이 측정 결합(×0.00941)을
> **1.8% 오차로 맞힌다**. **깨진 것은 표본을 건너뛴 합성**이다: `S4 ∩ K6 = 0` · `W3 ∩ K6 = 0` 이고
> 같은 출하 팔이 배치마다 **0.15301 / 0.16552 / 0.18399 = 20.2% 산포**한다
> ⟹ `(a,b)` 요인(+0.24%)은 **그 산포 안에 묻힌다**.
> ⛔**서로 다른 배치의 요인 비를 곱해 상한을 만들지 마라** — 두 요인을 곱하면 개선 ×1.58 인데
> 측정 격차는 ×110 이다(**어긋남 ×69.8**, 곱이 설명하는 **log 지분 9.7%**).
> ⚠`×110.00`(중앙값의 몫)과 `×112.82`(짝 비의 중앙)는 **다른 추정량**이다. 표기를 분리하라.

**`MEMORY.md` / `CLAUDE.md` 의 "축이 죽는 이유" 문장 대체안**:

> ★**축이 죽는 흔한 사인 하나 = 안 돌아가는(또는 거의 안 돌아가는) 코드를 연구한 것** —
> 2026-08-15 novelty 킬 라운드의 **K1·K4** 가 그 사유로 죽었다. `ARGUED`.
> ⛔**"모든 축이 같은 사유로 죽었다"로 일반화 금지** — **축별 귀속 자료가 0건**이고,
> **2026-08-19 에 K1·K4 의 사유 자체가 정정됐다**(코드는 돈다; 발화율이 9.2%·원자 1.69%라 지렛대가 없다).
> ⛔**누적 축 수를 총량으로 인용 금지**(이 저장소는 총량 표현 자체를 금지한다). **열거하라.**
> ★**살아남는 규율은 그대로다**: 새 축을 제안하기 전에 **원자 카운터로 발화율과 개입 크기를 함께 재라.**
> "돈다"와 "지렛대가 있다"는 다른 주장이다.

### ④ 근거

- `tmp/W4_SOLVER_2026-08-19/AMENDMENT_1.md:90-108` (A-9) `[R]` ·
  `tmp/W4_SOLVER_2026-08-19/recon/B2_factors.md:215-238, 247-265` `[R]`
- 산술 재도출 `[M]`:
  ```bash
  python3 -c "
  import math
  a,b=1.00245,0.63263; g=110.00
  print('product %.5f -> improvement x%.3f'%(a*b,1/(a*b)))
  print('mismatch x%.2f'%(g*a*b)); print('log share %.1f%%'%(100*math.log(1/(a*b))/math.log(g)))
  print('within-sample: %.5f vs measured 0.00941'%(1.00245*0.00922))
  print('shipped-arm spread %.1f%%'%(100*(0.18399-0.15301)/0.15301))"
  ```
- 축별 귀속 부재: `project_novelty_search_2026-08-15.md` (K1–K4 표만 존재)
- 총량 표현 금지: `reference_citation_bans.md` 머리말

---

## 3. ⛔"`merge_atoms` 는 공짜 티켓 (이미 구현·미노출)"

### ① 현재 문장 (인용)

**`MEMORY.md` 2026-08-15 novelty 항목**:
> ★공짜 티켓: 배포 분리 문턱 **0.03 이 측정 한계 양쪽보다 작다** ⟹ 위험은 "중복 원자"이고
> 수리 `merge_atoms` 가 **이미 구현·미노출**.

**`CLAUDE.md` 측정 규율절**:
> ⟹ 위험은 "바닥이 넓다" 가 아니라 **한 주엽 안 중복 원자**다. 수리 `merge_atoms` 가 **이미 구현·GT-free·
> 끄면 비트 동일**인데 `learned_merge_frac` 기본값이 `None`(no-op)이고 `configs/`·`run.py` 에 **미노출**.

### ② 실측 `[M]` — **저장소가 2026-07-26 에 이미 쟀고, 사전등록된 목표에서 실패했다.**

`tmp/unified_guard_2026-07-26/` (PREREG 존재, 동결 변형 **V1 = 0.25·kp**, **V2 = 0.375·kp**, 500런).
`merge_arms.jsonl` + 기준선(`wp_integration_2026-07-25/arm_runs.jsonl`, `unified_2026-07-26/arm_runs_extra.jsonl`)
으로 **내가 독립 재집계**했다(에피소드 중앙 = median-of-5 레인, `score_task2.py` 규약과 동일).

**(a) 혼잡 no-harm — 0승**

| `merge_frac` | n(에피소드) | 승 | 무 | 패 | 중앙 Δ | Wilcoxon p | `no_harm` |
|---|---|---|---|---|---|---|---|
| 0.25 | 10 | **0** | 9 | 1 | +0.00000 | 1.0 | PASS(무변화) |
| 0.375 | 10 | **0** | 4 | **6** | **+0.00088** | **0.03125** | ⛔**FAIL** |
| **합** | **20셀** | **0승** | 13 | 7 | | | |

레인 단위(10ep × 5레인 × 2변형 = 100셀)로도 **승 4 · 무 81 · 패 15**.
⟹ ★**"혼잡 20셀 0승, `0.375` 는 `no_harm` FAIL" 는 정확하다** `[M]`.
`task2_results.json` 의 인쇄값(`p=0.03125`, `median_delta=0.00088`, `no_harm_pass=false`)과 전건 일치 `[R]`.

**(b) 희소 파리티 — 두 변형 모두 `FAIL` 티어**

| | 값 | 사전등록 문턱 |
|---|---|---|
| pure-v2 희소 | 0.02037 | — |
| pure-learned 희소 (병합 없음) | 0.05345 | — |
| merge 0.25 | **0.05485** | FULL ≤ 0.02537 · PARTIAL ≤ 0.037 ⟹ **FAIL** |
| merge 0.375 | **0.05459** | ⟹ **FAIL** |

⟹ ★**병합은 희소를 개선하지 못했을 뿐 아니라 병합 없는 쪽보다 미세하게 나빴다**(+0.0014 / +0.0011).

**(c) 오늘 프로브가 기제를 준다** (`audit/probe_merge_scale.json`, ZERO 추정기 런, 200셀)

병합 반경은 `r = merge_frac × kp`, 혼잡 `kp` 중앙 **0.0765**:

| `merge_frac` | 반경 r | 원자 중앙 in→out | 제거 발생 셀 |
|---|---|---|---|
| 0.25 | 0.0191 | 15 → 14.5 | 12/200 |
| **0.375** | **0.0287** | 15 → **13** | **50/200** |
| 0.5 | 0.0383 | 15 → **11** | **110/200** |

★**"공짜 티켓" 논증이 여기서 끊긴다**: 논증은 *"배포 문턱 0.03 이 측정 분해능 한계(반경 **0.042–0.074**)보다
작으니 올려라"* 였다. 그런데 **실제로 측정된 두 설정의 반경은 0.0191·0.0287 로 여전히 0.042 미만**이고,
그런데도 **0.375 가 혼잡을 유의하게 해쳤다**. 분해능 한계 0.042 에 닿으려면 `merge_frac ≈ 0.55`,
0.074 에 닿으려면 `≈ 0.97` 이 필요한데, **이미 0.5 에서 셀의 55%가 원자를 잃는다**(중앙 15→11).
⟹ **미측정 구간은 "더 안전한 쪽"이 아니라 `no_harm` 을 이미 깬 설정보다 *더 공격적인* 쪽이다.**

**(d) 범위 한계** `[A]` — 이 측정은 **순수 `learned` 팔**이고 출하 `auto` 라우터가 아니다. 희소 코퍼스는
레거시 1comp/2comp 집합이고 λ 시대는 **1e-3** 이다. ⟹ **기록 결과는 무손상**(배포 전 실행)이나
**재실행하면 비트 동일하지 않다**(S1–S8 과 같은 규칙). 다만 **"공짜 티켓" 주장을 지우기에는 충분**하다 —
주장은 "재면 이긴다"였고, **재봤더니 졌기** 때문이다.

### ③ 정확한 대체 문장

**`CLAUDE.md` 대체안**:

> ⟹ 위험은 "바닥이 넓다" 가 아니라 **한 주엽 안 중복 원자**다.
> ⛔**그러나 이것을 "공짜 티켓"이라 부르지 마라 — 2026-07-26 에 이미 쟀고 실패했다**
> (`tmp/unified_guard_2026-07-26/`, PREREG 존재, 동결 변형 `0.25·kp`·`0.375·kp`, 500런):
> **혼잡 20셀 0승**(0.25 = 0승 9무 1패 · 0.375 = 0승 4무 **6패**), **`0.375` 는 `no_harm` FAIL**
> (Wilcoxon **p=0.03125**, 중앙 Δ **+0.00088** 악화), 희소는 **두 변형 다 `FAIL` 티어**
> (0.0549 / 0.0546 vs FULL 문턱 0.0254 · PARTIAL 0.037; **병합 없는 0.0534 보다도 나쁘다**).
> ⚠**"문턱 0.03 < 분해능 한계 0.042–0.074 이니 올리면 된다" 는 논증은 끊겼다**: 측정된 두 설정의 실제
> 반경은 **0.0191·0.0287 로 아직 0.042 미만**인데 이미 혼잡을 해쳤고, 0.042 에 닿으려면
> `merge_frac ≈ 0.55` 가 필요한데 **0.5 에서 이미 셀의 55%가 원자를 잃는다**(중앙 15→11,
> `audit/probe_merge_scale.json`).
> ✅**사실로 남는 것만 적어라**: `merge_atoms` 는 **구현돼 있고**(`learned_init.py:190`), **GT-free** 이며,
> `learned_merge_frac` 기본값 `None` 이라 **끄면 비트 동일**이고, `configs/`·`run.py` 에 **미노출**이다.
> ⟹ 허용 = **"노출은 안 됐지만 미검증은 아니다. 두 설정에서 검증됐고 떨어졌다."**
> ⚠재측정하려면 **출하 `auto` 라우터 위에서** 다시 재라(07-26 은 순수 `learned` 팔·λ=1e-3·레거시 희소 코퍼스).

**`MEMORY.md` 2026-08-15 항목 대체안** (해당 ★ 절 교체):

> ⛔**"공짜 티켓" 철회(2026-08-19)** — `merge_atoms` 는 **미노출이지 미검증이 아니다**.
> `tmp/unified_guard_2026-07-26` 이 동결 변형 2개로 **이미 쟀고**: 혼잡 **0승/20셀**,
> `0.375` **`no_harm` FAIL(p=0.03125)**, 희소 **양 변형 `FAIL` 티어**(병합 없는 쪽보다도 나쁨).
> ★교훈 = **"이미 구현돼 있다"를 "아직 안 재봤다"로 읽지 마라 — 후속 배치를 먼저 확인하라.**

### ④ 근거

- `tmp/unified_guard_2026-07-26/PREREG.md:74-106`(동결 티어·문턱) · `task2_results.json`(인쇄 판정) ·
  `score_task2.py:93-110`(no-harm 규약) · `merge_arms.jsonl`(500행)
- `[M]` 재도출 명령:
  ```bash
  cd /home/vexit/wavesplat/wavesplat && python3 -c "
  import json,numpy as np;from scipy.stats import wilcoxon
  L=[(42,0),(1,1),(2,2),(3,3),(4,4)]
  C=[f'episode_BF4_{c}comp_seed{s}_uniform.h5' for c in (15,30) for s in range(5)]
  ln={};
  for f in ('tmp/wp_integration_2026-07-25/arm_runs.jsonl','tmp/unified_2026-07-26/arm_runs_extra.jsonl'):
    for l in open(f):
      d=json.loads(l)
      if d.get('arm')=='learned' and d.get('avg_nRMSE'): ln[(d['episode'],d['est_seed'],d['diff_seed'])]=d['avg_nRMSE']
  mg={}
  for l in open('tmp/unified_guard_2026-07-26/merge_arms.jsonl'):
    d=json.loads(l); mg[(d['episode'],d['est_seed'],d['diff_seed'],d['merge_frac'])]=d['avg_nRMSE']
  M=lambda v:float(np.median([x for x in v if x]))
  for mf in (0.25,0.375):
    a=np.array([M([mg.get((e,E,D,mf)) for E,D in L]) for e in C])
    b=np.array([M([ln.get((e,E,D)) for E,D in L]) for e in C])
    print(mf,'WIN',int((a<b).sum()),'TIE',int((a==b).sum()),'LOSS',int((a>b).sum()),'p=%.5f'%wilcoxon(a,b).pvalue)"
  ```
- 반경 기제: `tmp/W4_SOLVER_2026-08-19/audit/probe_merge_scale.json` (+ `probe_merge_scale.py`)
- 코드: `wavesplat/optimizer/learned_init.py:190`(`merge_atoms`) · `:365-368`(기본 `None` no-op)

---

## 4. ⛔"k 표류 중앙 15% / 최대 34%"

### ① 현재 문장 (인용)

**`MEMORY.md` 2026-08-16 W3 항목**:
> ⚠CVX 는 4요인 묶음이고 ★**G-12 가 한 요인을 정량화**: 오라클 팔이 **k 를 중앙 15% 흘린다**(최대 34%).

**`project_w3_solver_2026-08-16.md:53-54`**:
> ★**G-12 가 그 한 요인을 정량화했다**: 오라클 팔이 **k 를 유지하지 못한다** — 중앙 상대 표류
> **14.9–16.2%**, 최대 34%(`ORA-K@L3` 138/144 유효).

### ② 실측 `[M]` — **두 겹 결함이 확인된다.**

`W3/results.jsonl` 의 `k_drift_valid=True` 행 **508개**(`ORA-K@L3` 138 · `ORA-K@L4` 138 · `ORA-K-NOREG` 138 ·
`ORA-K@L3-DUP` 94).

**(a) 결함 1 — 층 풀링.** 두 층이 **세 자릿수** 다른데 하나로 뭉갰다.

| 층 | n | `k_drift_rel_median` 중앙 | IQR | 최대 |
|---|---|---|---|---|
| **혼잡** | 376 | **0.2329 (23.3%)** | 0.1430–0.3058 | **0.5215 (52.1%)** |
| **희소** | 132 | **0.0004 (0.04%)** | 0.0002–0.0056 | 0.4307 (43.1%) |
| (풀링) | 508 | 0.1720 (17.2%) | — | 0.5215 |

⟹ ★**풀링값 15–17% 는 어떤 실제 모집단도 기술하지 않는다.** 혼잡도 희소도 그 근처에 없다.

**(b) 결함 2 — 서로 다른 두 필드, 다른 단위.**

- **"15%"** = `k_drift_rel_median` 의 **팔별 풀링 중앙값 범위**:
  `ORA-K@L4` 0.1494 · `ORA-K@L3` 0.1529 · `ORA-K-NOREG` 0.1618 ⟹ **14.9–16.2%**. 전사는 정확 `[R]`.
- **"34%"** = **`k_drift_max` 의 풀링 최대 = 0.3437**. ⛔**이 필드는 `rad/m` 단위의 절대 표류**이지
  **백분율이 아니다**(`run_w3.py:_k_drift` — `k_drift_max = np.max(d)`, `d = |k_est − k_true|`;
  상대량은 `k_drift_rel_median = median(d/|k|)` 로 **별도 필드**다).
- **상대 표류의 실제 최대는 0.5215 = 52.1%** 다.

⟹ ★**"중앙 15% / 최대 34%" 는 한 문장 안에서 `rel_median` 의 중앙과 `max`(절대)의 최대를 붙여 놓은 것**이다.

**(c) 층별 정직값 — 보고된 값과 전건 일치** `[M]`

| 팔 | 혼잡 중앙 | 혼잡 최대 | 희소 중앙 | 희소 최대 |
|---|---|---|---|---|
| `ORA-K@L4` | 22.49% | 52.15% | 0.03% | 34.97% |
| `ORA-K-NOREG` | 22.74% | 51.33% | 0.04% | 34.99% |
| `ORA-K@L3` | 24.95% | 51.11% | 0.07% | 43.07% |
| `ORA-K@L3-DUP` | 24.95% | 51.11% | — | — |

⟹ **혼잡 22.5–25.0% · 희소 0.03–0.07% · 혼잡 최대 52.1%**. 보고받은 정정값과 일치.

**(d) ★`k_drift_valid` 가 `SHIP` 에서 `0/144`** — **배포 팔의 표류는 미측정** `[M]`

| 팔 | valid / n | `k_drift_note` |
|---|---|---|
| `SHIP` | **0 / 144** | `not-an-oracle-arm` (144/144) |
| `SHIP-FITPHI` | 0 / 144 | 〃 |
| `KONLY` / `KONLY-FITPHI` | 0 / 144 각 | 〃 |
| `ORA-K-CVX` | 0 / 48 | 〃 |
| `ZERO` | 0 / 48 | 〃 |
| `ORA-K@L3` / `L4` / `NOREG` | 138 / 144 각 | 유효 |

진단기가 **설계상 오라클 팔에만 걸려 있다**(`run_w3.py:_k_drift`: `if not oracle or not sp: return`).
⟹ ★**"출하 팔이 k 를 흘린다/안 흘린다"는 어느 쪽도 이 자료로 말할 수 없다.**

**(e) 덤으로 — 무효 6행의 사유가 1번과 연결된다** `[M]`
오라클 팔조차 144 중 6행이 무효이고 사유는 `'N_final(%d)!=n_gt(%d): prune/grow broke the 1:1 map'` 이다.
⟹ **성장/가지치기가 발화하면 이 진단 자체가 꺼진다** — 1번의 "성장은 실제로 돈다"와 같은 사실의 뒷면이다.

### ③ 정확한 대체 문장

**`MEMORY.md` 2026-08-16 W3 항목 대체안**:

> ⚠CVX 는 4요인 묶음이고 ★**G-12 가 한 요인을 정량화한다 — 단 층별로 읽어라**:
> **오라클 팔은 혼잡에서 k 를 중앙 22.5–25.0% 흘리고(최대 52.1%), 희소에서는 사실상 안 흘린다
> (중앙 0.03–0.07%)** — 상대 표류 `k_drift_rel_median` 기준, 팔별 중앙의 범위, `n=376`(혼잡)/`132`(희소).
> ⛔**"중앙 15% / 최대 34%" 인용 금지**(2026-08-19 정정, 두 겹 결함):
> (i) **층 풀링** — 두 층이 세 자릿수 다르므로 풀링값 17.2% 는 **어느 모집단도 기술하지 않는다**;
> (ii) **단위 다른 두 필드 혼용** — "34%" 는 `k_drift_max`(**`rad/m` 절대량 0.3437**)이지 백분율이 아니다.
> **상대 표류의 실제 최대는 52.1%** 다.
> ⛔★**`k_drift_valid` 가 `SHIP` 0/144(`not-an-oracle-arm`) ⟹ 배포 팔의 표류는 미측정이다.**
> 이 자료로 출하 팔의 표류를 **어느 방향으로도** 주장하지 마라.
> ⚠오라클 팔조차 6/144 가 무효이고 사유는 *"prune/grow broke the 1:1 map"* — **성장이 발화하면 진단이 꺼진다.**

**`project_w3_solver_2026-08-16.md:53-54` 대체안**: 위와 동일 문안으로 교체하고
`(ORA-K@L3 138/144 유효)` 는 **`유효 508/912 = 오라클 3팔 각 138/144 + DUP 94/96`** 로 정확히 적을 것.

### ④ 근거

- `[M]` 재도출 명령:
  ```bash
  cd /home/vexit/wavesplat/wavesplat/tmp/W3_SOLVER_2026-08-15 && python3 -c "
  import json,numpy as np,collections
  R=[json.loads(l) for l in open('results.jsonl')]
  V=[r for r in R if r.get('k_drift_valid')]
  for st in ('crowded','sparse'):
    a=np.array([r['k_drift_rel_median'] for r in V if r['stratum']==st])
    print(st,'n',len(a),'median %.4f max %.4f'%(np.median(a),a.max()))
  print('POOLED rel_median median %.4f'%np.median([r['k_drift_rel_median'] for r in V]))
  print('POOLED k_drift_max(ABS rad/m) max %.4f'%max(r['k_drift_max'] for r in V))
  print('SHIP valid', sum(r.get('k_drift_valid',0) for r in R if r['arm']=='SHIP'),'/',
        sum(r['arm']=='SHIP' for r in R))
  print(collections.Counter(r['k_drift_note'] for r in R if r['arm']=='SHIP'))"
  ```
- 필드 정의: `tmp/W3_SOLVER_2026-08-15/run_w3.py:_k_drift` (`k_drift_max = np.max(d)` 절대 ·
  `k_drift_rel_median = np.median(d/kmag)` 상대 · `if not oracle: return` 게이트)
- `tmp/W4_SOLVER_2026-08-19/AMENDMENT_1.md:113-116` (A-10 항목 4) `[R]`

---

## 5. ⚠`amplitude_refit` 부호 결함 — **`CLAUDE.md` 에 경고 없음** (신규 항목 제안)

### ① 현재 문장

**없다.** `CLAUDE.md` 「Important Design Decisions」·「측정 규율」 어느 절에도 `amplitude_refit` 언급이 없다.
저장소 안 유일한 기술은 오늘 산출물(`recon/W1_path.md:193`, `recon/B3_pathfacts.md:164`)과
`tmp/tracker_debate_2026-08-05/` 뿐이다.

### ② 실측 `[M]` — 결함 확정, 과거 오염 0건, **그러나 자기진단이 못 잡는다**

**(a) 부호 결함 — 수학으로 확정**

모델 규약(`model/spectral_gaussian.py:6`, `:124-127`): `z = Σ wᵢ·cos(kxᵢx + kyᵢy − ωᵢt + φᵢ)`
설계행렬(`sequential_update.py:1296-1318`): 열이 `[cos θ, sin θ]`, `θ = kx·x + ky·y − ω·t` (φ 제외)

전개하면 `w·cos(θ+φ) = (w cos φ)·cos θ − (w sin φ)·sin θ` ⟹ **`a = w cos φ`, `b = −w sin φ`**
⟹ 올바른 역산은 **`φ = arctan2(−b, a)`**.

```
sequential_update.py:1334:  w_refit   = np.sqrt(a ** 2 + b ** 2)      # ✅ 올바름 (부호 무관)
sequential_update.py:1335:  phi_refit = np.arctan2(b, a)             # ⛔ 부호 누락 ⟹ φ → −φ
```

**저장소 자신의 올바른 구현**: `model/tnse.py:216` → `phi = np.arctan2(-b, a)` (같은 `θ` 규약).
배포 학습 경로는 이쪽을 쓴다(`learned_init.py:311-320 ls_fit_wphi` → `tnse.fit_amplitudes_phases`).

**(b) 수치 재현** `[M]` — 깨끗한 합성(2,000점, `w=1.7, φ=0.9`):

| | 값 |
|---|---|
| `arctan2(b, a)` (현 코드) | **−0.900000** |
| `arctan2(-b, a)` (올바름) | **+0.900000** (참값 0.9) |
| `w = √(a²+b²)` | 1.700000 (참값 1.7) — **무영향** |
| 잔차 rms (올바른 부호) | 3.6e−16 |
| **잔차 rms (틀린 부호)** | **1.9254 = 신호 rms 의 1.59배** |

⟹ **φ 만 부호 반전**되고 진폭은 멀쩡하다. `W1_path.md` 의 실프레임 값(잔차 1.87 > z rms 1.14 = **1.64배**)과
같은 크기 `[R]`.

**(c) ★가장 위험한 성질 — 함수의 자기보고가 결함을 숨긴다** `[M]`

```
sequential_update.py:1355:  residual_after = float(np.sqrt(np.mean((A @ coeffs - z_vec) ** 2)))
```
`residual_after` 는 **원시 `coeffs`(= 올바른 LS 해)** 로 계산된다. 반면 모델에 설치되는 것은
`:1361-1362` 의 **`phi_refit`(부호 반전)** 이다.
⟹ ★**함수는 "좋은 잔차"를 인쇄하면서 "나쁜 φ"를 설치한다.** `verbose` 출력(`:1366-1367`)도, 반환
dict(`:1371-1376`)도 이 불일치를 **드러내지 않는다**. 켠 사람은 정상 동작으로 읽게 된다.

**(d) 과거 오염 = 0건** `[M]` / `[R]`

- 유일 실행 게이트: `run.py:398 if cfg.get('amplitude_refit', False):` → `:402 updater.amplitude_refit(...)`. **기본 False**.
- 현 작업트리 전수 grep: 이 키를 **True 로 두는 곳 0곳**.
- `git log -S` 6패턴(`'amplitude_refit': True` / `amplitude_refit=True` / `"amplitude_refit": true` /
  `amplitude_refit'] = True` / `--amplitude_refit` 등) → **0 커밋**.
  최초 도입 커밋 `874419a`(2026-04-29) 메시지 자체가 *"amplitude_refit: … 구현 (기본 off)"*.
- 추가 방벽: `:1279 if self._bank is not None: return {}` (bank-MMSE 경로는 진입 자체를 안 한다).
- ⚠**부재증명 한계** `[A]`: 커밋되지 않은 REPL/일회성 세션은 잡을 수 없다.

**(e) 테스트 커버리지 0** `[M]`
`wavesplat/tests/test_bank_mmse_port.py:146` → `assert upd.amplitude_refit([new]) == {}`.
이것은 `:1279` 의 **bank 조기 반환**만 밟는다 ⟹ `:1330-1335` 의 수학은 **한 번도 실행되지 않는다**.

### ③ 정확한 대체 문장 (= `CLAUDE.md` 신규 경고, 「Important Design Decisions」 말미 삽입 제안)

> - ⚠**`amplitude_refit` 에 `(a,b) → φ` 부호 결함이 있다** (2026-08-19 확정, **런 0회 · 코드 직독 + 산술**).
>   `optimizer/sequential_update.py:1335` 이 `phi_refit = np.arctan2(b, a)` 인데, 모델 규약
>   `z = Σ w·cos(θ+φ)`(`model/spectral_gaussian.py:6`)와 설계행렬 `[cos θ, sin θ]`(`:1296-1318`)에서는
>   `a = w cos φ`, **`b = −w sin φ`** 이므로 올바른 식은 **`np.arctan2(-b, a)`** 다.
>   ⟹ **φ 가 부호 반전**된다(`w = √(a²+b²)` 는 무영향). 틀린 부호의 잔차는 **신호 rms 의 약 1.6배**로
>   재구성이 원신호보다 나빠진다. 저장소 안 **올바른 구현이 이미 있다: `model/tnse.py:216`**
>   (배포 학습 경로가 쓰는 `learned_init.py:311 ls_fit_wphi` 가 그쪽이다).
>   ✅**과거 결과 오염 0건**: 유일 게이트 `run.py:398 cfg.get('amplitude_refit', False)` 는 **기본 False**,
>   저장소에서 **True 로 두는 곳 0곳**, `git log -S` 6패턴 **0커밋**, 도입 커밋 `874419a`(2026-04-29)부터
>   이미 off. (⚠커밋되지 않은 세션은 부재증명 불가.)
>   ★★**그런데 이 결함은 자기진단으로 안 잡힌다** — `:1355 residual_after` 는 **원시 `coeffs`(올바른 LS 해)**
>   로 계산되고 모델에는 `:1362` 의 **부호 반전된 `phi_refit`** 이 설치된다 ⟹ **함수가 "좋은 잔차"를 인쇄하면서
>   "나쁜 φ"를 넣는다.** 켜서 돌려 보고 정상이라 판단하지 마라.
>   ⚠**테스트 커버리지 0**: `tests/test_bank_mmse_port.py:146` 은 `:1279` 의 bank 조기 반환만 밟는다.
>   ⟹ **처방 = 한 글자 수정(`-b`) + 왕복 회귀 테스트**(알려진 `(w,φ)` 로 프레임 합성 → refit → `φ` 복원 확인).
>   ⛔**`:1335` 의 부호 규약을 새 코드에 베끼지 마라. 올바른 원본은 `model/tnse.py:216` 이다.**
>   ⚠**축으로는 죽었다** — 닫힌형 `(a,b)` 재적합의 이득은 `PROBE-A` 에서 **+0.3%, 6/16 = 널**이다.
>   **버그는 그것과 무관하게 고친다.**

### ④ 근거

- 코드: `wavesplat/optimizer/sequential_update.py:1259-1376`(특히 `:1279`, `:1296-1318`, `:1334-1335`,
  `:1355`, `:1361-1362`) · `wavesplat/model/spectral_gaussian.py:6, 124-127` ·
  `wavesplat/model/tnse.py:216` · `wavesplat/optimizer/learned_init.py:311-320` · `wavesplat/run.py:396-402` ·
  `wavesplat/tests/test_bank_mmse_port.py:146`
- `[M]` 재도출 명령:
  ```bash
  python3 -c "
  import numpy as np
  rng=np.random.default_rng(0); w,p=1.7,0.9
  th=rng.uniform(-3,3,2000); z=w*np.cos(th+p)
  A=np.column_stack([np.cos(th),np.sin(th)]); (a,b),*_=np.linalg.lstsq(A,z,rcond=None)
  print('a vs w cos phi', a, w*np.cos(p)); print('b vs -w sin phi', b, -w*np.sin(p))
  print('code  arctan2(b,a) =',np.arctan2(b,a)); print('right arctan2(-b,a)=',np.arctan2(-b,a))
  bad=np.hypot(a,b)*np.cos(th+np.arctan2(b,a))
  print('resid(wrong)/rms(z) = %.2fx'%(np.sqrt(((z-bad)**2).mean())/np.sqrt((z**2).mean())))"
  ```
- 선행 산출물: `recon/W1_path.md:193-203, 372` · `recon/B3_pathfacts.md:164-217, 277` ·
  `AUDIT_NEUTRAL.md:196-197` · `recon/W2_upper_bound.md:370-399, 475, 516` `[R]`

---

## 6. `reference_citation_bans.md` — 제안

⚠**총량 표현 금지 준수**: 아래를 개수로 세지 말고 **날짜 태그 `(08-19)` 로 가리켜라**.
파일 머리말의 계수 규칙(`- ` 로 시작하는 줄, `⛔` 출현 수)에 맞춰 **한 줄 묶음**으로 적었다.

### 6.1 ★추가할 항목 (신규 묶음)

```
- **(08-19 W4 재감사 — 문서 정정)** ⛔**"빔포밍·`MIN_DK`·성장 분기·난수 위상·`RX2_GROW_FITPHI` 가 배포 혼잡에서 실행되지 않는다"**(`CLAUDE.md` 측정규율 · `MEMORY.md` 08-15) → **거짓**. `W3 SHIP` 실측 성장 발화 **혼잡 17/96**(학습 **8/87=9.2%** · `v2_konly` **9/9**) · 희소 6/48. `_initialize_learned` 가 우회하는 것은 **`coarse_to_fine.py` staging 하나뿐**(그 사이트는 학습 경로 **0/130** = 참으로 0회)이고 **두 번째 구현** `sequential_update.py:1065-1200` 은 조건절 `:1104` 가 `'learned'` 를 포함해 살아 있다(`:1111-1116` `MIN_DK`, `:1193` 난수 위상, 호출 지점 `:660`). 독스트링(`:620-626`)은 원래 *"the same adaptive prune+grow"* 라 **명시**했고 문서가 그 문장을 떨어뜨렸다. 허용 = **"성장 분기는 배포 혼잡 런의 82%에서 발화하지 않는다"** + **라우트·크기 병기**(넣는 원자 **28/1,660 = 1.69%**) · ⛔**`17/96` 과 `8/87` 을 단위 표시 없이 섞지 마라**(전자는 층, 후자는 라우트) · ⛔**S1 의 "출하 팔은 beamforming 을 아예 하지 않는다"** → 그 프로브는 **에피소드 1개·시드 1개**(`beamform_probe.txt`)다; 허용 = "그 에피소드에서 `auto` 0회 · `v2_konly` 3회" · ⛔**K1 의 사유 "출하 혼잡 경로가 `MIN_DK` 를 안 쓴다"**·⛔**K4 의 사유 "난수 위상 site 도달 불가"** → **둘 다 거짓**(판정 `사망` 은 유지, 사유는 **지렛대 부족 = 발화 9.2%·원자 1.69%** 로 교체) · ⛔**"격차는 곱이다"** → 두 요인 곱은 개선 **×1.58** 인데 측정 격차 **×110**(어긋남 ×69.8, **log 지분 9.7%**); ⚠**곱셈 분해 자체는 무죄** — 한 표본(S4 16ep) 안에서는 곱 ×0.00924 가 측정 결합 ×0.00941 을 **1.8% 오차로** 맞힌다; 허용 = **"한 표본 안에서 격차는 `k` 다(×0.00922, 16/16). `(a,b)` 는 널(6/16)"** + ⛔**서로 다른 배치의 요인 비를 곱해 상한을 만들지 마라**(`S4∩K6=0` · `W3∩K6=0`, 같은 출하 팔이 **0.15301/0.16552/0.18399 = 20.2% 산포**) · ⛔**`×110.00`(중앙값의 몫)과 `×112.82`(짝 비의 중앙)를 같은 값으로** → 다른 추정량, 표기 분리 · ⛔**"축이 죽은 이유 = 전부 하나만 고쳐서"** → **`ARGUED`**(축별 귀속 자료 **0건**, 게다가 K1·K4 사유가 08-19 에 정정됨) · ⛔**누적 축 수의 총량 인용**(이 파일 머리말의 총량 금지가 그대로 적용된다 — 열거하라) · ⛔**"`merge_atoms` 는 공짜 티켓 / 이미 구현·미노출"** → **미노출이지 미검증이 아니다**; `tmp/unified_guard_2026-07-26`(PREREG 존재, 동결 `0.25·kp`·`0.375·kp`, 500런)이 **이미 쟀고 떨어졌다**: 혼잡 **0승/20셀**(0.25 = 0승9무1패 · 0.375 = 0승4무**6패**), **`0.375` `no_harm` FAIL(Wilcoxon p=0.03125, 중앙 Δ +0.00088 악화)**, 희소 **양 변형 `FAIL` 티어**(0.0549/0.0546 vs FULL 0.0254·PARTIAL 0.037, **병합 없는 0.0534 보다 나쁨**); ⛔**"문턱 0.03 < 분해능 0.042–0.074 이니 올리면 된다"** → 측정된 두 설정의 실제 반경은 **0.0191·0.0287 로 아직 0.042 미만인데 이미 해쳤고**, 0.042 도달에는 `merge_frac≈0.55` 가 필요한데 **0.5 에서 셀 55%가 원자를 잃는다**(중앙 15→11) · ⛔**"k 표류 중앙 15%/최대 34%"**(`MEMORY.md` 08-16 · `project_w3_solver_2026-08-16.md:53-54`) → **두 겹 결함**: (i) **층 풀링**(혼잡 23.3% vs 희소 0.04% = 세 자릿수 차, 풀링 17.2% 는 **어느 모집단도 기술하지 않는다**), (ii) **단위 다른 두 필드 혼용** — "34%" 는 `k_drift_max` = **`rad/m` 절대량 0.3437** 이지 백분율이 아니다(**상대 최대는 52.1%**); 허용 = **"혼잡 중앙 22.5–25.0%(최대 52.1%) · 희소 중앙 0.03–0.07%"** · ⛔★**이 자료로 출하 팔의 k 표류를 어느 방향으로도 주장 금지** — `k_drift_valid` 가 `SHIP` **0/144**(`not-an-oracle-arm`), `KONLY`·`ORA-K-CVX`·`ZERO` 도 0 ⟹ **배포 팔의 표류는 미측정** · ⚠오라클 팔조차 6/144 무효, 사유 *"prune/grow broke the 1:1 map"* ⟹ **성장이 발화하면 진단이 꺼진다** · ⚠**`amplitude_refit`(`sequential_update.py:1335`) 부호 결함** — `arctan2(b,a)` → 올바름은 **`arctan2(-b,a)`**(올바른 원본 `model/tnse.py:216`); **과거 오염 0건**(기본 False·True 설정 0곳·`git log -S` 6패턴 0커밋) 이나 ⛔**"켜서 돌려 보니 정상"으로 판단 금지** — `:1355 residual_after` 는 **원시 `coeffs`(올바른 해)** 로 계산되고 설치는 `:1362` 의 **부호 반전 φ** 라 **자기진단이 결함을 숨긴다**; ⛔**`:1335` 의 부호 규약을 새 코드에 복제 금지**
```

### 6.2 ★뒤집혀서 **삭제/교체**해야 할 항목

**⚠먼저 정직하게**: 이번 감사에서 **기존 `reference_citation_bans.md` 항목 중 뒤집힌 것은 확인되지 않았다.**
`beamform`·`빔포밍`·`merge` 로 그 파일을 grep 한 결과 **해당 항목이 애초에 없다** — 1·3·4 의 잘못된 서술은
**금지목록이 아니라 `CLAUDE.md`·`MEMORY.md`·`project_*.md` 의 본문에 있었다.**
(2026-08-12 감사가 *"실패는 인덱스·금지목록·`CLAUDE.md` 에 몰려 있었다"* 고 한 패턴의 재발이다.)

⟹ **금지목록에서 삭제할 항목은 없다.** 대신 **본문에서 삭제/교체할 곳**은 아래와 같다(적용은 사용자).

| 파일 | 위치 | 조치 |
|---|---|---|
| `CLAUDE.md` | 측정 규율절 — "⟹ **빔포밍 · `MIN_DK` · 성장 분기 · 난수 위상 · `RX2_GROW_FITPHI` 는 배포 혼잡에서 실행되지 않는다.**" | **삭제 후 §1③ 문안으로 교체** |
| `CLAUDE.md` | 측정 규율절 — "수리 `merge_atoms` 가 **이미 구현·GT-free·끄면 비트 동일**인데 … **미노출**" | **"수리/공짜 티켓" 프레이밍 삭제**, §3③ 문안으로 교체(사실 나열은 유지) |
| `CLAUDE.md` | 「Important Design Decisions」 말미 | **§5③ 경고 신규 삽입** |
| `MEMORY.md` | 08-15 novelty 항목 — "★★★둘의 사인이 같고 … 실행되지 않는다" | **§1③ 문안으로 교체** |
| `MEMORY.md` | 08-15 novelty 항목 — "★공짜 티켓: … `merge_atoms` 가 이미 구현·미노출" | **§3③ 문안으로 교체** |
| `MEMORY.md` | 08-16 W3 항목 — "**k 를 중앙 15% 흘린다**(최대 34%)" | **§4③ 문안으로 교체** |
| `MEMORY.md` / `CLAUDE.md` | "★축이 죽는 것은 … 안 돌아가는 코드를 연구해서다" (+ 누적 축 총량) | **§2③ 문안으로 교체**(`ARGUED` 표기 + 총량 제거) |
| `project_novelty_search_2026-08-15.md` | K1 행 · K4 행 · §2 "★★★진짜 발견" | **판정 유지 · 사유만 §1③ 문안으로 교체** |
| `project_w3_solver_2026-08-16.md` | `:53-54` G-12 문장 | **§4③ 문안으로 교체** + 유효행 표기 정정 |

### 6.3 ⚠적용 시 지킬 것

1. **판정과 사유를 분리해서 고쳐라.** K1·K4 는 **사유가 틀렸지 판정이 틀린 게 아니다**. 판정을 뒤집으면 과잉정정이다.
2. **"돈다"로 뒤집지 마라.** 성장은 **9.2%에서 돌고 90.8%에서 안 돈다**. 두 숫자를 같이 적어라.
3. **라우트·층 접두어를 항상 적어라**(`혼잡 learned 8/87` 처럼). `17/96` 과 `8/87` 은 다른 질문의 답이다.
4. **`merge_atoms` 의 사실 나열(구현됨·GT-free·no-op 기본값·미노출)은 전부 참**이다. 지울 것은 **"공짜 티켓"이라는 가치판단**뿐이다.
5. **`k_drift` 를 인용할 때는 필드명을 적어라** — `rel_median`(무차원)인지 `max`(rad/m)인지.
6. **`amplitude_refit` 은 "고칠 버그"이지 "쓸 축"이 아니다**(닫힌형 `(a,b)` 이득은 `PROBE-A` 에서 널).

---

## 7. 이 문서의 한계 `[A]`

- **`W4/results.jsonl` 미열람**(사전노출 차단). 따라서 W4 배치가 1번·4번에 대해 무엇을 말할지는 여기 없다.
- 3번의 07-26 자료는 **순수 `learned` 팔 · λ=1e-3 · 레거시 희소 코퍼스**다. **기록 무손상 / 재실행 시 비동일**
  (S1–S8 과 같은 규칙). "공짜 티켓" 을 지우기엔 충분하나, **"영원히 죽었다"는 주장에는 부족**하다.
- 5번의 "과거 오염 0건" 은 **커밋된 이력에 한정**된 부재증명이다.
- 1번의 발화율 9.2%·1.69% 는 **W3 코퍼스·시드(1592–1594)** 에서의 값이다. 다른 코퍼스에서 달라질 수 있다.
- 2번의 "축별 귀속 자료 0건" 은 `project_novelty_search_2026-08-15.md` 와 `tmp/NOVELTY_SEARCH_2026-08-15/KILL.md`
  열람 범위에서의 판단이다.
