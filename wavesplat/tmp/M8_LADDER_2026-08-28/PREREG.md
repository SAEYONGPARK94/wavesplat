# M-8 사전등록 — ★**bank → 제안 표현 사다리**: 실측에서 기술은 어디서 죽는가 (2026-08-28)

> ⛔**결과 0행에서 작성.** ⛔**아직 동결되지 않았다** — §14 가 동결 전 남은 것을 적는다.
> 이 문서의 모든 수는 `premise_m8.py → premise_m8.json` 에서 왔다(**손복사 상수 0건**).
> ⚠**1 레코드 스모크로 방향이 노출됐다 — §13 이 축자로 공시한다.**

---

## 0. 질문과 뒤집기

원 질문: *"사전은 왜 실측에서 실패하나"* — **참조 없는 소거**였고, 후보 넷이
`B-14b`·`B-15`·`B-18`·`B-26`·`M4`·`M5b` 로 이미 죽었다.

★**뒤집는다**: ***"bank 가 가진 것을 사전은 무엇을 안 가졌나."***
실측 12 레코드에서 배포 bank 팔이 제안 표현을 **11/12** 로 이긴다
(`B-2` frozen 11/12 · ★**CPU 고정 `M5b` ANCHOR 에서도 11/12**, 유일 반례 `LJ18a` 동일).
⟹ **작동하는 참조 팔이 저장소 안에 있다** ⟹ 소거가 아니라 **대조**를 할 수 있다.

`DIRECTION = LOCATION-OR-DECOMPOSITION`. 산출은 *"경계가 이 고리 위에 놓여 있다"*
또는 *"단일 고리로 분해되지 않는다"* 까지다. ⛔*"원인은 X 다"* 라는 **기제 지목으로 올리지 않는다.**

---

## 1. ★★도달 가능성 판정 — **임무의 절반**. 축별로 먼저 답한다

두 팔은 실제로 **다른 코드 경로**다:

* `SHIP` = `sequential_update._initialize_inner` → **REAL-DOMAIN GATE**(`:706` GT 부재 탐지)
  → `_initialize_bank`(`:1100`) → `bank_mmse.fit_bank_mmse` → `BankMMSEFit.render`
* `PROP` = `_initialize_learned`(`:863`) → `learned_init.learned_init_params`
  → `propose_atoms`(확산+polish) → `pool_and_select` → `ls_fit_wphi`
  → `_adaptive_prune_grow`(`:1407`) → `varpro.apply_varpro` → `SpectralGaussianModel.predict_grid`

### 1.1 ★**컨테이너는 독립 인자가 아니다** (구조적 사실, 런 0회로 확정)

`BankMMSEFit.render` = `(Φ_q·γ) @ w_pred` = `Σ_b [A_b cos ψ_b + B_b sin ψ_b]`,
`SpectralGaussianModel` = `Σ_i w_i cos(θ_i + φ_i)` — **같은 k 집합이 주어지면 두 컨테이너는
같은 함수공간을 친다.** ω 도 양쪽 다 같은 분산관계 하드제약이다.
⟹ ★**"표현 컨테이너" 축은 존재하지 않는다.** 남는 것은 **어떤 k 를, 몇 개, 어디에 놓고,
계수를 어떻게 추정하는가** 넷뿐이다. (⚠유일한 예외 = `clamp_steepness` — 행에 `vp_clamp_hits` 로 계측.)

### 1.2 축별 판정표

| # | 축 | bank | 제안 | 도달 가능? | 근거 |
|---|---|---|---|---|---|
| **A1** | 앙상블 | median-of-3 (val-split 시드) | **없음** | ✅**bank 쪽만** `real_bank_ensemble_k=1` (config) | ⛔**제안 쪽은 구조적 불가** — `make_updater:2013` 이 `init_method=='auto'` 에만 앙상블을 주고, `MedianEnsembleUpdater.initialize:1925` 가 `route != 'v2_konly'` 면 `ensemble_k=1` 로 접는다 |
| **A2** | 후보 기수 `B` | 1904–1918 | ≤30 | ⚠**패키지 가산 패치 필요** (`bank_dk_frac`) — §12 | `dk_c` 가 `_bank_fit_window:1170` 에서 **넘어가지 않는다**(config 표면 0개) |
| **A3** | 파수 배치 | 고정 격자(kp 단위 등간격) + 환대 `(0.35,2.5)kp` | 학습된 **연속** | ⚠**bank 쪽 주입 패치 필요** (`bank_k_set`) ✅**제안 쪽은 무패치** (`k_snap_dk`, 이미 배포됨) | `_snap_k_to_grid:940` (B-16 노브, 기본 None) |
| **A4** | 결합 방식 | MMSE(스펙트럼 사전 γ + val 선택 σ + SBL-EM) | LS ridge 0.5 → **VARPRO 비볼록 k 하강** | ✅ `varpro_steps=0` (config) + A3 의 `bank_k_set` 로 **같은 원자에 두 결합기**를 붙일 수 있다 | — |
| **A5** | 적응 N (prune+grow) | **없다** | 있다 | ✅ `adaptive_n=False` (config) | ★§6 — 실측에서 **grow 가 실제로 발화**한다 |
| **A6** | 관측 예산 | 600 | 조건자 400 / LS·VARPRO 600 | ✅ `bank_max_obs=400`. ⚠단 부차 | 두 부분추출이 **같은 `default_rng(0).choice(600,400)`** 이라 지표집합이 일치한다 |
| **A7** | 신뢰 가중 | 없음 | `weighting='confidence'` | ⛔**구조적으로 무연산 — 측정 불가** | ★`make_gif_bank_deployed.py:78` 이 실측 프레임의 `confidence` 를 **항등적으로 1** 로 만든다(`premise_m8.json:P5` 실측 확인) ⟹ `varpro.py:127` `sqrt(clamp(conf))≡1` = `weighting='none'` 과 **비트 동일**. ⟹ `B-25`(`GT-WEIGHT-HURTS`)는 **실측에 전이되지 않는다** |
| **A8** | 원자 출처(규칙 vs 학습) | beamform 규칙 | 확산 사전 | ✅ 하지만 **A3 와 분리 불가** — bank 의 k 집합은 *격자 그 자체*라 "출처만" 바꿀 수 없다 | ⟹ A3 와 **묶어서** 한 고리로 잰다 |
| **A9** | 창 길이 선호 | — | — | ⛔**재측정 금지** (`M5b`) | 단서로만 |
| **A10** | 추적/재적합 | — | — | ⛔**이 하네스에 없다** | 러너는 `initialize` + `predict_field` 만 부른다(`update()` 0회) ⟹ `bank_refit_*`·`tracking_*` 는 **전부 무관**. 이것은 제약이 아니라 **단일요인을 지켜 주는 성질**이다 |

⟹ ★**결론: 사다리는 만들 수 있다.** 죽은 축은 **둘**이다 — A7(구조적 무연산)과
A1 의 제안 쪽 절반(구조적 불가). 나머지는 **config 5개 + 가산 패치 2키**로 도달한다.

---

## 2. ★사다리 — 9 계단 · 6 고리 · **한 번에 한 인자**

| 계단 | 이름 | 경로 | `DEFAULT_CONFIG` 위에 얹는 것 | 후보 수 |
|---|---|---|---|---|
| **R0** | `BANK` | 실측 게이트 → bank | *(배포 그대로)* | B 1904–1918, k=3 |
| R1 | `BANK-K1` | 〃 | `real_bank_ensemble_k=1` | 〃, k=1 |
| R2 | `BANK-B472` | 〃 | `+ bank_dk_frac=0.2` | **472** |
| R3 | `BANK-B118` | 〃 | `+ bank_dk_frac=0.4` | **118** |
| R4 | `BANK-B30` | 〃 | `+ bank_dk_frac=0.8` | **30** |
| R5 | `MMSE-PROP` | 〃 | `+ bank_k_set = <제안 원자>` | n_sel (≤30) |
| R6 | `PROP-LS` | learned | `real_domain_gate=False, init_method='learned', varpro_steps=0, adaptive_n=False` | n_sel |
| R7 | `PROP-LS-APG` | learned | 〃 (단 `adaptive_n=True` = 배포값) | 가변 |
| **R8** | `PROP` | learned | `real_domain_gate=False, init_method='learned'` *(나머지 배포 그대로)* | 가변 |

★`B(frac)` 는 **레코드 무관**이다(`premise_m8.json:P2`, 12 레코드 실측):
`0.2→472 · 0.4→118 · 0.8→30` 이 **12/12 에서 정확히 같은 수**다
(`0.1` 만 부동소수 끝점 때문에 1904–1918 로 흔들린다).
⟹ ★**R4 의 기수는 제안의 상한 `max_n=30` 과 정확히 일치**한다.

### 2.1 고리 = 인자

| 고리 | 계단 | 바뀌는 **단 하나** |
|---|---|---|
| **L1** | R0 → R1 | 앙상블(median-of-3 → 단일) |
| **L2** | R1 → R4 | 후보 **기수** 1908 → 30 *(용량-반응 R2·R3 4점)* |
| ★**L3** | R4 → R5 | 같은 기수에서 **k 를 어디에 놓는가** — 규칙 격자 vs 학습된 연속 |
| **L4** | R5 → R6 | **결합기** — MMSE(스펙트럼 사전+val σ+EM) vs LS ridge 0.5 |
| **L5** | R6 → R7 | 적응 N(prune+grow) |
| **L6** | R7 → R8 | VARPRO 400 스텝(비볼록 k 정련) |

⚠**L4 는 완전한 단일요인이 아니다**(정직 공시): MMSE 는 *"사전 γ + val 선택 σ + EM"* 세 가지를
한 덩어리로 갖는다. 이 배치는 그 덩어리를 **하나의 인자**로 취급하고, 분해는 별도 사전등록 몫이다.

### 2.2 부차 팔 (Holm 가족 **밖**, 사전등록)

| 이름 | 무엇을 가른다 | 구성 |
|---|---|---|
| `MMSE-PROP-BAND` | **환대 제약**(0.35–2.5 kp)을 배치와 분리 | R5 + 원자를 환대 안으로 필터 |
| `PROP-SNAP` | ★**실측 최초의 `B-16` 절제** — 배포 팔의 파수 연속성만 제거 | R8 + `k_snap_dk = kp_bank/10` |
| `PROP-LS-SNAP` | 같은 절제를 결합기·적응 N 없이 | R6 + `k_snap_dk = kp_bank/10` |

⚠`k_snap_dk` 는 **VARPRO 뒤에 사후 양자화**한다(`_initialize_learned:917`) — bank 처럼
*"처음부터 격자 위에서 적합"* 이 아니다. ⛔**둘을 같은 것이라 쓰지 않는다.**
`kp_bank` 는 bank 자신의 2단 beamform zoom(GT-free)에서 온다.

---

## 3. ★이미 답해진 부분 (⛔중복 측정 금지)

| 이미 잰 것 | 판정 | 이 배치가 **하지 않는 것** / 남은 것 |
|---|---|---|
| `M4` 제안 **용량 ↑** (max_n 30→128, e_th·ddk 0) | `CAPACITY-HURTS` 12/12 | ⛔같은 방향 재측정 금지. ★**남은 것 = 반대 방향** — *bank* 의 기수를 30 으로 **내리면** 기술이 남는가(L2). 그리고 M4 가 늘린 것은 **같은 확산 표본의 근중복 원자**이고 L2 가 줄이는 것은 **분산 다양체 격자 피복**이다 — 다른 양이다 |
| `B-16`/`B-19` 파수 연속성 절제 | 합성에서 **연속성이 번다**(46/48, 14/16) | ⛔합성 결론을 실측에 전이 금지. 실측에서 그 노브를 돌린 배치는 **0건** ⟹ `PROP-SNAP` 은 신규 |
| `W12` VARPRO 가 관측 목적함수 −63% 인데 장 오차는 제로필드보다 나쁨 | — | L6 가 실측 12 레코드에서 **짝으로** 재확인. ⛔`W12` 의 크기 인용 금지 |
| `B-8` `V2K` 실측 14/36 | 규칙 기반 k 도 무기술 | ⟹ `V2K` 팔을 **넣지 않는다**(1런 70 s 로 비싸고 이미 답했다) |
| `M5b` 창 길이 | `WINDOW-LONGER-IS-BETTER` | 단서로만. 창은 **W=40 고정** |
| `B-25` GT 유래 신뢰가중 | 합성에서 `GT-WEIGHT-HURTS` | ⛔실측 전이 불가 — A7(신뢰도 ≡ 1) |

---

## 4. 표본 · 채점 · 시드

* **표본**: 실측 **12 레코드**(`gates_b2.RECORDS` 임포트, 손복사 0) · 창 **win=3** ·
  잡음 **0.02** · `scene_station(cap_mode='uniform')`.
  ★**`cap_mode='uniform'` 을 쓰는 이유**: 이 배치는 `W` 를 움직이지 않으므로 `M5b` 가 닫은
  *"cap 추첨 모집단이 W 의 함수"* 경로가 **애초에 열리지 않는다**. 대신 `'uniform'` 은
  `B-2`·`M4` 동결 행과 **비트 앵커**를 준다(§5). ⟹ 규약 선택의 사유가 결과 무관하다.
* **채점**: `sqrt(mean((Z−gt)^2))/sg`, 러너 축자 동일. ⛔`ZERO_REF=1.0` **사용 금지** —
  기술 서술이 필요하면 `zero_ref.zero_ref_nowcast(record, win=3)` (실측 범위 **0.3375–1.1559**,
  `premise_m8.json:P6`). ★**판정은 전부 짝 비교**이고 제로필드는 **부수 서술 전용**.
* **분리**: `nowcast`(lead 0, PRIMARY) 와 `leadmean`(lead 0–14 평균, 사전등록 부차)을
  **따로** 보고한다(`B-3` 이 lead 에서 갈렸다). 행에 `curve` 전체를 싣는다.
* **추정기 시드**: `EST_SEEDS = (42, 43, 44)`, 계단·레코드별 **median-of-3**
  (`S13b`·`B-22`·`M4` 선례). ⛔단일 시드 금지.
  ⚠시드는 bank 쪽에서 **val 분할**(`fit_bank_mmse:139`)과 앙상블 오프셋을,
  제안 쪽에서 **확산 시드**와 **성장 위상 난수**를 동시에 움직인다 — 같은 이름의 다른 채널이다.
  ⟹ 행에 **시드 산포**(3값 전부)를 싣고 판정문에 병기한다.

---

## 5. ★★양성대조(앵커) — **하나는 성립하고 하나는 [BLOCKER] 다**

| 앵커 | 대상 | 상태 |
|---|---|---|
| **`G-ANCHOR-B`** | R0(seed 42) == `B-2` frozen `SHIP` 12/12 **비트** | ✅**성립한다** — 오늘 CPU 에서 `AA14` 재현 확인: `0.4500722313936653` / `0.5796657042696121` **비트 동일** |
| **`G-ANCHOR-P`** | R8(seed 42) == `M4` frozen `SHIP-K-INL` 12/12 **비트** | ✅**성립한다** — `AA14` `0.7301224313151458` / `0.8931412662608678` / `N=23` / `apg_dn=[-7]` **비트 동일**(두 번 독립 재현) |
| ~~R8 == `B-2` frozen `PROP`~~ | | ⛔⛔**성립 불가 — 12/12 전부 다르다** |

### 5.1 ⛔⛔[BLOCKER-A] **`B-2` 의 `PROP` 열은 CPU 에서 재현되지 않는다**

오늘 실측: `B-2 PROP` vs 오늘 CPU 재실행이 **12/12 레코드에서 다르다**
(`BS13a` 2.7722 → 0.7643 = **−72%**, `AA15a` 0.6204 → 1.4591 = **+135%**, `N` 도 갈린다).
패키지 코드는 **기능적으로 동일**하다(`9d09159`→HEAD 의 `sequential_update.py` 차이는
`k_snap_dk` **no-op** 하나, `learned_init.py`·`varpro.py` 는 **독스트링만**).

★**원인은 코드가 아니라 디바이스다**(코드 사실 두 줄로 확정, `premise_m8.json:P7`):

1. `learned_init.propose_atoms:130-132` — `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
   ★**이 함수는 `config['device']` 를 인자로 받지도 않는다** ⟹ 추정기 config 의 `device='cpu'` 가
   **배포 제안 경로의 디바이스를 통제하지 못한다.**
2. `scripts/d3_beamform_cond_diffusion.sample:160-163` — `torch.manual_seed(seed)` 뒤
   `torch.randn(n, M, 3, device=DEVICE)`. **같은 시드라도 CPU 와 CUDA 는 다른 수열**을 낸다.

⟹ `CUDA_VISIBLE_DEVICES` 를 세우지 않은 배치(`B-2`·`B-3`·`B-7` 러너 = **0회**)는 GPU 가 보이면
GPU 원자로 돌았고, 세운 배치(`M4`·`M5b` = 각각 5회·2회)는 CPU 원자로 돌았다.

⚠**이것은 가설이 아니라 코드 사실 + 재현 실패의 결합이다.** 결정적 시험(같은 레코드를
CUDA 가시 상태로 1런)은 ⛔**오늘 GPU 금지로 막혀 있다** — 그 시험을 §14 에 남긴다.

★**이 배치에 대한 귀결(전부 결과 전 결정)**:
* 제안 끝단의 앵커는 **`M4 SHIP-K-INL`** 이다. ⛔`B-2 PROP` 을 앵커로 쓰지 않는다.
* 러너는 `CUDA_VISIBLE_DEVICES=''` 를 **torch 임포트 전에** 세우고, 행마다
  `cuda_available` 을 싣는다(`G-DEV` 가 전 행 `False` 를 요구).
* ★**끝점 대비를 수입하지 않고 이 배치가 직접 낸다**(`G-END`) — `B-2` 의 11/12 를 전제로 삼지 않는다.
  (참고: CPU 고정 `M5b` ANCHOR 에서도 11/12 로 같다 ⟹ 현상 자체는 디바이스에 견딘다.)

---

## 6. ★★전역 RNG 결합 — 설계의 핵심 위험, 그리고 처방

★**실측에서 성장 가지가 실제로 발화한다**(오늘 확인, `AA14`):
배포 `PROP` 은 `apg_dn = [-7]` 인데 **성장을 막으면**(`grow_residual_threshold=1e9`)
`N = 10` 이다 ⟹ ***가지치기 30→10 뒤 성장이 13개를 넣었고, 그 13개의 위상은
`sequential_update.py:1617 torch.rand` = 전역 torch 스트림에서 나온다.***

⟹ **같은 시드라도 그 지점까지 소모된 draw 수가 다르면 결과가 갈린다.**
실측 증거: 같은 원자(비트 동일, `M4 G-15` maxΔ=0)를 **캐시로** 넣은 팔과 **인라인**으로
제안한 팔이 `AA14` 에서 `0.5997` vs `0.7301` (**+22%**) 로 갈린다 — 오직 `propose_atoms` 가
소모한 draw 때문이다.

### 6.1 처방 넷 (전부 결과 전 고정)

1. ★**계단마다 `propose_atoms` 를 인라인으로 돌린다.** ⛔원자 캐시 재사용 금지.
   ⟹ R6·R7·R8 과 부차 제안 팔은 **성장 지점까지 소모한 draw 수가 구조적으로 같다**
   (`sample()` 이 `torch.manual_seed` 로 스트림을 리셋하고 그 뒤 고정 개수를 뽑는다).
   ⚠단 `_build_model(n)` 이 `4n` draw 를 뽑으므로(`spectral_gaussian.py:57-60`)
   **n_sel 이 같은 계단끼리만** 정렬이 보장된다 ⟹ 행에 `n_sel` 을 싣고 `G-RNG2` 가 대조한다.
2. ★**R5(`MMSE-PROP`)는 RNG 무관하다** — bank 경로에 `torch.rand` 가 없다.
   ⟹ 원자는 러너가 인라인으로 한 번 만들어 주입하고, `atoms_md5` 가 R6–R8 과 **일치**해야 한다(`G-ATOM`).
3. ★**시드 3개 median-of-3** (§4). RNG 결합은 시드 축의 산포로 흡수되고, **산포를 함께 보고**한다.
4. ★**RNG 상태를 계측한다**: 행마다 `rng_hash_after_seed` · `rng_hash_after_init`
   (torch/np 상태의 md5) · `apg_calls` · `apg_dn` · `n_grown` · `n_pruned`.
   ⛔이 값들은 **판정에 쓰지 않는다**(공시·게이트 전용).

### 6.2 ⚠**L5 는 이 결합을 *측정* 하는 고리다**
R6(적응 off, 완전 결정적) → R7(적응 on, 난수 위상 유입). ⟹ ★***"성장의 난수 위상이
실측에서 얼마를 먹는가"*** 가 사다리의 한 칸으로 들어온다. 이것은 부작용이 아니라 산출이다.

---

## 7. MDE — **결과 전에 계산했고, 못 재는 것을 먼저 적는다** (`premise_m8.json:P4`)

n = 12 (레코드), 단측 부호검정, median-of-3.

| | 문턱 | α | 80% 검정력이 요구하는 승률 q |
|---|---|---|---|
| **PRIMARY** (L3, 단일 검정) | **10/12** | 0.0193 | **q = 0.869** |
| 사슬 6 고리 Holm — 1~4번째 기각 | **11/12** | 0.0083 / 0.0100 / 0.0125 / 0.0167 | **q = 0.931** |
| 사슬 6 고리 Holm — 5~6번째 기각 | 10/12 | 0.025 / 0.05 | q = 0.869 |

⟹ ★***이 배치는 "12 레코드 중 10–11개를 뒤집는" 크기만 잡는다.*** 부분 효과(q≈0.7)는
높은 확률로 널로 읽힌다. **부호검정 바닥 p = 2.44e-04**(12/12) — n=12 로는 원리적 상한이다.
⛔**따라서 "고리 하나에 전부 몰려 있다" 를 통계로 증명할 수 없다** — 증명할 수 있는 것은
*"이 고리는 문턱을 넘고 저 고리는 못 넘는다"* 이고, 그것은 **크기 순위가 아니다**(`F1` 교훈).

---

## 8. 판정 (⚠결과 전에 전부 적는다)

### 8.1 PRIMARY (확증, 검정 **1개**)

**L3 = `MMSE-PROP` vs `BANK-B30`**, `nowcast`, n=12(median-of-3), 단측 부호검정, 문턱 10/12.

| 라벨 | 조건 | 무엇을 쓰는가 |
|---|---|---|
| `PRIOR-ATOMS-EARN` | `MMSE-PROP < BANK-B30` ≥10/12 | ★**기여 ①의 실측 최초 양성** — *"같은 기수·같은 결합기·같은 관측에서 학습된 원자가 규칙 격자를 이긴다."* ⛔*"사전이 실측에서 작동한다"* 로 **올리지 않는다**(배포 팔은 여전히 진다) |
| `PRIOR-ATOMS-LOSE` | 역방향 ≥10/12 | *"원자 자체가 규칙 격자보다 나쁘다"* ⟹ 경계가 **사전 자신** 위에 있다 |
| `PRIOR-ATOMS-UNDECIDED` | 둘 다 아님 | ⛔*"차이 없다"* 금지(등가검정 안 함). MDE(q=0.869) 병기 |

★**PRIMARY 를 L3 으로 고른 사유(결과 전)**: L2 의 방향은 사전적으로 거의 자명하고
(기저 함수가 줄면 나빠진다), L3 만이 **학위 기여 ①을 실측에서 직접 시험**한다.
그리고 **양방향 모두 논문에 쓸 문장이 있다.**

### 8.2 사슬 (사전등록 부차 가족 6, Holm) — **국소화 라벨**

각 고리에서 *"제안 쪽 계단이 bank 쪽 계단보다 나쁘다"* 를 단측으로 검정한다.
Holm 생존 집합 `S` 로 라벨을 정한다:

| 라벨 | 조건 | 읽기 |
|---|---|---|
| `SKILL-DIES-AT-<Lk>` | `|S| = 1` | 그 고리 하나가 문턱을 넘었다. ★`Lk == L3` 이면 **CONFIRMATORY**, 아니면 **EXPLORATORY**(PRIMARY 가 아니므로) |
| `SKILL-DIES-DISTRIBUTED` | `|S| ≥ 2` | ⟹ ***"단일 원인이 아니다"***. 논문 §`sec:boundary` 에 **분해표**를 싣고 ⛔*"가장 큰 고리"* 를 **지목하지 않는다**(크기 순위는 이 검정력으로 불가) |
| `GAP-NOT-FACTORIZABLE` | `S = ∅` **이고** `G-END` 통과 | 끝점은 갈리는데(≥10/12) 어느 단일 고리도 문턱을 못 넘었다 ⟹ **상호작용**. 다음 설계는 **2요인 팔**이거나 **레코드 증설**이다 |
| `LADDER-VOID` | `G-END` 실패 | ⛔**판정 없음.** 끝점이 알려진 대비를 재현하지 못하면 사다리는 아무것도 국소화할 수 없다 |

★**크기 서술은 로그비 분해로만**(서술 전용, 검정 아님):
`Δ_k = median_records log(x_{k+1}/x_k)`, `Σ Δ_k` vs `log(R8/R0)` 를 표로 낸다.
⛔`F1` 교훈대로 **척도를 셋 병기**한다(짝 승수 · 절대 nRMSE · 참 제로필드 대비).

### 8.3 ⛔**"아무 데서도 안 죽는다" 는 설계 결함 먼저 의심한다**

`S = ∅` 이면 판정 전에 **개입 되읽기**가 통과했는지부터 본다(§9 `G-INT`).
`G-INT` 가 하나라도 실패하면 라벨은 `LADDER-VOID` 다. ⛔사후에 이 순서를 바꾸지 않는다.

---

## 9. 게이트 (✅**2026-08-28 `gates_m8.py` 로 구현 완료** — 결과 0행에서)

차단(blocking) / 범위축소(scoping)를 **결과 전에** 나눈다.
★**구현된 `SCOPING` = `{G-BAND, G-LEAD, G-0b2, G-0d2}`** — 뒤의 둘은 이 표를 코드로 옮기면서
**결과 0행에서 추가**한 것이고, 사유는 다음 한 줄이다: **부차 3팔은 Holm 가족 밖이므로
그 팔의 에러/비유한은 사다리 판정을 막을 이유가 없다**(반면 사다리 9계단의 에러는 차단이다).
⚠유일하게 예상되는 실패 경로는 `MMSE-PROP-BAND` 의 환대 필터가 원자를 0개로 만드는 경우다.

★**아래 표와 코드의 이름 대응**(코드가 정본, `gates_m8.gates()`):
`G-B` → **`G-INT-B`**(+`G-INT-B2` 단조성), `G-INT` → **`G-INT`**(경로·라우팅) +
`G-INT-N`/`-KSET`/`-KSET2`/`-ENS`/`-VP`/`-APG`/`-SNAP`(축별 되읽기),
`G-PATCH` → **`G-PATCH-SIG`**(배선 되읽기) + **`G-PATCH-REG`**(회귀 산출) + **`G-PATCH-OFF`**(두 키가
R0·R1 에서 꺼져 있다), `G-DEFAULT` → `G-DEFAULT`/`-2`/`-3`. 신설 = `G-0e`(작업목록 대조) ·
`G-SCENE-B2`(장면이 `B-2` 와 비트동일) · `G-KP`(beamform kp) · `G-SEED`(시드 3개 전건).
**총 38 게이트(차단 34 · 범위축소 4), 자가진단에서 커버리지 0 인 것 0개.**

| 게이트 | 무엇을 |
|---|---|
| `G-0/0b/0c/0d` | 런 수 == 선언값 · 에러 0 · 셀 중복 0 · `nowcast`·`leadmean` **유한**(NaN 은 `a<b` 에서 조용히 패배가 된다) |
| **`G-DEV`** | ★전 행 `cuda_available == False` (§5.1 BLOCKER-A 의 기계화) |
| **`G-INT`** | ★★**개입 되읽기 — 설정이 아니라 *객체* 에서**: `bank_B`(BankMMSEFit.n_candidates) · `bank_dk_c` · `bank_k_set_md5` · `n_gaussians` · `vp_steps_executed` · `apg_calls`/`apg_dn` · `k_snap.fired` 가 **계단별 선언값과 일치**. ⛔`effective_config` 를 입력 dict 복사로 만들지 않는다 |
| **`G-B`** | R2·R3·R4 의 `bank_B` 가 **472 / 118 / 30**, 12/12 레코드 전부(`premise P2` 와 대조) |
| **`G-ATOM`** | ★R5–R8 · 부차 제안 팔의 `atoms_md5` 가 (record, seed) 안에서 **전부 일치** |
| **`G-ANCHOR-B`** | R0(seed 42) == `B-2` frozen `SHIP` **비트** 12/12 |
| **`G-ANCHOR-P`** | R8(seed 42) == `M4` frozen `SHIP-K-INL` **비트** 12/12 |
| **`G-END`** | ★R0 < R8 이 ≥10/12 (끝점이 알려진 대비를 재현) — 실패 시 **판정 없음** |
| **`G-PATCH`** | ★`bank_dk_frac=None, bank_k_set=None` 이 **패치 전과 비트 동일**(§12 회귀가 낸 md5 를 행에서 대조) |
| `G-DEFAULT` | 배포 3키(`varpro_steps=400`·`learned_init_iters=0`·`auto_switch_thresh=0`)를 **객체**에서 되읽고 전 런 고정. ⛔빈 dict `{}` 로 팔을 정의하지 않는다 |
| `G-RNG` | `rng_hash_after_seed` · `rng_hash_after_init` 이 **비어 있지 않다**(계측 shim 사망 = 항등 통과 금지) |
| `G-RNG2` | R6·R7·R8 의 `n_sel` 이 (record, seed) 안에서 일치 |
| `G-ZERO` | `zero_ref` 가 `zero_ref.py` 에서 오고 **어느 레코드도 정확히 1.0 이 아니다** |
| `G-SCENE` | `obs_hash`·`q_hash`·`sigma_bits`·`kp_bits` 가 계단에 걸쳐 **비트 동일**(단일요인) |
| **`G-COV`** | ★**커버리지 메타게이트**(`M5b G-9`) — 모든 게이트가 실제로 본 행/짝 수를 세고 **0 이면 FAIL**. ⚠자가진단은 **러너의 작업목록·스탬프에서 만든 합성 행**으로 시험한다(`M4 G-13` 의 필터 항진) |
| `G-BAND` *(scoping)* | R5 의 `frac_in_band` 기록 — 실패 시 `MMSE-PROP-BAND` 해석만 빠진다 |
| `G-LEAD` *(scoping)* | `curve` 길이가 레코드별 lead 수와 일치 |

---

## 10. ⛔**널이면 무엇을 쓰는가** (이 질문에 답 없으면 설계하지 않는다)

1. **PRIMARY 널(`PRIOR-ATOMS-UNDECIDED`)** — *"같은 기수의 규칙 격자 대비 학습 원자의
   우열은 12 레코드·q=0.869 검정력에서 확정되지 않는다"*. ⛔*"차이 없다"* 금지.
   그래도 **사슬 표는 남고**(§8.2), 논문 §`sec:boundary` 는 *"경계는 원자의 질에서
   확정되지 않았고, 확정된 것은 아래 고리들이다"* 로 쓸 수 있다.
2. **`GAP-NOT-FACTORIZABLE`** — ★이것도 산출이다. 지금 논문은 실측 음성에 대해
   *"bank 가 무엇을 더 갖는지"* 를 **말할 자료가 0개**다. 이 라벨은 그것을
   *"넷 중 어느 하나도 아니다 — 상호작용이다"* 로 바꾸고, **다음 설계를 결정한다**
   (2요인 팔 또는 레코드 증설; 12 레코드가 상한이면 그 사실 자체가 학위 한계 절의 문장이다).
3. **`LADDER-VOID`** — 개입이 안 걸렸다는 뜻이므로 **결과를 열지 않고** 진단서를 쓴다
   (`M5`→`M5b` 선례: 0행 재동결).
4. ★**어떤 분기에서도 살아남는 산출 둘** (판정과 독립):
   (a) §5.1 **BLOCKER-A** — 배포 제안 경로가 `config['device']` 를 통제하지 못한다는 코드 결함과
       그로 인한 `B-2 PROP` 열의 재현 불가. **이미 확정됐다.**
   (b) §6 **성장 난수 위상**이 실측 배포 팔에서 발화하고 그 결합이 +22% 를 움직인다는 사실.

---

## 11. ⛔이 배치가 **주장하지 않는 것** (결과 전 공시)

1. ⛔**기제 지목 금지** — `DIRECTION = LOCATION-OR-DECOMPOSITION`.
2. ⛔**배포 권고 금지.** 어느 고리가 이겨도 *"bank 를 배포에 넣자"*·*"max_n 을 올리자"* 로 올리지 않는다
   (별도 사전등록 몫). ⚠특히 `M4` 가 제안 팔의 용량 증가를 **12/12 로 죽였다**.
3. ⛔**크기·순위 주장 금지**(`F1`) — 고리 간 크기 비교는 이 검정력으로 불가. 척도 셋 병기 의무.
4. ⛔**`B-16` 합성 결론의 실측 전이 금지** — `PROP-SNAP` 은 **사후 양자화**이지 격자 적합이 아니다.
5. ⛔**사이트 수준 일반화 금지**(사이트 4개) · ⛔**외부 기준선 추가 금지**(정지규칙 `S17`+`B-5`+`B2_WANGFIX`) ·
   ⛔**세 번째 LETKF 수리 금지**.
6. ⛔`ZERO_REF = 1.0` 금지. 짝 비교만 판정에 쓴다.
7. ⛔**`M5b`·`M4` 의 계수와 이 배치의 계수를 직접 비교 금지**(장면 규약·팔이 다르다).
8. ⛔**A7(신뢰 가중)·A1 제안쪽 앙상블을 "측정했다" 고 쓰지 않는다** — **도달 불가**로 공시한다.
9. ⛔**L4 를 "MMSE 가 낫다/못하다" 로 분해하지 않는다** — 사전 γ·val σ·EM 이 한 덩어리다(§2.1).

---

## 12. 패키지 패치 — **가산 2키, 기본값 보존형** (`PATCH.md` 가 정본)

`wavesplat/optimizer/bank_mmse.py` + `sequential_update.py:_bank_fit_window` 에
`dk_frac` · `k_set` 을 추가한다. **둘 다 기본 `None` = 패치 전과 비트 동일.**
선례 = `k_snap_dk`(B-16) · `bank_refit_every_s`(D) · `cap_mode`(M5b).
★**동결 전 필수 회귀**(`regress_bank_mmse.py`, 추정기 런 0회):
L1 전건 기본인자 비트 동일 · L2 `dk_frac=0.1` == 기본 · L3 `k_set` 주입이 손으로 만든 적합과 비트 동일 ·
L4 처치가 **실제로 다름**(B 가 움직인다). ⚠`test_bank_mmse_port.py` 가 계속 통과해야 한다.

---

## 13. ⚠★**노출 공시 — 이 설계는 blind 가 아니다**

설계 중 **도달 가능성 확인용 스모크**를 돌렸고, 그 과정에서 **`AA14` 한 레코드에서
12개 팔의 방향을 보았다**(산출물은 scratchpad 에 있었고 배치 디렉터리에 남기지 않았다).
본 것: `BANK`·`BANK-K1`·`BANK-dk{0.2,0.4,0.8}`·`MMSE-PROP`·`PROP`·`PROP-NOVP`·
`PROP-SNAP`·`PROP-NOVP-SNAP`·`PROP-NOGROW`·`PROP-NOADAPT`·`V2K` 의 `nowcast`/`leadmean`.

★**무엇이 그 노출의 영향을 받았나 (정직하게)**:
* **받았다**: 사다리에 어떤 계단을 넣을지(특히 `bank_dk_frac` 격자점 0.2/0.4/0.8 선택),
  그리고 PRIMARY 를 L2 가 아니라 **L3** 으로 고른 것.
* **받지 않았다**: 문턱(10/12) · Holm 규약 · 라벨 네 갈래 · MDE · 금지 목록 ·
  `DIRECTION` — 전부 저장소의 기존 규약을 그대로 가져왔다.
* ⚠**PRIMARY 의 라벨은 양방향 대칭으로 적혀 있다**(§8.1) ⟹ 본 방향이 라벨을 만들지 않는다.
⛔**이 사실을 결과 문서에서 지우지 마라**(`B-25` 선례: *"blind 아님, 방향 1표 노출 공시"*).

---

## 14. 동결 상태 (2026-08-28 갱신 — ✅**1–5 완료**, ⛔**6 만 미해결**)

1. ✅`gates_m8.py` — 상수·팔·**작업목록**·**행 스탬프**·문턱·게이트·라벨·**결과 전 해석 규칙**(`READING`)의
   **유일 정의** + 자가진단(변형 하네스 **34종**, ★**러너의 작업목록·스탬프에서 만든 합성 행**,
   ★**팔 12개를 하나씩 지워 "런 수 게이트 말고도 누가 FAIL 하는가"** 를 강제) + `--refcheck`.
   자가진단 **PASS(0 실패)** · 커버리지 0 게이트 **0개**.
2. ✅`run_m8.py` — §2 표를 `gates_m8` 에서 읽는 러너. `CUDA_VISIBLE_DEVICES=''` 를 **torch 임포트 전
   첫 줄**에서. `assert_shipping_defaults()` · `assert_bank_patch()` · `regress_pass()` 를
   **런 0회에서** 통과해야 시작한다. 1셀 연기 통과(§15 아래 기록).
3. ✅`analyze_m8.py` — 0행/부족 → exit 1 · 차단 FAIL → **판정 없이** exit 1 ·
   `G-END` 단독 실패 → **`LADDER-VOID`** 를 적고 PRIMARY·사슬·크기표를 **내지 않는다**(§10.3 축자).
4. ✅`regress_bank_mmse.py` — §12 의 네 겹 + 가드/항진 방지 겹. 산출 `regress_bank_mmse.out`
   (`regress_bank_mmse: PASS (0 실패)`, 비교 165건). `G-PATCH-REG` 가 그 산출을 **신선도까지** 검사한다
   (산출이 `bank_mmse.py`·`sequential_update.py`·회귀 소스보다 나중이어야 한다).
5. ✅`FROZEN_md5.txt` — 판정 코드 **전부 + 임포트 경로 파일 + 외부 앵커 자료** 35항목
   (패키지 8 · 스크립트 7 · 배치 단일출처 8 · 확산 체크포인트 2 · 앵커 jsonl 14).
   `md5sum -c` 전건 OK. ★*"동결한 적이 없다"* 가 더 나쁘다.
6. ⛔**미해결 시험 하나**: §5.1 의 결정적 확인(같은 레코드를 **CUDA 가시** 상태로 1런 → `B-2 PROP` 재현 여부).
   오늘 GPU 금지로 막혀 있다. ⚠**이 시험 없이는 BLOCKER-A 의 원인 지목이 "코드 사실 + 재현 실패"
   까지이고 "확정"이 아니다.**

★**방벽 4종 통과**(런 0회, 결과 0행 시점):
`check_batch_artifacts.py` = A/B/C/F/자기동결/빈dict **OK**(결과 `⚠미실행` 은 배치 전이라 당연) ·
`check_gate_refs.py` = **공허 참조 0** (경고 2건은 `'error' not in r` 방어 관용구) ·
`check_gate_blindspots.py` = **항등 0 · 가드없음 0**(`synthetic_rows()` 훅으로 사전등록 시점 검사) ·
`check_thresholds.py` = **우연과 구분되지 않는 문턱 0건** ·
`find_handcopied_refs.py` = `gates_m8.py` **적발 0건**.

---

## 15. 비용 · 실행 절차

**런 수** = 12 계단(R0–R8 9 + 부차 3) × 12 레코드 × 3 시드 = **432**.
⛔GPU 0회 · 에피소드 생성 0회 · 자료는 이미 디스크에 있다(`data/real_bergamasco/`, 12 레코드).

⚠**설계 시점의 단가 추정은 §15.1 의 실측으로 대체됐다**(하네스가 완성된 뒤 1셀을 실제로
돌려 쟀다). 종전 추정은 *"(record, seed) 당 55–70 s · 직렬 35–45 분 · 6 샤드 7 분"* 이었고
실측과 같은 자릿수다. ⚠`YS17` 의 bank 는 `B-2` 에서 9.5 s 였으므로 여유를 두고
**직렬 90분 · 6샤드 20분**을 상한으로 잡는다.

★**실행 절차 — 이 블록을 그대로 붙여넣어라. ⛔단계를 건너뛰거나 순서를 바꾸지 마라**
(오케스트레이터가 2026-08-28 에 **두 번** 절차를 안 읽어 사고를 냈다).

    cd /home/vexit/wavesplat/wavesplat      # ★반드시 이 디렉터리. 아래 경로는 전부 이 기준이다.

    # ── 0) 동결 대조 (⚠다른 에이전트 동시 작업 중 — **실행 직전** 반드시) ───────────
    md5sum -c tmp/M8_LADDER_2026-08-28/FROZEN_md5.txt | grep -v ': OK$'   # 출력 0줄이어야 한다

    # ── 1) 런 0회 — 전제 · 패치 회귀 · 게이트 자가진단 · 외부참조 ─────────────────
    #    ⚠회귀는 **반드시 `regress_bank_mmse.out` 으로 리디렉션**하라 —
    #      `G-PATCH-REG` 가 그 파일을 (신선도까지) 읽는다. 화면에만 찍으면 게이트가 FAIL 한다.
    CUDA_VISIBLE_DEVICES= python3 tmp/M8_LADDER_2026-08-28/premise_m8.py
    CUDA_VISIBLE_DEVICES= python3 tmp/M8_LADDER_2026-08-28/regress_bank_mmse.py \
        2>&1 | tee tmp/M8_LADDER_2026-08-28/regress_bank_mmse.out
    CUDA_VISIBLE_DEVICES= python3 tmp/M8_LADDER_2026-08-28/gates_m8.py            # selftest
    CUDA_VISIBLE_DEVICES= python3 tmp/M8_LADDER_2026-08-28/gates_m8.py --refcheck
    #    → 넷 다 exit 0 이어야 한다. `--refcheck` 가 MISSING 을 찍으면 **거기서 멈춰라**.

    # ── 2) 연기 1셀 (12팔 × AA14 × seed 42, ~55 s. 결과 파일은 스스로 지운다) ──────
    python3 tmp/M8_LADDER_2026-08-28/run_m8.py smoke
    #    → 마지막 두 줄의 **★앵커 … 비트 동일 OK** 를 눈으로 확인하라. 하나라도 ⛔면 멈춰라.

    # ── 3) 본 배치 432런 · 6 샤드 ─────────────────────────────────────────────
    #    ⚠**추적되는 방식으로 돌려라**(W15 교훈: `nohup &` 로 떼면 완료 알림이 안 온다).
    #    ⚠환경변수는 러너가 스스로 `setdefault` 하지만, 셸에서도 명시해 둔다(이중 방어).
    export CUDA_VISIBLE_DEVICES=
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    for s in 0 1 2 3 4 5; do
      python3 tmp/M8_LADDER_2026-08-28/run_m8.py run $s 6 \
        > tmp/M8_LADDER_2026-08-28/run_m8_$s.log 2>&1 &
    done; wait
    grep -c '"arm"' tmp/M8_LADDER_2026-08-28/results_m8_*.jsonl   # 합이 432 여야 한다

    # ── 4) 판정 ───────────────────────────────────────────────────────────────
    python3 tmp/M8_LADDER_2026-08-28/analyze_m8.py          # → verdict_m8.json

    # ── 5) 방벽 (배치 뒤) ─────────────────────────────────────────────────────
    python3 tmp/_audit/check_batch_artifacts.py
    python3 tmp/_audit/check_gate_refs.py     tmp/M8_LADDER_2026-08-28
    python3 tmp/_audit/check_gate_blindspots.py tmp/M8_LADDER_2026-08-28
    python3 tmp/_audit/check_thresholds.py    tmp/M8_LADDER_2026-08-28
    md5sum -c tmp/M8_LADDER_2026-08-28/FROZEN_md5.txt | grep -v ': OK$'   # 배치 후 재대조

★**샤드 규약** — `run_m8.py run <shard> <nshard>` 는 `gates_m8.expected_jobs()` 를
`i % nshard == shard` 로 자른다. 작업 순서가 `(record, seed)` 바깥 · `arm` 안쪽이고
팔이 12개라 **`nshard=6` 이면 샤드 k 가 `arm_idx % 6 == k` 인 두 팔을 전담**한다:
`0`=BANK+PROP-LS · `1`=BANK-K1+PROP-LS-APG · `2`=BANK-B472+PROP · `3`=BANK-B118+MMSE-PROP-BAND ·
`4`=BANK-B30+PROP-SNAP · `5`=MMSE-PROP+PROP-LS-SNAP.
⟹ **(record, seed) 를 쪼개지 않는다** — 장면과 원자 제안이 셀 안에서 재사용된다.
⚠**이어받기가 된다**(완료 셀을 건너뛴다) — 중단 후 같은 명령을 다시 돌리면 이어간다.
⛔`nshard` 를 6 이 아닌 값으로 바꾸면 팔이 샤드에 섞여 **원자 제안이 중복 실행**된다
(결과는 같지만 시간이 는다).

⛔`analyze_m8.py` 가 `label: null` 로 종료하면 **결과 파일을 열지 마라**(`M5` 의 교훈).

### 15.1 실측 단가 (2026-08-28 연기 1셀, `AA14`/seed 42, CPU 1스레드)

`BANK` 2.1–4.5 s · `BANK-K1` 2.1 s · `BANK-B472` 0.4 s · `BANK-B118` 0.1 s · `BANK-B30` 0.1 s ·
`MMSE-PROP` 4.5 s(원자 제안 포함) · `MMSE-PROP-BAND` 0.1 s(원자 재사용) ·
`PROP-LS` 3.8 s · `PROP-LS-APG` 7.9 s · `PROP` 8.1 s · `PROP-SNAP` 9.2 s · `PROP-LS-SNAP` 3.8 s
⟹ 셀(12팔) ≈ **45–55 s** ⟹ **직렬 ≈ 30–45 분 · 6 샤드 ≈ 8–15 분**(가장 무거운 샤드가 1·4번).
