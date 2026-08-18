# PREREG — `W4 · SOLVER-REMEDY` (진단을 처방으로 바꿀 수 있는지 판정한다)

**배치 ID** `W4_SOLVER_2026-08-19` · **산출 디렉터리** `tmp/W4_SOLVER_2026-08-19/`
**설계 시점 git HEAD** `cb4b4fb` (설계 중 런 0회)
**동결 방식**: 이 파일 + 러너 + 분석기의 md5 를 `FROZEN_md5.txt` 에 기록하고 **런 전에 git 커밋**한다
(W3 `196050e` · S17 `81c33b0` 선례 승계 ⟹ `self-attested + git-anchored`).
⛔**"preregistered" 대신 "self-attested freeze + git-anchored"** 로 표기한다.

**패키지 코드 `wavesplat/**` · `configs/` · `CLAUDE.md` · auto-memory · 논문은 이 배치가 건드리지 않는다.**
읽기 전용 입력: `tmp/W3_SOLVER_2026-08-15/` · `tmp/S17_WHITE25_2026-08-18/` · `tmp/S4_phase_oracle_2026-08-08/`
· `tmp/S5_window_loss_2026-08-08/`.

---

## §0. 한 문장

> **배포 혼잡 경로가 실행하는 500스텝 joint Adam 하강이 — 자기가 출발한 닫힌형 해보다 나은 곳으로 가는가,
> 아니면 그 해를 망가뜨리는가.** 그리고 망가뜨린다면, **빼는 것**과 **교대 분리최소제곱으로 바꾸는 것** 중
> 무엇이 처방인가.

---

## §1. 왜 이것이 W3 의 음성과 다른 질문인가 (★이 절이 이 배치의 존재 이유다)

### 1.1 W3 가 이미 죽인 것 (재탕 금지 목록)

| W3 대비 | 결과 | 무엇을 죽였나 |
|---|---|---|
| `SHIP-FITPHI` vs `SHIP` | 3승 **25동점**, `UNDECIDED-NO-INTERVENTION` | 배포 팔에 **개입 지점이 없다** |
| `KONLY-FITPHI` vs `KONLY` | **20/32 FAIL**(개입 96/96 = 완전) | ★**온전히 개입해도 안 산다** |
| `ORA-K-NOREG` vs `ORA-K@L4` (S-1) | **18/32 널** | ★**λ 축 완전 종결** — 출하 λ=1e-4 ≈ λ=0 |

⟹ ⛔**이 배치는 (a) 위상 재적합(`RX2_GROW_FITPHI`) · (b) `lambda_l1` 을 처치로 쓰지 않는다.**

### 1.2 W3 의 FITPHI 가 만진 것 vs W4 가 만지는 것 — **같은 코드가 아니다**

- **W3 FITPHI** = `_rx2_gate('RX2_GROW_FITPHI')` ⟹ **새로 성장한 원자에 어떤 위상값을 심을 것인가**.
  성장 분기(`coarse_to_fine._beamform_residuals` · `_adaptive_prune_grow`)의 **초기화 세부**다.
- **W4 PRIMARY** = `sequential_update.py:643` `learned_init_iters` ⟹ **주 하강 루프를 돌릴 것인가 말 것인가**.

**두 지점은 다른 함수, 다른 단계, 다른 기제다.** W3 의 음성은 *"위상 초기값을 더 잘 심어도 안 산다"* 이고,
W4 는 *"하강 루프 자체가 순손해인가"* 를 묻는다. **W4 의 어떤 결과도 W3 의 음성으로부터 연역되지 않는다.**

### 1.3 ★배포 혼잡 경로의 실제 구조 (코드에서 직접 확인, 2026-08-19)

`init_method='auto'` → 혼잡에서 `_initialize_learned` (`sequential_update.py:610-676`):

```
1. learned_init_params()                       # learned_init.py:323
     ├ pool_and_select(dedup_dk=0.03, energy_thresh=0.03, max_n=30)
     ├ merge_atoms(...)                        # :367-369  ← learned_merge_frac 기본 None = no-op
     └ ls_fit_wphi(kx,ky,frames,depth)         # :386 → tnse.fit_amplitudes_phases(lambda_reg=0.5)
                                               #   ★= k 고정 시 (a,b) 에 대한 닫힌형 ridge 최소제곱
2. for _ in range(learned_init_iters):         # :643-650  ★기본 500, joint Adam over (kx,ky,w,phi)
3. _adaptive_prune_grow(...)                   # :659
4. _build_tracking_optimizer()                 # :661
```

★★★**즉 배포 혼잡 팔은 닫힌형 해를 설치한 다음, 그 위에서 500스텝 경사하강을 돈다.**
`learned_init_iters` 는 **순수 config 노브**이고 (`self.config.get('learned_init_iters', 500)`),
**패키지 코드 수정 없이 0 으로 만들 수 있다.**

### 1.4 이 축이 CLAUDE.md 의 "안 돌아가는 코드" 함정에 걸리지 않는다는 증거 (런 0회, W3 원자료 재집계)

W3 `results.jsonl` 1,212행에서 `SHIP` 혼잡 96행을 되읽었다:
- `init_method_resolved`: **`learned` 87 / `v2_konly` 9 = 90.6%** (CLAUDE.md 의 90.6% 와 일치).
- 에피소드 단위: **26/32 전 시드 `learned`** · **5/32 시드에 따라 뒤집힘** · 1/32 전 시드 `v2_konly`.

⟹ ★**PRIMARY 족은 `init_method='learned'` 를 고정해 개입률 100% 를 구성적으로 보장한다**
(§2 Family A). 라우터를 그대로 둔 배포 가능성 질문은 **별도 SECONDARY**(Family C)로 분리한다.
**S1 의 "출하 팔은 beamform 0회 호출" 4연속 재발을 이 분리로 막는다.**

### 1.5 W3 원자료가 이미 말해주는 사전 신호 (⚠근거지 해석 금지 — 4요인 묶음)

동일 GT `k` 에서: `ORA-K-NOREG`(900 Adam 스텝, λ=0) 혼잡 중앙 **0.13797**
vs `ORA-K-CVX`(**0 스텝**, 무벌점 닫힌형) 혼잡 중앙 **0.00139** ⟹ **×99**.
⚠이 비는 {k 동결 × 닫힌형 × 무추적 × 무가중} **4요인 묶음**이고 **P-1 의 예측치가 아니다.**
P-1 은 그 4요인을 전부 정합시킨 **단일요인 대비**다. 이 문단은 검정력이 있다는 근거로만 쓴다.

---

## §2. 팔 정의 (동결)

**공통 바탕** = 출하 기본값(`run.py:DEFAULT_CONFIG`) 그대로 —
`use_dispersion=True, use_confidence=True, adaptive_n=True, prune_threshold=0.10,
grow_residual_threshold=0.05, max_n=30, tracking_window=3, tracking_mode='w_phi',
tracking_lr=0.005, tracking_steps=10, lr=0.01, lambda_smooth=1e-4, depth=50.0`.

★**`lambda_l1 = 1e-4` 를 모든 팔에 명시한다** (W3 `F-1` 함정 승계: `DEFAULT_CONFIG` 폴백에 맡기면
배포일 이동 시 조용히 무효화된다). `analytic_ensemble_k` 는 **Family A 에서 1**(라우터를 고정했으므로
해석적 폴백 앙상블이 도달 불가), **Family C 에서 3**(출하값).

### 2.1 Family A — 기제 족, `init_method='learned'` **고정** (개입률 100% 구성적 보장)

| # | 팔 | `L-SHIP` 대비 변경 | 요인 수 | 역할 |
|---|---|---|---|---|
| A0 | **`L-SHIP`** | — (`learned_init_iters=500`) | 대조 | 배포 혼잡 경로 대표 |
| A1 | **`L-ITER0`** | `learned_init_iters: 500 → 0` | **단일** | ★**PRIMARY 처치** |
| A2 | `L-ALT` | joint Adam → **교대 분리최소제곱** (§2.4) | **3요인 묶음**(라벨) | S-1 · "바꾸는" 처방 |
| A3 | `L-MERGE` | `learned_merge_frac: None → 0.5` | **단일** | S-2 · 중복 원자 병합 |
| A4 | `L-RIDGE0` | `ls_fit_wphi` ridge `0.5 → 0.0` | **단일** ⚠shrinkage 인접 | S-3 · 초기 닫힌형의 벌점 |
| A5 | `L-SHIP-DUP` | `L-SHIP` 과 **바이트 동일 cfg** | — | 재현 프로브(가설 아님, G-5) |

### 2.2 Family B — 양 끝 (가설검정 대상 아님, **서술 전용**)

| # | 팔 | 정의 | 역할 |
|---|---|---|---|
| B0 | **`ZERO`** | 0 원자, 평평한 바다 | **자명한 하한** · 퇴화 필요조건 |
| B1 | **`ORA-K-CVX`** | **W3 §2.1 정의를 무수정 재사용** — GT `(kx,ky)`, `ω=√(g|k|tanh(|k|d))`, 초기화창 관측 위 무벌점 `lstsq` 1회, 경사하강 0스텝, 추적 없음 | **상한** |

### 2.3 Family C — 배포 가능성 족, `init_method='auto'` (출하 라우터 그대로)

| # | 팔 | 변경 | 요인 |
|---|---|---|---|
| C0 | `A-SHIP` | — (= W3 `SHIP` 구성) | 대조 |
| C1 | `A-ITER0` | `learned_init_iters: 500 → 0` | 단일 |

⚠`v2_konly` 로 라우팅된 에피소드에서 `learned_init_iters` 는 **도달 불가**(개입 0) ⟹ **G-4b 의 대상**.

### 2.4 `L-ALT` 정의 (동결) — ⛔**"VARPRO 구현" 이라 부르지 않는다**

`_initialize_learned` 의 500스텝 joint 루프를 다음으로 **치환**한다 (총 Adam 스텝 500 = **연산 정합**):

```
for round in range(10):
    # (i) k 전용 하강 50스텝: kx,ky requires_grad=True / w,phi requires_grad=False
    # (ii) 닫힌형 재적합: A,phi = fit_amplitudes_phases(kx,ky,init_obs,depth,lambda_reg=0.5)
    #      → model.w, model.phi 에 설치   (★ridge 0.5 = 출하값 유지 ⟹ L-RIDGE0 과 교락 없음)
```

- 라운드마다 `_build_optimizer()` 로 Adam 을 재생성한다(모멘트 리셋 10회).
- 닫힌형 재적합 총 **10회**. 마지막 라운드 뒤 재적합이 마지막 연산이다.
- ⚠★**단일요인이 아니다.** 정직한 라벨 = **{파라미터 블록 교대 × 닫힌형 재적합 × Adam 모멘트 10회 리셋}
  3요인 묶음**. `RESULT.md` 는 이 라벨을 그대로 쓴다.
- ⛔**Golub–Pereyra(1973) VARPRO 가 아니다** — 투영자를 미분하지 않는다. 정확한 이름은
  **교대 분리최소제곱(alternating separable least squares)**.

### 2.5 `L-MERGE` 의 용량(dose)을 결과 전에 고정하는 근거

`merge_atoms` 반경 `r = merge_frac · kp`. 코퍼스 BF4/BF5 의 `kp ≈ 0.095–0.161 rad/m`
(`k=ω²/g`, `Tp≈5–6.5 s`) ⟹ **`merge_frac=0.5` ⟹ `r ≈ 0.048–0.081`**,
CLAUDE.md 가 실측으로 적은 **분해능 한계(반경 0.042–0.074)** 를 정면으로 겨눈다.
배포 분리 문턱 `learned_dedup_dk=0.03` 은 그 한계 **양쪽보다 작다** ⟹ 위험은 **한 주엽 안 중복 원자**.
⛔**용량 스윕을 하지 않는다** — 단일 용량 단일 대비다. 널이면 ⛔**"병합은 무용" 금지**,
허용 = *"`merge_frac=0.5` 에서 효과 없음"*(S6 `NO-DOSE` 교훈).

---

## §3. 표본 · 층 · 시드 (동결)

### 3.1 에피소드 — **신규 `.h5` 생성 0회**

| 층 | 코퍼스 | 구성 | n | 용도 |
|---|---|---|---|---|
| **혼잡** | `data_s8_crowded_2026-08-10` | BF4/BF5 × 15/30comp × seed 32–39 | 32 | **PRIMARY** |
| **혼잡** | `data_s9_crowded_2026-08-11` | BF4/BF5 × 15/30comp × seed 48–55 | 32 | **PRIMARY** |
| 희소 | `data_indep_sparse_2026-08-08` | BF4/BF5 × 2/5comp × seed 20–23 | 16 | **기술통계 전용** |
| 희소 | `data_s9_sparse_2026-08-11` | BF4/BF5 × 2/5comp × seed 48–51 | 16 | **기술통계 전용** |

- **혼잡 n = 64** (32+32). `corpus` 를 매 행에 기록하고 **32/32 분해를 서술로 병기**한다.
  **판정은 풀링된 64 위에서 한다 — 사전 동결.**
- ⚠**사전노출 고지**: `data_s9_crowded` 의 16 에피소드는 W3 에서 이미 사용됐고, `data_s8_crowded` 는 S8 에서
  사용됐다. **그러나 `learned_init_iters` 를 만진 배치는 저장소에 0건**이므로 이 대비에 대한 사전노출은 **0** 이다.
  이 문장은 `RESULT.md` 에 그대로 옮긴다.
- 희소는 **판정에 쓰지 않는다**(HOSGATE 가 희소 우위를 표현 아티팩트로 확정). 존재 이유는 **음성 대조**다:
  S4 가 혼잡 26.9배 vs 희소 1.09배 널이라는 **레짐 비대칭**을 기록했으므로, 처치가 혼잡에서만 듣는지 본다.

### 3.2 시드 (원장 `[1595–1818]` 에서 소모)

| 단계 | 추정기 시드 | 비고 |
|---|---|---|
| **Stage 1** | **1595** (단일) | Stage 2 와 **교집합 없음** |
| **Stage 2** | **1596 · 1597 · 1598** (median-of-3) | 확증 |

- 소모 후 원장 잔여 = **`[1599–1818]`**.
- ★**Stage 1 행은 Stage 2 의 어떤 판정에도 들어가지 않는다**(엿보기 오염 차단). 폐기가 아니라 **분리**다.
- **median-of-3 를 조건분기 없이 건다** (S13b: 혼잡 **18/64 에서 시드가 승패 부호를 가른다**; 시드 CV 혼잡 14.4%).
- `ZERO` · `ORA-K-CVX` 는 결정론적 ⟹ `est_seed=null`, 복제 1회.
- **충돌 검사**는 G-9 가 `tmp/**/*.jsonl` 전체를 훑어 런 전에 수행한다.

### 3.3 ★독립 단위 = **에피소드** (가짜복제 금지)

팔·에피소드마다 **3 시드의 중앙값**을 먼저 취하고, **그 다음** 에피소드 단위로 부호검정한다.
⛔`episode × seed` 를 표본으로 세지 않는다 (S4 가 `48/48` → 정직값 `16/16` 으로 정정된 사건).
G-6b 가 팔별 시드 퍼짐을 인쇄하고, 퍼짐이 정확히 0 인 팔은 `deterministic=True` 로 표시된다.

---

## §4. ★2단 설계 — 값싼 결정 실험을 먼저 돌린다

### 4.1 Stage 1 (`W4-P0`) — **256런 ≈ 25분**

| 팔 | 층 | ep | 시드 | 런 |
|---|---|---|---|---|
| `L-SHIP` · `L-ITER0` · `L-ALT` | 혼잡 | 64 | 1595 | 192 |
| `ZERO` | 혼잡 | 64 | — | 64 |

**비용 실측 근거**: W3 `SHIP` 혼잡 **중앙 21.1 s/런** (`results.jsonl` 96행 재집계).
192 × 21 s = **67분 직렬**; GPU 경합 탓에 **워커 3** ⟹ **≈ 25분**. `ZERO` 는 0.1 s/런.

**Stage 1 → Stage 2 게이트 (결과 전 동결)**

- **G0-a (하드, 개입)**: `L-SHIP.adam_steps_phase1 == 500` 64/64 · `L-ITER0.adam_steps_phase1 == 0` 64/64 ·
  `L-ALT.konly_adam_steps == 500` **및** `closed_form_refits == 10` 64/64 ·
  Family A 전 192행 `init_method_resolved == 'learned'`.
  **하나라도 위반 ⟹ 중단, Stage 2 없음, 판정 인쇄 없음.**
- **G0-b (신호)**: `L-ITER0` 또는 `L-ALT` 중 **적어도 하나**가 `L-SHIP` 대비 **≥41/64 또는 ≤23/64**.
  - **통과 ⟹ Stage 2 실행.**
  - **미달 ⟹ 판정 `NO-SOLVER-SIGNAL` 로 종결.** 그 문장은 —
    ✅*"단일 시드 n=64 에서 배포 Phase-1 옵티마이저를 **완전히 제거해도, 교대 분리최소제곱으로 대체해도**
    판정이 움직이지 않는다"* = **솔버 축의 25분짜리 폐쇄**.
    ⛔그 경우에도 *"옵티마이저는 무관하다"* 는 쓰지 않는다(단일 시드 · 두 처치 한정).

⚠**정직 고지 (조건부성)**: Stage 2 는 Stage 1 신호에 **조건부**다 ⟹ Stage-2 의 p 값은
**무조건부 I종 오류율이 아니다.** 완화 요소 두 가지를 결과 전에 적는다:
① **시드가 disjoint** 하므로 같은 자료를 두 번 보지 않는다.
② **방향을 고르지 않는다** — P-1 은 **양측**으로 동결돼 있어 Stage 1 이 방향 선택에 개입할 여지가 0 이다.

### 4.2 Stage 2 (확증, 조건부) — **1,740런 ≈ 3.0시간**

| 족 | 팔 | 층 | ep | 시드 | 런 |
|---|---|---|---|---|---|
| A | `L-SHIP` `L-ITER0` `L-ALT` `L-MERGE` `L-RIDGE0` | 혼잡 | 64 | 3 | 960 |
| C | `A-SHIP` `A-ITER0` | 혼잡 | 64 | 3 | 384 |
| A' | `L-SHIP` `L-ITER0` | 희소 | 32 | 3 | 192 |
| B | `ZERO` | 혼잡 64 + 희소 32 | 96 | — | 96 |
| B | `ORA-K-CVX` | 혼잡 64 + 희소 32 | 96 | — | 96 |
| A5 | `L-SHIP-DUP` | 혼잡 4 | 4 | 3 | 12 |

**총 = 1,740런.** 추정기 런 1,548 × 21 s ≈ 9.0 h 직렬 ⟹ **워커 3 에서 ≈ 3.0 h**.
**배치 전체 = 1,996런 ≈ 3.5 h.**

⚠**비용 상한이 걸리면 에피소드를 줄이지 않는다.** 줄일 수 있는 것은 팔뿐이고 순서는
`L-RIDGE0` → `L-MERGE` → 희소 슬라이스 → Family C 이며, 줄이면 해당 SECONDARY 는 `NOT-RUN` 으로 인쇄된다.
**PRIMARY 3팔(`L-SHIP`·`L-ITER0`·`ZERO`)은 절대 줄이지 않는다.**

---

## §5. PRIMARY — 질문 **1개**

> **P-1.** 혼잡 층 64 에피소드에서, `init_method='learned'` 를 고정한 팔 쌍에 대해
> **`avg_nRMSE(L-ITER0) ≠ avg_nRMSE(L-SHIP)`** — 추정기 시드 1596·1597·1598 의 **median-of-3** 를 먼저 취한 뒤,
> 에피소드 단위 **양측** 부호검정, 기각역 **≥41/64 또는 ≤23/64**.

- **정확 α = 0.032766** (양측). 바로 안쪽(`≥40`/`≤24`)은 `0.059941` ⟹ **40/64 는 통과가 아니다**(하나 차이 사건 사전 고지).
- **족 크기 1 ⟹ 다중성 보정 없음.**
- ★**양측인 이유**: 두 방향이 **모두** 과학적으로 결정적이다.
  `L-ITER0` 승 ⟹ 배포 옵티마이저가 **순손해**(빼는 것이 처방). `L-ITER0` 패 ⟹ 옵티마이저가 값을 벌고
  **"벽은 최적화 실패다" 라는 척추가 배포 경로에서 미지지**. ⟹ ⚠**방향을 사후에 고르지 않는다.**

**사전 검정력 (부호검정, 팔당 승률 p)**

| p | 0.60 | 0.65 | **0.6875** | 0.70 | 0.75 | 0.80 |
|---|---|---|---|---|---|---|
| power | 0.298 | 0.618 | **0.828** | 0.879 | 0.982 | 0.999 |

- **MDE**: power 0.80 ⟹ `p ≥ 0.682`; power 0.90 ⟹ `p ≥ 0.707`.
- ⚠**원리적 통과 가능성 확인 (S6 이 죽은 사유)**: 문턱 41/64 는 정확 이항으로 도달 가능하고,
  세 결과(`≥41` · `24–40` · `≤23`)가 **전부 명명된 판정으로 매핑**돼 있다(§8) ⟹ **원리적 통과 불가 구간 0**.
- ⚠**위음성 게이트 금지 확인 (S17 `KILL-2` 가 죽은 사유)**: 이 배치의 게이트는 전부 **카운터·스키마·비트동일성**
  같은 **관측 가능한 이산량**이고, 척도무관 통계량(`A_win` 류)을 게이트로 쓰지 않는다.

**IUT 필요조건 (실패 시 `UNDECIDED` 가 아니라 `FAIL`)**:
`L-SHIP < ZERO` 혼잡 **≥41/64**. (W3 실측 `SHIP` 0.153 vs `ZERO` 0.249 ⟹ 여유 큼.)

---

## §6. SECONDARY (전부 **SECONDARY 라고 라벨**, Holm-4 · 족 α=0.05 양측 · **m 은 4 로 고정**)

| ID | 대비 | 층 | n | 질문 | 요인 |
|---|---|---|---|---|---|
| **S-1** | `L-ALT` vs `L-SHIP` | 혼잡 | 64 | 교대 분리최소제곱이 joint Adam 을 이기는가 | **3요인 묶음** |
| **S-2** | `L-MERGE` vs `L-SHIP` | 혼잡 | 64 | 중복 원자 병합(`0.5·kp`)이 버는가 | 단일 |
| **S-3** | `L-RIDGE0` vs `L-SHIP` | 혼잡 | 64 | 초기 닫힌형의 ridge 0.5 가 해치는가 | 단일 ⚠shrinkage 인접 |
| **S-4** | `A-ITER0` vs `A-SHIP` | 혼잡 | 64\* | **출하 라우터 그대로**에서도 같은가 (배포 가능성) | 단일 |

**Holm-4 step-down** (W3 `B-2` 규율 승계): p 오름차순으로 보다가 **처음으로 기각되지 않는 가설에서 정지**하고
이후는 **전부 기각 불가**. `m = 4` 고정 — 어떤 SECONDARY 가 `NOT-RUN`·`UNDECIDED-NO-INTERVENTION` 이어도
**분모는 4**.

**순위별 실효 문턱 (양측, n=64 — 결과 전 명시)**

| p 순위 | Holm α | 필요 승수 | 정확 양측 p |
|---|---|---|---|
| 1 | 0.01250 | **≥43 또는 ≤21** | 0.00815 |
| 2 | 0.01667 | **≥43 또는 ≤21** | 0.00815 |
| 3 | 0.02500 | **≥42 또는 ≤22** | 0.01686 |
| 4 | 0.05000 | **≥41 또는 ≤23** | 0.03277 |

⛔**결과를 본 뒤 "41/64 로 문턱을 넘었다" 라고 쓰지 않는다** — 순위에 따라 43 이 필요할 수 있다(S8 이 22/32 로 죽은 전례).
\* **S-4 의 n**: §7 G-4b 의 **완전개입 부분집합**(3시드 전부 `learned` 로 라우팅된 에피소드) 위에서 판정한다.
라우팅은 두 팔에서 **동일**하므로 이 부분집합은 **팔에 균형**이다. **부분집합 < 48 ⟹ `UNDECIDED (n)`**.
(W3 실측 26/32 = 81.3% ⟹ 64 에피소드에서 **≈52 예상**.)

**희소 (n=32) — ⛔판정 아님, 기술통계 전용.** `L-ITER0` vs `L-SHIP` 만 병기하고,
양측 참조 기각역 `≥23/32 또는 ≤9/32` 를 **참고값으로만** 인쇄한다.

**퇴화 필요조건 (discipline 10, W3 `B-4` 승계)**: 어떤 팔이든 해당 층에서 **`ZERO` 를 이기는 에피소드가
≤50%** 이면 `DEGENERATE` 로 표시된다. **처치·대조 어느 쪽이라도** 퇴화면 그 행에 `(DEGENERATE-ARM: …)`
꼬리표를 붙이고, 쓸 수 있는 문장은 **인과-손상 형식뿐**:
✅*"X 가 그 팔의 오차를 Y 만큼 바꾼다 — 단 두 팔 모두 평평한 바다보다 나쁘다"* /
⛔*"X 가 추정을 개선한다"* · ⛔*"배포 가치가 있다"*. **수치는 그대로 인쇄한다.**

---

## §7. ★개입 게이트 (이 배치의 핵심 계측 — 카운터 이름·문턱·실패 분기를 결과 전에 동결)

### 7.1 원자 카운터 (러너가 `W4Updater` 안에서 증가시키고 매 행에 기록)

| 필드 | 정의 | 기대값 |
|---|---|---|
| `adam_steps_phase1` | Phase-1 초기화에서 `optimizer.step()` 호출 **총 횟수** | `L-SHIP`/`A-SHIP` **500** · `L-ITER0`/`A-ITER0` **0** |
| `konly_adam_steps` | `w,phi` 가 **frozen** 인 상태의 `step()` 호출 수 | `L-ALT` **500**, 그 외 **0** |
| `closed_form_refits` | `fit_amplitudes_phases` 가 **모델에 설치된** 횟수 | `L-ALT` **10** · 그 외 **1**(초기 `ls_fit_wphi`) |
| `ls_ridge_effective` | 실제로 넘어간 `lambda_reg` | `L-RIDGE0` **0.0** · 그 외 **0.5** |
| `merge_frac_effective` | 실제로 넘어간 `learned_merge_frac` | `L-MERGE` **0.5** · 그 외 **None** |
| `n_atoms_pre_merge` / `n_atoms_post_merge` | 병합 전후 원자 수 | 비-MERGE 팔에서 **동일** |
| `init_loss` / `final_init_loss` | Phase-1 하강 **직전 / 직후**의 `_compute_loss` 값 | `L-ITER0` 에서 **정확히 동일** |
| `init_method_resolved` | 라우터가 실제로 고른 값 | Family A **100% `learned`** |

### 7.2 게이트와 실패 분기 (동결)

| ID | 내용 | 문턱 | 실패 시 |
|---|---|---|---|
| **G-4** ★ | Family A 개입: 위 표의 기대값이 **전 행**에서 성립 · `init_method_resolved=='learned'` 전 행 | **100%** | **중단** — 판정 인쇄 없음 |
| **G-4b** ★ | Family C 개입률: `init_method_resolved=='learned'` 행 비율 · **완전개입 에피소드** 비율 | **행 ≥85%** **그리고** **에피소드 ≥75%** | **S-4 → `UNDECIDED-NO-INTERVENTION`** (다른 판정은 진행) |
| **G-6** ★ | `L-ITER0` 자기시험: `final_init_loss == init_loss` (비트) | 전 행 | 중단 |
| **G-7** | `merge_frac=None` 팔에서 `n_atoms_pre_merge == n_atoms_post_merge` | 전 행 | 중단 |

⚠★**정직 고지 (pilot-informed)**: `G-4b` 의 문턱 **85% / 75%** 는 **W3 원자료의 실측값
(행 90.6% · 에피소드 81.3%) 을 본 뒤에** 정했다. ⟹ **이 게이트는 blind 가 아니다.**
그럼에도 문턱을 이 값으로 두는 이유는 두 가지다: ① 실측보다 **낮게** 두어 게이트가
"이미 아는 답을 통과시키는 장치"가 되지 않게 하되, ② 실측 근처에 두어 **라우팅이 무너지면 반드시 발화**하게 한다.
⛔**결과를 본 뒤 이 문턱을 조정하지 않는다.**

### 7.3 나머지 게이트

| ID | 내용 | 실패 시 |
|---|---|---|
| **G-1** | 표본 파일 존재·이름 파싱·층 배정 일치 (혼잡 64 · 희소 32) | 중단 |
| **G-2** | DCC 스키마 검사 (§9) — 전 행 | `SCHEMA-FAIL`, 판정 인쇄 없음 |
| **G-3** | λ 조작검사: `effective_config['lambda_l1']`(★**객체 되읽기**, 의도 dict 아님) == 1e-4, 위반 0 | 중단 |
| **G-5** | 재현 프로브: `L-SHIP-DUP` 이 `L-SHIP` 과 **\|Δ\|=0** (혼잡 4ep × 3시드 = 12/12) | 중단 |
| **G-5b** ★ | **서브클래스 무해성**: `W4Updater(mode='joint')` 가 **stock `SequentialUpdater`** 와 혼잡 4ep × 3시드에서 **\|Δ\|=0** | 중단 |
| **G-6b** | 팔별 시드 퍼짐 인쇄; 퍼짐 0 인 팔은 `deterministic=True` 표시 | 표시만 |
| **G-8** | `code_md5` 전 행 **단일 digest** · `git_head` 단일 · `git_dirty_n` 기록 | 중단 |
| **G-9** | 시드 원장 충돌 0 · 신규 `.h5` **0** · 소모 시드 == {1595,1596,1597,1598} | 중단 |
| **G-10** | 에러 행 0 · **비유한(NaN/Inf) 값 0** | 중단 |
| **G-11** | `ZERO` 가 팔·시드에 무관하게 동일(퍼짐 0) | 중단 |
| **G-12** ★ | **팔별 셸 필드 존재 검사** — 모든 팔(`ZERO`·`ORA-K-CVX` 포함)이 `obs:` 9키를 **전부** 기록 | 중단 |
| **G-13** | `ORA-K-CVX` 자기검증: `cvx_rank == 2·n_gt`, `cvx_window_resid_rms` 가 진실의 창 잔차 이하 | 표시만 |

★**G-5b 가 이 배치에서 새로 필요한 이유**: `L-ITER0`·`L-ALT`·`L-RIDGE0` 와 `init_loss` 계측 때문에
러너가 `SequentialUpdater` 를 **서브클래싱**한다(`wavesplat/**` 무수정 원칙 유지). 서브클래스 자체가
교락이 되지 않음을 **비트 동일성으로** 증명해야 한다. W3 의 `SHIP-DUP`/`G-5b` 정신을 승계한다.

**구현 주의 (런 전 확인 사항)**: `learned_init_iters=0` 이면 `loss_history` 가 **빈 리스트**가 된다
(`sequential_update.py:645`). 러너는 `loss_history[-1]` 을 **가드 없이 읽지 않는다**.

---

## §8. 판정표 (사전 동결 — 결과를 보고 이름을 고르지 않는다. **코드가 계산한다.**)

### 8.1 1차 — P-1 단독

| P-1 결과 | 이름 | 뜻 |
|---|---|---|
| `L-ITER0` **승** ≥41/64 | **`OPTIMIZER-HARMS`** | 배포 Phase-1 하강이 **순손해**. 처방 = **빼는 것**. |
| 널 24–40/64 | **`OPTIMIZER-INERT`** | 500스텝이 **아무것도 하지 않는다**. ⟹ 벽은 하강이 아니라 **제안(k)** 에 있다. |
| `L-ITER0` **패** ≤23/64 | **`OPTIMIZER-EARNS`** | 하강이 값을 번다 ⟹ ★**"벽은 최적화 실패다" 척추가 배포 경로에서 미지지**. |

### 8.2 2차 — P-1 × S-1 (`L-ALT`)

| P-1 | S-1 | 이름 | 뜻 |
|---|---|---|---|
| HARMS | `L-ALT` 승(Holm) | **`SOLVER-REMEDY-FOUND`** | 처방이 존재하고 방향도 맞다 |
| HARMS | 그 외 | **`DELETE-NOT-REPLACE`** | 고치는 법은 "바꾸는 것"이 아니라 **"빼는 것"** |
| INERT | `L-ALT` 승(Holm) | **`REPLACE-NOT-DELETE`** | 하강 자체는 중립, 대체가 번다 |
| INERT | 그 외 | **`SOLVER-AXIS-CLOSED`** | 옵티마이저 축 종결 ⟹ 다음 축은 **제안(k)** |
| EARNS | 무관 | **`OPTIMIZER-EARNS`** | (2차 없음 — 1차가 결정적) |

### 8.3 ★판정 우선순위 **전순서** (S17 에서 우선순위 부재가 BLOCKER 였다)

분석기는 **정확히 이 순서로** 평가하고, 앞 단계가 종료 사유를 내면 **뒤 단계를 인쇄하지 않는다**:

```
1.  G-1, G-2                       → 실패: SCHEMA-FAIL / 중단 (아무것도 인쇄하지 않음)
2.  G-3, G-5, G-5b, G-8, G-9, G-10 → 실패: 중단 (아무것도 인쇄하지 않음)
3.  G-4, G-6, G-7, G-11, G-12      → 실패: 중단 (아무것도 인쇄하지 않음)
4.  [Stage 1 만] G0-b              → 미달: 판정 NO-SOLVER-SIGNAL, 종료
5.  IUT: L-SHIP < ZERO ≥41/64      → 미달: 판정 FAIL, 종료
6.  퇴화 스캔(§6) → 전 팔에 DEGENERATE 꼬리표 부착 (종료하지 않음)
7.  P-1                            → §8.1 이름 확정
8.  G-4b                           → 미달: S-4 를 UNDECIDED-NO-INTERVENTION 으로 고정
9.  S-1..S-4 Holm-4 step-down      → §8.2 이름 확정
10. 셸 분해(obs: 9키) · 크기 3규약 · 코퍼스 32/32 분해 · 희소 기술통계 인쇄
11. ORA-K-CVX 상한표 (⛔§10-1 라벨 의무)
```

**사망조건 요약**

| # | 조건 | 결과 |
|---|---|---|
| D-1 | §7 하드 게이트 중 하나라도 실패 | **판정 인쇄 없이 종료** |
| D-2 | Stage 1 `G0-b` 미달 | **`NO-SOLVER-SIGNAL`** (Stage 2 미실행) |
| D-3 | IUT `L-SHIP < ZERO` < 41/64 | **`FAIL`** |
| D-4 | `G-4b` 미달 | S-4 만 `UNDECIDED-NO-INTERVENTION` |
| D-5 | S-4 완전개입 부분집합 < 48 에피소드 | S-4 만 `UNDECIDED (n)` |
| D-6 | 동점(부동소수 정확 일치) 발생 | **처치의 패로 세고 `n` 에 남긴다**(W3 `U-1` 승계, 보수적). `ties` **반드시 인쇄** |

---

## §9. 지표 · 집계 · 필수 기록 필드 (DCC)

- **주 지표** = `avg_nRMSE` = `run_estimation` 의 `frame_results` 전 프레임 `nRMSE` 산술평균
  (`get_tracking_frames(tp_factor=2.0)` ⟹ 이미 `t > 2·Tp` 로 잘려 있다).
  **모든 팔**(`ZERO`·`ORA-K-CVX` 포함)이 **같은 프레임 집합 · 같은 `compute_all_metrics`** 로 채점된다.
- **집계 순서(동결)**: ① 팔×에피소드에서 시드 3개의 **중앙값** → ② 에피소드 단위 부호검정.
  ⛔평균-후-변환 금지 · ⛔프레임 단위 풀링 금지 · **결측은 리스트와이즈**(그 에피소드를 모든 팔에서 제외).
- ★**크기 3규약 의무 병기** (셋 다 안 쓰면 크기 인용 금지):
  **A** `median_ep(x_treat − x_ctrl)` · **B** `median_ep((x_treat − x_ctrl)/x_ctrl)` [%] ·
  **C** `(mean_ep x_treat − mean_ep x_ctrl)/mean_ep x_ctrl` [%].
- **셸 분해 의무**: 전역 승리는 반드시 셸로 분해해 병기한다. **키에 `obs:` 접두어를 붙인다** —
  `obs:0_5, obs:5_10, obs:10_20, obs:20_30, obs:30_50, obs:50_75, obs:75_100, obs:100_150, obs:150_inf`
  (관측점까지의 최소거리 기준). ⛔`ship:`(배-거리 고리)과 **혼동 금지** — 같은 표본에서 1.8–4.1배 갈린다.
  **어느 쪽을 쟀는지 접두어까지 적는다**(S13 이 여기서 사전등록을 이탈했다).

**필수 기록 필드 — 한 팔이라도 한 행에서 빠지면 `SCHEMA-FAIL` 로 종료(판정 인쇄 없음)**

**공통** (`REQUIRED_COMMON`):
`batch, stage, arm, episode, corpus, stratum, bf, ncomp, ep_seed, est_seed, n_gt,
avg_nRMSE, zero_nRMSE, n_frames_scored, shell_rmse(obs: 9키), zero_shell(obs: 9키),
shell_frac(obs: 9키), lambda_l1_intent, effective_config, code_md5, code_files_n,
git_head, git_dirty_n, wall_s, deterministic`

**추정기 팔 추가** (`REQUIRED_EST`):
`init_method_intent, init_method_resolved, lambda_l1_effective, adam_steps_phase1,
konly_adam_steps, closed_form_refits, ls_ridge_effective, merge_frac_effective,
n_atoms_pre_merge, n_atoms_post_merge, init_loss, final_init_loss, N_final,
decorr_mean, shrink_mean, obs_resid_rms_mean, elapsed_ms_mean`

**`ORA-K-CVX` 추가**: `cvx_n_obs, cvx_cond, cvx_rank, cvx_window_resid_rms, cvx_omega_source`

⚠★**`effective_config` 는 입력 dict 의 복사가 아니라 `make_updater()` 가 만든 객체에서 되읽는다**
(S13b 가 의도표 방식으로는 보이지 않던 "앙상블이 런의 14%에서만 작동" 을 이 방법으로 잡았다).
⚠★**`git_head` 만으로는 재현이 고정되지 않는다** — **임포트 경로 파일의 md5**(`code_md5`)를
동반 기록한다(S13·S13b 둘 다 배치 중 다른 세션이 임포트 경로 파일을 수정했다).

---

## §10. ⛔ 이 배치가 주장할 수 없는 것 (결과 전에 적는다)

1. ⛔★**`ORA-K-CVX` 대비 개선율을 "순수 솔버 이득"으로 인용 금지** (W3 `B-6b` 승계).
   그 격차는 **4요인 묶음**이다: **{k 동결 vs k 학습} × {닫힌형 vs Adam} × {무추적 vs 추적} × {무가중 vs confidence 가중}**.
   허용 문장 = ✅*"동일 관측·동일 채점기 위에서, k 를 진실에 고정한 무벌점 닫힌형 해가 우리 팔보다 [ratio]배
   정확하다 — 4요인 묶음이다"*. ⚠**모델계열이 진실을 정확히 포함하는 합성 우주의 값이므로 실측 일반화 0.**
2. ⛔**"VARPRO 를 구현/검증했다" 금지** — `L-ALT` 는 **교대 분리최소제곱**이고 **3요인 묶음**이다(§2.4).
3. ⛔**"post-lasso 축을 열었다" 금지** — λ 축은 W3 에서 **완전 종결**됐다. `L-RIDGE0` 은 **다른 블록(초기 닫힌형)
   의 L2 shrinkage** 이고, W3 의 `lambda_l1` 결과가 이 노브를 **덮지 않는다**는 사실만 주장할 수 있다.
4. ⛔**전역 nRMSE 헤드라인 금지** (S16 `S-1a` 실패 승계) — **판정은 혼잡 층 한정**.
5. ⛔**희소에 대해 판정하지 않는다** — 기술통계뿐이고, **부재증명이 아니다**.
6. ⛔**실측 일반화 0** — 합성 코퍼스 · 선형 GT(HOS 아님) · `direction_deg=0` · 단일 체크포인트.
   ⚠**저장된 `gt_wave_field` 재구성 오차 `max|Δ|=0.000e+00`** ⟹ 모델계열이 진실을 정확히 포함한다.
7. ⛔**지연(ms) 인용 금지** — 이 상자의 시계는 별도 재측정 대상.
8. ⛔**"비볼록성" 이라는 이름을 사지 않는다** (W3 §9.3 승계) — 다중 재시작 팔이 없다.
   안전한 이름은 **"위상 하위공간의 최적화 실패"**.
9. ⛔**배포 권고를 이 배치가 하지 않는다.** `learned_init_iters=0` 을 출하 기본값으로 제안하려면
   ① P-1 `OPTIMIZER-HARMS` ② 퇴화 필요조건 통과 ③ 셸 분해에서 **근거리(`obs:0_5`·`obs:5_10`) 거래 없음**
   ④ S-4(출하 라우터 팔) 통과 — **네 조건 전부** + **별도 확증 배치**가 필요하다(S7·S8 전례).
10. ⛔**기준선·기여 ⑥ 에 대해 아무 말도 하지 않는다** — S17 이 그 축을 `PREREG §8` 정지 규칙으로 종결했다.
11. ⛔**`SUPPORT-TRADEOFF` 를 "우리 방법의 이득은 먼바다에 있다" 로 읽지 않는다.**
    이 배치의 대비는 **전부 WaveSplat 내부 노브**이므로 허용된 읽기는
    ✅*"우리 노브를 돌리면 근거리를 팔아 먼바다를 산다"* 뿐이다. **횟수(“N번째 재현”)를 세지 않는다.**
12. ⛔**W3 의 음성을 이 배치 결과로 뒤집었다고 쓰지 않는다** — 서로 다른 코드 지점이다(§1.2).

---

## §11. 실행

```bash
cd /home/vexit/wavesplat/wavesplat

# --- Stage 1 (256런, ≈25분) ---
WORKERS=3 python3 tmp/W4_SOLVER_2026-08-19/run_w4.py stage1 \
  2>&1 | tee tmp/W4_SOLVER_2026-08-19/run_stage1.log
python3 tmp/W4_SOLVER_2026-08-19/analyze_w4.py stage1 \
  | tee tmp/W4_SOLVER_2026-08-19/VERDICT_stage1.log
# → NO-SOLVER-SIGNAL 이면 여기서 종료한다.

# --- Stage 2 (조건부, 1,740런, ≈3.0시간) ---
WORKERS=3 python3 tmp/W4_SOLVER_2026-08-19/run_w4.py stage2 \
  2>&1 | tee tmp/W4_SOLVER_2026-08-19/run_stage2.log
python3 tmp/W4_SOLVER_2026-08-19/analyze_w4.py stage2 \
  | tee tmp/W4_SOLVER_2026-08-19/VERDICT_stage2.log
```

- `results_stage1.jsonl` · `results_stage2.jsonl` 로 **분리**한다. 분석기는 지정된 파일만 읽는다.
- 스모크는 `smoke_results.jsonl` 로 **분리**되며 분석기가 그 경로를 열지 않는다
  (`assert RES.endswith(('results_stage1.jsonl','results_stage2.jsonl'))`).
- 스모크 스크립트는 **문턱을 덮어쓸 수 없다**(`assert THRESH == 동결값`).
- append + 재개(완료 `job_id` 스킵). ⛔두 단계를 동시에 돌리지 않는다(GPU 경합 + 같은 jsonl append).
- ⛔`CUDA_VISIBLE_DEVICES=''` 로 CPU 강제하지 않는다(학습 제안기의 수치가 바뀐다).
- **선택적 가속 (기본 OFF)**: `learned_atoms` 캐시로 GPU 확산 제안을 재사용하면 워커를 8로 올릴 수 있다.
  ⚠**쓰려면 게이트 필수** — 캐시 팔이 비캐시 `L-SHIP` 과 혼잡 4ep × 3시드에서 **\|Δ\|=0** 이어야 한다.
  통과하지 못하면 **캐시를 쓰지 않는다**.

---

## §12. 이 사전등록이 스스로에게 거는 제약

- 이 파일은 동결 후 **수정하지 않는다.** 변경은 `AMENDMENT_n.md` 로만 하고, **해당 단계의 판정 행이 0개일 때만**
  허용하며, **`results_*.jsonl` 을 몇 행 읽은 시점인지 명시**한다.
- 잠정 판독은 `provisional/` 로 격리하고 분석기는 그 경로를 열지 않는다.
- 분석기는 **자기 md5 를 판정 로그 첫 블록에 인쇄**한다(판정 경로가 동결 밖에 있지 않도록).
- **판정명은 코드가 계산한다**(`VERDICT_TABLE`). 사람이 사후에 이름을 고르지 않는다.
- **요약은 사전등록보다 넓게 나가지 않는다** — S13 의 PREREG 는 정확했는데 요약이 부풀었던 전례.
- **정지 규칙**: 이 배치 이후 **옵티마이저 축을 더 연장하지 않는다.** `NO-SOLVER-SIGNAL` 또는
  `SOLVER-AXIS-CLOSED` 가 나오면 그것은 "더 돌자"가 아니라 **"솔버 축은 이 코퍼스에서 종결"** 로 기록하고,
  다음 축은 §8.1 이 지목하는 **제안(k) 품질**이 된다.
- **이중 검증 기본**: 판정은 `analyze_w4.py` 와 **독립 재도출 스크립트** 양쪽에서 일치해야 인쇄한다
  (S17 선례).

---

## §13. 설계 중 실측한 것 (런 0회 · 전부 재현 가능)

| # | 사실 | 출처 |
|---|---|---|
| 1 | 배포 혼잡 경로가 **닫힌형 LS 설치 후 500스텝 joint Adam** 을 돈다 | `sequential_update.py:610-676`, `learned_init.py:386` |
| 2 | `learned_init_iters` 는 **순수 config 노브**(패키지 수정 불필요) | `sequential_update.py:643` |
| 3 | `ls_fit_wphi` 의 ridge 는 **0.5 하드코딩**(config 미노출) | `learned_init.py:318`, `tnse.py:176` |
| 4 | `merge_atoms` 는 **구현·GT-free·기본 no-op·`configs/`·`run.py` 미노출** | `learned_init.py:190,367` |
| 5 | `SHIP` 혼잡 라우팅 = **learned 87/96 (90.6%)**, 에피소드 **26/32 전 시드 learned · 5/32 뒤집힘** | W3 `results.jsonl` 재집계 |
| 6 | `SHIP` 혼잡 **중앙 21.1 s/런** (비용 산정의 근거) | W3 `results.jsonl` 재집계 |
| 7 | 혼잡 코퍼스 가용량 = indep 16 + s7 32 + s8 32 + s9 32 = **112** | `ls data_*_crowded_*/` |
| 8 | W3 혼잡 중앙값: `ORA-K-CVX` **0.00139** · `ORA-K-NOREG` 0.13797 · `ORA-K@L4` 0.14007 · `SHIP` 0.15301 · `ZERO` ≈0.2487 | `verdict_w3.json` |
