# C2 — `L-VP` (VARPRO) 팔 구현

**산출물**: `tmp/W4_SOLVER_2026-08-19/varpro.py` (747줄, md5 `5e920c91481af582475547021aa4a747`)
**추정기 에피소드 런 0회.** 이 문서의 모든 수치는 **닫힌형 항등식 · 합성 신호 · 합성 크기 벤치**에서 나왔다.
`wavesplat/**` **무수정**(`git status --porcelain wavesplat/wavesplat/` = 수정 0건; 유일한 untracked
`tests/test_shipping_defaults.py` 는 08-13 배치 산물로 이 작업과 무관).

---

## 0. 결론 먼저

| 항목 | 결정 | 실측/근거 |
|---|---|---|
| **자기시험 (PREREG G-7)** | **PASS (8/8)**, `python3 varpro.py --selftest` **exit 0** | 재구성 잔차 **max\|Δ\| = 3.331e-15 ≤ 1e-12** |
| **기울기** | **torch autograd**, float64, 정규방정식+ridge 를 통과 ⟹ **완전 Golub–Pereyra**(Kaufman 근사 **아님**) | 중앙 유한차분 대비 상대오차 **1.508e-11** (T4) |
| **가중** | **가중 LS** (행을 `√c_i` 로 스케일), `weighting='confidence'` 기본 | 목적함수가 배포 데이터항과 **rel 9.457e-15 로 일치**(T6) |
| **스텝** | 인자, 기본 **400**, **조기종료 없음** ⟹ `vp_steps` = 실제 실행 수 | G-3 이 요구한 상수성 확보 |
| **위상 규약** | `φ = atan2(−b, a)` (`model/tnse.py:216`) | 반대 규약(`sequential_update.py:1335`)은 **T3 에서 잔차 2.146e+00 으로 실패**(시험이 공허하지 않다는 증거) |
| **비용** | 400스텝 **+1.95–2.45 s/런**(8스레드) · **+4.1–6.2 s**(1스레드) | 합성 M=8080, N=20/30/45 |
| **결정성** | 같은 입력 2회 재실행 **max\|Δ\| = 0.0** (비트 동일) | G-5 복제 프로브와 양립 |

---

## 1. 수학

배포 전방모델 (`wavesplat/model/spectral_gaussian.py:118-127`):

```
z(x,y,t) = Σ_j w_j cos( kx_j x + ky_j y − ω_j t + φ_j ) ,   ω_j = √(g|k_j| tanh(|k_j| d))
```

θ_j ≡ kx_j x + ky_j y − ω_j t 로 두면

```
w cos(θ+φ) = (w cos φ)·cos θ + (−w sin φ)·sin θ  ≡  a·cos θ + b·sin θ
a = w cos φ ,  b = −w sin φ        w = √(a²+b²) ,  φ = atan2(−b, a)
```

⟹ **`(a,b)` 에 정확히 선형**. 선형성을 깰 수 있는 세 경로(stokes2 · neural corrector · free-ω)는
배포에서 전부 꺼져 있다(`W1_path.md §2.2`; `_build_model` `sequential_update.py:239-241`).

**VARPRO 축소 목적함수** — 자유변수는 `K=(kx,ky) ∈ R^{2N}` **뿐**:

```
Φ(K)  = [cos Θ | sin Θ]                        (M × 2N)
c(K)  = (Φ_wᵀΦ_w + εI)⁻¹ Φ_wᵀ z_w              (닫힌형, S = diag(√c_i), Φ_w = SΦ)
f(K)  = ‖Φ_w c(K) − z_w‖² / M  =  Σ_i c_i (ẑ_i − z_i)² / n_obs
```

마지막 등식이 **배포 데이터항 그 자체**다(`sequential_update.py:329-342` — 가중제곱합을
**비가중** `n_obs` 로 나눈다). 구현은 이 규약(가중합/비가중 M)을 문자 그대로 따랐고, T6 이
**shipped `_compute_loss(reg=False)` 를 직접 호출해** 비교했다.

⚠**ω 는 재유도하지 않고 패키지 함수를 호출한다** — `√(kx²+ky²+1e-12)` (`spectral_gaussian.py:100`)
→ `dispersion_omega`(`dispersion.py:33`, sqrt 안에 자체 `+1e-12`). 두 가드가 모두 재현된다.
(참고: `recon/probe_varpro.py:_fc` 는 `+1e-12` 없이 ω 를 재계산했다 — 무시할 수준이나 **다르다**.)

⚠**λ_l1·λ_smooth 는 축소 목적함수에 들어갈 수 없다** — `Σ|w| = Σ√(a²+b²)` 는 group-lasso 라
닫힌형 소거가 불가능하다. ⟹ `L-VP` 의 **재적합 단계는 구조적으로 λ=0** 이다.
W3 가 λ 축을 이미 닫았으므로(`ORA-K-NOREG` 30/32 이지만 `S-1` 18/32 널 = 배포 λ=1e-4 ≈ λ=0)
**VARPRO 이득이 나오더라도 λ 로 설명될 수 없다** — 사전 논증이고, 여전히 결과가 말해야 한다.

---

## 2. 기울기 방식 — **autograd = 완전 Golub–Pereyra**

`torch.linalg.solve(Φ_wᵀΦ_w + εI, Φ_wᵀ z_w)` 를 통과하는 **torch autograd**(float64).
자동미분은 `K → Φ → c → f` **합성 전체**를 미분하므로 `∂c/∂K` 경로가 실려 있다.
**그 항이 정확히 Kaufman 근사가 버리는 항**이다 ⟹ 구현은 **Golub–Pereyra(1973) 완전형**이며
**Kaufman 근사가 아니다**. 해석적 야코비안도 유한차분도 **팔에서는 쓰지 않는다**.

검증(T4): 같은 축소 목적함수에서 autograd 기울기와 **중앙 유한차분**(h=1e-7, 2N=14 성분)을 비교 —
`max|g_auto − g_fd| / max|g_fd| = 1.508e-11`.
⚠**진실점에서 검사하면 안 된다** — 잔차가 정확히 0 이라 기울기가 해석적으로 0 이고 비교가 퇴화한다
(첫 실행에서 실제로 이 함정에 빠져 `rel = 1.000e+00` 을 봤다). 최종 T4 는 **섭동된 k + 잡음 z** 에서 잰다.

`torch.linalg.lstsq` 는 autograd 지원이 제한적이라 **쓰지 않았다**(정규방정식 + ridge 1e-8).
옵티마이저는 Adam(lr **2e-3**), 400스텝 — 사전노출 프로브(`recon/probe_varpro.py`)에서 **무수정 상속**.
⛔W2 §3.5 가 400→4000 창 잔차 불변을 실측했으므로 **스텝수를 축으로 삼지 않는다.**

---

## 3. 가중 결정 — **가중(√c_i) LS**, 근거와 위험

**택한 것**: `weighting='confidence'` (기본). 행을 `√c_i` 로 스케일 ⟹ 목적함수가
`Σ c_i r_i² / n_obs`.

**근거 3개**
1. **비교 대상이 배포 팔이다.** `L-SHIP` 의 Phase-1·Phase-2 데이터항이 `Σ c_i r_i²`
   (`sequential_update.py:331-334`, `use_confidence=True` 기본 `run.py:62`). 다른 데이터항으로
   재적합하면 `L-VP` 가 **(매개화 × 가중) 2요인**이 된다. PREREG §2 는 `L-VP` 를
   **"단일(하강 매개화)"** 로 선언했다 ⟹ 가중을 배포와 같게 두는 것이 **단일요인을 지키는 쪽**이다.
2. `W1_path.md §3.1-2 · §6-6` 이 같은 처방을 명시했다.
3. 목적함수 동일성이 **측정으로 확인**된다(T6, rel 9.457e-15) — 주장이 아니라 검산이다.

**⚠위험 (사전등록에 반드시 적어야 함)**
- **사전노출 신호(−47.3%)는 비가중에서 나왔다** (`probe_varpro.py:_fc` 는 `Phi`/`Z` 를 그대로 쓴다).
  W3 상한 `ORA-K-CVX` 도 `weighting:'none'`(`run_w3.py:479`). ⟹ 가중을 택하면 확증이
  **프로브와 정확히 같은 양을 재지 않는다.**
- 합성 대조(균등 난수 신뢰도, 잔차와 무상관)에서는 차이가 거의 없었다:
  가중 win_rms 0.217351→0.002770 vs 비가중 0.217068→0.002763, `dk_med` 둘 다 0.00348.
  ⚠**실측 신뢰도는 입사각과 상관**이 있으므로 이 합성 대조는 **상한이 아니라 하한**이다.
- ⟹ **완화책**: 축을 코드 인자로 노출(`weighting='none'` 이 프로브를 정확히 재현)하고,
  모든 결과행에 **`vp_weighting` 을 기록**한다. 어느 쪽을 본런에 쓸지는 **PREREG 소유자가 동결**할 것.

---

## 4. 삽입 지점

**`SequentialUpdater._initialize_learned`(`sequential_update.py:617-673`) 를 클래스 메서드로 감싼다**
(원본 호출 → 반환된 모델 위에 VARPRO).

```python
import varpro
varpro.install(steps=400, weighting='confidence')   # ★워커 프로세스 안에서
try:
    ... run episode ...
    row.update(varpro.stats())                      # vp_entries / vp_steps / vp_k_shift ...
finally:
    varpro.uninstall(); varpro.reset_stats()
```

**왜 "교체"가 아니라 "덧붙이기"인가** `[근거]`
- PREREG §2 의 문자열이 **"Phase-1 *출력*에 VARPRO 400스텝"** 이다.
- 사전노출 신호(PROBE-B −47.3%)는 **추정기가 끝낸 원자집합**에서 출발해 측정됐다
  (`probe_varpro.py` 가 `S4_phase_oracle` 의 **저장된 최종 스펙트럼**을 읽는다). ⟹ Phase-1 **뒤**에
  얹는 것이 그 프로브의 **충실한 온라인 이식**이다.
- G-2 가 `adam_steps_phase1` 을 **`L-ITER0`·`L-SHIP` 에만** 구속하고 `L-VP` 에는 값을 지정하지 않았다
  ⟹ `L-VP` 는 배포 500스텝을 **유지**한다는 읽기와 정합.
- 그러면 `S-2`(`L-VP < L-ITER0`)가 **"하강 제거"와 "VARPRO"를 분리**하는 정확한 대조가 된다.

⚠**모호성 신고 — PREREG 소유자가 결정할 것.** §2 를 "500 Adam 을 VARPRO 로 **대체**" 로 읽을 수도
있다("단일(하강 매개화)" 표현이 그쪽을 시사한다). **코드 변경은 필요 없다** — 그 읽기를 택하면
팔 config 에 `learned_init_iters: 0` 을 넣으면 되고, 그 경우 `adam_steps_phase1` 이 0 으로 기록된다.
**둘 중 무엇을 동결하든 `varpro.py` 는 그대로 쓴다.**

**왜 이 자리가 안전한가** `[정적 확인]`
- 호출 지점은 저장소 전체에 **1곳뿐**: `sequential_update.py:517`
  `return self._initialize_learned(frames, x_range, y_range, verbose)` — **인스턴스 조회**라
  클래스 패치가 도달한다. `MedianEnsembleUpdater` 도 같은 클래스를 쓰므로 멤버 전부 덮인다
  (`:1493`, `:1518`). (패치 시그니처를 원본과 **동일한 이름**으로 맞춰 키워드 호출도 안전.)
- 반환 시점에는 `_adaptive_prune_grow`(`:660`)가 이미 끝나 `nn.Parameter` **교체**
  (`spectral_gaussian.py:191-194`, `:210-213`)가 완료됐다 ⟹ **제자리 `copy_` 안전**.
- `_build_tracking_optimizer`(`:662`)가 잡아둔 파라미터 참조가 그대로 유효하고, 그 Adam 상태는
  아직 비어 있다 ⟹ **옵티마이저 재구축 불필요**.
- ⚠`v2_konly` 로 라우트된 행은 **개입 0** 이고 `vp_steps=0` 으로 기록된다. G-1 이
  `init_method='learned'` 를 고정하므로 그 집합은 **구성적으로 공집합**이며, 카운터가 그것을
  **가정이 아니라 증명**한다.

**clamp**: 설치 후 `model.clamp_steepness(self.max_steepness)` 를 **호출한다**(기본 `clamp=True`).
배포는 Adam 매 스텝 뒤에 이 사영을 건다(`:650`). 안 걸면 배포 코드가 만들지 않는 상태를
Phase 2 에 넘기고, 어차피 `:979` 가 첫 추적 프레임에서 건다. 실제로 잘린 성분 수는
`vp_clamp_hits` 로 기록한다. ⟹ **이 한 줄은 사전등록 대상**(W1 §3.1-5 지적).

⛔**`amplitude_refit`(`sequential_update.py:1259-1355`)은 한 줄도 재사용하지 않았다.**
⛔**`_ls_w_update`(`:1215-1257`, 호출자 0개)도 재사용하지 않았다.**

---

## 5. 자기시험 결과 — **실제로 실행됨**

```
$ cd /home/vexit/wavesplat/wavesplat/tmp/W4_SOLVER_2026-08-19 && python3 varpro.py --selftest
```

| # | 검사 | 결과 | 판정 |
|---|---|---|---|
| T1 | 설계행렬 + `(w,φ)→(a,b)` 가 **shipped `SpectralGaussianModel.forward`** 를 재현 | `max|Δ| = 1.443e-15` | PASS |
| **T2** | **★G-7 게이트** — 알려진 `(A,φ)` → `(a,b)` 풀이 → `(A,φ)` 복원 → **재렌더** | **`max|Δ| = 3.331e-15 ≤ 1e-12`** | **PASS** |
| T2b | 파라미터 자체 복원 | `dw = 3.331e-16`, `dφ = 1.332e-15` | PASS |
| **T3** | **음성 대조** — `atan2(b,a)`(`sequential_update.py:1335`)를 쓰면 깨져야 한다 | `max|Δ| = 2.146e+00` | PASS (시험이 공허하지 않음) |
| T4 | autograd 기울기 vs 중앙 유한차분 | `rel = 1.508e-11` | PASS |
| T5 | 200스텝 하강이 목적함수↓ **그리고** k 를 진실 쪽으로 | `obj 3.4524e-02 → 2.2375e-06`, `|Δk| 0.00414 → 0.00001` | PASS |
| T6 | 우리 목적함수 == **shipped `SU._compute_loss(reg=False)`** | `rel = 9.457e-15` | PASS |
| T7 | `install()`/`uninstall()` 가 클래스 속성을 **원본 객체로** 복원 | 동일성 검사 | PASS |
| T8 | **진짜 `SequentialUpdater` + 진짜 모델**에 `apply_varpro` 적용 | `steps=50`, `dk=0.00355`, 배포 손실 `7.1439e-02 → 9.4269e-05`, 카운터 전건 비영 | PASS |

**`SELFTEST PASS`, exit code 0.** T1·T2·T6·T8 은 **패키지 코드를 직접 호출**한다(내가 재구현한
전방모델이 아니라 `spectral_gaussian.py`·`sequential_update.py` 자체). 즉 항등식은
**배포 코드에 대해** 성립한다.

★**T3 이 핵심이다** — G-7 이 "≤1e-12 을 통과했다"만 요구하면 **틀린 규약으로도 통과하는 시험**을
짤 수 있다. T3 은 틀린 규약이 **반드시 실패함**을 같은 자리에서 보인다.

**결정성**(별도 실행): 동일 입력 400스텝 2회 → 출력 `(kx,ky,a,b)` **max\|Δ\| = 0.0** (비트 동일).

---

## 6. 카운터 (PREREG §4·§5)

`varpro.stats()` 가 결과행에 그대로 넣을 dict 를 준다:

| 키 | 의미 | 기대값(`L-VP`, learned 라우트) |
|---|---|---|
| `vp_entries` | 패치가 진입한 횟수 | **1** (learned 는 `ensemble_k=1`; v2_konly 앙상블이면 3) |
| **`vp_steps`** | **실제 실행된 Adam 스텝 총합** (플래그 아님) | **400 × `vp_entries`** ← **G-3** |
| `vp_steps_per_entry` | 진입별 스텝 | `[400]` |
| `vp_skipped` | `None` / `bank_route` / `no_model` / `empty_window` | `[None]` |
| `vp_k_shift`, `vp_k_shift_max` | `‖Δk‖` 중앙/최대 | **0 이면 개입 없음** |
| `vp_dw_max`, `vp_dphi_max` | `max|Δw|`, `max|wrap(Δφ)|` | **0 이면 개입 없음** |
| `vp_clamp_hits` | `clamp_steepness` 가 실제로 스케일한 성분 수 | 닫힌형 해가 사영에 잘린 정도 |
| `vp_wall_s` | VARPRO 벽시계 | `D-2` 비용 분해 |
| `vp_obj_before/after` | **가중** 목적함수(배포 정렬) | LS 성질상 반드시 감소 |
| `vp_win_rms_before/after` | **비가중** 창 잔차 RMS | W2 표와 직접 비교 가능 |
| `vp_weighting` | `'confidence'` / `'none'` | ★§3 의 축을 행에 남긴다 |

`varpro.describe()` 는 **모듈 상태 되읽기**(의도 dict 아님, S13b 교훈)로 `installed` ·
패치된 메서드의 `__name__`(`_initialize_learned__L_VP`) · `code_md5` · 유효 인자를 준다
⟹ `effective_config` 와 `code_md5`(G-8)에 그대로 쓸 것.

⛔**대조 팔(`L-SHIP`·`L-ITER0`·`L-MERGE`)에서도 `varpro.stats()` 를 기록**해
`vp_entries == 0` 을 남길 것(= 누출 없음의 증거). 패치를 걸지 않으면 자동으로 그렇게 된다.

⚠**패치는 워커 프로세스 안에서 걸고 `finally` 로 되돌린다**(`W1_path.md §5`) — 부모에서 걸면
`fork` 시점에 따라 팔이 샌다. `install()` 은 이중 설치를 `RuntimeError` 로 막는다.

---

## 7. 비용 — 합성 크기 추정 (에피소드 미실행)

`python3 varpro.py --bench` (M=8080 = W2 가 보고한 배포 초기창 관측 수):

| M | N | 스텝 | 8스레드 | 1스레드 |
|---|---|---|---|---|
| 8080 | 20 | 400 | **1.95 s** | **4.11 s** |
| 8080 | 30 | 400 | **2.40 s** | **6.20 s** |
| 8080 | 45 | 400 | **2.45 s** | — |

⟹ 런당 추가 **≈ 2.0–2.5 s**(8스레드) 또는 **4–6 s**(워커당 1스레드). BLANKSLATE 가 잰
정직한 초기화 비용 **중앙 24.0 s** 대비 **+8~10%**(8스레드) / **+17~26%**(1스레드).
⚠**이 추정은 CPU float64 · 균등 난수 관측 기하**다. 실제 LiDAR 관측은 x 분포가 다르지만
비용은 `M×N` 에만 의존하므로 **크기 차수는 옮겨간다**. W2 의 독립 추정(+2.3 s)과 일치.
⚠병렬 워커 수를 늘리면 스레드당 비용이 위 1스레드 열로 간다 — **`D-2` 벽시계는 팔별로 실측할 것.**

---

## 8. 이 구현이 **주장하지 않는 것**

1. ⛔**VARPRO 는 우리 것이 아니다** — Golub–Pereyra 1973. PREREG §7-1 의 금지를 그대로 승계한다.
   `varpro.py` 는 **측정 도구**이지 기여가 아니다.
2. ⛔**"창 잔차가 내려간다" ≠ "필드가 좋아진다"** — LS 성질상 `vp_obj_after ≤ vp_obj_before` 는
   **정리이지 결과가 아니다**. W2 §3.4 가 창 잔차를 잡음 바닥까지 내려도 필드 ×37 임을 이미 쟀다.
   ⟹ 결과행에 **창 잔차와 필드 nRMSE 를 반드시 같은 행에** 두어야 한다(W2 §6.2-1).
3. ⛔**추적기와의 상호작용은 미측정** — 이 팔은 Phase-1 개입이고, 그 뒤 `tracking_mode='w_phi'` 가
   프레임당 10스텝 × 120–220 프레임으로 `(w,φ)` 를 다시 흔든다(`:970-974`). 프로브는 **추적기 OFF**
   표본에서 잰 값이다 ⟹ **온라인에서 이득이 부분적으로 씻길 수 있다.** W2 §6.2-2 가 제안한
   `tracking_lr=0` 동반 팔은 **PREREG §2 표에 없다** — 채택 여부는 소유자 결정.
4. ⛔**희소 주장 없음**, ⛔**전역 헤드라인 없음** — PREREG §7 을 그대로 승계.

---

## 9. 열람 범위 (= 주장 범위)

**읽은 것**: `PREREG.md`(전문) · `recon/W1_path.md`(전문) · `recon/W2_upper_bound.md`(§0 목차 +
§3.3–3.6 · §4.2–4.3 · §6) · `recon/probe_varpro.py`(전문) ·
`wavesplat/optimizer/sequential_update.py`(`:313-372`, `:400-420`, `:505-522`, `:609-690`,
`:1250-1360`, `_bank` grep) · `wavesplat/model/spectral_gaussian.py`(`:1-140`, `:215-235`) ·
`wavesplat/model/tnse.py`(`:200-240`) · `wavesplat/model/dispersion.py`(`:1-60`) ·
`wavesplat/data_loader.py`(`FrameData`).

**실행한 것**: `varpro.py --selftest`(4회, 최종 8/8 PASS, exit 0) · `varpro.py --bench` ·
결정성/가중 비교 스크립트 1회. **에피소드를 한 건도 열지 않았다** — `varpro.py` 는
`wavesplat.data_loader` 에서 **`FrameData` 만**(`:631`, `:666`) 가져와 합성 프레임을 만들고,
`EpisodeDataLoader` 는 **인스턴스화된 적이 없다**(`grep EpisodeDataLoader varpro.py` = 0건).
HDF5 접근·`wavesplat.run`·`compute_all_metrics` 호출 **0회**.

**하지 않은 것**: `L-SHIP`·`L-ITER0`·`L-MERGE`·`ZERO`·`ORA-K-CVX` 팔 · 러너 하네스 ·
결과 집계 · PREREG 감사 — 전부 다른 에이전트 몫.
