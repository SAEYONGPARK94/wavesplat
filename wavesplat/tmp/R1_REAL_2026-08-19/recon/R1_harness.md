# R1 정찰 — 실측(Bergamasco 스테레오) 하네스 사양

**2026-08-19. 추정기 런 0회.** 코드·기록 독해만.
표기: `MEASURED` = 파일을 열어 확인 · `INFERRED` = 읽은 것에서 도출(런으로 확인 안 됨).
모든 주장에 `절대경로:줄`.

---

## 0-A. ★★★**정정 (배치 중 도착) — 08-19 "평평한 상수장"은 추정기가 아니라 질의격자 버그였다**

배치 진행 중 다른 세션이 `tmp/COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py` 를 수정했다
(감사 R3 K-1). 새 주석 `:115-119` 가 축자로:
> *"the first version sliced the wrong axis … `Xn[:, 0]` and `Yn[0]` each held ONE unique value —
> every one of the 53,261 query points was evaluated at the same coordinate … The resulting flat
> panels were an artifact of this line, not a property of the estimator."*

★**내가 원자료로 독립 확인했다** (`MEASURED`, h5py 로 좌표 배열만 읽음 — 추정기 런 0회):

| 배열 | shape | `arr[:,0]` 고유값 | `arr[0,:]` 고유값 |
|---|---|---|---|
| `_nc4_non_coord_X` | (241, 221) | **241** | **1** |
| `Y` | (241, 221) | **1** | **221** |

구 코드는 `gx = Xn[:,0]`(= `Y[:,0]` ⟹ **고유값 1개**), `gy = Yn[0]`(= `X[0,:]` ⟹ **고유값 1개**)였다.
⟹ **53,261 질의점이 전부 같은 좌표**였고, 그 좌표는 팬 밖이다. **버그 확정.**

⟹ ⛔**"실측에서 배포 추정기가 붕괴했다 / 파도장 대신 상수장이 나왔다 / nRMSE 0.265 가
제로필드 0.250 보다 나쁘다" — 세 문장 모두 인용 금지.** 근거가 소멸했다.
⟹ **이 배치의 전제(`R1` 임무 배경문)가 무효다.** 사양은 여전히 유효하지만
**"왜"가 바뀌었다**: 실측 하네스를 다시 짜야 하는 이유는 "붕괴를 고치려고"가 아니라
**"아직 한 번도 제대로 재본 적이 없어서"** 다.

⚠**그럼에도 아래 §0-B 의 라우팅 사실은 그대로 유효하다** — 버그와 독립이다.
버그를 고쳐도 그 런은 여전히 **출하 경로가 아니라 W4/W5 학습 팔**을 잰다.
★**두 결함이 겹쳐 있었다. 하나를 고쳤다고 다른 하나가 사라지지 않는다.**

---

## 0-B. 08-19 런은 출하 경로가 아니었다 (버그와 독립, 여전히 유효)

`MEASURED` — 08-19 YS17 시도(`/home/vexit/wavesplat/wavesplat/tmp/COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py:94`)는
`arms_config_w5.arm_config('L-SHIP')` 를 썼고, 그 base config 는
`/home/vexit/wavesplat/wavesplat/tmp/W4_SOLVER_2026-08-19/arms_config.py:67` 에서 **`'init_method': 'learned'`** 다.

`/home/vexit/wavesplat/wavesplat/wavesplat/optimizer/sequential_update.py:445` 의 게이트는
`if self.init_method == 'auto':` 이다. ⟹ **`init_method='learned'` 는 그 블록 전체를 건너뛴다.**
W4 의 docstring 이 그 사실을 축자로 적어 놓았다 (`arms_config.py:63-66`):

> *"G-1: the router is REMOVED, not measured. `init_method != 'auto'` means `_initialize_inner`
> skips the whole auto/real-gate block (`sequential_update.py:445`)…"*

⟹ ★★★**08-19 런은 `REAL-DOMAIN GATE` 를 통과한 적이 없다.**
출하 기본값(`init_method: 'auto'`, `run.py:42`)으로 GT-free 입력을 넣었다면
`sequential_update.py:460-497` 이 발화해 **`bank_mmse` 로 라우팅**된다 — 그리고 그
`bank_mmse` 는 `d3_falcon_real_operator.py` 가 쓰는 연구 추정기의 **비트-정확 이식본**이다
(`wavesplat/optimizer/bank_mmse.py:124-125`: *"Verbatim transcription of
scripts/d3_sa_nomp.mmse_bank_estimate"*, 회귀 테스트 `wavesplat/tests/test_bank_mmse_port.py:84`).

⟹ **"배포 추정기가 실측에서 붕괴했다"는 문장은 현재 근거가 없다.**
붕괴한 것은 **W4/W5 연구 팔 `L-SHIP`(= 학습 초기화기, 라우터·실측게이트 제거)** 이다.
⛔이 구분 없이 "배포 추정기"라 부르면 저장소가 이미 4번 물린 *"안 돌아가는 코드를 연구했다"*
부류의 5번째 재발이 된다.

★**그리고 출하 경로는 이미 YS17 에서 돌았고 숫자가 디스크에 있다** (`MEASURED`):
`tmp/bankmmse_port_2026-07-29/verify_deployed_path.py` 가 출하 `make_updater` +
`DEFAULT_CONFIG(init_method='auto')` 로 실측 3창을 돌렸고,
`deployed_path_verify.json` 이 `"real_gate": true, "route": "bank_mmse", "bank_members": 3` 을
기록한다. YS17_w0 nowcast nRMSE **0.7228 (÷σ)** = **0.181 (÷4σ, 08-19 규약)** vs 제로필드 0.250.
⚠**정규화 분모가 4배 다르다** — 자세히는 §5.4-#2. ⛔**그래도 "출하가 이긴다"로 인용 금지**
(창·센서운동·stride 가 다르다). **다음 배치의 1순위 가설**이다.

⚠**동시에 반대 방향 증거도 있다**: 구 fallback `v2_konly` 는 실측 10창에서 nRMSE 0.750–1.099 로
**7/10 이 제로예측기 근처 이상**이고 YS17_w1 은 corr **−0.027** 이다
(`tmp/real_gif_2026-07-27/scan.jsonl`). LJ18g w3 은 bank 경로에서도 **1.1829 = 무기술**이다.
⟹ **"실측에서 무기술 필드가 나온다"는 것 자체는 이 저장소에서 처음이 아니다.**
⚠단 §0-A 이후 **08-19 은 그 사례 목록에 들어가지 않는다**(버그였다).

⚠★**§0-A 이 §5.4-#2 의 대비도 무효화한다** — "출하 0.181 vs 08-19 0.265" 에서
**우변이 버그값**이다. 좌변(출하 bank 경로 0.7228/σ = 0.181/4σ)은 그대로 유효하고
**제로필드 0.250 대비 우위라는 사실도 유효**하지만, **비교 상대가 사라졌다.**
⟹ 그 표는 이제 *"출하 bank 경로가 YS17 에서 제로필드를 이겼다(07-29, n=1창)"* 만 말한다.

---

## 1. `scripts/d3_falcon_real_operator.py` 전수 독해 — 작동시킨 조건 전부

파일: `/home/vexit/wavesplat/wavesplat/scripts/d3_falcon_real_operator.py` (426줄, 전수 독해)

### 1.1 추정기 (⚠**출하 `SequentialUpdater` 가 아니다**)

| 경로 | 줄 | 호출 대상 | 정체 |
|---|---|---|---|
| 기본(`--net` 없음) | `:322-327` | `mmse_bank_estimate` (`scripts/d3_sa_nomp.py:499`) | 연구 bank-MMSE |
| `--net` 지정 | `:318-320` → `:51-61` | `d3_prior_net_eval.net_gamma_mmse` (`scripts/d3_prior_net_eval.py:56`) | 학습 prior γ + **같은** physics MMSE |
| 참조 | — | `wavesplat.optimizer.sequential_update.SequentialUpdater` | **한 번도 임포트되지 않음** |

`MEASURED` — `d3_falcon_real_operator.py` 안에 `wavesplat.` 임포트는 **0건**(`:42-46` 이 전부 `scripts/` 모듈).
⟹ 이 스크립트의 모든 실측 수치는 **연구 bank-MMSE 계열**의 것이고, 출하 `SequentialUpdater` 를
직접 검증한 적은 없다. **단** 그 알고리즘은 `bank_mmse.py` 로 비트-정확 이식돼 있고
출하 실측 게이트가 그리로 라우팅한다 ⟹ **두 팔은 같은 수학**이다(아래 §2 의 세 델타 제외).

### 1.2 `kp_hint` 는 어떻게 계산되는가

```
:226-227   spread, lam_p, pdir_ax, kp = directional_spread(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
:322-327   mmse_bank_estimate(..., kp_hint=kp, ...)
```

`directional_spread` = `/home/vexit/wavesplat/wavesplat/scripts/realdata_analysis.py:95-118`
- 입력 = **단일 프레임 `Zmem[-1]`** (기억창의 마지막 스테레오 프레임)
- `:98-99` 2D FFT → power → `:107-109` `argmax(P)` 의 `|k|` = `kp`
- `:110` `peak_dir = degrees(arctan2(kpy, kpx))` — 실 FFT 라 **±180° 접힘**(`:114` 주석)
- `:112-117` 방향 스프레드는 `0.4kp < K < 2.5kp` 대역에서 원형 표준편차

★★★⚠**결정적 스코프**: `Zmem` 은 `rec.window(...)`(`:222`) = **전체 스테레오 패치**다.
팬이 표집한 관측이 아니다. ⟹ **`kp_hint` 는 시간적으로만 causal 이고 공간적으로는 오라클**이다.
docstring `:14` 의 *"causal in-window characterize"* 는 **시간 causal** 만 뜻한다.
`INFERRED` — 배 위 센서는 이 양을 스스로 만들 수 없다. 배포 서사로 쓰려면
**외부 파수원**(웨이브라이더 · X-band 레이더 · 예보 스펙트럼)을 전제로 사전등록해야 한다.

**같은 오라클 오염이 두 군데 더 있다**:
- `:225` `Hs = 4.0*np.nanstd(Zmem)` = **전체 필드** ⟹ `sigma0 = 0.05*Hs`(`:322`)도 오라클.
  출하 bank 는 **관측에서** 뽑는다: `sequential_update.py:861` `self._bank_Hs_obs = 4.0*np.std(z)`.
- `:230` `resolve_direction(Zmem, ...)` 도 전체 필드(§1.3).

### 1.3 `resolve_direction` (`:108-126`) — 하는 일과 왜 필요한가

```
:111  n = max(1, round(dt_gap / rec.dt))          # dt_gap=1.0 s 기본
:113-114  f0,f1 = lowpass_field(Zmem[-1-n]), lowpass_field(Zmem[-1])   # k_cut = 1.5*kp
:116-124  for sgn in (+1,-1): f0 를 sgn*cp*n*dt 만큼 이류(ndshift) → corr(fs, f1) 최대인 부호 채택
:125-126  pdir = pdir_axis  또는  pdir_axis + 180
```

**왜 필요한가** `MEASURED` — `directional_spread` 는 실수장의 2D FFT 이므로 스펙트럼이
켤레대칭이다. peak 가 `(kx,ky)` 와 `(−kx,−ky)` 에 동일 파워로 서므로 **전파 방향이 180° 모호**하다
(`realdata_analysis.py:114` 가 "fold direction to [−90,90] … to avoid +/− ambiguity" 라 명시).
시간축을 써서 한 프레임을 ±c_p·Δt 만큼 이류시키고 나중 프레임과 상관이 높은 부호를 고른다.

**무엇에 쓰이는가** — ⚠**추정기 입력이 아니다.** `:231-232` 에서
`w = (cos pdir, sin pdir)`, `h = -w` (**head seas**) 로 **배의 침로**를 정하고,
그 `h` 가 팬의 조준 방향(`:247-248`)과 조우 목표점(`:285`)을 만든다.
⟹ 180° 를 틀리면 **head seas 가 following seas 로 뒤집혀** 조우 문제 자체가 달라진다.

`MEASURED` 실측 품질: 저장된 `adv_corr` 는 YS17 0.93–0.94(6/6), LJ18g 0.79–0.96,
AA14 0.75–0.98, AA15a 0.69–0.93, AA15b 0.72–0.90
(`results/direction3/falcon_real_operator_*.json`, 아래 §5.1 표).

### 1.4 팬 기하 · 잡음 · 표집 (동결값)

| 항목 | 값 | 줄 |
|---|---|---|
| FOV | ±60° (`ca >= 0.5`) | `:248` |
| 사거리 `r_fan` | YS17/LJ18g `(20,100) m` · AA×3 `(8,40) m` | `:68-89` |
| 프레임당 유지 화소 `m_scan` | **12** (raydrop 대리 — 균일 랜덤) | `:136-137`, `:253` |
| 시공간 관측 상한 `m_cap` | **600** (`default_rng(0)` 결정적 subsample) | `:138`, `:312-315` |
| 거리 잡음 | `N(0, 0.02 m)` = Falcon 2 cm 1σ | `:139-140`, `:258` |
| 공간 stride | YS17·LJ18g `2` · AA×3 `1` | `:68-89`, `MRecord` `d3_sa_multirecord.py:103-112` |
| 선속 `U` | YS17 2.0 · AA×3 1.2 · LJ18g 2.4 m/s | `:68-89` |
| `lead_max` | YS17·LJ18g 20 s · AA×3 14 s | `:68-89` |
| 기억창 `w_mems` | `[10, 20, 40] s` | `:134` |
| 창 개수 `n_win` | 기본 **6** (docstring 은 4 라 적혀 있다 — 불일치) | `:135` vs `:27` |
| 선속 각도 | 직선·정속 **head seas**(`h = -w`) | `:231-237` |
| 배 초기위치 | 패치 중심에서 `-T_tr/2` 만큼 후퇴 → 창을 가로질러 통과 | `:208-209`, `:235-237` |

⚠**모든 `U` 가 설계 선속 11.3 m/s 보다 훨씬 낮다**(1.2–2.4 m/s).
`memory/reference_agreed_operating_point.md` 가 이 창을 "판정용"이라 규정한 그 범위다.

### 1.5 채점 대상 — ★**지지집합 밖을 채점하지 않는다**

```
:284-288   Pq_enc = 배의 FUTURE 위치 (lead 0..lead_max, 1 s 간격) ← 제어-관련 목표
:290-301   Pq_map = t0 의 팬 영역 내부 유효 화소 최대 400점 ← nowcast 진단
:302       Pq = vstack([Pq_enc, Pq_map])
```

⟹ `MEASURED` — d3 는 **팬이 본 적 있는 영역과 배가 갈 궤적**만 채점한다.
08-19 시도는 `real_estimate.py:14` 가 스스로 적었듯 *"80%+ 를 한 번도 관측하지 않은 전체 패치"*
를 채점했다. ⟹ **두 배치는 같은 양을 재고 있지 않다.**
저장소의 기존 판정과도 충돌한다: E-C *"지지집합 밖 식별 불가"* ·
E-D *"~30 m 안만 이득, 밖 84% 손해"*(`memory/project_ec_result_2026-08-07.md`,
`memory/project_ed_result_2026-08-07.md`).

### 1.6 창 시작점 배치

```
:212-215   lo = nfw + 1 ;  hi = T_all - n_lead - 2 ;  i0s = linspace(lo, hi, n_win)
:221       rng = default_rng(700 + wi)      # 창별 결정적 시드
```

---

## 2. ★★★배포 추정기에 `kp_hint` 를 줄 수 있는가 = **아니오. 경로가 없다.**

### 2.1 전수 grep 결과 (`MEASURED`)

`wavesplat/`·`configs/` 전체에서 `kp_hint` 는 **5개 위치**에만 등장한다:

| 파일:줄 | 내용 |
|---|---|
| `wavesplat/optimizer/bank_mmse.py:121` | `def fit_bank_mmse(P, z, depth, sigma0, kp_hint=None, ...)` ← **파라미터는 존재** |
| `wavesplat/optimizer/bank_mmse.py:133-134` | docstring: *"kp_hint: peak wavenumber if known; None (deployment) = two-stage beamform zoom"* |
| `wavesplat/optimizer/bank_mmse.py:145-153` | `if kp_hint is None: <2단 beamform> else: kp = float(kp_hint)` |
| ★`wavesplat/optimizer/sequential_update.py:863` | `fit_bank_mmse(P, z, self.depth, sigma0, kp_hint=None, seed=s)` ← **리터럴 None** |
| `wavesplat/tests/test_bank_mmse_port.py:83-106` | 테스트만 `kp_hint=0.13` 을 쓴다 |

⟹ ★**`sequential_update.py:863` 이 `kp_hint=None` 을 하드코딩한다. `self.config` 조회가 없다.**
`configs/estimation.yaml` · `run.py DEFAULT_CONFIG` 어디에도 대응 키가 없다.
`initializer.py` · `learned_init.py` 에는 `kp_hint`/`k_hint` 가 **0건**
(`learned_init.py:196-295` 의 `kp` 는 전부 **자기 빔포밍 피크**에서 내부 계산된 것).

⟹ **★최우선 산출: 수리는 한 줄이다.**
```python
# wavesplat/optimizer/sequential_update.py:863
kp_hint=self.config.get('bank_kp_hint', None)
```
+ `run.py DEFAULT_CONFIG` 와 `configs/estimation.yaml` 에 `bank_kp_hint: null` 미러.
`None` 기본값이면 **기존 동작과 비트 동일** ⟹ 회귀 위험 0.
(`RMSE_near` 가 46일간 계산되고 저장 안 된 것과 **정확히 같은 형태의 결함**이다:
계산 능력은 있는데 키 한 줄이 없어 도달 불가.)

### 2.2 배포 실측 경로 vs d3 — 남은 델타는 **3개뿐**

전제: `init_method='auto'` + GT-free 입력 ⟹ 실측 게이트 발화 ⟹ `bank_mmse`
(`sequential_update.py:460-497`, `run.py:57 real_estimator='bank_mmse'`).

| 항목 | d3 (작동함) | 배포 bank 경로 | 델타 |
|---|---|---|---|
| 수학 | `d3_sa_nomp.mmse_bank_estimate` | `bank_mmse.fit_bank_mmse` (비트-정확 이식, `bank_mmse.py:124`) | **없음** |
| `kp` | 외부 힌트 `kp_hint=kp` (`:322`) | **2단 blind beamform** (`bank_mmse.py:145-151`) | ★**Δ1** |
| `sigma0` | `0.05·4·nanstd(전체 필드)` (`:225`) | `0.05·4·std(관측)` (`sequential_update.py:861-862`) | Δ2 |
| 앙상블 | 단일 fit | median-of-3 (`run.py:60 real_bank_ensemble_k=3`) | Δ3(배포가 더 강함) |
| `m_cap` | 600 (`:138`) | 600 (`sequential_update.py:857 bank_max_obs`) | 없음 |
| 밴드 | `(0.35, 2.5)·kp` (`:155`) | `(0.35, 2.5)·kp` (`bank_mmse.py:121`) | 없음 |
| `dk_c` | `kp/10` | `kp/10` | 없음 |
| `--dual_band`/`--bound_wave`/`--bw_cross` | 있음(`:149-169`) | **이식 안 됨** | Δ4(옵션, 기본 off) |

### 2.3 ⚠blind kp 가 실제로 얼마나 나쁜지는 **이미 측정돼 있다**

`MEASURED` — `/home/vexit/wavesplat/wavesplat/tmp/M3_RESULT_2026-07-22.md`:
> 최난셀 Tp13_Hs5 (λ≈264 m): **beamform kp = 0.0327 vs 참 kp ≈ 0.0238 (+37% 과대)**.
> 오라클 kp 에서 `T−L = +0.1389` 이던 우위가 센서 beamform kp 에서 **+0.0222 (~6× 축소)**,
> `T−N` 은 **소멸(n.s.)**.

⟹ 판정 = **`FLAG (헤드라인 oracle-kp 의존)`**.
`INFERRED` — 긴 파장/좁은 개구에서 blind beamform kp 가 크게 편향되고,
`fit_bank_mmse:157-160` 의 밴드가 `(0.35kp, 2.5kp)` 라 **kp 가 틀리면 밴드 전체가 어긋난다**.

⛔★★★**이 가설은 §5.4-#3 에서 내가 스스로 반증했다.** `results/direction3/band_selection.json`
이 실측 YS17 에서 `kp` 를 **0.4×–1.0× 로 흔들어도 팬 nRMSE 가 0.671–0.690(±3%)** 임을 측정했고,
출하 blind beamform 이 YS17 에서 실제로 뽑은 `kp=0.1154` 는 참값 대비 **+7%** 에 불과하다
(`deployed_path_verify.json`). ⟹ **kp 는 YS17 에서 병목이 아니다.**
`kp_hint` 경로 부재는 **실제 결함이고 수리가 한 줄**이지만(§2.1),
**08-19 붕괴의 설명으로는 1순위가 아니다. 1순위는 §0 의 라우팅이다.**

★★★**동시에 M-3 가 반대 방향의 경고도 준다**: `kp_hint` 는 **공짜가 아니라 오라클 입력**이고,
그것을 주면 팔마다 다르게 작용해 **격차를 인위적으로 벌린다**(팔-대칭 전제가 M-3 에서 거짓으로 판명).
⟹ 다음 배치는 `kp_hint` 를 **처치 요인으로 사전등록**하고, `SHIP`·기준선 **양 팔에 동일하게** 주거나
동일하게 빼야 한다. 한 팔에만 주면 M-3 가 죽인 바로 그 오류가 반복된다.

---

## 3. 레코드 표 — 쓸 수 있는 것은 몇 개인가

원자료 = `/home/vexit/wavesplat/tmp/record_characterization.json` (13행, `MEASURED`)
+ 유효박스는 `/home/vexit/wavesplat/wavesplat/tmp/B3_regime_position_2026-08-12/RESULT.md:36-44`
및 `.../real_run.log` (`largest_valid_box`, `b3_regime.py:149,189-191`:
`frozen = np.all(Z==Z[0],axis=0)` 로 상수클램프까지 죽인 뒤의 **최대 전유효 직사각형**).

| # | 레코드 | d3 태그 | 공칭 ext | **유효박스** | valid_frac | 수심 | Hs | λ_p | k_p | T_p | dt | dur_s | t_coh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | YS_2017/20170513_050000 | **YS17** | 220×240 | **142×111** | 0.567 | 50 | 1.88 | 58.2 | 0.108 | 6.10 | 0.100 | 449.9 | 9.56 |
| 2 | LJ_2018/…Grid2_short | **LJ18g** | ⛔250×300(클램프) | **162×183** | ⛔1.0(거짓) | 45 | ⛔27.5(클램프) | 301.4 | 0.021 | 16.21 | 0.100 | 1799.8 | 7.42 |
| 3 | AA_2015/20150305_103500 | **AA15a** | 65×85 | **41×46** | 0.649 | 17 | 2.44 | 51.7 | 0.122 | 5.85 | 0.0833 | 1249.9 | 28.64 |
| 4 | AA_2014/20140327_091000 | **AA14** | 70×86 | **37×34** | 0.432 | 17 | 1.36 | 35.0 | 0.180 | 4.74 | 0.0833 | 1666.6 | 4.35 |
| 5 | AA_2015/20150515_090000 | **AA15b** | 55×80 | **45×60** | 0.897 | 17 | 1.18 | 40.1 | 0.157 | 5.09 | 0.0833 | 666.6 | 5.51 |
| 6 | LJ_2018/…Grid1_short | (LJ18n) | 40×40 | **40×40** | 1.0 | 45 | 7.46 | ⛔40.1(패치인공) | 0.157 | 5.07 | 0.100 | 1799.8 | 10.09 |
| 7 | BS_2011/20111001_161800 | (BS11b) | 13.5×13 | **14×13** | 1.0 | 30 | 0.58 | 9.4 | 0.671 | 2.45 | 0.0667 | 721.9 | 0.79 |
| 8 | BS_2011/20111004_113800 | (BS11a) | 14×13.5 | **14×14** | 1.0 | 30 | 0.36 | 14.0 | 0.450 | 2.99 | 0.0833 | 1197.9 | 1.53 |
| 9 | BS_2011/20111004_130700 | — | 13.5×13.5 | **13×14** | 1.0 | 30 | 0.54 | 9.6 | 0.654 | 2.48 | 0.0833 | 1797.8 | 5.78 |
| 10 | BS_2011/20111004_153000 | — | 19×14.5 | (미측정) | 1.0 | 30 | 0.63 | 18.9 | 0.333 | 3.48 | 0.0833 | 1797.9 | 4.76 |
| 11 | BS_2013/20130922_130001 | — | 19×19 | **19×19** | 1.0 | 30 | 0.50 | 13.5 | 0.466 | 2.94 | 0.100 | 1798.0 | 5.18 |
| 12 | BS_2013/20130925_121501 | (BS13a) | 19×19 | **19×19** | 1.0 | 30 | 0.72 | 19.1 | 0.329 | 3.50 | 0.0833 | 1798.1 | 5.93 |
| 13 | BS_2013/20130930_102001 | (BS13b) | 17×17 | **17×17** | 1.0 | 30 | 0.68 | 17.1 | 0.368 | 3.31 | 0.0833 | 1798.1 | 3.28 |

### 3.1 판정 — **d3 기하로 즉시 쓸 수 있는 레코드 = 5개** (`RECORDS`, `:66-90`)

`MEASURED` — `d3_falcon_real_operator.py:67` 주석: *"only records whose patch can host a
(possibly scaled) forward fan"*. 즉 나머지 8개는 **d3 저자가 이미 배제했다**.

⚠**5개조차 무조건은 아니다** — 세 가지 경고:

1. ★**YS17 의 docstring 이 틀렸다.** `:5-6` 이 *"220 x 240 m — the only record large enough
   for the real fan geometry"* 라 적었으나, **유효박스는 142×111 m** 다
   (`B3_regime_position_2026-08-12/real_run.log:2`). `r_fan=(20,100)` 팬은 **111 m 축을 넘는다**.
   `INFERRED` — 팬 표집이 `np.isfinite` 로 무효점을 버리므로(`:249`) 실행은 되지만
   **먼 링에서 커버리지가 비대칭으로 줄어든다**. `n_obs` 는 6/6 창에서 600 상한에 붙어 있어
   저장된 진단으로는 이 손실이 안 보인다. ⟹ **다음 배치는 링별 유효점수를 기록해야 한다.**
2. ⛔**LJ18g 의 3개 수치는 인용 금지**: 공칭 250×300 m 개구 · `mask_Z` 의 "100% 유효" ·
   `Hs≈26–27 m` (`B3.../RESULT.md:74`). 상수클램프 31.47% 인공물.
   ⚠단 d3 의 `MRecord(robust=True)` 는 `_robust_mask`(`d3_sa_multirecord.py:120-131`,
   `tsd > 0.1*med`)로 **동결화소를 이미 죽인다** ⟹ d3 의 LJ18g 저장값(Hs 9.5–13.0 m)은
   record_characterization 의 27.5 와 다르다. **어느 쪽이 맞는지 이 배치는 확인하지 않았다.**
3. **AA×3 은 유효박스 34–60 m 인데 팬이 `(8,40) m`** ⟹ 팬 사거리가 박스와 같은 크기다.
   `r_fan/λ_p` = AA14 1.14 · AA15a 0.77 · AA15b 1.00 · YS17 1.72 · LJ18g 0.33 ⟹ **레코드마다
   무차원 개구가 5배 갈린다.** ⛔**레코드를 뭉뚱그려 평균 내지 마라.**

### 3.2 나머지 8개를 살릴 수 있는가

`INFERRED` — BS 7개는 유효박스 13–19 m 에 λ_p 9.4–19.1 m ⟹ **개구/λ ≈ 0.7–2.0**.
d3 의 `LJ18n` 이 바로 그 레짐이고 저장소가 **"NEGATIVE CONTROL"** 이라 라벨했다
(`d3_sa_multirecord.py:88-93`). B-3 는 같은 자산에서 *"작은 개구에서는 혼잡이 희소처럼 보인다"*
를 측정했다(`B3.../RESULT.md:96-103`, AUC 0.596@45 m → 0.843@195 m).
⟹ **8개를 추가하면 n 은 늘지만 전부 "개구 부족" 셀**이고, 저장소의 기존 측정상
**그 셀에서는 레짐 구분 자체가 안 실린다**. **음성 대조군으로만 사전등록할 것.**

---

## 4. ★★★짝의 단위와 최대 유효 n

### 4.1 후보 3개와 각각의 독립성

| 단위 | 개수 | 독립성 | 판정 |
|---|---|---|---|
| **프레임** | 수만 | ⛔`t_coh` 0.79–28.6 s 안에서 완전 상관 | **사용 불가**(의사복제) |
| **레코드 × 시간창** | **101** (§4.2) | ⚠기억창은 안 겹치나 **바다 상태가 같다** | 2차 단위로만 |
| **레코드(= 바다)** | **5**(d3) / 13(전체) | ✅서로 다른 사이트·날짜·해상 | ★**군집 단위** |

★**"레코드 안의 창들은 같은 바다의 반복측정"이 추론이 아니라 측정이다** (`MEASURED`,
`results/direction3/falcon_real_operator_*.json`):
- **AA15a**: 6창 중 **5창이 `kp = 0.1216` · `pdir = −37.2°` 로 완전 동일**
- **YS17**: 창 1–3 이 `kp = 0.1342` · `pdir = −129°` 로 동일, 6창 전체가 `kp∈[0.077,0.134]`
- **AA14**: 창 2·4 가 `kp=0.2313 pdir=−39` 동일, 창 3·6 이 `kp=0.1939 pdir=−22` 동일

⟹ 창 수를 늘려도 **처치가 마주하는 바다의 다양성은 늘지 않는다.**
저장소가 이미 같은 규율을 세웠다: M-1 이 *"160행 = 32바다 × 5draw"* 를 **바다 수준 n=32** 로
집계했고(`tmp/M1_kp_hand_prereg_2026-07-22.md`, "바다가 군집 단위"),
M-3 는 *"바다-군집 t + wild bootstrap"* 을 썼다(`tmp/M3_sensor_kp_prereg_2026-07-22.md`).
`memory/feedback_control_stratifying_variable.md` = **pooled 상관 승격 금지(누적 5회)**.

### 4.2 레코드 × 창 최대치 (겹치지 않는 기억창, `MEASURED` 산술)

`d3` 규약 그대로: `lo = W_max/dt + 1`, `hi = T_all − lead_max/dt − 2`,
간격 ≥ `S = W_max + lead_max`(기억창끼리 겹치지 않는 최소 간격).

| 태그 | T_all | dt | S=W+L | 가용 시작범위 | **n_max** | t_coh | 비고 |
|---|---|---|---|---|---|---|---|
| AA14 | 20000 | 0.0833 | 54 s | 1611.8 s | **30** | 4.35 | |
| LJ18g | 17999 | 0.100 | 60 s | 1739.6 s | **29** | 7.42 | ⚠클램프 |
| AA15a | 15000 | 0.0833 | 54 s | 1195.3 s | **23** | 28.64 | ⚠t_coh 가 S 의 53% |
| AA15b | 8000 | 0.0833 | 54 s | 612.2 s | **12** | 5.51 | |
| **YS17** | 4500 | 0.100 | 60 s | 389.7 s | **7** | 9.56 | ★**병목**(449.9 s 레코드) |
| | | | | **합계** | **101** | | (d3 기본은 6×5 = 30) |

⟹ **레코드×창 최대 유효 n = 101** (d3 의 40 s 기억창 · 겹침 0 기준).
`W_max` 를 20 s 로 줄이면 `S` 가 34/40 s 로 떨어져 **≈157** 까지 늘지만,
YS17 은 `S=40` 에서도 **10** 이 상한이다.

### 4.3 검정력 — **정직한 상한**

레코드 군집 부호검정의 원리적 최소 단측 p:

| 군집 n | n/n 전승 | (n−1)/n | α=0.05 도달 |
|---|---|---|---|
| **5** (d3 즉시 가용) | **0.0312** | 0.1875 | ⚠**5/5 에서만** — 한 번만 지면 즉시 실패 |
| 7 (YS17 창을 군집으로 오인 시) | 0.0078 | 0.0625 | ⛔의사복제이므로 **금지** |
| **13** (전체 레코드, 팬 재척도 필요) | 0.0001 | **0.0017** | ✅11/13(p=0.0112)까지 여유 |

★★★**이것이 저장소가 이미 두 번 물린 함정이다**: Rx-4 에서 *"표준집합 4개 전부 α=0.05 원리적 불가"*
(`memory/project_session_2026-08-07_rx4.md`).
⟹ **다음 배치는 MDE 와 검정 방식을 런 전에 동결해야 하고, 부호검정을 쓸 거면
n=5 로는 5/5 만이 통과라는 사실을 PREREG 에 적어야 한다.**

### 4.4 ★권고하는 설계

1. **PRIMARY 통계 = 연속량의 레코드-군집 검정**, 부호검정 아님.
   창별 짝 Δ(같은 창·비트동일 관측에서 `SHIP` vs 기준선) → **레코드 내 중앙값** →
   레코드 5개에 대한 **wild/cluster bootstrap**(M-3 의 방식, `M3_sensor_kp_prereg:15`).
   ⟹ 창 101개가 **분산을 줄이는 데는 기여하되 n 을 부풀리지 않는다.**
2. **짝은 팔 사이에서 만든다**(같은 창·같은 팬 표집·같은 시드). 관측이 비트동일이면
   짝지음이 바다 변동을 완전히 제거하므로 **독립성 비용 0**.
   ★`memory/project_s15_direction` 의 **G-8/G-8b** 처럼 **팔별 필드 존재를 PREREG 게이트로 검사**할 것
   (S13·HOSGATE 가 기준선 셸을 0행 기록해 두 번 못 쟀다).
3. **층화는 레코드, 절대 pooled 금지.** `r_fan/λ_p` 가 0.33–1.72 로 5배 갈리므로
   (§3.1-3) 레코드를 **개구비로 층화**해 보고할 것.
4. **n=13 을 원하면 팬 척도 법칙을 런 전에 동결하라.** d3 는 그것을 **암묵적**으로만 했다
   (λ_p ≳ 58 m → `(20,100)`, λ_p 27–52 m → `(8,40)`). 사후에 정하면 자유도가 된다.
   BS 7개는 **음성 대조군**으로만(§3.2).
5. **YS17 은 7창이 상한**이다. YS17 하나로 배치를 세우면 **군집 n=1** 이고 어떤 검정도 못 한다.

---

## 5. 저장소가 실측에서 이미 낸 수 — 측정된 것 / 미측정

### 5.1 `MEASURED` — d3 팬 조우예보 (5레코드 × 6창, 2026-07-06/08)

원자료 `results/direction3/falcon_real_operator_{YS17,AA14,AA15a,AA15b,LJ18g}.json`
(내가 재집계함, 런 0회). 통계 = 창을 풀링한 Pearson corr(예측, 스테레오 GT).

| 레코드 | U | r_fan | W=10 (0–10 s) | W=20 (0–10 s) | W=40 (0–10 s) | W=40 (10–20 s) | map-nowcast corr (중앙, W=10) |
|---|---|---|---|---|---|---|---|
| **YS17** | 2.0 | 20–100 | +0.632 | +0.697 | **+0.708** | +0.455 | **0.84** |
| **LJ18g** | 2.4 | 20–100 | **+0.616** | +0.585 | +0.461 | +0.087 | 0.69 |
| **AA15a** | 1.2 | 8–40 | +0.602 | **+0.710** | +0.657 | +0.595 | 0.79 |
| **AA15b** | 1.2 | 8–40 | +0.367 | +0.218 | **+0.565** | −0.025 | **0.91** |
| **AA14** | 1.2 | 8–40 | **+0.428** | +0.288 | +0.282 | (n=2, nan) | 0.82 |

⟹ `MEASURED` — **작동 조건에서 YS17 의 nowcast corr 중앙값은 0.79–0.84 이고
조우 corr(0–10 s)는 +0.63~+0.71 이다.** 08-19 시도의 "평평한 상수장"과는 전혀 다른 상태다.
⚠**이 수치들은 corr 이지 nRMSE 가 아니다.** 저장된 파일에 nRMSE 열이 없다
⟹ 08-19 의 0.265 와 **직접 비교 불가**. 기준선 팔(ZERO/LETKF)도 이 파일에 **0행**이다.

### 5.2 `MEASURED` — 이미 죽은 실측 주장

- ⛔**"실측 = 희소형"** — B-3 가 클램프 인공물로 철회. 실측 레짐 질문 = **`UNANSWERABLE`**
  (필요 개구 ~195 m, 보유 최대 162×183 m, 그 지점 AUC≈0.78 = 여전히 "약함").
  `tmp/B3_regime_position_2026-08-12/RESULT.md:12,107`
- ⛔**M-3 `FLAG`** — B-2 헤드라인이 **oracle-kp 의존**. 센서 beamform kp 로 바꾸면
  `T−L` +0.139 → **+0.022 (~6× 축소)**, `T−N` **소멸**. `tmp/M3_RESULT_2026-07-22.md`
- ⚠**M-1 `ARGUED only`** — 5/5 방향 일치하나 7쌍 중 3쌍만 CI 가 0 제외, **n=32 바다로 미봉인**.
  `tmp/M1_RESULT_2026-07-22.md`
- **bank-MMSE 포트가 실측 3창에서 v2_konly 를 이겼다** (nowcast nRMSE
  0.858→0.498 AA15a w5 · 0.927→0.666 AA15b w3 · 1.048→0.718 YS17 w0)
  `sequential_update.py:800-802`. ⛔**"~2× 개선" 인용 금지** — 재도출 비 1.723/1.392/1.460,
  **중앙 1.46×**, 2× 도달 창 **0개** (`run.py:53-56` 의 P1 정정).
  ⚠원자료가 **두 파일에 갈라져 있다**: bank = `tmp/bankmmse_port_2026-07-29/m3_field.json`,
  v2_konly = `tmp/confirm2_2026-07-27/real_gif_info.json`.

### 5.3 **미측정** — 다음 배치가 실제로 여는 공백

| 질문 | 상태 |
|---|---|
| 출하 `SequentialUpdater`(auto + 실측게이트 → bank)가 실측에서 무슨 nRMSE 를 내는가 | ⛔**측정 0회** |
| `kp_hint` 유/무의 **개입 대조**(같은 창·같은 관측) | ⛔**측정 0회** — M-3 는 다른 아레나(HOS 합성) |
| 실측에서 `SHIP` vs `ZERO`(제로필드) vs `LETKF` **짝 비교** | ⛔d3 파일에 기준선 **0행** |
| 실측 nRMSE(파일에 있는 건 corr 뿐) | ⛔**컬럼 부재** |
| 지지집합 **밖**(팬이 못 본 영역) 실측 성능 | ⛔d3 는 채점 안 함 · 08-19 은 사전등록 없이 쟀다 |
| 실측 팬 링별 유효 관측수(YS17 유효박스 111 m < r_fan 100 m 문제) | ⛔미기록 |

⚠**아래 §5.4 는 병렬 서브에이전트의 광역 스윕 결과다** — D-17/D-18/D-20 · E-60 ·
"실측 전이 LOSS" 의 원 근거 · 출하 추정기의 실측 실행 이력 전수.

### 5.4 광역 스윕 (병렬 서브에이전트, 전건 재확인 대상)

⚠**서브에이전트의 추론 1건을 내가 반증했다** — 그것이 `#1` 이다. 나머지는 그가 준 `파일:줄` 을
그대로 인용하되, **내가 직접 열지 않은 항목은 `[2차]` 로 표시**했다(열람 범위 = 주장 범위).

#### ★#1 정정 — "08-19 런에서 실측 게이트가 발화했을 것" 은 **거짓** (내가 `MEASURED`)
서브에이전트는 *"`real_domain_gate` 를 끄지 않았으므로 게이트가 발화해 bank_mmse 로 갔을 것"*
이라 `INFERRED` 했다. **틀렸다.** 게이트의 바깥 조건이 `real_domain_gate` 가 아니라
**`init_method == 'auto'`** 다(`sequential_update.py:445`), 그리고 W4 base config 는
`arms_config.py:67` 에서 **`'init_method': 'learned'`** 로 못박는다.
⟹ 게이트 블록에 **진입 자체를 안 한다**. `real_domain_gate` 값은 읽히지도 않는다.
(W4 docstring `arms_config.py:63-66` 이 그 사실을 스스로 적었다.)
★이 오류가 정확히 §0 의 요지다 — **팔 이름만 보고 어느 경로가 도는지 추측하면 안 된다.**

#### 이미 측정된 것 (`[2차]` = 서브에이전트 인용, 내가 미열람)

| 항목 | 수치 | 출처 |
|---|---|---|
| ★**출하 `make_updater` 가 실측에서 실제로 돈 적 있다** | `real_gate=true`, `route=bank_mmse`, median-of-3 | `tmp/bankmmse_port_2026-07-29/verify_deployed_path.py` (**내가 직접 확인**) |
| ★출하 bank 경로 실측 nowcast nRMSE (/σ) | AA15a_w5 **0.4885** · AA15b_w3 0.6486 · YS17_w0 **0.7228** · LJ18g_w3 **1.1829(무기술)** | `deployed_path_verify.json`(내가 확인) + `real_gif_info.json`[2차] |
| 구 fallback `v2_konly` 실측 10창 | nRMSE 0.750–1.099, **7/10 이 제로예측기(≈1.0) 근처 이상** · YS17_w1 corr **−0.027** | `tmp/real_gif_2026-07-27/scan.jsonl` [2차] |
| **실측 refit(3레코드) 계보** | 07-29 원본 → **"탐색적" 재라벨·승격 철회** → 08-01 사전등록 held-out(5레코드×21신선창) 이 정본 | `tmp/P1_real_path_2026-08-11/DECISION.md:44` [2차] |
| **08-01 사전등록 판정** | **`CHEAP-ALTERNATIVE`** — §3.1 PASS(5/5, p=0.03125) · §3.2 FAIL · 해악게이트 FAIL | `tmp/prereg_heldout_2026-08-01/verdict.json` [2차] |
| **"실측 전이 LOSS" = 서로 다른 두 사건** | (A) 07-10 학습 prior 4레코드 **−0.0345** · (B) 07-26 ③b 27창 medianΔ **−0.0571, p=0.0046, 6/27 승** | `docs/AB_REAL_TRANSFER_AND_WINDOWS_2026-07-10.md:23-25` · `tmp/wp_real_2026-07-25/WP_REAL_2026-07-25.md:11,24-30` [2차] |
| (B') 08-06 위생 재검 | 부호 유지, **p 0.0046 → 0.100** (널통과창 n=19 만 보면 0.040) | `tmp/A2_realproto_2026-08-06/A2_RESULT.md:12-18` [2차] |
| **D-17** | ⚠**합성 전용**(BF3/BF4, U=11.3). G1 FAIL(팬 밖 corr→0.0) · G3 FAIL(4805 ms) | `results/direction3/d17_ctrl_eval.json` [2차] |
| **D-18** | ⚠**합성 전용**. learned 팔 **SKIPPED** | `results/direction3/d18_ctrl_tune.json` [2차] |
| ★**D-20 = 유일한 실측 제어 배치** | 5레코드 120창. 원 JSON PASS(5/5, pooled −0.1233)이나 **군집강건 JSON `headline_pass=false`**(cluster p 0.0732 · wild 0.0252 · sign 0.0625) | `results/direction3/d20_replay_eval{,_clusterrobust}.json` [2차] |
| D-20 이력 | 07-12 "t −4.02, 3레코드" → 07-14 **강등**(창을 독립 취급한 iid t) → 07-15 5레코드로 부분회복 | memory `project_kp_floor_audit_2026-07-14.md:81-85` [2차] |
| **E-60(E60K)** | 주 아레나 PASS 는 **HOS 합성**. 실측 다리는 **k=5 site-day "방향-일관 기록"** 뿐 | `tmp/E60K_RESULT_2026-07-20.md:9,34,97` [2차] |
| ★**출하 라우터의 실측 거동** | `auto` 가 실측에서 **30/30 learned 로 축퇴**, guard **0/30** 발화 ⟹ *"실측에서 auto 는 라우터가 아니다"* | `tmp/blind_auto_real_2026-07-26/RESULT.md:30-45` [2차] |
| **개입 지점 부재 4건** | `lambda_l1`·`sigma0`·`bank_memory_s`·가우시안 노브 15개가 실측 예측면을 **비트~바이트 동일**하게 둔다 | `tmp/P1_real_path_2026-08-11/DECISION.md:22-30` [2차] |
| **하네스 21% 불일치 원인** | `smult` 아니라 **관측 배열 ORDER**(600-cap 이중적용 → val 분할이 다른 점을 뽑음). 수정 후 **12/12 비트동일** | `tmp/P1_real_path_2026-08-11/RECONCILE.md:10-18` [2차] |
| **UQ 보정 붕괴** | `z_std 2.047`, `cov68 0.519`, `cov95 0.796` | `results/direction3/uq_gating_real.json` [2차] |
| **2-lobe 계열은 실측에서 오지정** | `ANCHOR FAIL … deliverable scoped to synthetic JONSWAP (scientific negative)` | `results/direction3/e46_real_supply.json` [2차] |
| **07-27 첫 정면대결이 유일** | *"this run is the **first** such measurement … an open item, **not a claim**"* | `tmp/confirm2_2026-07-27/real_gif_info.json:484` [2차] |

#### ★★★#2 — **정규화 함정이 이미 문서화돼 있다** (`MEASURED`, 내가 직접 확인)

| 하네스 | nRMSE 분모 | 근거 |
|---|---|---|
| d3 / bankmmse_port / deployed_path_verify | **σ(GT)** | `tmp/bankmmse_port_2026-07-29/measure_blind_vs_frozen.py:44,51` (`sg = np.std(Z…)`) |
| 08-19 `real_estimate.py` · 합성 배치 전체 | **4σ = Hs** | `tmp/COMPARE_REAL_SYNTH_2026-08-19/real_estimate.py:142-143` |

⟹ ★**두 숫자는 4배 차이가 난다.** 08-19 의 제로필드 기준선이 **정확히 0.250** 인 것이
그 증거다(σ/4σ = 0.25). 같은 규약으로 환산하면:

| | YS17 nRMSE (÷4σ) | 제로필드 (÷4σ) | 판정 |
|---|---|---|---|
| **출하 bank 경로**(07-29, w0 nowcast) | **0.181** | 0.250 | ✅**제로필드를 27.7% 이긴다**(n=1창) |
| ~~08-19 `L-SHIP`~~ | ~~0.265~~ | ~~0.250~~ | ⛔**§0-A 로 무효 — 질의격자 버그값** |

⛔**이 대비를 "출하가 이긴다"로 인용하지 마라.** 창 인덱스·기억창 길이·정지 vs 이동 센서·
프레임 stride 가 전부 다르고, **§0-A 이후 아랫줄이 사라져 비교 자체가 성립하지 않는다.**
남는 것은 윗줄 하나 = *"출하 bank 경로가 YS17 한 창에서 제로필드를 이겼다"*. **n=1.**
✅**단 채점 영역은 같다** — `scene_real`(`scripts/d3_compare_gifs.py:62-72`)의 `maps_gt` 는
`rec.gx[::sq], rec.gy[::sq]` 로 **전체 패치를 4 m 로 데시메이트**한 것이고 `score()` 는
`np.isfinite(g)` 인 전 화소를 채점한다(`measure_blind_vs_frozen.py:47-55`).
⟹ **08-19 의 "전체 패치 채점"과 같은 영역이다.** (§1.5 의 d3 조우/맵 채점과는 다르다.)

⚠**저장소가 이 함정에 이미 한 번 물렸다**: `tmp/P1_real_path_2026-08-11/DECISION.md:1-22` 가
*"배치의 동기 문장이 정규화 인공물이었다 — 같은 분모로 실측 0.171 Hs vs 합성 혼잡 0.178 = **0.96배**"*
라며 P1 배치를 **런 0회로 취소**했다. ⟹ **다음 배치는 분모를 PREREG 에 못박아야 한다.**

#### ★#3 — `kp_hint` 민감도가 **실측에서 이미 스윕돼 있다** (`MEASURED`, 내가 직접 확인)

`scripts/d3_band_selection.py` + `results/direction3/band_selection.json`.
`KP_FACTORS = [0.4 … 2.0]` × 실측 YS17·LJ18g, 팬 관측 nRMSE(÷σ, 팬 내부):

| kp/kp₀ | 0.4 | 0.7 | **1.0** | 1.6 | 2.0 |
|---|---|---|---|---|---|
| **YS17** fan nRMSE | 0.676 | 0.671 | **0.690** | 0.692 | 0.723 |
| **LJ18g** fan nRMSE | 0.537 | 0.552 | **0.550** | 0.659 | 0.721 |

★★★**결정적**: `kp` 를 **2.5배 범위(0.4×–1.0×)로 흔들어도 YS17 팬 nRMSE 는 0.671–0.690 = ±3%** 다.
⟹ ⛔**"kp 가 틀려서 평평해졌다"는 내 가설은 이 자료에서 지지받지 못한다.**
과대(1.6–2.0×) 쪽만 벌점이 있고 그마저 +5~+31% 다.

**이를 뒷받침하는 두 번째 측정** (`MEASURED`, `deployed_path_verify.json`):
출하 blind 2단 beamform 이 실제로 뽑은 `kp` = AA15a **0.0986** · AA15b **0.1423** · YS17 **0.1154**.
레코드 참값(`record_characterization.json`) 0.1216 / 0.1567 / 0.1080 대비 **−19% / −9% / +7%**.
⟹ **YS17 에서 blind beamform 은 +7% 밖에 안 틀린다.** M-3 의 +37% 붕괴는 **λ≈264 m 장파** 셀의 일이다.

⟹ ★**수정된 우선순위**: `kp_hint` 는 **여전히 만들 가치가 있다**(경로 부재는 실제 결함이고
수리가 한 줄이며 LJ18g 같은 장파 레코드가 M-3 레짐이다). 그러나 ⛔**08-19 붕괴의 설명으로
1순위에 두지 마라.** 1순위는 **§0 의 라우팅**이다.

★**대안이 이미 구현돼 있다** — `d3_band_selection.py` PART 2 의 **GT-free 밴드 선택**
(관측의 25%를 홀드아웃해 밴드 후보 중 잔차 최소를 고름). YS17 에서 적응규칙이 고른 밴드는
`(0.35,2.5)`, 오라클 밴드는 `(0.25,3.5)` 로 **오라클이 0.676 vs 적응 0.690 = 2% 차이**.
⚠**`bank_mmse.py` 에는 이 규칙이 이식되지 않았다**(`band` 가 고정 인자, `:121`).

#### #4 — 서브에이전트가 짚은 미해결/긴장

- ⚠`scene_real`(`d3_compare_gifs.py:171`)이 **배 항로 방향 `pdir` 을 조밀 진실면에서 뽑는다**.
  `A2_RESULT.md:44-46` 이 이를 *"미해결 부채로 남긴다"* 라 명시. ⟹ **내 §1.2/§1.3 의 공간-오라클
  지적이 이미 저장소의 공개 부채다.** 새 발견이 아니라 **미상환 항목**이다.
- ⚠**AA14 는 어느 배포경로 JSON 에도 없다**(서브에이전트가 확인). LJ18g 는 `additional_windows` 에 있다
  ⟹ `P1 DECISION.md:38-40` 의 *"AA14·LJ18g 는 배포 bank 경로를 통과한 적 없다"* 중 **LJ18g 부분이 낡았다**.
- ⚠**"실측" 이라는 한국어가 두 뜻**(실제 해양 데이터 / "이 런에서 잰 값")으로 쓰인다.
  S1–S9·W3–W5·S13–S17 의 "실측" 은 **거의 전부 후자이고 합성 코퍼스**다.
- `results/direction3/e60k_real_arena.json` 은 `config.records` 에 BS11a 를 등록했으나 **rows 0행**.

#### ★#5 — 군집 상한이 **저장소에서 두 번 독립 계산됐고 내 §4.3 과 일치한다**

- `tmp/P1_real_path_2026-08-11/DECISION.md:52-60` [2차]: *"BS 7개는 패치 13.5–19 m 로 팬 기하 불가,
  LJ Grid1≡Grid2 동시각 ⟹ **5군집 정확검정 바닥 양측 p = 0.0625**. E-12 의 '5→9/12 레코드' 처방은
  **현 데이터로 실행 불가**."*
- `tmp/DEBATE_novelty_2026-08-19/P1_ocean_reviewer.md:690` [2차]: *"독립 단위가 레코드 5개
  ⟹ 클러스터 바닥 p = 2⁻⁵ = 0.031"*

★★★⟹ **§4.3 의 결론이 3중으로 독립 확인됐다**: 이 자산으로 실측 정확도 우위를
α=0.05 로 사려면 **5/5 전승(단측 0.031)이 유일한 통과 경로**다.
⚠★**게다가 LJ Grid1 과 Grid2 는 같은 시각의 같은 바다**(`d3_sa_multirecord.py:30` 이
*"analysis renormalizes with lam_ref/c_p_ref taken from Grid2 (same sea)"* 라 적었다)
⟹ **13 레코드조차 독립 바다는 12개이고, 사이트로 묶으면 6개(AA/BS11/BS13/LJ/YS + 연도)** 다.
⛔**§4.3 의 "n=13 → p=1.2e-4" 는 상한이지 보장이 아니다.**

---

## 6. 다음 배치가 반드시 사전등록해야 할 것 (체크리스트)

0. ★★★**분모를 못박아라.** 실측 하네스는 `RMSE/σ`, 합성 배치와 08-19 은 `RMSE/4σ` 다(§5.4-#2).
   **4배**다. 저장소가 이 함정으로 배치 하나(P1)를 런 0회로 취소한 적이 있다
   (`tmp/P1_real_path_2026-08-11/DECISION.md:1-22`). PREREG 에 분모와 제로필드 기준값을 숫자로 적어라.
1. ★**팔 정의를 코드로 못박아라**: `init_method` 값을 행마다 기록하고,
   **`_auto_real_gate == True` 와 `_auto_selected == 'bank_mmse'` 를 런 전 게이트로 검사**하라
   (`sequential_update.py:468,494`). 08-19 은 이 게이트가 있었으면 즉시 잡혔다.
   ⚠**`init_method='learned'`(W4/W5 팔)는 실측 게이트를 통째로 우회한다** — 팔 이름이 아니라
   **`init_method` 값**이 경로를 정한다.
2. ★**원자 카운터로 개입 0회를 확인하라** — `fit_bank_mmse` 호출수와 `kp_hint` 실수신값.
   W3 설계가 그렇게 `SHIP-FITPHI` 0/3 개입을 런 전에 잡았다.
3. **`kp_hint` 는 양 팔에 대칭으로** 주거나 빼라 (M-3 의 팔-비대칭 오류 반복 금지).
   그리고 그 출처가 **공간 오라클**임을 문장에 적어라.
4. **`sigma0` 를 오라클(전체 필드 Hs)에서 뽑을지 관측 Hs 로 뽑을지 명시**하라 — d3 와 배포가 다르다.
5. **채점 영역을 접두어까지 적어라**: `enc:`(배 미래 궤적) / `map:`(팬 내부) / `full:`(전체 패치).
   d3 는 앞 둘, 08-19 은 셋째를 쟀다. **같은 이름으로 부르면 안 된다.**
6. **군집 = 레코드. n=5 면 5/5 만 α=0.05.** MDE 를 런 전에 동결.
7. **임포트 경로 파일의 md5 를 동반 기록**(`git_head` 만으로는 더러운 트리를 못 잡는다).
8. `effective_config` 는 **객체 되읽기**로 만들어라(입력 dict 복사 금지 — S13b 교훈).
9. ⚠**지연을 같이 기록하라.** 출하 bank 경로의 실측 `predict_ms_per_frame` 은
   AA15a **45.1** · AA15b **9.2** · **YS17 959.3 ms** 다(`deployed_path_verify.json`, `MEASURED`).
   YS17 은 ≤100 ms 제약을 **10배** 넘는다(격자 화소수 탓으로 추정, 미확인).
   ⟹ 정확도 배치라도 이 열을 버리면 "제어-호환" 문장을 또 못 산다.
10. ⚠**"실측"이라는 낱말을 쓰지 마라 — `real-data` / `measured-in-this-run` 을 구분해 적어라.**
    저장소에서 이 두 뜻이 섞여 있고, S1–S9·W3–W5·S13–S17 의 "실측"은 **전부 합성 코퍼스**다.

---

## 7. ⛔이 정찰이 **답하지 않은** 것 (다음 배치가 떠맡아야 함)

1. ~~08-19 붕괴의 실제 기제~~ — **§0-A 로 답이 나왔다: 질의격자 축 슬라이싱 버그.**
   ⟹ 남은 질문은 **"버그를 고친 뒤 그 팔이 무슨 값을 내는가"** 이고, 그것은 아직 미측정이다.
   그리고 **라우팅 델타(§0-B)는 별개로 여전히 미개입 확인 상태**다
   (`init_method='auto'` 로 같은 스크립트를 돌리는 것이 그 시험이다).
2. **YS17 유효박스 111 m < r_fan 100 m** 의 실제 커버리지 손실 — 링별 유효점수 미기록.
3. **LJ18g 의 `Hs`** — `record_characterization.json` 27.5 m vs d3 저장값 9.5–13.0 m.
   `MRecord(robust=True)` 가 차이를 만드는 것으로 보이나 **확인 안 했다**.
4. **08-19 런의 창 인덱스가 `deployed_path_verify` 의 `YS17_w0` 와 얼마나 겹치는가** — 미대조.
5. **`d3_band_selection` PART 2 의 GT-free 밴드 선택을 `bank_mmse.py` 에 이식할 때의 비용/이득** — 미평가.
6. 서브에이전트 `[2차]` 표시 항목 전부 — 나는 그 파일들을 **열지 않았다**.
