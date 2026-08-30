# 학습 경로 지도 (LEARNED PATH MAP) — 2026-08-30

**감사 범위(열람 범위 = 주장 범위).** 아래 파일만 실제로 열었다. 런 0회, 패키지 수정 0줄, 커밋 0회.

| 열람한 파일 | 비고 |
|---|---|
| `wavesplat/run.py` | 전체 (792행) |
| `wavesplat/optimizer/sequential_update.py` | 1–1100, 1341–1720, 1820–2032 (2032행 중 약 85%) |
| `wavesplat/optimizer/learned_init.py` | 전체 (409행) |
| `wavesplat/optimizer/varpro.py` | 전체 (275행) |
| `wavesplat/optimizer/coarse_to_fine.py` | 1–60, 140–220 |
| `wavesplat/optimizer/bank_mmse.py` | 30–60 |
| `wavesplat/model/spectral_gaussian.py` | 90–140 + 함수 목록 |
| `wavesplat/model/confidence.py` | 전체 |
| `wavesplat/model/tnse.py` | 176–230 |
| `wavesplat/real_data.py` | 285–315 + grep |
| `configs/estimation.yaml` | 전체 |
| `wavesplat/tests/test_shipping_defaults.py` | 1–70 + grep |
| `wavesplat/tests/test_config_exposure.py` | grep |
| `scripts/d3_beamform_cond_diffusion.py` | 20–45, 155–200 |
| `scripts/d3_beamform_net.py` | grep |
| `scripts/d3_setdiff_prior.py` · `d3_setdiff_posterior.py` · `d3_setdiff_posterior_b2.py` | grep + 20–100 |

⛔**안 연 것**: `initializer.py` 본문(상수 3개만 grep) · `sequential_update.py` 1100–1341(bank 경로) ·
`data_loader.py` 본문 · `tmp/` 의 판정 산출물(별도 조사) · 학습 스크립트 본문(별도 조사).

---

## 1. 호출 사슬 — 배포 합성 경로 (`init_method='auto'`, 혼잡)

```
run.py:791  main()
 └─ run.py:723-766   CLI 화이트리스트 → config dict
 └─ run.py:779       run_estimation(episode, config)
     ├─ run.py:204-206   cfg = DEFAULT_CONFIG.copy(); cfg.update(config)
     │                   ⟹ varpro_steps=400 · learned_init_iters=0 · auto_switch_thresh=0
     │                      · init_method='auto' · lambda_l1=1e-4 · analytic_ensemble_k=3
     ├─ run.py:212       set_seed(42)   (torch.manual_seed + np.random.seed)
     ├─ run.py:215-216   EpisodeDataLoader → cfg['depth'] = loader.depth
     ├─ run.py:226       init_frames = loader.get_init_window(tp_factor=2.0)
     ├─ run.py:233       updater = make_updater(cfg)
     │    └─ sequential_update.py:2029-2032
     │         init_method=='auto' AND analytic_ensemble_k>1 ⟹ MedianEnsembleUpdater(cfg)
     │         └─ :1905  self._primary = SequentialUpdater(cfg)
     │              └─ :399  self.config = config   ⚠**복사하지 않는다**(§5-D 참조)
     │                 :413  self.init_method = cfg.get('init_method', 'v2_konly')  → 'auto'
     ├─ run.py:234       updater.initialize(init_frames, x_range, y_range)
     │    └─ MedianEnsembleUpdater.initialize  :1930
     │         └─ :1936  self._primary.initialize(...)
     │              └─ SequentialUpdater.initialize  :644
     │                   └─ :672  _initialize_inner  :677
     │                        ├─ :697   if init_method == 'auto':
     │                        ├─ :714   실측 게이트: frames 에 GT 가 없으면 → bank_mmse (§2-J)
     │                        └─ :765   self.init_method = self._auto_select_init(frames)
     │                             └─ _auto_select_init  :989
     │                                  ├─ :1016  thresh = cfg['auto_switch_thresh'] = 0
     │                                  ├─ :1026  propose_atoms(...)   ← ★확산 제안 (아래)
     │                                  ├─ :1042  cfg['learned_atoms'] = {kx,ky,energy}  (캐시)
     │                                  ├─ :1043  pool_and_select(...) → n_sel
     │                                  └─ :1049  chosen = 'learned' if n_sel >= 0  ⟹ **항상 learned**
     │                        └─ :770-771  if init_method == 'learned': _initialize_learned(...)
     │                             └─ _initialize_learned  :871  ★핵심
     │                                  ├─ :885  learned_init_params(frames, depth, self.config, seed)
     │                                  │    └─ learned_init.py:327
     │                                  │         ├─ :346  cache = config['learned_atoms']  ← **적중**(라우터가 넣었다)
     │                                  │         ├─ :358  pool_and_select(e_th=0.03, ddk=0.03, max_n=30)
     │                                  │         ├─ :363  peak_anchor_filter   (기본 False → no-op)
     │                                  │         ├─ :370-372  merge_atoms      (기본 None → no-op)
     │                                  │         └─ :389  ls_fit_wphi → tnse.fit_amplitudes_phases(ridge 0.5)
     │                                  ├─ :888-894  _build_model(n) + kx/ky/w/phi 설치 + _build_optimizer()
     │                                  ├─ :897-908  결합 Adam 루프 — **range(0) = 0회 실행**(배포)
     │                                  ├─ :913-914  _adaptive_prune_grow(...)   (§2-D)
     │                                  ├─ :916      _build_tracking_optimizer()  (w_phi: kx/ky 동결)
     │                                  ├─ :921-926  VARPRO 400스텝  ← ★배포 Phase-1 의 유일한 최적화
     │                                  │    └─ varpro.apply_varpro :197 → varpro_refine :109
     │                                  └─ :929      _snap_k_to_grid  (k_snap_dk=None → **None 반환, 무연산**)
     │         └─ :1941  route != 'v2_konly' ⟹ **ensemble_k = 1** (앙상블 미구성, §2-I)
     ├─ run.py:250       updater.predict_field(...)  → :2014 members==1 → _primary.predict_field :1820
     └─ run.py:311-339   프레임 루프 → updater.update(frame)
          └─ :1979 members==1 → SequentialUpdater.update :1341
               ├─ :1355  self._bank is None (합성) → 계속
               ├─ :1368  TRACK_PHASE_PREDICT=False ⟹ **위상 전진 건너뜀**
               ├─ :1378-1380  _frame_buffer (tracking_window=3)
               └─ :1404-1408  tracking_steps=10 × Adam(tracking_lr=0.005) on _compute_loss(window)
                              tracking_mode='w_phi' ⟹ **w 와 φ 만** (kx/ky 동결)
```

### 확산 제안 내부 (`learned_init.propose_atoms:124`)

```
learned_init.py:133-134  device = 'cuda' if torch.cuda.is_available() else 'cpu'   ⚠config['device'] 무시(§6)
learned_init.py:135      _load_cond(ckpt, device)  :98
                          ├─ :108-111  sys.path 에 scripts/ 삽입 → import d3_beamform_cond_diffusion as cd
                          │            ⚠임포트 부작용: cd 가 os.chdir(repo root) 한다 (:25)
                          ├─ :113      stats = torch.load(cd.PRIOR_CKPT)['stats']
                          ├─ :114-115  betas = cd.cosine_betas(cd.T); abar = cumprod(1-betas)
                          └─ :116-117  model = cd.CondSetDenoiser(); load_state_dict(ckpt)
learned_init.py:137-140  _gather_init_obs(frames) → (x, y, z, t_absolute)
learned_init.py:143-146  len>400 이면 np.random.default_rng(subsample_seed=0) 로 400점 부분추출
learned_init.py:147-149  t0 = min(t); Ps[:,2] -= t0   (빔포밍/폴리시용 시간원점 이동)
learned_init.py:151      Bmap = cd._bmap_from_obs(Ps, zf)   → d3_beamform_net.beamform, cd.DEVICE 사용
learned_init.py:152      X = cd.sample(model, Bmap, abar, betas, n=4, seed=diffusion_seed)
                          └─ d3_beamform_cond_diffusion.py:49-61
                               :51  torch.manual_seed(seed)  ⚠**전역 재시드**
                               :54  x = torch.randn(n, M, 3, device=DEVICE)     ⚠장치 의존 스트림
                               :55-60  T=500 스텝 역확산, :60 torch.randn_like
learned_init.py:158      for b in range(4): cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
                          └─ d3_setdiff_posterior_b2.py:64-... Adam(lr=0.02) × 80, **난수 없음**
learned_init.py:159      denorm_atoms(xb, stats) → (kx, ky, a=energy=A²)
learned_init.py:163-164  4샘플 원자 concat 반환 (길이 4×M)
```

**따라서 배포된 학습 Phase-1 을 한 문장으로**:
확산 제안(4샘플 × M원자) → 관측 우도 Adam 폴리시 80회 → 에너지 3% 문턱 + `dedup_dk=0.03` 탐욕 선택(최대 30) →
**ridge 0.5 무가중 LS 로 (w, φ)** → **결합 경사하강 0회** → 에너지 90% 가지치기 → (조건부 성장) →
**신뢰가중 VARPRO 400스텝**(선형부 폐형해, `kx,ky` 만 하강) → 추적은 `w,φ` 만 프레임당 10스텝.

---

## 2. 성분별 실행 판정표

판정 기호: ✅**실행됨** / ⚠**조건부** / ⛔**구조적 무연산**

| # | 성분 | 판정 | 근거 (파일:줄) |
|---|---|---|---|
| A | `auto_switch_thresh=0` ⟹ 합성 항상 학습 | ✅ | `sequential_update.py:178` 상수 `=0`; `:1016` 읽기; `:1049` `n_sel >= 0` 은 항상 참(`n_sel` 은 `len()` ⟹ 음수 불가). **원장 확인.** ⚠단 `:1001` 주석이 적듯 `n_sel==0` 도 learned 로 가서 `learned_init_params:374-382` 축퇴 폴백(랜덤 3성분)에 떨어진다 |
| B | `varpro_steps=400` | ✅ | `sequential_update.py:49` 상수; `:921` 읽기; `:923` `if vp_steps > 0` ⟹ 발화. **`_initialize_learned` 안에서만**(`:918-926`) — 해석적 경로(`_initialize_inner:773-869`)에는 VARPRO 호출이 **없다**. `varpro.py:204-206` 이 bank 경로를 명시적으로 건너뛴다 |
| C | `learned_init_iters=0` | ⛔ | `sequential_update.py:52` 상수 `=0`; `:897` 읽기; `:899 for iteration in range(0)` ⟹ **0회**. ⟹ 배포 학습 Phase-1 에서 `_compute_loss` 를 쓰는 결합 Adam 은 **한 스텝도 안 돈다**. 부수결과: `:894 _build_optimizer()` 가 만든 옵티마이저는 여기서 안 쓰이고, `loss_history` 는 **빈 리스트**로 반환된다(`:937-938`) |
| D | 가지치기 (`adaptive_n=True`, `prune_threshold=0.10`) | ✅ | `:913-914` 호출 → `:1430-1491`. 에너지 90% 유지, 잔차 30% 악화 시 되돌림(`:1475`). **가지치기 반은 항상 돈다** |
| E | 성장 (같은 함수 후반) | ✅**배포에서는 대부분 발화한다 — ★원장 반전** | `:1500-1501` `n_gaussians >= max_n` 조기반환, `:1527` 잔차 `< 0.05 m` 조기반환. **실측 발화율은 혼잡 배포 구성에서 71–73%** (§7-B). ⛔**원장의 "9.2% / 82% 미발화" 는 배포 전(2026-08-22 이전) 구성의 값이고 배포 후에는 거의 반전된다** |
| F | 성장의 **두 번째 구현** | ✅ 살아 있음 | 원장은 `sequential_update.py:1065-1200` 이라 적었으나 **줄번호가 낡았다**. 현재 위치는 **`:1493-1653`**(`MIN_DK` `:1551,:1555,:1589`, 난수 위상 `:1633`, 난수 kx/ky 패딩 `:1619-1620`, `add_gaussians` `:1634`). 호출 지점은 `_initialize_learned:914`(원장의 `:660` 아님) |
| G | `coarse_to_fine.py` 성장 | ⛔ 학습 경로에서 도달 불가 | `_initialize_learned:871-945` 안에 `CoarseToFineSchedule` 이 **없다**. 그 클래스는 `_initialize_inner:779` 에서만 생성되고 학습 경로는 `:770-771` 에서 그 앞으로 빠져나간다. ⟹ 독스트링 `:876-878` *"coarse-to-fine N-staging is bypassed"* 는 **이 파일에 대해서만 참** |
| H | `learned_dedup_dk = 0.03` | ✅ | `learned_init.py:344` 기본값; `:359` `pool_and_select(dedup_dk=…)`; `:184-185` 사용. ⚠비교가 **유클리드가 아니라 `|Δkx|<dk AND `|Δky|<dk` 축별 상자**(`:184`) — 대각선 방향 분리는 `0.03·√2 = 0.042` 다. 라우터도 같은 값을 쓴다(`sequential_update.py:1046`) |
| H2 | `learned_konly` | ⛔ 기본 무연산 | `learned_init.py:384` `config.get('learned_konly', False)` — `DEFAULT_CONFIG`·yaml·CLI **어디에도 없다**. False 면 `:389 ls_fit_wphi` 로 간다 |
| H3 | `merge_atoms` / `learned_merge_frac` | ⛔ 기본 무연산 | `learned_init.py:370-372` `if merge_frac:` — 기본 `None` ⟹ 비트 동일. `merge_atoms:208-209` 도 자체 방어. **`DEFAULT_CONFIG`·yaml·CLI 미노출** |
| H4 | `peak_anchor_filter` / `learned_peak_anchor` | ⛔ 기본 무연산 | `learned_init.py:363` 기본 False. 미노출 |
| I | `analytic_ensemble_k = 3` 앙상블 | ⛔**구조적으로 죽었다 — 원장 확인** | `sequential_update.py:1941` `if self.k_requested <= 1 or route != 'v2_konly' or real_gate:` → `ensemble_k = 1`. 임계 0 이면 route 는 항상 `'learned'` ⟹ **조건 성립 ⟹ 멤버 1개**. 값 3 은 `DEFAULT_CONFIG:64`·yaml `:56` 에 남아 있으나 **작동하지 않는다**. ★독립 확증: `wavesplat/tests/test_shipping_defaults.py:52-53` 이 같은 사실을 이미 기록해 두었다 |
| J | 실측 게이트 | ✅ (실측에서) | `:714-715` GT 부재 감지 → `:723` `real_estimator=='bank_mmse'` → `:751 return self._initialize_bank(...)`. ⟹ **실측은 학습 경로에 절대 들어가지 않는다**(`real_domain_gate=False` 로만 우회) |
| K | `tracking_mode='w_phi'` | ✅ | `:525-531`: `kx.requires_grad_(False)`, `ky.requires_grad_(False)`, `w`·`phi` True, `lr=tracking_lr=0.005`. `update:1404-1408` 이 프레임당 10스텝. ⟹ **학습이 놓은 파수는 추적 내내 고정**이고, 추적이 미는 것은 진폭과 위상뿐 |
| K2 | `TRACK_PHASE_PREDICT` 위상 전진 | ⛔ | `:311` `= False` ⟹ `:1368` 조건 거짓 ⟹ `phi += ω·dt` 실행 안 됨 |
| L | `use_confidence=True` | ⚠**부분 실행 — 원장 정정 필요** | 두 소비처: ①`_compute_loss:570-572` ②`varpro.apply_varpro:218-220 → varpro_refine:117-118 sqrt_c`. ★**배포 학습 Phase-1 에서 ①은 0회 호출**(`learned_init_iters=0`, C 항목) ⟹ **Phase-1 에서 신뢰가중은 오직 VARPRO 를 통해서만 작동**한다. ①은 추적(`:1406`)과 성장 미세조정(`:1646`)에서만 산다 |
| L2 | 실측에서 `confidence ≡ 1`? | ⚠**미확인(코드상으로는 가능)** | 실측은 `real_data.py:297-301 observation_weights(inc, dist, d_ref=mount.d_ref)`, `d_ref` 기본 **5.0 m**(`real_data.py:35`). `confidence.py:38-41` 은 `cos θ / (d/d_ref)` 를 `[0,1]` 로 **클램프**하므로 `d < d_ref·cosθ` 인 모든 점이 정확히 1.0 이 된다. 고정 관측소 스테레오에서 그럴 개연성은 높으나 **나는 실측 h5 를 열지 않았다** ⟹ 수치 확인은 미이행. ★그러나 **실측 판정과는 무관하다**: 실측은 게이트(J)가 `bank_mmse` 로 보내고, `bank_mmse` 는 `confidence` 를 아예 읽지 않는다(§7-7) |
| M | `k_snap_dk` | ⛔ 기본 무연산 | `:189` `K_SNAP_DK_DEFAULT = None`; `:954-956` `if not dk: return None`. 모델을 건드리지 않으므로 비트 동일. **CLI·yaml 미노출** |
| N | `lambda_l1 = 1e-4` | ⚠**부분 실행 — ★새 발견** | `_compute_loss:589-597` 에서만 쓰인다. B/C 항목과 결합하면: 배포 학습 Phase-1 에서 `_compute_loss` 호출 0회 ⟹ ★**L1 벌점이 학습 경로의 초기화에는 전혀 걸리지 않는다.** 사는 곳은 **추적 루프**(`:1406`)와 성장 미세조정(`:1646`)뿐. ★그리고 **VARPRO 의 축약 목적함수에는 λ 가 없다**(`varpro.py:82-88` 이 스스로 그렇게 적었다; `varpro.py` 에 `lambda` 문자열 0회, `VP_RIDGE_DEFAULT=1e-8` 만) |
| O | `use_dispersion=True` | ✅ | `spectral_gaussian.py:99-101` · `varpro.py:61-70` 이 같은 `dispersion_omega` 를 부른다. ⟹ ω 는 자유 파라미터가 아니다 |
| P | `max_steepness=0.15` | ✅ | `varpro.apply_varpro:242-246` (초기화 끝) · `update:1413` (매 프레임) |
| Q | `n_gaussians=[3,8,15]` 스테이지 | ⛔ 학습 경로 무연산 | `self.stages` 는 `:454-458` 에서 설정되지만 `_initialize_learned` 는 `CoarseToFineSchedule` 을 만들지 않고 `n = len(제안 원자)` 로 곧장 모델을 짓는다(`:885-888`) |
| R | `amplitude_refit` | ⛔ 기본 무연산 | `run.py:463` `if cfg.get('amplitude_refit', False)` — 이 키는 `DEFAULT_CONFIG`(`run.py:29-114`)에 **없다**. yaml·CLI 에도 없다 |
| S | `neural_corrector` | ⛔ 기본 무연산 | `:452` 기본 None ⟹ `_build_model:481` 조건 거짓, `predict_field:1833` 조건 거짓 |
| T | `_ls_w_update` | ⛔**완전 사망 코드** | `sequential_update.py:1655` 정의. 저장소 전체(`grep -rn`)에서 호출 지점 **0곳**(스냅샷 `tmp/rx2_init_2026-08-05/snap/…` 의 사본 제외) |

---

## 3. 학습된 것은 정확히 무엇인가

*(체크포인트 파일 사실·학습 코퍼스는 별도 조사 결과를 §3.2/§3.3 에 반영. 아래 §3.1 은 내가 직접 코드에서 확인한 것.)*

### 3.1 코드가 말하는 것 (내가 직접 확인)

- **배포 체크포인트 상수**: `learned_init.py:61-63`
  `DEFAULT_CKPT_REL = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'`,
  `DEFAULT_CKPT_SHA256 = '636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9'`,
  경로는 `_repo_root()`(`:40-42`) 로 절대화된다.
  ⚠**`ckpt_sha256()`(`:69-77`) 은 정의만 되어 있고 `_load_cond` 는 그것을 부르지 않는다**
  (`:98-120` 에 sha 검사 없음) ⟹ **출처 검증이 자동으로 걸리지 않는다.**
- **두 번째 체크포인트**: `_load_cond:113` 이 `cd.PRIOR_CKPT` 를 읽는다 =
  `scripts/d3_beamform_cond_diffusion.py:40` `'results/direction3/setdiff_prior.pt'` 의 `['stats']`.
  ⟹ **학습 경로는 체크포인트를 하나가 아니라 둘 쓴다**(가중치 + 정규화 통계).
- **모델**: `cd.CondSetDenoiser`(`d3_beamform_cond_diffusion.py:60`), 조건부 인코더 `CondEncoder`(`:44`).
- **입력**: `Bmap = beamform(P[None], z[None])` — `d3_beamform_net.py:31-32` 에서
  `KLIM = 0.6`, `NK = 64`, `M_OBS = 400` ⟹ **관측 400점에서 만든 64×64 빔포밍 파워맵**.
  관측 상한 400 은 `learned_init.py:65 M_OBS_CAP = 400` 이 미러한다.
- **출력**: `sample(...)` → `(n, M, 3)` 정규화 원자, `n = learned_nsamp = 4`,
  `M = 32`(`d3_setdiff_prior.py:25`), 채널 3 = (kx, ky, log-energy) 정규화.
  확산 스텝 `T = 500`(`d3_setdiff_prior.py:28`), `clamp(-1,1)`(`sample:61`).
  ⟹ **원자 풀 = 4 × 32 = 128 개**가 `pool_and_select` 로 들어가 최대 30 개로 줄어든다.
- **역정규화**: `d3_setdiff_prior.py:67-70 denorm_atoms` —
  `kx = Z[...,0]·K_LIM`, `ky = Z[...,1]·K_LIM`, 에너지는 `(Z+1)/2·(hi_e−lo_e)+lo_e` 후 exp 계열.
  `stats` 키: `K_LIM`, `hi_e`, `lo_e` (최소한 이 셋).
- **학습된 것이 아닌 것**: 폴리시(`polish_atoms`, Adam 80회, 관측 우도)·`pool_and_select`·
  `ls_fit_wphi`·VARPRO·가지치기/성장·추적은 **전부 비학습 고전 연산**이다.
  ⟹ ★**"학습"은 파이프라인에서 딱 한 자리다: (kx, ky, energy) 집합의 제안.**

### 3.2 체크포인트 파일 사실 (실측)

배포 경로가 **실제로 여는 `.pt` 는 정확히 둘**이다.

| # | 절대경로 | bytes | mtime | sha256 | git |
|---|---|---|---|---|---|
| 1 | `/home/vexit/wavesplat/wavesplat/results/direction3/beamform_cond_diffusion_mixed_v1.pt` | 5,856,104 | 2026-07-27 00:32:38 | `636c72a7…65239d9` | ✅추적됨 |
| 2 | `/home/vexit/wavesplat/wavesplat/results/direction3/setdiff_prior.pt` | 2,279,733 | 2026-06-27 16:06:50 | `5e8418cc…24efbb34` | ⛔**미추적** |

- ✅**#1 의 sha256 은 `learned_init.py:62` 의 핀과 일치한다(MATCH).** 파일은 학습 종료 이래 불변.
- ⛔★**#2 는 `/home/vexit/wavesplat/.gitignore:81`(저장소 루트) 의 `*.pt` 에 걸려 git 에 없다.**
  ⟹ ***배포된 합성 경로가 저장소에 없는 파일 하나에 의존한다.***
  ⚠단 배포가 쓰는 것은 그 안의 `['stats']` **스칼라 넷**뿐이고(`learned_init.py:113`),
  prior 망 가중치는 배포 경로에서 **읽히지 않는다** ⟹ 위험은 "다른 망"이 아니라 **정규화 상수 소실**이다.
  실측 값: `{'lo_e': -15.656161308288574, 'hi_e': -1.0779345035552979, 'K_LIM': 1.1, 'M': 32}`.
- ⚠**`ckpt_sha256()`(`:69-77`) 은 정의만 있고 `_load_cond` 가 부르지 않는다** ⟹ 출처 검증 자동 실행 0회.

### 3.3 무엇이 들어가고 무엇이 나오는가 (텐서 형상, 실측)

- **입력** `Bmap`: `beamform(P (1,400,3), z (1,400))` → **`(1, 64, 64)` 실수 파워맵 1채널**.
  `d3_beamform_net.py:31-32 KLIM=0.6, NK=64`, `:39` 분산관계로 초점, `:53-54` 최대값 정규화 후 `log1p(·×50)`.
- **망** `CondSetDenoiser`(`d3_beamform_cond_diffusion.py:60-83`): 조건 인코더가 `Bmap` → **`(B,256,128)` 토큰**,
  원자 `(B,32,3)` 에 self-attn(원자 간) + cross-attn(원자→맵) 블록 4개(heads 4, d=128).
  출력 `(B,32,3)` = **ε 예측**. 실측 **101 텐서 / 1,454,723 파라미터**.
- **샘플링**: `sample()` = 조건부 ancestral DDPM, `T=500`, `n=4` ⟹ `(4, 32, 3)`, `clamp(-1,1)`.
- **역정규화** `denorm_atoms`(`d3_setdiff_prior.py:67-72`):
  `kx = Z[...,0]·K_LIM`, `ky = Z[...,1]·K_LIM`, `a = exp(le)` = **에너지 = A²**(진폭 아님).
- ★**ω 와 φ 는 망에서 나오지 않는다.** ω 는 분산관계로, φ 는 `ls_fit_wphi` 로 따로 만든다.
  ⟹ ***학습된 것은 (kx, ky, 에너지) 집합의 조건부 사전 하나뿐이다.***

★**새 발견 — 사전의 하드 표현 한계**: `sample` 이 `clamp(-1,1)`(`:61`) 하고 `K_LIM=1.1` 로 선형 역정규화하므로
**제안 파수는 축별 `|kx|,|ky| ≤ 1.1 rad/m` 로 구조적으로 갇힌다.** 폴리시가 `clamp(-1.1,1.1)`
(`d3_setdiff_posterior_b2.py:55`)로 10% 넓혀 **최대 1.21** 까지 간다.
⟹ **사전이 제안할 수 있는 최단 파장은 축별 λ ≈ 5.2 m, 대각 |k| ≤ 1.71 에서 λ ≈ 3.7 m.**
⚠단 이것은 **초기화의 한계**다 — VARPRO(`varpro.py:109-165`)는 `kx,ky` 를 자유롭게 움직이고 클램프가 없다.
⚠또한 **조건화 맵의 대역(`KLIM=0.6`)이 사전의 대역(`K_LIM=1.1`)보다 좁다** — 두 상수가 다른 파일에 있고
서로 다르다. 이것이 의도인지 사고인지는 **미확인**.

### 3.4 어떤 자료로 학습됐는가 (실측)

| 대상 | 학습 스크립트 | 산출 |
|---|---|---|
| prior(`stats` 출처) | `scripts/d3_setdiff_prior.py:116-153 train()` | `setdiff_prior.pt` |
| 구 조건부 denoiser | `scripts/d3_beamform_cond_diffusion.py:127-154 train()` | `beamform_cond_diffusion.pt` |
| ★**배포본 `_mixed_v1`** | **`tmp/mixed_retrain_2026-07-26/train_mixed.py`** + 장면 생성기 **`gen_mixed.py`** | `beamform_cond_diffusion_mixed_v1.pt` |

**★핵심: 학습 장면은 에피소드 파일에서 오지 않는다. 에피소드에서 오는 것은 LiDAR 기하 `(x,y,t)` 뿐이다.**

- **(i) 파장 장면 = 매 스텝 새 합성 추첨** — `tmp/mixed_retrain_2026-07-26/gen_mixed.py:270-351 sample_scene()`
  - 장면 시드 `rng.integers(SEED_LO, SEED_HI)`, `:88 SEED_LO, SEED_HI = 2_000_000_000, 2_147_483_647`
  - `:79 ANCHORS = ((0.3,4.0),(0.6,5.5),(1.2,7.0),(2.0,9.0))` = `wavesim/pipeline.py:35-40` 의 **BF2–BF5 (Hs,Tp) 와 동일**,
    `:80-83 ANCHOR_FRAC=0.5 / ANCHOR_JITTER=0.15`, 나머지 절반은 `HS_BOX=(0.25,3.2)`·`TP_BOX=(3.8,11.5)`
  - `:74-76` 층화: 혼잡 0.30(`nc∈{15,30}`) · 희소 0.20(`nc∈{1,2}`) · 중간 0.10(`nc=5`) ·
    연속 완만 0.10 · 연속 강 0.15 · 중간확산 0.15
  - `:54 DEPTH=50.0`, `:56 M_OBS=400`, `:57 NOISE_SIGMA=0.02`, `:341 Tobs = U[1.9,2.3]·Tp`
  - ⛔**관측 렌더는 엄격히 선형**: `:343-344 elevation(..., nonlinear=False)`
- **(ii) 관측 기하 = 도너 에피소드 14개의 실제 LiDAR 창 재생** — `gen_mixed.py:66-71 DONOR_FILES`, `:92-143 GeometryBank`
  - `data_finemesh_extras_2026-07-24/episode_BF{6,7}_{1,5,15,30}comp_seed0_uniform.h5` (8개)
  - `data_finemesh_2026-07-23/episode_BF{3,4}_5comp_seed{2,3,4}_slope.h5` (6개)
  - `:105-124` 는 `lidar_points`·`lidar_valid_mask`·`timestamps` 만 읽고 **`z` 는 읽지 않는다**(항상 재렌더).
- **(iii) 학습 하이퍼(실측 `train_curve.json` 의 `args`)**: `steps 40000`, `bsz 32`, `lr 2e-4`,
  **`seed 20260726`**, `train_mixed.py:92` **from scratch**(미세조정 아님), AdamW + cosine.
  실측 wall **4711.7 s**, 최종 `train 0.05955 / val 0.08619`.
  `VAL_SEED=987654321` 홀드아웃은 `train_mixed.py:43` 축자 *"reporting only — never selects a checkpoint"*.
- **(iv) ⛔동결 검사 실패**: `sha256sum -c tmp/mixed_retrain_2026-07-26/FROZEN.sha256` → **3/3 FAILED**
  (`PREREG.md`·`gen_mixed.py`·`train_mixed.py` 전부 기록 해시와 불일치).
  ⟹ ★***v1 을 만든 코드 바이트는 디스크에 남아 있지 않다.*** 변경은 문서화돼 있고(라운드 2/3 용
  `target_mode` 추가) 세 파일 mtime 이 v2 학습 직전이므로 **v1 이후 편집**으로 보이지만,
  **v1 시점 판본의 복원 가능성은 미확인**이다.

### 3.5 ★원장의 두 문장 검증 — *"담지자는 체크포인트"* / *"평가와 같은 분포로 학습"*

**(A) "담지자는 프레임워크가 아니라 체크포인트"** → ✅**코드가 이를 강하게 지지한다.**
학습된 성분은 파이프라인에서 단 한 자리(제안)이고(§3.3), 그 자리가 파일 **한 개**(`…mixed_v1.pt`)에 있으며,
그 파일은 2026-07-27 이후 **불변**이고 sha 가 핀과 일치한다.
★그리고 저장소가 이미 이 담지를 **개입으로 측정했다**: `B-8` `V2K` 팔(학습을 빼고 해석적으로 대체)이
실측에서 14/36 = 무기술.

**(B) "평가와 같은 분포로 학습"** → ⚠**절반만 참이다. 정확한 문장은 「in-family, out-of-realisation」이다.**
저장소 자신의 사전등록이 축자로 그렇게 적었다(`tmp/mixed_retrain_2026-07-26/PREREG.md §2.2`).

- ✅**시드 축은 분리가 검증됐다**: 학습 시드 `[2×10⁹, 2³¹)` vs 평가 시드 **전부 < 2000**
  (`data_finemesh_2026-07-23` 0–4 · `heldout40` 777/888 · `confirm2` 1111/1212 ·
  `s8/s9_crowded` 32–39/48–55 · `HOSGATE` 1581–1584·1633–1640).
  도너 14개도 평가 목록과 **파일 교집합 0**.
- ✅**파랑 분포는 의도적으로 정합**: 앵커 (Hs,Tp) 가 BF2–BF5 와 동일, `n_components` 층이 `{1,2,5,15,30}` 로
  평가의 혼잡(15,30)·희소(1,2)와 **정확히 일치**.
- ⛔★**그러나 문자 그대로는 세 축에서 반증된다**:
  1. **`random_amplitudes` / `random_frequencies` / `decorrelate_stream`** — 최근 평가 코퍼스는 셋 다 **True**
     (`data_s8_crowded_2026-08-10/_manifest.json`; `tmp/HOSGATE_2026-08-13/gen_hos.py:169`).
     학습 추첨은 `gen_mixed.py:304-305` 가 이 셋을 **안 넘겨 기본 False**(`wave_generator.py:144-146`).
     ⟹ **진폭·주파수 추첨 방식이 학습과 평가에서 다르다.** 그리고 이 규약의 코퍼스는 **08-10 이후 산물** =
     체크포인트(07-27)보다 **나중**이다.
  2. **비선형성** — 학습 관측은 엄격히 선형. `B-11`/`B-12`/`B-16`/`B-22`/`M4` 의 HOS 팔은 2차 비선형
     ⟹ **HOS 절반은 학습 분포 밖**이다(저장소가 이미 *"단서 ①선형 GT 가 제거된다"* 로 쓴 사실과 정합).
  3. **기하 도너의 해상상태** — BF6/BF7(평가가 안 쓰는 상태) + BF3/BF4 **slope** raydrop.
     평가는 BF4/BF5 **uniform**. `gen_mixed.py:61-65` 주석이 이 선택을 **의도**라 적었다.

⟹ ★**권고 문안**: *"체크포인트는 평가와 **같은 계열**(같은 (Hs,Tp) 앵커·같은 성분수 층)에서 학습됐고,
시드·잡음 추첨 규약·비선형성·기하 도너 해상상태는 **다르다**."*
⛔*"같은 분포"* 도 ⛔*"완전한 out-of-sample"* 도 둘 다 과장이다.

### 3.6 ⚠배포 인가와 배포 구성의 불일치 (실측)

- `learned_init.py:44-53` + `tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §1`: v1·v3 **양쪽이 사전등록 관문 전부 통과**,
  **동결 tie-break**(혼잡 짝 ≥11/16)가 **정확히 11/16** 으로 v1 을 골랐다.
  저장소가 스스로 *"결정 규칙이지 우월성의 증거가 아니다(p=0.105)"* 라 못 박았다.
- ⚠**자체 게이트에서 v1 은 `G2a` 를 FAIL 했다**(`tmp/mixed_retrain_2026-07-26/RESULT.md §4`: 바 ≤6, v1 = **12.0 FAIL**, v3 = 6.0 PASS).
- ⛔★**인가된 구성은 `auto_v1_m3` = v1 + 라우터 th=8 + guard 은퇴 + median-of-3 이다.**
  현재 배포는 **th=0**(`sequential_update.py:178`)이고 그 결과 **median-of-3 앙상블이 구조적으로 죽었다**(§2-I).
  ⟹ ***인가 문서가 서술하는 세 요소 중 둘이 지금 배포되고 있지 않다.***
  (변경 근거 W11/W13/W14·W15 는 존재하므로 "재측정 없이 바뀌었다"는 주장은 아니다.
  주장은 **인가 문서를 배포 근거로 인용할 수 없다**는 것이다.)

---

## 4. 학습 경로 노브 전수 목록

세 표면을 대조했다:
`run.py DEFAULT_CONFIG`(`:29-114`) / `configs/estimation.yaml` + `load_yaml_config`(`run.py:125-188`) / CLI `main()`(`:668-766`) /
그리고 `sequential_update`·`learned_init`·`varpro` 가 실제로 읽는 키.

★**중요한 구조**: `run_estimation(h5, config=…)` 은 `:204-206` 에서 **화이트리스트 없이** `cfg.update(config)` 한다.
⟹ **파이썬 API 로는 모든 키가 도달 가능**하고, 막히는 것은 **CLI 와 YAML 두 표면뿐**이다
(`load_yaml_config` 은 `cfg = {}` 로 시작해 아는 키만 옮긴다 `:131-188` ⟹ **모르는 키는 조용히 버려진다**).

### ① 배포 노출 (DEFAULT_CONFIG 에 있음)

| 키 | 배포값 | yaml | CLI | 학습 경로에서의 효과 |
|---|---|---|---|---|
| `init_method` | `'auto'` | ✅ `:156` | ✅ `--init_method` | 라우터 진입. 명시하면 실측 게이트가 사라진다(`:686-696`) |
| `auto_switch_thresh` | `0` | ✅ `:144` | ✅ | 0 = 항상 learned |
| `varpro_steps` | `400` | ✅ `:140` | ✅ | Phase-1 유일 최적화 |
| `learned_init_iters` | `0` | ✅ `:142` | ✅ | 0 = 결합 적합 없음 |
| `lambda_l1` | `1e-4` | ✅ `:138` | ✅ | ⚠추적에만 (표 §2-N) |
| `lambda_smooth` | `1e-4` | ✅ `:139` | ✅ | ⚠추적에만 (같은 사유) |
| `lr` | `0.01` | ✅ `:136` | ✅ | ⚠**학습 경로에서 실질 무연산**: `_build_optimizer` 는 만들어지지만 결합 루프가 0회. 성장 미세조정(`:1643-1648`)에서만 산다 |
| `tracking_mode` / `tracking_lr` / `tracking_steps` / `tracking_window` | `w_phi` / `0.005` / `10` / `3` | ✅ | ✅ (window 는 CLI 있음) | 추적 |
| `adaptive_n` / `prune_threshold` / `grow_residual_threshold` / `max_n` | `True` / `0.10` / `0.05` / `30` | ✅ `:174-178` | `--adaptive_n`, `--max_n` 만 | 가지치기·성장 |
| `use_dispersion` / `use_confidence` | `True` / `True` | ✅ `:172-173` | ⛔ | ω 구속 / VARPRO 가중 |
| `n_gaussians` | `[3,8,15]` | ✅ `:133` | ✅ | ⛔학습 경로 무연산 (§2-Q) |
| `depth` | `50.0`(에피소드가 덮어씀 `run.py:216`) | ✅ `:134` | ⛔ | ⚠§5-C 이중 출처 |
| `analytic_ensemble_k` | `3` | ✅ `:158` | ⛔ | ⛔구조적으로 죽음 (§2-I) |
| `k_snap_dk` | `None` | ⛔ | ⛔ | 표현 절제 노브 — **파이썬 API 전용** |
| `seed` | `42` | ✅ `:186` | ✅ | `learned_diffusion_seed` 기본값이 된다 |
| `device` | `'cpu'` | ✅ `:184` | ⛔ | ⚠**제안 경로가 무시한다** (§6) |
| `real_estimator` / `real_bank_ensemble_k` / `bank_refit_every_s` | `bank_mmse` / `3` / `None` | ✅ / ✅ / ✅`:168` | `--bank_refit_every_s` 만 | 실측 전용 |
| `bank_refit_async` | `False` | ⛔ | ⛔ | 실측 전용, **파이썬 API 전용** |

### ② 연구 전용 — 도달 가능하나 미노출 (파이썬 API `run_estimation(config=…)` 또는 직접 생성자로만)

`learned_ckpt` · `learned_nsamp` · `learned_diffusion_seed` · `learned_energy_thresh` ·
`learned_dedup_dk` · `learned_konly` · `learned_atoms` · `learned_merge_frac` ·
`learned_peak_anchor` · `learned_anchor_frac` · `learned_anchor_sigfrac` ·
`auto_guard` · `auto_guard_ratio` · `real_domain_gate` · `init_k_lr_scale` ·
`max_steepness` (yaml `:90` 에 **쓰여 있으나 `load_yaml_config` 이 안 읽는다** — §5-E) ·
`amplitude_refit` · `neural_corrector_path`(CLI `--neural_corrector` 로는 노출) ·
`bank_dk_frac` · `bank_k_set` · `bank_max_obs` · `bank_memory_s` ·
`loss_thresholds` · `max_iters_per_stage`(yaml `:181-183` 있음).

⚠**`learned_*` 열한 개 전부가 CLI·YAML 양쪽에서 도달 불가**다. 학습 경로의 노브가
배포 설정 표면에 **하나도 나와 있지 않다** — `learned_init_iters` 만 예외이고 그건 이름만
`learned_` 이지 실제로는 `sequential_update` 의 결합 루프 카운트다.

### ③ 도달 불가 (설정으로 못 바꿈 = 모듈 상수)

`SCALE_INVARIANT_GROW`(`:282`) · `SCALE_INVARIANT_L1`(`:291`) · `TRACK_PHASE_PREDICT`(`:311`) ·
`coarse_to_fine.GROW_RANDOM_PHI`(`:18`) · `coarse_to_fine.SCALE_INVARIANT_ADVANCE`(`:27`) ·
`initializer.MIN_DK = 0.15` / `RECOMPUTE_MIN_DK = 0.02` ·
`learned_init.M_OBS_CAP = 400`(`:65`) · `propose_atoms` 의 `polish_iters=80`·`subsample_seed=0`
(`learned_init.py:125` 기본값 — `learned_init_params:355-356` 이 **둘 다 안 넘긴다** ⟹ 설정에서 도달 불가) ·
`varpro.VP_LR_DEFAULT = 2e-3` / `VP_RIDGE_DEFAULT = 1e-8`(`varpro.py:34-35`;
`_initialize_learned:925` 가 `steps` 만 넘기고 `lr`·`ridge` 는 안 넘긴다) ·
`ls_fit_wphi` 의 `lambda_reg=0.5`(`learned_init.py:322` 하드코딩) ·
`d3_setdiff_prior.M = 32` / `T = 500` · `d3_beamform_net.KLIM = 0.6` / `NK = 64` ·
`pool_and_select` 의 `max_n` 기본 30 (설정 `max_n` 이 덮음) ·
환경변수 게이트 `RX1_*`(`:327-347`) · `RX2_GROW_*` · `S5_WINDOW` · `S2_ALPHA` ·
`S8_L1_TRACK_ONLY` / `S8_L1_PHASE1_ONLY`.

★**연구 축 제안 시 주의**: VARPRO 의 `lr`·`ridge` 와 `ls_fit_wphi` 의 `lambda_reg`,
확산의 `polish_iters` 는 **배포 경로에서 설정으로 도달할 수 없다**. 이 넷을 건드리는 배치는
패키지 수정이나 몽키패치가 필요하고, 그것은 이 저장소가 `varpro.py` 로 대가를 치른 부류다
(`varpro.py:3-4`: *"몽키패치로만 존재했고 패키지에는 없었다"*).

---

## 5. 같은 기능이 두 곳(이상)에 구현된 것

### A. ★**(w, φ) 최소제곱 적합이 다섯 곳** — 이 중 **둘이 배포 학습 경로에서 연달아 돈다**

| # | 위치 | 정규화 | 가중 | 상태 |
|---|---|---|---|---|
| 1 | `model/tnse.py:176-218 fit_amplitudes_phases` (← `learned_init.py:309-323 ls_fit_wphi`) | **ridge `λ=0.5`** (`:322` 하드코딩) | **무가중** | ✅배포 학습 경로 |
| 2 | `optimizer/varpro.py:89-102 reduced_objective` + `:48-54 ab_to_wphi` | **ridge `1e-8`** (`:35`) | **`sqrt(confidence)`** (`:117-118`) | ✅배포 학습 경로 |
| 3 | `sequential_update.py:1699- amplitude_refit` | 없음 (`lstsq`) | 무가중 | ⛔기본 미사용 (§2-R). **위상 부호 결함이 있던 자리** |
| 4 | `sequential_update.py:1655-1697 _ls_w_update` | 없음 | 무가중 | ⛔**완전 사망** (호출 0곳) |
| 5 | `initializer.py:417 _linear_projection` | — (단일 성분 사영) | 무가중 | ⚠성장 분기에서 사용 (`:1582`) |

★**새 발견 — 이것이 판정에 닿는다.** 1번과 2번은 **같은 선형 부분문제**를 **다른 정규화·다른 가중**으로
푼다. 배포 순서는 `1 → (가지치기) → 2` 이고, **2번이 1번의 (w, φ) 를 완전히 덮어쓴다**
(`varpro.py:231, 236-237`). 그런데 ★**가지치기가 1번의 결과를 본다**:
`_adaptive_prune_grow:1432` 가 `energy = self.model.w ** 2` 로 자르는데, 그 시점의 `w` 는
**ridge 0.5 무가중 적합의 진폭**이다.
⟹ ***어떤 원자가 살아남는가는 `λ_reg=0.5` 가 결정하고, 최종 진폭은 `ridge=1e-8` 신뢰가중이 결정한다.***
이 두 값은 서로 다른 파일에 하드코딩돼 있고 어느 쪽도 설정 표면에 없다.
⛔**나는 이 이중성이 정확도에 영향을 준다고 주장하지 않는다** — 그것은 측정된 적이 없다.
주장하는 것은 **구조**뿐이다.

### B. ★**성장(residual beamform) 이 두 파일에 두 벌**

- `coarse_to_fine.py:192-… _beamform_residuals` + `:158-190` (패딩·위상·`add_gaussians`)
- `sequential_update.py:1543-1640` (같은 탐욕, 같은 `MIN_DK` 억제, 같은 `RX2_GROW_*` 게이트)

**원장이 이미 아는 중복이지만 줄번호가 낡았다**(원장 `:1065-1200`·`:1188`·`:173` → 현재
`:1543-1640`·`:1633`·`coarse_to_fine.py:177`). **학습 경로에서는 `sequential_update` 쪽만 도달 가능**(§2-G).

### C. **`depth` 가 두 곳에서 읽힌다**

`sequential_update.py:400 self.depth = config.get('depth', 50.0)` 와
`:965 float(self.config.get('depth', 50.0))`(`_snap_k_to_grid` 안). `run.py:216` 이 둘 다 채우므로
현재는 일치하지만, `updater.depth` 를 사후에 바꾸는 코드가 있으면 갈라진다.
★이것은 원장이 기록한 **`B-2` 수심 17→30 사고와 같은 부류**(상수가 두 자리에 있고 한쪽만 갱신)다.

### D. ★**`self.config` 를 복사하지 않는다 + 라우터가 그 dict 에 쓴다 — 잠재 오염 경로**

`sequential_update.py:399 self.config = config`(복사 없음) →
`_auto_select_init:1042 cfg['learned_atoms'] = dict(kx=…, ky=…, energy=…)`.
⟹ **호출자의 config dict 에 원자가 주입된다.**
`run.py` 는 안전하다(`:204` 가 매 호출 새 dict 를 만든다). 그러나
**같은 config 객체를 에피소드 루프에서 재사용하는 러너는 두 번째 에피소드부터 첫 에피소드의 원자를 쓴다.**
`grep -rln learned_atoms tmp/` 가 러너 여러 개를 찾았다(`W4_SOLVER_2026-08-19/run_w4.py`,
`B22_PRIOR_ABLATION_2026-08-27/run_b22.py`, `confirm2_2026-07-27/run_arms_confirm2.py`,
`heldout40_2026-07-26/…`, `unified_guard_2026-07-26/…`, `learned_only_2026-07-26/…`).
⛔**나는 이 러너들이 실제로 오염됐다고 주장하지 않는다** — 각 러너가 dict 를 새로 만드는지 열어보지 않았다.
★**주장하는 것은 기제의 존재와 검사 방법**이다: 러너마다 `config` 객체가 에피소드마다 새로 만들어지는지 확인하고,
행에 `n_selected` 와 원자 해시를 실어라.

### E. **`max_steepness` 가 yaml 에 있는데 로더가 읽지 않는다**

`configs/estimation.yaml:90 max_steepness: 0.15` 인데 `load_yaml_config`(`run.py:135-170`) 에
그 키가 없다 ⟹ **yaml 로 못 바꾼다**. 값이 코드 기본값(`:448`)과 같아서 거동은 일치하지만,
★**`bank_refit_every_s` 가 겪은 "죽은 미러" 와 정확히 같은 부류**다(`estimation.yaml:85` 이 스스로 그 사고를 적어 두었다).

### F. **장치 정책이 일곱 자리에 손복사돼 있다** — §6 참조.

---

## 6. 장치 의존성 — 정확한 진입점

### 어디서 들어오는가

`torch.cuda.is_available()` 로 장치를 **자기 혼자 고르는 자리가 임포트 사슬에 일곱 개** 있다:

| # | 파일:줄 | 시점 | 학습 경로 관여 |
|---|---|---|---|
| 1 | `wavesplat/optimizer/learned_init.py:133-134` | **호출 시점** (`propose_atoms`) | ✅ 모델·`abar`·`betas`·`Pt`·`zt` |
| 2 | `scripts/d3_beamform_cond_diffusion.py:39 DEVICE` | **임포트 시점** | ✅ `sample:54,56,60`, `_bmap_from_obs:66-67` |
| 3 | `scripts/d3_beamform_net.py:29 DEVICE` | 임포트 시점 | ✅ `beamform:36,62-64` |
| 4 | `scripts/d3_setdiff_posterior.py:23 DEVICE` | 임포트 시점 | ✅ `polish_atoms` 가 쓰는 `_to_t`(`d3_setdiff_posterior_b2.py:34-35`)·`loglik_atoms` |
| 5 | `scripts/d3_setdiff_prior.py:22 DEVICE` | 임포트 시점 | ⚠`denorm_atoms` 는 장치 무관하나 모듈은 로드된다 |
| 6 | `scripts/d40_learned_acquisition.py` (grep 1건) | 임포트 시점 | ⚠`cd` 가 임포트한다(`d3_beamform_cond_diffusion.py:32`) |
| 7 | `wavesplat/optimizer/bank_mmse.py:41 _DEV` | 임포트 시점 | ⛔실측 전용 |

### 왜 결과가 갈리는가 (기제, 파일:줄)

1. **`propose_atoms` 는 `config['device']` 를 보지 않는다.** `learned_init.py:124-125` 서명의
   `device=None` 기본값이고, 두 호출 지점 모두 인자를 안 준다:
   `learned_init.py:355-356`(`learned_init_params`) 와 `sequential_update.py:1026-1030`(라우터).
   ⟹ `:133-134` 자동 감지가 이긴다. **배포 기본값 `device='cpu'`(`run.py:112`) 는 제안 경로에 도달하지 않는다.**
2. **난수는 `cd.DEVICE` 위에서 뽑힌다.** `d3_beamform_cond_diffusion.py:51` `torch.manual_seed(seed)` →
   `:54 torch.randn(n, M, 3, device=DEVICE)` → `:60 torch.randn_like(x)`.
   CPU 기본 생성기와 CUDA(Philox) 생성기는 **같은 시드에서 다른 수열**을 낸다.
   ⟹ ★**T=500 스텝 역확산의 첫 잡음부터 다르다 ⟹ 다른 원자 집합 ⟹ 다른 성분 수·다른 파수.**
   이것이 "0/12 비트 불일치, 최대 상대차 245.1%" 의 자리다.
   폴리시(`d3_setdiff_posterior_b2.py:64-55`)와 부분추출(`learned_init.py:144`
   `np.random.default_rng(0)`)은 **난수가 없거나 명시 생성기라 장치 무관**하다
   ⟹ **발산의 유일한 원천은 `sample()` 의 두 `randn` 이다.**
3. **부수 결함 ①**: `:51` 은 **전역** RNG 를 재시드한다. 이후 `_adaptive_prune_grow:1633 torch.rand`
   와 `coarse_to_fine.py:177 torch.rand` 가 그 스트림을 이어 쓴다 ⟹ 제안이 추적/성장의 난수를 옮긴다.
   (원장 `reference_global_rng_coupling.md` 가 같은 결합을 기록했다. 여기서 **파일:줄로 확정**한다.)
4. **부수 결함 ②**: 만약 누가 `propose_atoms(device='cpu')` 를 CUDA 기계에서 명시하면
   모델은 CPU(`:135 _load_cond(ckpt, 'cpu')`)인데 `Bmap` 은 `cd.DEVICE`=cuda(`:151`) ⟹
   `cd.sample:53 model.enc(Bmap)` 에서 **장치 불일치 예외**. ★**즉 `device` 인자는 무시될 뿐 아니라, 쓰면 깨진다.**

### 결정적으로 만드는 최소 수정 (⛔적용하지 않았다)

★**최소 = `_load_cond` 안에서 모듈 전역 장치를 고정한다** — `scripts/` 를 안 고치고 3줄.

```python
# learned_init.py:_load_cond, `import d3_beamform_cond_diffusion as cd` 직후
import d3_beamform_net as _bn, d3_setdiff_posterior as _sp
cd.DEVICE = _bn.DEVICE = _sp.DEVICE = torch.device(device)
```
그리고 `propose_atoms:133-134` 의 자동 감지를 **호출자가 준 `config['device']`** 로 바꾸고
(`learned_init_params:355`·`sequential_update.py:1026` 에서 `device=config.get('device','cpu')` 전달),
CPU 로 고정하면 **CUDA 기계와 CPU 기계가 같은 수열**을 쓴다.
✅**CPU 전용 기계에서는 비트 동일**(현재 이미 cpu 로 자동 감지되므로) ⟹ 기록된 합성 배치가 무효화되지 않는다.

★**더 강한 대안**(장치를 자유롭게 두면서 재현성 확보): `cd.sample` 의 난수를 **CPU 생성기에서 뽑아 옮긴다**.
```python
g = torch.Generator(device='cpu'); g.manual_seed(seed)
x = torch.randn(n, M, 3, generator=g).to(DEVICE)
...  x = mean + (betas[t].sqrt() * torch.randn(x.shape, generator=g).to(DEVICE) if t > 0 else 0)
```
이쪽은 **전역 재시드도 함께 없앤다**(부수 결함 ①의 수리). CPU 에서는 같은 MT19937 수열이므로
**여전히 비트 동일**이어야 하지만 — ⚠**그것은 검증 대상이지 가정이 아니다.**

⚠**두 수정 중 어느 것도 부동소수점 수준의 장치 차이를 없애지 못한다**(cuBLAS vs MKL).
★**따라서 처방은 "고치고 끝" 이 아니라 "고친 뒤 같은 셀을 두 장치에서 돌려 잔여 차이를 재라" 이다.**
이 저장소의 규율(*"개입의 효과를 그것이 들어가는 양에서 재지 말고 판정에서 재라"*)이 그대로 적용된다.

### ★배치 위생 처방 (수정 없이 지금 당장 쓸 것)

행에 `torch.cuda.is_available()` 값과 `learned_init._CACHE` 키의 장치 문자열,
그리고 `n_selected`·제안 원자 해시를 **함께 기록하라**. 그러면 장치가 바뀐 셀이 사후에 탐지된다.
(현재 어느 것도 결과 행에 실리지 않는다 — `_initialize_learned:937-945` 반환 dict 에 장치 필드가 없다.)

---

## 7-B. ★★★성장 분기 발화율 — 원장 수치가 배포에서 반전됐다

동결 `.jsonl` 에서 **원 수치를 재집계**했다(런 0회).

### 원장 수치는 그 배치에서는 **정확하다**

원 자료 = `tmp/W3_SOLVER_2026-08-15/results.jsonl`(**2026-08-15**, `git_head 196050e`),
계수기 = `tmp/W3_SOLVER_2026-08-15/run_w3.py install_probe()` 가 `add_gaussians` 를 감싼 `add_calls`.
재집계 결과가 전건 일치:

| 층 | 경로 | n | `add_calls>0` | 비율 |
|---|---|---|---|---|
| 혼잡 | learned | 87 | **8** | **9.2%** |
| 혼잡 | v2_konly | 9 | 9 | 100% |
| 혼잡 | (합) | 96 | **17** | 17.7% |

원자 = 혼잡 learned `randphi_atoms` 합 **28**, `sum(N_final)` **1660** ⟹ **1.69%**, 중앙 `N_final` 19,
발화 8행 전부 `add_calls == 1`. `coarse_to_fine.py:173` = learned **0/130** · v2_konly **14/14** 도 일치.

### ⛔**그러나 그것은 배포 전 구성이다**

`W3`(08-15)·`W4`(08-19) 는 배포 커밋(**2026-08-22**: `98883a8` VARPRO · `d673892` 임계 8→0)보다 **앞선다**.
W3 행의 `effective_config` 스냅샷에 `varpro_steps`·`learned_init_iters`·`auto_switch_thresh` 세 키가
**아예 없다**(그때 코드에 없었거나 배포 전 기본값).

★**배포 구성의 측정은 디스크에 이미 있다** — `tmp/W5_WHY_2026-08-19/` 의 `L-VP-ONLY` 팔이
`learned_init_iters=0` + `varpro_steps=400` + route=learned 192/192 = **오늘의 출하 삼종과 구성 동일**:

| 팔 | 구성 | n(혼잡) | 성장 발화 | 비율 |
|---|---|---|---|---|
| `L-SHIP-DUP2` | iters 500, vp 0 = **배포 전** | 64 | 11 | **17.2%** |
| `L-VP800` | iters 500, vp 800 | 192 | 47 | 24.5% |
| ★**`L-VP-ONLY`** | **iters 0, vp 400 = 출하** | 192 | **141** | ★**73.4%** |

독립 확증 = `tmp/B_DEPLOY_2026-08-19/results.jsonl` 의 `A-VP` 팔
(코드가 `sequential_update.py:47` 에서 축자로 *"B 의 `A-VP` 팔 … 그것을 그대로 배포한다"* 라 적은 그 팔):
혼잡 learned **123/173 = 71.1%** (배포 전 `A-SHIP` 은 31/173 = 17.9%).

### ★그리고 이 반전은 **배포 3일 전에 예측되고 기록됐다**

`tmp/W4_SOLVER_2026-08-19/recon/B3_pathfacts.md:141-144` 축자:
> ★**`=0` 은 잔차를 키워 성장 분기 발화율을 올린다** … **9.2%보다 훨씬 자주** 켜질 수 있다.

`tmp/W5_WHY_2026-08-19/RESULT.md:49-50` 축자:
> `learned_init_iters=0` 은 결합 적합만 끄지 않는다 — **성장 분기 발화가 3배가 된다**.

⟹ ★***측정은 있었고 원장으로 전파되지 않았다.*** 이 저장소가 08-24 에 이미 대가를 치른
*"측정의 부족이 아니라 문서의 지연"* 과 같은 부류다.

### 왜 그런가 (기제)

`learned_init_iters=0` ⟹ 결합 적합이 잔차를 안 줄임 ⟹ `_adaptive_prune_grow:1514` 의
`residual_rms` 가 높게 남음 ⟹ `:1527` 문턱(0.05 m 절대)을 넘김 ⟹ 성장 발화.
★**즉 배포 스위치 하나(`learned_init_iters` 500→0)가 성장 분기를 켠 것이다.** 두 노브는 직교가 아니다.

### `final_n` 분포 (출하 등가 팔, 혼잡 192런)

| 결과 | n | 비율 |
|---|---|---|
| 성장 분기 **실행** | 141 | 73.4% |
| `n >= max_n` 조기반환(`:1500`)에 막힘 | **15** | **7.8%** |
| 잔차 문턱(`:1527`)에 막힘 | 36 | 18.8% |

`final_n == 30` 은 21/192 = 10.9%, 중앙 19.
⟹ ⛔**내가 §2-E 초안에서 쓴 *"가지치기가 되돌려지면 성장이 구조적으로 불가"* 는 과장이었다** —
그 관문이 막는 것은 **7.8%** 뿐이다. 정정한다.
⟹ ⛔**코드 주석 `sequential_update.py:1495-1496` 의 *"pool_and_select returns exactly max_n components"* 도
`max_n=30` 에서 거짓**이다(에너지 문턱·dedup 이 먼저 자른다). 그 문장이 참인 것은 주석이 실제로
인용하는 실측 사례(`max_n=10`, `tmp/B2_REAL_2026-08-23/RESULT_B2F.md:88-93`)뿐이다.

### ⚠남은 미확인

- **배포 커밋 이후(08-22 이후) git head 에서 잰 혼잡 `final_n`·`add_calls` 는 없다.**
  `B-16`(432행) `final_n` **전열 null** · `B_DEPLOY` `n_atoms_final` 전열 null ·
  `M4` 는 필드 자체가 없음. ⟹ W5/B_DEPLOY 는 **구성 등가이지 커밋 등가가 아니다**.
- **배포 후 "원자 비중" 대응치 없음** — `randphi_atoms`/`add_calls` 가 W5·B_DEPLOY 스키마에 없다.
  ⟹ ⛔**"1.69%" 의 배포 후 값은 미측정**이다.
- 배포 전 값조차 코퍼스 안정적이지 않다(W3 9.2% vs `A-SHIP` 17.9%, 다른 코퍼스·다른 시드).
  `E2_doc_corrections.md:669` 이 이 위험을 이미 명시했다.

### ★판정에 닿는 결과

원장은 축 **K1(`MIN_DK`)** 과 **K4(난수 재시작 위상)** 를 *"지렛대 부족 = 발화 9.2% · 원자 1.69%"* 라는
사유로 죽였다(`E2_doc_corrections.md:156,163`).
⟹ ⛔***그 사유의 앞 절반은 배포 구성에서 성립하지 않는다***(발화율이 약 8배).
뒤 절반(원자 비중)은 **배포 후 미측정**이다.
⛔**나는 그 축들을 되살리라고 주장하지 않는다** — 판정을 뒤집으려면 재측정이 필요하다.
주장하는 것은 **기각 사유가 더 이상 자료에 의해 지탱되지 않는다**는 것뿐이다.

---

## 8. 줄번호 드리프트 정정표 (원장 → 현재)

| 원장이 적은 것 | 현재 실측 |
|---|---|
| `sequential_update.py:1065-1200` (성장 두 번째 구현) | **`:1493-1653`** |
| `:1111-1116` `MIN_DK` | **`:1551, :1555, :1589`** |
| `:1188` `RX2_GROW_FITPHI` 게이트 사이트 | **`:1628`** |
| `:1193` 난수 위상 | **`:1633`** (그리고 난수 kx/ky 패딩은 `:1619-1620`) |
| `:660` (`_initialize_learned` 의 성장 호출) | **`:914`** |
| `:1335` `arctan2(b, a)` 부호 결함 자리 | 현재 `amplitude_refit` 은 `:1699-` (수리 완료, `varpro.py:49-54` 가 규약을 문서화) |
| `coarse_to_fine.py:173` | 변동 없음 ✅ |
| `learned_init.py:341` `learned_dedup_dk=0.03` | **`:344`** (기본값 읽기), 사용은 `:184-185` |
| `learned_init.py:386 ls_fit_wphi(ridge 0.5)` | **`:389`** 호출, ridge 는 `:322` |
| `learned_init.py:129` 독스트링 | 현재 `:130` 부근 (`propose_atoms` 반환 설명) |

---

## 9. ★원장(`CLAUDE.md` · `MEMORY.md`)에서 정정해야 할 문장

형식 = *"원장은 X 라 적었으나 실측은 Y"*. ⛔**나는 원장 파일을 수정하지 않았다.**

### ⛔C-1 (BLOCKER) — 성장 발화율이 배포에서 반전됐다
- **원장(`CLAUDE.md` 「측정 규율」)**: *"성장은 배포 혼잡 런의 82%에서 발화하지 않고, 발화해도 넣는 원자는
  28/1,660 = 1.69% 다"* · *"학습 경로만 8/87 = 9.2%"*.
- **실측**: 그 수치는 `tmp/W3_SOLVER_2026-08-15`(**배포 전 구성**)의 값으로 재집계에서 **전건 재현**되지만,
  **출하 삼종 구성**에서는 `tmp/W5_WHY_2026-08-19` `L-VP-ONLY` **141/192 = 73.4%**,
  `tmp/B_DEPLOY_2026-08-19` `A-VP` 혼잡 learned **123/173 = 71.1%** 이다.
- ⟹ **"82% 미발화" 를 "약 73% 발화" 로 바꿔야 한다.** 그리고 **1.69% 는 배포 후 미측정**이다(카운터 부재).
- ★부수: 이 반전은 `tmp/W5_WHY_2026-08-19/RESULT.md:49-50` 이 **배포 3일 전에** 기록했다 —
  *"성장 분기 발화가 3배가 된다"*. **문서 전파 실패**다.
- ★부수: 원장이 이 수치를 근거로 죽인 축 **K1·K4** 의 기각 사유 앞 절반이 무너진다(§7-B 끝).

### ⛔C-2 — 줄번호 여섯 개가 낡았다
- **원장**: *"같은 기능의 두 번째 구현이 `sequential_update.py:1065-1200` 에 살아 있고 …
  `:1111-1116 MIN_DK`, `:1193` 난수 위상, 호출 지점은 `_initialize_learned` 자신 `:660`"*.
- **실측**: 각각 **`:1493-1653`**, **`:1551/:1555/:1589`**, **`:1633`**, **`:914`**.
  (`:1188` → **`:1628`**.) 전체 대조표 = §8.
- ⚠**구조적 주장 자체는 유효하다** — 두 번째 구현은 살아 있고 조건절에 `'learned'` 가 있다(`:1505`, `:1544`).

### ⛔C-3 — *"테스트가 배포 기본값을 전혀 지키지 않는다"* 는 낡았다
- **원장(`CLAUDE.md`)**: *"⛔테스트가 배포 기본값을 전혀 지키지 않는다 (2026-08-13 실측). …
  동시에 되돌려도 `111 passed`. 테스트 전체에 `DEFAULT_CONFIG`·`configs/` 문자열이 0회 등장한다."*
- **실측**: `wavesplat/tests/test_shipping_defaults.py` 가 **바로 그 결함을 고치려고 만들어졌고**
  (파일 독스트링 `:3-21` 이 그 감사 소견을 인용한다), `SHIPPING_DEFAULTS`(`:44-68`)가
  `lambda_l1`·`init_method`·`tracking_mode`·`analytic_ensemble_k`·`varpro_steps`·`learned_init_iters`·
  `auto_switch_thresh`·`bank_refit_every_s`·`bank_refit_async` **아홉 개를 핀**하고
  yaml 미러(`:86-93`, `:345-347`)와 단일출처(`:80-83`, `:129-132`, `:182-187`)까지 검사한다.
- ⟹ ★**이 문장은 방벽이 세워지기 전의 진단이다. 지금 인용하면 거짓이다.**

### ⚠C-4 — *"실측 confidence≡1 = 구조적 무연산"* 은 근거가 다르다
- **원장(`MEMORY.md`)**: *"A7 신뢰 가중(실측 confidence≡1 = 구조적 무연산 ⟹ ⛔`B-25` 실측 적용 금지)"*.
- **실측**: `confidence≡1` 이 **성립할 조건**은 코드에 있다(`confidence.py:38-41` 의 `[0,1]` 클램프 +
  `real_data.py:35` `d_ref=5.0`), 그러나 **나는 실측 h5 를 열어 수치로 확인하지 않았다** ⟹ **미확인**.
- ★**그런데 결론은 더 강한 사유로 성립한다**: 실측은 게이트(`sequential_update.py:714-751`)가
  `bank_mmse` 로 보내고, **`bank_mmse` 는 `confidence` 를 읽지 않는다**
  (`_bank_obs_from_frames:1093` 가 `P=(x,y,t)` 와 `z` 만 꺼낸다).
  ⟹ *"값이 1이라 무연산"* 이 아니라 **"실측 추정기가 그 필드를 아예 안 본다"** 가 정확한 문장이다.

### ⚠C-5 — *"`analytic_ensemble_k` 앙상블이 합성에서 구조적으로 죽었다"* 는 ✅**참**이고, 원장에 **없는 귀결**이 있다
- **확인**: `sequential_update.py:1941` 이 `route != 'v2_konly'` 이면 `ensemble_k = 1`. 임계 0 ⟹ 항상 learned ⟹ 항상 1.
  `test_shipping_defaults.py:52-53` 이 같은 사실을 이미 기록.
- ★**원장에 없는 귀결**: 인가 문서(`tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §4`)가 인가한 것은
  **`auto_v1_m3` = v1 + 라우터 th=8 + guard 은퇴 + median-of-3** 이다. 현재 배포는 **th=0** 이고
  **median-of-3 이 죽었다** ⟹ ***인가 문서가 서술하는 세 요소 중 둘이 배포되지 않고 있다.***
  ⟹ ⛔**`RESULT_CONFIRM2.md §4` 를 현재 배포의 인가 근거로 인용하지 마라.**
  (변경 근거 W11/W13/W14/W15 는 따로 있으므로 *"재측정 없이 바뀌었다"* 는 주장이 **아니다**.)

### ⚠C-6 — 체크포인트 서술 두 가지
- **원장(`MEMORY.md`)**: *"담지자는 프레임워크가 아니라 체크포인트"* → ✅**코드가 강하게 지지**(§3.5-A).
- **원장**: *"평가와 같은 분포로 학습"* → ⚠**절반만 참**. 정확한 문장은 저장소 자신의 사전등록 문언
  (`tmp/mixed_retrain_2026-07-26/PREREG.md §2.2`) 인 **"in-family, out-of-realisation"** 이다.
  시드 축은 ✅분리 검증됨(학습 `[2e9, 2³¹)` vs 평가 전부 `< 2000`), (Hs,Tp)·성분수 층은 ✅의도적 정합,
  ⛔**그러나 `random_amplitudes`/`random_frequencies`/`decorrelate_stream`(평가 True, 학습 False) ·
  비선형성(학습 `nonlinear=False`) · 기하 도너 해상상태(BF6/BF7·slope)는 다르다**(§3.5-B).
- ⛔**신규 사실 — 원장에 없다**: `results/direction3/setdiff_prior.pt` 가 **git 미추적**
  (`/home/vexit/wavesplat/.gitignore:81 *.pt`)인데 **배포 합성 경로가 매 런 그것을 연다**(`learned_init.py:113`).
  ⟹ **배포가 저장소에 없는 파일에 의존한다.**
- ⛔**신규 사실**: `tmp/mixed_retrain_2026-07-26/FROZEN.sha256` 이 **3/3 FAIL**
  (`PREREG.md`·`gen_mixed.py`·`train_mixed.py` 전부) ⟹ **v1 을 만든 코드 바이트가 디스크에 없다.**
- ⚠**신규 사실**: v1 은 자체 게이트 `G2a` 를 **FAIL** 했고(바 ≤6, v1 = 12.0; v3 = 6.0 PASS,
  `tmp/mixed_retrain_2026-07-26/RESULT.md §4`), 선택은 **동결 tie-break 11/16**(p=0.105)이었다.
  코드 주석(`learned_init.py:50-53`)이 이미 *"우월성의 증거가 아니다"* 라 적었으므로 **모순은 아니지만**,
  학위논문이 체크포인트를 기여로 말할 때 반드시 병기해야 한다.

### ⚠C-7 — `lambda_l1` 의 작용 자리 (원장에 없는 사실)
- **실측**: 배포 학습 Phase-1 에서 `_compute_loss` 호출이 **0회**(`learned_init_iters=0`) ⟹
  ★**L1 벌점이 학습 경로의 초기화에 전혀 걸리지 않는다.** 사는 곳은 추적(`:1406`)과 성장 미세조정(`:1646`).
  그리고 **VARPRO 의 축약 목적함수에도 λ 가 없다**(`varpro.py:82-88` 자인, ridge 1e-8 만).
- ⟹ ⛔**원장의 λ 축 서술(*"λ 벽"*, *"출하 1e-4≈0"*)을 학습 경로 초기화의 성질로 읽지 마라.**
  ★단 원장은 이미 λ 축을 *"W3 종결"* 로 닫아 두었으므로 **새 축을 열자는 제안이 아니다.**

### ⚠C-8 — 장치 의존성 (원장에 부분만 있다)
- **원장**: *"`bank_mmse.py:41` 은 임포트 시점 `_DEV`, `learned_init.py:133-134`·`:152` 는 `cd.sample` 이
  장치를 따라간다"*.
- **실측 보강**: 자동 감지가 **일곱 자리**다(§6 표). 발산의 **유일한 원천**은 `cd.sample` 의 두 `randn`
  (`d3_beamform_cond_diffusion.py:54, :60`)이고, 폴리시·부분추출은 장치 무관이다.
  그리고 ★**`propose_atoms(device=…)` 는 무시될 뿐 아니라 CUDA 기계에서 쓰면 장치 불일치로 깨진다**(§6-4).

### ⚠C-9 — 소소한 것 넷
1. `learned_init.py` 의 dedup 은 **유클리드가 아니라 축별 상자**(`:184`) ⟹ 대각 분리는 `0.03·√2 = 0.042`.
   원장의 *"분리 문턱 0.03 은 측정된 분해능 한계(반경 0.042–0.074) 양쪽보다 작다"* 는
   **축 방향에서는 참, 대각에서는 경계값**이다.
2. 사전의 파수 지지가 **`|kx|,|ky| ≤ 1.1`(폴리시 후 1.21)로 하드 캡**돼 있다(§3.3).
   원장 어디에도 이 한계가 적혀 있지 않다. ★학위 서술 *"격자 비의존 연속 파수 표현"* 의 **범위 단서**다.
3. `configs/estimation.yaml:90 max_steepness` 는 **로더가 안 읽는다** = `bank_refit_every_s` 와 같은 죽은 미러(§5-E).
4. `_ls_w_update`(`:1655-1697`)는 **호출 지점 0곳 = 완전 사망 코드**.

---

## 10. ★학습 위주 연구를 위한 관측 (판정 아님, 제안 아님)

원장 규율에 따라 **새 축을 제안하지 않는다.** 다만 이 지도가 드러낸 **구조적 사실**만 적는다.

1. ★**"학습" 은 파이프라인에서 정확히 한 자리다** — (kx, ky, 에너지) 집합의 조건부 제안.
   ω·φ·진폭·추적·가지치기·성장·VARPRO 는 전부 비학습이다.
   ⟹ 학위 서술이 *"학습된 스펙트럼 사전"* 을 주어로 삼으려면 **이 경계를 명시**해야 한다.
2. ★**배포 학습 Phase-1 에서 실제로 도는 최적화는 VARPRO 400스텝 하나뿐이다**(결합 적합 0회).
   ⟹ *"학습 초기화 + 결합 적합"* 이라는 서술은 **배포 구성에 대해 거짓**이다.
3. ★**두 스위치가 직교하지 않는다**: `learned_init_iters=0` 이 성장 발화율을 17%→73% 로 올렸다(§7-B).
   ⟹ 배포 삼종을 **하나씩** 되돌리는 팔은 측정된 적 없는 조합이 된다
   (`pre_2026_08_22_config()` 독스트링이 이미 경고하는 바로 그것).
4. ★**학습 경로 노브 열한 개가 CLI·YAML 어디에도 없다**(§4-②).
   ⟹ 학습 축 배치는 **전부 파이썬 API 경로**를 타야 하고, 그 경로에는 화이트리스트가 없다
   ⟹ *"설정을 넣었다 ≠ 설정이 먹었다"* 를 막을 기계가 **없다**. 개입 계수기를 행에 실어라
   (`_snap_k_to_grid` 가 `fired` 를 반환하는 방식이 이 저장소의 본보기다, `:980-983`).
5. ⚠**결과 행에 장치·체크포인트 sha·`n_selected` 가 실리지 않는다**
   (`_initialize_learned:937-945` 반환 dict 에 없다) ⟹ 재현이 사후에 고정되지 않는다.
