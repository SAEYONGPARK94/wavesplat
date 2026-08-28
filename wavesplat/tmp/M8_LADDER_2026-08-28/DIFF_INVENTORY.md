# M8 DIFF_INVENTORY — bank_mmse(SHIP) vs 제안 표현(PROP) 전수 차이 목록 (2026-08-28)

> **임무 B 산출물. 설계 아님** — 사다리(ladder) 설계는 임무 A 의 몫이다. 이 문서는
> 「무엇이 다른가」·「어느 차이가 개입 가능한가」·「무엇이 이미 측정됐는가」만 적는다.
> ⛔이 문서의 어떤 문장도 측정 주장이 아니다(§6 은 명시적 *가설*).
> **런 0회. `wavesplat/**` 무수정. PREREG/gates 미작성(임무 A 소관).**

## 0. 열람 범위 (= 주장 범위)

| 파일 | 읽은 범위 |
|---|---|
| `wavesplat/optimizer/sequential_update.py` | :40–330(상수) · :380–660(__init__/옵티마이저/loss) · :660–770(실측 게이트) · :863–1310(learned/bank init·update) · :1325–1640(추적·prune/grow) · :1804–1935(predict/ensemble) · :2006(make_updater) |
| `wavesplat/optimizer/bank_mmse.py` | 전문 (222행) |
| `wavesplat/optimizer/learned_init.py` | 전문 (409행) |
| `wavesplat/optimizer/varpro.py` | 전문 (275행) |
| `wavesplat/model/tnse.py` | :176–205 (`fit_amplitudes_phases`) |
| `wavesplat/model/confidence.py` | :24–41 · `wavesplat/real_data.py` :285–312 |
| 배치 문서 | `M4_CAPACITY`/`M5b_WINDOW` RESULT+PREREG, `B2_REAL_2026-08-23/scene_station.py`, `M5b/run_m5b.py`, `blind_auto_real_2026-07-26/run_blind_auto.py:38-54`(frames_from_scene 원형) |

두 경로의 정의(실측 12 레코드 아크의 팔 정의, `M5b/PREREG.md:156-161` 그대로):
- **SHIP** = `init_method='auto'` → 실측 게이트(`sequential_update.py:706-716`) → `_initialize_bank`(:1100) → `bank_mmse.fit_bank_mmse` × median-of-3.
- **PROP** = `real_domain_gate=False` + `init_method='learned'` → `_initialize_learned`(:863) → `learned_init.learned_init_params` → VARPRO 400(:913-918).

---

## 1. 차이 표 (축별, 코드 근거 포함)

| # | 축 | SHIP (bank_mmse) | PROP (learned+VARPRO) | 코드 근거 |
|---|---|---|---|---|
| D1 | **표현** | 고정 연속 뱅크의 **선형 계수장**: `z=Σ(a_b cosψ_b + b_b sinψ_b)`, ψ=k·x−ωt. 원자는 안 움직인다 — 계수만 축소(shrink)된다 | **희소 자유 원자**: `z=Σ w_i cos(k_i·x−ω_i t+φ_i)`, (kx,ky) 자체가 최적화 변수 | `bank_mmse.py:150-176` vs `spectral_gaussian` 극형식, `varpro.py:87-99` |
| D2 | **원자 수** | 후보 **B=1,908**(kp 무관: 환대 0.35–2.5kp, 격자 kp/10 → 계수 2B=3,816) | **N≤max_n=30** (실측 B-2 행 실측치 25–30; M4 K 중앙 30) | `bank_mmse.py:157-162`(격자 셈 재계산) vs `learned_init.py:168`(cap), B-2 `results.jsonl` |
| D3 | **파수 배치·원천** | 관측 자신의 2단 beamform zoom 이 kp 를 정하고(**kp_hint=None**), 그 주위 **고정 격자** | diffusion 체크포인트(합성 학습, `_mixed_v1`)가 beamform 지도 조건으로 **제안** → polish 80 iter → dedup 0.03 → VARPRO 가 **연속 이동** | `bank_mmse.py:146-152` vs `learned_init.py:124-166,168-191`, `varpro.py:118-147` |
| D4 | **적합기(결합)** | **폐형 선형-가우시안 MMSE**(듀얼 공간 n×n solve). 볼록, 반복 0회, 국소해 없음 | **비볼록 경사하강**: VARPRO(Adam 400 스텝, lr 2e-3)로 k 만 내리고 (a,b)는 ridge 1e-8 폐형 소거. `learned_init_iters=0`(결합 Adam 없음, W5) | `bank_mmse.py:215-219` vs `varpro.py:100-115,118-147`, `sequential_update.py:889` |
| D5 | **정칙화·잡음 모형** | **명시적 관측잡음** s²=(smult·0.05·Hs_obs)², smult∈{1,3,10} **val 선택** + 후보별 사전분산 gam ∝ beamform power^pexp(pexp∈{0.5,1,2} val 선택), 총에너지 var(z) 앵커, **val-gated SBL-EM ≤6회** | VARPRO 축약목적에 **λ_l1 부재**(파일에 `lambda` 0회, 문서화된 미측정 교락), ridge **1e-8**≈보간. `ls_fit_wphi` ridge 0.5. L1/smooth 는 추적·grow 폴리시에만 잔존 | `bank_mmse.py:140,177-213` vs `varpro.py:87-99` 상단 주석, `learned_init.py:309-323`, `sequential_update.py:544-600` |
| D6 | **자기검증(GT-free)** | **내장 held-out**: 관측 25% val 분리(`vi=rng.choice`), 초모수·EM 을 **val 예측오차로 게이트**, `vr`/`vr_rel` 을 적합 신호로 반환 | **전무**. 모든 관측에 적합, 자기 예측오차를 재는 장치가 없다 | `bank_mmse.py:138-144,183-213` vs `_initialize_learned` 전체(:863-937) |
| D7 | **앙상블** | **median-of-3 장**(val-split 시드 s,s+1000,s+2000; `real_bank_ensemble_k=3`) | **단일 런**. `MedianEnsembleUpdater` 는 v2_konly 경로만 감싼다 — learned 경로는 구조적으로 앙상블 없음 | `sequential_update.py:1126-1131,318-324` vs :1924-1934, `make_updater`:2006 |
| D8 | **시간 전개** | 동일 선형 분산 전파(ω=√(gk·tanh kd) — **두 팔이 같은 ω**, ⛔`DISPERSION-EARNS` 금지 기존 판정 유지). 롤링 관측창 40 s + **재적합 노브**(`bank_refit_every_s`, 기본 None; async 옵션) | 동일 전파(−ωt). 재적합 개념 없음; 대신 **추적기**(w·φ Adam, k 동결) — 단 M5b·B-2 프로토콜은 initialize 후 동결이라 **추적 미사용** | `bank_mmse.py:47`, `sequential_update.py:1271-1310` vs :1325-1400, `M5b/run_m5b.py:190-208`(update 호출 없음) |
| D9 | **관측 사용 — 개수** | `bank_max_obs=600` cap(결정적 default_rng(0) 추출) | diffusion 제안만 `M_OBS_CAP=400` cap(모듈 상수, 미노출); `ls_fit_wphi`·VARPRO 는 **창 전체(600)** 사용 | `sequential_update.py:1160-1178` vs `learned_init.py:65,143-145`, `varpro.py:160-176` |
| D10 | **관측 사용 — 가중** | **무가중**(`_bank_obs_from_frames` 가 confidence 를 아예 안 나른다) | `_compute_loss`·VARPRO 가 confidence 가중 — ⚠단 **실측 하네스에서는 항진**: `frames_from_scene` 이 `confidence=torch.ones(n)` (B-2/M5b 계열 공통 원형) | `sequential_update.py:1085-1098` vs :560-575, `varpro.py:196-199`, `blind_auto_real_2026-07-26/run_blind_auto.py:52` |
| D11 | **난수 사용** | 전부 **국소 RNG**: val 분리 `default_rng(seed)`, 600 cap `default_rng(0)`. 전역 스트림 소모 0 | diffusion 샘플 시드 + **전역 torch RNG**(grow 위상 `torch.rand` :1617) + 전역 `np.random`(grow 패딩 :1602-1606) — ★배경문서의 "소모 draw 수" 위험은 **PROP 계열에만** 있다 | `bank_mmse.py:137,` `sequential_update.py:1171` vs :1602-1617 |
| D12 | **정밀도·장치** | float64 numpy(beamform 만 float32 torch) | 모델·추적 float32 torch / VARPRO 내부 float64 | `bank_mmse.py:41-44` vs `varpro.py:38` |
| D13 | **적응 N (prune/grow)** | 없음(성분 0개 경로) | 있음 — 단 `n≥max_n` 이면 grow **가지 전체가 구조적 사문**(B-2f: max_n 도달 시 비트동일). 실측 n_selected 25–30≈30 이라 대체로 닫혀 있다 | `sequential_update.py:1407-1637`, :1500-1506 주석 |

**같은 것**(차이 아님, 헛개입 방지용): 전방 물리(선형 중첩 + 같은 분산관계), 채점 규약,
init 창(두 팔이 같은 frames — `M5b G-1` 계열이 obs_hash 로 강제), depth.

---

## 2. 개입 가능성 판정 (⛔`wavesplat/**` 무수정 규율 하)

채널 구분: **[A] 노출 인자**(config dict — M 배치 하네스는 `make_updater(cfg)` 직접 호출이라
run.py CLI 화이트리스트에 안 막힘) / **[B] 하네스 수준**(패키지 함수 직접 호출·shim; 선례 =
M5b 의 `SequentialUpdater.__init__` 캡처 shim, W4 varpro 몽키패치, `test_bank_mmse_port` 비트 핀) /
**[C] 코드 수정 필요 = 이번엔 불가**.

| # | 차이 | 채널 | 구체 수단 |
|---|---|---|---|
| D1+D4 | 표현·적합기 통째 교환 | **[B]** | PROP 원자 위에 bank 의 MMSE 수학을 하네스에서 실행(fit_bank_mmse 는 격자를 내부 생성하므로 **후보 주입은 함수 호출로 불가** — gam/K solve ~30행을 하네스 재구현해야 함). ⚠**「같은 기능 두 곳 구현」결함 부류** — 기본 인자에서 `fit_bank_mmse` 와 비트일치 핀 게이트 필수(선례 `test_bank_mmse_port.py`) |
| D2 | 원자 수 (PROP 쪽) | **[A]** | `max_n`·`learned_energy_thresh`·`learned_dedup_dk`·`learned_atoms`(원자 직접 주입 — beamform 격자를 PROP 파이프라인에 먹일 수 있음). ⚠M4 가 이 방향 이미 측정(§3) |
| D3 | 원자 원천 | **[A]** | `learned_atoms` 주입 / `real_estimator='v2_konly'`(beamform 원천). ⚠후자는 B-8 기측정 |
| D4' | VARPRO 용량 | **[A]** | `varpro_steps` ∈ {0, …, 400} — 실측에서 **미측정**(§3) |
| D5 | 잡음 모형·ridge | **[A+B]** | [A] `varpro_steps=0` 으로 배포 호출 끄고 [B] `apply_varpro(..., ridge=…, lr=…)` 직접 호출(스텝·ridge·가중이 인자로 노출돼 있으나 `_initialize_learned:916-918` 은 steps 만 config 에서 읽는다). λ_l1 쪽은 [A] `lambda_l1` — ⚠B-2c 기측정(실측 널) |
| D6 | held-out 자기검증 | **[B]** | 하네스 래퍼: 관측 75% 로 PROP 적합 → 25% 예측오차로 소가족 선택/중지. 패키지 무수정 |
| D7 | 앙상블 | **[A]**(SHIP 쪽) / **[B]**(PROP 쪽) | SHIP: `real_bank_ensemble_k=1`(1..5 검증됨 :1128). PROP: `learned_diffusion_seed` 3개 런의 장 median 을 하네스에서 계산 |
| D8 | 재적합·창 | **[A]** | `bank_refit_every_s`·`bank_refit_async`·`bank_memory_s`; 창 길이는 장면 인자(W). ⚠B-3·B-20·M5b 기측정 |
| D9 | 관측 수 등화 | **[A]** 반쪽 | `bank_max_obs=400` 으로 SHIP 를 PROP 의 제안 cap 에 맞추는 방향은 노출. `M_OBS_CAP`(400→600)은 모듈 상수 = **[C]**(shim 으로는 가능하나 학습 분포 이탈 교락) |
| D10 | 가중 | **[A]** | `use_confidence=False` — ⚠단 실측 하네스는 conf≡1 이라 **항진 개입**(G-4 류 비트동일 예상). 실측 후보로 부적격 |
| D11 | 난수 | **[B]** | 개입 아님 — **사다리 설계 위험**: 단(rung)마다 소모 draw 수가 다르다(배경문서 :1617). PROP 계열 팔은 grow 가지 도달 여부(D13)로 전역 RNG 소모가 갈림 ⟹ 원자 카운터 + RNG 스냅샷 병기 필요 |
| D12 | 정밀도 | **[C]** | 개입 후보 아님(교락 소음원으로만 기록) |
| D13 | prune/grow | **[A]** | `adaptive_n=False` — 단 실측 n≈max_n 이라 grow 는 이미 사문(B-2f), prune 만 남음. 개입 계수는 **가지 단위**로 |

---

## 3. 이미 측정된 차이 (중복 측정 금지 목록)

| 차이 | 배치 | 판정 (요지) | 남는 미측정 구석 |
|---|---|---|---|
| D2 (PROP 용량) | **M4** (2026-08-28, 실측 12 + 합성 48) | `CAPACITY-HURTS` 12/12 · 48/48 — PROP 적합기를 든 채 원자를 늘리면 **나빠진다**, 사다리 8점 단조 | **용량×축소(shrinkage) 상호작용**: "많은 원자 + MMSE 사전" 칸은 bank 그 자체뿐, 중간 칸 0개 |
| D8 (창 길이) | **M5b** (2026-08-28, 456런) | `WINDOW-LONGER-IS-BETTER` 반증형: PROP 2/12(+31%), SHIP 11/12(−13%) — **부호가 팔에 따라 반대** | 갈림의 **기제**(§6) — M5b 행에는 `varpro_fired` 불리언뿐, `vp_k_shift` 등 미기록 |
| D1 (파수 연속성) | **B-16/B-19** (⚠**합성**) | k-격자 스냅만으로 46/48 −38.61% 악화(혼잡) · 14/16(희소) — 연속성이 **합성 이득**의 담지자 | **실측**에서 같은 절제(`k_snap_dk`) 0건. ⚠단 bank 가 격자(kp/10)로 실측을 이기므로 "연속성 무용" 방향 해석은 적합기 교락(D4·D5)과 분리 안 됨 |
| D4+D5+D6 (과적합 노출) | **W12** (2026-08-22, 실측 5레코드) | 학습 강제 시 VARPRO 가 관측 목적 −63% 인데 장은 제로필드보다 나쁨 — **관측 잔차가 과적합을 탐지 못 한다** | 기제의 **개입** 0건: VARPRO 용량 사다리·ridge·held-out 게이트 전부 미개입. W12 는 관찰이지 절제가 아님 |
| D8 (재적합) | **B-3**(36/36 −23.96%)·B-9/9b·B-20 | bank 재적합은 기술을 더한다(동결이 손해) | PROP 쪽 재적합(re-init 롤링)은 미측정 — ⚠단가 6.11 s/런(M5b) |
| D3 (원자 원천) | **B-8** (실측, `V2K` 14/36) | 학습 제안을 beamform 원천으로 빼도 **똑같이 무기술** ⟹ 원천 단독은 담지자 아님 | — (닫힘에 가깝다) |
| D5 (λ_l1) | **B-2c** (실측 36런) | λ 를 어떻게 해도 7–8/12, **벌점 전제거도 동일** ⟹ 실측 한계는 λ 축이 아님 | VARPRO 내부 ridge·잡음항은 λ_l1 과 별개 — 미측정 |
| D10 (신뢰 가중) | **B-25** (⚠합성, GT-유래 가중) | `GT-WEIGHT-HURTS` — 참 법선 가중 절제 시 배포 팔이 이긴다(정보 자체가 손실원) | 실측은 conf≡1 이라 축 자체가 항진 — 측정 불요 |
| D4 (결합 Adam) | **W5** (합성) | `learned_init_iters` 잉여(P-2 널) | — |
| D2' (merge/anchor) | W4·2026-07-26 (합성) | `learned_merge_frac` 무익~유해 | — |
| D7 (앙상블 k) | **미측정** — `real_bank_ensemble_k` 는 effective_config 되읽기 기록만 있고(W4/W5/HOSGATE jsonl), **개입 0건** | | SHIP k=1 vs 3, PROP median-of-3 둘 다 빈 칸 |
| D6 (held-out) | **미측정** — 저장소 전체에서 PROP 계열에 val 분리를 붙인 팔 0건(열람 범위 내) | | 빈 칸 |

---

## 4. 유망 차이 상위 3 (기준: ①실측 개입 가능 ②미측정 ③기술 차이 설명 개연성)

**1위 — D5+D6: 잡음 모형·held-out 자기검증 (bank 의 "통계적 척추")**
- ①: `varpro_steps` 사다리는 노출 인자[A]; ridge·val-게이트는 하네스[B]로 도달(패키지 무수정).
- ②: 실측에서 VARPRO 용량·ridge·held-out 전부 개입 0건(§3).
- ③: **가장 강한 정황이 이미 둘** — W12(관측 잔차 ↓ 장 오차 ↑ = 무검증 적합의 교과서적 서명),
  B-2c(λ 축 소거로 "정칙화 총량"이 아니라 **어디에·무엇으로** 거는가가 남음). bank 는 s² 를
  관측에서 스스로 정하고(0.05·Hs_obs × val 선택 배수) 사전분산을 beamform power 에 앵커한다;
  PROP 은 ridge 1e-8 로 사실상 보간한다. 두 팔의 실측 갈림 중 **적합기가 관측 잡음을 아는가**
  하나로 환원될 수 있는 부분이 미측정 최대 덩어리다.
- "널이면 무엇을 쓰는가": VARPRO 용량·ridge 사다리가 전부 널이면 *"PROP 의 실측 실패는
  적합 통계가 아니라 표현/지지집합에 있다"* 로 소거표가 한 칸 좁혀지고 D1/D2 상호작용(2위)로 넘어간다.

**2위 — D7: 앙상블 비대칭 (median-of-3 vs 단일)**
- ①: SHIP 쪽 `real_bank_ensemble_k=1` 은 **순수 노출 인자 한 개**[A]; PROP 쪽 시드-median 은 하네스 3런.
- ②: 개입 0건(§3) — 그리고 이것은 **모든 실측 SHIP<PROP 비교 안에 이미 들어앉은 교락**이다
  (SHIP 수치는 전부 median-of-3, PROP 은 전부 단일 런).
- ③: 설명 개연성은 1위보다 낮다(bank 단일 멤버도 v2_konly 를 이겼던 2026-07-29 포트 기록이
  있으므로 앙상블이 전부일 리는 낮음) — 그러나 **사다리의 0단**으로서의 가치가 크다:
  이 칸을 닫지 않으면 다른 어떤 단의 크기도 순수하게 읽히지 않는다. 비용도 최소다.
- "널이면": *"앙상블은 담지자가 아니다"* 가 확정되고 이후 모든 단에서 k=3/단일 비대칭을
  공시 한 줄로 처리할 수 있게 된다.

**3위 — D2×D5: 용량×축소 상호작용 (지지집합 교환)**
- ①: 반쪽은 노출[A] — `learned_atoms` 로 beamform 격자 원자를 PROP 파이프라인에 주입(+`max_n`↑,
  `varpro_steps=0` 으로 이동 금지). 반대쪽(PROP 원자 위 MMSE)은 하네스 재구현[B] + 비트 핀 게이트 필요.
- ②: M4 는 **PROP 적합기를 고정**하고 용량만 움직였다 — "격자 지지집합 + MMSE 축소" 와
  "희소 자유 원자" 사이의 **중간 칸이 0개**다. B-16/B-19 는 합성이고 방향이 반대(연속성이 번다)라서,
  실측에서 격자 bank 가 이기는 것과 정면 긴장 관계 — 이 긴장을 가르는 측정이 없다.
- ③: M4(용량 단독 유해) + B-8(원천 단독 무관) + 1위 축 결과가 나오면, 남는 후보는
  "원자가 **움직이는가**(자유 k) × 계수가 **축소되는가**" 상호작용뿐이다.
- "널이면": PROP 표현 자체가 실측에서 성립 불가 쪽으로 소거가 진행되고, 논문의
  "작동 경계" 서사(합성 승·실측 순손실)의 기제 후보가 적합기에서 표현으로 옮겨 적힌다.

(차순위로 D9 관측 수 등화 `bank_max_obs=400` — 노출 인자 하나로 값싸게 닫히는 공시용 칸.)

---

## 5. 사다리 설계에 넘기는 위험 메모 (임무 A 참조용)

- **전역 RNG(D11)**: PROP 계열 단은 grow 가지 도달 여부·diffusion draw 수로 전역 torch/np 스트림
  소모가 단마다 다르다(`sequential_update.py:1617`, :1602-1606). bank 계열 단은 소모 0.
  ⟹ 단마다 `torch.get_rng_state` 해시 + 가지 단위 개입 카운터(B-2f 규율)를 행에 실을 것.
- **필터 항진(M4 G-13 부류)**: `use_confidence` 개입은 실측 하네스에서 conf≡1 항진(D10),
  `adaptive_n` 개입은 n≈max_n 에서 grow 사문(D13) — 게이트가 세는 집합이 팔을 비우지 않는지
  러너 작업목록 합성 행으로 먼저 시험.
- **하네스 재구현 칸(D1+D4, D6)**: 「같은 기능 두 곳 구현」 결함 부류 — 기본 인자 비트일치 핀
  (`test_bank_mmse_port.py` 선례) 없이는 팔이 아니라 버그다.
- **PROP 단가**: 실측 1런 6.11 s vs SHIP 1.64 s(M5b PREREG :344) — CPU 여유는 있으나 단 수 설계에 반영.

---

## 6. M5b 갈림(같은 개입, 반대 부호)의 가설 — ⛔전부 미측정 가설이다

관찰(동결, M5b): 창 40 s→2·Tp 단축이 SHIP 에는 **11/12 −13.13%**, PROP 에는 **2/12 +31.30%**,
PROP 쪽 용량-반응 단조(TP6 7/12 → TP4 4/12 → TP2 2/12).

**H1 (주가설) — 과적합 노출 비대칭 (D5+D6).** 창 단축 = 관측 수·시간 기저 축소 = 분산 비용 증가.
bank 는 그 비용을 **흡수하는 기제를 셋** 가진다(val 예측오차가 커지면 smult 그리드가 더 센
축소를 고르고, EM 이 val 로 게이트되고, median-of-3 이 멤버 분산을 깎는다) — 정확도에 남는 것은
창 단축의 **이득**(예측 시각에 가까운, 더 정상성 좋은 관측)뿐. PROP 은 흡수 기제가 0개라
(ridge 1e-8, 검증 없음, 단일 런) 분산 비용이 그대로 장 오차로 간다. **부호를 뒤집는 것은
개입이 아니라 팔의 분산 통제 유무**라는 가설. → 1위 축(D5+D6)의 창-교호 팔로 검정 가능
(예: TP2 에서 varpro_steps/ridge/held-out 사다리 — 창 이득이 회복되면 H1 지지).
- 판별 예측: PROP 실패 창에서 `vp_obj_after` 는 낮은데 장 오차가 크고(W12 형), `vp_k_shift` 가
  창 단축과 함께 커진다. ⚠M5b 행에는 `varpro_fired` 불리언만 있어 **기존 자료로는 못 가른다**.

**H2 — k-식별성(피셔 정보) 비대칭 (D1/D4).** PROP 만 (kx,ky)를 **추정**한다. k 의 식별성은
시간 개구에 1차로 의존하므로 창 단축은 PROP 에만 있는 추정 문제 하나를 정확히 타격한다;
bank 는 k 를 추정하지 않고 고정 후보를 가중할 뿐이라 그 채널 자체가 없다. H1 과의 차이:
H1 은 "잡음을 덜 걸러서", H2 는 "추정할 것이 하나 더 있어서". 검정 분리는 `varpro_steps=0`
(k 이동 금지, LS 만) 팔 — H2 라면 k 동결만으로 창 단축 손해가 크게 줄어야 한다.

**H3 (반증 우세로 격하) — 학습 지지집합.** "창이 학습 분포(tp_factor 2.0)에 가까울수록 낫다"는
방향은 **M5b 자신의 용량-반응이 반대**다(지지집합 쪽으로 줄일수록 나빠짐) ⟹ 주가설 자격 없음.
잔여 형태(제안 원자의 질이 창에 무감하고 하류 적합만 갈린다)는 H1/H2 로 흡수된다.

공통 프레임: **양팔은 같은 물리로 외삽하고(D8 동일 ω), 창은 bias(정상성)–variance(적합) 거래의
지점만 옮긴다** — bank 는 variance 쪽 방벽이 있어 bias 이득이 드러나고, PROP 은 방벽이 없어
variance 손실이 이긴다. H1·H2 는 그 방벽의 정체가 통계(D5+D6)냐 추정 부담(D4)이냐로 갈린다.
