# 사전등록 — 타이밍 전면 재측정 (2026-08-05)

**작성 시각**: 런 시작 **전**. 이 문서가 커밋된 뒤에 harness 를 돌린다.
**배경**: `docs/DEBATE_2026-08-05_LATENCY_AND_REPRESENTATION.md` §A. 논문의 지연 주장
(`~10×`, `132.6 ms`, `13.4/11.0 ms`, `on a single CPU thread`, `Intel i7-12700H`)이 전부 철회 대상.
이 재측정은 **그 자리를 대체할 숫자를 처음부터 다시 만드는 것**이지, 기존 숫자를 방어하는 것이 아니다.

⛔ **사전 공약**: 결과가 어느 방향이든(비가 1.0배든 3배든, WS 가 지든) 그대로 보고한다.
지연 우위는 이 측정의 **가설이 아니라 측정 대상**이다.

---

## 1. 측정 대상 (arms)

| arm | 클래스 | 설정 (논문 튜닝값, `tmp/rerun_2026-07-25/batchA_baselines.py` 와 비트 동일) |
|---|---|---|
| **WS-P1** | `run_estimation` 경로 | `init_method='v2_konly'`, `use_dispersion=True`, `use_confidence=True`, `adaptive_n=True`, `tracking_window=3` |
| **ETKF** | `ETKFGrid` | 40×30 격자, `n_ensemble=100`, `mult_inflation=1.0`, `add_inflation=0.005` (기본 `obs_noise=0.05`, `max_obs=40`) |
| **LETKF** | `LETKFGrid` | 위와 동일 + `loc_radius=50.0` |
| **GPR-1** | `GPRBaseline` | `kernel_type='matern'`, `length_scale=20.0`, `noise_sigma=0.02`, `window_frames=1` |
| **GPR-disp** | `GPRDispersion` | `length_scale=20.0`, `noise_sigma=0.02`, `window_frames=3`, `depth=50.0` |

**에피소드**: P1 12ep 세트(`batchA_baselines.py:TABLE1_EPS`), `data_finemesh_2026-07-23/`.
**반복**: 3회. WS 는 추정 시드 `[42, 123, 456]`(rep 1/2/3).
베이스라인은 3회 전부 `np.random.seed(42)`(**논문 프로토콜 충실**; 앙상블 내용은 compute cost 를
바꾸지 않으므로 시드 변경은 타이밍에 정보를 더하지 않는다).
⚠ **비대칭 공개**: WS 3반복은 *기계 잡음 + 시드로 인한 N 변동*을 섞고, 베이스라인 3반복은
*기계 잡음만* 잰다. WS 쪽 산포가 구조적으로 더 넓게 나오는 것은 결함이 아니라 이 설계의 결과다.

---

## 2. 계측 위생 (측정 **전에** 고정)

1. **시계**: 모든 타이밍은 `time.perf_counter()`(= `CLOCK_MONOTONIC`).
   `time.time()` 기반 프로덕션 `elapsed_ms` 도 **나란히 기록**하여 편향을 직접 잰다(무수정 관찰).
2. **프로덕션 코드 무변경**: harness 가 `updater.update(frame)` / `baseline.update(...)` 호출을
   **바깥에서 감싼다**. `wavesplat/`, `wavesim/`, `configs/`, `paper/` 파일 수정 0건.
3. **스레드 고정**: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
   `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` (numpy import **전**에 설정) + `torch.set_num_threads(1)`.
   런 시작 시 실제 적용값을 provenance 에 기록.
4. **직렬 실행**: 워커 1개, `multiprocessing` 사용 안 함. **전 팔이 같은 프로세스·같은 세션**.
   이것이 A-2 (BLAS 스레드 370배 교란)의 처방이다.
5. **순서 고정**: `for rep in 1..3: for ep in 12: for arm in [WS, ETKF, LETKF, GPR-1, GPR-disp]`.
   3반복 모두 **같은 순서**. 에피소드 내에서 5팔이 연속 실행되므로 팔들은 동일한 머신 상태를 본다.
6. **워밍업**: 각 (arm, ep, rep) 런의 **처음 5 프레임은 PRIMARY 집계에서 제외**(별도 기록).
7. **스텝 삼킴 탐지**: 런마다 `time.time()` 총경과 vs `perf_counter()` 총경과를 대조하고,
   프레임 단위로 `Δrealtime − Δmonotonic < −0.05 s` 인 사건을 센다. 런별 건수를 산출물에 기록.

---

## 3. 프레임 구간 정렬 (사전 결정)

WS 는 초기화 창(2·Tp) **이후**의 tracking 프레임만 갱신하고, 베이스라인은 논문 프로토콜상
**전 프레임**을 순회한다. 프레임 구성이 다르면 per-frame 비가 오염된다.

⟹ **PRIMARY 집계는 `t ≥ t_init_end` 인 프레임만** 사용(WS 가 시간을 재는 바로 그 구간).
베이스라인은 전 프레임을 돌리되(스핀업 보존, 프로토콜 충실) **집계에서만** 잘라낸다.
`t < t_init_end` 구간도 기록하여 SECONDARY 로 보고.

---

## 4. 지표 · 열화 방향 · 집계 · MDE

★ **집계는 median 을 PRIMARY 로 사전 지정**한다. 이유(측정됨): 이 박스의 `CLOCK_REALTIME` 은
~32.5 s 마다 ~1.25 s 뒤로 스텝하고, 그 이산 사건이 **평균을 오염**시키지만 **중앙값은 사실상 면역**이다
(mean 11.008→11.329 vs median 10.062→10.081). mean 은 **오염 지표로만** 보고한다.

| # | 지표 | 열화 방향 | 집계 (PRIMARY) | 이 n 에서의 MDE |
|---|---|---|---|---|
| M1 | per-frame `update()` 지연 [ms] | ↑ 나쁨 | 팔별 **pooled median**(전 프레임·전 ep·전 rep), 부수로 p95 / mean | 프레임 n≈12×3×160≈5,760 ⟹ median 의 부트스트랩 95% CI 반폭 **≲1%**. 실질 하한은 시계 해상도 아님 |
| M2 | 팔 대 WS **비** (paired, per-episode) | — (양방향) | ep 별 median-of-3-rep 을 취해 12쌍 → **비의 median + 부트스트랩 95% CI**(10,000 percentile) | n=12. 파일럿 CV≈5% 가정 시 CI 반폭 ≈1.96·0.05/√12 ≈ **±3%** ⟹ **비 차이 >6% 분리 가능**. 방향 검정 = 부호검정, 12/12 시 양측 **p=4.9e−4**(바닥) |
| M3 | **≤100 ms 예산** 충족 | 초과 = 실패 | 팔별 `update+predict` per-frame **p95** vs 100 ms | 이진 판정. **변별력 지표 = 통과 팔 수 / 5**. 5/5 통과면 "예산은 기여가 아니라 전제"(A′-3)로 확정 |
| M4 | 초기화 지연 [s] | ↑ 나쁨 | 팔별 ep 단위 median(3rep) → 12ep median + range | n=12. 배율이 10⁰ vs 10¹ 수준이면 부호검정 12/12 로 충분 |
| M5 | 시계 편향 = (time.time 보고) / (perf_counter 실측) − 1 | — | 팔별 median, 예측값 **−3.86%**(=1255.5 ms / 32,496 ms) | 예측이 팔 무관 상수라는 **A-1 주장의 직접 검정**. 팔 간 차이가 CI 안에서 0 이면 A-1 확증 |
| M6 | 스텝 삼킴 건수 | — (계측 건강) | 런별 건수 합 | 기술통계 |

**부수 기록(가설 없음, 사전 선언)**: `predict` 지연, 프레임당 관측점 수, WS `final_n`,
런별 총 wall time, 워밍업 5프레임의 지연.

---

## 5. 사전 공약된 판정문 (결과에 따라 자동으로 결정되는 것)

- **P1 (배율)**: M2 의 비 median 과 CI 를 그대로 인용한다.
  비가 [1.0, 1.5) 면 "우위 없음~미미", [1.5, 3) 면 "2배 내외", ≥10 이면 비로소 "order-of-magnitude".
  ⛔ **"~10×" 를 되살리려면 CI 하한이 10 을 넘어야 한다.** 그 외에는 철회 유지.
- **P2 (예산)**: M3 에서 5/5 통과면 **"≤100 ms 예산은 변별력이 없다"** 를 결론으로 쓴다(강등 확정).
  1개 이상 탈락하면 예산이 실제로 팔을 가른다는 뜻이고, 그때만 기여로 승격 가능.
- **P3 (초기화)**: M4 를 **반드시 프레임당 지연과 분리하여** 보고한다. WS 가 초기화를
  지불하고 필터가 지불하지 않는다는 사실은 지연 절에 **명시**한다(논문에 현재 0회 언급).
- **P4 (시계)**: M5 가 팔 무관 상수를 확증하면, 시계 결함은 **비를 바꾸지 않는다**는 A-1 을 재확인하고
  ⛔"WS 만 오염" 인용금지를 유지한다. 팔 간 차이가 유의하면 A-1 을 **철회**한다.

## 6. 범위 · 한계 (사전 고지)

- 단일 박스(AMD Ryzen 7 7800X3D, WSL2) 단일 구성. **이식성 주장 없음.**
- `1 스레드` 는 논문이 스스로 쓴 조건이자 공정 비교 조건이지 **최적 배포 구성이 아니다**.
  멀티스레드 레짐의 배율은 여기서 측정하지 않는다(A-2 가 370배 민감성을 이미 보였다).
- GPU 경로 미측정. 전 팔 CPU.
- 정확도는 이 문서의 범위 밖. 타이밍 harness 는 metric 을 계산하지 않는다(계산 비용 오염 방지).
