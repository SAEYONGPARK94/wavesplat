# 타이밍 전면 재측정 — 결과 (2026-08-05)

사전등록: [`PREREG.md`](PREREG.md) (런 **전** 작성) · 하네스: `harness.py` · 분석: `analyze.py`
원자료: `results.jsonl` (180 런, **오류 0**) · 집계: `summary.json` · 로그: `harness.log`
부록: `addendum.py`/`addendum_summary.json` · `contention_probe.py`/`contention.out`

> **범위**: 이 문서는 **타이밍만** 다룬다. 정확도 수치는 하나도 재계산하지 않았다
> (metric 계산 자체가 실제 연산이라 측정 대상을 오염시킨다).
> **논문 파일·`configs/`·프로덕션 기본값 무변경.** 프로덕션 코드 수정 **0줄**.

---

## 1. 사전등록 요약

- **PRIMARY 집계 = median** 을 런 **전에** 고정. 근거(측정됨): 이 박스의 `CLOCK_REALTIME` 이
  ~32.5 s 마다 ~1.25 s **뒤로 스텝**하고, 그 이산 사건이 **평균을 오염**시키지만 **중앙값은 면역**이다.
- 지표 6개(M1 프레임지연 / M2 비 / M3 예산 / M4 초기화 / M5 시계편향 / M6 스텝) 각각에
  **열화 방향·집계·이 n 에서의 MDE** 를 표로 사전 명시.
- **사전 공약**: "~10×" 를 되살리려면 **비의 CI 하한이 10 을 넘어야 한다**. 그 외에는 철회 유지.
- 프레임 구간 정렬을 **사전 결정**: WS 는 초기화 창 이후만 갱신하므로 **PRIMARY 는 `t ≥ t_init_end`**
  프레임만 집계(베이스라인은 전 프레임을 돌리되 집계에서만 자름) + 각 런의 **첫 5프레임 워밍업 제외**.

## 2. 계측 위생 — 검증 결과

| 항목 | 사전 공약 | 검증 |
|---|---|---|
| 시계 | `time.perf_counter()` | ✅ 전 타이밍. 프로덕션 `time.time()` 값도 **나란히 기록**해 편향을 직접 측정(M5) |
| 프로덕션 무변경 | harness 가 **바깥에서** 감쌈 | ✅ `wavesplat/`·`wavesim/` 수정 0줄. 래퍼 오버헤드 **≤ 15 µs**(M5 의 `prod MEDIAN` −0.03~−0.21%) |
| 프로토콜 동일성 | — | ✅ harness WS 경로 vs `run_estimation`: `final_n` 15=15, `init_iters` 900=900, tracking 190=190 **일치** |
| 스레드 | 5개 변수 전부 =1 | ✅ 라이브 프로세스 `/proc/<pid>/environ` 확인, `torch.get_num_threads()=1`, **CPU 96.2% ≈ 정확히 1코어** |
| 직렬 | 워커 1개 | ✅ 라이브 `pgrep -P` = **자식 0개**. `multiprocessing` 미사용 |
| 같은 세션 | 전 팔 1 프로세스 | ✅ 180 런 전부 단일 파이썬 프로세스, 순서 `rep → ep → arm` 3회 동일 |
| 경쟁 부하 | 없음 | ✅ 런 중 차상위 프로세스 6.9% |

## 3. 결과표

**하드웨어**: AMD Ryzen 7 7800X3D 8-Core (16 logical), WSL2 Linux 6.18.33.1
**소프트웨어**: Python 3.12.3 · numpy 2.5.0 (scipy-openblas 0.3.33) · torch 2.12.1 · scipy 1.18.0 · sklearn 1.9.0
**설계**: 12 ep × 3 rep × 5 arm = 180 런 / 팔당 5,970–6,186 프레임 / 총 38.8 분

### 3.1 per-frame `update()` 지연 [ms] — PRIMARY

| arm | **median** | 95% CI | p95 | mean(오염지표) | ep 범위 |
|---|---|---|---|---|---|
| **WS-P1** | **6.19** | [6.09, 6.27] | 6.69 | 6.12 | 6.00–6.33 |
| ETKF | 10.22 | [10.20, 10.25] | 10.74 | 10.28 | 10.18–10.26 |
| LETKF | 16.39 | [16.33, 16.42] | 17.00 | 16.41 | 16.29–16.47 |
| GPR-1 | 18.40 | [18.17, 18.92] | 26.12 | 19.10 | 17.04–20.13 |
| GPR-disp | 57.75 | [55.66, 64.61] | 94.43 | 62.77 | 51.97–73.56 |

### 3.2 per-frame **총** 비용 (update + GT 격자 필드 생산) [ms]

| arm | median | p95 | mean | predict | interp |
|---|---|---|---|---|---|
| **WS-P1** | **6.48** | 6.98 | 6.38 | 0.29 | — |
| ETKF | 10.44 | 10.97 | 10.50 | 0.05 | 0.17 |
| LETKF | 16.61 | 17.22 | 16.63 | 0.05 | 0.17 |
| GPR-1 | 19.65 | 27.43 | 20.31 | 1.20 | — |
| GPR-disp | 65.59 | 105.73 | 70.93 | 5.83 | — |

### 3.3 WS 대비 비 (에피소드 단위 paired, n=12)

| arm | update only | 95% CI | update+필드 | 95% CI | WS 보다 느린 ep | 부호검정 p |
|---|---|---|---|---|---|---|
| ETKF | **1.65×** | [1.63, 1.68] | **1.62×** | [1.59, 1.65] | 12/12 | 4.9e−4 |
| LETKF | **2.65×** | [2.62, 2.69] | **2.57×** | [2.54, 2.63] | 12/12 | 4.9e−4 |
| GPR-1 | **2.97×** | [2.92, 3.10] | **3.05×** | [2.99, 3.14] | 12/12 | 4.9e−4 |
| GPR-disp | **9.38×** | [8.97, 10.46] | **10.21×** | [9.90, 11.29] | 12/12 | 4.9e−4 |

★ **사전 공약 적용**: ETKF 비 CI 상한 **1.68 ≪ 10** ⟹ **"~10×" 는 되살아나지 않는다. 철회 확정.**
순위(WS 가 5팔 중 가장 빠름)는 **12/12 로 전 팔에 대해 성립**하고 이것이 살아남는 진술이다.

### 3.4 부수 발견 — WS 프레임 비용은 사실상 **N 에 평평하다**

관측 `final_n` 범위 **1–22**(36 런). `final_n` vs per-frame median: **r = 0.694, p < 1e−4**
(효과는 실재) 이지만 **기울기 0.038 ms/성분** ⟹ N=1→22 전 구간에서 **0.8 ms(≈13%)** 만 움직인다.
N=1–2 런 median **6.00 ms** vs N≥19 런 **6.33 ms**.
⟹ 복잡도 항 `O(K·K_w·M·N)` 은 **측정되지만 지배하지 않는다**. 이 규모에서는 고정 오버헤드가 지배한다.

## 4. ≤100 ms 예산 — 판별력이 **있다** (강등 판정 뒤집힘)

기준(사전 지정): per-frame **총** 비용(update + 필드 생산)의 **p95 ≤ 100 ms**.

| arm | p95 | 최악 ep p95 | max | 예산 초과 프레임 | 판정 |
|---|---|---|---|---|---|
| WS-P1 | 6.98 | 7.18 | 10.33 | 0.00% | PASS |
| ETKF | 10.97 | 11.09 | 14.58 | 0.00% | PASS |
| LETKF | 17.22 | 17.28 | 26.48 | 0.00% | PASS |
| GPR-1 | 27.43 | 29.39 | 42.56 | 0.00% | PASS |
| **GPR-disp** | **105.73** | **120.43** | **173.15** | **7.06%** | **FAIL** |

⟹ **4/5 통과 = 예산이 팔을 실제로 가른다.**
★ 토론의 A′-3 **"전 팔 통과 ⟹ 변별력 없음"** 은 **정정된다** — 단, 변별되는 것은 WS 가 아니라
**GPR-dispersion 하나뿐**이고, 그것도 median(65.6 ms)이 아니라 **꼬리(p95)** 에서만 탈락한다.
정직한 진술은 "예산이 기여를 만든다"가 아니라 **"예산은 팔 5개 중 1개만 탈락시킨다"** 이다.

## 5. 초기화 지연 — 논문에 **0회** 등장하는 항목

| arm | median [s] | 95% CI | min–max |
|---|---|---|---|
| **WS-P1** | **25.3** | [22.3, 31.2] | 16.3–40.9 |
| ETKF | 0.003 | [0.003, 0.003] | — |
| LETKF | 0.003 | [0.003, 0.003] | — |
| GPR-1 / GPR-disp | ~0.000 | — | — |

- **WS 초기화 = 자기 프레임 비용의 3,909배.** 필터·GPR 베이스라인은 **초기화를 지불하지 않는다**
  (생성자만; 스핀업은 계산비용이 아니라 정확도 수렴 문제다 — ARGUED, 여기서 측정 안 함).
- **해상상태에 강하게 스케일**: BF2 16.3 s → BF3 22.3 → BF4 29.7 → **BF5 40.4 s**.
- 기존 `results/compute_cost_comparison.json` 의 **77.4 s** 는 재현되지 않는다(측정 **25.3 s**).
  같은 파일의 WS `inference_ms` 는 5개 에피소드 전부 **하드코딩 13.4** 로, 측정값이 아니다.

## 6. 시계 결함 — 실측

- **스텝 삼킴: 180 런 중 69 런에서 총 69회**(38.8 분). 팔별 {WS 31, ETKF 3, LETKF 7, GPR-1 7, GPR-disp 21}
  — 런 **길이**에 비례할 뿐 팔 정체와 무관.
- **런 수준 편향 = −3.774%** [−4.910, −2.758]. 예측 **−3.86%** 가 CI 안 ⟹ **A-1 확증**.
  팔 간 편차 **1.46 pp** 뿐 ⟹ **팔 무관 상수**가 맞다. ⛔"WS 만 오염" 인용금지 **유지**.
- ★**mean 은 오염되고 median 은 면역** — 직접 측정:

| arm | `time.time()` **mean** 편향 | `time.time()` **median** 편향 |
|---|---|---|
| WS-P1 | **−9.51%** | −0.21% |
| ETKF | −3.73% | −0.03% |
| LETKF | −5.95% | −0.03% |
| GPR-1 | −5.69% | −0.04% |
| GPR-disp | −3.33% | −0.05% |

  median 잔차(−0.03~−0.21%)는 시계가 아니라 **내 래퍼의 호출 오버헤드**(≤15 µs)다.
  ⟹ 프레임당 **평균** 집계(`run.py:274`)는 최대 **−9.5%** 까지 틀릴 수 있고,
  이는 −3.86% 상수보다 **크다**. 이유: 프레임 구간이 짧아 스텝이 **드물지만 치명적인 이상치**로
  들어오기 때문(기댓값은 −3.86% 이나 **분산이 거대**). **median 사전지정이 옳았다.**

## 7. 부록 — 논문의 나머지 타이밍 주장 2건

**ADD-1 앙상블 크기** (1스레드, 4 ep): ETKF `ne=50` **4.96 ms** → `ne=100` **10.07 ms** = **2.03×**.
논문은 같은 문단에서 `ne=50` 을 31.7 ms 라 쓰고 `ne=100` 을 132.6 ms 라 써 **"~4×"** 를 함의한다.
`results/etkf_tuning_sweep.json` 은 2.81×. **측정값은 2.03×** — 2배 앙상블에 2배 비용, 즉 선형이다.
(31.7 과 132.6 은 **서로 다른 파일·서로 다른 부하 레짐**에서 왔다 ⟹ "4×" 는 측정이 아니라 **파일 교차 아티팩트**.)

**ADD-2 / 경쟁 프로브 ★★ 토론 A-2 의 인과귀속이 틀렸다.** ETKF `ne=100`, 같은 코드·같은 에피소드:

| 스레드/프로세스 × 프로세스 | 요청 스레드(16 논리코어) | median | 1t×1p 대비 |
|---|---|---|---|
| 1 × 1 | 1 | 10.11 ms | 1.00× |
| **16 × 1** | 16 | **12.73 ms** | **1.26×** |
| 1 × 4 | 4 | 10.34 ms | 1.02× |
| 4 × 4 | 16 | 12.92 ms | 1.28× |
| **16 × 4** | **64** | **1983.84 ms** | **196×** |

⟹ **스레드 수 단독으로는 1.26배밖에 못 만든다.** 토론이 "BLAS 스레드 수, 크기 370배"라고
기소한 것은 **과잉 귀속**이다. 진범은 **오버섭스크립션**(요청 스레드 ≫ 코어)이고,
그것은 알고리즘의 성질도 스레드 플래그의 성질도 아니라 **그 순간 박스에서 무엇이 돌고 있었는가**의
성질이다. 토론의 `115.88 ms` 는 12.73 과 1983.84 **사이**에 있어 **부분 오버섭스크립션**과 정합적이다.
⟹ 새 인용금지 후보: ⛔**"ETKF 의 132.6/115.88 ms 는 16스레드 때문"** — 허용은
**"동시 실행 프로세스와 스레드의 곱이 코어 수를 넘긴 배치 레짐에서 나온 값"**.

## 8. ★ 논문에 쓸 수 있는 정확한 문장

> **(A) 지연 본문용 — 조건 전부 명시**
>
> All timings were re-measured in a single dedicated session on one machine
> (AMD Ryzen 7 7800X3D, 8 cores / 16 threads; WSL2 Linux; Python 3.12.3, NumPy 2.5.0
> with scipy-openblas 0.3.33, PyTorch 2.12.1), with `OMP_NUM_THREADS`,
> `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and
> `VECLIB_MAXIMUM_THREADS` all pinned to 1, a single worker process, no concurrent jobs,
> and `time.perf_counter()` (CLOCK_MONOTONIC) as the clock. All five methods were run
> **in the same process and the same session**, over the same 12 episodes × 3 repetitions
> (≈6,000 timed frames per method). We report the **median** per-frame latency: the
> distribution has a heavy discrete tail on this host, and mean aggregates are biased low
> by up to 9.5% while medians are unaffected (≤0.2%).
>
> Per-frame update latency was **6.19 ms** for WaveSplat, **10.22 ms** for the tuned
> ETKF-grid ($n_\text{ens}$=100), **16.39 ms** for LETKF, **18.40 ms** for GPR-1 and
> **57.75 ms** for GPR-dispersion. Paired per-episode ratios against WaveSplat are
> **1.65× (95% CI 1.63–1.68)** for ETKF, **2.65× (2.62–2.69)** for LETKF,
> **2.97× (2.92–3.10)** for GPR-1 and **9.38× (8.97–10.46)** for GPR-dispersion;
> WaveSplat is the fastest arm on **12 of 12** episodes against every baseline
> (sign test $p=4.9\times10^{-4}$). Including field production on the evaluation grid
> changes these ratios by less than 8%.
> **We do not claim an order-of-magnitude speed advantage**: against the primary
> ETKF baseline the measured advantage is a factor of **1.6–1.7**.
>
> **(B) 예산 문장**
>
> At 10 Hz, four of the five methods meet a 100 ms per-frame budget with margin
> (p95 of update + field production: 6.98 / 10.97 / 17.22 / 27.43 ms for WaveSplat, ETKF,
> LETKF and GPR-1). Only GPR-dispersion exceeds it, and only in the tail
> (median 65.6 ms, p95 105.7 ms, 7.1% of frames over budget). The budget is therefore
> **not a discriminating criterion among the grid-free and filter baselines**; we report
> it as a feasibility precondition rather than as a contribution.
>
> **(C) 초기화 문장 — 현재 논문에 없는 비용의 공개**
>
> These per-frame figures exclude initialisation. WaveSplat pays a one-off
> coarse-to-fine fit over the first $2T_p$ of data costing a median of **25.3 s**
> (range 16.3–40.9 s over the 12 episodes, rising with sea state: 16.3 s at BF2 to
> 40.4 s at BF5), i.e. about **3,900× its own per-frame cost**. The ensemble and
> GPR baselines have no comparable initialisation (constructor ≤3 ms). Any real-time
> claim for WaveSplat is therefore a claim about the **tracking** phase after an
> offline or warm-start initialisation, not about cold start.
>
> **(D) 재현성 각주 — 반드시 함께 갈 것**
>
> Per-frame latency on this workload is dominated by host configuration rather than by
> algorithmic cost: for the same ETKF code and episode, the median moves from 10.11 ms
> (1 BLAS thread, 1 process) to 12.73 ms (16 threads, 1 process) to **1,984 ms**
> when four 16-thread processes share 16 logical cores. Latency numbers measured under
> an unpinned, concurrently-loaded batch harness are therefore not comparable with these,
> and we report the pinned single-process configuration throughout.

**⛔ 아직 쓰면 안 되는 것**: WS 6.19 ms 의 **정확도 짝**(이 harness 는 metric 을 안 쟀다) ·
다른 하드웨어로의 이식 · 초기화의 warm-start 대안(미측정) · GPU 경로(미측정).

## 9. 확인 못 한 것 (정직한 공백)

1. **정확도 미측정**. 이 재측정은 타이밍 전용이다. 논문의 지연 문장을 갈아끼울 때
   nRMSE 숫자는 **다른 출처**에서 와야 하고, 그 출처는 이 런과 **다른 런**이다.
2. **단일 박스·단일 OS**. WSL2 이고 네이티브 리눅스도 다른 CPU 도 시험 안 했다. §8(D) 각주가
   이 한계를 흡수하도록 썼다.
3. **GPU 경로 0건.** 전 팔 CPU.
4. **`nproc` provenance 오점**: `results.jsonl` 의 provenance 레코드에 `nproc: 1` 이 박혔다 —
   `nproc` 이 `OMP_NUM_THREADS` 를 존중하기 때문이다. **실제 값은 16 논리 / 8 물리**
   (`/proc/cpuinfo` 로 별도 확인). 원자료를 읽을 때 이 필드를 믿지 말 것.
5. **베이스라인 3반복은 시드 동일**(42). 즉 기계 잡음만 재고 WS 3반복은 시드까지 바꾼다
   — PREREG §1 에 사전 공개한 비대칭. WS 쪽 산포가 넓은 것은 이 설계의 결과다.
6. **필터의 "스핀업"** (수렴까지 걸리는 프레임 수)은 계산비용이 아니라 정확도 문제라
   여기서 재지 않았다. WS 초기화와 대칭 비교를 하려면 **별도 측정이 필요**하다 — 미실행.
7. **경쟁 프로브의 16t×8p 셀**은 사전 계획했으나 런타임이 과도해 완주를 보장하지 않았다.
   결론은 16t×4p(196×)만으로 이미 확정된다.
8. **토론의 `115.88 ms` 자체를 재현하지 않았다.** 오버섭스크립션이라는 **기제**는 보였지만
   그 특정 숫자가 정확히 어떤 (스레드, 프로세스, 동시 부하) 조합에서 나왔는지는 **모른다**.
9. `results/p1_comparison.json` 의 ETKF(26–37 ms)가 기본 `n_ensemble=50` 인지 확정하지 않았다
   (클래스 기본값은 50). 논문의 "31.7 ms = ne50" 귀속은 **미검증**.
