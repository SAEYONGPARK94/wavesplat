# 시뮬레이터 데이터 현실성 검토 (2026-07-02)

> 사용자 의문 2건에 대한 근거 기반 검토.
> (1) HOS 비선형 성분이 주도적이지 않은데 추정이 안 맞는 게 "너무 현실과 다른 파"를 보기 때문 아닌가?
>     형태가 안 맞아도 에너지는 맞는 편인가?
> (2) 우리 합성 데이터가 현실 바다의 성분·형태를 잘 반영하는가?
>
> 산출 스크립트: `tmp/review_simulator_realism.py`, `tmp/hos_residual_decomp.py`
> 산출 데이터: `tmp/simulator_realism.json`
> 참조 결과: `results/direction3/steepness_nonlinearity.json`, `results/stokes2_gap.json`, `results/hos_vs_linear_gap.json`

---

## Q1. 에너지 vs 형태 — 그리고 "자잘한 파(bound harmonics)"의 실제 기여

### bound harmonic은 오차의 주범이 아니다 (3~5%)
비선형 HOS는 dispersion shell **밖**(합/차 파수)에 위상고정된 bound harmonic("자잘한 파")을 만든다.
우리 추정기는 전부 dispersion-constrained라 이걸 표현 못 한다. 이 성분을 **깨끗이 격리**하는 방법은
dense GT에 최적 dispersion-fit을 한 뒤 남는 잔차(`steepness_nonlinearity.json`의 `disp_ceiling`):

| steepness ε | 0.02 | 0.06 | 0.12 | 0.20 |
|---|---|---|---|---|
| disp-ceiling (bound-harmonic 분율, nRMSE) | 0.054 | 0.029 | 0.030 | 0.039 |
| 실제 추정기(RELAX) nRMSE | 0.239 | 0.293 | 0.285 | 0.281 |

→ **bound harmonic은 nRMSE의 3~5%만 설명. 추정 오차(24~29%)의 주범이 아니다.**
   주범은 (i) 희소 역문제(60점 → 전장 복원), (ii) 합성↔실측 도메인 갭(아래 Q2).
   **사용자의 의심("HOS 성분이 주도적이지 않다")이 맞다.**

### 에너지는 대체로 맞고, 형태가 크게 틀린다
- **Forward 재현**(`stokes2_gap`, `hos_vs_linear_gap`): `Hs_hos ≈ target Hs`로 **<1%** — bound harmonic은
  순 에너지를 더하지 않고 재분배할 뿐이라 Hs(에너지)는 보존.
- **우리 추정기 잔차 decomp**(강비선형 Hs 2.0~3.5, `tmp/hos_residual_decomp.py`):
  - Hs 오차 11~17%, 분산비 est/GT = 0.69~0.79 → 에너지 크기는 대략 맞으나 **~30% 과소평가(smoothing)**.
  - **상관계수 corr = 0.19~0.30** → 강비선형+희소 설정에서 **형태(위상) 복원이 사실상 실패**.
  - ⚠️ 이 스크립트의 `k_split=1.5·kp` 밴드분할은 bound harmonic 격리에 **부적절**(자유파 JONSWAP 고주파
    꼬리 k≤~9·kp까지를 "high-k"로 오분류). bound harmonic 결론은 위 disp-ceiling만 신뢰할 것.

→ **답: 에너지는 대체로 맞음(과소평가 경향), 형태는 강비선형+희소에서 크게 틀림. 단 그 이유는
   HOS 자잘한 파가 아니라 희소 역문제.**

---

## Q2. 합성 데이터 현실성 (측정값)

### 생성 방식 (`wavesim/wave_generator.py::jonswap_components`)
- JONSWAP 스펙트럼(Hasselmann 1973), γ=3.3, ω∈[0.5, 3.0]·ωp.
- **방향 spreading: Gaussian, std 기본 30°, 평균방향 0°** — `pipeline.py`가 인자를 안 넘겨 **항상 30° 고정**
  (config 오버라이드 없음). **주파수 독립·단봉(unimodal)**.
- Gerstner(trochoidal) + Q steepness. HOS 모드는 M=3 bound harmonic 추가.
- Sea state: BF2 Hs0.3/Tp4, BF3 0.6/5.5, BF4 1.2/7, BF5 2.0/9. n_components ∈ {1,2,5,15,30}.

### 측정 (seed0, `tmp/simulator_realism.json`)
| sea | Hs | Tp측정 | 방향분산(에너지가중 σ) | ε | aperture(peak λ) |
|---|---|---|---|---|---|
| BF2(1comp) | 0.30 | 2.3* | **0.0°** (단일파) | 0.116 | 29 λ |
| BF3 | 0.60 | 4.4 | **17.7°** | 0.062 | 8 λ |
| BF4 | 1.2 | 6.3 | **19.1°** | 0.064 | 4 λ |
| BF5 | 2.1 | 8.5 | **19.8°** | 0.062 | 2.2 λ |

\* BF2 1-comp은 단일 성분이 [0.5,3]ωp의 중앙(1.75ωp)에 놓여 **off-peak 단일 정현파** = degenerate.
설계 30° Gaussian이 이산+에너지가중 후 **실효 ~18~20°**로 나타남(전 sea-state·전 seed 동일).

### 실측 Bergamasco Black Sea 기록과 비교
| | 방향분산 | Tp | Hs | aperture | 비고 |
|---|---|---|---|---|---|
| **우리 합성** | ~19° | 4~8.5s | 0.3~2.3 | 2~8 λ | 좁음·long-crested·단봉·고정 |
| **실측 Black Sea** | **~109°** | 3.3s | 0.29~0.44 | ~2.2 λ | 극단 short-crested·다방향 |
| 일반 wind-sea(문헌) | 30~50° | — | — | — | 우리보다 넓음 |

### 현실성 진단
**강점(물리 타당):** JONSWAP 스펙트럼, moderate steepness(ε~0.06), HOS 비선형, 깊은수심.

**약점(현실성 갭, 중요도 순):**
1. **방향 구조 = 최대 약점.** 실효 ~19° 단봉 Gaussian, 주파수 독립, swell+windsea 혼합/crossing sea 없음.
   실측(109°)은 물론 일반 wind-sea(30~50°)보다도 **좁다(long-crested 편향)**. 전 에피소드가 사실상
   **한 방향분포** → directional 일반화 검증 불가.
2. **저성분(1~2 comp)** = 단일/이중 정현파, 비대표적 → 헤드라인·검증에서 제외 권장.
3. **정상성·무-breaking·무-forcing.** 짧은 nowcast엔 허용, 장기 예보·강비선형엔 한계.

---

## 종합 함의
- 추정이 안 맞는 주원인은 **HOS 자잘한 파(3~5%)가 아니라** (i) 희소 역문제, (ii) **합성(좁은 long-crested)
  ↔ 실측(넓은 short-crested) 도메인 갭**. → 사용자 의심 방향이 옳다.
- 에너지(Hs)는 대체로 맞으나 강비선형+희소에서 형태(corr)는 붕괴.

## 다음 단계 (권장)
1. **방향 다양성 주입**: sea-state별 spread 샘플링(20~60°), 주파수 의존 cos-2s spreading, 일부 bimodal
   (swell+windsea). → 현실 방향구조 반영 + directional 일반화 검증 가능.
2. **1~2 comp 에피소드**는 controlled test로만, 헤드라인 제외.
3. **실측 데이터로 검증/학습** (아래) — GT(dense elevation field) 보유.

## 실측 데이터 현황
- **Guimarães, P.V. et al. (2020),** *A data set of sea surface stereo images to resolve space-time wave
  fields*, **Scientific Data 7:145**, DOI 10.1038/s41597-020-0492-9 (Ifremer/SEANOE 공개).
  사이트: Black Sea·Adriatic(Acqua Alta)·Yellow Sea·Iroise. up to 15~20 Hz, ~10⁴ m².
  post-processed **surface elevation map = dense GT** 제공.
- ⚠️ **현재 디스크에 원본 없음** — 이전 세션 scratchpad에서 처리 후 삭제(파일럿 스크립트도 소실).
  → **재다운로드 필요, 이번엔 레포에 저장**(`data/real_bergamasco/` 예정).
- 참조 기록: `results/PILOT_realdata_bergamasco_2026-07-01.md` (게이트 b = PARTIAL/미확정).
