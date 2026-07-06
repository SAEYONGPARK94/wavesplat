# 학습 방법·하이퍼파라미터 개선 여지 검토 (2026-07-02)

> 별도 결론(코드 버그 검토): 결론을 바꿀 correctness 버그 **없음** (RELAX·fit/render 부호·forecast·
> GP·metrics·전치·누수 전부 clean). 이 문서는 **버그가 아니라 튜닝/방법 개선 headroom**.
> 핵심 관점: **sim에 맞춰진 파라미터가 실측·복잡해로 전이될 때 "회피 가능한 gap"을 정보한계 위에
> 얹고 있는지**를 가른다.

## RELAX 추정기 (헤드라인 nowcast)
1. **`beamform_top` klim=0.6 rad/m 하드코딩** (never overridden). 파수 탐색 [-0.6,0.6]에 국한.
   - swell/BF4·5(kp~0.06~0.09)엔 충분, 하지만 **짧은 wind-sea(BS kp≈0.45, 스펙트럼 ~1.3)엔 너무 낮아
     대부분 성분을 seed 못 함** → 실측 BS nowcast를 인위적으로 악화 가능.
   - → **적응적 klim ≈ 3~4×kp(레코드별)** 또는 스펙트럼에서 설정.
2. **`refine_params`: 200 iters, lr=0.01, SINGLE-START, amp reg 1e-3.**
   - 비볼록(off-grid kx,ky+계수) 최적화를 **단일 시작점**으로 200스텝 → 국소최적 위험(원래 d16→d17
     불안정의 근원, warm-start로 *부분* 완화). **관찰된 variance ratio 0.69~0.79(진폭 과소)의 유력 원인.**
   - → iters↑ / LR 스케줄 / 소수 restart / 더 나은 init / 수렴 리포트.
3. **정규화 불일치·비스케일**: refine_params reg=1e-3 vs `fit_amplitudes_phases` lambda_reg=0.5(절대값).
   - fact-check: 헤드라인 M~3000에선 DᵀD(~M/2=1500) 대비 0.5는 **무시 수준**(과수축 원인 아님). 단
     **저-M(희소)에선 진폭 수축** → 레짐 의존적. → **reg를 trace(DᵀD)/2N 또는 Hs²에 스케일.**
4. beamform 그리드 n=80 over [-0.6,0.6] → dk=0.015 (거침). off-grid refine이 보정하므로 minor.
5. MCAP=3000 obs 캡 — 적정.

## GP space-time (융합/예보)
6. **`SIGMA_OBS=0.02 m` 고정** (모든 GP 스크립트). sim 잡음(0.02)에 맞춤 → **실측 스테레오엔 과소**
   (실측 잡음 ~0.1~0.5m, 고주파 변동 ~17%). under-noise → GP가 잡음 과적합 → 실측 nowcast 악화.
   - → **데이터적응 SIGMA_OBS**(고주파 잔차/wave gauge로 잡음바닥 추정).
7. 커널 진폭 A = WaveSplat 추정(오라클 아님, 정상). #2·#3의 진폭 편향과 연동.

## Diffusion / DPS
8. **DPS `GUID=0.1` 고정**, 스윕 없음. DPS는 guidance 스케일에 매우 민감 → **guidance 스윕/anneal.**
9. prior 학습 120 epochs, lr 2e-4 — under-train 가능(더 길게/큰 모델 시험).

## 교차 이슈
10. **계통적 진폭 과소(VR<0.8)** — 분산제약 적합 전반. → 사후 variance/Hs 매칭 또는 #2/#6 개선.
11. **수심**: 실측 중간수심(BS 30·AA 17·YS 50·LJ 30~60m, La Jument kd≈0.7)엔 정확 수심 필수
    (realdata_analysis.py에서 수정 완료).
12. **"튜닝 gap" vs "정보벽" 분리**: 현재 파라미터가 정보한계 *위에* 회피가능 gap을 얹었을 수 있음.
    정보한계 결론 전에 **한 레코드에서 개선 파라미터로 재실행** → 여전히 naive 못 이기면 진짜 정보벽,
    개선되면 벽의 일부는 튜닝이었음.

## 권장 첫 실험 (저비용·고가치)
- **A.** La Jument nowcast에서 **적응 klim + refine iters↑ + 소수 restart** → nRMSE·VR 변화 측정.
- **B.** 실측 GP에 **데이터적응 SIGMA_OBS**.
- **C.** DPS **guidance 스윕**.
→ A가 헤드라인·실측 결론에 직접 영향이라 최우선.

---

## 실험 A 결과 (La Jument, 2026-07-02) — **모델차수 미제어가 1차 원인 확증**
long-crested 대서양 폭풍 swell (Hs3.9m·Tp12.8s·패치40m·실제 λ~256m→개구비 0.11λ):

| 방법 | N | nRMSE | corr |
|---|---|---|---|
| dense-GT dispersion 천장 | 7 | **0.049** | **0.980** |
| 저차수 RELAX (max_comp=4,prune0.3) | 2 | 0.393 | 0.753 |
| 기본 RELAX | 20 | 0.530 | 0.220 |
| naive 보간 | — | 0.160 | 0.769 |

- **dense 천장 0.05 → 필드는 dispersion으로 완전 표현 가능**(표현벽 아님).
- **기본 RELAX 0.53의 대부분은 과적합**: N20→N2로 **0.53→0.39, corr 0.22→0.75(3배)**. → **#2(모델차수 미제어)가
  실증됨. "nowcast 안 됨" 비관 결론이 튜닝으로 부풀려진 부분 있음.**
- 단 최선 튜닝(0.39)도 naive(0.16) 못 이김 → 개구비<1λ면 필드가 매끈 → 보간 near-optimal, 물리 이점 소멸.

### nowcast 실패의 원인 순위 (실증 기반)
1. **모델차수 미제어 → 과적합** (가장 큰 레버, 고칠 수 있음). La Jument서 확증. HOS 복잡해 decomp의 corr
   0.19~0.30·VR<0.8도 동일 징후 → **복잡해·실측 nowcast 전반에 영향 가능.**
2. **개구비(aperture-per-wavelength)** — patch<~1-2λ면 필드 smooth·near-planar → naive near-optimal, 물리
   식별성 이점 소멸 (근본적, CRB 일관).
3. **방향 식별성**(short-crested) — BS 분석 대기.
4. **실측 스테레오 잡음** — band-limit 효과 미미(파일럿) — 정량 대기.

### 다음 (확정 필요)
- **모델차수 제어(BIC/CV/강한 prune)를 estimator에 구현 → 합성 헤드라인 + 실측 전 레코드 재실행.**
  헤드라인(0.119)이 이미 적정 N이면 영향 적고, 과차수면 개선 → **헤드라인 재검증 필수.**

---

## 실험 결과 (BS_2011, 2026-07-02) — klim·모델차수 fix 실증 + 냉정한 nowcast 판정
short-crested wind-sea (패치14m·Hs0.23·Tp2.4s·방향분산 **27°**(파일럿 109° 아님)·개구비 ~1λ).
`relax_step`에 `klim` 파라미터 추가(기본 0.6, backward-compatible) — BS는 스펙트럼 k~1.35라 klim=0.6이 절반 못 봄.

**단일프레임(운 좋음, 참고용):** default 0.183 → klim-fix 0.168 → +차수제어 0.162/corr0.80 vs naive 0.200.

**확증(5프레임×3seed, 공정 in-hull 채점):**
| method | nRMSE | corr |
|---|---|---|
| default RELAX (klim0.6) | 0.189±0.024 | 0.648 |
| **fixed (klim1.5 + 차수제어)** | **0.178±0.017** | **0.746** |
| naive interp | 0.185±0.018 | 0.670 |
→ fixed가 naive 이김 **9/15, margin +0.007(미미).**

**정직한 결론:**
1. **fix가 WaveSplat 자체를 확실히 개선**(0.189→0.178, corr 0.65→0.75) → methods-review #1·#2 유효.
2. **단 nowcast는 fixed vs naive 대략 무승부**(nRMSE 동률, corr만 소폭 우위). "WaveSplat이 naive를 이긴다"는 과장.
3. **어제 "naive 확실 우세/정보벽" → "대략 무승부"로 완화** — method artifact가 pessimism 부풀린 건 맞으나 물리가
   nowcast를 이기는 것도 아님.

**(방향성 × 개구비) 정리:** La Jument(0.11λ) 물리 짐 / BS(~1λ) 무승부 → **개구비가 binding.** 헤드라인은 naive가
원리적으로 못 하는 **forecast·unobserved 외삽**에서 나야 함(La Jument·BS 공통 시사).
⚠️ 교훈: 단일프레임 흥분 금지, 항상 multi-seed + 공정(hull내) 채점.
