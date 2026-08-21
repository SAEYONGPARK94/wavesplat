# B-1 증거표 — **원자료에서 기계 생성** (손으로 옮긴 숫자 없음)

생성 스크립트가 이 파일을 만들었다. 숫자가 의심되면 아래 경로를 직접 열어라.

## 1. `traj_step1b_relax.json` — 오라클 k vs 추정 k (24 ep)

| 팔 | 오라클 k nRMSE | 정지보다 나은 | 오라클 CI | 추정 k nRMSE | 정지보다 나은 | 추정 CI |
|---|---|---|---|---|---|---|
| stationary | 0.0456 | 0/24 | [+0.0000,+0.0000] | 0.2403 | 0/24 | [+0.0000,+0.0000] |
| straight-cross | 0.0603 | 11/24 | [-0.0312,-0.0001] | 0.3065 | 4/24 | [-0.1028,-0.0300] |
| greedy | 0.0409 | 12/24 | [-0.0038,+0.0128] | 0.2997 | 4/24 | [-0.0945,-0.0227] |
| CEM | 0.0250 | 21/24 | [+0.0137,+0.0269] | 0.2504 | 12/24 | [-0.0438,+0.0238] |
| learned | 0.0273 | 22/24 | [+0.0120,+0.0243] | 0.2794 | 7/24 | [-0.0732,-0.0040] |

**verdict**: `FAIL(clean): oracle advantage (CEM +0.021) DISAPPEARS under estimated wavenumbers = oracle-wavenumber artifact`

**궤적 공간 범위 (m)**: stationary 20x20 · straight-cross 116x122 · greedy 98x126 · CEM 114x103 · learned 112x106

## 2. `traj_step1_netmmse.json` — 비오라클 확인 (24 ep, 8.0 m/s)

speed=8.0 smult=3.25 note=`net-MMSE (beamform kp, non-oracle) achievability, linear seas`

| 팔 | nRMSE | 정지보다 나은 | CI |
|---|---|---|---|
| stationary | 0.1838 | 0/24 | [+0.0000,+0.0000] |
| straight-cross | 0.2339 | 0/24 | [-0.0622,-0.0396] |
| greedy | 0.2242 | 0/24 | [-0.0540,-0.0293] |
| CEM | 0.1908 | 10/24 | [-0.0305,+0.0169] |
| learned | 0.2067 | 7/24 | [-0.0409,-0.0027] |

**verdict**: `FAIL: advantage washed out under net-MMSE = oracle-wavenumber artifact`

## 3. `platform_aperture.json` — 정보(CRB) 측

군속도 c_g=5.231 m/s.  속도 그리드 = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

**방향 CRB (deg) — 낮을수록 정보 많음**

| 기하 | 0 m/s | 1 m/s | 2 m/s | 3 m/s | 5 m/s | 7 m/s | 10 m/s |
|---|---|---|---|---|---|---|---|
| (이상) along-track (∥ wave) | 0.260 | 0.252 | 0.268 | 0.243 | 0.256 | 0.246 | 0.282 |
| (이상) cross-track (⊥ wave) | 0.260 | 0.253 | 0.239 | 0.222 | 0.184 | 0.153 | 0.119 |
| (이상) diagonal 45° | 0.260 | 0.246 | 0.247 | 0.230 | 0.153 | 0.112 | 0.099 |
| **(현실) along-track** | **9.61** | **12.36** | **16.84** | **23.28** | **23.55** | **27.09** | **18.97** |
| **(현실) cross-track** | **9.61** | **9.48** | **9.60** | **7.48** | **3.88** | **2.73** | **2.16** |

**verdict**: `CONFIRMED: cross-track platform motion collapses the directional CRB (≥2× sharper); along-track does not / hurts. Platform-motion synthetic aperture is the directional cure.`

## 4. 오늘(08-12) B-3 이 잰 개구 요구량

| 개구 | 45 | 60 | 90 | 120 | 150 | 180 | 195 |
|---|---|---|---|---|---|---|---|
| AUC | 0.596 | 0.612 | 0.598 | 0.667 | 0.746 | 0.823 | 0.843 |

게이트: AUC≥0.90 판정가능 · 0.70–0.90 약함 · <0.70 판정불가.

## 5. 오늘 잰 조우기하 (런 0회, `data_s9_crowded_2026-08-11`)

- BF4 피크 위상속도 **c_p = 10.77 m/s** ⟹ 설계선속 11.3 m/s 에서 **U/c_p = 1.049** (서프라이딩 임계)
- 에피소드 = 30.0 s, 301 프레임. 프레임당 유효 반사점 중앙 50/71.
- **정지 배도 30 s 동안 173x197 m 를 훑는다** (파면 상하동으로 반사점 이동). 피복 4,188 m².
- following 11.3 = 19,024 m²(4.5x) · crossing 11.3 = 12,396 m²(3.0x) · **과거 실패한 cross 8.0 = 10,068 m²(2.4x)**

