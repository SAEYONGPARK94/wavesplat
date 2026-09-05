# `T16-BANDFIX` 도달성 — **1단 격자를 절대 → 상대로 바꾸는 방법이 출하 팔을 건드리는가**

작성 2026-09-05 · ⛔**추정기 실행 0회** · 빔포머(규칙)만 합성 관측 위에서 돌렸다
(`systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0`, `CUDA_VISIBLE_DEVICES=""`, 1스레드).
⛔이 문서는 **설계 근거**다. 러너·게이트·분석기는 여기 없다.
**열람 범위 = 주장 범위** — 인용한 파일:줄은 전부 이번에 직접 열었다.
⛔**12 GB zip 은 `namelist` 만 열었다** — `.mat` 내용은 한 바이트도 읽지 않았다.

> ★**결론 먼저 (세 줄)**
> 1. **패키지를 고칠 필요가 없다.** `bank_mmse.py:49` 의 `_beamform_top(P, r, depth, klim=…, n=…)` 이
>    `klim` 을 **인자로 받는다** ⟹ 수리는 **하네스의 규칙 모듈** 안에서 끝난다.
>    그리고 그 함수는 배포 학습 경로에서 **실행되지 않는다**(§2 전수 grep) ⟹ 출하 팔은 **한 비트도 안 바뀐다**.
> 2. **채택 = `klim₁ = (n−1)/2 · 2π/L_obs`** ⟹ `Δk₁ = 2π/L_obs`(개구의 Rayleigh 한계). **자유 파라미터 0개.**
> 3. **합성으로 미리 쟀다**(§4, 런 0회): 새 규칙의 등변 편차는 시험한 **58측정 전부에서 `0.000e+00`**
>    (`c=0.5`·`c=2.0`, 수심 고정·재라벨 둘 다). 같은 셀에서 옛 규칙은 최대 **1.1e+03**.

---

## 1. 결함 지도 (파일:줄)

| 위치 | 내용 |
|---|---|
| `wavesplat/optimizer/bank_mmse.py:49` | `def _beamform_top(P, r, depth, klim=0.6, n=80):` — ★**`klim`·`n` 이 인자다** |
| `:60` | `ax = torch.linspace(-klim, klim, n, device=_DEV)` — 격자. 간격 = `2·klim/(n−1)` |
| `:65` | `om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)` — 분산관계로 구속된 빔포머 |
| `:73` | `j = int(torch.argmax(pw))` — **이산 argmax** (등변성이 「같은 칸이 이기는가」로 환원되는 지점) |
| ★`:159-160` | `_beamform_top(…, depth, klim=2.0, n=120)` — ⛔**1단이 절대 격자다.** `Δk₁ = 2·2.0/119 = 0.033613` |
| `:161` | `kp = max(np.hypot(bx, by), 1e-3)` — **절대 바닥**(결속하면 등변성이 구조적으로 깨진다) |
| `:162-163` | `_beamform_top(…, depth, klim=3.5*kp, n=120)` — ★**2단은 이미 상대적이다**(손대지 않는다) |
| `:164` | `kp = max(np.hypot(bx, by), 1e-3)` — 같은 바닥 |
| `tmp/T14_AUTOSCALE_2026-09-04/scale_rules_t14.py` `kp_hat()` | 위 `:159-164` 절차를 **하네스에 복제**한 것. 독스트링이 *"`bank_mmse.py:158-165` 절차와 축자 동일"* 이라 적는다 |
| `tmp/T14_AUTOSCALE_2026-09-04/AMENDMENT_2_POSTRESULT.md §1` | 그 절대 격자 때문에 `T14` 가 `BLOCKED`(`G-EQUIVAR`). `k_p/Δk₁` 중앙 **1.16** · 최소 **0.88** |

**기제 재진술**: `k_p < Δk₁` 이면 1단이 첨두를 담은 칸을 못 고른다 ⟹ `k̂₁` 이 엉뚱한 값이 되고
2단의 `klim = 3.5·k̂₁` 이 통째로 틀어진다. `c = 0.5` 재라벨은 장면의 `k` 를 **절반으로** 만드는데
격자는 그대로이므로 **정확히 그 영역으로 들어간다** ⟹ `T14` 가 잰 `|c=0.5|` 편차 ≈ **0.99**.

---

## 2. 도달 경계 — **패키지를 건드리면 안 되는 이유와, 건드릴 필요가 없는 이유**

### 2.1 `_beamform_top` 을 부르는 곳 (저장소 전수)
```
$ grep -rn "_beamform_top" --include=*.py wavesplat/ scripts/
wavesplat/optimizer/bank_mmse.py:49    def _beamform_top(...)
wavesplat/optimizer/bank_mmse.py:159   ← fit_bank_mmse 안
wavesplat/optimizer/bank_mmse.py:162   ← fit_bank_mmse 안
```
⟹ **패키지 안의 호출자는 `fit_bank_mmse` 뿐이다.**
그리고 `fit_bank_mmse` 는 **실측 게이트 뒤**의 경로다 —
`wavesplat/optimizer/sequential_update.py:794-795` 가 `real_domain_gate` ∧ `not _frames_have_gt(frames)`
일 때만 `:803` 에서 `bank_mmse` 로 보낸다. 이 배치의 팔은 **상수 `gt_field`** 를 붙이므로
`_frames_have_gt`(`:411`)가 참이 되어 라우터가 `learned` 로 간다(`G-ARM` 이 되읽어 증명).
⟹ ★***규칙이 `_beamform_top` 을 어떤 `klim` 으로 부르든, 배포 합성 경로는 그 함수를 한 번도 실행하지 않는다.***

### 2.2 학습 경로가 쓰는 빔포머는 **다른 함수**다
`wavesplat/optimizer/learned_init.py:151 cd._bmap_from_obs(Ps, zf)` → `scripts/d3_beamform_net.py:46 beamform`
이고 그것은 모듈 전역 `KX_G, KY_G, OM_G`(`:43`)를 쓴다. 상수는 `:30 G=9.81; DEPTH=50.0` · `:31 KLIM=0.6; NK=64` ·
`:32 M_OBS=400`. ⟹ **규칙의 `klim` 과 조건화 지도의 `KLIM` 은 서로 다른 격자다.** 섞지 마라.

### 2.3 그래서 처치는 어디에 넣는가
★**`tmp/T16_BANDFIX_2026-09-05/scale_rules_t16.py`(하네스) 안에서 끝난다.**
`T13-DEPTHFIX` 는 처치가 **모듈 전역**(`OM_G`)이라 컨텍스트 매니저로 패치해야 했지만,
여기서는 **인자**이므로 그런 장치가 **필요 없다** — 규칙이 값을 골라 넘기면 그만이다.
⟹ `T13 §3.3` 이 ⛔로 적은 위험(모듈 리로드 · 패키지 경로 임포트 · `sys.modules` 정체 불일치)이
**이 배치에는 하나도 해당하지 않는다.**

⛔**비채택 — 패키지 수정**(`bank_mmse.py:159` 의 `klim=2.0` 을 직접 바꾸기): 설계 단계 금지이고,
바꾸면 **실측 경로의 출하 추정기가 바뀐다**(그 경로가 `fit_bank_mmse` 를 실제로 쓴다).
⟹ **하네스 처치가 `T13`·`T14` 가 지킨 경계이고, 이 배치도 그것을 따른다.** §7 에 미래형만 적는다.

---

## 3. 채택한 규칙 (⛔러너 아님 — 구현자가 이 형태로 쓴다)

```python
"""scale_rules_t16.py — ⛔`PREREG.md §1` 이 정본. 관측 5배열만 받는다."""
import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'T14_AUTOSCALE_2026-09-04'))
import scale_rules_t14 as SR14                 # ★임포트. ⛔손복사 금지 (G-RULEID 가 객체 동일성을 단언)

S_MIN, S_MAX = SR14.S_MIN, SR14.S_MAX          # 0.25 / 4.0
L_TRAIN, K_TRAIN = SR14.L_TRAIN, SR14.K_TRAIN  # 220.55 / 0.06779
s_ap_raw, s_ap = SR14.s_ap_raw, SR14.s_ap      # ★정확히 등변인 옛 규칙 ①  (그대로 재사용)
s_band_raw, s_band = SR14.s_band_raw, SR14.s_band   # ⚠ill-posed 한 옛 규칙 ②  (비교자로만)

N1 = 120        # bank_mmse.py:160 / :163 의 `n=120`
ZOOM = 3.5      # bank_mmse.py:163 의 `3.5 * kp`
FLOOR = 1e-3    # bank_mmse.py:161 / :164 의 `max(·, 1e-3)`


def L_obs(x, y):
    """★`SR14.s_ap_raw` 가 쓰는 것과 **같은 양**이다 (개구)."""
    return max(float(np.ptp(np.asarray(x))), float(np.ptp(np.asarray(y))))


def klim1(x, y):
    """★1단 격자 간격을 관측 개구의 Rayleigh 한계로 고정: Δk₁ = 2π/L_obs.

    `linspace(-klim, klim, n)` 의 간격이 `2·klim/(n−1)` 이므로 `klim₁ = (n−1)/2 · 2π/L_obs`.
    ⛔자유 파라미터 0개 — `n` 은 `bank_mmse.py:160`, `2π/L` 은 개구의 Rayleigh 한계다.
    ⛔결과를 본 뒤 이 식을 바꾸는 것은 금지(새 규칙 = 새 사전등록).
    """
    return 0.5 * (N1 - 1) * (2.0 * math.pi / L_obs(x, y))


def kp_hat_rel(x, y, t, z, depth):
    """2단 줌 — ★1단만 상대 격자로 바꿨다. 2단은 `bank_mmse.py:162-163` 그대로."""
    from wavesplat.optimizer.bank_mmse import _beamform_top
    P = np.column_stack([x, y, t]).astype(np.float32)
    r = np.asarray(z, dtype=np.float32)
    bx, by = _beamform_top(P, r, depth, klim=klim1(x, y), n=N1)
    k1 = max(float(np.hypot(bx, by)), FLOOR)
    bx, by = _beamform_top(P, r, depth, klim=ZOOM * k1, n=N1)
    return max(float(np.hypot(bx, by)), FLOOR)


def s_bandr_raw(x, y, t, z, depth):
    """규칙 ③ **비클리핑** — 첨두 파수를 사전 밴드로, 1단 격자는 개구 상대."""
    return float(K_TRAIN / kp_hat_rel(x, y, t, z, depth))


def s_bandr(x, y, t, z, depth):
    return float(min(max(s_bandr_raw(x, y, t, z, depth), S_MIN), S_MAX))


RULES = {'AUTO-AP': s_ap, 'AUTO-BAND': s_band, 'AUTO-BANDR': s_bandr}
RULES_RAW = {'AUTO-AP': s_ap_raw, 'AUTO-BAND': s_band_raw, 'AUTO-BANDR': s_bandr_raw}
CANDIDATES = ('AUTO-AP', 'AUTO-BANDR')      # ★`G-EQUIVAR` 가 **차단**하는 두 규칙
COMPARATORS = ('AUTO-BAND',)                # ⚠기록 전용(`G-EQUIVAR-OLD`) — 후보가 아니다
```

### 3.1 왜 등변인가 (해석)
재라벨 `x' = x/c, y' = y/c, t' = t/√c` 아래:
- `L_obs' = L_obs/c` (⟹ `ptp` 는 `c` 가 2의 거듭제곱이면 **정확히** 나눠진다)
  ⟹ `klim₁' = c·klim₁` ⟹ **격자 전체가 정확히 `c` 배**가 된다.
- 물리 첨두도 `k' = c·k` 로 옮겨간다 ⟹ **같은 격자 칸이 이긴다** ⟹ `k̂' = c·k̂` (정확히)
  ⟹ `s' = K_TRAIN/(c·k̂) = s/c` ⟹ `|s'·c/s − 1| = 0`.
- 2단은 이미 `3.5·k̂₁` 이므로 자동으로 따라간다.
⟹ ★**등변성이 「같은 argmax 칸이 이기는가」로 환원된다**(`bank_mmse.py:73`). 옛 규칙은 격자가 안 따라가서
칸이 바뀐다. 새 규칙은 따라간다.
⚠**정확히 0 이 아닐 수 있는 자리 둘**: (a) `√c` 가 무리수라 `t'` 에 반올림이 있다
(⟹ 위상 상대오차 ~1e-7, 두 칸이 1e-6 이내로 접전이면 argmax 가 뒤집힐 수 있다) ·
(b) `tanh(k·depth)` 는 `depth` 를 재라벨하지 않으면 정확히 보존되지 않는다.
⟹ **§4 가 두 경우를 다 쟀고, 시험한 58측정 전부에서 편차가 정확히 0 이었다.**
⟹ 그래도 `G-EQUIVAR` 의 허용오차는 **`1e-6`**(0 이 아니라)로 등록한다 — (a)를 알고 있기 때문이다.

### 3.2 ⛔비채택 후보 (기록 — 왜 아닌가)
| 후보 | 왜 안 되는가 |
|---|---|
| `klim₁ = C · 2π/L_obs` (C 를 자유롭게) | 등변하지만 **자유 파라미터 1개**가 생긴다. 채택안은 `C = (n−1)/2` 를 **코드에서 유도**해 그 자유도를 없앤다(⟹ *"C 를 자료 보고 골랐다"* 를 원천 차단) |
| 시간창에서 유도 (`klim₁ = π²/(g·dt²)` = Nyquist) | 등변하지만 (i) `dt` 를 산란 관측점에서 **추정**해야 해 취약하고 (ii) 이 코퍼스에서 `Δk₁ = 2·1.006/119 = 0.0169` ⟹ `k_p/Δk₁ ≈ 1.8–2.9` 로 **채택안(3.4–5.4)보다 나쁘다** |
| **다단(3단) 줌** | 1단이 여전히 절대라 **등변성을 고치지 못한다**(정확도만 는다). ⛔차단 사유를 해소하지 못한다 |
| `bank_mmse.py:159` 를 직접 수정 | ⛔출하 팔이 아니게 된다(§2.3) |
| 바닥 `1e-3` 제거 | 배포 절차에서 이탈한다. **유지하고 `G-FLOOR` 로 미결속을 강제**한다 |

---

## 4. ★합성 등변성 프로브 — **실제로 돌렸다** (추정기 런 0회)

**장면**: 제3자 코퍼스의 기하를 축자로 흉내 — 128×128 격자, `dx = dy = 4.878999999999905 m`,
40 프레임 `dt = 1.0`, 프레임당 200점 균등 무작위(= 8,000점), `depth = 400`,
JONSWAP 유사 30성분(로그정규 `k` 산포 0.22, 방향 산포 20°). ⛔프리즌 표본은 **건드리지 않았다**.
스크립트는 `_beamform_top` 을 **패키지에서 임포트**해 쓴다(⛔재구현 아님).

**장면 생성기 (재현용 — ⛔러너 아님)**
```python
DX = 4.878999999999905; NG, NF, NPTS = 128, 40, 200; G_ACC = 9.81
def scene(kp, seed, ncomp=30, spread_deg=20.0, dirdeg=30.0, hs=2.0, depth=400.0):
    rng = np.random.default_rng(seed)
    gx = (np.arange(NG) - (NG - 1) / 2.0) * DX; gy = gx.copy()
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    kk = kp * np.exp(rng.normal(0.0, 0.22, ncomp))
    th = np.deg2rad(dirdeg + rng.normal(0.0, spread_deg, ncomp))
    kx, ky = kk * np.cos(th), kk * np.sin(th)
    om = np.sqrt(G_ACC * kk * np.tanh(kk * depth))
    amp = np.exp(-(np.log(kk / kp) ** 2) / (2 * 0.22 ** 2))
    amp = amp / np.sqrt((amp ** 2).sum()) * (hs / 4.0)
    ph = rng.uniform(0, 2 * np.pi, ncomp)
    xs = ys = ts = zs = None; XS, YS, TS, ZS = [], [], [], []
    for j in range(NF):
        idx = rng.choice(NG * NG, NPTS, replace=False)
        iy, ix = np.unravel_index(idx, (NG, NG))
        xp, yp = X[iy, ix], Y[iy, ix]; zp = np.zeros(NPTS)
        for c in range(ncomp):
            zp += amp[c] * np.cos(kx[c]*xp + ky[c]*yp - om[c]*float(j) + ph[c])
        XS.append(xp); YS.append(yp); TS.append(np.full(NPTS, float(j))); ZS.append(zp)
    return (np.concatenate(XS), np.concatenate(YS), np.concatenate(TS), np.concatenate(ZS))

def equivar(rule, x, y, t, z, depth, c, relabel_depth):
    k1 = rule(x, y, t, z, depth)
    d2 = depth / c if relabel_depth else depth
    kc = rule(x / c, y / c, t / math.sqrt(c), z, d2)
    return abs(kc / (c * k1) - 1.0)          # `s ∝ 1/k̂_p` 이므로 s 편차와 같은 양이다
```
A: `k_p ∈ {0.0307, 0.0360, 0.0429, 0.0491}`, seed 11 · B: `k_p ∈ {0.010 … 0.200}` 9점, seed 23 ·
C/D: seed `1000+i` (i = 0..11), `k_p ~ U(0.030, 0.050)`, 방향 `10 + 10·(i mod 6)` 도.

```
device of bank_mmse beamform: cpu
L_obs = 619.633 m · 2π/L = 0.010140 rad/m
klim1_new = 0.603340  (Δk₁_new = 0.010140)
klim1_old = 2.0                      (Δk₁_old = 0.033613)

=== A. 실제 코퍼스 대역(k_p 0.031–0.049) · 등변성 ===
    k_p  rule    kp_hat  kp/Δk1   |c=.5|(d고정)   |c=2|(d고정)    |c=.5|(d재라벨)    |c=2|(d재라벨)
 0.0307   old   0.03460    1.03     2.616e+02    1.428e-02       2.616e+02      1.428e-02
 0.0307   new   0.03430    3.38     0.000e+00    0.000e+00       0.000e+00      0.000e+00
 0.0360   old   0.03956    1.18     9.365e-04    1.219e-02       9.365e-04      1.219e-02
 0.0360   new   0.03937    3.88     0.000e+00    0.000e+00       0.000e+00      0.000e+00
 0.0429   old   0.04642    1.38     2.309e-02    1.077e-03       2.309e-02      1.077e-03
 0.0429   new   0.04598    4.53     0.000e+00    0.000e+00       0.000e+00      0.000e+00
 0.0491   old   0.05310    1.58     4.325e-02    1.343e-03       4.325e-02      1.343e-03
 0.0491   new   0.05445    5.37     0.000e+00    0.000e+00       0.000e+00      0.000e+00

=== B. k_p 스트레스 스윕 (어디서 깨지는가) ===
     k_p   old kp̂   old|c.5|    old|c2|   new kp̂ new kp/Δk1   new|c.5|    new|c2|
  0.0100   7.54150  1.433e-01  9.986e-01   0.01054       1.04  0.000e+00  0.000e+00
  0.0150   0.01470  1.138e+03  1.181e-02   0.01463       1.44  0.000e+00  0.000e+00
  0.0210   0.02095  8.256e+02  4.879e-02   0.02046       2.02  0.000e+00  0.000e+00
  0.0310   0.02928  3.895e+02  1.652e-02   0.03015       2.97  0.000e+00  0.000e+00
  0.0430   0.04223  5.320e-02  2.406e-02   0.04028       3.97  0.000e+00  0.000e+00
  0.0600   0.05853  4.242e-04  1.419e-02   0.05816       5.74  0.000e+00  0.000e+00
  0.0900   0.08622  5.880e+01  2.488e-02   0.08686       8.57  0.000e+00  0.000e+00
  0.1300   0.12877  4.125e+01  6.689e-02   0.12347      12.18  0.000e+00  0.000e+00
  0.2000   0.19990  1.301e-02  2.505e-02   0.18879      18.62  0.000e+00  0.000e+00

=== C. 씨앗 30개 · 코퍼스 대역 · 새 규칙만 (분포) ===          ⚠제목 문자열은 낡았다 — 실제 반복은 **12**
  |c=0.5| 중앙 0.000e+00 최대 0.000e+00 · >5.88e-2 인 셀 0/12
  |c=2.0| 중앙 0.000e+00 최대 0.000e+00 · >5.88e-2 인 셀 0/12
  k̂_p/Δk₁ 중앙 3.16 최소 2.80 최대 4.68
```
(⟹ **D. 옛 규칙 같은 12셀 대조**는 §4.3 에 적었다.)

### 4.1 ★읽기
- **새 규칙의 등변 편차 = `0.000e+00`, 시험한 58측정 전부**(A 16 · B 18 · C 24)**.** `c = 0.5`·`c = 2.0` 두 방향,
  **수심 고정**(`T14 G-EQUIVAR` 의 실제 프로토콜)과 **수심 재라벨**(수학적으로 정확한 변환) **둘 다**.
  ⟹ §3.1 의 (a)(b) 두 위험이 이 대역에서 **실현되지 않는다**.
  ⚠C 절의 인쇄된 제목은 *"씨앗 30개"* 이지만 **실제 반복은 12** 다(비용 절감으로 줄이면서 제목 문자열을 안 고쳤다 — 기록).
- **옛 규칙은 같은 셀에서 무너진다** — `c=0.5` 방향 최대 **1.138e+03**.
  ⚠그리고 `k_p = 0.010` 에서는 **재라벨 없이도** `k̂ = 7.54` 를 낸다(참값의 **754배**).
  ⟹ `T14 AMENDMENT_2` 의 기제 서술이 합성에서 **재현**된다.
- **여유가 3배 개선된다**: `k̂_p/Δk₁` 이 옛 **1.03–1.58** → 새 **3.38–5.37**(A), 12셀 중앙 **3.16**(C).
  `T14` 본실행의 실측 중앙 **1.16**·최소 **0.88** 과 같은 자다.
- ★**새 규칙의 붕괴 지점은 원리적이다**: `k_p → 0.010` 에서 `k̂_p/Δk₁ → 1.04`.
  그런데 `Δk₁ = 2π/L_obs` 는 **개구가 분해할 수 있는 최소 파수** 그 자체다
  ⟹ ***규칙은 관측 개구가 그 파장을 분해하지 못하는 곳에서 정확히 함께 한계에 닿는다.***
  ⛔이것을 *"규칙이 더 튼튼하다"* 로 확장하지 마라 — 잰 것은 **등변성과 격자 여유**이지 **정확도**가 아니다.

### 4.2 ⛔이 프로브가 **증명하지 않는 것**
1. ⛔**정확도를 증명하지 않는다.** `k̂_p` 가 참 `k_p` 에 얼마나 가까운지는 여기서 판정하지 않는다
   (A 에서 새 규칙은 `k_p` 를 **+2~11% 과대추정**한다 — `T14 §9.3-9` 의 편향과 같은 부호다. ⛔보정하지 않는다).
2. ⛔**제3자 자료에서의 등변성을 증명하지 않는다.** 이 장면은 **합성**이다.
   ⟹ 본실행의 `G-EQUIVAR` 가 **판정 자료 위에서** 다시 잰다(전 행). 그것이 차단 게이트다.
3. ⛔**추정 성능을 증명하지 않는다.** 추정기를 한 번도 돌리지 않았다.
4. ⚠**`L_obs` 는 이 코퍼스에서 상수다** ⟹ `klim₁ = 0.60334` 도 상수다.
   그것은 `d3_beamform_net.py:31` 의 `KLIM = 0.6` 과 **0.56% 차**로 **자료만 보면 구별되지 않는다**.
   ★구별은 **`G-KLIM1-LIVE`(재라벨에서 `klim₁` 이 `c` 배로 움직이는지 되읽는 행동 검사)** 가 한다.
   ⛔우연한 근접을 유도로 서술하지 마라.

### 4.3 D. 옛 규칙 같은 12셀 대조 (§4 프로브의 마지막 절)

```
=== D. 옛 규칙 동일 30셀 (대조) ===          ⚠제목 문자열은 낡았다 — 실제 반복은 **12**(C 와 같은 셀)
  |c=0.5| 중앙 3.490e-02 최대 3.905e+02 · >5.88e-2 인 셀 5/12
  |c=2.0| 중앙 1.077e-02 최대 3.471e-02 · >5.88e-2 인 셀 0/12
  k̂_p/Δk₁ 중앙 0.97 최소 0.82 최대 1.39
```
⟹ ★**같은 12셀에서 새 규칙 0/12 위반 vs 옛 규칙 5/12 위반**(`c=0.5` 방향, `T14` 의 코드 유래 허용 5.88% 기준).
⟹ ★**여유 3.3배**: `k̂_p/Δk₁` 옛 **중앙 0.97 · 최소 0.82** → 새 **중앙 3.16 · 최소 2.80**.
⚠**합성 장면이 `T14` 의 실측 국면을 재현한다** — `T14` 본실행 28블록의 실측은 중앙 **1.16** · 최소 **0.88**
이었고, 이 프로브의 옛 규칙은 중앙 **0.97** · 최소 **0.82** 다(자료가 아니라 **같은 자**로 잰 값).
⛔이 근접을 *"프로브가 코퍼스를 재현한다"* 로 확대하지 마라 — 격자·창·밀도만 맞춘 합성이다.
⚠`c=2.0` 방향에서 옛 규칙이 12/12 통과한 것도 `T14`(초과 **1/34**)와 정합이다
⟹ ***`T14` 를 차단한 것은 그 방향의 한 셀이었지만, 규칙이 실제로 무너지는 방향은 처음부터 `c=0.5` 였다.***

---

## 5. 표본 — **형제 배치와의 겹침을 결과 0행 시점에 계산했다**

(교훈: `T13` 이 `T12` 와 24행 **비트 동일**인데 *"독립 확증"* 이라 적었다가 철회했다.)

```
아카이브 sorted() 126 mats:
   idx   0– 59  Data/mixed_data          60
   idx  60–119  Data/mixed_data2         60
   idx 120–125  Data/outofdistribution    6

추정기 팔이 채점된 적이 있는 블록  (T11·T12·T13·T14 의 동결 jsonl 을 **전건 개봉해** 계산)
   T11   24  (mixed_data2)                     T12   52  (mixed_data2)
   T13   52  (mixed_data2)                     T14   36  (mixed_data2 30 + OOD 6)
   합집합 : mixed_data2 **54/60** · outofdistribution **6/6** · mixed_data **0/60**

⟹ ★Tier-P (= Data/mixed_data 전수 60) ∩ (T11 ∪ T12 ∪ T13 ∪ T14) = **∅**
```

### 5.1 층 안의 노출 등급 (⛔같은 무게로 읽지 마라)
| 등급 | n | 근거 |
|---|---|---|
| **성과(outcome) 노출** | **0/60** | 어떤 배치도 `mixed_data` 에서 `nRMSE` 를 낸 적이 없다 |
| 속성(property) 노출 | 36/60 | `T1`(12)·`T1b`(24)·`T10`(12)이 **비선형성·분산관계 통계**만 쟀다(`skew`·`ct`·`fit_depth`). ⛔이 배치의 설계는 그 수를 쓰지 않는다 |
| **완전 미접촉** | **24/60** | `angle40 h{3.5,4.0,4.5,5.0}` · `angle50 h{0.5…5.0}` · `angle60 h{0.5…5.0}` ⟹ `SEC-BLIND` |

### 5.2 남은 blind 층 계산 — ★**이것이 마지막이다**
```
mixed_data2 : 60 중 54 가 성과 노출.  남은 6 은 `AUDIT_SMOKE_IDX` 의 **감사 노출** 파일이다
              (angle10h3.0 · angle10h4.5 · angle20h1.0 · angle20h2.5 · angle20h4.0 · angle30h0.5)
              ⟹ ⛔blind 아님.
outofdistribution : 6/6 성과 노출 (T14 Tier-3).
mixed_data  : **60/60 성과 미노출** ⟹ ★이 아카이브에서 이 규약으로 남은 blind 층은 여기뿐이다.
```
⟹ **재실행은 blind 가 아니다.** 그래서 이 배치는 **전수 60** 을 쓰고, 표본 크기를 정한 것은
검정력이 아니라 **blind** 다(⚠`PREREG §9.3-14` 공시).

### 5.3 `mixed_data` 가 `mixed_data2` 와 같은 기하인가 — ✅확인 / ⛔미확인
```
✅ T1_THIRDPARTY 동결 행  Data/mixed_data/angle10h0.5.mat : shape [502,128,128] · dx 4.878999999999905 · dt 1.0
✅ T10_EXTDISP  동결 행  Data/mixed_data/angle30h2.5.mat : dx 4.878999999999905 · dy 동일 · dt 1.0
✅ T14          동결 행  Data/mixed_data2/*             : e_shape [502,128,128] · dt 1.0 · L_obs 619.645
⛔ 생성 시드 · Tp · 방향 분포는 **열지 않았다** (zip 은 namelist 만).
```
⟹ 차단 게이트 `G-GRID`(런 시점 되읽기)와 기록 게이트 `G-TP`(`Tp̂` 유도)가 그 자리를 메운다.
⚠`T14 §9.3-6` 의 *"`Tp` 전 파일 10 s"* 는 **`mixed_data2` 의 진술**이다 — 이 배치에서는 미확인이다.

### 5.4 시드 구간 충돌 검사 (실측)
```
T11 20260905–20260928 · T12 20260905–20260932 · T13 20260905–**20261032** · T14 20260905–20260948
T16 SEED16 = 20261300  ⟹ 20261301 … 20261360        교집합 **∅**
   ⛔초안 20261100 은 T15(20261101–228)와 **완전 충돌**이었다 — 결과 파일만 보는 절차의 사각지대
ANCHOR 2파일만 T11 시드 20260905 / 20260906 으로 덮어쓴다(그래야 `G-ANCHOR-T11` 이 성립한다)
```

### 5.5 `T15-HOMEREG`(같은 날 형제)와의 관계
`T15` 는 **우리 코퍼스**(`data/*.h5` 110 에피소드 + 고정생성기 128)를 쓴다(`T15 PLUMBING.md §1`)
⟹ **표본 교집합 0**, 처치도 다르다(`A-MAPSCALE` 를 자국 코퍼스에). ⛔순서 의존 없음.
`G-SIBLING`(기록 전용)이 런 시각의 형제 결과 파일 존재·mtime 을 남긴다.

---

## 6. `check_prereg.py` — 설계 단계에서 이미 돌렸다

```
$ systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- \
      python3 tmp/_audit/check_prereg.py tmp/T16_BANDFIX_2026-09-05
  FAIL 4 · WARN 1 · OK 6 · N/A 9
```
(★`T14` 의 설계 단계 프로필 `FAIL 4 · WARN 1 · OK 6 · N/A 9` 와 같다.)
**OK**: `C4`(양성 이름 라벨이 전부 이김 다리) · `C11`(스모크) · `C13`(다중성 공시) ·
`C14`(blind/노출) · **`C15`(PRIMARY 통계량·방향·n=60)** · `C16`(시드).

**남은 FAIL 4 + WARN 2 는 전부 「구현이 아직 없다」이고, 구현자가 닫아야 한다**:
| # | 항목 | 닫는 법 |
|---|---|---|
| `C0` | `gates_t16.py`·`analyze_t16.py` 0개 | 러너보다 **먼저** 쓴다(2026-09-04 규율) |
| `C8` | 사전등록 게이트 19개 미구현 | `PREREG §6.1–6.4` 전부 분석기에 |
| `C7` | `FROZEN*.txt` 없음 | **결과 0행 시점에** 생성(`PREREG §9.1` 측정면/분석면 분리) |
| `C9` | ⛔공시 토큰이 분석기에 없다 | `PREREG §9.2·§9.3` 을 `disclosures` 에 **기계로** 넣는다(⛔재작성이 조용히 떨어뜨리는 부류 — `T11 A3 §1` 이 하루에 세 번 당했다) |
| `C1`(WARN) | MDE 상수가 gates 에 없다 | `PREREG §12-2` 의 상수 목록을 `gates_t16.py` 로 |

★설계 중 검사기가 잡아 **이미 고친 것**: `C15` 가 처음엔 WARN(*"PRIMARY 줄은 있으나 방향 기호 없음"*)이었다
⟹ `PREREG §3` 의 가족 선언 줄에 **방향 `<` 와 `n = 60`** 을 명시해 OK 로 닫았다.
⚠`C10` 은 **N/A** 다 — 이 디렉터리가 아직 git 미추적이라 소실 게이트 검사가 불가능하다.
구현자는 `run_t16.py`·`scale_rules_t16.py`·`sample_t16.json` 과 함께 **문서도 커밋**한 뒤 다시 돌려라.

⛔`N/A` 9개는 **미검증**이지 통과가 아니다 — 러너·분석기가 생기면 **다시 돌려라**.
결과 0행 시점에 `check_gate_refs.py`·`check_gate_blindspots.py` 도 함께 돌린다.

---

## 7. 미래형 (⛔이 배치에서 하지 않는다)

`T16` 이 양성이면 다음 질문은 *"패키지의 `fit_bank_mmse` 1단도 상대로 바꿔야 하는가"* 다.
그것은 **실측 경로**(`sequential_update.py:803` 뒤)를 바꾸는 일이므로 **별도 사전등록**이 필요하고,
최소한 (i) 실측 12 레코드에서의 `no_harm`, (ii) 우리 코퍼스에서의 `no_harm`, (iii) 단가가 남는다.
⚠그리고 그 경로에서는 `bank_mmse` 가 **이미 자기 `k̂_p` 에 상대적인 뱅크**를 만들므로
(`:168` `dk_c = kp / 10.0` · `:183` `ax = np.arange(-b_hi*kp, …, dk_c)` · `:122` `band=(0.35, 2.5)`) 효과는 **1단 줌 한 단계에 국한**될 것이다.
⛔이 예상을 결과로 인용하지 마라 — **측정하지 않았다**.
