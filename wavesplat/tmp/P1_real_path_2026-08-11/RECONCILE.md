# RECONCILE — 실측 경로 두 구현의 불일치 규명·수정·증명 (2026-08-11)

작성 시각의 모든 수치는 **이 문서를 쓰면서 직접 실행해 얻은 값**이다. 문서·JSON 에서
베낀 수치는 없다. 재현 스크립트는 전부 이 디렉터리에 있다.

---

## 0. 한 줄 결론

> **원인은 `smult` 선택 로직이 아니라 관측 배열의 ORDER 다.**
> `fit_bank_mmse` 는 검증분할을 **배열 위치**로 뽑는다(`bank_mmse.py:140`
> `rng.choice(n, ...)`). GIF 하네스가 프로토콜 600-캡을 **두 번** 적용하면서 순서가
> 바뀌었고, 같은 600점이 서로 **다른 물리점**을 val 로 골라(겹침 38/150 = 우연 수준)
> 다른 `(pexp, smult)`·다른 prior·다른 예보를 냈다.
> **패키지는 옳았고 하네스가 틀렸다.** 하네스 한 곳을 고치니 두 경로가 **비트 동일**해졌다
> (`np.array_equal` = True, max|Δ| = **0.000e+00**, 3창 × 4 lead = 12/12).

---

## 1. 재현 (수정 전)

`probe_deployed_gap.py` — AA15a w5, 같은 창·같은 격자·같은 GT.

| lead | 기본설정 경로 | 스크립트 경로 | 차 |
|---|---|---|---|
| 0.0 | 0.4885 | 0.4982 | −0.0098 |
| 1.0 | 0.4820 | 0.4924 | −0.0105 |
| 7.0 | 0.7843 | 0.6027 | **+0.1815** |
| 14.0 | 0.9043 | 0.7049 | **+0.1994** |

시간 ±4 s 스캔에서 최적 δ = **+0.00 s** (τ=7, τ=14 모두) ⟹ 시계 어긋남 아님.

---

## 2. 원인 규명 — 인자 분리 측정

배포 래퍼가 연구 호출 대비 관측 배열에 가하는 변형은 정확히 셋뿐이다
(`make_gif_bank_deployed.frames_from_scene` + `sequential_update._bank_obs_from_frames`).
셋을 하나씩 켜서 측정했다 (`probe_factor_isolate.py`, AA15a w5, seeds {42,1042,2042}).

| 팔 | kp | pexp | smult | lead14 nRMSE |
|---|---|---|---|---|
| **연구 기준** (원 순서, f64, 절대시각) | 0.0986 ×3 | [2,2,1] | [1,3,3] | **0.7049** |
| +ORDER 만 | 0.0986 ×3 | [2,2,**2**] | [1,3,**1**] | **0.9043** |
| +FLOAT32 만 | 0.0986 ×3 | [2,2,1] | [1,3,3] | 0.7049 |
| +TSHIFT 만 | 0.0986 ×3 | [2,2,1] | [1,3,3] | 0.7049 |
| ORDER+F32+TSHIFT (배포 재구성) | 0.0986 ×3 | [2,2,2] | [1,3,1] | 0.9043 |
| **실제 배포 경로** | 0.0986 ×3 | [2,2,2] | [1,3,1] | 0.9043 |

- 재구성 vs 실제 배포 예측면 **max|Δ| = 0.000e+00** ⟹ 세 인자가 차이의 전부다.
- **ORDER 단독으로 전체 차이를 재현**한다. FLOAT32·TSHIFT 는 4자리에서 무영향.
- `kp` 는 모든 팔에서 동일 ⟹ beamform zoom 은 용의자가 아니다.
- seed 사용 순서도 동일 (`_bank_seeds` = [42,1042,2042] = `RR.SEEDS`).

### 2.1 왜 순서가 바뀌는가 (정확한 지점)

```
scene_real (d3_compare_gifs.py:196-201)
    P_all = 시간순 5760점  →  ii = default_rng(0).choice(5760, 600)
    sc['P'] = P_all[ii]                       ← **셔플된** 600점
                    ↓
make_gif_bank_deployed.frames_from_scene:53   for tt in np.unique(P[:,2])
    → 그 600점을 **시간순으로 되정렬**
                    ↓
sequential_update._bank_fit_window:844        if len(z) > cap:   # cap=600
    → len == 600 이므로 **캡이 발동하지 않음** (no-op)
                    ↓
bank_mmse.fit_bank_mmse:140                   vi = rng.choice(n, 150)
    → **위치 기반** val 분할이 다른 물리점을 고른다
```

측정: seed 42 의 val 150점, 연구 순서 vs 배포 순서의 **교집합 38/150 = 25.3%**
(= 우연 기대값 150/600 = 25%) ⟹ 두 분할은 사실상 독립이다.
순열이 항등인가: **False**.

### 2.2 어느 쪽이 설계 의도인가 — 코드·주석 근거

**패키지가 옳다.** 근거:

1. `sequential_update._bank_fit_window` 독스트링:
   > "The window is capped at bank_max_obs points with the **SAME deterministic
   > subsample the frozen research harness used (scene_real m_cap=600,
   > default_rng(0))**: every published real-sea bank number was measured on
   > 600-obs windows"

   ⟹ 패키지는 **자기가 `scene_real` 의 캡이 되겠다**고 선언한다. 이 계약은 **미캡 창을
   받아야만** 지킬 수 있다. 이미 캡된 600점을 주면 계약이 무효화된다.

2. `bank_mmse.py` 모듈 독스트링: "The math below is a **verbatim transcription** of the
   `mmse_bank_estimate` default path"; "`test_bank_mmse_port.py` **pins bit-exactness**
   against the research function."
   ⟹ 추정기 내부(val 분할 규칙 포함)를 바꾸면 **모든 발표된 실측 수치가 움직인다**. ⛔건드리면 안 된다.

3. `PROTOCOL_ROLLING_REFIT.md`: "all = deployed stack: … **600-obs cap via rng(0)**"
   ⟹ 연구 하네스 스스로 "출하 스택을 돌린다"고 사전등록했다. 두 팔은 같아야 한다.

4. `frames_from_scene` 독스트링: "Format adapter only … **No value altered, nothing
   filtered**" — **거짓 주장**이었다. 조용히 재정렬한다. 이 문장이 결함의 이름이다.

**검증 측정** (`probe_order_origin.py`): 미캡 40 s 스트림(5760점)을 그대로 배포
`SequentialUpdater` 에 먹이면

```
rng(0) 캡이 scene_real 의 600점을 재현: True
[FrameData float64] pexp=[2,2,1] smult=[1,3,3] vr=[0.199638,0.200258,0.203152]
                    vs 연구(f64) 최대|Δ| = 0.000e+00      ← 비트 동일
[FrameData float32] vs 연구(f32 관측) 최대|Δ| = 0.000e+00
                    vs 연구(f64)      최대|Δ| = 3.940e-08 ← 관측 양자화만
```

---

## 3. 수정 — 최소 1파일

**변경 파일: `tmp/bankmmse_port_2026-07-29/make_gif_bank_deployed.py` 단 하나.**
패키지(`wavesplat/`, `wavesim/`), `configs/`, `paper/` 는 **한 줄도 건드리지 않았다.**

| 변경 | 내용 | 왜 |
|---|---|---|
| `scene_uncapped(record, win)` 신규 | `scene_real(..., m_cap=10**9)` | 배포 추정기에게 **미캡 창**을 줘서 자기 rng(0) 캡이 발동하게 한다 |
| `estimate(sc, seed, sc_obs=None, dtype=torch.float32)` | 프레임을 `sc_obs` 에서 만든다. `sc` 는 격자/GT/Pq 만 공급 | 연구 팔(`estimate_bank`)은 계속 `sc['P']` 를 쓰므로 **무영향** |
| `main()` 배포 팔 호출 | `estimate(sc, a.seed, sc_obs=scene_uncapped(a.record, a.win))` | 이중 캡 제거 |
| `frames_from_scene(..., dtype=)` | 기본값 `torch.float32` 유지 | float64 는 **잔차 4e-08 이 관측 양자화뿐임을 증명**하는 용도로만 |
| 독스트링 2건 | "No value altered" 가 거짓임을 정정 + 근거·측정치 기입 | 같은 결함 재발 방지 |

⚠**패키지는 의도적으로 고치지 않았다** — 이유는 §5.

---

## 4. 일치 증명 (`prove_reconcile.py` → `reconcile_proof.json`)

3창 × 4 lead. 배포 = 수정된 하네스(미캡 창) → `SequentialUpdater` → 실측게이트 →
`_initialize_bank`; 연구 = `fit_bank_mmse(sc['P'], sc['z'], …)` seeds {42,1042,2042}.

### L1b — 시간원점을 맞춘 엄밀 비교

| 창 | lead | max abs Δ | `np.array_equal` |
|---|---|---|---|
| AA15a w3 | 0 / 1 / 7 / 14 | **0.000e+00** ×4 | **True** ×4 |
| AA15a w5 | 0 / 1 / 7 / 14 | **0.000e+00** ×4 | **True** ×4 |
| YS17 w3 | 0 / 1 / 7 / 20 | **0.000e+00** ×4 | **True** ×4 |

**12/12 비트 동일.** 선택된 하이퍼도 창별로 전부 일치:

| 창 | kp | pexp | smult | 미캡 n |
|---|---|---|---|---|
| AA15a w3 | 0.1175 | [2,2,2] | [1,1,1] | 5760 |
| AA15a w5 | 0.0986 | [2,2,1] | [1,3,3] | 5760 |
| YS17 w3 | 0.0832 | [2,1,2] | [3,3,3] | 4800 |

### L1 — 하네스의 상수 시간이동을 남겨둔 경우

max|Δ| = 4.87e-14 … **1.27e-12**. 이 잔차는 **전부 t_shift 다**:
TSHIFT 단독 효과(L0)와 lead 7 에서 **자릿수까지 정확히 일치**
(AA15a w3 6.660e-13 = 6.660e-13, w5 9.057e-13 = 9.057e-13, YS17 5.168e-14 = 5.168e-14).
시간이동은 수학적으로 **정확한 재매개화**다(prior `gam = concat[g,g]` 가 각 (cos,sin)
쌍에서 등방이라 회전과 교환된다). 남는 것은 float64 반올림뿐.
⟹ **1e-12 문턱을 넘고 싶으면 시간원점을 맞추면 되고, 맞추면 정확히 0 이다.**

### L2 — 배포 실물(FrameData float32)

max|Δ| = 2.67e-08 … 5.84e-08. **알고리즘 차이가 아니라 관측 양자화**다
(`wavesplat/real_data.py:305` 가 실측 프레임을 float32 로 만든다 = 출하 규약).
같은 float32 관측을 연구 호출에 주면 **0.000e+00** (§2.2).

### L3 — `predict_field` 의 멤버 중앙 취하기

`upd.predict_field(...) == np.median(members, 0).astype(float32)` → **True** (3창 전부).
float64 중앙값 대비 3.0e-08 ~ 4.8e-08 = `_bank_predict_grid` 의 마지막
`torch.as_tensor(med, dtype=torch.float32)` **한 줄뿐**이며, 이는
`SpectralGaussianModel.predict_grid` 와 맞춘 **의도된 API 규약**이다(버그 아님).

### 부수 측정 — 이 불일치의 크기

수정으로 성능이 좋아진 창도, 나빠진 창도 있다(**목표가 아니라 결과**):

| 창 | lead | 연구 = 수정후 배포 | 수정 전 배포 |
|---|---|---|---|
| AA15a w3 | 14 | 1.7168 | **1.1533** (수정 전이 더 좋았다) |
| AA15a w5 | 14 | **0.7072** | 0.9072 |
| YS17 w3 | 20 | 1.0117 | 1.0074 |

⟹ **어느 순서가 "맞는" 게 아니라, val 분할 하나가 lead14 nRMSE 를 0.56 (49%) 움직인다.**
median-of-3 앙상블이 이 분산을 못 잡는다. ⛔따라서 "수정해서 좋아졌다/나빠졌다" 로
읽으면 안 된다.

---

## 5. 고치지 **않은** 것과 그 이유

### 5.1 `fit_bank_mmse` 의 위치기반 val 분할 — ⛔건드리면 안 됨
분할 규칙을 순서 불변으로 바꾸면 **연구 호출의 출력도 함께 바뀐다**. 이 함수는
`mmse_bank_estimate` 의 verbatim 이식이고 `test_port_bit_exact_vs_research` 가 비트
동일성을 고정하고 있다. 발표된 모든 실측 수치가 이동한다. **일치가 목적이지 개선이 아니다.**

### 5.2 `_bank_fit_window` 의 `if len(z) > cap` — 잔존 위험, **보고만**
출하 경로의 출력은 관측 **집합**이 아니라 **배열 순서**의 함수다. 측정
(`probe_order_risk.py`, 합성 바다, 같은 집합을 순열만 바꿔 투입):

| 상황 | perm1 max abs Δ | perm2 max abs Δ |
|---|---|---|
| cap 미발동 (n=560 ≤ 600) | 7.83e-03 m | 4.62e-03 m |
| cap 발동 (n=800 > 600) | 2.51e-02 m | 1.93e-02 m |

★**캡이 발동해도 순서 불변이 아니다** — 캡도 *위치*를 고르므로 입력이 섞이면 다른
600점을 고른다. 캡은 "같은 입력 배열에 대해 결정적"으로 만들 뿐이다.
실측 재현이 성립한 것은 `scene_real` 과 패키지가 **원 스트림을 같은 시간순으로
조립**하기 때문이다.

**왜 안 고쳤나**: 무조건적 결정적 재배열(`choice(n, min(cap,n))`)은 n ≤ 600 구간의
**출하 동작을 바꾼다**. 그 구간은 측정된 적이 없고, 이 프로젝트 규율상 출하 동작 변경은
사전등록된 측정 위의 **사용자 결정**이다(`PROTOCOL_ROLLING_REFIT.md`: "Any change of the
shipped default is a USER decision on these numbers; this run changes no code").
게다가 그 변경은 §4 의 일치에 **필요하지 않다**. ⟹ 권고만 남긴다.

**권고 (사용자 결정 사항)**: 배포 계약을 "관측 집합의 함수"로 만들고 싶다면
`_bank_fit_window` 에서 캡 여부와 무관하게 정준 정렬(예: `(t, x, y)` lexsort) 후
rng(0) 추출을 하도록 바꾸되, **먼저 사전등록된 측정**을 걸어야 한다.

---

## 6. 회귀 증명

### 6.1 변경 파일
```
git status --short  (내 변경분)
 M wavesplat/tmp/bankmmse_port_2026-07-29/make_gif_bank_deployed.py   ← 유일
```
`wavesplat/wavesplat/`, `wavesplat/wavesim/`, `wavesplat/configs/`, `wavesplat/paper/`
아래 변경 **0건**.
(`wavesplat/wavesplat/evaluation/metrics.py` 와 `results/gifs/*.gif` 는 이 세션 **시작
시점에 이미** 수정되어 있던 것으로 내 작업과 무관하다.)

### 6.2 테스트
```
python3 -m pytest wavesim/tests/ wavesplat/tests/ -q
수정 전:  196 passed in 56.15s
수정 후:  196 passed in 55.42s
```

### 6.3 합성 경로 비트 동일 (`regress_synth.py`)
하네스를 HEAD 로 되돌려 `before`, 수정본으로 `after` 를 각각 실행.
`run_estimation(seed=42, device='cpu')`, 프레임별 nRMSE 를 `repr()` 전체 정밀도로 기록.

| 에피소드 | 프레임 | avg_nRMSE (before = after) |
|---|---|---|
| `episode_BF4_5comp_seed0_uniform.h5` | 160 | `0.06685207008073728` |
| `episode_BF2_1comp_seed0_uniform.h5` | 220 | `0.3660015564298991` |

**`regress_synth_before.json` 과 `regress_synth_after.json` 이 바이트 동일 = True**
(380개 프레임별 nRMSE 전체 정밀도 포함).
추가로 합성 실행 중 `'make_gif_bank_deployed' in sys.modules` = **False** ⟹ 수정한
모듈은 합성 경로에서 **import 조차 되지 않는다**.

---

## 7. 재현 스크립트

| 파일 | 무엇을 |
|---|---|
| `probe_deployed_gap.py` | 수정 전 격차 재현 + 시간 δ 스캔 (기존) |
| `probe_harness_gap.py` | 관측 100% 교집합 확인 (기존) |
| `probe_factor_isolate.py` | ★ORDER/FLOAT32/TSHIFT 인자 분리 |
| `probe_order_origin.py` | ★미캡 스트림 → 비트 동일 확인 |
| `probe_order_risk.py` | ★출하 경로 순서 의존성 크기 |
| `prove_reconcile.py` | ★3창 일치 증명 → `reconcile_proof.json` |
| `regress_synth.py` | 합성 회귀 → `regress_synth_{before,after}.json` |
