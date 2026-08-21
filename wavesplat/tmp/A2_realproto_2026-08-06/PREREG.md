# PRE-REGISTRATION — A2: 실측 비교 프로토콜 위생 (GT-유래 입력 대칭화)

Frozen **2026-08-06, 결과를 보기 전에**. Branch `learn-boundwave-retrain`.
No commits, no paper edits, no config/default changes, no training on real data, checkpoints
read-only. 출력은 `tmp/A2_realproto_2026-08-06/` 에만. 선행 자산
(`tmp/wp_real_2026-07-25/`, `tmp/bankmmse_port_2026-07-29/`)은 읽기 전용.

---

## 0. 전제 (1단계 조사에서 코드로 확인 — 동결 전 완료)

`tmp/wp_real_2026-07-25/run_wp_real.py:50-52` (hand arm):
```
s0 = 0.05 * sc['Hs']
fq = mmse_bank_estimate(..., sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'])
```
`scripts/d3_compare_gifs.py:scene_real` 에서
- `Hs = 4.0 * np.nanstd(Zmem)`  — `Zmem` = **조밀 진실면**의 과거 40 s 메모리 창 (전 그리드).
- `kp` = `directional_spread(np.nan_to_num(Zmem[-1]), dx, dy)` 의 4번째 반환값 — **조밀 진실면
  마지막 프레임**에서 유도.
⟹ hand 팔은 희소 LiDAR 관측 `z` 가 아니라 **조밀 진실면**에서 유도된 두 스칼라를 받는다.

출하 경로 `wavesplat/optimizer/sequential_update.py:817-825`:
```
self._bank_Hs_obs = 4.0 * float(np.std(z));  sigma0 = 0.05 * self._bank_Hs_obs
fit_bank_mmse(P, z, self.depth, sigma0, kp_hint=None, seed=s)   # seeds = base+{0,1000,2000}
```
⟹ 출하는 `kp_hint=None` + `Hs_obs = 4·std(z)` + median-of-3.

Learned 팔 (`run_wp_real.py:143-171`)은 **GT 를 안 쓴다**: `_bmap_from_obs(Ps,z)` → `sample` →
`polish_atoms(...,Pt,zt,...)` → `atoms_to_field` 는 `target['gt']/['Hs']` 를 **버려지는 nrmse
계산에만** 쓰고 반환 필드 `pred = render_wave_field(kx,ky,A,phi,depth,gx,gy,t)` 는 GT 무관.
`fit_amplitudes_phases` 는 관측만 사용. ⟹ **비대칭은 hand 팔 한쪽에만 있다.**

부수 확인(개입 대상 아님, 보고만): learned 팔은 관측을 `M_OBS=400` 으로 재추출(`rng(0)`),
hand 팔은 600 전부를 받는다 ⟹ **관측 예산도 비대칭**(hand 유리). 아래 셀 BB400 으로 측정한다.
`scene_real` 의 배 항로 방향 `pdir` 도 조밀 `Zmem` 에서 유도되지만 이는 **장면 생성**이며
두 팔에 동일하게 적용되므로 팔 간 비대칭이 아니다(기록만 한다).

### 헤드라인 재도출 (동결 전 완료, 원자료에서)
`wp_real.json` 정상 27창: hand **mean 0.7441 / median 0.7407**, learned **mean 0.8051 /
median 0.8361**, 짝별 median(learned−hand) **+0.0571**, learned 승 **6/27**,
양측 Wilcoxon **p=0.004613**. ⟹ 과제문의 0.7441/0.8050 은 **평균**, 사전등록 1차 지표인
median 은 0.7407/0.8361. **재현 확인.**
`blind_vs_frozen.json` 3창: GT 입력 제거 비용(nowcast) **+0.0554/+0.0233/−0.0218, 중앙 +0.0233**;
짝별 격차(learned−hand) 중앙 **+0.0572 → +0.0339**, learned 승 **1/3 → 1/3**. **재현 확인.**

---

## 1. 팔 / 셀 정의 (동결)

Learned 팔은 4셀 전부에서 **동일**하다(위 0절: GT 를 안 받으므로 개입 대상이 없다).
따라서 4셀은 **hand 팔의 2×2 요인 설계**이고, 각 셀은 동일한 learned 값과 짝지어진다.

| 셀 | `kp_hint` | `sigma0` | 관측 | 이름표 |
|---|---|---|---|---|
| **FF** | `sc['kp']` (조밀 GT) | `0.05·Hs_GT` | 600 | 발표된 연구팔 = **묶음**(GT 손잡이 2개 동시 ON) |
| **BF** | `None` (내장 2단 beamform) | `0.05·Hs_GT` | 600 | `kp_hint` 단독 OFF |
| **FB** | `sc['kp']` | `0.05·Hs_obs` | 600 | `sigma0` 단독 OFF |
| **BB** | `None` | `0.05·Hs_obs` | 600 | **출하 구성** = 묶음 2개 동시 OFF |
| **BB400** | `None` | `0.05·Hs_obs` | 400 (learned 와 동일 `rng(0)` 부분추출) | 2차: 관측예산까지 대칭화 |

`Hs_obs = 4.0·np.std(sc['z'])` — 출하 `_bank_fit_window` 와 **문자 그대로 동일**.
`Hs_obs` 는 창의 LiDAR 관측 `z` 만 쓴다(GT 무접촉, 코드로 확인).

**init_method 명시**: 이 실험은 `wavesplat.run` 경로를 쓰지 않는다. hand 팔은
`scripts/d3_sa_nomp.mmse_bank_estimate` 직접 호출이며, 대응하는 출하 설정은
`configs/estimation.yaml: real_estimator: bank_mmse` (→ `sequential_update._bank_fit_window`).
learned 팔은 `results/direction3/beamform_cond_diffusion.pt` 고정 체크포인트 추론.

**시드 규약**: hand 는 5셀 **전부** 출하 규약 median-of-3, `seeds = base+{0,1000,2000}`,
`base = 42` ⟹ **{42, 1042, 2042}** (= `blind_vs_frozen` 이 쓴 시드와 동일 ⟹ 그 3창은
자동 재현 대조가 된다). 셀 간 시드 동일 ⟹ 짝별 비교에서 추정기 시드분산이 상쇄된다.
추가로 **FF-exact** = `seed=sc['seed']`(=i0), 단일 solve — `wp_real.json:hand_now` 와의
**비트 재현 대조**용(1차 판정에는 쓰지 않는다).
learned 값은 `wp_real.json:learned_now`(diffusion 시드 0–4 의 median)를 그대로 쓰고,
**6창 재현 스팟체크**(레코드당 1창 + 붕괴창 1개)를 GPU 로 재실행해 대조한다.

---

## 2. 데이터 (동결)
`RECORDS_ORDER = ['YS17','AA15a','AA14','AA15b','LJ18g']` × `win 0..5` = **30창**.
`scene_real(rec, win)` 기본 인자(W=40, m_scan=12, m_cap=600, noise=0.02) 그대로.
실측 데이터는 **TEST 전용**. 학습에 쓰지 않는다.

---

## 3. 지표 / 집계 규약 (동결 — `wp_real` PREREG 와 동일)

- 지표 = **nowcast(lead-0) map nRMSE**, `sg = std(모든 lead 에 pooled 된 유한 GT)`
  = `run_wp_real.score_maps` 와 문자 그대로 동일. (`blind_vs_frozen` 의 `dense_gt_std`
  변형은 쓰지 않는다 — 두 정의는 창당 ~0.002 차이가 나며, 본 실험은 `wp_real` 헤드라인과
  직접 비교되어야 하므로 `wp_real` 정의를 채택한다.)
- 짝 단위 = 창. **Δ_cell(w) = nRMSE_hand(cell,w) − nRMSE_learned(w)**, **양수 = learned 우세**.
- 1차 요약 = **정상창 median Δ** + 양측 **Wilcoxon signed-rank, α=0.05** + 승수(learned 승 창수)
  + rank-biserial. (`wp_real` PREREG §2 와 동일. **바꾸지 않았다.**)
- 평균 기반 요약은 **참고로만** 병기하고 판정에 쓰지 않는다(⛔기존 인용금지: 실측 mean 기반
  learned 승리는 붕괴창 3개가 지배).

### 판정 집합
- **S1 (1차)** = `wp_real` PREREG 가 동결한 **정상 27창** (사전 제외:
  `AA14_w0`, `AA15a_w3`, `LJ18g_w3`). 셀과 무관하게 고정 ⟹ 5셀이 같은 창에서 비교된다.
- **S2 (2차, 널 기반)** = **모든 셀의 모든 팔이 평평바다 널을 이긴 창**의 교집합.
  규칙은 지금 고정하되 창 목록은 실행 후 산출된다; 셀 간에는 **동일 집합**이므로 비교 가능.
- S1 밖 3창(붕괴창)은 **별도 표**로 보고하며 1차 판정에 넣지 않는다.

### 평평바다 널 (병기 의무)
창마다 아래 두 널을 계산해 **모든 표에 병기**한다.
- `null_zero(w) = sqrt(mean(gt0[m]^2)) / sg`   — z≡0 예측
- `null_mean(w) = sqrt(mean((gt0[m]−mu)^2)) / sg`, `mu = mean(pooled 유한 GT)` — 상수면 예측
"널을 넘는다" = `nRMSE < null_zero(w)`. 널을 못 넘는 셀·창은 **결과가 아니다**:
S2 에서 제외되고, S1 표에서는 널 미달 여부를 플래그로 표시한다.

---

## 4. 합격선 / 실패조건 (결과 보기 전 동결)

**1차 판정은 셀 BB(출하 구성) 의 S1 위에서 내린다.**

- **FLIP (부호 뒤집힘)** — `median Δ_BB > 0` **그리고** 양측 Wilcoxon `p < 0.05`
  ⟹ "두 팔을 모두 GT-free 로 놓으면 실측에서 learned 가 hand 를 이긴다."
  ⟹ **"실측 정확도 니치 없음" 판정을 뒤집는다.**
- **BOUND** — `median Δ_BB > 0` 이지만 `p ≥ 0.05`
  ⟹ 시사적·검정력 부족. 주장 금지, 효과크기+CI 만 보고. 니치 판정은 **유지되되 약화**로 기록.
- **HOLD (유지)** — `median Δ_BB ≤ 0`
  ⟹ "가장 큰 교란(GT 유래 입력)을 제거해도 learned 가 진다."
  ⟹ **"실측 정확도 니치 없음" 판정 유지·강화.** `p < 0.05` 면 확증적 패배로 기록.
- **부수 지표(판정 아님)**: 격차 축소율 `1 − median(Δ_BB) / median(Δ_FF)` 를 보고한다.
  이 값이 어떻든 위 3분류를 바꾸지 않는다.
- **BB400** 이 BB 와 다른 분류를 내면, 그 사실을 **판정 문장에 병기**한다(관측예산 비대칭이
  결론을 지배한다는 뜻이므로).
- **프로토콜 무효 조건**: FF-exact 가 `wp_real.json:hand_now` 와 창당 |Δ| > 1e-6 로 어긋나면
  재현 실패 ⟹ 전 실험을 무효로 보고하고 판정하지 않는다.

## 5. 손잡이 분리 (동결)
전부 **짝별(창별) 값을 먼저 만들고 그 median** 을 보고한다(중앙들의 차 금지 —
2026-08-05 B2 의 가짜 초가법성 전례). 양수 = **나빠짐**.
- `E_kp | σ=GT`   = median_w [ BF(w) − FF(w) ]
- `E_kp | σ=obs`  = median_w [ BB(w) − FB(w) ]
- `E_σ  | kp=GT`  = median_w [ FB(w) − FF(w) ]
- `E_σ  | kp=None`= median_w [ BB(w) − BF(w) ]
- `I (상호작용)`  = median_w [ (BB(w) − FB(w)) − (BF(w) − FF(w)) ]
- 가법성 검정: `median_w[ BB − FF ]` vs `median_w[(BF−FF)+(FB−FF)]` 를 **짝별로** 비교.
각 항에 양측 Wilcoxon p 와 부호수를 병기. **묶음 개입에 단일 원인 이름을 붙이지 않는다** —
FF 는 두 손잡이가 동시에 ON 인 묶음 셀임을 모든 문장에서 명시한다.

## 6. 층화 (동결)
레코드별(5개, 창 5–6개) median Δ 와 부호수. **레코드 단위 n=5–6 ⟹ 양측 부호검정 바닥 p:
n=5 → 0.0625, n=6 → 0.03125.** ⟹ **레코드 단위 α=0.05 주장 금지**(n=5 는 원리적 불가).
층화는 **기술적 보고**이며 판정 근거가 아니다.

## 7. 검정력 / 바닥 p (동결)
- S1 n=27: 양측 부호검정 바닥 p = 2·2⁻²⁷ ≈ 1.5e-8; `p<0.05` 에 필요한 최소 승수 = 19/27
  (양측 이항, 실행 전 재계산해 보고).
- MDE: 관측된 짝별 Δ 분포에서 부트스트랩(10,000회)으로 "Wilcoxon p<0.05 를 80% 확률로 얻는
  최소 위치이동"을 산출해 관측 효과크기와 나란히 보고한다.

## 8. 실행 (동결)
- hand: CPU 워커 8개, `OMP/OPENBLAS/MKL=1`, `nohup … &` + 로그, 폴링. 죽이지 않는다.
- learned 스팟체크: GPU, 6창, `wp_real` 과 동일 코드경로/시드.
- 총 solve 수 = 30창 × (5셀 × 3시드 + FF-exact 1) = **480**.
- 출력: `cells.json`, `null.json`, `spot_learned.json`, `A2_RESULT.md` — 이 디렉터리에만.
- 4단계: 동결 규약대로 채점한 뒤 **처음부터 다시 쓴 독립 스크립트**로 재도출해 대조한다
  (`verify_independent.py`). 불일치는 전부 보고한다.

## 9. 이 시험이 허가/불허하는 범위
- 허가: 두 팔을 동일한 GT-free 입력 위에 놓았을 때의 실측 짝별 비교 결론.
- 불허: 합성 혼잡 니치에 대한 어떤 변경도 아니다(다른 대상·다른 분포).
  실측 레코드 n=5 로 α=0.05 주장 금지. mean 기반 learned 승리 인용 금지.
  "3/3 붕괴 수리"·"insurance/보험" 프레이밍 금지. 벽시계는 기록하되 성능 주장 금지.
- 어떤 방향이 나오든 그대로 보고한다. 부호가 안 뒤집히는 것도 결과다.
