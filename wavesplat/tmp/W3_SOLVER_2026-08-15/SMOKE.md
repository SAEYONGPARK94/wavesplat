# W3_SOLVER — SMOKE (2 에피소드 × 10팔), 2026-08-15

**사전등록** `PREREG.md` md5 `9f4e9c7e7bf534253b4049b7c91cca71`
(**self-attested freeze** 2026-08-15 20:04:01 KST, **모든 런 이전**) + `AMENDMENT_1.md` md5 `db6d32039dc275ce45a5fff2b87ea647`.
**원자료** `smoke_results.jsonl` (49행) · 로그 `smoke.log` · 분석기 리허설 `dryrun_smoke.log`.
**에피소드** `BF4_30comp_seed20`(혼잡, 가장 비싼 셀) · `BF4_2comp_seed20`(희소) — 둘 다 `uniform_ra_rf_ds`, 채점 프레임 160.

> ⚠**스모크는 판정 자료가 아니다.** `smoke_results.jsonl` 은 `results.jsonl` 과 **분리**돼 있고
> `analyze_w3.py` 는 후자만 읽는다. 아래 숫자는 **계측이 작동함을 증명**하는 용도이고,
> n=1/층 이므로 **어떤 방향 주장도 아니다.**

---

## 1. 게이트 판정

| 게이트 | 실측 | 판정 |
|---|---|---|
| **G-2** DCC 스키마 | 48/48 행 위반 **0** (셸 9 · 제로셸 9 · 면적 9 · `effective_config` 포함) | ✅ |
| **G-3** λ 조작검사(**객체 되읽기**) | 불일치 **0** — `1e-3 / 1e-4 / 0.0` 이 팔별로 그대로 확인됨. 앙상블 멤버 불일치 0 | ✅ |
| **G-4** FITPHI 조작검사(**원자 단위**) | OFF 팔 전 행 `fitphi_atoms=0`; ON 팔은 아래 §2 | ⚠ (§2) |
| **G-5** 누출 프로브 | `ORA-K@L3-DUP` vs `ORA-K@L3` **\|Δ\| = 0.000e+00** (3/3 시드) | ✅ |
| **G-6** 시드 퍼짐 | 어떤 팔도 퍼짐 0 아님 ⟹ **가짜복제 아님**(오라클 팔도 `torch.rand` 위상 때문에 시드 의존) | ✅ |
| **G-7** 코드 지문 | 87파일 단일 digest `696fa102bb05dd23f04ea18c8421ee12`, 전 행 동일 (`code_fingerprint.json` 과 일치) | ✅ |
| **G-8** `ZERO` 안정 | 10팔에 걸쳐 에피소드당 **단일값**(0.240048 / 0.250006) | ✅ |
| **G-9** 시드 원장 | 소모 `{1592,1593,1594}`, 1500–1900 기존 최대 **1591** ⟹ 충돌 **0**, 신규 `.h5` **0** | ✅ |
| **G-11** CVX 자기검증 | rank **60/60 · 4/4** (full), 창 잔차가 **진실보다 나쁜 행 0** | ✅ |
| G-1 / G-10 | 스모크라서 **설계상 FAIL**(48/1200 행) | — |

---

## 2. ★★★G-4 — 배포 팔에는 **개입 지점이 없다** (설계 중 예측 → 실측 확인)

원자 단위 카운터(`_linear_projection` 반환 위상 ↔ `add_gaussians` 의 `init_phi` 대조):

| 팔 | 층 | 개입한 런 | 평균 `fitphi_atoms` | 게이트 호출 지점 |
|---|---|---|---|---|
| `KONLY-FITPHI` | 혼잡 | **3/3** | **13.3** | `coarse_to_fine.py:173` (사이트 ①) |
| `KONLY-FITPHI` | 희소 | **3/3** | **12.0** | 〃 |
| **`SHIP-FITPHI`** | **혼잡** | **0/3** | **0.0** | — |
| `SHIP-FITPHI` | 희소 | 1/2 | 18.0 | 〃 |
| OFF 팔 8개 전부 | 양 층 | 0 | 0.0 | — |

- **원인**(코드에서 확인): `sequential_update._initialize_learned` 가 *"the coarse-to-fine
  N-staging is bypassed"* ⟹ `init_method='auto'` 가 `learned` 로 라우팅되면 사이트 ①은 **0회 호출**된다.
  스모크에서 `SHIP`/`SHIP-FITPHI` 는 혼잡 3/3 이 `learned` 로 라우팅됐다.
- **직접 증거**: 혼잡에서 `SHIP` 과 `SHIP-FITPHI` 의 세 시드 값이 **완전히 동일**하다
  (0.2379 / 0.2257 / 0.2549) ⟹ 처치가 **한 비트도 닿지 않았다**.
- ⟹ **S-3(배포 위 수리)은 전량에서 `UNDECIDED-NO-INTERVENTION` 이 될 가능성이 높다.**
  위상 수리를 실제로 시험하는 것은 **S-4(`KONLY` 짝)** 다. 이걸 **런 이전에** 알아낸 것이
  이 배치 설계의 소득이다(저장소 전례: "처치 팔이 해당 코드를 0회 호출" 4회 이상).

---

## 3. ★★★상한 `ORA-K-CVX` — 저장소 채점기 위에서 확인

| 에피소드 | `ORA-K-CVX` | `ZERO` | cond(Φ) | rank | 창 잔차(해) | 창 잔차(진실) |
|---|---|---|---|---|---|---|
| 혼잡 `BF4_30comp_seed20` | **0.00201** | 0.24005 | 32.5 | 60/60 | 0.00548 | 0.00583 |
| 희소 `BF4_2comp_seed20` | **0.00144** | 0.25001 | 1.0 | 4/4 | 0.00576 | 0.00601 |

- **경사하강 0 스텝**, 관측은 추정기와 **같은 초기화 창**(혼잡 7,034점 / 희소 7,012점), 채점은 **같은
  `compute_all_metrics`**, 격자는 **미관측 영역 포함 전역 GT 격자**.
- 해의 창 잔차가 **진실보다 작다**(전역 최소점이므로 당연) ⟹ 파이프라인이 옳게 물려 있다는 증거.
- 셸이 **평평하다**(0.0023–0.0025, 근/원 차이 사실상 없음) ⟹ `SUPPORT-TRADEOFF` 가 **여기엔 없다**.
- 진단: `cvx_gtomega_nrmse` 가 주값과 소수 5자리까지 일치 ⟹ **분산관계 ω = GT ω** (일관성 확인).

> ⚠**해석 한계(반드시 병기).** 합성 GT 필드는 `Σ A·cos(k·x − ωt + φ)` 이고 모델도 **같은 식**이다
> ⟹ `ORA-K-CVX` 는 **모델 계열이 진실을 정확히 포함하는** 조건에서의 상한이다.
> 따라서 이 값이 재는 것은 **솔버 격차**이지 **모델-계열 격차가 아니다.**
> (MEMORY 08-13 의 "희소 −93% 는 표현 아티팩트" 와 같은 함정. 같은 주의를 여기에 적용한다.)

---

## 4. 스모크 관측치 (n=1/층 — **방향 주장 아님**)

혼잡 `BF4_30comp_seed20`, 시드 3개 중앙값:

| 팔 | nRMSE | `ZERO` 대비 |
|---|---|---|
| `ORA-K-CVX` | **0.00201** | 이김 (×119) |
| `ORA-K-NOREG` (λ=0) | 0.14521 | 이김 |
| `ORA-K@L4` (λ=1e-4) | 0.17299 | 이김 |
| `SHIP` = `SHIP-FITPHI` | 0.23790 | 이김 |
| `ORA-K@L3` = `ORA-K@L3-DUP` (λ=1e-3) | 0.25337 | ⛔**짐** |
| `KONLY` 0.26435 · `KONLY-FITPHI` 0.26914 | | ⛔**짐** |
| `ZERO` | 0.24005 | — |

- ⚠**PRIMARY 대조 `ORA-K@L3` 이 이 에피소드에서 `ZERO` 보다 나쁘다** ⟹ 전량에서도 그러면
  `AMENDMENT_1 A-3` 에 따라 판정에 **`(DEGENERATE-CONTROL)` 꼬리표**를 붙이고,
  "λ=0 이 이긴다" 가 아니라 **"λ=1e-3 이 오라클 팔을 평평한 바다보다 못하게 만든다"** 로 쓴다.
- 최선의 경사하강 팔(`ORA-K-NOREG` 0.145)과 닫힌형 상한(0.002) 사이가 **72배**다.

---

## 5. 비용 — **전량 ≈ 1.8 h** (2위상 실행)

팔별 **최소** `wall_s`(경쟁 없는 조건에 가장 가까운 값). ⛔중앙값은 경쟁 오염으로 최대 5배 부풀었다
(`SHIP|sparse` 24.3 s → 113.5 s).

| 팔 | 혼잡 30comp | 희소 2comp | 잡 수 | 순차 |
|---|---|---|---|---|
| `ZERO` | 1.7 s | 1.7 s | 48 | 0.02 h |
| `ORA-K-CVX` | 3.9 s | 2.2 s | 48 | 0.04 h |
| `ORA-K@L3` | 34.0 s | 30.7 s | 144 | 1.32 h |
| `ORA-K-NOREG` | 31.8 s | 31.2 s | 144 | 1.26 h |
| `ORA-K@L4` | 33.2 s | 31.2 s | 144 | 1.30 h |
| `ORA-K@L3-DUP` | 33.8 s | — | 96 | 0.90 h |
| `SHIP` | 23.6 s | 24.3 s | 144 | 0.95 h |
| `SHIP-FITPHI` | 23.1 s | 23.2 s | 144 | 0.93 h |
| `KONLY` | 40.8 s | 37.4 s | 144 | 1.59 h |
| `KONLY-FITPHI` | 37.7 s | 37.6 s | 144 | 1.51 h |
| **합계** | | | **1,200** | **9.82 h 순차** |

- 스모크 병렬 효율 실측: `sum(wall)=1,601 s` / `elapsed=210 s` = **×7.62** (8워커).
- **위 순차 합은 상한이다** — 혼잡 32 중 16 이 15comp(오라클 팔은 N 에 비례해 더 싸다),
  희소 16 중 8 이 5comp 인데 전부 가장 비싼 셀 값으로 곱했다.
- **예상**: 8워커 단일 위상 **≈1.3 h**, `AMENDMENT_1 A-2` 의 2위상 실행 **≈1.8 h**(권장).

### ⚠GPU 병목 (실측)

49잡 중 **1잡이 `CUDA error: out of memory`** 로 실패했다 — `init_method='auto'` 의 학습 제안기가
GPU 를 쓰는데 8워커가 4080 Super 하나를 동시에 물기 때문이다. 저장소에서 **재발**하는 결함이다.
⟹ `AMENDMENT_1 A-2` 의 위상 분리로 회피한다. 러너는 실패 잡을 `done` 에 넣지 않으므로
**같은 명령 재실행이 곧 재시도**다.

---

## 6. 실행 (권장, `AMENDMENT_1 A-2`)

```bash
cd /home/vexit/wavesplat/wavesplat
# Phase A -- GPU 미사용, ≈1.1 h
WORKERS=8 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full \
  ZERO,ORA-K-CVX,ORA-K@L3,ORA-K-NOREG,ORA-K@L4,ORA-K@L3-DUP,KONLY,KONLY-FITPHI \
  2>&1 | tee -a tmp/W3_SOLVER_2026-08-15/run.log
# Phase B -- init_method='auto' (GPU), ≈0.7 h
WORKERS=3 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full SHIP,SHIP-FITPHI \
  2>&1 | tee -a tmp/W3_SOLVER_2026-08-15/run.log
# Phase C -- 실패 잡 재시도 (성공 잡은 스킵)
WORKERS=3 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full 2>&1 | tee -a tmp/W3_SOLVER_2026-08-15/run.log
# 판정
python3 tmp/W3_SOLVER_2026-08-15/analyze_w3.py | tee tmp/W3_SOLVER_2026-08-15/VERDICT.log
```

한 줄로 돌릴 경우(감사가 위상 분리를 불필요하다고 판단하면):

```bash
WORKERS=8 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full 2>&1 | tee tmp/W3_SOLVER_2026-08-15/run.log
```
