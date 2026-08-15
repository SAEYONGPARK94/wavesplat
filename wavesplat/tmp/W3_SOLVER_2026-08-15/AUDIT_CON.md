# W3_SOLVER — AUDIT_CON (런 전 감사, 2026-08-15)

**역할** = 이 배치를 런 전에 죽인다. **READ-ONLY** — 이 파일 외 어떤 파일도 수정하지 않았다.
**전량 실행 0회.** 하위 에이전트 0회.

**열람 범위 (= 주장 범위)**
`tmp/W3_SOLVER_2026-08-15/{PREREG.md, AMENDMENT_1.md, SMOKE.md, NOTES.md, FROZEN_md5.txt,
run_w3.py, analyze_w3.py, dryrun_smoke.py, dryrun_smoke.log, smoke.log, smoke_results.jsonl(49행),
code_fingerprint.json}` ·
`wavesplat/{run.py, data_loader.py, evaluation/metrics.py, optimizer/{sequential_update.py,
coarse_to_fine.py, initializer.py, learned_init.py}, model/spectral_gaussian.py}` ·
`tmp/rx2_init_2026-08-05/run_rx2b.py` · `tmp/{S4,S3,S1,K6}_*/run_*.py`(λ grep) ·
`memory/reference_citation_bans.md`(전문) · 코퍼스 h5 `config` 속성 4개.
동결 md5 2건 재검증: `PREREG.md 9f4e9c7e…` ✅ · `AMENDMENT_1.md db6d3203…` ✅.

**판정: `GO-WITH-FIX`** — BLOCKER **7** / WARN **11**.
7건 모두 **런 전 수정 가능**하고 **재실행을 요구하지 않는다**(5건 문서·분석기 편집, 1건 무비용 기록 필드,
1건 12잡 ≈5분 프로브).

---

## BLOCKER (7) — 심각도 순

### B-1. PREREG §1.1 이 **인용금지 수치를 근거로 쓴다** (배치 동기 자체)

`PREREG.md:23` — *"**`ORA-K` 계열 4연속 널** — K6 (**p=0.79**) · S1 (8/16) · S3 (8/16) · S4 (6/16)."*

`memory/reference_citation_bans.md:74` 가 이 문장을 **명시적으로 금지**한다:

> ⛔**"K6: 완벽한 k 를 줘도 0(p=0.79)"** → 그 팔은 `ORACLE-K` = **3요인 묶음**; 같은 파일
> `ORACLE−AUTO` = **−0.0640, 14/14, p=1.22e-4** ⟹ **k 통로는 열려 있다**.

⟹ 이 배치의 **한 문장(§0)과 동기(§1.1)가 금지 수치 위에 서 있다.** 규율상 이대로 결과를 쓰면
**보고가 무효**다. 또한 금지목록의 정정("k 통로는 열려 있다")은 *"네 번 모두 널"* 이라는 프레임을
직접 깎는다 — 네 사례 중 하나는 널이 아니라 **묶음 때문에 널로 보인 것**으로 이미 재판정돼 있다.

**수정안 (AMENDMENT_2, 런 전, 문턱·팔·통계량 불변)**
`§1.1(1)` 을 *"`ORA-K` 계열 널 **3건**(S1 8/16 · S3 8/16 · S4 6/16) + K6 **1건은 인용금지·묶음
재판정**(`bans:74`)"* 로 재기술하고, `§0`·`§8` 에서 **"4연속 널"** 표현을 제거한다.
⛔RESULT.md 가 *"네 번 모두 널이었다"* 를 상속하지 못하게 못 박을 것.
(`§9.5` 의 3요인 묶음 라벨은 `bans:58` 요구를 이미 충족 ✅ — 이건 유지.)

---

### B-2. Holm-4 에 **step-down 정지 규칙이 없다** — FWER 팽창

`analyze_w3.py:312-318`
```python
order = sorted(sec.items(), key=lambda kv: kv[1]['p_exact'])
for i, (sid, rr) in enumerate(order):
    adj = ALPHA / (m - i)
    holm = rr['p_exact'] <= adj and rr['wins'] >= rr['threshold']
```
Holm 은 **가장 작은 p 가 α/m 를 못 넘는 순간 나머지 전부 기각 불가**다. 위 루프는 각 가설을
**독립적으로** 문턱과 비교하므로, `p₁ > α/4` 인데 `p₂ ≤ α/3` 이면 **S-x 를 PASS 로 인쇄한다.**
`PREREG.md:186` 도 정지 규칙 없이 *"오름차순으로 α/4, α/3, α/2, α 와 비교"* 라고만 적어 같은 오류를 공유한다.

**왜 치명적인가** — 이 족(族)에 **S-2**(`ORA-K-NOREG < SHIP`)가 들어 있고, S-2 는 `§8` 판정표의
두 축 중 하나다. 즉 **판정 이름이 잘못된 다중성 보정 위에서 결정될 수 있다.**

**수정안** (3줄, 런 전)
```python
alive = True
for i, (sid, rr) in enumerate(order):
    adj = ALPHA / (m - i)
    alive = alive and rr['p_exact'] <= adj and rr['wins'] >= rr['threshold']
    rr['holm_pass'] = bool(alive) and rr.get('verdict') != 'UNDECIDED-NO-INTERVENTION'
```
그리고 `PREREG` 의 Holm 문장에 *"첫 미기각에서 정지"* 를 AMENDMENT_2 로 명시.

---

### B-3. SECONDARY 문턱 **23/32 는 자기 Holm 규칙 아래에서 도달 불가능하다**

정확 이항 단측(내가 직접 계산, `n=32, p=0.5`):

| X≥ | 21 | 22 | **23** | **24** | 25 |
|---|---|---|---|---|---|
| p | 0.055092 | 0.025051 | **0.010031** | **0.003500** | 0.001051 |

`ALPHA=0.025` (`analyze_w3.py:26`) ⟹ Holm 문턱 α/4=**0.00625** · α/3=**0.00833** · α/2=0.0125 · α=0.025.
⟹ **p 가 가장 작은 두 SECONDARY 는 24/32 가 필요**하고(23/32 의 0.010031 > 0.00833),
`PREREG.md:181-184` 표가 네 칸 모두에 적어 둔 **"문턱 23/32" 는 그 두 칸에서 거짓**이다.
(코드는 `p ≤ adj` **와** `wins ≥ threshold` 를 **둘 다** 요구하므로 실제 동작은 보수적 ✅ —
깨지는 것은 **문서**이고, 그 문서가 결과 해석에 쓰인다.)

**왜 BLOCKER 인가** — 이 저장소는 **"하나 차이"로 판정이 갈린 전례**가 있다(S8 희소 22/32,
`MEMORY` 08-11). 결과를 본 뒤 *"S-2 가 23/32 로 문턱을 넘었다"* 라고 쓰는 사고가 **구조적으로 열려 있다.**

**수정안** — AMENDMENT_2 에 SECONDARY 실효 문턱을 **명시적으로 표로** 동결:
`p 순위 1·2 → 24/32` · `3·4 → 23/32`. 문턱 열을 지우고 "Holm 정지 규칙 + 정확 p" 만 남겨도 된다.

---

### B-4. `§4.3` 의 **DEGENERATE 금지가 SECONDARY 에 구현돼 있지 않다**

`PREREG.md:194-195` — *"`ZERO` 를 이기는 에피소드가 ≤50% 이면 `DEGENERATE` … **그 팔이 들어간 어떤
우월성 문장도 쓰지 않는다**"*.

`analyze_w3.py:290` 은 **P-1 에만** 꼬리표를 붙인다:
```python
tag = '' if PRIMARY[2] not in degen['crowded'] else '  (DEGENERATE-CONTROL)'
```
`analyze_w3.py:308-321` 의 SECONDARY 루프는 `degen` 을 **한 번도 참조하지 않는다**
(`dead` = FITPHI 개입 부재만 본다).

**실측 근거 — 이건 반드시 발화한다.** 스모크 혼잡 `BF4_30comp_seed20`(시드 3 중앙값,
`smoke_results.jsonl`):
`ZERO` **0.24005** vs `KONLY` **0.26435** · `KONLY-FITPHI` **0.26914` — **둘 다 평평한 바다에 진다.**
⟹ **S-4(`KONLY-FITPHI < KONLY`) = 위상 수리를 실제로 시험하는 유일한 대비**가
**두 퇴화 팔 사이의 대비**인데 꼬리표 없이 PASS/FAIL 로 인쇄된다.

**수정안** (3줄, 런 전) — SECONDARY 루프에서
```python
if (t in degen['crowded']) or (c in degen['crowded']):
    rr['degenerate'] = True      # 인쇄에 (DEGENERATE-ARM) 꼬리표
```
그리고 `§4.3` 의 "우월성 문장 금지"를 S-4 에 어떻게 적용할지 **결과 전에** 한 줄로 동결
(A-3 가 P-1 에 대해 한 것과 같은 형식).

---

### B-5. **배포 팔의 전제(§1.4)가 PRIMARY 층에서 거짓이다** — 그리고 §9.1 의 배포 후보가 사라진다

`PREREG.md:30-34` 는 P1 축의 동기를 *"배포 경로는 `coarse_to_fine.py:177` 의
`torch.rand(n_new)*2π`"* 라고 적었다. **혼잡(PRIMARY) 층의 출하 경로는 거기로 가지 않는다.**

코드 증거 (내가 읽음):
- `sequential_update.py:443` `init_method=='auto'` → `:511` `_auto_select_init` → 혼잡에서 `'learned'`.
- `sequential_update.py:516-517` → `_initialize_learned` → `:619-624` 문서 *"the coarse-to-fine
  N-staging is **bypassed**"* ⟹ `coarse_to_fine.py:173-179`(FITPHI 사이트 ①) **0회 호출**.
- ★그리고 `_initialize_learned` 가 심는 위상은 난수가 **아니다**:
  `learned_init.py:361` `w, phi = ls_fit_wphi(kx, ky, frames, depth)` →
  `learned_init.py:306-320` → `fit_amplitudes_phases(..., lambda_reg=0.5)`
  ⟹ **출하 혼잡 경로는 이미 (w, φ) 를 LS 적합해서 설치한다.**
- 실측 확인: 스모크 혼잡 `SHIP`/`SHIP-FITPHI` 6행 전부 `init_method_resolved='learned'`,
  `lp_calls=0`, `add_calls=0`, `fitphi_atoms=0`, 그리고 세 시드 값이 **비트 동일**
  (0.237897 / 0.225657 / 0.254893).

⟹ **정정 3건이 필요하다.**
1. *"배포 경로 = 난수 위상"* 은 **혼잡 출하 경로에 대해 거짓**이다. 참인 범위는
   `v2`/`v2_konly` **사다리 성장 경로**뿐이다(라우터가 그쪽으로 보낸 에피소드에 한해).
2. `PREREG.md:275-276` §9.1 *"배포 후보는 `SHIP-FITPHI` 하나뿐"* ⟹ **무효**.
   개입 지점이 없으므로 **이 배치에는 배포 후보가 0개**다. 그렇게 적어야 한다.
3. SMOKE §2 의 결론 *"배포 팔에는 개입 지점이 없다"* 는 맞지만 **불완전**하다. 정직한 문장은
   *"혼잡 출하 경로는 위상을 버리지 않는다 — 그 사이트를 아예 지나지 않고, 이미 LS 적합 위상을
   설치한다(ridge 0.5)"* 이고, 이건 S-3 의 `UNDECIDED` 를 *"시험 실패"* 가 아니라
   ***"여기엔 고칠 것이 없다"*** 로 바꾼다.

**수정안** — AMENDMENT_2 로 §1.4·§9.1·SMOKE §2 세 곳을 재기술. **런 전 문서 작업, 비용 0.**

---

### B-6. `ORA-K-CVX` 상한 — **미신고 오라클 요인 1개 + §8 의 첫 분기가 사실상 반증 불가**

**정직한 부분은 먼저 인정한다**(§"공격 실패" 참조): 적합 집합·채점기·프레임 정합은 깨끗하다.
문제는 **72배를 무엇에 귀속시키는가**이다.

**(a) 미신고 요인: CVX 는 k 를 진실에 고정하고, `ORA-K` 는 k 를 학습한다.**
- `run_w3.py:65` `ORA = dict(init_method='oracle', …)` 에 `init_k_lr_scale` **없음** ⟹ 기본 1.0
  (`sequential_update.py:262` `k_lr = self.lr * 0.5 * init_k_lr_scale`) ⟹ **k 학습 가능**.
- `tmp/rx2_init_2026-08-05/run_rx2b.py:174-176` `__oracle_k__` 도 k 학습 가능(재현 일치 ✅),
  `:170-173` `__oracle__` 만 `init_k_lr_scale=0.0`.
- `run_w3.py:371-381` CVX 는 `kx, ky` 를 GT 에서 **그대로 고정**하고 (a,b)만 푼다.
⟹ 72배는 **{k 동결 vs k 학습} × {닫힌형 vs 900 Adam 스텝} × {무추적 vs 추적} × {무가중 vs
confidence 가중}** 의 **4요인 묶음**이다. `PREREG.md:285-286` §9.7 은 **뒤 두 개만** 신고했다.

**(b) §8 의 첫 분기는 결과 전에 이미 결정돼 있었다.**
`memory/reference_citation_bans.md`(08-13 R3): 저장된 `gt_wave_field` 를 `Σ A·cos(kx·x+ky·y−ωt+φ)`
로 재구성하면 **`max|Δ| = 0.000e+00`**, 그리고 *"'표현 아티팩트' 단서를 희소에만 붙이지 마라 —
**코퍼스 전체에 해당한다**"*. ⟹ 모델 계열이 진실을 **정확히** 포함하고 k 를 진실로 주면,
7,034 관측 · 60 미지수 · `cond=32.5` 에서 `CVX ≈ 0` 은 **사전에 보장된다**.
`PREREG.md:268` *"`CVX ≈ 0` 이면 ⟹ … **솔버**가 벽이다"* 는 **거의 반증 불가능한 분기**다.
(SMOKE §3 의 한계 문단은 이 함정을 **이미 적어 두었다** ✅ — 고쳐야 할 것은 §8 의 추론 사다리다.)

**수정안 (둘 다 런 전, 추가 런타임 0)**
1. **k 표류를 기록하라.** `run.py:480` 이 이미 `spectrum_params`(`model.get_spectrum_params()`)를
   반환한다. `run_w3.run_est` 에서 오라클 팔의 최종 `(kx,ky)` 와 GT `(kx,ky)`(순서 보존 —
   `initializer.py:624-627` *"order as stored"*)의 `k_drift_rms` 를 한 필드로 남긴다.
   표류 ≈ 0 이면 (a)의 요인이 **실측으로 소거**되고 "솔버" 귀속이 살아난다. 비용 0.
2. §8 을 **격차 서술**로 재기술: *"CVX 는 4요인 묶음 상한이며, 재는 것은 **솔버 격차**이지
   모델계열 격차가 아니다"*(SMOKE §3 문장을 PREREG 로 승격).

---

### B-7. **GPU/learned 경로에 재현 프로브가 없고, A-1 의 `NONDETERMINISM` 안전밸브는 구조적으로 발화 불가**

- `run_w3.py:659-668`: `done` 은 **성공 행의 `job_id`** 로 만들어지고 `todo` 에서 제외된다
  ⟹ **한 번 성공한 잡은 두 번째 성공 행을 절대 갖지 않는다.**
- `analyze_w3.py:56-60`: `dup` 은 *같은 `job_id` 의 두 성공 행*이 있을 때만 채워진다.
  ⟹ **`AMENDMENT_1 A-1` 의 `NONDETERMINISM` 검사는 죽은 코드다.** 그런데 A-1 은 이걸
  *"재시도가 결과를 흔들었다면 게이트를 실패시킨다"* 는 **보호 장치로 제시**하고 있다.
- 유일한 누출 프로브 `ORA-K@L3-DUP` (`run_w3.py:71-72`)은 **혼잡·오라클·CPU 전용**이다.
- 하필 **S-2 의 대조인 `SHIP` 이 위험한 팔**이다: `learned_init.py:130-131` 이
  `device = 'cuda' if torch.cuda.is_available() else 'cpu'` 로 **cfg 의 `device='cpu'` 를 무시**하고
  GPU 확산 제안기를 돌린다(스모크 49잡 중 **1잡 CUDA OOM**), 그리고 저장소는
  `learned↔v2_konly` 라우트가 **시드에 따라 31/128(24%) 뒤집힌다**고 이미 측정해 두었다
  (`bans`, S13b). A-2 는 이 팔을 **다른 워커 수(8 → 3)** 에서 돌리도록 나눈다.

**수정안** — `SHIP-DUP`(=`SHIP` 과 바이트 동일, 혼잡 **4 에피소드 × 3 시드 = 12잡**, WORKERS=3 에서
**≈5분**)을 추가하고 게이트 **G-5b**: `|Δ|=0` 아니면 중단. Phase B 안에서 같이 돈다.
비용은 `SHIP-FITPHI` 144잡(≈0.93 h)의 **1.4%** 다(W-10 참조).

---

## WARN (11)

| # | 내용 | 근거 | 수정안 |
|---|---|---|---|
| **W-1** | `AMENDMENT_1 A-3` 이 *"우리 주장을 **약화시키는 것뿐**"* 이라 자평하는데 **부정확**하다. `PREREG §4.3` 은 퇴화 팔이 든 **우월성 문장을 금지**했고, A-3 는 그 **금지를 꼬리표부 허용으로 바꾼다** ⟹ **허용 방향의 사후 완화**이기도 하다. 스모크 방향을 본 뒤(20:11→20:13) 작성됐다. | `AMENDMENT_1.md:51-59` vs `PREREG.md:194-195`; `FROZEN_md5.txt` 시각 | A-3 자기평가 문장을 *"약화 + 금지→조건부 허용"* 으로 정직하게 재기술 |
| **W-2** | G-4 경계 불일치. PREREG *"개입 부재 ≥50% ⟹ UNDECIDED"*(⟺ `h ≤ n/2`) vs 코드 `h < 0.5*n`. **정확히 16/32 에서 판정이 갈린다.** | `PREREG.md:117-119` vs `analyze_w3.py:120-121` | `h <= 0.5*n` 로 통일 |
| **W-3** | **15comp 혼잡 셀이 스모크에 없다.** 거기선 `n_gt=15 < max_n=30` 이라 오라클 팔이 `_adaptive_prune_grow` 의 beamform 성장 분기에 **들어갈 수 있다**(30comp 는 `N=max_n` 이라 `n_add≤0` — 그래서 스모크가 `lp_calls=0`). ⟹ (i) `ORA-K` 가 PRIMARY 층 절반에서 **4요인 묶음**이 되고, (ii) **G-4 의 "OFF 팔 전 행 `fitphi_atoms==0`" 이 그 사이트가 발화할 수 있는 바로 그 셀에서 미검증**이다. 발화하면 게이트가 **1.8 h 뒤에** 배치를 죽인다. | `sequential_update.py:1095-1104`; 스모크 30comp 행 `lp_calls=0` | 스모크에 15comp 혼잡 1 에피소드 추가(오라클 2팔 × 3시드 ≈ 3분) |
| **W-4** | §2.1(6)/§9.7 이 CVX 의 **무추적**을 *"우리에게 유리한 방향이 아니다"* 로 **부호까지 단정**한다. 저장소 자신의 측정은 반대다: `Rx-7` 관측 −0.1634(13/13) vs 미관측 +0.0400(0/13), S3 *"추적기 끄기 **5번째 재현**(15/16)"*. | `MEMORY.md` 08-07·08-08 | *"부호 미상, 저장소 기록은 무추적이 **유리**할 수 있음을 시사"* 로 재기술 |
| **W-5** | **유한성 게이트 부재.** `missing_fields` 는 **키 존재만** 본다 ⟹ `avg_nRMSE=NaN` 행이 G-2 를 통과하고, 부호검정에서 `(d<0)=False` 로 **조용히 패로 계수**된다. `abs(nan-nan)>0` 도 False 라 dup 검사도 못 잡는다. | `run_w3.py:583-596`; `analyze_w3.py:58,227` | G-2 에 `np.isfinite(avg_nRMSE, zero_nRMSE)` 추가 |
| **W-6** | **판정 로직이 모든 봉인 밖**이다. `code_fingerprint()` 의 자체 파일은 `run_w3.py` + `PREREG.md` **둘뿐**이고, `analyze_w3.py` 와 `AMENDMENT_1.md` 는 지문에도 `FROZEN_md5.txt` 에도 없다 ⟹ `results.jsonl` 이 생긴 뒤 분석기를 고쳐도 **어떤 게이트도 못 잡는다.** | `run_w3.py:151`; `code_fingerprint.json`(자체 2파일); `FROZEN_md5.txt` | 지금 두 파일 md5 를 `FROZEN_md5.txt` 에 추가 + 분석기가 자기 md5 를 `VERDICT.log` 첫 줄에 인쇄 |
| **W-7** | **numpy BLAS 스레드가 고정되지 않는다.** `import numpy` 는 워커 spawn 시 모듈 최상단(`run_w3.py:28`)에서 실행되는데 `OPENBLAS/MKL/OMP_NUM_THREADS` 는 그 **뒤에** `run_one` 에서 setdefault 된다(`:552-553`). torch 는 함수 안 임포트라 고정됨 ✅, **CVX 의 `np.linalg.lstsq` 만 비고정**. | `run_w3.py:28, 552-553` | setdefault 3줄을 `import numpy` **위로** 이동(1줄 이동) |
| **W-8** | `float(r['lambda_l1_effective'])` 가 되읽기 실패값(`None`/`'<absent>'`)에서 **FAIL 이 아니라 예외**로 죽는다 ⟹ 게이트 메시지 없이 트레이스백. | `analyze_w3.py:95` | `try/except` → G-3 FAIL 로 강등 |
| **W-9** | **라우터 불안정이 미보고.** `learned↔v2_konly` 가 시드에 따라 **31/128(24%)** 뒤집힌다(S13b). 스모크에서 이미 발생: 희소 `SHIP` 시드 1592/1593 → `learned`, 1594 → `v2_konly`. ⟹ `SHIP` 의 에피소드 중앙값이 **서로 다른 두 추정기의 중앙값**일 수 있다(집계는 동결돼 있으므로 규약 구멍은 아님). | `bans`(S13b); `smoke_results.jsonl` `init_method_resolved` | 분석기가 S-2/S-3 옆에 `SHIP`/`SHIP-FITPHI` 의 **에피소드별 라우트 구성**을 인쇄 |
| **W-10** | **비용 배분.** `SHIP-FITPHI` 144잡(≈0.93 h 순차)은 **코드 + 스모크로 이미 확립된 사실**(혼잡 개입 0)을 전량 n 으로 재진술하는 데 쓰인다. PREREG §10 의 삭감 순위에서 이미 2번째다. 반면 GPU 경로 재현 프로브는 **0개**(B-7). | `SMOKE.md:109`; `PREREG.md:300-302` | `SHIP-FITPHI` 일부를 **B-7 의 `SHIP-DUP` 12잡**과 맞바꾸는 것을 권고 |
| **W-11** | 계수 인용. §5 가 *"`SUPPORT-TRADEOFF` **7회** 재현"* 이라 적는데, 08-15 축 감사가 후보를 **10건**으로 재열거하며 **`K6` 을 후보에서 실격**시켰다 ⟹ 계수가 불안정. 또 `SMOKE.md:64` *"셸이 평평하다 ⟹ `SUPPORT-TRADEOFF` 가 여기엔 없다"* 는 **내부 노브 대조가 아닌 팔**에 그 명제를 읽는다(허용된 읽기는 *"우리 노브를 돌리면 근거리를 팔아 먼바다를 산다"*). | `bans:96-100`, `bans:78` | 계수 삭제(의무는 유지) · SMOKE §3 문장 재라벨. W3 의 셸 분해는 **λ·FITPHI = 내부 노브** 대조이므로 **허용된 읽기 그 자체다** ✅ |

---

## 공격 실패 목록 (= 이 배치의 **단단한 곳**)

내가 죽이려 했으나 **원자료·코드가 방어한** 항목. 이건 감사 통과로 기록해도 된다.

1. **`ORA-K-CVX` 가 GT 격자로 적합했다** → **거짓.** `run_w3.py:365-369` 가
   `loader.get_init_window(tp_factor=2.0)` 의 `(x_obs,y_obs,z_obs,t)` 만 쌓는다 =
   **추정기 Phase-1 이 보는 바로 그 관측**(혼잡 `cvx_n_obs=7034`). GT 격자는 **평가에만** 쓰인다.
2. **채점기·프레임·정규화가 다르다** → **거짓.** CVX/ZERO/추정기 모두
   `get_tracking_frames(tp_factor=2.0)` 의 **동일 160 프레임**(`n_frames_scored=160` 전 행 일치),
   동일 `compute_all_metrics`(`run_w3.py:404-406` vs `run.py:282-288`, 인자 순서·keyword 동일),
   동일 `nRMSE = RMSE/Hs`(`metrics.py:50`), `zero_nRMSE = √mean(gt²)/Hs` = 제로장의 nRMSE ✅.
3. **셸이 팔마다 다른 정의다** → **거짓.** 경계 `run_w3.py:57` == `run.py:358` **바이트 동일**,
   둘 다 **최근접 관측점 거리** 최소값(`run_w3.py:302-305` vs `run.py:354-357`), 둘 다 **원시 RMSE[m]**
   ⟹ 단위 혼합 없음. `obs:` 접두어로 `bans:78`("셸을 배-거리로 인용 금지")를 **선제 준수** ✅.
4. **CVX 가 데이터를 더 봤다** → **반대다.** CVX 는 **초기화 창만** 보고 **추적 갱신 0회**,
   추정기 팔은 160 추적 프레임의 관측을 **추가로** 받는다. G-11 실측: 해의 창 잔차 **0.005482** <
   진실의 창 잔차 **0.005833**(전역 최소점이므로 정상), rank **60/60**, cond **32.5**.
5. **가짜복제(S4 의 48/48 결함) 재발** → **불가.** `analyze_w3.py:203-208` 이 **시드 중앙값 먼저**,
   `:220-233` 이 **에피소드 단위** 부호검정, `n = 에피소드 수`. G-6 은 보고 전용이며 그게 맞다.
   스모크 실측 시드 퍼짐 **8.8e-3 ~ 5.9e-2**(퍼짐 0 팔 없음).
6. **DCC 가 스모크에만 걸린다(S15 결함)** → **거짓.** `analyze_w3.py:82-90` 이 `results.jsonl`
   **전 행**에 `missing_fields` 를 돌리고 위반 시 **`sys.exit(3)` 로 어떤 판정도 인쇄 전에 종료**.
   스모크 48행 위반 0(셸 9 × 3족 × 48 = 1,296 항목 전부 유한).
7. **FITPHI 프로브가 처치를 바꾼다** → **거짓.** 세 래퍼 모두 **RNG 를 소비하지 않고 반환값을
   바꾸지 않는다**(`run_w3.py:193-233`). `coarse_to_fine.py:32` 가 `_linear_projection` 을
   **모듈 최상단 임포트**하므로 `C._linear_projection` 은 `I._linear_projection` 과 **같은 객체**
   ⟹ 같은 래퍼로 감싸는 것이 안전하고, `sequential_update.py:1110` 은 **호출 시점 임포트**라
   사이트 ②도 덮인다. `_linear_projection` 은 **파이썬 float 2개**를 반환하므로
   (`initializer.py:450`) float32 대조가 정확히 성립한다. 증명: `ORA-K@L3` vs `-DUP` **|Δ|=0.000e+00**,
   혼잡 `SHIP` vs `SHIP-FITPHI` **비트 동일**.
8. **env 게이트가 자식 워커에 안 간다** → **거짓.** `run_w3.py:465-472` 가 **워커 프로세스 안에서**
   4개 키를 전부 `'0'` 으로 덮어쓴 뒤 팔 값을 넣고, `:485-489` finally 가 다시 `'0'` 으로 되돌린다.
   `initializer.py:24-30` `_rx2_gate` 는 **호출 시점**에 읽는다. 카운터는 잡 단위로
   install/undo 되므로 **팔별로** 기록된다. 실측: OFF 팔 24행 전부 0, `KONLY-FITPHI` 6/6 개입
   (원자 12–16, 게이트 지점 `coarse_to_fine.py:173`).
9. **λ 조작검사가 의도 dict 를 읽는다(S13 결함)** → **거짓.** `run_w3.py:275-298` 이
   **살아 있는 객체**에서 되읽고 앙상블 멤버별 λ 까지 남긴다(`members` 프로퍼티는 실존:
   `sequential_update.py:1483`). 스모크 불일치 0, `0.001/0.0001/0.0` 팔별 확인.
10. **F-1(λ 폴백 함정)이 과장** → **참으로 확인.** `tmp/rx2_init_2026-08-05/run_rx2b.py:61-64`
    `BASE_CFG` 에 `lambda_l1` **없음**, `tmp/{K6,S1,S3,S4}_*/run_*.py` 에 `lambda_l1` **grep 0건**
    ⟹ 전부 당시 기본값 상속. W3 는 전 팔 λ 명시 ✅.
11. **W3 의 `BASE_CFG` 가 frozen 셀을 재현하지 않는다** → **거짓.** 추가된 6키
    (`lr=0.01, lambda_smooth=1e-4, tracking_steps=10, tracking_mode='w_phi', tracking_lr=0.005,
    device='cpu'`)는 **전부 `run.py:26-68` `DEFAULT_CONFIG` 와 동일** = 무해한 명시화.
    `ORA` 는 `run_rx2b.py:174-176` `__oracle_k__` 와 **정확히 일치**.
12. **혼잡 32 = 두 코퍼스 풀링이 이질적** → **미지지.** 두 코퍼스의 h5 `config` 속성이
    `Hs=1.2 · Tp=7.0 · gamma=3.3 · direction_spread_deg=30 · mesh_dx/dy · decorrelate_stream=True ·
    random_amplitudes/frequencies=True · mesh_verdict=ADEQUATE` 로 **동일**하고, 다른 것은
    시드별 난수 추첨(`lambda_min/peak`)뿐. 둘 다 **생성기 결함 수정(08-07) 이후** 산출물이고
    ep_seed 도 겹치지 않는다(20–23 vs 48–51). 48 파일 **전부 존재**.
13. **시드 충돌** → **없음.** 저장소 전체 `*.json`/`*.jsonl` 에서 `"(seed|est_seed|filter_seed": 159[234]`
    **0건**(W3 자체 제외).
14. **동결 통계량이 틀렸다** → **전건 재현.** 23/32 → **0.010031**, 22/32 → 0.025051,
    power(p=0.80) = **0.9102**, MDE(power .80) = **0.766**, 희소 13/16 → **0.010635**. 전부 PREREG 기재와 일치.
15. **`git_head` 로는 더러운 트리를 못 잡는다(S13b 처방)** → **이미 이행됨.** `code_md5` 가
    `wavesplat/**` + `wavesim/**` + `configs/*.yaml` + `run_w3.py` + `PREREG.md` **87파일** 지문이고
    (`run_w3.py:141-155`), 스모크 전 행 단일 digest `696fa102bb05dd23f04ea18c8421ee12` ✅.
    **런 도중 러너를 고치면 G-7 이 즉시 잡는다** — 의도된 설계. (구멍은 W-6 의 분석기뿐.)

---

## 자진신고 6건에 대한 판정

| # | 자진신고 | 감사 판정 |
|---|---|---|
| 1 | 배포 팔 개입 지점 부재, 처방 = `KONLY` 짝 | **처방 불충분 — 단 이유가 다르다.** `KONLY` 는 배포 구성이 아니므로 **배포 질문에 답하지 않는다**(자진신고가 맞다). 그러나 더 중요한 것은 **애초에 혼잡 배포 경로에 고칠 것이 없다**는 사실(B-5): 그 경로는 이미 LS 적합 위상을 설치한다. ⟹ 처방은 "`KONLY` 추가" 가 아니라 **"§9.1 의 배포 후보를 0개로 정정"** 이다. |
| 2 | `BASE_CFG` λ 부재 | **검증 완료 · 처방 충분.** 전 팔 λ 명시 + G-3 객체 되읽기. 공격 실패 #10. |
| 3 | PRIMARY 가 4연속 널을 직접 안 묻는다 → S-2 | **처방 방향은 옳으나 미흡.** S-2 가 Holm-4 에 들어가면서 **실효 문턱이 24/32 로 올라가는데 문서는 23/32**(B-3), 그리고 **step-down 부재로 보정 자체가 부정확**(B-2). 게다가 §1.1 의 "4연속" 프레임이 **인용금지 수치 위에 있다**(B-1). |
| 4 | PRIMARY 대조 퇴화 가능 | **치명적이지 않다 — 단 A-3 의 자기평가는 부정확하다.** 퇴화 대조를 이기는 것이 사는 것: `ORA-K@L3`(0.2534) vs `ORA-K-NOREG`(0.1452) 는 **λ 하나만 다른 짝**이므로 **"λ=1e-3 이 오라클 팔에 입힌 손상의 크기"** 라는 **인과 주장**은 유효하고, 처치 팔은 `ZERO`(0.2401)를 **이긴다**. 사지 **못하는** 것은 "λ=0 이 좋다"·"배포 개선" 이며 A-3 가 그걸 막는다 ✅. 남는 문제는 (i) A-3 가 §4.3 의 **금지를 허용으로 바꾼 사후 완화**라는 점(W-1), (ii) **SECONDARY 에는 그 규율이 아예 구현돼 있지 않다**는 점(B-4). ⟹ 스모크에서 관측된 λ 사다리 `L3 0.2534 → L4 0.1730 → NOREG 0.1452 → CVX 0.0020` 이 **용량-반응**이라 퇴화 대조 하나로 해석이 무너지지는 않는다. |
| 5 | GPU OOM 1/49 | **A-1 의 완화는 수용 가능하나 그 안전밸브가 죽어 있다**(B-7). 진짜 위험은 OOM 이 아니라 **GPU 경로에 재현 프로브가 0개**라는 것. |
| 6 | `ORA-K` 3요인 묶음 · 혼잡 32 풀링 · CVX 무가중·무추적 | **묶음 라벨 ✅**(단 15comp 에서 **4요인**이 된다 — W-3). **풀링 ✅**(공격 실패 #12). **CVX 비대칭 신고는 불완전** — **k 동결 요인 미신고**(B-6) 이고 **무추적의 부호 단정이 저장소 기록과 반대**(W-4). |

---

## 결론

**`GO-WITH-FIX`.**
설계의 계측(λ 되읽기 · 원자 단위 FITPHI 카운터 · 누출 프로브 · 코드 지문 · 집계 순서 · DCC 하드 종료)은
**이 저장소가 지금까지 만든 것 중 가장 단단한 축**에 든다 — 위 15건의 공격이 전부 코드·원자료에서 막혔다.
깨지는 곳은 **계측이 아니라 (i) 통계 보정, (ii) 동기 문장, (iii) 상한 팔의 귀속**이다.

**런 전 처리 순서 (총 소요 ≈ 15분 + 프로브 5분)**
1. `AMENDMENT_2.md` 작성·동결 — **B-1**(K6 금지수치) · **B-3**(24/32) · **B-5**(배포 전제·후보 0개) ·
   **B-6(b)**(§8 재기술) · **W-1**(A-3 자기평가) · **W-4**(무추적 부호) · **W-11**(계수).
   `results.jsonl` **부재 시점**임을 명시하고 md5 를 `FROZEN_md5.txt` 에 추가.
2. `analyze_w3.py` 수정 — **B-2**(Holm step-down 3줄) · **B-4**(SECONDARY 퇴화 꼬리표 3줄) ·
   **W-2**(경계) · **W-5**(유한성) · **W-8**(예외) · **W-9**(라우트 인쇄). 수정 후 md5 를
   `FROZEN_md5.txt` 에 기록(**W-6**) — **결과가 생기기 전에.**
3. `run_w3.py` 수정 — **B-6(a)** k 표류 필드 · **B-7** `SHIP-DUP` 팔 + G-5b · **W-7** BLAS 고정 1줄.
   (지문이 바뀌므로 **반드시 런 개시 전에.**)
4. 스모크 보강 — **W-3** 15comp 혼잡 1 에피소드(오라클 2팔 × 3시드, ≈3분). G-4 OFF 팔 청결 재확인.
5. 그 다음 `AMENDMENT_1 A-2` 의 2위상 실행.

⛔**어떤 문턱·팔·표본·통계량·방향도 바꾸지 않는다.** 위 7건은 전부 **판정을 더 엄격하게** 만들거나
**문서를 사실에 맞추는** 방향이다.
