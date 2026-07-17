# E-42 Design D1 — Amortized JOINT-hypothesis posterior + tempered exact evidence + calibrated BMA
Designer 1 / 3 (독립 경쟁 설계). 2026-07-17. 대상 상금: cross-Tp crossing seas oracle − cur_grid = +0.0982 ± 0.0204 (n=32).

---

## 0. Hypothesis class — 한 문장 선언

**입력 = 기존 6ch dispersion-constrained beamform maps (mu_hat 기준 회전 정준화 + 2 스칼라) → 출력 = 구조화된 저차원 sea-system 파라미터의 JOINT 사후분포** — model space {1-system(δ1, a1)} ∪ {2-system(δ1, a1, δ2, a2, w)}에 대한 von-Mises/Gaussian mixture density (모델 사후확률 포함) — **학습 신호 = 참 ensemble 파라미터의 NLL (proper scoring rule; 시뮬 라벨, task loss 아님)**; 추론 시 frozen hypothesis lattice 위에서 이 amortized posterior를 **tempered exact type-II evidence(70s 시간축, e41 엔진)와 stacked log-linear 결합으로 재가중**하고, 최종 출력은 **posterior-mixture γ**(각 hypothesis를 e41 lobe_grid 규약으로 렌더 후 가중합) → 불변의 `solve_with_grid`.

즉: `maps → q(model, u | x)` (NLL 학습) `→ w_j ∝ exp(β1·logE_j + β2·log q_j + β0·1[M2])` (β는 held-out 라벨 시뮬에서 교정, 동결) `→ γ_out = Σ w_j γ(u_j)`.

incumbent(beamform map → CNN → 고정뱅크 γ, task loss)와의 차이는 **출력 파라미터화**(γ맵의 조건부 평균이 아니라 joint 저차원 파라미터의 분포), **학습 신호**(solver 관통 task loss가 아니라 파라미터 NLL), **추론 구조**(비-amortized exact likelihood 항 결합) 세 축 모두다. E-28 붕괴(같은 γ-over-fixed-bank 류의 조건화 변주)가 아니다 — γ는 최종 렌더링 표적일 뿐 학습 대상이 아니다.

**Litcheck 주의 (상시지시)**: 이 클래스는 방법론적으로 simulation-based inference의 **neural posterior estimation(NPE; Papamakarios&Murray 2016, Cranmer 2020 리뷰)** + Bayesian model averaging이다. 방법 novelty 주장 금지. 기여 주장은 "E-41 분해가 지목한 5-파라미터 공간에 대한 진단-유도 적용 + PERM-안전 교정 스택"으로 한정.

---

## 1. 진단 — incumbent가 구조적으로 못 하는 것 (사실 근거)

1. **결합 정보는 입력에 있다.** `beamform_power`(d3_sa_nomp.py:320)는 각 k-원자에 ω(|k|)를 분산관계로 묶는다 → 2-system 바다는 맵 위에 (ring1,d1)·(ring2,d2) 두 blob으로 **결합된 채** 나타난다. e37 진단이 정합: 맵/net의 lobe-2 radial 질량 포착 0.88 수준인데 "(방향2×ring2) 결합 집중 실패".
2. **incumbent = 정확히 all-marginal smear 수준** (E-41: m2 − cur_grid = −0.0049 ± 0.0237). 데이터 추가 무효(E-39: in-distribution에서 갭 지속, crossing +0.059). 원인 진단: **task loss(≈ 제곱 위험)의 최소화자는 조건부 평균 γ**. 입력이 모호할 때 다봉 사후의 평균 = smear. 이 프로젝트가 반복 관측한 mean-collapse(raw 점회귀, pixel-MSE, v2 라벨회귀)와 같은 기제이며, **γ맵-출력 + task-loss 클래스 안에서는 어떤 조건화/데이터로도 못 벗어난다** — 그 클래스의 베이즈 최적해 자체가 smear이므로. E-39가 "구조 실패"라 부른 것의 구체적 이름이 이것이라는 게 본 설계의 걸이가설.
3. **상금의 자리 = joint 커밋**: E-41에서 sum(MAIN)=+0.048 << oracle−m2=+0.103. d2 MAIN −0.005인데 TOTAL +0.044 = 방향2는 **다른 파라미터가 맞을 때만** 값한다(어느 ring에 어느 방향이 붙는가의 assignment 구조). mixture-of-joint-hypotheses는 각 성분이 일관된 (d1,ring2,d2,w) 할당이므로 **성분 내부에 상호작용이 보존**된다. m2는 정확히 "prior에 대한 mixture"이고, 사후가 참에 집중할수록 oracle로 연속 이동한다 — 이 클래스의 두 극한이 m2(포기)와 oracle(완전 커밋)이다.
4. **시간축 likelihood는 맵이 못 푸는 케이스를 푼다**: E-41 detector에서 evidence fit이 mu_hat의 wrong-lobe lock(104.9/108.4/140.0/154.4°)을 구제(paired +26.8±11.0°, 13/24), ring은 3–7%로 해결. 단 비-tempered ML-II는 speckle에 과신(corr(d2_err,ΔlogE)=+0.419; negatives 24 중 23이 w=0.06=grid 바닥의 nuisance lobe). → evidence는 **교정된 가중치의 한 항**으로만 쓴다, 판정 threshold로 쓰지 않는다.

---

## 2. 설계 상세

### 2.1 JPN (Joint Posterior Net)
- **입력**: `maps6(P, z, kp)` 6ch를 mu_hat 기준 −mu_hat 회전(정준화; `rotate_map` 재사용) + 스칼라 2개(log10(σ0/std z), log10 kp_hat). 정준화로 표본효율 확보(회전 등변성 학습에 표본 안 태움). wrong-lobe lock 케이스는 δ1이 커지는 것으로 자연 표현.
- **출력**: p(M2|x) 로짓 1개 + M1용 mixture K1=4 {vM(δ1)×N(log a1)} + M2용 mixture K2=12 {vM(δ1)×N(log a1)×vM(δ2)×N(log a2)×N(logit w)} (대각). δ = mu_hat 기준 상대각. 백본은 PriorNet conv trunk 재사용 + GAP + MLP heads (~1.5M params).
- **라벨**: 시뮬 생성기에서 공짜 — (is2, d1, d2, k1/kp_hat, k2/kp_hat, r2). a-라벨을 **kp_hat 상대**로 정의(렌더 규약과 일치; kp_hat 편향 교정까지 학습).
- **손실 = NLL**: −log[p(M|x)·q_M(u_true|x)]. Proper scoring rule → 최소화자 = 생성분포 하 참 사후 → **in-distribution 자동 교정**(체계적 confidently-wrong은 위험 함수상 벌점) — PERM 대응의 1차 방어선.
- **학습 데이터**: 자체 생성기 = e37 cross-Tp family를 약간 넓힌 것(Tp2/Tp1~U(0.70,1.45), offset U(50,165)°, r2~U(0.15,0.5)) + **unimodal fraction 0.40**(negatives 없이 학습하면 amortized 수준에서 ΔlogE의 nuisance-lobe 실패를 재생산한다). 40k 샘플, 시드는 모든 eval/교정 시드와 분리(결정적 시드, E-14b 교훈). GT 필드 렌더/solve 불필요(라벨=파라미터) → 샘플당 ~0.3s, 40k ≈ 3–4h(병렬 시 ~1.5h).
- **학습**: 30k step, bs 64, AdamW, cosine, val-NLL early stop, **seed 3개**(deployed q = 3-시드 사후 평균; incumbent 관례와 동일). "충분히 학습" 상시지시 충족 — E-28의 step-800 조기수렴은 solver-관통 task loss + 2.5–8k 샘플의 성질이지, 40k 샘플 밀집 NLL의 성질이 아니라는 게 예상이며, 아니면 val 곡선으로 드러난다.

### 2.2 Hypothesis lattice + tempered evidence + stacking
- **Lattice(동결)**: e41 detector의 grid를 재사용하되 **w 바닥 0.15**(negatives 23/24가 w=0.06 바닥에 앉은 nuisance 채널 봉쇄; 참 r2는 0.2–0.5), a2∈[0.5,2.0]. 후보 = q 상위 192 lattice 노드 ∪ **m2 prior-quadrature 백스톱 노드(항상 포함)** ∪ M1 lattice. 결정적(샘플링 없음).
- **evidence**: e41 `EvidenceEngine` 그대로(smult=3, var(z) 스케일 — solver와 동일 규약). 후보 ~300개/sea, batched Cholesky 400×400 → ~5s/sea.
- **가중치**: log w_j = β1·logE_j + β2·log q_j + β0·1[j∈M2] (+ log π_j 상수). (β0,β1,β2)는 **held-out 교정 세트 64 seas**(cross-Tp 32 + unimodal 32, 시드 분리)에서 참 파라미터 NLL 최소화로 적합 후 **동결**. β1은 fractional-likelihood temperature 역할 — ML-II 과신(오지정 하 evidence 과신)의 표준 처방. β→0 극한에서 출력 = m2 ≈ incumbent(−0.005) = **in-span do-no-harm**: 이 설계는 실패하면 incumbent로 퇴화하지, PERM처럼 −0.077을 내지 않는다(단 §4-FM1 참조 — 그래서 이길지는 별문제).
- **렌더**: γ(u) = e41 `lobe_grid` 규약. **커밋 축은 (d1, d2, ring2≡a2, w)만** — E-41에서 gam/sp MAIN ≈ 0이므로 spread/γ_J는 quadrature smear 유지(X 아레나에선 상수라 무해, Y 일반화 시 필수). 최종 γ_out = Σ w_j γ(u_j) → `solve_with_grid`. **threshold 없음** — 검출/판정 이진화가 어디에도 없다(unqualified-detector 문제 원천 회피).

### 2.3 이 클래스가 incumbent가 못 하는 무엇을 표현하는가 (요구사항 응답)
| | incumbent | D1 |
|---|---|---|
| 출력 | γ 51×51 조건부 평균 (베이즈 최적 = smear) | joint 저차원 파라미터 **분포** (다봉 표현 가능, 커밋·포기 연속 조절) |
| 학습 신호 | task loss (제곱 위험 → mean-collapse) | 파라미터 NLL (proper scoring rule → 사후 자체를 표적) |
| 상호작용 | 원자별 독립 에너지 = marginal product 구조로 붕괴 | hypothesis 내부에 (d1,ring2,d2,w) 결합 보존; mixture 극한이 m2↔oracle |
| 시간축 | 맵의 분산-구속으로 간접 | + exact 70s likelihood 항(맵 모호 시 wrong-lobe 구제, 사실 7) |
| 과신 방어 | 해당 없음(커밋 자체를 안 함) | proper score + temperature + 교정 스택 + no-threshold BMA |

E-28 재현 아님의 증명 책임: 공유하는 것은 maps6/bank/solve_with_grid **경계 인터페이스**뿐. 학습되는 객체(사후밀도)와 위험 함수(NLL)와 추론 연산(evidence 재가중)이 전부 다르다. informativeness 스칼라 조건화(E-28에서 ~0)와 p(M2|x)는 역할이 다름 — 전자는 γ의 변조 입력, 후자는 model posterior로서 **라벨에 의해 직접 지도**되고 교정 세트에서 검증된다.

---

## 3. 사전등록 (동결; 이 문서가 등록본)

**아레나 / 표본**
- 주 평가: `build_sea_crosstp(si)`, **si = 100..195 (n=96, fresh — si 0..31은 발견 세트라 오염)**. 비교 가능성용으로 si=0..31도 병기 보고(라벨: discovery set).
- 안전 평가: `build_sea_bimodal(2000+j, unimodal=True)`, n=32 (detector 교정의 3000+와 분리).
- 일반화(secondary): `build_sea_openbi(si)` si=100..163, n=64.
- 교정 세트(β, coverage): cross-Tp si=500..531은 detector 교정에 사용됐으므로 **si=600..631 + unimodal 3100..3131** 사용.
- 학습/교정/평가 시드 3-way 분리 목록을 코드 상수로 동결.

**Arms**: `cur_grid`, `cur_dep`(상시지시: 두 cur 다), `m2`(정직 smear 대조), `ens1`, **`D1`(본 설계)**, `D1_noev`(β1=0 ablation — evidence 기여 분리), `D1_noq`(β2=0 — net 기여 분리), `perm_ctrl`(D1에서 각 hypothesis의 d2만 prior 재추첨, 가중치 유지 — 질량-배치 아티팩트 대조), `oracle`(상한 앵커).

**통계**: paired mean ± SE, t, 95% CI, win-rate. 예상 SE: per-sea diff SD ≈ 0.134 (m2−cur에서 역산) → n=96에서 SE ≈ 0.014, **MDE(CI 0 배제, 80% power) ≈ +0.039**. n=32의 MDE ≈ ±0.06이라 부족 — 그래서 n=96.

**판정 (전부 D1 − cur_grid AND D1 − cur_dep 기준)**
- **PASS**: 주 평가 mean ≥ +0.03, 95% CI 0 배제, win ≥ 60%, **AND** unimodal 안전: mean(D1−cur_grid) ≥ −0.01.
- **STRONG**: mean ≥ +0.05 (상금 +0.098의 절반 이상 회수).
- **DEAD (자기 선언 사망)**: 주 평가 mean < +0.02 (n=96), 또는 unimodal harm < −0.02, 또는 아래 G-JOINT 실패.
- 애매 구간(mean ∈ [0.02,0.05], CI 걸침): **1회 한정** n=160으로 확장(사전등록된 escalation; 그 외 peeking 없음).

**기제 게이트 (조기 kill 신호 — 풀 평가 전 판정)**
- **G-JOINT**: val에서 joint-MDN NLL이 product-of-marginals-MDN(같은 용량, 헤드만 인수분해) NLL보다 **≥ 0.10 nat** 낮아야 함. 실패 = 맵에 marginal 초과 결합 정보 없음 = 본 설계의 핵심 전제 사망 → 학습 중단 선언. (이게 E-39 "구조 실패" 해석의 직접 검증이기도 함 — 실패해도 논지 가치 있음.)
- **G-CAL**: 교정 홀드아웃에서 (d1,d2) 50%/90% credible set coverage가 명목 ±10pt 이내.
- **G-PERM**: 상위-가중 hypothesis의 d2 오차와 그 가중치의 corr ≤ +0.20 (detector의 +0.419 재생산 금지).
- **G-EVID**: 교정에서 β1 > 0 (아니면 evidence 항 제거하고 D1_noev로 진행 — 그것도 결과).
- 기제 상관: Spearman(per-sea gain, 2-system 커밋 질량) > 0.

**컴퓨트**: 데이터 3–4h + 학습 3×45min + 교정 10min + 평가 (96+32+64) seas × ~8s ≈ 30min. 4080S 1대, 총 < 10h, 학습 자체 < 3h. 충족.

---

## 4. 적대적 자기비판 — 실패 양식 (사전 선언, 순위)

- **FM1 (최유력): 교정된 포기 = 이득 0.** 400점/70s에서 d2·(ring2×d2) 결합의 식별력이 진짜로 약하면(detector의 accepted-median d2 40.9°가 경고), 정직한 사후는 평평하고 BMA는 m2로 퇴화 → D1 ≈ cur ≈ m2, PASS 실패. 이 설계는 **지면 안전하게 지도록** 만들어졌고, 그 대가로 "교정을 지키면서 이길 수 있는가"가 전부 식별력에 걸려 있다. 조기 신호: G-JOINT nat 차이, 교정 세트 커밋 질량 분포. 이 경우 산출물은 "joint 정보가 맵+likelihood에 없다/약하다"의 정량 — E-39의 구조 실패를 정보 한계로 좁히는 음성 결과.
- **FM2: slice-conditional 미교정.** NLL 교정은 학습 생성분포 평균에서의 교정이다. 아레나 X는 Tp1=7 슬라이스 — 슬라이스/조건(고속 U, 낮은 r2)별로는 국소 과신이 남아 PERM-형 손실이 국소 발생 가능. 완화: 교정 세트를 아레나 family 자체에서 추출, per-U-bin coverage 보고. 잔여 위험: β 2–3 스칼라를 64 seas로 적합 → 교정 자체의 표본 오차.
- **FM3: MDN mode collapse / 표현 실패.** vM mixture가 side 대칭(d2 = d1±offset)의 이봉을 못 잡으면 사후가 뭉개짐. 완화: K2=12 + 3-시드 평균 + 사전등록 fallback(G-JOINT 판정 전 val-NLL에서 MDN이 product 대비 못 이기면 **kernel-smoothed joint-lattice softmax 헤드로 1회 교체** — 39k-노드 CE, 다봉 표현이 공짜).
- **FM4: 기대치 산수의 벽.** MAIN d1 +0.050은 **exact** d1 기준이다. 현실 사후(d1 ~20°, ring 3–7%, d2 불량)로는 회수분이 각 항의 일부 — 정직한 기대는 +0.03~+0.06이며 PASS 경계에 걸친다. wrong-lobe 구제(치명 케이스 = cur의 최악 seas일 개연성)가 상호작용 항을 얼마나 여는지가 승부처. 이것이 n=96을 요구한 이유다.
- FM5: 렌더 family 오지정(spread=30·γ_J=3.3 상수)은 X에선 구성적 참, Y/실측 일반화 주장 시엔 약점 — Y는 secondary로만.

---

## 5. (B) 회수 실험(`e41_blind_recovery.py`) 의견 — 오케스트레이터 실행용

**Arms 추천**
1. `fit1` — 유지. threshold 무관 + "evidence의 d1/a1만으로 얼마 회수?"를 격리. 단 **대조는 cur만이 아니라 `ens1`과 `m2`에 대해서도**: ens1−cur = −0.062이므로 fit1이 cur에 지더라도 ens1 대비 +가 크면 "evidence의 d1·ring 값"은 실증됨(가치는 lobe-구조가 아니라 1-lobe fit 개선에 있음이 판명).
2. `fit2_hard` @ frozen TAU=14.64 — 사전등록이므로 돌리되 **expected-fail 라벨**. TAU가 두 outlier(w=0.06/0.12 nuisance lobe)로 정해진 게 이미 알려져 있어 TPR 33%로 거의 아무것도 안 잡는다. post-hoc TAU=3.5 arm은 절대 판정에 쓰지 말 것(탐색 라벨만).
3. `fit_bma` — 유지하되 **주의: 현 구현의 family-내 가중치는 raw exp(logE) ≈ argmax**(ΔlogE 수십 단위) = 사실상 point 커밋 = PERM 노출. 이대로 지면 "회수 불가"는 **과신-가중치가 만든 위음성**이다.
4. **★추가 `fit_bma_temp` — 내가 PRIMARY로 추천**: family-내·family-간 가중 log w ∝ β·logE, β 하나를 **fresh negatives에서** 교정(예: unimodal 세트에서 E[p2] ≈ 사전 odds 또는 coverage 기준) 후 동결. threshold 없음 + 과신 방어. 이거 없이 돌리는 raw BMA 단독 실험은 E-37의 실수(자기 검출기의 해악 측정)를 가중치 공간에서 반복할 위험이 가장 크다.
5. **★추가 2-lobe family에 w 바닥 0.15**: negatives 23/24가 w=0.06 바닥에 앉음 — nuisance-speckle 채널의 구조적 봉쇄. 단 **family를 바꾸면 TAU/β는 fresh negatives(예: 3100+j, n=24)로 재도출** — 기존 24 negatives를 본 뒤의 재사용은 post-hoc 오염.
6. **컨트롤**: `m2`(정직 smear — 어떤 fit arm이든 cur를 이기고 m2를 못 이기면 회수가 아니라 smear 가치), `ens1`(universal-radial 특화 가치 분리), `perm_ctrl`(fitted d2를 prior 재추첨, 가중 유지 — 질량-배치 아티팩트 검출), unimodal negatives n≥24에 **같은 arms 전부**(harm 사전등록: mean ≥ −0.01 vs cur). 그리고 stage-2의 greedy lobe1-freeze 완화: stage-1 상위 2 후보 각각에서 stage-2 시드(2× 비용, wrong-lobe seas의 위음성 방지).

**타겟/문턱**: "+0.03, CI 0 배제, n≥32"의 **문턱은 유지하되 n=32는 부족** — blind arm의 paired SE ≈ 0.031(e37 실측) → MDE ≈ +0.06. **n=64 최소, 가능하면 96** (SE ≈ 0.022/0.018). 그리고 E-41이 말하는 기제(d1 이득 = tail rescue 13/24)상 pooled가 +0.01~0.03에 그칠 수 있으므로 **subgroup을 지금 사전등록**: mu_err > 60°(wrong-lobe lock) vs ≤ 60° 분리 대조. 판정 티어: FULL(pooled ≥ +0.03 CI>0) / RESCUE-ONLY(pooled ≥ 0 & lock-subgroup CI>0 & harm ≥ −0.01) / FAIL. 평가 seas는 si=0..31(발견 세트, 비교 가능성) **+ fresh si=100..163을 confirmatory로** — 판정은 fresh에서.

**위양성 최대 경로**: (a) 발견 세트 재사용 + family에 아레나 상수(Tp1=7 radial, spread 30)가 박혀 있어 regime-특화 이득을 "2-lobe 회수"로 오독 — fresh si + m2/ens1/perm_ctrl 대조로 차단. (b) family 변경 후 옛 negatives로 TAU/β 재도출(post-hoc) — fresh negatives 강제.
**위음성 최대 경로**: (a) 비-tempered BMA의 사실상-argmax 커밋이 PERM 해악으로 이득을 상쇄 — fit_bma_temp가 방어. (b) greedy lobe1-freeze가 wrong-lobe seas에서 2-lobe family의 swap을 봉쇄 — top-2 시딩으로 방어. (c) n=32 검정력.

**내 설계와의 관계**: 회수 실험의 `fit_bma_temp`는 내 D1에서 q-net을 뺀 것(D1_noq)과 거의 동일하다. 회수 실험이 RESCUE-ONLY 이상이면 D1의 evidence 항 가치가 선검증되는 셈이고, FAIL이면 내 설계의 부담이 전부 JPN(G-JOINT)으로 넘어온다 — 어느 쪽이든 D1 사전등록은 그대로 유효.
