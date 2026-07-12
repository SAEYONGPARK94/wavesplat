# 방법 절 초안 — 미분가능 물리-사후 관통 task-loss 하이퍼 학습 (B-2)

> 작성 2026-07-12 (자율, CPU 전용). 방법 후보 B-2
> (`docs/ADVISOR_MEMO_thesis_type_2026-07-12.md` §2)의 논문-이식 정식화.
> 프레임: "small, certified learning" (`PAPER_CH_C_draft_2026-07-12.md`).
> 인용 자리는 `[CITE:...]`로 표기 — litcheck는 메인 에이전트가 수행 중.
> 수치는 전부 출처 JSON 명기, `scripts/`에서 재도출 가능.

---

## M.0 요지 (한 문단)

분산-구속 스펙트럴 은행 위의 사후평균(MMSE) 추정기는 두 하이퍼—진폭
스펙트럼 prior $\gamma(\mathbf{k})$와 잡음바닥 $\sigma$—가 성능을 지배한다.
표준 처방인 우도·창-안 잔차 최소화는 **실패 창에서 체계적으로 오선택**한다
(H5의 n-독약). 우리는 이 두 하이퍼를 **과제 손실**(조우/전장 예측 오차)로,
**미분가능한 Woodbury 사후평균을 관통**해 역전파하는 amortizer($\gamma$-CNN +
$s^2$-head)로 학습한다. 우도를 쓰지 않는다는 것이 이 방법의 정의적 선택이며,
그 동기는 우도가 틀린다는 판정 자체다. 학습은 오지정-주입 합성 분포에서만
이루어지고 실측에는 **zero-shot** 배포된다.

---

## M.1 문제 설정 — 사후평균을 지배하는 하이퍼, 그리고 우도의 실패

### M.1.1 분산-구속 스펙트럴 은행 사후분포

관측은 이동 플랫폼 LiDAR가 전방-부채꼴에서 취득한 희소 표고 표본
$z_i = \eta(\mathbf{x}_i, t_i) + \varepsilon_i$, $i=1{:}n$이다. 파랑장을
분산관계로 구속된 진행파 은행으로 전개한다:

$$
\eta(\mathbf{x},t)=\sum_{b=1}^{B}\big[a_b\cos\psi_b(\mathbf{x},t)+c_b\sin\psi_b(\mathbf{x},t)\big],\qquad
\psi_b(\mathbf{x},t)=\mathbf{k}_b\!\cdot\!\mathbf{x}-\omega_b\,t.
\tag{1}
$$

은행의 파수 $\mathbf{k}_b$는 추정 첨두파수 $k_p$로 정규화된 $51\times51$
환형 격자의 셀이고, 각 셀의 각주파수는 **분산관계 경성 구속**으로 묶인다:

$$
\omega_b=\sqrt{g\,|\mathbf{k}_b|\tanh(|\mathbf{k}_b|\,d)}.
\tag{2}
$$

식 (2)는 후보 원자를 물리적으로 실현 가능한 (파수, 주파수) 쌍에만 국한시켜
자유도를 대폭 줄인다("분산-구속"). 계수 $\{a_b,c_b\}$에 대각 가우시안
prior를 부과한다: $a_b,c_b\sim\mathcal{N}(0,\gamma_b)$,
$\boldsymbol{\Gamma}=\mathrm{diag}(\gamma_1,\dots,\gamma_B,\gamma_1,\dots,\gamma_B)$.
prior 질량은 **관측 분산으로 구속**된다 —
$\sum_b\gamma_b = \mathrm{Var}(z)$, $\boldsymbol{\gamma}$는 은행 위 정규화된
분포 $\gamma_b\!\ge\!0,\ \sum_b\gamma_b/\mathrm{Var}(z)=1$ ("분산-구속").
관측 잡음은 $\varepsilon_i\sim\mathcal{N}(0,\sigma^2)$.

설계행렬 $\boldsymbol{\Phi}\in\mathbb{R}^{n\times 2B}$,
$\boldsymbol{\Phi}_{i,\cdot}=[\cos\psi_b(\mathbf{x}_i,t_i),\ \sin\psi_b(\mathbf{x}_i,t_i)]_b$
로 두면, 임의의 질의점 집합 $Q$에서의 사후평균(MMSE)은 함수공간(Woodbury/GP
쌍대) 형식으로

$$
\hat{\boldsymbol{\eta}}(Q)=\boldsymbol{\Phi}_Q\,\boldsymbol{\Gamma}\,\boldsymbol{\Phi}^{\!\top}
\big(\underbrace{\boldsymbol{\Phi}\,\boldsymbol{\Gamma}\,\boldsymbol{\Phi}^{\!\top}+\sigma^2 \mathbf{I}_n}_{\displaystyle \mathbf{K}}\big)^{-1}\mathbf{z},
\tag{3}
$$

$n\times n$ 계 $\mathbf{K}$의 역만 요구하므로 은행 크기 $B$에 무관하게
관측 수 $n$으로 비용이 결정된다 (Woodbury 축약; $n\!\ll\!2B$). 사후분산도
동일 커널에서 닫힌형식으로 나온다 (§M.2.4, UQ). 식 (3)이
`d3_prior_net_eval.py::net_gamma_mmse`의 추론 경로이자
`d3_prior_net_taskft.py::task_loss_one`의 미분가능 학습 경로다 (동일 은행·동일
사상, `bound_wave` 옵션은 §M.2.5).

### M.1.2 하이퍼가 성능을 지배한다 — 그런데 우도가 그것을 틀리게 고른다

식 (3)의 품질은 사실상 **$(\boldsymbol{\gamma},\sigma)$ 두 하이퍼**로
결정된다: $\boldsymbol{\gamma}$는 어느 (파수) 원자에 에너지를 둘지,
$\sigma$는 관측을 얼마나 신뢰할지를 정한다. 표준 처방은 이 둘을 창 안
데이터에 대한 우도(또는 held-out $z$-잔차)로 고르는 것이다. 우리는 이것이
**실패 창에서 체계적으로 틀린다**는 것을 판정했다 (H5, n-독약 —
`docs/H5_EM_PATHOLOGY_AND_RECIPE_2026-07-11.md`,
`results/direction3/h5_em_pathology.json`):

- 관측이 많아질수록 held-out $z$-잔차 그리드는 잡음바닥을 **축소**($\sigma$
  배수 $3\!\to\!1$)하고 prior를 **평평화**($p_{\exp}\!\to\!0.5$, 고-$k$ 질량
  $0.29\!\to\!0.44$)한다. 창 안 적합은 좋아지지만, 이는 시간 외삽(조우예보)을
  전혀 감지하지 못한다. "창 안을 더 세게" $\neq$ "미래를 더 잘".
- 근본 원인은 대역-절단·off-shell 에너지 등 **모델 오지정**이다. 잔차-선택은
  이 오지정을 창 안 자유도로 흡수해버려, $\sigma$를 센서 잡음 수준으로
  과소평가한다. 올바른 값은 $\sigma^2\!\approx\!$ 모델 오차 수준
  (GP 오지정의 nugget-inflation 처방과 정합 [CITE: GP-misspecification /
  nugget inflation, arXiv 2312.09225 계열]).

**동기 요약**: 하이퍼는 성능을 지배하지만, 창-안 우도로는 실패 창에서 옳게
고를 수 없다. 그러므로 우리는 하이퍼를 **관측 우도가 아니라 과제 오차로**
정한다.

---

## M.2 방법 — 미분가능 사후를 관통하는 task-loss 하이퍼 amortizer

### M.2.1 하이퍼를 내는 amortizer

두 하이퍼를 장면(관측)에서 직접 예측하는 amortizer를 둔다. 입력은 관측을
분산관계로 초점맞춘 **빔형성 파워맵** — 관측을 두 다리로 분할해
($\mathbf{X}\!\in\!\mathbb{R}^{6\times51\times51}$: 다리1/다리2/최소 파워맵과
그 로그) 정규화 $k_p$ 은행 위에 얹은 것 (`net_gamma_mmse`의 입력 레시피와
동일).

**진폭 spectrum prior — $\gamma$-CNN.** $\log\gamma$ 맵을 내는 소형 U-Net
$g_\theta$ (6ch→48ch, 3-스케일 인코더·스킵, 셀당 $\log_{10}\gamma$ 출력;
`d3_prior_net_train.py::PriorNet`):

$$
\log_{10}\gamma_b=\big[g_\theta(\mathbf{X})\big]_b,\qquad
\gamma_b\leftarrow \frac{\gamma_b}{\sum_{b'}\gamma_{b'}}\,\mathrm{Var}(z).
\tag{4}
$$

**과제-보정 잡음바닥 — $s^2$-head.** 장면 특징(파워맵의 채널별 평균·표준편차)
을 풀링해 로그 잡음배수를 내는 MLP $h_\phi$
(`d3_l1_s2head_train.py::SmultHead`), 잡음바닥 $\sigma_0=\nu\,H_s$ (센서
잡음율 $\nu$·유의파고)에 곱한다:

$$
\log s=b+h_\phi\!\big(\mathrm{pool}(\mathbf{X})\big),\qquad
\sigma=s\cdot\sigma_0.
\tag{5}
$$

$s$의 초기 바이어스는 $b=\log 3$ (실측 무학습 레시피의 손맞춤 잡음배수).
스칼라 전역-$s$(v3, `taskft.py`의 `log_smult`)에서 **장면-적응 head**(L1-s2)로
확장한 것이며, head가 상수로 붕괴하면 아무것도 학습되지 않은 것으로 간주하는
정직-널 감사를 둔다 (`d3_l1_s2head_train.py`의 smult 분산 출력).

### M.2.2 과제 손실 — 우도가 아닌 예측 오차

훈련 표본마다 관측점 $P$·관측 $\mathbf{z}$와, **held-out 질의점** $Q$에서의
참 표고 $\boldsymbol{\eta}^\star(Q)$ (전장 픽셀 및/또는 조우예보 리드-질의)를
둔다. 손실은 정규화된 질의-필드 오차:

$$
\mathcal{L}_{\text{task}}=\frac{1}{|Q|}\,
\frac{\big\|\hat{\boldsymbol{\eta}}(Q)-\boldsymbol{\eta}^\star(Q)\big\|^2}{(H_s/4)^2},
\tag{6}
$$

$\hat{\boldsymbol{\eta}}(Q)$는 식 (3)의 사후평균, 정규화항 $(H_s/4)^2$는
해상상태(유의파고) 간 손실 스케일을 맞춘다. 결정적으로 $Q$는 **관측창 밖**
질의(전장 비관측 영역·미래 리드)를 포함하므로, 손실은 창-안 적합이 아니라
**배포 과제 자체**(조우예보·전장)를 직접 최적화한다 — §M.1.2가 진단한
우도-외삽 괴리를 설계상 우회한다. (`task_loss_one`이 정확히 식 (3)+(6)이다:
`pred = (design(Q)*gam) @ (Phi.T @ solve(K, z))`.)

### M.2.3 미분가능 관통과 역전파 경로

핵심은 식 (3)의 선형계 풀이가 **미분가능**이라는 점이다. $\mathbf{K}$가
$\boldsymbol{\gamma}$(식 4)와 $\sigma$(식 5)에 매끄럽게 의존하므로,
자동미분이 $\mathbf{K}^{-1}\mathbf{z}$ 풀이(torch `linalg.solve`)를 관통해
하이퍼로 기울기를 전달한다. 야코비는 표준 선형계 미분

$$
\frac{\partial \boldsymbol{\alpha}}{\partial \theta}
=-\mathbf{K}^{-1}\frac{\partial \mathbf{K}}{\partial \theta}\,\boldsymbol{\alpha},
\qquad \boldsymbol{\alpha}=\mathbf{K}^{-1}\mathbf{z},
\tag{7}
$$

를 따르며, 여기에

$$
\frac{\partial \mathbf{K}}{\partial \gamma_b}=\boldsymbol{\varphi}_b\boldsymbol{\varphi}_b^{\!\top},\qquad
\frac{\partial \mathbf{K}}{\partial \log\sigma}=2\sigma^2\mathbf{I}_n
\tag{8}
$$

($\boldsymbol{\varphi}_b$ = $b$번째 은행원자의 설계열)를 결합해
$\partial\mathcal{L}_{\text{task}}/\partial(\theta,\phi)$가 CNN·head
파라미터까지 흐른다. 실제 구현은 이 연쇄를 손으로 쓰지 않고 미분가능
`solve`에 위임한다 (AdamW, $\gamma$-CNN lr $3\!\times\!10^{-5}$·head lr
$10^{-3}$, grad-clip 1.0; `taskft.py`/`l1_s2head_train.py`). 요컨대 **물리
사후분포가 학습의 마지막 층으로 미분가능하게 삽입**되고, 학습되는 것은
장(場)이 아니라 그 사후를 형성하는 하이퍼다 (mean-collapse 회피 규율,
HANDOFF §4-7과 정합).

### M.2.4 불확실도 정량화 (동일 커널에서 무상)

동일 $\mathbf{K}$에서 질의점 사후분산이 닫힌형식으로 나온다:

$$
\mathrm{Var}\big[\hat\eta(\mathbf{q})\big]
=\boldsymbol{\phi}_q^{\!\top}\boldsymbol{\Gamma}\boldsymbol{\phi}_q
-\big(\boldsymbol{\phi}_q^{\!\top}\boldsymbol{\Gamma}\boldsymbol{\Phi}^{\!\top}\big)
\mathbf{K}^{-1}\big(\boldsymbol{\Phi}\boldsymbol{\Gamma}\boldsymbol{\phi}_q\big),
\tag{9}
$$

(`net_gamma_mmse(..., return_var=True)`). 학습된 prior가 UQ까지 함께
교정한다는 것은 별도 검증 대상이며 본 절의 범위 밖이나, 방법이 점추정과 UQ를
같은 물리 사후에서 일관되게 낸다는 구조적 이점만 명기한다.

### M.2.5 훈련 분포(오지정-주입)와 zero-shot 배포

**훈련 분포**는 합성 셔드(JONSWAP 다성분·광대역·잡음·전방-fan 관측 프로파일)
로, 배포에서 마주칠 오지정을 의도적으로 주입한다: 대역 절단, 관측 잡음,
그리고 2차 속박파(super-harmonic, $(2\mathbf{k},2\omega)$ off-shell) 후보를
자유파 prior에 종속·하향 결합한 은행(`bound_wave`; 학습된 자유 prior가 에너지
둔 곳에서만 발화 → 깨끗한 바다 무-오발화). $\gamma$-CNN은 오지정을 어디서
채울지를, $s^2$-head는 오지정 수준의 잡음바닥을 학습한다.

**배포는 zero-shot**: 실측 레코드에서 재훈련·미세조정 없이 식 (3)+(4)+(5)를
그대로 적용한다. 유일한 외부 입력은 첨두파수 힌트 $k_p$
(`characterize()`) — 빔형성 argmax $k_p$가 추정기의 가장
도플러/해류-취약 단계이기 때문(2026-07-05 해류-아크 판정)이며, 이는 은행
중심을 고정할 뿐 prior·잡음바닥은 여전히 학습이 낸다.

---

## M.3 차별화 — 인접 방법론과의 4축 구분

이 방법의 부품(amortized prior, 미분가능 추론, 결정-지향 학습)은 각각 선행이
있으며, 우리는 원리 신규성을 주장하지 않는다 (difference-first,
ADVISOR_MEMO §4). 구분은 네 축의 **결합·보장·도메인**에 있다.

| 축 | 본 방법 | 인접 선행 | 차이 |
|---|---|---|---|
| ① 물리 은행 | 분산-구속(식 2) 스펙트럴 은행 위 prior + 분산-구속 정규화 | amortized GP 하이퍼prior — 커널 하이퍼를 관측에서 예측 [CITE: AHGP, Liu et al. NeurIPS 2020] | 우리는 **파수-물리 은행**(분산관계·에너지 구속)이 학습축이고, 커널이 아니라 스펙트럼 형태를 amortize |
| ② 손실 | 비-우도 **task-loss**(식 6, 창-밖 질의) | 미분가능 필터/추론이 대개 우도·NLL·상태-MSE로 학습 [CITE: KalmanNet, Revach et al. 2022; deep kernel/marginal-likelihood, Wilson et al.] | **우도가 실패 창에서 틀린다는 판정(§M.1.2)이 손실 선택의 동기** — task-loss는 그 실패를 설계상 우회 |
| ③ 관측 연산자 | 이동 희소 **전방-fan** LiDAR (식 1의 $\mathbf{x}_i,t_i$가 궤적 위 부채꼴) | 결정-지향 학습(DFL/SPO)은 대개 정적·조밀 관측 [CITE: DFL Wilder et al. 2019; SPO Elmachtoub & Grigas 2022] | 관측 연산자가 **이동·희소·시야제한**이고, 그 위에서 사후를 관통 |
| ④ 검증 | 실측 **zero-shot** + **제어 전달**까지 | 파랑-학습 이득 상당수가 시뮬 또는 관측연산자 교체(레이더) 효과 [CITE: Ehlers et al.] | 재훈련 없는 실측 전이 + 제어 폐루프 가치 전달을 사전등록 게이트로 확증(§M.4) |

**비자명성 논거** (발표 규율, ADVISOR_MEMO §4): 부품 결합이 자명하다면 임의
결합이 작동해야 하나, 우리는 사전등록 판정 20여 건에서 대부분의 결합이
죽는다는 것을 수치로 보였다. 작동하는 결합을 찾고 왜 작동하는지를 판정
체계로 증명하는 것이 기여다.

> **인용 자리 목록** (litcheck = 메인): AHGP(amortized GP prior),
> KalmanNet/differentiable filtering, deep kernel/GP marginal-likelihood
> learning, DFL(Wilder), SPO(Elmachtoub-Grigas), Ehlers(radar+learning),
> GP misspecification/nugget-inflation, (보조) SBL(Tipping)·RELAX/NOMP·
> Woodbury/GP regression(Rasmussen-Williams).

---

## M.4 검증 요약 (사전등록 게이트, 군집-보정)

전 항목 창-짝지음·레코드 군집 부트스트랩으로 재보정
(`scripts/d3_cluster_bootstrap.py`,
`results/direction3/cluster_bootstrap_2026-07-12.json`). "창-수준 유의"는
레코드-군집 CI 생존을 뜻한다.

| 판정 | 지표 (learned vs 무학습 기준) | 통계 | 출처 JSON |
|---|---|---|---|
| **γ-prior (nowcast)** | 창-짝지음 상관 **+0.143** | $t_{\text{win}}$ 4.42·$t_{\text{rec}}$ 4.32·39/47창·8/8레코드·CI [+0.087,+0.206]·부호검정 $p{=}0.004$ | `cluster_bootstrap_2026-07-12.json` (A); `gate_promotion_eval.json`·`head_anatomy.json` |
| **잡음바닥+prior (조우, 60창)** | head−hand 전창 **+0.088** / 실패창 **+0.227** | $t_{\text{win}}$ 3.14/4.75·$t_{\text{rec}}$ 4.59/5.01·5/5레코드·CI [+0.052,+0.117]/[+0.142,+0.297] | `cluster_bootstrap_2026-07-12.json` (B) |
| 바닥 단독 기여 (분해) | fix3−hand 전창 +0.083 / 실패창 +0.211; 학습 head 수렴 $s\!\approx\!3.2$ ($\approx\!3\sigma_0$) | 2×2 가법 분해, 상호작용 ~0 | `cluster_bootstrap...json` (C)·`head_anatomy.json` (smult_head 3.18–3.23) |
| **시드 재현** | 훈련 시드 $\{0,1,2\}$ 전이 유지 (bw_s0/s1/s2) | A-5, 부호·크기 일관 | `gate_promotion_eval.json` (bw_s0/s1/s2 열) |
| **HOS 재현 (전장)** | 비선형 HOS M=3에서 pooled **+0.070**, 7/8셀 | 부호 일관 (Hs1.2 +0.075 / Hs3.0 +0.064) | `a1_field_hos.json` |
| 오지정 기울기 (3무대) | clean 선형 −0.015 / HOS +0.070 / 실측 +0.033 | 훈련-표적(오지정) 정합 = task-loss가 물리가 모르는 것만 채움 | `a1_field_hos.json`·`sa_multirecord_mmse(_net).json`·RUNLOG |
| **제어 전달 (D-20)** | 실측 리플레이 폐루프, learned-FF 잔여 < hand-FF **−0.086** | $t$ **−4.02**·$n{=}72$·3/3레코드 (LJ18g 실패창 −0.195 주역) | `d20_replay_eval.json` |
| 속도 | amortizer 111× (정확도 손실 없음); GPU 재추정 5 ms p50 | — | RUNLOG·D-20 (동등성 1.0000) |

**중심 문장**: 학습은 물리가 모르는 것을 채울 때만 이긴다 — 과제별로 다른
학습($\gamma$-net은 nowcast, 잡음바닥은 forecast)이, 창-수준 유의성으로,
가법적으로.

---

## M.5 한계

1. **아키텍처 단순 (CNN·MLP head).** amortizer는 소형 U-Net + 풀링 MLP다.
   이는 심사-방어를 위해 표현 축을 정면 검증했다: 점군-native
   perceiver-lite로 교체 시 동일 셔드·동일 task-loss에서 **열세**
   (point +0.472 vs CNN +0.620, Δ−0.148, $t$−3.90; `d3_m3_point_eval` /
   M3 판정, RUNLOG). 갭은 실측이 아니라 **합성 도메인부터** 발생 —
   맵+$k_p$-정규화 귀납편향이 유효 기여이며, attention이 등가 정보에서
   그것을 재발견하지 못했다. 따라서 "CNN은 최소-충분 인스턴스"이지 방법의
   본질이 아니다 (아키텍처-독립 주장). M3 민감도 표를 심사 자산으로 부록화.
2. **표본 규모.** 실측 = 공개 스테레오 전량 계열(YS17·AA15·AA14·LJ18g 등
   4사이트, 조우 60창 / nowcast 8레코드·47창). 선박 탑재 LiDAR 실물 부재로
   벤치마크는 고정-T 사양-모사(v2)로 보완. 대규모 실선 데이터는 미확보.
3. **잡음바닥 head의 상·하한 미탐.** 학습이 수렴한 $s\!\approx\!3.2\sigma_0$은
   과제-보정 상수에 가깝다 (H5의 손맞춤 $s{=}10$과 다른 무대). head의 장면
   민감도(대량 $n$·극단 오지정에서의 상·하한)와 nowcast에서의 부호 반전
   (nowcast는 기본 EM 우세, P4-①)은 별도 스윕이 필요.
4. **과제 특이성.** $\gamma$-net은 nowcast에서, 잡음바닥은 forecast에서
   유의하며(과제-상보), 통짜로 한 팔에 귀속하면 안 된다. 결합 배치의
   가법성은 60창 2×2 분해로만 확인됨 — 다른 과제-축 일반화는 열린 문제.
5. **선택 상금 미회수.** 창별 하이퍼 선택(비평가) 상금 +0.14는 실측
   550창+CNN으로도 회수 실패(비평가 v1/v2 FAIL) — 잔여 신호가 관측가능
   특징에 없거나 극소. 본 방법은 **전역-amortized** 하이퍼까지만 주장하고,
   창별-적응 선택은 원리적 한계로 기록.

---

## 부록: 절 구성·수식·인용 요약

- **절**: M.0 요지 / M.1 문제 설정(분산-구속 은행·우도 실패) / M.2 방법
  (amortizer·task-loss·미분가능 관통·UQ·훈련분포) / M.3 차별화 4축 표 /
  M.4 검증 요약표 / M.5 한계 5항.
- **수식 9개**: (1) 은행 전개, (2) 분산관계 구속, (3) Woodbury 사후평균,
  (4) $\gamma$-CNN prior+정규화, (5) $s^2$-head 잡음바닥, (6) task-loss,
  (7) 선형계 미분, (8) $\partial\mathbf{K}$, (9) 사후분산.
- **인용 자리 9종** (§M.3 목록): AHGP · KalmanNet/differentiable filtering ·
  deep kernel/GP marginal-likelihood · DFL(Wilder) · SPO(Elmachtoub-Grigas) ·
  Ehlers(radar+learning) · GP misspecification(nugget-inflation) ·
  (보조) SBL(Tipping) · RELAX/NOMP·Woodbury(Rasmussen-Williams).
</content>
