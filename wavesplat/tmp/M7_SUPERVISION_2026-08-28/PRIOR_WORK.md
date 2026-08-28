# M7 선행 확인 — 이 축(실측-유사 어려운 데이터 재학습 / 감독 변경)의 선행 기록 (2026-08-28)

> 임무 B (Fable). **런 0회 — 읽기 전용 감사.** 열람 범위 = 이 문서가 인용하는 파일 전부(경로 명기).
> 결론 먼저: **축은 형식적으로 죽지 않았으나, "실측 전이" 를 표적으로 삼는 형태는
> 측정된 사실 네 건이 정면으로 막는다(§6). 살아 있는 표적은 셋뿐이다(§7).**

---

## 0. 한 줄 요약

- **손실 변경**: 배포 아키텍처에서 **1회 시도, 사전등록 널**(`wp_e2e` 2026-07-25). 강한 가중은
  디노이저를 **능동적으로 파괴**했다. 단 ⚠**"장 손실 × 정합된 관측 기하" 조합은 미시도** — e2eft 는
  기하 수리(`mixed_retrain` D3) **이전**에 돌았고, e2eft 자신이 그 기하 갭을 널의 원인으로 지목했다.
- **어려운 데이터 학습**: 4회 시도 중 **데이터-축 1회만 PASS**(`mixed_retrain`, 배포됨).
  실측-유사 기하(fan) 재학습은 **합성 PRIMARY FAIL + 실측 무개선 = 축 폐쇄**(2026-07-24, 구 아키텍처).
- **HOS/비선형 학습**: **1회 있었다**(2026-07-03, `ds` vs `dslin` 대조) — HOS 혼합 학습이
  **HOS 평가에서조차 이득 0**(오히려 근소 열위). 배포 체크포인트는 **`nonlinear=False` 축자**.
- **체크포인트 계보**: `beamform_cond_diffusion_mixed_v1.pt`(2026-07-27 00:32, sha 실측 일치).
  원장의 *"평가와 같은 분포"* 문장은 **정확하다**(사전등록 자신이 "distribution-matching" 이라 선언).
- **비용**: 40k step **69–78.5 min**(RTX 4080 Super), fine-tune 4k step **≈3 min**. VRAM 미기록
  (state_dict 5.86 MB — 소형). ⚠지금은 `M1` 이 **14.7/16.4 GB** 점유 중(M4 RESULT §8 실측).
- ★**축을 제약하는 최고가치 사실**: **`B-8b` 가 `CHECKPOINT-LIMITED` 를 반증했다** — 실측에서
  학습을 통째로 빼고 해석적 초기화를 같은 기계에 넣어도 기술이 생기지 않는다(29/72).
  **체크포인트를 무엇으로 재학습하든, 같은 기계에 꽂아 실측 기술을 얻는다는 가설에는
  측정된 반대 증거가 이미 있다.**

---

## 1. 질문 1 — 손실 변경을 시도한 기록

### 1.1 ★`wp_e2e_2026-07-25` = 장-오차 보조손실 fine-tune, **사전등록 FAIL(널)**

`tmp/wp_e2e_2026-07-25/{PREREG.md, WP_E2E_2026-07-25.md, e2e_finetune.py}` ·
체크포인트 `results/direction3/prior_net_e2eft_{A,B}.pt`(2026-07-26 00:58/01:01).

- **정확히 M7 이 제안하는 부류다**: ε-MSE 앵커 + **λ·field-MSE**(x̂₀ 디코드 → denorm →
  **미분가능 torch ridge-LS 로 w/φ 적합** → held-out 격자 렌더 → (RMSE/Hs)² vs 참 장).
  장 그래디언트가 kx/ky 배치에 집중되도록 설계됨(w/φ 는 LS 함수).
- **PRIMARY FAIL**: 혼잡 40 에피소드, ft vs 현행 ckpt — median Δ **−0.0007**, 21/40, **p=0.30**.
  6개 층 분해 전부 널. 통합 팔 스팟체크도 널(4/10, p=0.35).
- ★**config B(λ_field=4)는 명백한 음성**: 장 손실은 내렸는데 **앵커가 0.207→0.366 으로 붕괴**,
  ancestral 샘플러 val 0.242 = **미조정보다 나쁨**. 동결 선택 규칙이 올바르게 기각.
  ⟹ 교훈(당시 기록 축자): *"field loss must be a light auxiliary … or it detaches the net
  from being a valid DDPM denoiser."*
- ★**널의 국소화(§3, MEASURED)**: fine-tune 은 자기 학습 관측분포(make_batch)에서 **실제 −9%**
  개선을 만들었는데 에피소드 프로토콜(LiDAR 스윕 기하)에서 **증발**했다.
  §9 권고 = *"관측 분포를 에피소드 프로토콜에 정합시켜라"* — **그 처방은 이튿날
  `mixed_retrain` D3(GeometryBank)로 이행됐지만, 장 손실 없이** 이행됐다.
  ⟹ ⚠**"장 손실 × 정합 기하" 조합은 이 저장소에서 한 번도 측정된 적이 없다.**
- ⚠단 같은 문서 §8 ARGUED: *"제안 넷은 이 관측에서 beamform-조건 사전이 배치할 수 있는
  천장 근처일 가능성"* — 조합 재시도의 기대값을 낮추는 논거(측정 아님).

### 1.2 `cd_task`(task-loss fine-tune, 2026-07-22) — **HOS 재검에서 사망**

`tmp/taskft_cd.log`(`scripts/d3_prior_net_taskft.py`, best val @2200 step) →
`results/direction3/prior_net_cd_task.pt`. 판정 = `tmp/HOS_LADDER_RESULT_2026-07-22.md` Q2:
**v3 대비 −0.027 열위, "더 나은 net forward-path 사망 확정"**. 같은 문서 Q1 이
합성 "+0.45 headroom" 을 **inverse crime 인위산물**로 확정(HOS 천장 0.52 < net 0.79).

### 1.3 B2 = DPS likelihood-guidance 계보 (학습 아님 — 시험시점 유도)

`scripts/d3_setdiff_posterior.py`(DPS, Chung 2023) → `d3_setdiff_posterior_b2.py`(**B2** =
ΠGDM 잡음연동 유도 + **Gauss-Newton polish**, *"Reuses the frozen prior — no retraining"*).
`d3_beamform_cond_diffusion.py:4-6` 이 이 B2 의 유도를 *"weak"* 라 부르며 **beamform 조건화
(amortized)** 로 대체했다. ★단 **B2 의 polish 는 죽지 않았다** — 배포 경로
`learned_init.propose_atoms`(`:123-126`, polish_iters=80)가 지금도 관측 log-likelihood 를
직접 최대화한다. ⟹ **관측-우도는 배포 경로의 시험 시점에 이미 들어가 있다**(polish + VARPRO 400).
빠져 있는 것은 **학습 시점의 장-오차 벌점** 하나이고, 그것이 1.1 의 널이다.

### 1.4 사전등록 자신의 회계

`mixed_retrain PREREG` 서두 축자: 시도 1–3(`prior_net_compdiverse`, `prior_net_fan/fanTF`,
`prior_net_e2eft_A/B`)은 *"all targeted **architecture / loss**. That card is spent
(three independent FAILs)."* — **손실/아키텍처 카드는 저장소 스스로 "소진" 으로 동결해 뒀다.**
(⚠단 이 문장은 2026-07-26 시점 회계이고, 미시도 조합(1.1)의 존재와 모순되지 않는다.)

비고: direction-3 동물원에 `amortized_field_matched.pt`·`amortized_ff_mse.pt`·
`h2_taskloss_sites_*.pt` 등 field/MSE-족 구형 체크포인트가 더 있다(07-02~11, 배포 계보 밖).
**열지 않았다** — 주장 범위 밖.

---

## 2. 질문 2 — "어려운 데이터로 학습" 시도 기록

### 2.1 ★`mixed_retrain_2026-07-26` — 프로그램 4번째 시도이자 **유일한 PASS**, 그리고 배포됨

`tmp/mixed_retrain_2026-07-26/{PREREG.md(+ADDENDUM 1·2, APPENDIX A), LOOP_LOG.md, RESULT.md}`.

- **데이터만 변경**(손실·아키텍처·옵티마이저 byte-for-byte 동일, eps-MSE): 카디널리티 층화
  `nc∈{1,2,5,15,30}` + 컨티뉴엄(465 성분→32 k-means) + (Hs,Tp) 50% BF2–BF5 앵커 ±15% +
  **GeometryBank 실 스윕 기하**(D3). ⚠**관측 잡음 `N(0,0.02)`·선형 forward 는 "출하 레시피
  그대로" 동결** — 즉 이 배치는 **뾰족함/잡음 축을 건드리지 않았다**. M7 이 노리는 축은 여기서
  비어 있다.
- 3 라운드(라운드당 단일 변경, 게이트 동결·1회): r1 G2a FAIL(12>6) → r2 에너지-graded 타깃
  FAIL(8>6) → **r3 FULL PASS**(G1 0.17491 ≤ 0.17878 · G2a 6 · G2b 0.01047 < v2 0.01355).
- ★**정직한 대차대조**(RESULT §5.3–5.4): 이득 = **파국 제거**(BF2 0.3286→0.0158, 제안이
  자기 조건자를 10–50배 초해상). 비용 = **혼잡 정확도의 단조 침식**(r3 +0.0061), 그리고
  40 에피소드 짝-중앙값에서는 **개선 아님**(v3 는 구넷 대비 p=0.029 로 오히려 유의 열위).
  *"the retrained net stops failing catastrophically … not a broad accuracy shift."*
- **남은 약점(재학습 표적 후보)**: moderate 5comp **0.042–0.045 vs 해석적 0.023** —
  혼합 10% 로는 못 고쳤다(RESULT §5.4).
- **운영 규율의 본보기**: max 3 라운드 · 라운드당 addendum 동결 · 체크포인트 절대 미덮어쓰기 ·
  from scratch(*"a data fix must not inherit the OOD mode"*) · 시드 구간 분리(학습 ≥2e9,
  평가 <2000 = 검사 가능한 구간 성질) · 하네스 동일성 대조 max|Δ|=0.
- G4(실측): 0.805→0.783 이지만 **여전히 hand 0.744 에 진다** — 실측 주장 불허 그대로.

### 2.2 fan-geometry 재학습(2026-07-24) — **실측-유사 기하 학습의 직접 선례, 축 폐쇄**

`tmp/G1_waveprior_power/G1_SESSION_2026-07-24_EVENING_REPORT.md` §5–7 ·
`results/direction3/prior_net_fan.pt`·`prior_net_fanTF.pt`(구 prior_net 아키텍처).
- 200시드 fan_frac 혼합 데이터로 재학습. **합성 PRIMARY FAIL**(fanTF vs hand d +0.0159,
  18/30, p=0.10). SECONDARY 는 PASS(구넷 대비 +0.0649) = **기하 미스매치는 실제로 고쳐졌는데
  그래도 해석적을 못 이겼다**.
- **실측 재검**: fanTF−v3 paired **−0.014 = 무승부~소폭 악화**, 정상 27창 n.s.
  판정 축자: *"fan-geometry 축 폐쇄 … 'novelty≠우월성' 판정 3중 확인(v3 재학습·40k 재학습·
  fan-geometry 재학습)"*.
- ⟹ ★**"학습 데이터를 실측 기하처럼 만든다" 는 이미 한 번 시도됐고, 기하 정합에 성공하고도
  이득이 없었다**(단 구 아키텍처·구 관측모형 시점이라는 한계는 있다).

### 2.3 compdiverse(2026-07-22) — nc-다양화 데이터, **이중 검증 FAIL**

`tmp/BETTERNET_fable_2026-07-22.md`: *"dual-verified FAIL (candidate net regresses on BOTH
regimes)"* + 평가 사슬 전체의 inverse-crime 판정. `tmp/train_compdiverse.log`.

---

## 3. 질문 3 — HOS/비선형 자료로 **학습**한 적이 있는가: **있다(1회), 이득 0**

- `scripts/build_dataset.py`(축자: *"ONE fixed training dataset (**linear + real-HOS
  nonlinear**) … HOSSimulator init→top-32 free modes as labels; bound harmonics present in obs,
  labels stay the free modes → nonlinearity-robust supervision"*, `--frac_hos 0.35`) →
  `beamform_cond_diffusion_ds.pt`(HOS 혼합) vs `_dslin.pt`(**선형-전용 대조** — 데이터 규모/
  표준화 효과에서 HOS-증강 효과를 분리하는 설계, `scripts/d3_hos_eval.py:31-37` 주석).
- **판정**(`results/direction3/hos_eval_new.json`, HOS M=3 평가 15창):
  `b1_ds` **0.2462** vs `b1_dslin` **0.2403** · `a1_ds` 0.2734 vs `a1_dslin` 0.2724.
  ⟹ ★**HOS 혼합 학습은 HOS 평가에서조차 이득이 없었다(근소 열위)**. ⚠n=15, 짝 검정 미기록 —
  강한 널은 아니지만 방향은 명확.
- 배포 계보의 현행 학습은 **선형 확정**: `gen_mixed.py:344 nonlinear=False` 축자,
  `d3_beamform_cond_diffusion.py` 의 `make_batch(hos_frac=0.0)` 는 **train() 이 인자를 아예
  안 넘긴다**(`:144` — hos_frac 는 死 파라미터). 2nd-order Stokes 훅과 `FixedSampler(nl_filter)`
  훅은 **코드에 살아 있으나 배포 학습에 연결된 적 없음**.
- `HOSGATE`·`B-11`·`B-12`·`M1`(진행 중)은 전부 **평가**다 — 학습 아님(임무 전제 확인됨).

---

## 4. 질문 4 — 체크포인트 계보 (확정)

```
setdiff_prior.pt (06-27, 무조건 사전; stats/M=32/K_LIM 만 상속)
  → beamform_cond_diffusion.pt (07-03 14:01, 15k step, make_batch:
     Hs~U[0.8,3.2]·Tp~U[5,11]·선형·σ=0.02·풋프린트×독립 t)
  → ★beamform_cond_diffusion_mixed_v1.pt (07-27 00:32, 40k step from scratch,
     gen_mixed 혼합 커리큘럼) = 현 배포
```

- 로더 = `wavesplat/optimizer/learned_init.py:61-63` `DEFAULT_CKPT_REL =
  'results/direction3/beamform_cond_diffusion_mixed_v1.pt'`. **sha256 실측 재검 일치**
  (`636c72a7…` == `DEFAULT_CKPT_SHA256`). 손실 = eps-MSE 단독(`train_mixed.py:125` ==
  `d3_beamform_cond_diffusion.py:150`).
- **배포 결정**: `tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md` §1 — v1·v3 둘 다 전 바 PASS,
  **동결 tie-break(혼잡 paired 11/16)로 v1** 선택(⛔"v1 우월" 서술 금지 — p=0.105,
  `learned_init.py:47-58` 주석이 명문화).
- ⚠**체크포인트는 07-27 이후 불변인데 하류 기계는 08-22 에 바뀌었다**(varpro_steps 400 ·
  learned_init_iters 0 · auto_switch_thresh 0). W15 가 합성 정확도 불변(70% 정확 동점)을 검증.
  `auto_switch_thresh=0` 실측 확인(`sequential_update.py:178`) ⟹ **합성 경로 100% 가 이
  체크포인트를 탄다**(실측은 `:715` 게이트가 `bank_mmse` 로 — 체크포인트 무관).
- ★**원장 문장 검증** — *"체크포인트가 평가와 같은 분포"* 는 **정확하다**:
  `PREREG §2.2` 자신이 *"anchoring half the scenes on the BF2–BF5 (Hs,Tp) points … It is
  distribution-matching … It does make G1/G2 **in-family, out-of-realisation** tests"* 라
  선언했고, 평가 코퍼스(B-1/S13/B-12/B-26 SS-BF4)는 그 앵커 위다. 시드만 구간 분리
  (학습 ≥2e9 / 평가 <2000) ⟹ *"out-of-sample 은 시드 축에서만"* 도 정확.
  단서의 현재 상태: ①forward 축은 평가에서 이동됐고(HOS, B-12 43/48 생존) 학습은 선형 그대로
  ②(Hs,Tp) 지지집합 밖 코너는 B-26 이 **탐색적으로만** 벌었다 — `reference_citation_bans.md:743`
  축자: *"'체크포인트 = 평가 분포' 단서(심사 사유 ③)를 닫지 않는다 — 닫히는 것은 (Hs,Tp) 두
  좌표뿐. 확증은 사전등록 배치로만(M3)"* — **M3 는 진행 중(verdict 없음, 08-28 실측)**.

---

## 5. 질문 5 — 학습 비용 실측

| 항목 | 값 | 출처 |
|---|---|---|
| 전체 재학습 40k step, batch 32 | **4712 s = 78.5 min**(r1) · 69 min(r2/r3) | `train.log` 말미 · RESULT §3 |
| step 단가 | 140 ms(예산) / **104 ms 실측**(r2) | PREREG §4 · LOOP_LOG r2 |
| e2eft fine-tune 4k step | **≈3 min/config**(A+B ≈6 min) | WP_E2E §Deviations |
| 게이트 포함 1 라운드 | 밤 1회(23:13→r3 저장 04:07) | LOOP_LOG |
| GPU | RTX 4080 Super(16.4 GB) · **VRAM 미기록**(state_dict 5.86 MB, 소형 모델) | RESULT §3 · ls |
| ⚠현재 | `M1` 이 **14.7/16.4 GB 점유**(~17h 잔여) — 동시 학습 불가 | M4 RESULT §8 |

---

## 6. ★이 축을 죽이거나 제약하는 사실 (최고가치 산출)

**표적을 "실측 전이" 로 잡는 형태는 측정 4건이 막는다:**

1. ★★**`B-8b` 가 `CHECKPOINT-LIMITED` 를 반증했다**(`reference_citation_bans.md:332-338`):
   실측 n=72 에서 학습을 통째로 빼도(`V2K` 29/72) 기술이 없고, 학습 경로는 **21/72 =
   무작위보다 유의하게 나쁘다**(p=2.7e-04). 동결 라벨 `SPLIT-UNDECIDED`(표현 한계도 미확정).
   ⟹ **"체크포인트만 더 잘 만들면 실측이 열린다" 는 가설은 이미 반대 증거를 가졌다** —
   재학습된 체크포인트도 같은 기계(polish+VARPRO+추적기)에 꽂힌다.
2. ★**소거표가 "실측-유사 어려움" 의 자연 축들을 전부 배제했다**(`tmp/STATUS_2026-08-28/
   STATUS.md §5` + 오늘 `M4`): 잡음의 형태(`B-14b`) · 잡음 수준·희소성·조합(`B-18`) ·
   센서 모형(`B-15`) · 해상상태(`B-26`) · ★**표현 용량(`M4`, 오늘: 방향이 반대 —
   `CAPACITY-HURTS` 12/12·48/48, 용량-반응 단조)**. 남은 후보(관측 기하·창/시간축·정상성·
   스테레오면 오차·사이트 구성)는 **전부 미측정** ⟹ **"뾰족하고 어렵게" 를 어느 축으로 만들지가
   소거표 위에서 정의되지 않으면, 이미 배제된 축을 학습 데이터에 넣는 배치가 된다.**
   `B-23`(실측 분포 재학습)이 정확히 이 사유로 탈락했다(`bans:708`, *"결과가 처방을 죽였다"*).
3. ★**W12 의 기제는 넷의 학습 손실이 아니라 시험 시점 적합에 있다**: 관측 목적함수를 63.2%
   낮춘 것은 **VARPRO(시험 시점, 400 step)** 이고, 장 오차 악화는 그 뒤에 온다. 넷의 학습
   손실을 바꿔도 **시험 시점 목적함수는 그대로다** — M7 가설(*"벌점을 받은 적이 없어서"*)은
   제안 단계에만 닿고, 실측 과적합의 측정된 발생 지점(VARPRO)에는 닿지 않는다.
   (`B-2c` 도 실측에서 λ 벌점을 전면 제거해도 7–8/12 불변을 측정했다.)
4. **직접 선례 전패**: fan-geometry 재학습(실측-유사 기하) = 합성 FAIL + 실측 무개선(§2.2) ·
   HOS-혼합 학습 = HOS 평가에서도 이득 0(§3) · e2eft 장 손실 = 널 + 강가중은 파괴적(§1.1) ·
   cd_task = 사망(§1.2). **성공한 유일한 재학습은 "실측을 닮게" 가 아니라 "평가 프로토콜을
   닮게"(분포 정합) 만든 것이었다**(§2.1).

**금지 규율(양방향)**:
- ⛔*"재학습 무익"* 주장 금지 — 두 번 명문화(`bans:338` B-8 은 무학습 대안을 쟀을 뿐 ·
  `bans:710` *"시험된 적 없음 — 08-23 단서 유지"*).
- ⛔*"B-23 을 다음 축으로"* 인용 금지(`bans:708`) — **우선순위 근거가 죽었다**(무익 판정 아님).

---

## 7. 그럼에도 살아 있는 표적 3개 (설계 임무 A 에 넘기는 것)

1. ★**심사 단서 ③("체크포인트 = 평가 분포")은 학습-측 개입으로만 닫힌다** — 평가-측은
   B-26(탐색적)·M3(진행 중, (Hs,Tp) 두 좌표 한정)가 전부다. **의도적으로 이동/확장된 분포로
   재학습 → 기존 동결 코퍼스에서 재평가**는 이 단서를 정면으로 겨누는 유일한 부류이고,
   ML 심사가 남긴 **마지막 사유**다(원장 08-15 축자). 이 표적은 **합성 헤드라인의 자격**에
   관한 것이지 실측 전이가 아니다 — §6 의 방벽에 걸리지 않는다.
2. ⚠**"장 손실 × 정합 기하" 미시도 조합**(§1.1) — e2eft 널의 자체 진단이 지목한 결측 셀.
   저비용(fine-tune ≈3 min, 전체 40k ≈70 min). 반대 논거 = e2eft §8 천장 ARGUED + §6-3
   (실측에는 못 닿음 — 표적은 합성 한정으로 선언해야 함).
3. **`M1` 결과 대기 축**: HOS M=3 × 실측 해상상태 교호 칸이 BREAKS 면, "비선형 자료로
   재학습" 이 처음으로 **측정된 표적**을 얻는다(현재 학습은 `nonlinear=False`, §3 의 ds/dslin
   널은 구 아키텍처·구 대조라 재측정 여지 있음). ⛔M1 결과 전에는 아무것도 읽지 않는다.
   (부수 표적: moderate 5comp 잔존 약점 §2.1 · M4 §9.8 이 미측정으로 남긴 제안 풀 축
   `nsamp·M` — 단 M4 방향은 용량 확대에 불리.)

**설계 시 주의(측정된 함정)**: ①전역 RNG 결합 — 성장 위상이 전역 torch RNG(`sequential_update.py:1617`),
한 draw 로 실측 nowcast ±429%(M4 `G-15x` 실측) ⟹ 팔 간 draw 소모 정합/계측 필수.
②`{}` 팔 정의 금지·세 배포 키 명시(08-22 변경 흡수 함정). ③재학습 거버넌스는 `mixed_retrain`
템플릿(라운드 상한·addendum 동결·from scratch·시드 구간 분리·하네스 Δ=0 대조)을 따를 것.
④fine-tune 이 아니라 from scratch 인 사유가 PREREG 에 명문("*a data fix must not inherit
the OOD mode*") — 감독 변경 배치도 같은 결정을 명시적으로 내려야 한다.

---

## 8. 열람한 파일 (주장 범위)

`tmp/mixed_retrain_2026-07-26/{PREREG.md,RESULT.md,LOOP_LOG.md,train.log}` ·
`tmp/wp_e2e_2026-07-25/WP_E2E_2026-07-25.md` · `tmp/G1_waveprior_power/G1_SESSION_…md` ·
`tmp/BETTERNET_fable_2026-07-22.md` · `tmp/HOS_LADDER_RESULT_2026-07-22.md` ·
`tmp/taskft_cd.log` · `tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md`(구조 스킴) ·
`tmp/W12_UNIFY_REAL_2026-08-22/RESULT.md` · `tmp/M4_CAPACITY_2026-08-28/RESULT.md` ·
`tmp/STATUS_2026-08-28/STATUS.md`(§5·M1 행) · `scripts/d3_beamform_cond_diffusion.py` ·
`scripts/d3_setdiff_posterior{,_b2}.py` · `scripts/d3_hos_eval.py` · `scripts/build_dataset.py` ·
`scripts/fixed_dataset.py` · `tmp/mixed_retrain_2026-07-26/gen_mixed.py`(grep) ·
`wavesplat/optimizer/learned_init.py` · `wavesplat/optimizer/sequential_update.py`(grep) ·
`results/direction3/hos_eval_new.json` · `results/direction3/*.pt`(ls) ·
`memory/reference_citation_bans.md`(관련 절) · `memory/project_persona_debate_novelty_2026-08-14.md`
(§2) · `memory/project_seastate_2026-08-28.md`(§1). sha256 재계산 1회(배포 ckpt).
