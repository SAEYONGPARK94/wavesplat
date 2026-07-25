# 논문 골격 초안 — 센서-인지 CRB-앵커 운용성 Envelope (NAOE) 2026-07-18

> 🛑🛑 **2026-07-22 정정 배너 (인용 전 필독, 권위 = `tmp/SESSION_MASTER_2026-07-22.md`·`tmp/HOS_LADDER_RESULT`·
> `tmp/PERSONA_DEBATE_novelty_phd_2026-07-22`)**:
> 1. **"estimator-independent"**: C8이 **한정형 SUPPORT**(under-det서 다른 추정기 클래스 예측, 이중검증)이나 synthetic
>    sandbox → **C8 HOS 재검 아직 남음**. **★§8 envelope HOS 재검 = 완료·PARTIAL·생존**(CRB↔actual HOS +0.30, 거리
>    넘어 partial +0.28) + **★CRB envelope이 경험UQ를 HOS서 이김(0.30 vs 0.21, 8바다 전부) = 핵심 delta HOS-검증**
>    (synthetic선 동률이었음). = spine A 핵심 자산이 HOS 살아남음. "예측 우위" 헤드라인 금지(PARTIAL). 권위 `tmp/HOS_S8_RESULT_2026-07-22.md`.
> 2. **§4(envelope-CRB, d1_crb 공유) ↔ §5(파티션 θ-CRB, e51 별개구현) = 다른 엔진** — "하나의 엔진" 통합서술 금지(F-2).
> 3. **B-2 "+0.139" 인용 금지 → "+0.022"(beamform-eval)**: oracle 참-피크 의존 판명(M-3/M-3G), F1 "훈련=oracle kp"
>    거짓(훈련=beamform). B = **별도 챕터·헤드라인 아님.**
> 4. **다채널 조건② = 이미 CLOSED**(F-3, E-55/56). THESIS_FRAMING_v2 §9/10만 stale.
> 5. **§8 = level-calibration을 1차 훅으로**(F-5, MEASURED). **⛔①under-det "학습 realizable forward-path"=철회**
>    (inverse crime, HOS서 −0.27 확인). **better-net=FAIL/parity.**
> 6. **★HOS-검증 positive(살아남음)**: 현 학습 net-MMSE가 **실제 HOS서 classical 이김**(0.79 vs 0.60) = spine A
>    "학습이 이기는 곳" 실물근거(단 "큰 갭 닫기" 아님·Ehlers OE2023 선점).
> 7. **박사 판정 = CONDITIONAL PASS(lower-mid)**: "방법 SOTA" 아니라 **"CRB-앵커 운용성 envelope+판별+정직음성"**
>    으로 재프레임. 스코프: ⛔"LiDAR"·"정보벽"·"학습 압도"·"이동/희소가 novelty".


> fable 박사심사 = **CONDITIONAL PASS**(조건 7개, 관건=envelope skill 검증 실험 미실행). 사용자 "괜찮으면
> 추천대로 작업" → 골격 초안. **규율 적용**([[feedback_direction_flip_discipline]]): 각 절 내용을 **MEASURED**
> (검증됨) / **PENDING**(조건 실험) 태그. 헤드라인은 검증 전까지 **설계(keystone-to-be-sealed)**로 표기.

## 제목 (후보)
"A Sensor-Aware Operability Envelope for Phase-Resolved Wave Forecasting from Sparse Moving Elevation
Observations: An Information-Theoretic (Cramér–Rao) Approach"
- ⛔ "LiDAR" 금지(미관측) · ⛔ "information wall" 금지(NAOE 부적합). 얼굴 = **운용성 지도**, CRB = 엔진.

## Abstract (fable 초안 기반)
선상 희소 이동 표고 관측으로부터의 phase-resolved 파랑 예보에서, 예보 신뢰 시공간 영역(**운용성 envelope**)을
학습 추정기의 경험적 불확실성이 아니라 **센서 사양·궤적·해상에 대한 정보이론 하한(Cramér–Rao)에 앵커하여
계산·보정·검증하는 방법**을 제시. envelope의 내부 엔진 = 보조-정보 채널을 5요소(가치·회수성·공급·달성성·
정보한계)로 회계하고 CRB로 "회수 실패가 정보벽인가 추정기 효율인가"를 판별하는 프로토콜. 두 채널(레짐
파티션·표면 해류)에 적용해 **채널마다 다른 사망요인 판별**을 실증. 보완 = 우도-실패 판정 동기의 비-우도
task-loss로 추정기를 제어가치에 정렬(제어전달 5/5 레코드 방향 일관, 군집급 경계선 — §7 참조). 모든 판정 = 사전등록·군집보정·HOS 재현·대칭감사로
봉인. 스코프 = 모사 센서 연산자(스테레오 실측 anchor 병기).

## 절 구조 + MEASURED/PENDING 태그

**§1 Introduction** — phase-resolved 예보의 운용 질문("언제·어디서 믿나"). gap: 기존 envelope = 기하-해석적
(Naaijen) 또는 경험 ML-UQ(Zhao), **정보이론 앵커 부재**(litcheck 확인, MEASURED).

**§2 Related work / difference-first** — DSWP·predictable zone·ML-UQ. 포지셔닝: "그들 = 추정기-종속 경험 UQ /
우리 = 추정기-독립 정보이론 하한, 정보벽 vs 효율갭 판별 가능"(MEASURED: litcheck). ⚠경쟁 Zhao 그룹·시간리스크.

**§3 Method — 추정기(vehicle)** — 분산제약 스펙트럴 GP: z=Φw+noise, ω=√(g|k|tanh), 사후평균 장 = 선형 solve.
prior γ(k)는 빔포밍+학습(MEASURED: 확립된 기계, novelty 아님·difference-first만).

**§4 Method — ★CRB-앵커 운용성 envelope (기여)** — FIM=(1/σ²)JᵀJ, D_eff=L+c_g·T_obs, envelope=CRB<임계 시공간
영역, 사후분산 보정. (MEASURED: CRB 백본 direction1_crb·2클래스 FIM 성분급+파티션급 E-51/52 / **PENDING:
envelope→skill 예측 검증 = 조건①**).

**§5 정보-채널 회계 프로토콜 (내부 엔진)** — 5요소 + CRB 판별(정보한계 vs 효율갭). (MEASURED: E-52 결과 B
판별 — regime은 정보벽 아니라 효율갭 +0.046, 정보잔여 +0.040).

**§6 채널 연구 (판별타당도)** — regime: 가치 +0.043(n=957)·blind≈net(arena X·Y, 전수열거)·achievability 사망
(부이급 부재). current: value 사망(kp-fix, radar급 무가치, base-robust). **두 채널 다른 사망요인 = 프로토콜
판별 실증**(MEASURED: E-42~E-56 3중검증 / PENDING: 조건② 개구 채널 추가 = 3채널 판정표).

**§7 B-2 task-loss 학습 (효율갭 닫기)** — 우도-실패 동기 비-우도 task-loss, 미분가능 물리-사후 관통, 제어정렬.
차별=물리 추정기 유지(vs end-to-end PoF2023)+검증사슬.
**⚠2026-07-19 E-60 정정**: 구 "제어전달 t−4.02"는 **iid t(레코드 3개 안 창 72개)** = ⛔인용 금지.
현재 = 5레코드 확장(`d20_replay_eval_clusterrobust.json`) **5/5 부호 일관·효과 −0.123·cluster-t p 0.073·
Webb wild p 0.027·`headline_pass: false`** ⟹ **MEASURED = 방향 일관 / ARGUED = 통계적 확립(경계선, k≈10 필요)**.
동일 정정: T−L 정면대결은 동결 게이트 PASS이나 **군집급 미확립**(p 0.19, 3/5); HOS 재현 **~null**
(pooled +0.010, 5/8, 게다가 "바다 2개" 아레나); s²-head는 **고정 smult 3.25 대비 이득 없음**(T_fix 0.618 ≥ T_own 0.615).
권위 = `tmp/E60_B2_staleness_audit_2026-07-19.md` + `tmp/E60_B2_headline_fable_review_2026-07-19.md`.

**§8 ★Validation (KEYSTONE — 조건①=MEASURED, 조건③=PENDING)** — envelope 예측 신뢰영역 vs 실제
phase-resolved skill(에피소드별 Spearman(σ_CRB, σ_actual), n=36; 사전등록 E-58 + 독립감사 E-58b, 권위
`tmp/E58_AUDIT_findings_2026-07-18.md`). **결과 = PARTIAL(예측바 ρ≥0.5 미달), 감사가 3중 정련**:
- **(i) under-determination 스케일링** (MEASURED): pooled ρ +0.33은 well-det 지배 — 분해하면 kp에 강하게
  스케일 **Spearman(kp, ρ)=−0.874 (p=3e-12)**, under-det 레짐(15/30comp) ρ **0.74~0.78 = ≥0.5 통과**,
  well-det(5comp) 0.31. partial|dist도 under-det서 상승(0.15→0.63, 거리 초과 생존). ⟹ envelope은
  **운용성이 실제 위험한 under-det 레짐서 강함**(개구비축, kp_floor 감사 정합).
- **(ii) estimator-비의존** (MEASURED): well-det modest ρ은 **비효율 아님** — 분리반분 감쇠보정 0.326≈관측·
  ρ가 ridge λ{1e-4~1e-2}·R{12~96}서 평평. **진짜 천장**(well-det skill map이 flat: 공간분산도 0.47 vs
  under-det 0.88) = "운용성 균일히 높음, 경고할 것 없음"을 올바로 보고.
- **(iii) 경험-UQ 대비** (MEASURED): σ_emp(ridge 사후분산, Zhao VB-NN 대리)가 **전 레짐서 동률**
  (0.31≈0.30, 0.74≈0.71, 0.78≈0.78) ⟹ **"예측 우위" 없음(견고)**. 가치축을 재배치: envelope의 기여는
  더 나은 예측이 아니라 **추정기 없이 설계시점 계산 + 정보한계 vs 효율갭 판별**(경험 UQ 원리적으로 불가).
- **판정(goalpost 동결)**: 예측바로는 PARTIAL 유지(사후 PASS 금지)이나 **FAIL 트리거(거리 흡수·envelope
  무효) 미발동** ⟹ 헤드라인이 회계/진단으로 후퇴하지 않음 = **방법 헤드라인 생존(재특정, 봉인 아님)**.
- PENDING: 스테레오 실측 anchor 1건(조건③), 이동-센서 실측 미검·regime/HOS 확장·보정 피복.

**§9 정직한 음성 + 방법론** — 사전등록·7규율(검정력 게이트·MEASURED/ARGUED·대칭감사)·척추 동결·스코프 정직화.
= "결과 믿을 수 있나"에 대한 실질 답(MEASURED: 방법론적 성숙).

**§10 Conclusion** — 성능은 정보-속박. **센서+채널마다 "추정기 만들지 vs 외부정보 공급할지" 결정규칙** =
운용성 지도(NAOE 산출물).

## ★학위 성립 조건 (fable 7개, 로드맵)
1. **[✅해소] §8 envelope skill 검증 실행+독립감사** — PARTIAL이나 감사가 정련(under-det 스케일·estimator-
   비의존·경험UQ동률) ⟹ 회계/진단 후퇴 아님, 방법 헤드라인 생존. 남은 것=조건③ 실측 anchor·서술 갱신.
2. **다채널 완주**(⚠E-54와 조정 필요): 원안 "개구 채널 추가=3채널 판정표"였으나 **E-54가 개구=채널 아님·
   CRB 백본(Direction-I) 재분류**(supply/recoverability 파싱불가=범주오류) ⟹ 채널은 2개(regime=achievability
   사망·current=value 사망). 개구는 §4 CRB 엔진으로 흡수. **판정=2채널 상이-사망요인 + CRB 판별기 2클래스.**
3. **[△QUALIFIED PARTIAL] 실측 anchor(E-59, fable 2라운드+버그2 수정)**: strip-obs로 실측 스테레오서 envelope↔
   skill 시험 — 버그수정 후 Δρ(coherent 신호 beyond 기하)=**+0.085 CI[+0.022,+0.113]**(3레코드 일관·0제외,
   ridge↑서 +0.5까지=보수하한) but model-class 천장의 **~8~19%**뿐, crb≈dist≈emp≈0.7=대부분 기하. **약-양성·
   미refuted·미clean검증.** 캐비엇: 스테레오(실물 이동 LiDAR 미관측)·작은패치는 under-det 원천불가(YS17만 자격)·
   linear twin=천장과대(HOS 미반영). ⛔"실측 검증됨"·"반증됨" 둘 다 금지. 권위=`tmp/E59_AUDIT_findings_2026-07-18.md`.
4. **[✅MEASURED] §7 random-phase 캐비엇**: §8 레짐서 필드 Gaussianity 확인 — Shapiro-W 5comp 0.991/
   15comp 0.995/30comp 0.995(CLT, comp수↑서 개선), 초과첨도 −0.58→−0.19. **envelope 강한 under-det
   레짐이 가장 Gaussian ⟹ 감사 강-레짐이 CRB 근사 가장 탄탄.** +§8은 rank corr라 절대-CRB 근사 무영향.
5. **지도교수 논문 유형 명시 수용**(v1 "조합" 지적에 difference-first+판별타당도 선제, 문서화).
6. **속도**: envelope 논문(채널1+프로토콜) 6개월 내 투고(Zhao 창 방어).
7. **명명·스코프 규율**: LiDAR/information-wall 금지·배포주장 삭제 유지.

## 정직한 현 위치 (2026-07-18 E-58 감사 후 갱신)
- **논문-급 확정 / 박사-급 = 다채널 일반성 완주 조건부**(fable). 자산 = 저널 2~3편, 통합 서사 존재 = 박사 하한 만족.
- **★조건①(최대 리스크) 해소**: §8 검증+독립감사 완료 = PARTIAL이나 감사가 정련(방향 재고 불필요, FAIL 미발동).
  헤드라인이 "예측 우위"→**"추정기-독립 설계시점 운용성 envelope, under-det서 강함(kp ρ−0.874), 정보/효율
  판별"**로 재특정됨(봉인 아님). 남은 관건 = **조건③ 실측 anchor + 조건⑤ 지도교수 difference-first 축 합의**.
- **다음 실제 작업 후보**: (a) 정련 헤드라인으로 §8/Abstract/§2 서술 갱신[진행중] → (b) 조건③ 스테레오
  실측 anchor 1건(data/real_bergamasco/, envelope↔skill 실측 재현) → (c) 조건④ §7 random-phase ARGUED→MEASURED.
