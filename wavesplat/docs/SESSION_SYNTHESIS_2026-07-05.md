# 세션 종합 2026-07-05 — "연속-k 초분해" 서사의 해체와 재구성: 상금은 차수, 열쇠는 정규화된 선형추정

어제(07-04)의 최종 특정 "chirped 이동샘플링 하 연속-k 초분해 sparse 추정(NOMP 계열)"을
직접 실행해 본 세션. 결과: **그 특정은 다시 한 번 틀렸고**, 더 깊은 구조가 드러났다.
DPS×이동 양성도 기각. 오늘의 사슬은 어제의 결론 두 개를 모두 뒤집었다.

## 발견의 사슬 (오전 세션, 시간순)

1. **DPS×이동 +0.51 기각.** `d3_sa_dps_moving.py`에 multi-window 루프(3ep×5win×3조건,
   45런, frozen prior 재학습 없음). track043 DPS unobs corr 평균 +0.04/중앙값 +0.02,
   track100 평균 −0.00. 원조건(30comp seed0 ti=25)도 새 obs 추첨에서 −0.13.
   DPS는 nrmse ~0.3 유지(RELAX >1.0)로 "우아한 실패"는 맞지만 미관측 영역 위상
   외삽은 0. → `results/direction3/sa_dps_moving_multiwin.json`
2. **NOMP 그대로 구현하면 진다.** FD-Newton cyclic NOMP(성분별 profile-power 뉴턴 +
   cyclic re-refine + CV stop): 10-sea 평균 0.172/0.169/0.089 < RELAX
   0.199/0.222/0.183. joint-Adam interleave 변형도 마찬가지.
3. **★oracle_ref의 재독해 — 자유 refine이 참 k도 파괴한다.** 저장된 cont 하니스에서
   oracle_ref(참 k init + 비정규화 joint Adam refine) = 0.23/0.245/0.22로 **참 k를
   주고도 RELAX 평탄선까지 붕괴**. 반면 oracle_ls2(참 k + ridge λ=1.0 선형피팅)는
   0.28/0.41/0.46. oracle_ls(λ=0.1)→oracle_ls2(λ=1.0)만으로 fixed 0.145→0.278.
   병목의 절반은 검출이 아니라 **근접-공선 모드쌍의 과소정규화 피팅**
   (JONSWAP 피크 클러스터링 → 약한 ridge에서 계수쌍 상쇄폭주).
4. **★차수 프로브 — 상금은 모델 차수다.** 참 top-N k + ridge λ=1.0의 N 스윕
   (`d3_sa_orderprobe.py`): CV가 멈추는 N~10에선 **참 k조차 0.15~0.32 ≈ RELAX**.
   N=48~64에서 track043 0.47/0.48, track100 **0.56/0.60**. 즉 greedy+CV 조기중단은
   구조적으로 상금 접근 불가; AA14의 "강제성장 악화"는 약한 ridge 탓이었다.
   → `results/direction3/sa_orderprobe.json`
5. **★정밀도 요구는 느슨, 검출은 유령.** 참 top-48 k에 δk 섭동(`tmp/probe_kprecision.py`):
   δk=0.01에서도 상금 대부분 생존(track100 0.534→0.477). 그런데 greedy 검출 k의
   nearest-true-mode 오차 중앙값 = **0.05~0.09** (요구의 5~9배) — 약간 빗나간 추정이
   아니라 **궤적-chirp 간섭이 만드는 유령 피크**. "초분해 정밀도" 서사 기각:
   필요한 건 초분해가 아니라 유령 억제 + 차수 + 정규화.
6. **fine-bank 동시선별(bank48)이 유령을 잡는다.** 0.015 간격 연속 후보은행 +
   group-FOCUSS IRLS(eps-annealing, graduated non-convexity) + top-48 + Newton cyclic
   + λ-CV refit: 선별 k 오차 0.010~0.013 (grow48 greedy의 1/5~1/7). 10-sea 성능
   0.272/0.222/0.204 — 전 조건 RELAX 이상, fixed는 oracle_ls2@32(0.278)에 근접.
   greedy 순차 잔차의 오류전파 제거가 유령 억제의 핵심 (시간-반분할/leg-분할
   교차-간섭 검출은 부분 효과만).
7. **★★승자: 스펙트럼-가중 선형 MMSE (식별 완전 포기).** fine bank 전체에
   beamform-power 형상의 prior 분산(지수 p val-선택) + val-게이팅 SBL-EM 2~3회
   (empirical Bayes) + 순수 선형 예측 = `mmse_bank_estimate`:
   **10-sea 0.272/0.239/0.241** — 모든 조건에서 RELAX·bank48 이상, 비볼록 피팅 전무.
   oracle_ls2가 애초에 "bin-클래스 linear-MMSE의 plug-in"이었음을 상기하면 자연스러운
   귀결: **이동 샘플링 하 wave field 추정은 sparse 식별 문제가 아니라 스펙트럼 추정 +
   정규화 선형추정(kriging) 문제다.**
8. **실측 재도전 (BS11a + AA14, estimator=mmse, 풀런 완료).** 아래 §실측 참조 —
   이동 10셀 중 8셀 절대 개선, AA14 이동 해로움 −0.23→−0.04 (추정기 아티팩트
   확증), BS11a 이동이득은 +0.11로 축소·생존.

## 실측 결과 (BS11a + AA14, 6win×3rep, matched budget, 윈도우 페어링)

`sa_multirecord_mmse.json` vs `sa_multirecord.json`(RELAX), common_unobs corr_bl:

| | RELAX | MMSE | Δabs (win+) |
|---|---|---|---|
| BS11a A(fixed) | +0.257 | **+0.474** | +0.217 |
| BS11a B_r0.43 | +0.485 | **+0.585** | +0.100 (5/6) |
| BS11a B_r0.7 | +0.559 | +0.374 | **−0.184 (2/6)** |
| BS11a B_r1 | +0.534 | **+0.588** | +0.054 (3/6) |
| AA14 A(fixed) | +0.545 | +0.505 | −0.040 |
| AA14 B_r0.1 | +0.114 | **+0.343** | +0.230 (5/6) |
| AA14 B_r0.43 | +0.172 | **+0.405** | +0.234 (5/6) |
| AA14 B_r1 | +0.313 | **+0.464** | +0.150 (4/6) |

판독 3줄:

1. **MMSE 실측 전이 성공** — 이동 10셀 중 8셀 절대 개선(평균 +0.10~0.23), 학습
   없는 순수 물리·선형이라 sim2real 게이트 무관 (DPS/net 계열과 대조).
2. **AA14의 "이동 전패"는 대부분 추정기 아티팩트였다**: r1 gain −0.232(RELAX)
   → −0.041(MMSE). 이동 샘플링이 해로웠던 게 아니라 greedy 식별이 chirp에서
   무너졌던 것 (07-04 식별붕괴 발견의 실측 확증).
3. **이동 gain 서사는 축소·정화**: MMSE가 fixed 기아를 크게 해소(BS11a A
   0.257→0.474)하면서 이동 프리미엄이 줄어듦 — corr(gain, A_abs)=−0.56 구조의
   추정기 성분이 제거된 것. BS11a에서 r0.43/r1 gain +0.11 양수 잔존(앵커 이득이
   축소된 형태로 생존). **"이동의 가치" 주장은 강한 추정기 기준으로만 유효** —
   thesis 재프레임 필수.

주의: B_r0.7만 두 레코드 모두 악화(BS11a −0.184, AA14 −0.096) — 윈도우 분산인지
구조인지 미확인 (내일 8-record 확장에서 감시).

## 방법론적 교훈

- **oracle 사슬을 끝까지 읽어라.** oracle_ref(0.22 평탄)는 어제부터 저장돼 있었다.
  "참 k + refine도 붕괴"를 봤다면 초분해 서사는 하루 일찍 죽었다.
- **2-sea 스모크로 변형 튜닝 금지.** sea 간 분산이 변형 간 차이보다 크다. 오늘
  joint/leg-split 튜닝에 쓴 시간의 절반이 노이즈 추적이었다.
- **val(on-track) ≠ 일반화(off-track).** 이동 조건에서 track 위 holdout은 미관측
  영역 스킬과 정렬이 약함 — EM 게이팅이 가끔 역효과(sea0 track100). 공간-분할
  val은 내일 후보.
- 실행 인프라: 백그라운드 파이프에선 print 버퍼링 — `python3 -u` 또는 flush 필수.

## Novelty 상황 (litcheck 수행함, 과대주장 금지)

- 정규화 LS/선형 superposition 재구성은 DSWP·radar 계열에 확립(예: CN-RLS 2024
  MDPI Remote Sensing — condition-number 정칙화 LS; GP 음장 재구성; CS 스펙트럼 복원).
  **"스펙트럼-가중 정규화 선형추정"은 방법으로서 신규가 아니다.**
- 우리의 기여 후보는 방법이 아니라 **발견**: ⓐ 이동 샘플링 하 sparse 식별
  (RELAX/OMP/SBL/NOMP/강제성장/학습검출 전 계열)의 구조적 차수-한계 규명
  (oracle 3층 분해 + 차수/정밀도/유령 정량화), ⓑ 같은 조건에서 empirical-Bayes
  MMSE가 일관 우위라는 실증, ⓒ 이동 상금(ceiling 성장분)의 미회수 잔여 정량화.
- 헤드라인화 전에 "moving platform + sparse point sampling + phase-resolved MMSE"
  협대역 재검색 필요.

## 오후 연장 세션 (사용자 지시: 멈추라 할 때까지 계속)

진행: ④실측 6-record 확장(AA15a·BS11b·BS13a·BS13b·AA15b·YS17, LJ는 kp 패치
아티팩트로 bank 대역 어긋남 — lam_ref 처리 후 별도) ⑤공간-분할(extrapolation)
val 프로브 ⑥HOS M=3 검증(`d3_sa_hos_mmse.py`) ⑦협대역 novelty 정독 완료
(2504.13943 EnKF-HOS-CMI=동화·예보, 2508.03315 PINO=부이/레이더, Ocean Eng 2025
SNU/FIT=이동선박 스테레오 영상 DL — 셋 다 "sparse 점 샘플링 병목 해부" 각도 아님.
DSGP-Net 스니펫만 출처 미확정, 추적 필요).

**⑤ 공간-분할(extrapolation) val 판정 — 한계효과, 채택 안 함** (10 seas,
`tmp/probe_mmse_spatial.py`): track100만 +0.047(7/10), fixed −0.007·track043
−0.019. "on-track val 미정렬"은 남은 상금의 binding constraint가 아님.
**보너스 발견**: bank 간격 kp/10(≈0.008, promote본 기본값)이 오전 프로브의
0.015보다 강함 — random-val 기준 **0.340/0.264/0.234** (오전 보고치
0.272/0.239/0.241 대비 fixed +0.07). 공식 수치는 kp/10 기준으로 갱신.

**⑥ HOS M=3 검증 — 통과** (6 seas, `sa_hos_mmse_M3.json`): fixed RELAX
+0.016(2 seas −0.33/−0.22 붕괴) vs **MMSE +0.263**(무붕괴), 이동 동급
(0.241/0.228 vs 0.221/0.232). 비선형에서 헤드라인 유지 — 오히려 RELAX의
greedy가 bound-harmonic에 취약. caveat: M=1 대조군은 HOS 시뮬레이터
수치불안정(NaN) — 선형 대조는 합성 하니스가 대체.

**⑧ 학습 방향 재개 (사용자 지시: 학습 정확도가 논문/학위의 골격).**
설계 = `docs/DESIGN_amortized_prior_2026-07-05.md`: "net은 prior만, 추정은
물리가" (γ(k) 예측 → 물리 MMSE; mean-collapse 면역 + sim2real 안전판 +
gap-to-bound 평가 내장). 정규화 k-공간(51×51, dk=0.1kp) bank로 net이
구조적으로 record-agnostic. **v1(8k 샘플, 라벨회귀): G1 미달**
(netEM−hand −0.08/−0.01/+0.02; 3k step 과적합 = 데이터 부족, 판단 보류).
v2: 40k 샘플 + log 입력채널(raw 파워맵 피크포화 → 약모드 불가시 문제;
상금이 약모드에 있음). 인프라: `d3_prior_net_{data,train,eval}.py`.

**④ 판정 기준 (사전 등록, 결과 보기 전):**
1. 1차: record별 Δabs(MMSE−RELAX), supra-임계 이동셀(B_r0.43·B_r1)과 A(fixed).
2. 2차: record-레벨 corr(gain, A_abs)=−0.56 구조가 MMSE에서 약화되는가
   (= "starvation"의 추정기 성분 비율).
3. 3차: MMSE 기준 이동 gain ≥ +0.1 생존 record 집합 (BS11a형 개구이득의 실재 범위).

**⑧-G1 통과 (v3 task-loss)**: 라벨회귀(v2)는 G1 미달(동률선), **미분가능
MMSE 관통 task-loss fine-tune(v3)이 돌파** — net-MMSE 0.387/0.446/0.468
(hand 0.340/0.264/0.234, RELAX 0.199/0.222/0.183), oracle_ls2@32 초과,
**이동 gain 최초 양수(+0.08)**, ceiling 속도-성장 추적. "목적함수가 병목"
진단 적중. in-distribution 합성 주의 문구 유지.

**⑧-G3 통과 (실측 무-FT 전이, 사상 최초)**: BS11a fixed 0.579/AA14 0.582
(hand 0.474/0.505), 이동 9/10셀 hand 초과. **B_r0.7 이상이 net에서 양 레코드
해소**(AA14 r0.7 gain +0.002, 6/6 윈도우 +0.338) — 학습분포가 U/cp≤1.2의
encounter-frozen 대역(k*=g/U², r0.7일 때 2kp 대역 내)을 본 덕으로 해석.
잔여 퇴행 1건: BS11a B_r1 (net 0.409 vs hand 0.588; RELAX gain +0.28이
여전히 왕좌) — r1은 k*=kp(피크 동결)라 학습분포 경계, 추적.
정규화 k-공간(51×51, dk=0.1kp) bank + "net은 prior만, 추정은 물리" 구조가
역대 sim2real 전멸을 뚫은 첫 사례.

**B_r0.7 4연속 악화(hand-MMSE)의 기제 가설**: encounter-frequency 영점
k*=g/U²이 U=0.7cp에서 2.04kp(대역 내) — following-leg에서 해당 대역 모드가
시간 동결→설계 준특이→dense bank가 greedy보다 취약. U=0.43은 k*=5.4kp
대역 밖. 검증 프로브(대역 downweight) 내일 후보.

**④ 8-record 완주 판정 (hand-MMSE, 사전 등록 기준):**
1. Δabs: 이동 절대치 광범위 상승. **완전 반전 2건**(AA15b r0.43 +0.001→+0.222·
   r1 −0.162→+0.143; BS13b r1 −0.071→+0.153), 대폭 완화 2건(AA14, AA15a),
   불변/악화 2건(BS13a 혼조, YS17 역행 — 거리희석형).
2. **corr(gain_supra, A_abs): RELAX −0.58 → MMSE −0.24** — "이동은 고정
   정보기아일 때만 돈" 구조의 절반 이상이 추정기 아티팩트로 판명. 평균
   supra gain −0.148 → −0.047.
3. gain≥+0.1 집합: {BS11a} → {BS11a, AA15b} (+BS13b r1 셀).
오버나이트: net-MMSE(G3 승자)로 나머지 6 record 실행 중
(`sa_multirecord_mmse_net.json` 병합) — 내일 8-record 3-way 완전판.

**⑧-완결: 8-record 3-way 완전판 (net-MMSE 오버나이트 완주):**

| 평균 (8 record) | RELAX | hand-MMSE | net-MMSE |
|---|---|---|---|
| A(fixed) abs | 0.401 | 0.396 | **0.538** (+34%) |
| supra-moving abs (r0.43,r1) | 0.253 | 0.350 | **0.375** (+48%) |
| motion gain | −0.148 | −0.047 | −0.163 |
| corr(gain, A_abs) | −0.58 | −0.24 | **−0.11** |
| gain≥+0.1 집합 | {BS11a} | {BS11a,AA15b} | **∅** |

**이중 헤드라인 확정:**
1. **정확도(학습)**: net-MMSE가 실측 전 레코드에서 최강 — fixed 8/8 승
   (BS11a 0.26→0.58, YS17 0.20→0.45), supra 7/8 승. 무-fine-tune 전이.
2. **과학(이동 서사의 역전)**: 추정기가 강해질수록 이동 이득이 소멸
   (−0.58→−0.24→−0.11; 이득 집합 → 공집합). BS11a 앵커 +0.31(3회 재현)조차
   net-fixed 0.58 앞에서 증발 — **"개구의 벽"은 대부분 "알고리즘의 벽"이었다.**
   단, 합성(2λ·10Tp)에선 net-MMSE 이동 gain +0.08 양수 → 실측-합성 긴장이
   내일의 첫 질문 (패치 스케일? valid-mask? broadband 시간정보?).
   주장 강도 조절: "이 패치 스케일·예산에서 검출 불가" (worthless 아님).

**UQ calibration (저녁 프로브, `tmp/probe_uq_calibration.py`):** MMSE 해석적
posterior 분산의 정직도 — hand prior는 이동에서 과신 악화(z-std 1.10/1.20/
**1.47**, cov95 0.83@track100), **net prior는 전 연산자 균일 근사-보정**
(z-std 1.12~1.17, cov95 0.90~0.93). 학습 prior가 정확도뿐 아니라 이동-조건
UQ 정직도까지 개선. 잔여 ~15% 과신은 단일 인플레이션 계수 교정 대상.

**⑨ 실측-합성 이동이득 괴리 규명 (심야, `tmp/probe_gain_{bandwidth,
patch_current}.py`, `probe_current_fit.py`):**
- H2 스펙트럼 폭 기각: gain이 협대역일수록 커지지 않음 (+0.09/+0.13/+0.07).
- H5 패치 스케일 부분 인정: 1~1.3λ에서도 gain 건재(1.3λ 피크 +0.17), 3λ부터
  희석 → **YS17(4.2λ)형만 설명**.
- **H6 해류 Doppler 불일치 = 유력 원인**: 0.15 m/s에 hand gain 전멸
  (+0.088→−0.003), 0.3 m/s에 net gain 75% 증발(+0.126→+0.032). 연안 실측
  사이트 전형 해류 0.1~0.5 m/s. 물리: 이동 이득 = 장궤적 위상간섭 → 분산
  왜곡에 고정보다 취약.
- **처방 v1/v2 실패 (그 자체가 발견)**: 해류 2-파라미터 피팅을 ①on-track
  val RMSE ②분산-셸 집중도(고전 X-band 역산 목적함수)로 선택 → 둘 다
  스퓨리어스 해류 선택(|Uc|-hat 0.29~0.47 @ 진실 0). **sparse 이동 샘플에선
  해류 파라미터도 궤적 chirp 자기-간섭으로 오식별** — dense FFT 볼륨용
  고전 목적함수가 전이 안 됨. 다음 후보 = GP 증거 최대화
  log p(z|Uc) = −½zᵀK⁻¹z − ½log|K| (원리적, val/셸 불필요).

**⑩ 해류 사슬 완결 (재개 세션, `tmp/probe_current_{fit,dof}.py` + lib):**
- GP-증거 선택(v3)도 실패(|Uc|-hat 0.19~0.42 스퓨리어스): 단일 sparse 궤적에서
  해류는 약식별 — 관측은 (궤적방향 파수, 조우주파수)뿐이라 해류 시프트와
  이웃-k 재배정이 구별 불가(우도면 평탄). 3연속 실패 = 발견.
- **eff_dof 불변(135/262 유지) = "해류×이동 구조적 정보손실" 가설 기각.**
- **완전-oracle(Uc+kp 기지)에서 완전 복원**(track 0.294 ≈ 무해류 0.311):
  해류 해악 = 추정기 채널 2개 ①설계 분산관계(ω+k·Uc) ②kp/대역 배치
  (무보정 beamform). 처방 = 외부 해류(선박 드리프트/ADCP, 실무 표준) 주입
  또는 다중-윈도우 공동추정(해류는 윈도우 간 공유) — 후자 미구현.
- **⚠️경고 플래그(정직 기록)**: 16-sea·evidence-변형 추정기에서 무해류
  이동 gain ~0 (win+ 7/16) — 합성 이동 gain 크기(+0.08~0.13)는 추정기
  변형·시드 민감. "이동 이득 실재" 주장 전 대표본(32+ seas) 재평가 필수.

**⑪ 대표본 이동 gain 확정 (n=32, 시드 7000+, `tmp/probe_gain_largeN.py`):**
hand gain 0 (+0.013/−0.029 비유의 — 8-sea +0.088은 시드 럭),
**net는 고속(1.0cp)만 +0.069±0.033 (t≈2.1, 22/32)**, 0.43cp 없음.
정착 주장: "합성 이동 이득 = net-prior × 고속 한정, +0.07 수준(2σ)".
절대 정확도 우위(net fixed 0.372/t100 0.441)는 견고.

**⑫ 해류 아크 최종 폐쇄 + BMA (5번째 실패)**: BMA-over-Uc도 실패
(bma < plain 전 셀; ESS 2.2~4.6/25 = evidence 면이 **틀린** Uc에 뾰족 —
평탄이 아니라 편향). v4 해류-증강 학습도 무효(γ-prior는 전 스펙트럼
시프트를 보상 불가). **kp_known 조건에선 해류 손실 자체가 미미(−0.002~
−0.04)** → 해악의 주 채널 = kp/대역 배치 (beamform argmax의 Doppler 취약).
실전 처방 확정: ①외부 해류 주입 ②kp_hint(characterize)를 net 경로에도
전달(실측 net은 내부 beamform kp 사용 중 — 즉시 개선 후보) ③광대역 밴드.
폐쇄 정리: 맹목 해류 복원 5/5 실패(val·shell·evidence·multiwin·BMA),
oracle 복원, 채널 특정 — "sparse 이동 궤적의 해류 비식별성" 소정리.

## 팩트체크 + novelty/contribution 리뷰 (사용자 요청, 재개 세션 말)

**팩트체크**: 모든 헤드라인 수치가 저장 JSON에서 재도출됨 (10-sea, G1,
HOS, orderprobe, 8-record 3-way, 14커밋). 표현 교정 2건:
① "net이 oracle_ls2@32 초과" → track043 +0.040 유효 / track100 +0.009
   = **동률**로 완화.
② 합성 이동 gain은 대표본 교정본(+0.069±0.033, net×고속 한정)이 공식.
추가 정직 노트: 실측 hand는 kp_hint(characterize) 사용, net은 내부
beamform kp — net이 불리한 kp 소스로도 이김 (주장 강화 + 개선 여지).

**novelty 위협 갱신**: ★AHGP(Liu et al., NeurIPS 2020) — 신경망이 데이터
→GP 커널 스펙트럼밀도 예측, 합성학습→실측 zero-shot까지 모듈1과 방법
골격 동일. **"amortized spectral prior" 자체는 우리 것이 아님 — 인용 필수.**
차별점으로 재배치: ⓐ분산-구속 시공간 커널(물리 다양체) ⓑtask-loss
(정확 선형해 관통; AHGP는 주변우도 계열) — v2 라벨회귀 실패/v3 성공
ablation이 이 차별점의 실증 ⓒ이동 연산자 분포 ⓓ해양 실측 검증.
PFN(Müller'22)·neural process 계열도 인용권. DSGP-Net 소스 미확인(리스크).

**contribution 구조 (재배치 후):**
- C1 [강]: 이동 sparse 파랑 추정의 병목 해부 (oracle 사슬 + FIM + 해류
  채널·비식별) — 조합 선행 없음.
- C2 [중, AHGP 인용 하 재프레임]: 해양파-특화 amortized spectral prior ×
  물리 MMSE, task-loss가 결정적이라는 ablation, 실측 zero-shot.
- C3 [강·도발적]: 추정기 사다리로 본 이동 프리미엄 증발 (corr −0.58→−0.11,
  8 record) — synthetic-aperture 서사 교정.
- C4 [소·깔끔]: sparse 이동 궤적 해류 비식별성 (5 목적함수 + oracle + 채널).
- C5 [보조]: UQ 근사-보정.
박사 골격: C1-C3 = 3챕터, C4/C5 보조. 리스크: 단일 프로토콜 패밀리
(Tp/패치/수심 ablation 필요), net 합성평가 in-distribution, BS11a B_r1
미해명, DSGP-Net 미추적.

**⑬ 거리별 스킬 분해 (사용자 "잘 안 맞는 것 같다" → 정직 정량,
`tmp/probe_zone_metrics.py`, 3 seas·track100):** 절대 기준 근접(0~0.15λ)도
corr 0.59/nRMSE 0.72 — "반쯤 맞는" 수준이 사실. 원거리(>0.6λ)는 oracle48
(참 파수)도 0.34, 전 추정기 ~0.07 = **관측예산의 정보 한계**. 0.6λ 이내에선
net ≈ oracle(0.59/0.57/0.41 vs 0.65/0.55/0.52) = 정보 소진 상태.
**논문 주장 형태 확정: "예측가능 반경(~0.6λ) 내 이론한계 근접 + 경쟁 대비
2~6배" (predictable-zone 프레임, Direction I 연결) — "절대적으로 잘 맞음"
주장 금지.** GIF: `results/gifs/ladder_track100_sea4.gif`.

**⑭ ablation + 예산 스윕 (오후 재개 세션 후반):**
- Tp/패치 ablation 통과(`tmp/probe_ablation_tp_patch.py`): Tp 5/7/9 net
  0.51~0.55(track)·0.37~0.38(fixed) 안정, 패치 1.3/2/3λ 전 셀 net>hand
  — k/kp 정규화의 스케일-공변 실증, 프로토콜-패밀리 과적합 리스크 폐쇄.
- **관측예산 스윕 = 무효 발견**(`tmp/probe_budget_sweep.py`): n_obs
  400→4800(12×)에 net 0.58→0.52(근접)/0.08→0.06(원거리) — 1D 궤적 위
  추가 표본은 커버리지 0. **3단 결론: ①반경내 net≈oracle(추정기 종결)
  ②표본수 무효 ③병목=관측 기하(커버리지)** → 다음 챕터 = 센싱 설계
  (교차 궤적·2D 스캔·footprint·순차 누적; d40 학습-acquisition 자산 연결).

## 남은 것 (내일 후보)

1. 실측 풀런 판독 + (양성이면) 나머지 8 record로 확장 — 특히 supra-0.7λ 음수
   6개 record가 MMSE로 뒤집히는지 (starvation 상관 −0.56 서사 재검).
2. **공간-분할 val**: track 세그먼트 holdout으로 EM/λ 게이팅 정렬 — 남은 이동
   상금(ceiling 0.41/0.46 vs 현 0.24)의 다음 레버.
3. HOS 비선형 재현 (상시지시): MMSE는 선형모델 — 비선형 바다에서의 열화 정량 필수.
4. 협대역 novelty 재검색 + (생존 시) 논문 프레이밍: "이동 플랫폼 파랑장 추정의
   병목 해부 + 처방" 포지션 페이퍼 각도.

## 산출물

- 스크립트: `scripts/d3_sa_nomp.py` (NOMP 계열 전 변형 + `bank_irls_estimate` +
  `mmse_bank_estimate`), `scripts/d3_sa_orderprobe.py`,
  `tmp/probe_kprecision.py`, `tmp/probe_mmse.py`
- 하니스 확장: `d3_sa_uspeed.run_window` estimator 스위치,
  `d3_sa_multirecord --estimator mmse`
- 결과: `results/direction3/sa_nomp.json`, `sa_orderprobe.json`,
  `sa_dps_moving_multiwin.json`, `sa_multirecord_mmse.json`(+smoke)
