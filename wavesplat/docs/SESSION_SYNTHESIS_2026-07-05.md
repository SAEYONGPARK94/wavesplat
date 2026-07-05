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

**④ 판정 기준 (사전 등록, 결과 보기 전):**
1. 1차: record별 Δabs(MMSE−RELAX), supra-임계 이동셀(B_r0.43·B_r1)과 A(fixed).
2. 2차: record-레벨 corr(gain, A_abs)=−0.56 구조가 MMSE에서 약화되는가
   (= "starvation"의 추정기 성분 비율).
3. 3차: MMSE 기준 이동 gain ≥ +0.1 생존 record 집합 (BS11a형 개구이득의 실재 범위).

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
