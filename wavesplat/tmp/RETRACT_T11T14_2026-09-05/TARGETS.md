# `RETRACT_T11T14_2026-09-05` — 철회·축소 표적 (그룹 C / 축 C3)

> 작성 2026-09-05. **런 0회 · 원고 무수정** (읽기 전용 사냥). 이 파일만 신설한다.
> 근거 = `tmp/T11_EXTACC_2026-09-03/verdict_t11.json`(+`AMENDMENT_3`) ·
> `tmp/T12_ISOLATE_2026-09-04/verdict_t12.json`(+`AMENDMENT_2`) ·
> `tmp/T13_DEPTHFIX_2026-09-04/verdict_t13.json`(+`AMENDMENT_2/3`) ·
> `tmp/T14_AUTOSCALE_2026-09-04/verdict_t14.json`(+`AMENDMENT_2/3`) ·
> `memory/reference_citation_bans.md` 맨 끝 `## 2026-09-04/05` 절 · 두 원고 전문 grep.
>
> ⚠**줄번호는 아래 판본 기준**이다(다른 에이전트가 편집 중 — 줄이 움직이면 **현재 문장**으로 찾아라):
> `paper/waveprior.tex` md5 `852c5bcf31c83eb9f20af3dfe84809d2` (3,851줄, HEAD `5b27a7e` 청정)
> `thesis/thesis_main.tex` md5 `5df9872f37f6562352357c6f625039c5` (11,029줄, HEAD `5b27a7e` 청정)
>
> ★**열람 범위 = 주장 범위.** 두 원고 전체를 정규식으로 훑고, 표적 후보 문단은 전문을 열었다.
> 열지 않은 것: 두 원고의 그림 파일·`paper/changes_appendix.tex` 본문 61줄 이후·`thesis/` 의
> 다른 tex 입력 파일. ⛔*"아마 있을 것"* 은 적지 않았다.

---

## 0. 오늘 판정 넷 — 원고에 넣을 수 있는 **정본 원자**와 동반 의무 (편집자용 요약)

| 배치 | 라벨 | 인용 가능 원자 (verdict 축자) | ⛔동반 의무 |
|---|---|---|---|
| `T11-EXTACC` (17/17, 96행, 24블록) | `BAND-MISMATCH` | `SHIP<ZERO` **8/24** p=0.968(무기술) · `RESCALE<ZERO` **24/24** p=5.96e-8 · `ORACLE<ZERO` **24/24** · 중앙 nRMSE ZERO .2476 / SHIP .2445 / RESCALE .1957 / ORACLE .1549 · MDE q80 **0.76** | 라벨 = **귀속**(우위/열세 아님) · `RESCALE` 은 **단위를 바꾼 사고실험**(성능·배포 아님) · 정본 읽기 = **"배포 추정기의 절대 스케일 설계 범위"** (⛔*"학습 사전의 파장 밴드"* — k_p 장면은 학습 밴드 83백분위 **안**) · ⛔`E2` 와 **대각선 비교** 금지 · ⛔저들 논문 수치 비교 금지 · ⛔solver 미명시(*"HOS"·"완전 비선형"* 금지) · ⚠Tp 전 파일 10 s = 한 주기 조건 · ⚠제안 단계 캡 400/8,000 · ⚠완전 blind 아님 · ⚠`d3_beamform_net.py:30 DEPTH=50` 하드코딩 = **비대칭 교락** 공시 |
| `T12-ISOLATE` (25/25, 424행) | `MULTIPLE-AXES-CARRY` | `A-MAPSCALE<ZERO` **24/24** · held-out **28/28** · 회수율 ρ 중앙 **0.507** · `A-MAPCELL` **19/24**(문턱 19 정확히) · 음성 6축 문턱 미달(SEC q80 **0.8332**) | 정본 = **"단일 서브시스템(조건화 맵)의 절대 스케일"** (⛔*"단일 상수"* · ⛔*"~때문"* 단일원인) · ⛔음성 축을 *"효과 없다/무죄/원인 아님"* 으로 · ⛔`C-MINDK` 널은 **6/24 다름** 병기 · ⛔ρ 팔 간 합산 · ⛔*"독립 확증·blind·out-of-sample"* · ⚠라벨이 한 블록 차로 뒤집힘(강건성 진술 병기) · ⚠처치 팔은 배포 구성이 아님 |
| `T13-DEPTHFIX` (24/24, 256행) | `FIX-UNDETECTED (WITHIN-PERTURBATION-FLOOR)` | `FIX<SHIP` **26/52** p=0.555 = **10/24(T12 행 재판독) + 16/28(신규)** · `SHIP<ZERO` 14/52 · 효과 −0.000148 vs 섭동 바닥 0.03664 · MDE q80 **0.6795** | ⛔*"수심은 원인이 아니다·교락은 무해·무효과 확증"* · ⛔⛔*"T12 와 독립적으로"* (절반이 비트 동일) · ⛔`FIX` 절대 nRMSE·배포 권고 · ⛔*"T11 BAND-MISMATCH 철회"* 로 읽기 |
| `T14-AUTOSCALE` (31/32, `blocked_by=['G-EQUIVAR']`) | **BLOCKED — 라벨 없음** | ✅단서형만: `G-INVERT` **0/176**(원 단위 원자를 뽑는 배포 가능 경로 작동) · `G-S-BLIND` **0/68** · `G-ANCHOR` T11→T14 비트 동일 | ⛔라벨·*"자동 스케일이 된다"* · ⛔`AUTO-AP` 28/28·회수율 1.634 (차단 배치 부분 통계) · ⛔*"규칙이 장면에 적응"*(s 는 sd=0 상수) · ⛔`AUTO-BAND` 는 `G-BAND-FRAGILE` 병기 없이 · ⛔스윕을 규칙의 증거로 |

**네 배치 공통**: ⛔층별(Hs·각도) 라벨 · ⛔`wall_s` 팔 간 비교 · ⛔T11 층별 수를 FRESH-28 과 직접 비교.
⚠**자료 정체**: Zenodo 17359600(*Phys. Fluids* 37, 117116; 원고 매크로 `\TOneCite{}` = `Zheng2025SyntheticWaveFields`)
`Data/mixed_data2`, 격자 128×128 · **L = 619.64 m** · Tp 10 s · k_p 0.030–0.040 rad/m(λ 157–209 m) ⟹ **개구/λ ≈ 3.0–3.9**
(실측 12 레코드의 1.0–4.2 안) · 조건화 맵 PSF **0.53칸**(학습 1.37–1.65칸).
⚠**원고 매크로 부재**: `paper/paper_constants.tex`·`scripts/gen_paper_constants.py`·`scripts/check_paper_disclosures.py` 에
`T11/T12/T13/T14/EXTACC/ISOLATE/DEPTHFIX/AUTOSCALE` 토큰 **0회**(grep) ⟹ ★새 수치는 **생성기를 거쳐 매크로로만** 넣고,
공시검사기에 위 동반 의무를 규칙으로 추가해야 한다(손타이핑 리터럴 금지 — `P8` 5차 감사 선례).

---

## 1. `BLOCKER` — **거짓이 된 문장** (3건 · 4지점)

### B-1 `paper/waveprior.tex:3405-3406` — *"what remains undone"* 이 이제 거짓
**현재 문장**
```
So the objection is answered in one direction and not the other: a reference field
built by someone else, from radar-imaging work with no connection to ours, carries the
phase coupling that a random-phase linear superposition cannot. What remains undone is
the accuracy comparison --- this test reads their ground truth, it does not run our
estimator on it.
```
**왜 거짓** — `T11` 이 **같은 아카이브**(`mixed_data2`, `T1`·`T1b` 와 서로소 24블록) 위에서 **출하 진입점**(`make_updater`,
`init_method=auto`, `G-WRAPPER`·`G-ARM` 되읽기)을 돌려 채점했다. 결과는 **무기술 8/24**(p=0.968).
⚠**단 "외부 기준선 팔" 은 여전히 0개다** — T11 에 제3자 *방법* 팔은 없다(§2 M-8 참조). 두 개념을 섞지 마라.

**제안 대체 문장** (수는 매크로로; 아래는 자구 초안)
```
So the objection is answered in one direction and not the other: a reference field
built by someone else, from radar-imaging work with no connection to ours, carries the
phase coupling that a random-phase linear superposition cannot. The accuracy
comparison on that ground truth has since been run (batch T-11, \TElevenN{} blocks
disjoint from the phase-coupling test): the shipped learned arm does not clear a
per-block zero field (\TElevenShipZero{}/\TElevenN{}, $p = \TElevenShipP{}$; MDE
$q_{80} = \TElevenMdeQ{}$), while an oracle given the true wavenumbers does
(\TElevenOracleZero{}/\TElevenN{}), so the information is in the observations. The
same arm on the same samples, with the coordinates relabelled by a
dispersion-preserving factor of two ($x' = x/2$, $t' = t/\sqrt{2}$), clears the zero
field on \TElevenRescaleZero{}/\TElevenN{}. That relabelling is a thought experiment
and not a configuration --- it changes the units, not the data --- and it attributes
the failure to the absolute-scale design range of the deployed estimator rather
than to the data; it does not make our method better or worse than anyone's, no
external method was run on this corpus, and we do not compare it diagonally with
the external reconstructor of Section~\ref{sec:res_scope}. The scene peak period,
$10$~s in every file, lies inside the prior's training band, so this is not a
statement about the prior's wavelength band; Section~\ref{sec:init} states which
constants move with the scale.
```
**처분** — 문장 교체 + 원문을 `%` 주석으로 보존(선례 `paper:2242-2248` E1 정정 형식) + `changes_appendix.tex` `tab:changes` 에
**R6 행** 신설(§4). ⚠같은 문단의 *"asked one question of it"*(`:3345`)도 함께 *"asked first one question"* 로 좁혀라(N곳 규칙).

### B-2 `thesis/thesis_main.tex:3302-3303` (+`:3267` *"질문은 하나다"*) — 같은 문장의 국문판
**현재 문장**
```
⛔남은 것은 정확도 비교다 --- 이 시험은 저들의 참값을 \emph{읽을} 뿐 그 위에서 우리
추정기를 돌리지 않았다.
```
(`:3267` *"질문은 하나다: \emph{그 표면에 위상 결합이 있는가 …}"* — 이제 질문이 둘이다.)
**제안 대체 문장**
```
★\textbf{[2026-09-04 갱신]} 그 정확도 비교는 이후 \textbf{같은 아카이브 위에서} 실행되었다
(배치 \texttt{T11}, 위상 결합 시험과 서로소인 $24$ 블록, 게이트 $17/17$): 출하 학습 팔은
블록별 제로필드를 넘지 못하며($8/24$, $p = 0.968$; 검출가능효과 $q_{80} = 0.76$), 참 파수를
받은 오라클은 넘는다($24/24$) ⟹ \textbf{정보는 관측에 있다}. 같은 표본을 분산관계를 보존한
채 좌표만 두 배로 재라벨($x' = x/2$, $t' = t/\sqrt{2}$)하면 같은 팔이 $24/24$ 로 넘는다.
⛔이 재라벨은 \textbf{단위를 바꾼 사고실험}이지 배포 구성이 아니고, 판정 \texttt{BAND-MISMATCH} 는
\textbf{귀속}(실패의 소재가 자료가 아니라 \emph{배포 추정기의 절대 스케일 설계 범위}에 있다)이지
우열이 아니다 --- 이 코퍼스에 외부 방법 팔은 없고, \S\ref{sec:init-bank} 의 제3자 재구성기와
\textbf{대각선으로 비교하지 않는다}. ⛔첨두주기가 전 파일 $10$\,s 로 학습 밴드 \emph{안}이므로
\emph{``학습 사전의 파장 밴드''} 로 읽지 않는다(어느 상수가 스케일과 함께 움직이는지는
\S\ref{sec:pipeline} 참조). ⚠제안 단계는 프레임당 $8{,}000$ 관측 중 캡 $400$ 만 쓴다.
```
`:3267` → *"질문은 \textbf{먼저} 하나였다: …"* 로.
**처분** — 인라인 정정 + 원문 `%` 보존(선례 `thesis:10195-10201`). N=2(논문 B-1 과 짝) — **둘을 같은 커밋에서**.

### B-3 `thesis/thesis_main.tex:7234-7235` — 전칭 *"유일한 out-of-family 게이트"* 가 거짓
**현재 문장**
```
\textbf{(7)}~★\textbf{실측 $12$ 레코드는 본 연구의 유일한 out-of-family 게이트이고, 이 배치가
그것을 한 번 더 소모했다.}
```
**왜 거짓** — `T11` 코퍼스(우리와 무관한 레이더 영상 연구의 참값, `T1`·`T1b`·감사 스모크와 서로소 24블록 + `T13/T14` 의
FRESH-28)가 **두 번째 out-of-family 게이트**로 이미 소모됐다(배포 팔 무기술 8/24 · 14/52 · 10/28, 세 시드 8·6·10/28).
⚠이 절(`M-7` 공시)은 실측 레코드에 대한 진술이므로 **"실측 자료 중 유일"** 로 좁히면 참이 유지된다.
**제안 대체 문장**
```
\textbf{(7)}~★\textbf{실측 $12$ 레코드는 본 연구가 보유한 \emph{실측} out-of-family 게이트의
전부이고, 이 배치가 그것을 한 번 더 소모했다.} ⚠2026년 9월 4일 이후에는 \emph{합성} out-of-family
게이트가 하나 더 있다 --- 제3자 참값 코퍼스(배치 \texttt{T11}; \S\ref{sec:forward}) --- 이며 그것도
이미 소모되었다(배포 팔 무기술 $8/24$, 후속 두 배치가 같은 $52$ 파일을 재사용).
```
**처분** — 인라인 정정. ⚠`\S\ref{...}` 는 B-2 정정문이 들어가는 절의 라벨로(3장 forward 모델 절 = `sec:forward` 인지 편집자가 확인).

---

## 2. `MAJOR` — **좁혀야 하는 문장** (7건 · 학위 4곳 동일 문구 1건 포함)

### M-1 `thesis:9957-9958` — *"무차원 파라미터를 직접 추정하므로 스케일 효과 제한적"* (수조 장)
**현재 문장**
```
\WS{}는 무차원 파라미터(파수, 분산관계)를 직접 추정하므로
스케일 효과의 영향은 제한적이다.
```
**왜 좁혀야 하나** — ①파수는 무차원이 아니다(rad/m). ②배포 학습 경로는 **절대 파수 상수**를 갖는다(조건화 맵 $|k|\le0.6$,
체크포인트 클립 $1.1$, `MIN_DK 0.15`, `dedup 0.03` — 원고 자신이 `thesis:3782,3824` · `paper:1117-1130` 에 적었다).
③**그 절대 스케일 의존이 오늘 개입으로 측정됐다**: T11 `SHIP` 8/24 → `RESCALE`(단위만 변경) 24/24; T12 조건화 맵
절대 스케일 하나를 절반으로 옮기면 24/24 · held-out 28/28 · ρ 0.507. ④수조 조건(표 `tab:tank_scaling` Tp 1.2 s, d 3.5 m)의
첨두 파수는 $k_p = (2\pi/1.2)^2/9.81 \approx 2.8$ rad/m — 맵 상한 0.6 의 **4.7배**, 클립 1.1 의 **2.5배** ⟹ 학습 제안기는
수조 첨두를 **표현조차 못 한다**. ⑤단 수조 입력은 참값이 없어 실측 게이트가 bank 경로로 보낸다(`thesis:9332`) — 그 경로는
$\hat k_p$ 상대 뱅크이나 **그 스케일 불변성은 T14 가 재지 않았고 인용 금지**다(T14 disclosure 10).
**제안 대체 문장**
```
⚠\textbf{스케일 효과는 팔에 따라 다르며, 제안 표현에 대해서는 ``제한적''이 아니다.}
파수는 무차원이 아니고(rad/m), 배포 학습 경로는 절대 파수 상수(조건화 지도 $|k| \le 0.6$\,rad/m,
체크포인트 클립 $1.1$\,rad/m, \S\ref{sec:pipeline})를 가지므로 스케일에 불변이 아니다 --- 제3자
참값 코퍼스에서 좌표를 두 배 재라벨하는 것만으로 같은 팔이 무기술($8/24$)에서 $24/24$ 로 바뀌고,
조건화 지도의 절대 스케일 하나를 절반으로 옮기면 $24/24$(held-out $28/28$, 회수율 중앙 $0.507$)이다
(배치 \texttt{T11}$\cdot$\texttt{T12}, \S\ref{sec:forward}). 수조 조건(표~\ref{tab:tank_scaling})의
첨두 파수 $k_p \approx 2.8$\,rad/m 는 그 지도 상한의 약 $4.7$ 배 \emph{밖}이다. ⚠단 수조 LiDAR 입력은
참값이 없어 위 설계 경고대로 bank-MMSE 경로로 가며, 그 경로는 추정된 $\hat k_p$ 에 상대적인 뱅크를
쓰므로 스케일 의존이 다를 수 있다 --- \textbf{그것은 측정되지 않았다}(자동 정규화 배치
\texttt{T14} 는 자기 검증 게이트에서 차단되어 라벨을 내지 않았다).
다만, LiDAR 거리 해상도($\sim$1\,cm)와 파고(수\,cm)가 같은 차수이므로 …
```
**처분** — 인라인 교체 + 원문 `%` 보존. ⛔*"자동 스케일이 된다"*·`AUTO-AP` 수치 금지.

### M-2 `thesis:3779-3780` — *"어떤 이산 격자에도 묶이지 않는다"*
**현재 문장**
```
집합 표현을 쓰는 이유는 격자 비의존성이 여기서 시작되기 때문이다 --- 원자의 파수는
연속값이며, 어떤 이산 격자에도 묶이지 않는다.
```
**왜 좁혀야 하나** — 표현(원자 좌표)은 연속이나, **그 표현을 채우는 제안은 조건화 지도의 절대 스케일(칸 크기 $1.2/64$ rad/m ·
$|k|\le0.6$ · 사전 밴드 $1.1$)에 묶여 있음이 개입으로 측정됐다**(T12: 그 스케일 하나를 옮기면 24/24; 매개열: `SHIP` 제안 원자
중앙 파수 0.449 vs 참 0.0415 = **10.4배**, `A-MAPSCALE` 0.078 = 2.45배 — ⛔기제 지목 금지, 기술통계). 오늘 원고의 문장은
독자에게 *"격자에 안 묶이므로 스케일에도 안 묶인다"* 로 읽힌다.
**제안 대체 문장**
```
집합 표현을 쓰는 이유는 격자 비의존성이 여기서 시작되기 때문이다 --- 원자의 파수는
연속값이며, 출력이 어떤 이산 격자에도 묶이지 않는다. ⚠\textbf{단 입력은 묶여 있다}: 제안은
아래 조건 입력(고정 칸의 $64\times64$ 빔포밍 지도)과 사전 밴드의 \emph{절대} 스케일에 의존하며,
그 의존은 제3자 참값 코퍼스에서 개입으로 측정되었다(단일 서브시스템의 절대 스케일 하나를
절반으로 옮기면 무기술 $8/24$ 가 $24/24$ 로; 배치 \texttt{T12}, \S\ref{sec:forward}).
연속 파수는 \emph{위치 정밀도}의 성질이지 \emph{스케일 불변}의 성질이 아니다.
```
**처분** — 인라인 추가(교체 아님).

### M-3 `thesis:3828-3830` + `:3841-3843` — 수심 50 고정 공시와 *"이식에서 조건 입력이 먼저 어긋난다"*
**현재 문장**
```
⚠\textbf{조건 인코더의 분산관계는 수심 $50$\,m 에 고정되어 있다}
(\texttt{d3\_beamform\_net.py} 의 모듈 상수). 관측된 수심은 이후의 물리 정련 단계에만
전달된다.
…
\textbf{다른 수심으로의 이식에서는 조건 입력이 먼저 어긋난다.}
```
**왜 좁혀야 하나** — 구조 서술(고정·미전달)은 **참이고 T11 이 재확인**했다(`d3_beamform_net.py:30`, config depth 가 지도에 0회 도달,
학습 코퍼스도 d=50 ⟹ `SHIP` 만 디포커스 = 비대칭 교락). 그러나 *"먼저 어긋난다"* 는 **기제의 우선순위 주장**이고, 그 수리를
개입으로 재자 **검출되지 않았다**: T12 `E-MAPDEPTH` 8/24 · T13 `FIX<SHIP` 26/52 (= 10/24 재판독 + 16/28 신규), 효과가 자기
섭동 바닥의 1/247. 반면 스케일 축은 24/24. ⛔*"수심은 원인이 아니다"* 는 금지(검정력 진술, MDE q80 0.6795) — 그래서 **삭제가
아니라 병기**다.
**제안 대체 문장**
```
\textbf{다른 수심으로의 이식에서는 조건 입력이 어긋난다} --- 다만 그 어긋남이 \emph{먼저}
작용한다고는 쓰지 않는다. 수심 $400$\,m 의 제3자 참값 코퍼스에서 이 상수를 자료의 수심으로
바꾼 개입은 검출 문턱에 이르지 못했고($26/52$ = 선행 배치 행의 재판독 $10/24$ + 신규 $16/28$,
효과가 자기 섭동 잡음 바닥 안; 배치 \texttt{T13}, ⚠검출가능효과 $q_{80} = 0.6795$ 이므로
\emph{``수심은 원인이 아니다''} 로 읽지 않는다), 같은 코퍼스에서 조건 입력의 \emph{절대 스케일}을
옮긴 개입은 $24/24$ 였다(배치 \texttt{T12}). ⛔두 배치는 독립이 아니다 --- 앞의 $24$ 행이 비트
동일하다.
```
**처분** — `:3841-3843` 교체 + `%` 보존. ⚠`paper` 에는 이 공시 자체가 없다(§5 D-1).

### M-4 `paper:1117-1136` — 상한(0.6/1.1)만 적은 *"bounded region"* 문단
**현재 문장(발췌)**
```
The learned proposal cannot place an atom outside a bounded region: …
$|k_x|, |k_y| \le \DepKLim{}$~rad/m. For the sea states evaluated here the largest true
wavenumber is near $1.0$~rad/m, so the bound is satisfied with little margin, and it
is a property of the checkpoint rather than of the representation …
A tighter bound applies upstream: … that grid spans $\pm 0.6$~rad/m. … Any application
to shorter waves would need the prior retrained, and we have not measured what happens
when a true component lies outside the bound.
```
**왜 좁혀야 하나** — 이 문단은 **위쪽** 경계만 다루고 *"밖은 미측정"* 이라 적는다. T11/T12 는 **안쪽**($k_p \approx 0.03$–$0.04$,
상한의 1/15, 학습 Tp 밴드 83백분위 안)에서 배포 팔이 무기술이고, 지도 칸·사전 밴드·`MIN_DK`·성장 DC블록·VARPRO lr 가 **스케일과
공선**으로 움직이는 **절대 스케일 설계 범위**가 양쪽으로 닫혀 있음을 개입으로 보였다(⛔단일 상수 귀속 금지).
**제안 추가 문단** (기존 문단 뒤)
```
The bound is not only an upper one. On a third-party ground-truth corpus whose peak
wavenumber sits near $0.04$~rad/m --- inside every bound above, and at a peak period
inside the prior's training band --- the shipped learned arm has no skill against a
zero field (\TElevenShipZero{}/\TElevenN{}), while relabelling the coordinates by a
dispersion-preserving factor of two restores it (\TElevenRescaleZero{}/\TElevenN{};
batch T-11, a thought experiment, not a configuration). Moving the absolute scale of
one subsystem alone --- the conditioning grid and the band it feeds, halved --- restores
it on \TTwelveMapScale{}/\TTwelveN{} of the same blocks and \TTwelveHeldOut{}/\TTwelveHeldN{}
held-out blocks, recovering a median \TTwelveRho{} of the relabelling gain (batch T-12;
a second axis also crosses its threshold, so no single constant is named). What is
outside the training distribution in that corpus is not the wavelength but the ratio of
observation aperture to conditioning-grid cell (\TElevenPsfShip{} cells against
\TElevenPsfTrain{} in training). We therefore describe the deployed estimator as having
an absolute-scale design range, closed at both ends, and we have not measured a rule
that selects that scale from observations alone (batch T-14 was blocked by its own
equivariance gate and carries no label).
```
**처분** — 추가. ⛔`E-MAPDEPTH`·`F-STEEP` 에 ρ 인쇄 금지 · ⛔`C-MINDK` 널 인용 시 6/24 병기.

### M-5 `paper:946-948` — 도너 기하 14 = *"generalisation limit rather than defend it"*
**현재 문장**
```
That bank of \TrainDonors{} geometries is small, and it is the only view of sensor
geometry the prior ever had; we note it as a generalisation limit rather than defend it.
```
**왜 좁혀야 하나** — 그 한계가 이제 **한 방향에서 측정됐다**: 관측 개구 **619.64 m**(학습 220.55 m)·균등 200점/프레임인 제3자
코퍼스에서 배포 팔 무기술(8/24), 지도 PSF 0.53칸(학습 1.37–1.65). *"note it rather than defend it"* 는 미측정을 함의한다.
**제안 대체 문장**
```
That bank of \TrainDonors{} geometries is small, and it is the only view of sensor
geometry the prior ever had. It is a measured limit, not only a noted one: on a
third-party corpus whose observation aperture is \TElevenAperture{}~m against the
\TrainAperture{}~m of the training pool, the shipped arm has no skill until the
absolute scale is moved (Section~\ref{sec:init}, batch T-11/T-12).
```

### M-6 `thesis:683 · :1444-1445 · :1589 · :7284` — *"개구는 처치가 아니고(두 코퍼스가 λ 단위로 같은 자리)"* (**4곳 동일 문구**)
**현재 문장**(`:683`, `:1589` 축자 동일; `:1445`, `:7284` 는 축약형)
```
⚠\textbf{[2026-09-03]} 그 셋의 지위는 이후 좁혀졌다 --- \textbf{개구}는 처치가 아니고(두
코퍼스가 $\lambda$ 단위로 같은 자리), \textbf{폐색}은 …
```
**왜 좁혀야 하나** — **λ 단위로는** 참이다. 그러나 T11 은 개구/λ ≈ 3.0–3.9(실측 1.0–4.2 안)를 **고정한 채** 절대 단위만
바꿔 8/24 → 24/24 를 냈고, 그 절대량이 곧 **조건화 지도의 개구/픽셀 비**다(T11 AMENDMENT_2 §6, T14 규칙 `AP` 의 정의).
⟹ *"개구는 처치가 아니다"* 는 *"λ 단위 개구는 처치가 아니다; 절대 단위 개구(지도 칸 대비)는 처치이고 제3자 코퍼스에서 기술을
켠다"* 로 좁혀야 한다. ⚠실측 레코드는 방향이 **반대**(개구 12–240 m < 학습 220 m ⟹ PSF 가 학습보다 **넓다**)이고 그 방향은
**미측정** ⟹ ⛔소거표에 넣지 마라(다른 코퍼스, 제로필드 대비, 기준선 대비 아님).
**제안 대체 문장** (4곳 전부)
```
\textbf{개구}는 $\lambda$ 단위로는 처치가 아니고(두 코퍼스가 같은 자리) --- ⚠단 \emph{절대} 단위
(조건화 지도의 칸 대비 개구)로는 처치이며, 제3자 참값 코퍼스에서 그 절대 스케일을 옮기는 개입이
기술을 켠다($8/24 \to 24/24$; 배치 \texttt{T11}$\cdot$\texttt{T12}). 실측 레코드의 개구는 학습
개구보다 \emph{작은} 쪽이어서 방향이 반대이고 그 방향은 미측정이다 ---
```
**처분** — 4곳 동시 정정(⛔N−1 금지). `:683`/`:1589` 는 초록 미러라 **영문 초록 대응 문장**도 확인 필요(영문에 *"aperture is not a
treatment"* 가 있는지 이 사냥에서는 확인하지 못했다 — 편집자가 `thesis:900-1100` 을 grep 할 것).

### M-7 *"합성 = 양성 / 실측 = 음성"* 이분법 — `paper:3172-3175` · `paper:3745-3748` · `thesis:10160-10230(결론 「실측 자료」 목록)`
**현재 문장**
- `paper:3172-3175` — *"The positive results are synthetic … Every margin reported in Section 5.1 and its ablations is measured on
  simulated fields with a simulated sensor. Measured data enters this paper through the negative …"*
- `paper:3745-3748` — *"the representation that earns its gain on crowded synthetic seas has not been shown to earn it on the measured
  data we hold"*
- `thesis:10160-10163` — *"결론은 다섯이다"* (실측 음성 목록)
**왜 좁혀야 하나** — 문장 자체는 참이나, **합성 자료 중 우리가 만들지 않은 것**에서 두 번째 음성이 생겼다: 배포 팔이 제3자 참값
(Tp 10 s, 학습 밴드 안)에서 무기술 8/24 · 전수 52 에서 14/52 · FRESH-28 세 시드 8·6·10/28, 그리고 정보는 관측에 있다(오라클 24/24).
독자가 *"합성이면 된다"* 로 읽는 것을 막아야 한다. ⛔단 이 음성은 **기준선 대비가 아니라 제로필드 대비**이고 관측 기하가 다르므로
헤드라인(격자 기준선 대비 혼잡 우위)을 **철회하지 않는다** — 척추 *"경계는 파랑 조건 축을 따라 놓여 있지 않다"* 와도 **정합**
(Tp 가 학습 밴드 안인데 실패 ⟹ 파랑 조건 축이 아니다).
**제안** — `paper:3175` 뒤: *"Synthetic data we did not generate enters through a second negative: on a third-party ground-truth
corpus the shipped arm has no skill against a zero field at its shipped scale (Section~\ref{sec:init}), and the failure is attributed
to the estimator's absolute-scale design range rather than to the data."* · `paper:3746` *"on the measured data we hold"* →
*"on the measured data we hold, nor on a third-party synthetic corpus at the shipped scale"*. · `thesis` 결론 목록에 **여섯째 항목**
추가(B-2 정정문과 같은 원자·같은 공시; *"결론은 여섯이다"* 로 개수 표기 동시 갱신 — 이 저장소의 「개수가 목록과 어긋나는」 부류 방지).
⚠초록(`paper:118-151`)은 250단어 관례라 **선택**: 넣는다면 마지막 문장에 *"and on a third-party synthetic corpus it has no skill at
its shipped scale until the scale is moved"* 한 절 — 공시검사기 규칙(대각선 금지·RESCALE 사고실험·MDE) 이 초록에 걸리므로
`P8` 5차 감사 방식대로 **초록에는 의무 수치를 넣지 않는** 편이 안전하다.

### M-8 ✅**유지 — 고치지 마라** (편집자가 오늘 판정을 보고 잘못 "정정"할 위험이 있는 참 문장)
| 위치 | 문장 | 왜 여전히 참 |
|---|---|---|
| `paper:2253-2256` | *"The learned prior … has still never been compared against an external method"* | T11–T14 에 외부 **방법** 팔은 0개. T11 은 외부 **참값** 위의 자기 팔 비교다 |
| `thesis:508-509` · `:894-895` · `:1591-1598` · `:2172-2180` · `:10195-10201` | *"제안 표현은 여전히 외부 방법과 비교된 적이 없다"* / E1 정정 항목 | 동일 |
| `thesis:496` · `:1696` · `:6160` · `:6417` · `:6499` | *"외부 기준선 행이 $0$개"* (각 절제 배치 한정) | 그 배치들의 사실 |
| `thesis:1370-1378 §boundary` 척추 · `paper:2765-2772` | *"경계는 파랑 조건 축을 따라 놓여 있지 않다"* / four candidates measured | T11 은 격자 기준선 팔이 없고 제로필드 대비이며, Tp 가 학습 밴드 안에서 실패 = 파랑 조건 축 **아님**과 정합. ⛔T11 을 `tab:boundary`(실측 12 레코드 전용 표)에 넣지 마라 |
| `thesis:9702-9704` (수조 장) | *"진단된 실패 기제는 관측 개구가 파장의 1–5배 …"* | 주어가 **ETKF 기준선**이다. T11 에 ETKF 팔이 없어 무관 |
| `thesis:1342-1346` | *"실측 입력은 예외다 --- 실측 게이트가 bank-MMSE 로"* | T11 은 상수 `gt_field` 를 붙여 게이트를 **끄지 않고** 학습 경로에 도달했다(`G-NOLEAK`·`G-ARM`). 게이트 서술 불변 |

---

## 3. 오늘 금지 자구가 **기존 원고에 이미** 있는가 — grep 결과

`reference_citation_bans.md` 맨 끝 절의 금지 자구를 두 원고 전체에 걸었다(대소문자 무시, 주석 포함).

| 금지 자구 | paper | thesis | 판정 |
|---|---|---|---|
| 단일 상수 / single constant | 0 | 0 | — |
| 학습 사전의 파장 밴드 / wavelength band of the prior | 0 | 0 | — |
| 자동 스케일 / auto-scale / automatic(ally) scale·select / 자동 정규화 / self-normali | 0 | 0 | — |
| 수심은 원인이 아니 / 교락은 무해 / depth is not the cause / confound is harmless | 0 | 0 | — |
| 효과 없 / 무죄 / 원인 아님 | 0 | 0 | — |
| 규칙이 장면에 적응 | 0 | 0 | — |
| 대각선 / diagonal | 2 (`:1955`,`:2079` 그림의 대각선) | 1 (`:2751` 격자 대각선) | **문맥 무관** (기하) |
| 완전 비선형 / fully non-linear | 1 (`:3357`) | 1 (`:3303`) | ✅**준수 형태** — 둘 다 *"we do not call the corpus … fully non-linear"* / *"부르지 않고"* 부정문 |
| HOS | 85 | 56 | 전부 **우리** HOS M=3 코퍼스 지칭. 제3자 코퍼스를 HOS 라 부른 곳 **0** (`paper:3356-3357`, `thesis:3303` 이 명시적으로 거부) |
| 독립 확증 | 0 | 14 | 전부 `B-20b`/`M8b` 맥락의 **금지 선언·부정문**(`:482`,`:1958`,`:4848`,`:6270-6272`,`:7392`,`:10363` + 주석). T12/T13 지칭 0 |
| blind / out-of-sample | 18 / 8 | 29 / 1 | 전부 타 배치의 *"not blind"* 공시 또는 *"seed axis only"* 한정. T12 지칭 0 |
| 인수분해되지 않 / 상호작용 / 도달 불가 | 0 / 0 / 1(주석) | 10 / 13 / 19 | `M8b`·`D3` 등 **08-29~09-01 묶음**의 자구(오늘 목록이 아니라 그때 목록). 오늘 판정 지칭 0 |
| `T11·T12·T13·T14·EXTACC·ISOLATE·DEPTHFIX·AUTOSCALE·BAND-MISMATCH·MULTIPLE-AXES·FIX-UNDETECTED·mixed_data` | 0 | 0 | 원고 반영 **0회** (bans 파일의 09-05 grep 재확인) |

⟹ **오늘 판정을 가리키는 형태의 금지 자구 = 0건.** 준수 형태(부정문) 2건. ⚠편집 시 주의: `paper:2270-2300` 의 `E2` 문단(*"their
method on our corpus"*)은 T11(*"our method on their corpus"*)과 **2×2 를 이룬다** — 원고가 그 표를 그리게 되면 ⛔**대각선 비교 금지**
(T11 PREREG §6 의무 공시)를 같은 절에 적어야 공시검사기 부류의 위반이 안 생긴다.

---

## 4. 철회 표시의 관례와 **이동 위치** 제안

- 저장소 원칙: *"철회는 삭제가 아니라 이동"* (`thesis:10973` 선언 · `paper:1816` 주석). 선례 `P8`(2026-09-01): **절 단위**는
  `tmp/PAPER_LEGACY_2026-09-01/retracted_{results,discussion,tables}_*.tex` 로 이동; **문장 단위**는 인라인 정정 + 원문 `%` 주석 보존
  (`paper:2242-2248` E1, `thesis:10195-10201`); 논문은 `changes_appendix.tex` `tab:changes` 에 **번호 행**(R1–R5), 학위논문 초록 블록은
  `app:abstract_ledger`.
- **오늘 표적은 전부 문장 단위**다 ⟹ 이동할 절이 없다. 처분:
  1. `paper`: B-1·M-4·M-5·M-7 은 인라인 + 원문 `%` 보존, 그리고 **`tab:changes` 에 R6 행 신설** —
     *"R6 | 'What remains undone is the accuracy comparison' (2026-09-03) | Run since (T-11): the shipped learned arm has no skill on
     that ground truth (8/24); attributed by intervention to the estimator's absolute-scale design range, not to the data. No external
     method arm exists on that corpus."* 본문 포인터 `App.~\ref{app:changes}, R6` 을 B-1 자리에.
  2. `thesis`: B-2·B-3·M-1·M-2·M-3·M-6 인라인 + `%` 보존(`% ⛔[2026-09-05 정정] 종전 문안 …` 형식). ⚠`app:abstract_ledger` 는
     초록 전용이므로 쓰지 않는다. **M-1(수조 장 문단)을 통째로 뺀다면**만 이동: `tmp/PAPER_LEGACY_2026-09-01/` 와 같은 규약으로
     `tmp/RETRACT_T11T14_2026-09-05/moved/thesis_tank_scaling_2026-09-05.tex` 에 원문+사유 보존.
- ⚠**래칫**: 양방향 래칫(`P8`)이 *"개선이 기록되지 않았다"* 로 막을 수 있다 — 기준선을 내릴 때 사유를 코드 주석에 적는 선례를 따를 것.

---

## 5. 두 원고 간 **불일치**

| # | 논문 | 학위논문 | 성격 |
|---|---|---|---|
| D-1 | `paper:852` 알고리즘 표 3단계 *"Conditioning map: 64×64 … $\omega$ from the dispersion relation"* — **수심 50 m 고정·config 미전달 공시 없음** (grep: `hard-coded`/`module constant`/`d=50` 이 조건화 문맥에 0회) | `thesis:3828-3830` 이 **공시함** | ⚠**논문만 결손**. T11 이 이를 비대칭 교락으로 공시했고 T13 이 그 수리를 쟀으므로 이제 **하중 있는 사실** ⟹ `paper:852` 에 *"($d$ fixed at 50 m in the map; the configured depth does not reach it --- batch T-13 measured the repair as undetected)"* 추가 |
| D-2 | `paper:3405-3406` *"undone"* | `thesis:3302-3303` *"돌리지 않았다"* | **같은 거짓 문장 2곳**(B-1·B-2) — 같은 커밋에서 |
| D-3 | 논문에 *"유일한 out-of-family"* 전칭 **없음**(`paper:3342-3345` 는 T1 용도만 서술) | `thesis:7234` 전칭 **있음** | 학위논문만 결손(B-3) |
| D-4 | `paper:1128-1136` 이 조건화 지도 ±0.6 을 *"the only view of the data the network receives"* 로 **구속 상한**이라 명시 | `thesis:3780` *"어떤 이산 격자에도 묶이지 않는다"* 가 `:3824` 의 0.6 서술과 **연결 없이** 서 있다 | M-2 로 해소 |
| D-5 | 제3자 코퍼스 서술: `\TOneNMat{}`=126 · 12.8 GB · *"radar-imaging work"* | 126개×502 · 12.8 GB · *"레이더 영상 연구"* | ✅일치 |
| D-6 | `paper:3172-3177` 실측 진입 지점 **셋** 열거 | `thesis:10160` 실측 결론 **다섯** | 범위가 달라 불일치 아님. M-7 이 양쪽에 같은 원자를 넣으면 **개수 표기 동시 갱신** 필요 |

---

## 6. 집계

| 구분 | 건수 | 지점 |
|---|---|---|
| ★거짓이 된 문장 (`BLOCKER`) | **3건 / 4지점** | `paper:3405-3406` · `thesis:3302-3303`(+`:3267`) · `thesis:7234-7235` |
| ★좁혀야 하는 문장 (`MAJOR`) | **7건 / 13지점** | M-1 `thesis:9957` · M-2 `thesis:3780` · M-3 `thesis:3841-3843` · M-4 `paper:1117-1136` · M-5 `paper:946-948` · M-6 `thesis` 4곳 · M-7 `paper:3172-3175`,`:3745-3748` + `thesis` 결론 목록 |
| 두 원고 간 불일치 | **3건**(D-1·D-3·D-4; D-2 는 B-1/B-2 와 동일) | |
| 기존 원고에 이미 있는 **오늘** 금지 자구 | **0건** (준수 형태 부정문 2건 · 타 배치 묶음 자구 다수는 오늘 판정 미지칭) | |
| 유지해야 할 참 문장(오정정 위험) | 6묶음 (M-8) | |

⛔**이 파일은 지시서다. 원고는 이 세션에서 한 글자도 수정하지 않았다.**

---

# 부록 A — **동시 편집 상태 반영** (2026-09-05, 2차 열람)

> §0–§6 은 청정 HEAD `5b27a7e`(md5 `852c5bcf…`/`5df9872f…`) 기준이다. 그 뒤 다른 두 에이전트가
> 두 원고에 **+458줄**을 넣었다. 이 부록은 아래 판본 기준이며 **내용 검색**으로 재확인했다
> (줄번호는 또 움직인다 — 문장으로 찾아라):
> `paper/waveprior.tex` md5 `e4c7877969a58f2986e15107065d2b15` ·
> `thesis/thesis_main.tex` md5 `758d7977fba737ca691207dc6f2736e6`.
> ⛔이 세션은 여전히 두 원고에 **한 글자도 쓰지 않았다**.

## A.1 §1–§5 표적의 현재 상태

| 표적 | 상태 (2차 열람) | 비고 |
|---|---|---|
| **B-1** `paper` *"What remains undone…"* | ✅**정정됨** — 원문 `%` 보존 + *"This test reads their ground truth; it does not score our estimator against it. That comparison has since been run … §sec:res_extacc"* | 준수 |
| **B-2** `thesis` *"돌리지 않았다"* | ✅**정정됨** — `% ★CORRECTED 2026-09-05` + *"그 정확도 비교를 했다 … 무기술 $\TElevenShip$"* + solver 미명시·검출률 비교 금지 병기 | 준수 |
| **B-3** `thesis` *"유일한 out-of-family"* | ✅**정정됨** — *"실측 out-of-family 게이트의 전부"* + *"합성 out-of-family 게이트가 하나 더"* + ⛔계수 직접 비교 금지 | 준수 |
| **M-1** `thesis` 수조 *"무차원 파라미터"* | ✅**정정됨** — 파수 단위·절대 상수·T11/T12 개입·수조 첨두 파수가 지도 상한 밖·bank 경로 스케일 의존 **미측정**·T14 차단 전부 병기 | 준수 (내 §2 M-1 초안과 동형) |
| **M-2** `thesis` *"어떤 이산 격자에도"* | ✅**좁혀짐** — *"\textbf{출력이} 어떤 이산 격자에도 묶이지 …"* | 준수 |
| **M-3** `thesis` *"조건 입력이 먼저 어긋난다"* | ✅**정정됨** — *"어긋난다 --- 다만 먼저 작용한다고는 쓰지 않는다"* + 26/52 분해·바닥·MDE·⛔수심 원인 아님 부정·⛔독립 아님 | 준수 |
| **M-4** `paper` *"we have not measured what happens when a true component lies outside the bound"* (`:1143`) | ⛔**미정정 · 생존** | §2 M-4 추가 문단 그대로 유효 |
| **M-5** `paper` *"we note it as a generalisation limit rather than defend it"* (`:955`) | ⛔**미정정 · 생존** | §2 M-5 대체 문장 그대로 유효 |
| **M-6** `thesis` *"개구는 처치가 아니고"* 4곳 | 🔄**진행 중** — `:689` 에 `% ★NARROWED 2026-09-05 (T11 + T12)` 주석 + `:1513` *"개구는 $\lambda$ 단…"* 로 좁혀짐 확인. `:1658`·`:7414` 는 이 md5 에서 첫 70자만 봤다(*"[2026-09-03] 그 셋의 지위는…"*) ⟹ **N=4 전부 좁혀졌는지 편집자가 확인** | ⛔N−1 주의 |
| **M-7** 이분법 | 🔄**부분** — ✅`paper` 결론 *"A second boundary, this one with a mechanism attached"* 신설 · ✅`paper` `tab:boundary` 뒤 *"A second negative … the two must not be merged"* · ✅`thesis` 결론 *"결론은 여섯이다"*(T11 4회 언급) · ⛔**`paper` §Limitations *"The positive results are synthetic"* 문단은 그대로**(third-party 언급 0회) | 남은 1곳 = §2 M-7 첫 항목 |
| **D-1** `paper:859` 알고리즘 표 3단계 수심 미공시 | ⛔**미정정 · 생존** | *"($d$ fixed at 50 m inside the map; the configured depth does not reach it)"* 추가 |
| **M-8** 유지 목록 | ✅전부 그대로 참 — 새 절도 *"외부 기준선 행 0개"*·*"실측 12 레코드에 대해서는 아무것도 말하지 않는다"* 를 명시 | — |

## A.2 ★동시 편집이 **새로 들여온** 문장 — 사냥 결과 (인쇄면 3건 MAJOR · 1건 MINOR)

### N-1 `MAJOR` — 학위 §T11 문단 **제목**이 금지된 읽기를 유도한다 (3곳)
| 위치 | 현재 문장 |
|---|---|
| `thesis:7757` | `\paragraph{★\texttt{T11-EXTACC} --- 실패는 우리 대역이고 자료가 아니다}` |
| `thesis:7767` | `★\textbf{좌표를 재라벨해 저들 바다를 우리 대역으로 옮기면 기술이 나타난다}` |
| `thesis:1679` (§boundary 신설 문단) | *"좌표만 분산관계 보존 재라벨로 우리 대역에 옮기면 기술이 나타나고"* |

**왜** — *"우리 대역"* 은 `T11 PREREG` v2 의 초판 자구(`BAND-MISMATCH` = *"실패는 우리 대역"*)인데, 결과 후 `AMENDMENT_2 §6`
이 그 읽기를 **철회**했다: 재척도는 단일요인이 아니고 k_p 장면은 학습 밴드 **83백분위 안**이므로 정본 = **"배포 추정기의 절대 스케일
설계 범위"**, ⛔*"학습 사전의 파장 밴드"* 금지. 같은 문단 **본문**은 이미 그렇게 적고 있다(`:7772-7775`) — **제목이 본문과 어긋난다**
(`T4 §8-3` 선례: *"제목이 금지된 읽기를 유도 ⟹ 제목 완화"*).
**제안** — `:7757` → `\paragraph{★\texttt{T11-EXTACC} --- 실패는 배포 추정기의 절대 스케일 설계 범위이고 자료가 아니다}` ·
`:7767`·`:1679` → *"… 재라벨해 배포 추정기의 절대 스케일 설계 범위 안으로 옮기면 기술이 나타난다"*. **3곳 동시**.

### N-2 `MAJOR` — 학위 §T13 **제목·서사** *"무엇 때문이 아닌가"* (인쇄 2곳 + 주석 1)
| 위치 | 현재 문장 |
|---|---|
| `thesis:7752` | *"\emph{무엇 \textbf{때문이 아닌가}}(\texttt{T13})"* (사다리 서사) |
| `thesis:7810` | `\paragraph{★\texttt{T13-DEPTHFIX} --- 무엇 \emph{때문이 아닌가}}` |
| `paper:2460` (주석) | *"→ **무엇 때문이 아닌지**(T13)"* |

**왜** — `T13` 라벨은 `FIX-UNDETECTED` 이고 verdict 공시 4 가 축자로 ⛔*"수심은 원인이 아니다·교락은 무해하다 로 읽지 마라 —
검정력의 진술"* 이다. *"무엇 때문이 아닌가"* 는 정확히 그 금지된 읽기(원인 배제)를 제목으로 건다. 본문(`:7823-7826`)은 준수하나
제목이 되감는다. ⚠주석도 N 에 든다(원장 규칙 *"「N곳 전부」의 N 에는 주석이 포함된다"*).
**제안** — `:7752` *"무엇을 고쳐도 검출되지 않는가(\texttt{T13})"* · `:7810` `\paragraph{★\texttt{T13-DEPTHFIX} --- 수리가 검출되지 않은 후보}` ·
`paper:2460` 주석 *"→ **수리가 검출되지 않는 것**(T13)"*. **3곳 동시**.

### N-3 `MAJOR` — 영문 초록 *"the scale responsible for that"* = 단일원인 귀속
| 위치 | 현재 문장 |
|---|---|
| `thesis:1145-1147` | *"makes skill appear ($\TElevenPrimary$), and the scale **responsible for** that is reproduced inside a \emph{single subsystem}, the absolute scale of the conditioning map"* |

**왜** — `T12` 라벨은 `MULTIPLE-AXES-CARRY`(두 축이 각자 문턱 초과, ρ 0.507 = 절반), 금지 = ⛔*"제안 스케일 **때문**"* 단일원인 귀속.
*"responsible for"* 는 그 자구의 영역(英譯)이다. 국문 미러(`:704` *"그 배율이 **사는 곳은** …"*)는 준수 ⟹ **두 초록이 어긋난다**(미러 규칙).
**제안** — *"and the scale that relabelling stands in for is recovered inside a single subsystem, the absolute scale of the conditioning map, though not there alone"*.

### N-4 `MINOR` — 논문 §res_extacc *"halved, one at a time, a single absolute-wavenumber constant"* (`paper:2510`)
PRIMARY 팔 `A-MAPSCALE` 은 `knob_diffs` 가 **4키**(`KLIM`·`KX_G00`·`K_LIM`·`OM_Gmd5`)라 *"a single constant"* 가 아니고, 두 문장 뒤가
*"not a single constant"* 라 자기모순이다. → *"moved, one axis at a time, an absolute-wavenumber setting of the deployed path"*.

### N-5 `MINOR` — 방벽 비대칭 재발 (지시)
- `scripts/check_paper_disclosures.py` 에 **T-규칙 21개**가 들어갔고 실행 결과 **전건 OK**(rc=0). `scripts/gen_paper_constants.py --check`
  = ✅**364 상수 일치**.
- 반면 `scripts/check_thesis_disclosures.py` 의 T-토큰 **0회** — 위 N-1·N-2·N-3 는 **논문 검사기라면 잡았을** 부류(제목·초록의 자구)다.
  ⟹ 학위 검사기에 최소 `T11-대역귀속금지`·`T13-원인배제금지`·`T12-단일원인금지` 3규칙을 옮겨라.
- `scripts/gen_thesis_t11t14_constants.py` 는 `--show` 만 있고 **`--check` 가 없다**(무조건 `OUT` 에 쓴다) ⟹ 논문 쪽 `--check` 가드와
  비대칭. ✅단 `thesis/t11t14_constants.tex` 의 **핀 58개는 내가 읽은 verdict 네 파일의 값과 전건 일치**(눈대조; `T14` 라벨·팔 계수 부재 확인).
- `paper/changes_appendix.tex` 의 `R6` 는 이미 다른 항목(추적 한계 서술)이다 ⟹ *"undone"* 철회를 표에 넣으려면 **R7**. 편집자는 인라인 주석을
  택했고 그것도 선례(E1 정정) 안이다 — **선택**.

## A.3 새 절의 준수 확인 (인쇄면 · 이 md5 기준)
- ✅`paper` §`sec:res_extacc`: RESCALE 사고실험 · *"does not identify the wavelength band … as the cause"* · 학습 밴드 안 · MDE(T11·T12·T13·T14) ·
  *"not a single constant"* · 한 블록 차 · `C-MINDK` 6/24 · *"not an independent confirmation, neither blind nor out of sample"* ·
  *"undetected and not exoneration"* · 10/24+16/28 분해 · *"did not reach that conclusion independently … withdrawn"* · T14 *"prints no label"* ·
  `G-INVERT 0/176`·`G-S-BLIND 0/68` 만 · 한 주기 · solver 미명시 · 저들 수치 비교 금지 · **외부 기준선 팔 0** · §res_scope 와 대각선 읽기 금지 ·
  캡 400 · wall-clock 금지 · 결과 후 검사 공시. **위반 0**.
- ✅`paper` 초록 추가문: 수치 0 · *"thought experiment about units, not a deployable configuration"* · *"absolute-scale design range"*. **위반 0**.
- ✅`thesis` §`sec:results_t11` 본문·§boundary·국문 초록: 위와 동형 + 상수 `gt_field` 누출 불가 공시. **위반 = N-1·N-2 의 제목/서사뿐**.
- ⚠`thesis` 영문 초록: N-3.

## A.4 집계 (2차)
| 구분 | 1차(§6) | 2차 |
|---|---|---|
| 거짓 문장 | 3건/4지점 | **3건 전부 정정됨** |
| 좁힐 문장 | 7건 | **5건 정정/진행**, 생존 = **M-4 · M-5 · M-7(논문 Limitations 1곳) · D-1** |
| 동시 편집이 들여온 금지 읽기 | — | **MAJOR 3건(7지점: 인쇄 6 + 주석 1) · MINOR 1건** |
| 기존 원고의 오늘 금지 자구 | 0건 | 0건(변동 없음) |
