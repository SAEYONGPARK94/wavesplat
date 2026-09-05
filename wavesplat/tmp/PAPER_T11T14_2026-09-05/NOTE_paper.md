# `PAPER_T11T14_2026-09-05` — `T11`~`T14` 를 `paper/waveprior.tex` 에 반영한 기록

> 그룹 C / 축 C1. **런 0회** — 동결 `verdict_*.json` 만 읽었다.
> ⛔`thesis/` 무수정 · ⛔`tmp/T1*_*` 배치 디렉터리 무수정(읽기만).
> 근거 = 네 배치의 `verdict_*.json`(+`PREREG.md`·`AMENDMENT_*.md`) ·
> `memory/reference_citation_bans.md` 맨 끝 `## 2026-09-04/05` 절 ·
> 형제 축 C3 의 `tmp/RETRACT_T11T14_2026-09-05/TARGETS.md`.

---

## 0. 한 줄

원고에 **0회**였던 판정 넷을 넣었다. 서사는 *"우리 방법이 낫다"* 가 아니라 **작동 경계**이고,
넷이 그 경계에 기제를 붙인다: **안 되는 것**(T11 음성) → **무엇 때문인지**(T12) →
**무엇 때문이 아닌지**(T13) → **자동화는 아직**(T14 차단).

---

## 1. 넣은 것 — 어디에, 왜

### 1.1 ★신설 절 `§5.7 Running the estimator on a third party's ground truth` (`\label{sec:res_extacc}`)
`§5.6 Scope, stated as measured` 와 `§5.8 Summary` 사이. **아홉 문단**.

| 문단 | 무엇 | 왜 여기 |
|---|---|---|
| 1 | `T11` 음성 — 배포 학습 팔 `\TElevenShipWin{}/\TElevenN{}`, 문턱 `\TElevenThresh{}`, MDE `\TElevenMDE{}` | §Limitations 의 위상결합 시험이 *"정확도 비교는 남았다"* 라 적던 자리를 채운다 |
| 2 | 오라클 `\TElevenOracleWin{}/\TElevenN{}` = **정보는 관측에 있다** | 음성을 「막다른 길」이 아니라 **귀속**으로 만드는 첫 다리 |
| 3 | 좌표 재라벨 `\TElevenRescaleWin{}/\TElevenN{}`, $p=$ `\TElevenRescaleP{}` | 두 번째 다리. **사고실험**·**절대 스케일 설계 범위**·**밴드 안** 세 공시를 같은 문단에 |
| 4–5 | `T12` — 조건화 맵 축 `\TTwelveWin{}/\TTwelveN{}` · held-out `\TTwelveHeldWin{}/\TTwelveHeldN{}` · ρ `\TTwelveRho{}` · 라벨 경계(`\TTwelveCellWin{}` == `\TTwelveSecThresh{}`) · 음성 축 `\TTwelveBelowThresh{}/\TTwelveArms{}` + `\TTwelveSecMDE{}` · `C-MINDK` no-op `\TTwelveMinDkDiff{}` | *"무엇 때문인지"* |
| 6 | `T12` 노출 공시 (⛔독립확증·blind·out-of-sample 금지) | 등급을 문면에서 못박는다 |
| 7–8 | `T13` — `\TThirteenWin{}/\TThirteenN{}` vs 문턱 `\TThirteenThresh{}` · MDE `\TThirteenMDE{}` · 섭동 바닥 `\TThirteenFloor{}` vs 효과 `\TThirteenEffect{}` · **분해** `\TThirteenReuseWin{}/\TThirteenReuseN{}` + `\TThirteenFreshWin{}/\TThirteenFreshN{}` · 3시드 재현 (`\TTwelveShipHeldWin{}`·`\TThirteenShipFreshWin{}`·`\TFourteenShipHeldWin{}`) | *"무엇 때문이 아닌지"* + ⛔*"T12 와 독립"* 철회를 **인쇄면에** |
| 9–10 | `T14` — 차단(`\TFourteenGates{}` · `\TFourteenBlocked{}`) · 기제(`\TFourteenBandMed{}`/`\TFourteenBandMin{}`/`\TFourteenBandN{}`) · MDE `\TFourteenMDE{}` · ✅게이트만(`\TFourteenBlind{}`·`\TFourteenInvert{}`) | *"자동화는 아직"* |
| 11 | 네 배치 공통 범위 4항 | solver 미명시 · **외부 기준선 0개** · ⛔대각선 비교 · 예산 비대칭 · `wall_s` 금지 · 사후추가 게이트 |

### 1.2 기존 문장 수리 (전부 **N−1 방지**)
| 위치 | 무엇 | 왜 |
|---|---|---|
| §Limitations `\paragraph{Someone else's ground truth…}` | *"What remains undone is the accuracy comparison --- … it does not run our estimator on it."* → 시험의 **범위 서술**로 좁히고 `§5.7` 포인터 | ⛔**오늘 거짓이 됐다**(C3 의 `B-1`). `T11` 이 같은 아카이브에서 출하 진입점을 돌렸다 |
| §5.8 도입부 | *"**Three** contrasts are reported outside them"* → **Four groups**, `§5.7` 항목 추가 | 신설 절이 표 밖 대조를 하나 더 만든다 |
| `tab:boundary` 뒤 | *"두 번째 음성은 §5.7 에 있고 **뭉치지 마라**"* 문장 추가 | ⛔실측 12 레코드 음성과 제3자 합성 음성은 **다른 코퍼스·다른 실패면**. `tab:boundary` 자체에는 **넣지 않았다** |
| §Method `bounded region` 문단 | *"The bound is not only an upper one."* 문단 추가 | 그 문단이 **위쪽** 경계만 다루고 *"밖은 미측정"* 이라 적는데, 오늘 판정은 **안쪽**(상한의 1/15)에서 무기술을 보였다 (C3 `M-4`) |
| §Method 도너 기하 | *"we note it as a generalisation limit rather than defend it"* → *"now a measured limit in one direction"* | *"note rather than defend"* 는 **미측정**을 함의한다 (C3 `M-5`) |
| §Limitations `The positive results are synthetic` | *"Synthetic is not the same as ours"* 문단 추가 | 독자가 *"합성이면 된다"* 로 읽는 것을 막는다 (C3 `M-7`) |
| 초록 | 한 문장 추가(**수 0개**) | 판정 넷이 원고에 0회였다 |
| 결론 | `\paragraph{A second boundary, this one with a mechanism attached}` (**수 0개**) | 같은 이유 |
| 결론 본문 | *"…on the measured data we hold"* → *"…nor on a third party's synthetic fields at the shipped scale"* | 요약 문장이 단독으로 읽힐 때 낡지 않게 |

★**초록·결론·§Method 에는 매크로를 하나도 넣지 않았다.** 사유: 공시 규칙은 **매크로 사용**에
걸리므로, 그 절들에 수를 넣으면 **동반 의무 아홉 종을 그 절에도 전부 반복**해야 한다
(`P8` 5차 감사가 초록에서 정확히 그 부류로 6건을 냈다). 수와 병기는 `§5.7` 이 진다.

---

## 2. 방벽 — 규약대로 했다

### 2.1 `scripts/gen_paper_constants.py` (⛔손타이핑 리터럴 0)
- **ATOMS 29원자 신설**(T11 **7** · T12 **11** · T13 **10** · T14 앵커 **1**) — 전부 동결
  `verdict_*.json` 경로 + **핀**. 배치가 재분석되면 `AssertionError` 로 죽는다.
  ⟹ 헬퍼 5개가 내는 **16개**를 더해 **신설 매크로 45개**(전부 본문에서 쓰인다 — 미사용 0).
  ⚠부수효과 1건: `_donor_disjoint()` 의 `CitedBatchFiles` 가 **108 → 117**(인용 배치가 넷 늘었으므로
  도너 부재를 확인한 결과 파일 수도 늘었다). 값이 바뀐 기존 매크로는 그것 하나뿐이다.
- **헬퍼 5개 신설**:
  - `_t11_protocol()` — 관측 규약(밀도·창)을 `meta_t11.json` 에서. 파일 수 == 검정 n 교차 확인.
  - `_t12_family()` — 2차 가족 크기 8 과 문턱 미달 **7**(⛔*"여섯"* 이라 손으로 적을 뻔했다 —
    금지목록이 이름을 든 「음성 6축」 밖에 `A-BAND` 15/24 가 하나 더 있다) + `C-MINDK` 의
    `6/24` 를 `effect_vs_ship` **문자열에서 파싱**.
  - `_t13_fresh()` — 신규 절반의 `SHIP<ZERO 6/28` 을 **동결 `disclosures` 축자**에서 파싱
    (그 수는 verdict 의 별도 필드에 없다; 선례 `_spec_controls`).
  - `_t14_blocked()` — ★**팔 계수를 하나도 내지 않는다.** 차단 사실·기제·행동검사 두 개·MDE 만.
    `label is None` 이 깨지면 **죽는다**(그때는 문안을 다시 써야 한다). 덤으로
    `T11`·`T12`·`T14` 의 `G-SIZE` 가 **같은 바이트 수**임을 assert — *"같은 아카이브"* 를
    산문으로 주장하기 전에 기계로 확인한다.
  - `_map_depth()` — 조건화 맵의 **수심·파수 범위**를 `d3_beamform_net.py` **소스 리터럴에서**
    정규식으로(선례 `T14 G-SRC-PIN`). §5b 의 `M-4` 수리가 쓴다.
- **FMT 2종 신설**: `abs4`(MDE·바닥 — 금지목록이 넷째 자리까지 적어 의무로 걸었다) ·
  `sig3`(부호 있는 미세 상대효과 — 고정소수점은 `-0.0001` 로 뭉갠다).
- 결과: **366 상수 · 핀 불일치 0**(`_map_depth()` 2개 포함). `--check` 통과.

### 2.2 `scripts/check_paper_disclosures.py` — **동반 공시 규칙 18종 신설**
| 규칙 | 무엇을 강제 |
|---|---|
| `T11-사고실험` | *"thought experiment about units"* + *"No sample is altered"* |
| `T11-대역귀속금지` | *"absolute-scale design range"* 정본 자구 |
| `T11-밴드안` | *"already sits inside the prior's training band"* |
| `T11-MDE` | `\TElevenMDE` |
| `T11-solver미명시` | *"no documentation naming its solver"* |
| `T1114-외부기준선없음` | *"no external baseline arm"* (네 배치 전체 트리거) |
| `T12-단일상수아님` | *"not a single constant"* |
| `T12-노출표본` | *"not an independent confirmation"* / *"neither blind nor out of sample"* |
| `T12-라벨경계` | *"a single block in the other direction"* |
| `T12-MDE` | `\TTwelveSecMDE` |
| `T12-MINDK` | `\TTwelveMinDkDiff` (75% no-op 병기) |
| `T13-MDE` · `T13-바닥` · `T13-분해` · `T13-독립아님` | MDE · 섭동바닥 · 10/24+16/28 분해 · *"did not reach that conclusion independently"* |
| `T14-라벨없음` · `T14-BAND취약` · `T14-MDE` | *"prints no label"* · `\TFourteenBandMin` · `\TFourteenMDE` |

★**항진 검사 `--selftest`: 78/78 · 미검증 0.** 신설 18종 전부 (A) 근거를 지우면 FAIL하고
(B) 트리거를 지우면 침묵한다 = **규칙이 규칙이다**.

### 2.3 래칫·전체 방벽
- `audit_paper_bans.py` 양방향 래칫: **기준선 12 → 오늘 12 (동일)**, `tex` 편집 후·**PDF 재조판 후** 둘 다 확인.
- `scripts/check_all.py --fast`: **29종 전부 통과**(`--fast` 는 pytest 만 건너뛴다).
  포함 = 게이트 구현 대조 · 판정 파일 위생 · **원고 인용 성숙도**(신규 인용 배치 넷 통과) ·
  상수 생성기 `--check` · 동반 공시 · 항진 검사 · 인용금지 래칫 · 원장 수치 대조 · 인쇄면 손상 등.

---

## 3. ⛔**넣지 않기로 한 것과 사유**

| 넣지 않은 것 | 사유 |
|---|---|
| `T14` 의 **어떤 팔 계수도** (`AUTO-AP` 28/28 · 회수율 1.634 · `AUTO-AP<FIX2` · 스윕 11/12·12/12·12/12) | ⛔금지목록: **차단된 배치의 부분 통계**. 라벨이 없으므로 결론이 될 수 없다. 인쇄한 것은 게이트(`0/176`·`0/68`)와 차단 기제뿐 |
| `T14` 의 `R`(회수율) | ⛔`T12` 의 `ρ` 와 **병치 금지**(정의가 다르다 — 선형 vs 로그비, 분모도 다르다). `ρ` 만 인쇄했다 |
| `T11`·`T13` 의 **층별(Hs·각도) 수** | ⛔금지목록 네 배치 공통. 각도 6클러스터·Hs 두 층은 기술통계이고 라벨을 만들지 않는다. 부등호 규약도 갈린다 |
| `T11` 층별 수 ↔ `FRESH-28` 직접 비교 | ⛔모집단이 다르다(Hs 3/25 치우침) |
| `wall_s`·지연 | ⛔팔 간 비교 금지(BLAS 스레드가 팔마다 다르다). 대신 *"not compared between arms"* 를 **본문에 적었다** |
| `T11` 초판 `ORACLE` 수치 | ⛔개정 전 값은 **상한이 아니라 하한**이었다. `AMENDMENT_2 §5` 의 ±ω 4열 정의 값만 썼다 |
| `E2`·`T1`·`T10` 과의 **대각선 비교표**(*"우리 방법 × 저들 자료"* vs *"저들 방법 × 우리 자료"*) | ⛔`T11 PREREG §6` 의무 공시. 만들면 그림이 좋지만 **금지**다. 대신 *"not comparable with §5.6"* 을 명시 |
| 저들 논문(`Zheng2025`)의 수치와 비교 | ⛔같은 사유(채점·격자·기준이 다르다) |
| `T11`~`T14` 를 `tab:boundary` 행으로 | ⛔그 표는 **실측 12 레코드 전용**이다(캡션이 그렇게 선언한다). 제3자 합성은 다른 표본이다 |
| `T11`~`T14` 를 `tab:earns` 행으로 | ⛔그 표는 *"표현이 버는 것"* = **기준선 대비**다. 이 넷은 제로필드 대비이고 기준선 팔이 0개다 |
| `\FIX` 의 절대 nRMSE · `FIX` 배포 권고 | ⛔`T13` 공시. 대신 *"harness intervention … nothing here recommends adopting it"* |
| 하이라이트(6번째 항목) | Elsevier 관례 3–5항이고 현재 5항이다. 85자 안에 의무 병기를 실을 수 없으므로 **구조적으로 위반**이 된다(`P8` 2차 감사 선례) |
| `changes_appendix.tex` 의 새 철회 행(`R6`) | 이 원고가 **철회한 문장이 없다**. §Limitations 의 한 문장은 *"거짓이 됐다"* 가 아니라 **범위가 좁아졌다**(같은 자리에서 좁히고 포인터를 달았다). ⚠C3 `TARGETS.md §4` 는 `R6` 신설을 제안한다 — 그것은 **thesis 쪽 처분과 함께 결정할 사안**으로 남긴다 |
| `T12`·`T14` 의 **1단 격자 상대화 처방** | ⛔`T14 AMENDMENT_2 §3`: *"다른 규칙이므로 새 사전등록"*. 본문은 그 사실만 적는다 |
| 83백분위(학습 밴드 안) 수치 | 손타이핑이 될 뻔했다(verdict 본문이 아니라 공시 산문의 수). 자구 *"already sits inside the prior's training band"* 로 **수 없이** 병기했다 |
| 개구 619.64 m / 220.55 m | `s_ap`(=2.8095, 차단 배치 통계)에서 유도해야 한다 ⟹ **위험**. 대신 *"substantially wider"* 로 적었다 |

---

## 4. 조판

⛔**`paper/` 전용 빌드 스크립트는 저장소에 없다**(`thesis/build.sh` 는 학위논문 전용).
`thesis/build.sh` 와 **같은 절차**를 손으로 돌렸다:

```bash
export PATH="$HOME/texlive/2026/bin/x86_64-linux:$PATH"
cd wavesplat/paper && pdflatex … waveprior.tex ; bibtex waveprior ; pdflatex ×2
```

| | 값 |
|---|---|
| 쪽 | **94** (편집 전 87) |
| 오류 | **0** |
| 미정의 인용 | **0** |
| 미정의 참조 | **0** |
| 재실행 필요 | 아니오 |
| Overfull / Underfull | 7 / 14 (편집 전과 **동일** — 내 편집이 조판 미세문제를 늘리지 않았다) |
| 초록 | 344 단어 (편집 전 300 — 수는 넣지 않았다) |

⚠**PDF 를 다시 만들었으므로 래칫이 PDF 를 다시 읽는다** — 재조판 **후**에도 12 → 12 로 동일함을 확인했다.

---

## 5. ⛔금지 자구를 하나도 쓰지 않았음을 어떻게 확인했나

1. **기계** — `check_paper_disclosures.py` 18규칙 신설 + `--selftest` **78/78**.
   각 규칙은 *"이 매크로가 절에 나타나면 이 근거 표현이 **같은 절에** 있어야 한다"* 이고,
   근거를 지우면 반드시 실패함을 검사기가 스스로 증명한다.
2. **기계** — `audit_paper_bans.py` 양방향 래칫 12→12(악화도 개선도 없음), `check_all.py` 29/29.
3. **기계** — `literals_outside_macros()`: 매크로 값이 리터럴로 인쇄된 곳 **0건**.
4. **손** — `git diff` 의 **추가된 212줄**(주석 제외)에 금지 자구 정규식을 걸었다:
   `single constant|wavelength band of the prior|auto-?scal|automatically select|
    depth is not the cause|confound is harmless|no effect|independent confirmation|
    out of sample|diagonal|fully non-linear|28 of 28|1\.634|recovery fraction`
   → **히트 6건, 전부 부정문 안**:
   - *"not a single constant"* ×2
   - *"candidate for insufficient leverage rather than for **no effect**"*
   - *"We do not call this an **independent confirmation**, and it is neither blind nor **out of sample**"*
   - *"we quote from it no arm count, no **recovery fraction**"*
   - *"we do not call it higher-order or **fully non-linear**"*
5. **손** — 조판된 PDF 에서 신설 절 전문을 `pdftotext` 로 뽑아 **인쇄면**을 읽었다.
   ★그 과정에서 **사실 역전 1건을 잡아 고쳤다**: `\TFourteenBlind{}`(=`0/68`)·
   `\TFourteenInvert{}`(=`0/176`)는 **위반 수**인데 초판 문장이
   *"reproduced … on 0/68 checks"* 라 적어 **0건에서만 재현됐다**로 읽혔다
   ⟹ *"--- \TFourteenBlind{} mismatches ---"* 로 고쳤다.
   같이 고친 것 둘: 숫자로 시작하던 문장(*"7 of the 8 …"* → *"The other …"*) ·
   *"31/32 of its gates pass"*(분수로 읽힌다) → *"its gate tally is 31/32"*.
6. **교차** — 형제 축 C3 의 `tmp/RETRACT_T11T14_2026-09-05/TARGETS.md` 와 대조.
   그 문서의 `paper/` 표적 **4건**(B-1 · M-4 · M-5 · M-7) 을 전부 처리했고,
   `M-8`(*"고치지 마라"* 목록 — *"제안 표현은 여전히 외부 방법과 비교된 적이 없다"* 등)
   은 **손대지 않았다**.

---

## 5b. ★독립 감사(축 C3)가 **내 편집에서** 잡은 것 — 전부 닫았다

| # | 무엇 | 처분 |
|---|---|---|
| **N-4** | 내가 쓴 *"halved, one at a time, **a single absolute-wavenumber constant**"* 이 **두 문장 뒤의 *"not a single constant"* 와 자기모순**. 실측 `knob_diffs_measured['A-MAPSCALE']` = **4키**(`KLIM`·`KX_G00`·`K_LIM`·`OM_G` md5) | ✅자구를 **정본**으로 교체: *"halved, one arm at a time, **the absolute wavenumber scale of exactly one subsystem**of the deployed path --- for some arms that is one constant and for others, including the one that reproduces the effect, it is the several constants that define that subsystem's scale together"* |
| **M-4** | 알고리즘 표 3단계에 **조건화 맵 수심 고정** 공시가 없다(학위논문에는 있다 = 두 원고 불일치) | ✅표 3행에 *"$\omega$ from the dispersion relation at a **fixed** $d=\DepMapDepth{}$~m baked into the module at import --- the configured depth does not reach this map"* + §Method 에 문단 *"That grid also fixes a depth."* 신설. ★수는 **생성기 신설 헬퍼 `_map_depth()`** 가 `d3_beamform_net.py` **소스 리터럴에서 정규식으로** 뽑는다(`\DepMapDepth`·`\DepMapKLim`) --- 손타이핑 0. 덤으로 §Method·표에 손으로 적혀 있던 `0.6` 두 곳도 `\DepMapKLim{}` 로 바꿨다 |

⚠**C3 가 「생존」으로 표시한 3지점은 이미 이 작업에서 닫혀 있었다**(그 목록은 HEAD `5b27a7e`
기준으로 작성돼 내 편집을 보지 못했다):
- `:1143` *"bounded region"* 상한만 → **`\textbf{The bound is not only an upper one.}` 문단 신설**(§1.2)
- `:955` 도너 기하 *"note rather than defend"* → ***"now a measured limit in one direction"*** 로 교체
- §Limitations *"The positive results are synthetic"* → ***"Synthetic is not the same as ours"* 문단 신설**

✅**지우지 않은 참 문장**: *"The learned prior … has still never been compared against an
external method"*(`:2296`). `T11`~`T14` 에 외부 **방법** 팔은 0개이고, `T11` 은 외부 **참값**
위에서 **우리 팔끼리** 비교한 것이다 --- 두 개념을 섞지 않았다.

★**그 수리가 결함을 하나 더 만들었고 기계가 잡았다**: `N-4` 를 고치며 넣은 주석 블록이
**본문 한 절을 삼켰다**(`scripts/check_comment_swallow.py` ⛔`:2554` *"주석 뒤 본문이 소문자로
시작한다"*). 주석을 문단 **앞**으로 옮겨 해소했다. ⟹ 이 저장소가 이름 붙인
*"수리 자체가 인쇄 파손을 만든다"* 부류의 재발이고, **오늘도 사람이 아니라 기계가 잡았다.**

---

## 6. ⚠남긴 것 (다음 사람에게)

- `changes_appendix.tex` `R6` 행 신설 여부 — **철회가 아니라 범위 축소**라 넣지 않았다. 재검토 대상.
- `T11`(*"우리 방법 × 저들 자료"*)과 `E2`(*"저들 방법 × 우리 자료"*)의 **2×2 표**는
  독자에게 매력적이지만 ⛔**대각선 비교 금지**가 걸려 있다. 그리려면 금지 공시를
  같은 절에 넣어야 하고, 그 경우 규칙을 하나 더 추가해야 한다.
- `T13 AMENDMENT_3 §3` 이 스스로 적은 **공시 5건 부채**(`FIX` 절대 nRMSE 금지 등)는
  아직 `verdict['disclosures']` 에 없다 — 원고는 그 다섯을 **자구로** 지키고 있으나
  기계가 verdict 쪽에서 강제하지는 않는다.
- `unused_macros` 자문 15건은 **내 신설 매크로를 하나도 포함하지 않는다**(45/45 사용, 기계 확인).
