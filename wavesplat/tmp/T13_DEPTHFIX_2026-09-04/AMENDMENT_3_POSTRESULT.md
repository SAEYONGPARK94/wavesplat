# `T13-DEPTHFIX` 개정문 3 — ⚠**결과 이후** · 공시 + 동결 확장 · 새 런 0회

작성: 2026-09-05. 계기: **판정 간 정합 감사**와 **소급 구현 감사**(다중 에이전트).
⛔**판정 불변** — `FIX-UNDETECTED (WITHIN-PERTURBATION-FLOOR)` 그대로.

## 1. ⛔⛔"독립" 은 거짓이다 — **이 배치의 절반은 새 자료가 아니다**

실측(내가 직접 재집계, 런 0회):
```
T12 E-MAPDEPTH(PRIMARY 24행)  vs  T13 FIX(T11-24 층 24행)
  nrmse 불일치 0/24 · proposal_md5 불일치 0/24 · seed 불일치 0/24 · max|Δ| = 0.000e+00
```
같은 처치(`d3_beamform_net.DEPTH` 50→400 + `kgrid()` 재호출)를 **같은 시드**(`20260904+fi`)로
두 번 돌린 것이다. ⟹ `T12` 결과 파일만으로 `E-MAPDEPTH<SHIP` = **10/24** 가 계산되고,
그것이 `T13` PRIMARY 의 절반이다.

⟹ ⛔**철회한다**:
- 커밋 `9ed3503` 본문 *"T12 와 T13 이 **독립적으로** 같은 결론에 왔다"*
- 원장 *"T12 의 E-MAPDEPTH 8/24 널을 더 민감한 설계로 **확증**"*
(원장 3파일은 2026-09-05 에 정정했다.)

**정확한 서술**: PRIMARY **26/52 = 10/24(T12 행의 재판독) + 16/28(신규)**.
신규 절반만: `FIX<SHIP` **16/28**(문턱 19) · `FIX<ZERO` 13/28 · `SHIP<ZERO` 6/28.
⟹ **라벨은 어느 쪽으로 세도 `FIX-UNDETECTED` 다** — 판정 불변.

⚠**노출 미공시**: `PREREG.md:90` 표가 T11-24 층을 *"`FIX`·`RESCALE-FIX` 미노출"* 이라 적었다.
작성 시점(21:46)엔 참이었으나 `T12` 커밋(23:34) 뒤 본실행(00:13)에서는 **거짓**이다.
그리고 `AMENDMENT_1 §5` 가 `T12` 의 `E-MAPDEPTH` 결과를 **인용하면서도** 동일 처치임을 적지 않았다.
⚠**`FRESH-28` 의 `SHIP` 도 형제 노출**이다 — `T12` SEC-R 이 같은 28파일을 (다른 시드로) 먼저
돌렸다. `FIX` 는 blind 가 맞다.

★**처방(방벽)**: `tmp/_audit/check_batch_artifacts.py` 에 **형제 배치 간 `(block, seed,
proposal_md5)` 교집합 검사**를 붙이면 이 부류가 기계로 잡힌다. (별건으로 남긴다.)

## 2. 동결 목록이 판정 경로 파일 **셋**을 빠뜨렸다 — `G-FREEZE-EXT` 신설

`run_t13.py:37-40` 의 `IMPORT_FILES` 8개에 없는 것:
| 빠진 것 | 판정 경로 증거 |
|---|---|
| `wavesplat/optimizer/initializer.py` | `sequential_update.py:1682` 이 **호출 시점에** `_beamform_coarse`·`MIN_DK` 임포트 (성장 경로). ★`T12` 가 같은 파이프라인에서 실측: `C-GROWGRID` **24/24 변경** |
| `scripts/d3_setdiff_prior.py` | `learned_init.py:112` 가 제안 경로에서 `denorm_atoms` 임포트 |
| `sha256:results/direction3/setdiff_prior.pt` | ★**이 배치의 BLOCKER 를 만든 바로 그 파일** — 러너 주석(`:25-31`)이 축자로 적어놓고 **동결은 안 했다** |
`T12` 는 셋 다 동결했다(`consts_t12.py:114-131`).

⛔**러너(측정면)를 고치지 않는다.** 대신 분석기가 **`T12` 의 결과 0행 동결값**과 대조한다 —
`T12` 동결(09-04 밤)이 `T13` 본실행(09-05 00:13)보다 **앞서므로 진짜 앵커**다.

## 3. ⚠기록만 하는 것
- `gates_t13.py:127` 의 차단 목록에 `G-FREEZE` 가 있으나 분석기는 그 이름의 게이트를 만들지
  않는다(동결 검사는 `G-ENV` 안에 있다). **잔재** — 라벨에 영향 없다.
- `PREREG.md §11` 의 금지 5건(`FIX` 절대 nRMSE 인용 · `FIX-HURTS` 지목 · `SKILL-FIX-UNDETECTED`
  의 `F1` 부류 경고 · `FRESH-28` Hs 치우침 읽기 · 재학습 미측정)이 `verdict['disclosures']` 에
  없다. ⚠이 개정문이 §1 의 공시 둘만 추가했다 — **나머지 5건은 다음 개정문의 부채**로 남긴다.
  ⛔*"공시를 떨어뜨렸다"* 는 사실 자체를 여기 기록한다(N−1 부류).
