# E-LIVE 결과 — 출하 기본값 `auto_v1_m3` 는 자기를 인가한 수치를 재현하는가

실행 2026-08-06 10:02–10:21 KST. 원자료 `arm_runs_auto_live_confirm2.jsonl`(200 lane, err 0,
중복 0), 로그 `live_auto.log`, 채점 `VERDICT_E_LIVE.json`(동결 채점기) ·
`INDEPENDENT_CHECK.json`(내가 처음부터 다시 쓴 재도출기 `verify_independent.py`).

---

## 0. 한 줄

**PASS — 그리고 "통과"보다 강하다: 200/200 lane 이 차이 정확히 0.0 (근사치가 아니라 부동소수
비트 일치).** 출하 기본값을 한 프로세스로 라이브 실행한 결과가, 두 팔을 오프라인으로 조립해
만든 예측과 lane 단위로 완전히 같다. 즉 `wavesplat/run.py:35-38` 주석의 다섯 수치는
**조립물이지만 정확한 조립물**이었다.

---

## 1. 무엇이 문제였나 (실험의 이유)

배포 인가 문서 `tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §4` 가 `auto_v1_m3` 를 인가하고,
그 슬라이스 평균이 출하 기본값 주석에 박혀 있다. 그런데 `tmp/confirm2_2026-07-27/` 의
lane 레코드 **600행 전부 `auto_selected=None` 이고 `arm='auto'` 레코드는 0건**이다
(arm 분포: learned 200(v1) + learned 200(v3) + v2_m3 200). 인가 수치는
`arm='learned'`(v1) 와 `arm='v2_m3'` 를 lane 별 `n_selected>=8` 로 **오프라인 조립**한 값이고,
그 구성은 **한 프로세스로 실행된 적이 없었다.**

그 위에 더 큰 구멍이 있었다. RESULT_CONFIRM2 §4 자신이 이렇게 적는다:

> m3 폴백은 현재 **평가 하네스로만 구현**됨 — 운용 경로(wavesplat/) 이식은 별도 작업

즉 인가 당시의 median-of-3 은 `tmp/confirm2_2026-07-27/run_v2m3.py` 의 **별도 구현**(run.py 의
`compute_all_metrics`/`nrmse` 를 몽키패치로 가로채 pred 필드를 캡처하고 numpy 로 pointwise
median)이었고, 오늘 출하되는 것은 그 뒤 `wavesplat/optimizer/sequential_update.py` 에 이식된
`MedianEnsembleUpdater`(멤버별 RNG 스냅샷/복원 + `torch.median`)다. **서로 다른 두 코드가 같은
수를 내는지 아무도 확인한 적이 없었다.** E-LIVE 는 그것을 잰다.

합격선을 통계가 아니라 **결정론**으로 잡은 이유: 조립이 라이브의 정확한 에뮬레이터여야 하므로
허용 오차는 검정력 문제가 아니라 등가 문제다(<=1e-9).

---

## 2. 실행 전 검증 (4건 전부 통과, 그러나 발견 3건)

| # | 검증 | 결과 |
|---|---|---|
| V1 | `score_e_live.py` 집계 규약이 `score_confirm2.py` 의 실제 규약과 같은가 | **동일.** 원본을 열어 대조했고, 그것과 별개로 규약을 **처음부터 다시 구현**해 lane 200/200·route 200/200 일치, 5개 슬라이스 평균이 `confirm2_verdict.json` 의 `candidates.v1.means` 와 전자리 일치 |
| V2 | `reference_frozen.json` 이 `run.py:35-38` 의 수치를 재현하는가 | **재현.** 최대 이탈 4.9e-5 = 4자리 반올림 오차 |
| V3 | `run_arms_confirm2.py` 에 `auto` 팔이 배선돼 있는가 | **배선됨** (L80–88: `cfg['init_method']='auto'` 설정, L101: `auto_selected=r.get('auto_selected')` 기록) |
| V4 | 체크포인트가 오늘의 출하 기본값인가 | **동일.** `results/direction3/beamform_cond_diffusion_mixed_v1.pt` sha256 `636c72a7…239d9` = `learned_init.DEFAULT_CKPT_SHA256`. 캐시 `atoms_confirm2_v1.jsonl` 은 `precompute_confirm2.py:29` 가 **그 경로 그대로** 써서 만든 산물 |

추가로 사전에 확인한 것:

- **n_selected 재계산**: 캐시된 원자에 라우터 자신의 기본 손잡이
  (`energy_thresh=0.03, dedup_dk=0.03, max_n=30`)로 `pool_and_select` 를 다시 돌려
  저장값과 **200/200 정확히 일치**. 즉 P1 은 실행 전부터 예측 가능했다.
- **RNG 중립성 프로브**: 비트 일치가 성립하려면 (a) 라우터의 `pool_and_select` 와
  (b) `EpisodeDataLoader`+`get_init_window` 가 전역 RNG 를 소모하지 않아야 한다
  (앙상블 멤버 1·2 는 `_seed_globals(s)` 직후 초기화되지만, 오프라인 멤버는 로더를 거친 뒤
  초기화된다). **둘 다 RNG 중립 확인.**
- 하네스가 명시하는 값이 전부 출하 기본값과 같음: `learned_init_iters=500`
  (`sequential_update.py:603` 기본값), `auto_switch_thresh=8`(기본값),
  `TRACK_PHASE_PREDICT=False`(모듈 기본값과 동일 — no-op), `analytic_ensemble_k` 미지정 →
  `DEFAULT_CONFIG` 의 3, `auto_guard` 미지정 → False(은퇴 상태).

### 검증에서 나온 문제 3건

1. **`reference_frozen.json` 의 슬라이스가 5개가 아니라 4개다.** 출하 주석은
   `all / sparse / 5comp / crowded / 2comp` 다섯 수치를 적는데, 동결 예측과 `score_e_live.py`
   의 `ship` dict 모두 **`2comp` 를 빠뜨렸다**. 내 독립 재도출로 메웠다:
   라이브 2comp = 오프라인 2comp = **0.016015** (주석 0.0160). 판정은 바뀌지 않지만,
   동결된 예측이 주석의 5분의 4만 덮고 있었다는 건 사실이다.
2. **`score_e_live.load_rows` 와 `score_confirm2.load_rows` 는 같은 함수가 아니다.**
   후자는 중복 lane 을 만나면 `sys.exit`(PREREG Sec 2 규칙)하고 err lane 을 `inf` 로 넣어
   P1c 에 불리하게 센다. 전자는 err 행을 **조용히 버리고**(→ "missing" 으로 판정 보류, 안전)
   중복 행은 **조용히 덮어쓴다**(→ 마지막 것이 이김, 안전하지 않음).
   이번 런은 err 0 · 중복 0 이라 차이가 실제로 발생하지 않았고, 나는 중복 검출을 내
   재도출기에 넣어 별도로 확인했다. 그래도 이 로더는 재사용 전에 고쳐야 한다.
3. **라이브 출력 행에 `ensemble_k` 와 `auto_real_gate` 가 기록되지 않는다.**
   `run_estimation` 은 둘 다 summary 로 내보내는데(`run.py:299,304`) 하네스 스키마가
   `auto_selected`/`auto_n_selected` 만 옮긴다. 그래서 **"v2_konly 로 라우팅된 73개 lane 에서
   앙상블이 실제로 3멤버로 만들어졌다"는 것은 로그에서 직접 확인되지 않는다.**
   간접 증거는 강하다(그 73개 lane 의 중앙 실행시간 85.1 s vs learned lane 18.0 s ≈ 3배 초기화,
   그리고 값이 3멤버 오프라인 m3 와 비트 일치), 하지만 직접 기록은 없다.

---

## 3. 판정

동결 채점기와 내 독립 재도출기가 **독립적으로 같은 결론**에 도달했다.

| 바 | 내용 | 결과 |
|---|---|---|
| **P1** 라우팅 충실도 (결정론, n=200 lane) | 라이브 `auto_selected` == 예측 route | **200/200 PASS** |
| **P2** lane 정확 일치 (결정론, n=200 lane) | \|라이브 − 조립\| <= 1e-9 | **200/200 PASS, 최대 편차 정확히 0** |
| **P3** 통계 폴백 (P2 실패 시에만 의미) | 에피소드 paired Δ, 슬라이스 등가대 | **PASS (Δ median = 0, 40/40 동률)** |
| **P4** 희소 lane 안전문 (결정론, n=80 lane) | 어떤 sparse lane 도 0.10 초과 금지 | **0/80 위반, max 0.0768 PASS** |
| | | **=== E-LIVE VERDICT: PASS ===** |

> P1·P2·P4 는 결정론적 검사다. **p-value 를 붙이지 않는다.**
> P3 의 부호검정은 여기서 퇴화했다(+0 / −0 / 동률 40 → p=1). 이 p=1 은 "차이 없음의 증거"가
> 아니라 **비교할 부호가 하나도 없다**는 뜻이다. 등가는 P2 의 결정론적 0 이 말하는 것이지
> 이 검정이 말하는 게 아니다. (참고: n=40 부호검정 바닥 p = 2·0.5^40 = 1.819e-12.)

### 라이브 슬라이스 평균 (lane→에피소드 MEDIAN, 에피소드→슬라이스 MEAN)

| 슬라이스 | n | 라이브 | 오프라인 조립 | 차이 | 출하 주석 |
|---|---|---|---|---|---|
| all | 40 | 0.087674 | 0.087674 | **+0.000000** | 0.0877 |
| sparse | 16 | 0.017509 | 0.017509 | **+0.000000** | 0.0175 |
| moderate (5comp) | 8 | 0.031452 | 0.031452 | **+0.000000** | 0.0315 |
| crowded | 16 | 0.185951 | 0.185951 | **+0.000000** | 0.1860 |
| 2comp | 4 | 0.016015 | 0.016015 | **+0.000000** | 0.0160 |

라이브 vs 오프라인 조립의 차이: **어디에도 없다.** lane 200개 전부에서
`live.avg_nRMSE - offline.avg_nRMSE == 0.0` (1e-9 이 아니라 0). 최대 편차 lane 도 0.

---

## 4. 라우터는 실제로 어디로 보냈나

`auto_n_selected` 도 캐시 저장값과 **200/200 일치**했다.

| | learned | v2_konly |
|---|---|---|
| 전체 lane | 127 / 200 | 73 / 200 |
| sparse (n=80 lane) | 22 | 58 |
| moderate (n=40 lane) | 29 | 11 |
| crowded (n=80 lane) | **76** | 4 |

에피소드 다수결(5 lane 중 3 이상 learned) 기준: sparse **2/16** · moderate **6/8** ·
crowded **16/16**. 이것은 RESULT_CONFIRM2 §3-3 이 적은 "v1 희소 2/16 · 5comp 6/8" 와
정확히 같다 — 라우팅 censo 도 재현됐다.

crowded 인데 분석팔로 간 4 lane (전부 uniform, n_selected 4–5):
`BF4_30comp_seed1111_uniform e2d2`(0.2744) · `BF5_15comp_seed1111_uniform e1d1`(0.2234) ·
`BF5_15comp_seed1212_uniform e2d2`(0.2388) · `BF5_15comp_seed1212_uniform e3d3`(0.2486).

---

## 5. 참고 맥락 (판정 바 아님)

- crowded 16/16 에피소드가 평탄해면 널(`tmp/E_depth2_2026-08-06/D2_nulls_confirm2.json`,
  ≈0.25)보다 낮다, 중앙 마진 −0.0670. **이것은 바닥을 넘었다는 뜻이지 성능 주장이 아니다.**
  사전등록된 E-LIVE 바가 아니므로 판정에 넣지 않는다.
- crowded/slope 평균 0.1530 vs crowded/uniform 0.2189 (각 n=8 에피소드).
- 벽시계: 200 lane, 8 워커, **1106 s**. lane 중앙 실행시간 v2_konly 85.1 s / learned 18.0 s.
  **이 수치는 성능 주장에 쓰지 않는다** — 워커 경합·BLAS 스레드 아티팩트 전례가 있다
  (⛔인용금지 37).

---

## 6. 이 결과로 인용해도 되는 문장

- "출하 기본값 `init_method='auto'` + `analytic_ensemble_k=3` 을 **한 프로세스로** 신선
  예비시드 40(200 lane)에 라이브 실행한 결과가, `auto_v1_m3` 를 인가한 오프라인 팔 조립과
  **lane 200/200 에서 부동소수 비트 일치(차이 정확히 0)** 했다."
- "GT-free 라우터(th=8)는 라이브에서 **200/200 lane 예측대로 라우팅**했고, `n_selected` 도
  캐시 저장값과 200/200 일치했다."
- "`wavesplat/run.py:35-38` 주석의 다섯 슬라이스 수치(all 0.0877 / sparse 0.0175 /
  5comp 0.0315 / crowded 0.1860 / 2comp 0.0160)는 **라이브 실행으로 재현된다**
  (반올림 오차 <= 5e-5)."
- "인가 당시 **평가 하네스로만 존재하던** median-of-3 필드 앙상블(`run_v2m3.py` 의 몽키패치
  구현)과, 이후 운용 경로에 이식된 `MedianEnsembleUpdater` 는 이 40 에피소드에서
  **비트 일치**한다 — 이식은 충실했다."
- "희소 lane 안전문(>0.10 금지)이 라이브 경로에서 **0/80 위반**으로 유지된다(max 0.0768)."
- "이 40 에피소드에서 라우터는 crowded 16/16 · 5comp 6/8 · sparse 2/16 에피소드를 학습팔로
  보낸다."

## 7. 이 결과로 **여전히 인용 불가**인 문장

- ⛔ **"출하 기본값이 정확하다 / 잘 작동한다 / 검증되었다."** 이것은 **재현 시험**이지
  정확도 시험이 아니다. 0.0877 이 좋은지, 다른 추정기를 이기는지, 이 40개 밖으로 일반화되는지에
  대해 E-LIVE 는 **아무 말도 하지 않는다.**
- ⛔ **"라우터/제안기가 결정론적이다."** 라이브 런은 제안 원자를 **동결 캐시**
  (`atoms_confirm2_v1.jsonl`)에서 받았다. GPU 확산 제안(`propose_atoms`)은 **한 번도 실행되지
  않았다.** 캐시 없이 배포되는 경로(그때 `learned_diffusion_seed` 는 `self.seed` 로 기본값이
  바뀐다)는 여기서 시험되지 않았다.
- ⛔ **실측(GT-free) 경로에 대한 어떤 진술도 불가.** 실측 도메인 게이트는 40개 합성
  에피소드에서 **한 번도 발화하지 않았다**(전부 GT 있음). `real_estimator='bank_mmse'` 경로
  미실행.
- ⛔ **"P3 가 등가를 통계적으로 확인했다."** P3 의 부호검정은 40/40 동률로 퇴화했고 p=1 은
  정보가 없다. 등가를 말하는 것은 P2 의 결정론적 0 이다.
- ⛔ **"v1 이 v3 보다 낫다."** E-LIVE 는 v3 를 돌리지 않았다. RESULT_CONFIRM2 의 tie-break
  금지 조항(11/16)은 그대로 유효하다.
- ⛔ **벽시계/지연 수치를 성능 비교에 사용 금지**(§5).
- ⛔ **"앙상블이 73개 lane 에서 3멤버로 돌았다"를 로그 근거로 인용 금지** — `ensemble_k` 가
  기록되지 않았다(§2 발견 3). 간접 증거만 있다.

---

## 8. 재현

```bash
cd /home/vexit/wavesplat/wavesplat
nice -n 5 python3 tmp/confirm2_2026-07-27/run_arms_confirm2.py --arms auto \
    --out tmp/E_live_2026-08-06/arm_runs_auto_live_confirm2.jsonl \
    --cache tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl --workers 8
python3 tmp/E_live_2026-08-06/score_e_live.py --score          # 동결 채점기
python3 tmp/E_live_2026-08-06/verify_independent.py            # 독립 재도출
```

`tmp/confirm2_2026-07-27/` 의 어떤 파일도 수정하지 않았다(전부 mtime 2026-07-27 유지).
`paper/`·`configs/`·출하 기본값·체크포인트 무변경, git commit 없음.
