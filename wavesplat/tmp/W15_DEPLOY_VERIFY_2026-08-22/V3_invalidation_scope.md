# V3 — 기록 무효화 범위 감사 (2026-08-22 배포 3건)

**추정 런 0회.** 감사 대상 = `wavesplat/tmp/` 의 **디렉터리 120개**(`.py` 를 가진 것; `fowd_2026-08-02`
= venv 제외) + 패키지 소스 + 메모리/문서 4종.
라벨 = `MEASURED`(파일을 열어 확인) · `ARGUED`(추론) · `UNVERIFIED`(미확인).

---

## 0. 요약 표 — 3분류 집계

| 분류 | 뜻 | 건수 |
|---|---|---:|
| **기록손상** | 이미 저장된 결과 파일의 수가 바뀐다 | **0 / 120** |
| **재실행시 무효** | 같은 러너를 오늘 다시 돌리면 다른 구성이 돈다 (전 팔) | **30** |
| **재실행시 부분무효** | 일부 팔만 이동 (오라클·`v2_konly` 팔은 비트 동일) | **19** |
| **무효화 없음** | 전 팔이 `v2_konly`/`oracle`/`v1`/`v2`/`bank_mmse` 명시 또는 오늘 배포 후 설계 ⟹ 비트 동일 | **31** |
| (추정 런 0회 배치) | 분석·문서 전용 | **40** |

★**세 키 중 하나라도 **명시**한 배치 = 120 중 **18개**뿐이고,
★★★**`varpro_steps` 를 명시한 배치는 08-22 이전에 `0`개**다.**

| 키 | 명시한 배치 수 | 누구 |
|---|---:|---|
| `varpro_steps` | **5** | W8·W11·W13·W14·W15 — **전부 오늘, 배포 이후** |
| `learned_init_iters` | **13** | B·S11·W4·W5·W8·confirm2·confirm40·heldout40·learned_only·mixed_retrain·unified·unified_guard·wp_integration |
| `auto_switch_thresh` | **3** | `blind_auto_real_2026-07-26`(실측) · `confirm2`(그림/린트 전용) · W15(오늘) |

⟹ ★★★**λ 사고가 정확히 재현될 조건이 갖춰져 있었다**: 러너들이 값을 안 적었고, 값은 오늘 바뀌었다.

### 0.1 두 질문을 반드시 분리한다 (`CLAUDE.md` λ 절의 규약)

| 질문 | 답 | 근거 |
|---|---|---|
| **기록된 결과가 손상되는가?** | ⛔**아니다. 0건.** | 세 변경은 커밋 `98883a8`(08-22 10:12)·`d673892`(08-22 13:58)이고, 그 이전에 끝난 배치의 `results.jsonl`/JSON 은 정적 파일이다. 코드 기본값 변경은 소급하지 않는다. `MEASURED` |
| **재실행하면 달라지는가?** | ✅**49개 배치에서 그렇다** (무효 30 + 부분 19). | 아래 census |
| **그 배치를 "현행 배포의 성질" 로 인용해도 되는가?** | ⛔**아니다** — 도달 팔이 이동한 49개 전부. | §2 |

⚠**따라서 옳은 문언은 "무효" 가 아니라 "구 배포 구성에서 측정된 것" 이다.** 결과는 살아 있고,
**그것이 서술하는 대상**이 바뀌었다.

---

## 1. 세 키가 실제로 무엇을 건드리는가 (코드 실측)

`MEASURED` — `wavesplat/wavesplat/optimizer/sequential_update.py` 현행본.

| 키 | 읽는 곳 | 도달 조건 |
|---|---|---|
| `varpro_steps` (400) | `:764` `_initialize_learned` **말미** | `init_method == 'learned'` 인 런에서만 |
| `learned_init_iters` (0) | `:740` `_initialize_learned` 결합적합 루프 | 동일 |
| `auto_switch_thresh` (0) | `:806` `_auto_select_init` | `init_method == 'auto'` 인 런에서만 |

- `_initialize_learned` 는 **`:613 if self.init_method == 'learned'`** 로만 들어간다 ⟹
  **`oracle`·`v2_konly`·`v1`·`v2`·`relax*`·`bank_mmse` 팔은 세 키를 한 번도 읽지 않는다.**
- **실측 게이트(`:540–607`)가 라우터보다 먼저 반환**한다 ⟹ **실측 경로는 세 변경 전부와 무관**.
  회귀 시험 `test_unification_did_not_disable_the_real_gate` 가 이를 고정. `MEASURED`
- **라우터가 이제 상수다**. 런 0회 실증(오늘 실행):

```
n_atoms=1  -> learned    n_atoms=7  -> learned
n_atoms=2  -> learned    n_atoms=8  -> learned
n_atoms=5  -> learned    n_atoms=30 -> learned
```

- **얼마나 많은 런이 실제로 이동하는가** — `B` 코퍼스 384런(`A-SHIP`, 구 임계 8)의 실측 라우팅:

| 층 | `learned` | **`v2_konly`(이동 대상)** |
|---|---:|---:|
| 혼잡 15comp | 80/96 | **16 (16.7%)** |
| 혼잡 30comp | 93/96 | **3 (3.1%)** |
| 희소 2comp | 61/96 | **35 (36.5%)** |
| 희소 5comp | 79/96 | **17 (17.7%)** |
| **합** | 313 | **71 / 384 = 18.5%** |

⟹ **혼잡은 9.9%만 이동**(작다) · **희소는 27.1% 이동**(크다). `MEASURED`

---

## 2. 전수 census (120 배치)

범례: ✔ = 그 키를 러너가 **명시**한다 · · = 미명시.
"재실행" = **무효**(전 팔 이동) / **부분**(일부 팔만) / **불변**(비트 동일) / **런0**(추정 런 없음).
⚠**기록손상 열은 만들지 않았다 — 120개 전건 `없음` 이라 열이 상수다.**

| 배치 | `varpro_steps` | `learned_init_iters` | `auto_switch_thresh` | 재실행 | 도달 팔 / 근거 |
|---|:--:|:--:|:--:|:--:|---|
| `AUDIT_recheck_2026-08-13` | · | · | · | **무효** | (`init_method` 부재 ⟹ `DEFAULT_CONFIG` = `auto`) |
| `B_DEPLOY_2026-08-19` | · | ✔ | · | **무효** | `A-SHIP`/`A-VP` 둘 다 라우터 경유(`init_method` 부재). **라우터가 사라져 `ITT` 설계 자체가 소멸** |
| `COMPARE_REAL_SYNTH_2026-08-19` | · | · | · | **무효** | `auto`,`learned` |
| `G1_waveprior_power` | · | · | · | **무효** | (`init_method` 부재 ⟹ `DEFAULT_CONFIG` = `auto`) |
| `HOSGATE_2026-08-13` | · | · | · | **무효** | `SHIP` = `dict(DEFAULT_CONFIG)`. `LETKF_PLUS`·`ZERO` 팔은 면역 |
| `P1_real_path_2026-08-11` | · | · | · | **무효** | `auto` |
| `S11_null_corrected_2026-08-12` | · | ✔ | · | **무효** | `L1-*` 셀이 `AUTO` 파생 (`learned_init_iters` 는 명시) |
| `S13_oos_2026-08-12` | · | · | · | **무효** | `SHIP` = `dict(DEFAULT_CONFIG)` **런타임 스냅샷** ⟹ 세 기본값을 그대로 흡수 |
| `S13b_seed_robust_2026-08-13` | · | · | · | **무효** | 동일 (`run_s13b.py:60`) |
| `S16_BASELINE_2026-08-14` | · | · | · | **무효** | `SHIP` = `init_method=auto`. 기준선 `SPEC*`/`WHITE*` 팔은 면역 |
| `S17_WHITE25_2026-08-18` | · | · | · | **무효** | 동일 — **단 S17 의 PASS 는 기준선 팔만의 성질이라 무영향**(§3) |
| `S2_control_metric_2026-08-08` | · | · | · | **무효** | **전 셀 `AUTO*`** |
| `S6_l1_sweep_2026-08-08` | · | · | · | **무효** | **전 셀 `AUTO*`** |
| `S7_confirm_n32_2026-08-10` | · | · | · | **무효** | **전 셀 `AUTO*`** |
| `S8_deploy_confirm_2026-08-10` | · | · | · | **무효** | **전 셀 `AUTO*`** |
| `S9_lr_x_l1_2026-08-11` | · | · | · | **무효** | **전 셀 `AUTO*`** |
| `W4_SOLVER_2026-08-19` | · | ✔ | · | **무효** | `L-SHIP`/`L-VP` 전부 `learned`. **`varpro_steps` 미명시 ⟹ 패키지 400 + 몽키패치 = 이중 적용, 기준선 팔도 처치를 받는다** |
| `W5_WHY_2026-08-19` | · | ✔ | · | **무효** | `run_w4` 의 `L-*` 팔(전부 `learned`) 재사용. `learned_init_iters` 는 명시하나 **`varpro_steps` 미명시 ⟹ 몽키패치와 이중 적용** |
| `bankmmse_port_2026-07-29` | · | · | · | **무효** | `auto` |
| `confirm40_2026-07-27` | · | ✔ | · | **무효** | (`init_method` 부재 ⟹ `DEFAULT_CONFIG` = `auto`) |
| `discreteness_2026-07-26` | · | · | · | **무효** | (`init_method` 부재 ⟹ `DEFAULT_CONFIG` = `auto`) |
| `figs_opus_diag_2026-07-26` | · | · | · | **무효** | `auto` 진단 그림 — 재생성 시 다른 경로 |
| `figs_program_2026-07-26` | · | · | · | **무효** | `auto` |
| `learned_only_2026-07-26` | · | ✔ | · | **무효** | `learned` |
| `mixed_retrain_2026-07-26` | · | ✔ | · | **무효** | (`init_method` 부재 ⟹ `DEFAULT_CONFIG` = `auto`) |
| `real_gif_2026-07-27` | · | · | · | **무효** | `auto` |
| `unified_2026-07-26` | · | ✔ | · | **무효** | `auto`,`learned` |
| `unified_guard_2026-07-26` | · | ✔ | · | **무효** | `auto`,`learned` |
| `wp_crowded_2026-07-25` | · | · | · | **무효** | (`init_method` 부재 ⟹ `DEFAULT_CONFIG` = `auto`) |
| `wp_integration_2026-07-25` | · | ✔ | · | **무효** | `learned` |
| `B3_rx3_2026-08-06` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `C_ENCOUNTER_2026-08-20` | · | · | · | **부분** | `SHIP`=`auto` 이동 / `V2K`=`v2_konly`·`ZERO` 면역 |
| `DEBATE_blankslate_2026-08-12` | · | · | · | **부분** | `auto`,`v2_konly` |
| `EB_instrument_2026-08-07` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `EC_oos_2026-08-07` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `ED_lead_shell_2026-08-07` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `K6_run_2026-08-07` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `Rx4_tracker_init_2026-08-06` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `Rx5_channel_matched_2026-08-07` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `Rx6_channel_curve_2026-08-07` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `Rx7_tracker_deploy_2026-08-07` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `S15_DIRECTION_2026-08-14` | · | · | · | **부분** | `auto`,`oracle` |
| `S1_indep_sparse_2026-08-08` | · | · | · | **부분** | `AUTO`·`AUTO-G1P1` 이동 / `X-BASE`·`G1`·`P1`·`ORACLE-K` 면역 (셀은 `rx2_init/run_rx2b.CELLS`) |
| `S3_place_x_track_2026-08-08` | · | · | · | **부분** | `AUTO`·`AUTO-NOTRACK`·`AUTO-DUP` 이동 / `ORACLE-K`·`ORACLE-NOTRACK` 면역 |
| `S4_phase_oracle_2026-08-08` | · | · | · | **부분** | `AUTO-NT`·`AUTO-NT-DUP` 이동 / `ORA-*` 면역 ⟹ **PRIMARY(오라클) 무영향** |
| `S5_window_loss_2026-08-08` | · | · | · | **부분** | `AUTO-NT*` 4셀 이동 / `ORA-WPHI-*` 면역 ⟹ **PRIMARY 무영향** |
| `W3_SOLVER_2026-08-15` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `confirm2_2026-07-27` | · | ✔ | ✔ | **부분** | `v2_konly` 팔 면역 / **`auto`·`learned` 팔 이동** (`learned_init_iters=500` 만 명시) |
| `rx2_init_2026-08-05` | · | · | · | **부분** | `auto`,`oracle`,`v2_konly` |
| `BLANKSLATE_2026-08-12` | · | · | · | **불변** | `v2_konly` |
| `W10_SMOOTH_2026-08-22` | · | · | · | **불변** | `ORA-WPHI-*` 전건 — `init_method_readback=oracle` 실측 |
| `W11_UNIFY_2026-08-22` | ✔ | · | · | **불변** | 오늘. `ROUTED`={} 는 배포 그대로, `FORCE-LEARNED` 명시 |
| `W12_UNIFY_REAL_2026-08-22` | · | · | · | **불변** | 오늘. 실측 — 게이트가 먼저 반환 |
| `W13_SPARSE_HEAD2HEAD_2026-08-22` | ✔ | · | · | **불변** | 오늘. 두 팔 다 `init_method` 명시 |
| `W14_CROWDED_HEAD2HEAD_2026-08-22` | ✔ | · | · | **불변** | 오늘. 두 팔 다 `init_method` 명시 |
| `W15_DEPLOY_VERIFY_2026-08-22` | ✔ | · | ✔ | **불변** | **오늘 배포 후 설계** — 두 팔 모두 `varpro_steps` 계승, `PRE` 가 `auto_switch_thresh=8` 명시 |
| `W7_LAMBDA_CROSS_2026-08-22` | · | · | · | **불변** | `ORA-WPHI-*` 전건 — `init_method_readback=oracle` 실측 |
| `W8_VARPRO_DEPLOY_2026-08-22` | ✔ | ✔ | · | **불변** | 오늘. `varpro_steps`·`learned_init_iters` 전 팔 명시 |
| `W9_PENALTY_BUDGET_2026-08-22` | · | · | · | **불변** | `ORA-WPHI-*` 전건 — `init_method_readback=oracle` 실측 |
| `X2_no_learned_2026-08-11` | · | · | · | **불변** | `V2K` = `v2_konly` 명시 |
| `adaptn_probe_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `blind_auto_real_2026-07-26` | · | · | ✔ | **불변** | 실측 배치 — 실측 게이트가 `bank_mmse` 로 먼저 반환. 게다가 `auto_switch_thresh=8` 을 **명시**한 유일한 구(舊) 배치 |
| `e1_grid_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `e2_grid_sep_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `encounter_rescore_2026-07-26` | · | · | · | **불변** | `v2_konly` |
| `heldout40_2026-07-26` | · | ✔ | · | **불변** | `v2_konly` |
| `latency_bug_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `p1_penalty_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `prereg_heldout_2026-08-01` | · | · | · | **불변** | `v2_konly` |
| `relax_extctrl_2026-08-05` | · | · | · | **불변** | `relax`,`relax_konly`,`v2_konly` |
| `rerun_2026-07-24` | · | · | · | **불변** | `v1`,`v2`,`v2_konly` |
| `rerun_2026-07-25` | · | · | · | **불변** | `v2_konly` |
| `rerun_2026-08-03` | · | · | · | **불변** | `v1`,`v2`,`v2_konly` |
| `rx1_tracker_2026-08-05` | · | · | · | **불변** | `oracle`,`v2_konly` |
| `s1s2_2026-08-05` | · | · | · | **불변** | `oracle`,`v2_konly` |
| `s4_within_spread_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `spread_xdesign_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `timing_remeasure_2026-08-05` | · | · | · | **불변** | `v2_konly` |
| `tracker_ampdecay_2026-07-25` | · | · | · | **불변** | `v2_konly` |
| `tracker_debate_2026-08-05` | · | · | · | **불변** | `oracle`,`v2_konly` |
| `A2_realproto_2026-08-06` | · | · | · | **런0** | 추정 런 없음 |
| `AUDIT_full_2026-08-12` | · | · | · | **런0** | 추정 런 없음 |
| `B1_precondition_debate_2026-08-12` | · | · | · | **런0** | 추정 런 없음 |
| `B2_g1p1_ext_2026-08-06` | · | · | · | **런0** | 추정 런 없음 |
| `B3_regime_position_2026-08-12` | · | · | · | **런0** | 추정 런 없음 |
| `DEBATE_novelty_2026-08-14` | · | · | · | **런0** | 추정 런 없음 |
| `E1_effect_size_2026-08-11` | · | · | · | **런0** | 추정 런 없음 |
| `E_depth2_2026-08-06` | · | · | · | **런0** | 추정 런 없음 |
| `E_live_2026-08-06` | · | · | · | **런0** | 추정 런 없음 |
| `K6_scope_2026-08-07` | · | · | · | **런0** | 추정 런 없음 |
| `W6_ALS_2026-08-19` | · | · | · | **런0** | 추정 런 없음 |
| `X1_bank_on_synth_2026-08-11` | · | · | · | **런0** | 추정 런 없음 |
| `X3_audit_and_next_2026-08-11` | · | · | · | **런0** | 추정 런 없음 |
| `audit_2026-08-04` | · | · | · | **런0** | 추정 런 없음 |
| `audit_independence_2026-08-07` | · | · | · | **런0** | 추정 런 없음 |
| `cdip_support_2026-08-03` | · | · | · | **런0** | 추정 런 없음 |
| `collapse_controls_2026-07-26` | · | · | · | **런0** | 추정 런 없음 |
| `depth3_ceiling_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `doppler_real_2026-07-25` | · | · | · | **런0** | 추정 런 없음 |
| `figs_opus_real_2026-07-26` | · | · | · | **런0** | 추정 런 없음 |
| `frf_lidar_2026-08-01` | · | · | · | **런0** | 추정 런 없음 |
| `heldout_real_2026-07-26` | · | · | · | **런0** | 추정 런 없음 |
| `identifiability_debate_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `indep_data_2026-08-07` | · | · | · | **런0** | 추정 런 없음 |
| `litproj_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `obsaxis_2026-07-26` | · | · | · | **런0** | 추정 런 없음 |
| `paper_stage3_2026-08-04` | · | · | · | **런0** | 추정 런 없음 |
| `placement_field_debate_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `realspread_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `routability_2026-08-01` | · | · | · | **런0** | 추정 런 없음 |
| `router_review_2026-07-27` | · | · | · | **런0** | 추정 런 없음 |
| `timing_ledger_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `topicC_depth2_2026-08-06` | · | · | · | **런0** | 추정 런 없음 |
| `topicC_verify_2026-08-06` | · | · | · | **런0** | 추정 런 없음 |
| `transformer_review_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `v1v3_compare_2026-07-27` | · | · | · | **런0** | 추정 런 없음 |
| `waveprior_audit_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `waveprior_debate_2026-08-05` | · | · | · | **런0** | 추정 런 없음 |
| `wp_e2e_2026-07-25` | · | · | · | **런0** | 추정 런 없음 |
| `wp_real_2026-07-25` | · | · | · | **런0** | 추정 런 없음 |

**집계**: 무효 **30** · 부분 **19** · 불변 **31** · 런0 **40** · 합 **120** · **기록손상 0**

### 2.1 census 가 드러낸 함정 — "명시했다" 가 지켜주지 않는다

★★★**`S13`·`S13b`·`HOSGATE` 의 `SHIP` 팔은 `dict(DEFAULT_CONFIG)` 다** (`run_s13.py:52-54`,
`run_s13b.py:60`). 그 배치들의 PREREG 는 *"SHIP is DEFAULT_CONFIG written out explicitly —
no fallback"* 이라 적어 안전을 주장했다. ⛔**그 안전장치는 이 부류의 변경을 막지 못한다** —
막는 것은 *추정기 내부 폴백 표류*이지 *`DEFAULT_CONFIG` 값 표류*가 아니다.
**런타임 스냅샷은 오늘의 기본값을 그대로 흡수한다.** `MEASURED`

⟹ ★**처방: 사전등록 게이트는 `DEFAULT_CONFIG` 를 복사하지 말고 그 시점의 값을 리터럴로 동결하고,
`effective_config` **객체 되읽기**로 대조하라**(S13b 가 이미 도입한 방식이 옳았다).

★**두 번째 함정 = 이중 적용.** `W4`·`W5` 는 VARPRO 를 `tmp/W4_SOLVER_2026-08-19/varpro.py:396`
의 **몽키패치**(`SequentialUpdater._initialize_learned` 자체를 감싼다)로 넣었다. 패키지가 이제
**같은 함수 안에서** VARPRO 를 부른다(`sequential_update.py:764`) ⟹ 그 러너를 오늘 돌리면
**처치 팔은 VARPRO 를 두 번, 기준선 팔(`L-SHIP`)은 한 번** 받는다 = **대조가 붕괴한다.**
(`W8` 은 이걸 알고 `MP-VP` 팔에 `varpro_steps: 0` 을 명시했다 — `gates_w8.py:8`. `MEASURED`)

---

## 3. `STATUS_2026-08-22/STATUS.md` §2 "살아있는 측정" 재판정

⚠**그 §2 표는 11:06 에 쓰였고 `auto_switch_thresh` 배포는 13:58 이다** ⟹ 표가 배포보다 낡았다.

| # | 주장 | 근거 배치 | 도달 팔 | **재판정** |
|---|---|---|---|---|
| 1 | 혼잡 레짐 우위 **(부호)** | S13 56/64 · S13b 61/64 · HOSGATE 14/16 | 셋 다 `auto` | ⚠**"현행 배포의 성질" 로 인용 불가.** 허용 = *"2026-08-22 이전 배포 구성에서"*. 부호가 새 구성에서도 살 **개연성**은 높으나(혼잡은 9.9%만 이동, W14 가 같은 방향 63/64) 그건 `ARGUED` |
| 2 | `SHIP > ZERO` | S16 63~64/64 | `auto` | ✅**살아남되 근거를 갈아야 한다** — 새 경로의 제로필드 이김은 W11/W13/W14 가 **128/128**(해석적은 100/128)로 독립 측정했다. **주장은 더 강해졌고 파일이 바뀐다** |
| 3 | 정보는 충분, 격차는 솔버 (오라클 k 고정 `lstsq` ρ=0.999964) | 런 0회 · 오라클 | 없음 | ✅**완전 무영향** |
| 4 | 진실을 줘도 출하 구성이 진실을 못 지킨다 (**16/16 · 1.5163× · p=2.0e-5**) | W7 | **`ORA-WPHI-*` 전건, `init_method_readback='oracle'` 실측** | ✅**완전 무영향** |
| 5 | 그 이탈은 λ 에 용량-반응 (λ=0 널 / 1e-4 1.52× / 1e-3 3.66×) | W7 | 오라클 | ✅**완전 무영향** |
| 6 | VARPRO 처방이 배포 구성에서 재현 (`B` ITT 혼잡 55/60 −25.28%) | B | `auto` (라우터 경유) | ⚠**수식어를 떼야 한다.** ★`B` 의 가치는 *"배포 라우터를 그대로 두고 ITT"* 였는데 **라우터가 없어져 ITT ≡ PP 가 됐다** ⟹ 그 각주가 소멸. 또 `B` 의 `v2_konly` 71/384 행은 "정확한 동점(no-op)" 이었으나 새 구성에선 처치를 받는다 ⟹ **−25.28% 는 구 라우팅 하의 크기**. ⛔현행 배포의 크기로 인용 금지 |
| 7 | 조우 기하에서도 제로필드를 이긴다 (`C` 13/16 −17.6%) | C | `SHIP`=`auto` 이동 / `V2K`·`ZERO` 면역 | ⚠**이미 `exploratory`. 여기에 "구 라우팅" 표기를 더해야 한다** |

### 3.1 §2 밖이지만 같이 물리는 것

| 항목 | 판정 |
|---|---|
| **`S17` = `PASS`(기준선 축 폐쇄)** | ✅**무영향.** S17 이 실제로 산 것은 *"백색 배경에서 국소화를 줄이면 기준선이 단조 악화한다"* = **기준선 팔만의 성질**이고 `SPEC*`/`WHITE*` 는 세 키를 읽지 않는다. (그 배치의 `SHIP<SPEC25 59/64` 는 이미 인용금지) |
| **`S16` = `UNDECIDED`** | ✅**판정 자체는 무영향**(R-2 실패가 기준선 팔 내부 대조였다). SHIP 열만 이동 |
| **`W8` 이식 16/16 비트 동일** | ✅**무영향** — 몽키패치 vs 패키지 비교이고 라우터와 무관 |
| **`W11` `ROUTER-UNNECESSARY`** | ✅**무영향, 그리고 이 배포의 유일하게 올바른 근거다** — `ROUTED`={} 가 **진짜 배포 팔**(라우터 + median-of-3 앙상블)이었다 |
| **`W13`/`W14` 63/64·27/64** | ⚠**비교자 결함**(§5-④). 판정은 유지, 크기는 낙관 편향 가능 |
| **`S4`·`S5` 오라클 PRIMARY** (`OPTIMIZER-FAILURE`, 26.9×) | ✅**무영향** — `ORA-*` 셀 |
| **`S6`–`S9` λ 축** | ⚠**전 셀 `AUTO*` ⟹ 재실행 전건 이동.** 단 λ 축은 W3·W7 로 이미 종결이라 재측정 동기가 없다 |
| **`K6`(오라클 k)·`Rx4`–`Rx7`·`E-B/C/D`** | ⚠**부분** — `AUTO` 셀만 이동. 이 배치들의 결론은 대부분 이미 사망/폐쇄 |
| **실측 계열 전부**(`R1`·`W12`·`D`·`bankmmse_port`·`blind_auto_real`·`real_gif`·`heldout_real`) | ✅**구조적으로 무영향** — 실측 게이트가 라우터보다 먼저 `bank_mmse` 로 반환한다 |

---

## 4. 재측정 우선순위 + 런 수 추정

**비용 기준(실측)**: `STATUS_2026-08-22` §6 = **48런 ≈ 4분**(혼잡, 8워커) ⟹ **≈ N/12 분**.
★**절약 규칙**: 기준선 팔(`LETKF*`·`SPEC*`·`ZERO`·`WHITE*`)은 세 키를 읽지 않으므로
**같은 시드로 다시 돌리면 비트 동일**이다 ⟹ **`SHIP` 팔만 재실행하고 기록된 기준선 열을 재사용**할 수 있다.
⚠**단 신선 시드 규율을 지키려면 기준선도 그 시드로 돌려야 한다** — 아래는 **기존 시드 재사용** 전제다.

### 4.0 ⛔**감사 중 발견: `W15` 가 죽어 있다** (`MEASURED`, 이 감사의 부수 산출)

- `results.jsonl` **124행 / 768**(16%), **전부 `PRE` 팔** ⟹ **`POST` 행 0개 = 아직 아무것도 못 쟀다.**
- `run.log` 는 `[124/768]` 에서 끊기고 이후 **`BrokenPipeError` 캐스케이드**(`multiprocessing.pool`
  워커들이 부모에게 쓰다 실패) · `watchdog.log` 도 **14:10:15 · 115/768 에서 정지** ·
  `ps` 에 `run_w15` **0개** ⟹ **부모 프로세스가 통째로 죽었다**(워커 예외가 아니다).
- **OOM 아님(정황)**: 현재 여유 14 GB. **런처 세션 종료로 프로세스 그룹이 같이 죽은 것**으로 보인다. `ARGUED`
- ✅**재개 가능**: `run_w15.py:51` 이 `todo = [j for j in jobs if j['job_id'] not in done]` ⟹
  **124행을 읽고 남은 644런만 돈다.** 같은 명령을 다시 띄우면 된다.
- ⚠**동시에 감시자도 함께 죽었다** ⟹ ★**`feedback_watchdog_supervision_split` 의 의도(감시자 분리)가
  프로세스 그룹 수준에서는 지켜지지 않았다** — 러너와 감시자가 **같은 부모에 매달려 있다**.
  ★**처방: 감시자를 `setsid`/`nohup` 으로 러너와 다른 세션에 띄울 것.**
- ✅**살아있는 부분 정보**: `PRE` 124행에 `route=learned` 와 `route=v2_konly` 가 **둘 다 있다**
  ⟹ 게이트 `G-2`(대조가 실재하나)는 현재까지 만족.

| 순위 | 무엇 | 왜 | 런 수 | 벽시계 |
|:--:|---|---|---:|---:|
| **P0** | ⛔**`W15` 는 죽었다 — 재개 필요** (§4.0) | 배포 전/후 ITT. **`PRE` 팔이 진짜 구 배포**(라우터 + 앙상블)라 §3 #1·#6·#7 의 이동 폭을 직접 준다 | **잔여 644** | ~54분 |
| **P1** | **혼잡 헤드라인 재측정** — `SHIP`(통일) vs 기록된 `LETKF_PLUS`·`ZERO` | §3 #1. 논문의 유일한 살아있는 **부호** 주장 | **384** (128ep×3시드, SHIP만) | **~32분** |
| **P2** | **`HOSGATE` 비선형 게이트, `SHIP` 팔만** | 상시지시 *"검증에 비선형 필수"* 를 만족하는 **유일한 라인**. 매우 싸다 | **64** (16셀×2 GT×2시드) | **~5분** |
| **P3** | **분산 처방 재인가** — 학습 경로의 시드-간 분산 | ★`analytic_ensemble_k` 앙상블이 구조적으로 죽었다(§5-①). **CV 53% 처방이 배포에서 사라졌는데 대체물이 없다** | **160** (32ep×5시드) | **~14분** |
| **P4** | **`C` 조우, `SHIP` 팔만** | §3 #7. `exploratory` 이므로 급하지 않음 | ~150 | ~13분 |
| **P5** | `W13`/`W14` 를 **앙상블된 해석적 팔**로 재측정 | §5-④ 비교자 결함 교정. **W15 가 대체할 수 있다** | 256 | ~21분 |
| ⛔ | `S16`/`S17` `SHIP` 열 | 기준선 축 **정지 지시** + 이미 `UNDECIDED`/폐쇄 | — | — |
| ⛔ | `S6`–`S9` λ 축 | W3·W7 로 종결 | — | — |

**P1+P2+P3 = 608런 ≈ 51분**으로 §2 표의 #1 을 새 구성에서 되살릴 수 있다. `ARGUED`(비용 추정)

⚠**시드 원장**: `W13`=1620–1622 · `W14`=1623–1625 · `W15`=1626–1628 소모 실측 ⟹ **잔여 ≈ [1629–1818]**.
⛔`STATUS_2026-08-22` §6 의 **"[1619–1818] = 200개" 는 낡았다.**

---

## 5. ★커밋 메시지에 없는 부수효과 (임무 5)

### ① ★★★`analytic_ensemble_k` 앙상블이 **합성에서 구조적으로 사망**했다 — 커밋 어디에도 없다

`MedianEnsembleUpdater.initialize` 는 **`route == 'v2_konly'` 일 때만** 멤버를 만든다:

```
sequential_update.py:1612-1616
        route = self._primary.init_method
        real_gate = bool(getattr(self._primary, '_auto_real_gate', False))
        if self.k_requested <= 1 or route != 'v2_konly' or real_gate:
            self.ensemble_k = 1
```

`auto_switch_thresh = 0` ⟹ **합성에서 route 는 항상 `learned`** ⟹ **`ensemble_k` 는 항상 1**.
런 0회 실증(오늘): `make_updater(DEFAULT_CONFIG)` → `MedianEnsembleUpdater`, `k_requested=3`,
그러나 라우터가 `n_atoms∈{1,2,5,7,8,30}` 전건에서 `learned` 를 고른다. `MEASURED`

**그런데 `DEFAULT_CONFIG['analytic_ensemble_k']` 는 여전히 3 이고 yaml 도 3 이고 시험도 3 을 지킨다.**
⟹ ★**λ·`v2_konly` 폴백과 정확히 같은 부류의 "죽은 기본값" 이 하나 새로 생겼다.**

**무엇이 사라졌나**: 2026-07-27 `confirm2` 가 인가한 처방 = *"해석적 팔의 위상 난수 초기화가
CV 53% 의 이봉 분산을 만든다 ⟹ 예측 **필드**의 median-of-3"*. 신선 40 에서 희소 위반 0/80,
**28 레인 불일치·8 건 실제 수리**로 인가됐다. **그 처방이 배포에서 사라졌고 대체물이 없다.**
⚠`CLAUDE.md` 는 이 앙상블이 *"런의 14%(36/256)에서만 작동"*(S13b 실측)이라 적었다 — **이제 0%다.**

### ② ★★★저장소 자신의 주석이 이 상태를 *"배포된 적도 측정된 적도 없다"* 고 선언한다

`sequential_update.py:104-111` (**2026-08-21 작성 = 배포 하루 전**):

> `A bare SequentialUpdater({}) carrying 'auto' would be a THIRD behaviour — routed but
> un-ensembled — that has never shipped and has never been measured.`

**그 "제3의 거동" 이 오늘부터 배포 거동이다**(합성 · 체크포인트 존재 시).
⟹ ★**주석이 지금 거짓이고, 그 거짓이 배포 계약에 대한 진술이다.**
⚠단 **완화**: `learned` 경로 자체는 W4·W5·`B`·W11·W13·W14 가 광범위하게 쟀다. 미측정인 것은
**"라우터가 `v2_konly` 를 골랐을 러너들이 앙상블 없이 학습으로 가는 것"** = 희소 27.1%·혼잡 9.9% 다.
★**`W15` 가 그것을 재도록 설계됐으나 죽었다(§4.0) — 현재 `POST` 행 0개.** `MEASURED`(주석) / `ARGUED`(완화)

### ③ ★`pre_varpro_config()` 되돌리기 헬퍼가 **오늘 아침에 이미 낡았다**

`sequential_update.py:77-85` 의 docstring:
> *"★스위치가 하나가 아니라 **둘**이다. 한쪽만 끄면 배포된 적도 측정된 적도 없는 제3의 팔이 된다."*

이 헬퍼는 라운드2 감사(`554c838`, 08-22 오전)가 **정확히 이 결함을 막으려고** 만든 것인데,
`d673892`(13:58) 이후 **스위치는 셋**이다. `pre_varpro_config()` 만으로 되돌리면
`auto_switch_thresh` 가 0 으로 남아 **또 다른 제3의 팔**이 나온다.
⟹ ★★**같은 결함이 같은 날 두 번째로 재현됐다** — 그 헬퍼가 막으려던 바로 그 결함으로.
`AUTO_SWITCH_THRESH_PRE_UNIFY = 8` 은 정의만 돼 있고 **셋을 묶는 단일 헬퍼가 없다.** `MEASURED`

### ④ ★`W13`/`W14` 의 해석적 팔이 **배포된 해석적 팔보다 약했다** (배포 근거의 비교자 결함)

`gates_w13.py:17` / `gates_w14.py:17` = `'ANALYTIC': {'init_method': 'v2_konly'}`.
`make_updater` 는 **`init_method=='auto'` 일 때만** 앙상블 래퍼를 준다(`:1702-1705`) ⟹
그 `ANALYTIC` 팔은 **median-of-3 필드 앙상블이 없는 단일 런**이다.
반면 **배포가 실제로 쓰던 해석적 경로에는 앙상블이 있었다.**
⟹ ⚠**커밋 메시지가 인용한 `혼잡 63/64 −50.98%`·`희소 27/64 +49.2%` 는
"학습 vs (배포보다 약한) 해석적" 이다.** 앙상블의 문서화된 이득이 **희소 폭발 수리**였으므로
**희소 비용은 과소평가됐을 수 있다**(방향: 진짜 배포 해석적이 더 좋았을 것 ⟹ 학습이 더 불리).
✅**다만 `W11` 은 이 결함이 없다** — `ROUTED`={} 가 진짜 배포 팔이었고, 거기서 라우터는
16/44 로 지고 **희소 꼬리 +1183.9%** 를 보였다. **⟹ 배포의 하중은 W11 이 진다.** `MEASURED`
⚠크기 편향의 **정도는 `UNVERIFIED`** (W15 재개 시 답한다 — §4.0).

### ⑤ `init_iters` 결과 열이 학습 런에서 **상수 0** 이 됐다

`learned_init_iters=0` ⟹ `init_result['n_iters'] = 0`. `S8`(`analyze_s8.py:476`)·`S9`
(`analyze_s9.py:286`)가 이 값을 **공변량으로 상관 분석**한다 ⟹ 재실행 시 **퇴화 공변량**.
하드 게이트로 쓰는 곳은 **0건**(전수 grep) ⟹ 오류는 안 나고 조용히 무의미해진다. `MEASURED`

### ⑥ 지연·비용 원장이 배포를 더는 서술하지 않는다

`confirm2` 인가문의 *"멤버당 6.5 ms/frame → k=3 에서 19.7 ms, 100 ms 예산 안"* 은
**앙상블이 안 도니 이제 단일 멤버 비용**이다(프레임 지연은 내려간다). 반대로 **초기화**에는
VARPRO 가 **`vp_wall` ≈ 2.50 s** 를 더한다. ⟹ **두 방향으로 동시에 틀렸다.** `MEASURED`

### ⑦ "통일" 은 **체크포인트 존재에 조건부**다

`_auto_select_init` 의 `except (FileNotFoundError, OSError, ImportError)` 분기
(`:821-830`)는 여전히 `'v2_konly'` 를 반환한다 ⟹ 체크포인트가 없으면 **해석적 + 앙상블 3** 으로
되돌아간다. **통일은 절대적이지 않고, 그 이탈 경로는 앙상블까지 함께 켠다.** `MEASURED`

### ⑧ 시험 두 개가 공허해졌다

- `test_ensemble_k3_members_seeds_and_pointwise_median`(`test_deploy_auto_v1_m3.py:197`)은
  `init_method='v2_konly'` 를 명시해 앙상블을 만든다 ⟹ **배포 라우터가 결코 만들지 않는 구성**을 시험한다.
- `test_auto_router_follows_the_shipped_threshold` 는 `want` 를 코드와 **같은 식**으로 계산하므로
  라우팅 단언이 항등식이다(살아있는 단언은 `_auto_n_selected == n_atoms` 뿐).
⟹ ⚠`CLAUDE.md` 가 이미 적은 *"테스트가 배포 기본값을 지키지 않는다"* 결함 부류의 **재발**. `MEASURED`

### ⑨ 무영향 확인 (부수효과 아님을 실측)

- **실측 경로**: 게이트가 `:540-607` 에서 먼저 반환 ⟹ 세 키 전부 미도달. `real_bank_ensemble_k=3` 유지.
- **`oracle`/`v2_konly`/`v1`/`v2`/`relax*` 팔**: `_initialize_learned` 미진입 ⟹ 비트 동일.
- **`w_phi` 추적·`lambda_l1`·`use_dispersion`·`adaptive_n`**: 미접촉.

---

## 6. 메모리·문서 전파 누락 (⛔고치지 않음 — 목록만)

| 파일 | 누락/어긋남 | 심각도 |
|---|---|:--:|
| **`CLAUDE.md`** | ⛔**세 변경 전부 0회 언급.** "Important Design Decisions" 절이 `init_method: auto` 를 여전히 *"GT-free 라우터"* 로 서술 — 이제 **상수 라우터**다 | ★★★ |
| **`CLAUDE.md`** | `analytic_ensemble_k` 를 *"런의 14%(36/256)에서만 작동"* 이라 적음 ⟹ **이제 0%**. 그 줄이 이 배포의 부수효과를 가리는 위치에 있다 | ★★★ |
| **`CLAUDE.md`** | `sequential_update.py:426`·`:967` λ 하드코딩 경고는 유지되나, **같은 부류의 신규 죽은 기본값 3개**(`analytic_ensemble_k`, `MedianEnsembleUpdater` 래퍼, `auto_guard`)가 미기재 | ★★ |
| **`MEMORY.md`** | ⛔**`auto_switch_thresh` 8→0 이 0회 언급.** VARPRO 배포(키 2개)만 08-20~08-21 묶음에 있다 | ★★★ |
| **`MEMORY.md`** | ⛔**W9·W10·W11·W12·W13·W14·W15 묶음이 없다** (W9 만 STATUS §5.5 에 한 줄) | ★★★ |
| **`MEMORY.md`** | RESUME(line 8) 의 *"초기화 24 s ⟹ 08-19 VARPRO 가 8.5 s"* 가 라운드2 정정(**22.50→9.52 s = 2.37배**)을 반영 안 함 | ★★ |
| **`MEMORY.md`** | `STATUS_2026-08-21` 묶음이 아직 최상단 — `STATUS_2026-08-22` 가 그것을 **대체**한다는 표시 없음 | ★ |
| **`reference_citation_bans.md`** | ⛔**`auto_switch_thresh` 관련 금지 0건.** `varpro_steps=0` 금지(08-22 라운드2)만 있고, **그 항목 자체가 이제 불완전**하다(*"스위치는 둘"* → 셋) | ★★★ |
| **`reference_citation_bans.md`** | ⛔**"S13·S13b·HOSGATE·B·C 를 현행 배포의 성질로 인용 금지" 가 미기재.** 그 문장은 **커밋 `d673892` 메시지 안에만** 있다 ⟹ 인용금지 목록을 열어보는 규율로는 도달 불가 | ★★★ |
| **`STATUS_2026-08-22/STATUS.md` §1** | ⛔**표에 `auto_switch_thresh` 행이 없다**(11:06 작성 vs 13:58 배포). *"✅끄는 법: `varpro_steps=0` + `learned_init_iters=500`"* 도 **이제 틀렸다**(세 번째 스위치 필요) | ★★★ |
| **`STATUS_2026-08-22/STATUS.md` §2** | #1·#6·#7 이 §3 의 재판정을 반영 안 함 | ★★★ |
| **`STATUS_2026-08-22/STATUS.md` §0·§5.5** | 오늘 사건 표에 W10–W15 없음. **테스트 수 "221"/"223" ↔ 현재 226** | ★★ |
| **`STATUS_2026-08-22/STATUS.md` §6** | 시드 잔여 **[1619–1818]** 은 W13·W14·W15 소모(1620–1628) 미반영 ⟹ **[1629–1818]** | ★★ |
| **`configs/estimation.yaml`** | `varpro_steps` 줄 주석 *"⛔0 하나로는 배포 전 거동이 아니다 (아래와 함께 바꿀 것)"* 가 **`auto_switch_thresh` 를 포함하지 않는다** — 세 줄이 인접해 있는데 묶는 문장이 없다 | ★★ |
| **`sequential_update.pre_varpro_config()`** | docstring 이 *"스위치가 **둘**"* 이라 단언 — **셋**이다 (§5-③) | ★★★ |
| **`sequential_update.py:104-111`** | *"routed but un-ensembled … never shipped, never measured"* 가 **오늘부터 거짓** (§5-②) | ★★★ |

---

## 7. 범위와 한계 (열람 범위 = 주장 범위)

**열었다**: 커밋 `98883a8`·`d673892` 전문 diff · `sequential_update.py`(상수부·`initialize`·
`_initialize_learned`·`_auto_select_init`·`MedianEnsembleUpdater`·`make_updater`) · `run.py`
(`DEFAULT_CONFIG`·`load_yaml_config`·`run_estimation` 병합) · `configs/estimation.yaml` ·
`test_deploy_auto_v1_m3.py`·`test_shipping_defaults.py`(diff) · `tmp/` 120 배치 전 `.py` 의
정규식 스캔 + 15개 배치의 팔 정의 직접 열람 · `B_DEPLOY/results.jsonl`(384행 라우팅 집계) ·
`W7/W9/W10/results.jsonl` 첫 행(`init_method_readback`) · `W11/RESULT.md` ·
`W15/PREREG.md`·`gates_w15.py` · `MEMORY.md` · `reference_citation_bans.md` ·
`CLAUDE.md` · `STATUS_2026-08-22/STATUS.md`.

**안 열었다 ⟹ 주장 안 함**: `wavesplat/optimizer/varpro.py` 수학 검증(W8 이 이미 이식 충실도를
16/16 비트 동일로 쟀다) · `results/` 의 1,000여 JSON · 논문/학위논문 본문 · `scripts/` 413개 ·
`tmp/fowd_2026-08-02`(venv).

**추정 런 0회.** §1 의 라우터 실증과 §5-① 의 래퍼 확인은 **에피소드를 로드하지 않는
객체 생성·메서드 호출**이고 추정을 돌리지 않았다.

**미확인(`UNVERIFIED`)**: ⓐ §5-④ 비교자 결함의 **크기**(W15 재개 시 답한다 — §4.0) ·
ⓑ 재측정 시 §3 #1 의 **부호가 유지되는지** · ⓒ 학습 경로의 시드-간 분산이 해석적 경로의
CV 53% 보다 작은지(= 앙상블 소실이 실제로 해로운지) — **P3 가 그것을 재는 배치다** ·
ⓓ `varpro_steps` 를 안 적은 러너 중 몽키패치 이중 적용이 실제로 **에러가 아니라 조용한 오염**인지
(코드 읽기로는 조용한 오염이나 실행 확인은 안 했다).
