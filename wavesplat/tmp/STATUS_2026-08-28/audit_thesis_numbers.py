#!/usr/bin/env python3
"""논문 정합 감사 #4 — **인용 수치의 단일출처 대조**(손복사 사냥). 런 0회.

★설계: 기대 문자열을 **verdict JSON 에서 계산**하고, 그 문자열이 논문 본문에
실제로 있는지 찾는다. 논문 쪽 정규식은 '어디를 보는지' 표시일 뿐 판정에 쓰지 않는다.
⟹ verdict 가 바뀌면 이 검사가 먼저 깨진다(= 논문이 낡았다는 신호).
불일치가 하나라도 있으면 exit 1.
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_numbers.py
"""
import json, re, sys, hashlib, pathlib

ROOT = pathlib.Path("/home/vexit/wavesplat/wavesplat")
TEX  = ROOT / "thesis/thesis_main.tex"
lines = TEX.read_text(encoding="utf-8").splitlines()
print(f"# tex md5 = {hashlib.md5(TEX.read_bytes()).hexdigest()}")

BODY = []
for i, l in enumerate(lines, 1):
    if l.lstrip().startswith('%'):
        continue
    b = re.sub(r'(?<!\\)%.*$', '', l)
    if b.strip():
        BODY.append((i, re.sub(r'[{}$\\]|\\,|mathbf|textbf|emph|texttt', '', b)))

_c = {}
def jget(rel, path):
    if rel not in _c:
        _c[rel] = json.load(open(ROOT / "tmp" / rel))
    cur = _c[rel]
    for k in path.split('.'):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return cur

B16 = "B16_KGRID_2026-08-24/verdict_b16.json"
B19 = "B19_KGRID_SPARSE_2026-08-25/verdict_b19.json"
B20 = "B20_REFIT_CONTROL_2026-08-27/verdict_b20.json"
B20B= "B20b_QUANT_CROWDED_2026-08-27/verdict_b20b.json"
B21 = "B21_MEDIATION_2026-08-27/verdict_b21.json"
B22 = "B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json"
B25 = "B25_CONFWEIGHT_2026-08-27/verdict_b25.json"
B26 = "B26_SEASTATE_2026-08-27/verdict_b26.json"
M2  = "M2_LOCTUNE_2026-08-28/verdict_m2.json"
M4  = "M4_CAPACITY_2026-08-28/verdict_m4.json"
M5B = "M5b_WINDOW_2026-08-28/verdict_m5b.json"
M8B = "M8b_LADDER_2026-08-28/verdict_m8b.json"
M1  = "M1_NONLIN_REAL_2026-08-28/verdict_m1.json"
M3  = "M3_SUPPORT_2026-08-28/verdict_m3.json"
# ── 2026-08-30 추가 (M-7): 감독 축 + 분포 이동 검사. 판정 파일이 셋으로 갈린다.
M7  = "M7_SUPERVISION_2026-08-28/verdict_m7.json"
M7A = "M7_SUPERVISION_2026-08-28/verdict_m7_e1.json"
M7B = "M7_SUPERVISION_2026-08-28/verdict_m7_e2.json"

def pct(v, n=2):  return f"{v:.{n}f}"
def cnt(v, tot):  return f"{v}/{tot}"

CHECKS = [
 ("B-16 혼잡 절제 승수",      B16, "primary.better",                    lambda v: cnt(v,48)),
 ("B-16 혼잡 절제 크기",      B16, "primary.rel_median_pct",            lambda v: pct(v)),
 ("B-16 용량반응 승수",       B16, "dose.better",                       lambda v: cnt(v,48)),
 ("B-19 희소 절제 크기",      B19, "primary.rel_median_pct",            lambda v: pct(v)),
 ("B-19 절제팔 vs 제로",      B19, "neg_kgrid1_vs_zero.better",         lambda v: cnt(v,16)),
 ("B-20 (a) 승수",           B20, "primary_a.better",                  lambda v: cnt(v,16)),
 ("B-20 (a) 크기",           B20, "primary_a.rel_median_pct",          lambda v: "+"+pct(v,3)),
 ("B-20 (b) 크기",           B20, "primary_b.rel_median_pct",          lambda v: pct(v)),
 ("B-20 |dphi| 중앙",        B20, "median_phi_abs_change_max.NOOP-REFIT",lambda v: f"{v:.3f}"),
 ("B-20 |dw| 중앙",          B20, "median_w_abs_change_max.NOOP-REFIT", lambda v: f"{v:.4f}"),
 ("B-20 시드밴드 pp",        B20, "magnitudes.pp_band",                lambda v: pct(v)),
 ("B-20 재적합 몫(로그)",     B20, "log_decomposition.median_d_refit",   lambda v: f"{v:.4f}"),
 ("B-20 재적합 몫(%)",       B20, "log_decomposition.share_of_medians", lambda v: pct(v*100)),
 ("B-20 게이트",             B20, "gates_pass",                        lambda v: cnt(v,28)),
 ("B-20b 혼잡 직접대조 승수",  B20B,"primary.better",                    lambda v: cnt(v,48)),
 ("B-20b 크기",              B20B,"primary.rel_median_pct",            lambda v: pct(v)),
 ("B-20b 중앙 nRMSE a",      B20B,"primary.median_a",                  lambda v: f"{v:.4f}"),
 ("B-20b 중앙 nRMSE b",      B20B,"primary.median_b",                  lambda v: f"{v:.4f}"),
 ("B-20b 처치팔 vs 제로",     B20B,"negative.n2.better",                lambda v: cnt(v,48)),
 ("B-20b 연역 하한",         B20B,"thresholds.deducible_lower_bound",   lambda v: str(v)),
 ("B-21 게이트",             B21, "n_gate_pass",                       lambda v: cnt(v,24)),
 ("B-21 희소 매개 승수",      B21, "strata.sparse.primary.better",       lambda v: cnt(v,16)),
 ("B-21 혼잡 매개 승수",      B21, "strata.crowded.primary.better",      lambda v: cnt(v,48)),
 ("B-21 혼잡 교락 승수",      B21, "strata.crowded.confound.better",     lambda v: cnt(v,48)),
 ("B-21 혼잡 교락 크기",      B21, "strata.crowded.confound.rel_median_pct", lambda v: pct(v)),
 ("B-21 혼잡 몫(%)",         B21, "strata.crowded.attribution_share.share", lambda v: pct(v*100)),
 ("B-21 희소 몫(%)",         B21, "strata.sparse.attribution_share.share",  lambda v: pct(v*100,3)),
 ("B-21 TOST 밴드안(혼잡)",   B21, "strata.crowded.confound_tost.n_inside",  lambda v: str(v)),
 ("B-22 L2 승수",            B22, "comparisons.L2.better",             lambda v: cnt(v,48)),
 ("B-22 L2 크기",            B22, "comparisons.L2.rel_median_pct",     lambda v: pct(v)),
 ("B-22 LEARN-NOVP vs 제로", B22, "neg_vs_zero.LEARN-NOVP.better",     lambda v: cnt(v,48)),
 ("B-22 LEARN-NOVP 크기",    B22, "neg_vs_zero.LEARN-NOVP.rel_median_pct", lambda v: pct(v,1)),
 ("B-22 ANA-SHIP vs 제로",   B22, "neg_vs_zero.ANA-SHIP.better",       lambda v: cnt(v,48)),
 ("B-22 ANA-SHIP 크기",      B22, "neg_vs_zero.ANA-SHIP.rel_median_pct",lambda v: pct(v,1)),
 ("B-22 ANA-NOVP vs 제로",   B22, "neg_vs_zero.ANA-NOVP.better",       lambda v: cnt(v,48)),
 ("B-22 ANA-N-NOVP vs 제로", B22, "neg_vs_zero.ANA-N-NOVP.better",     lambda v: cnt(v,48)),
 ("B-25 P 승수",             B25, "comparisons.P.better",              lambda v: cnt(v,48)),
 ("B-25 P 크기",             B25, "comparisons.P.rel_median_pct",      lambda v: pct(v)),
 ("B-25 S2 승수",            B25, "comparisons.S2.better",             lambda v: cnt(v,48)),
 ("M2 SS-REAL SPEC12 승수",  M2,  "results.SS-REAL.d2.SPEC12.better",  lambda v: cnt(v,24)),
 ("M2 SS-REAL SPEC12 크기",  M2,  "results.SS-REAL.d2.SPEC12.rel_median_pct", lambda v: "+"+pct(v)),
 ("M2 SS-REAL SPEC6 크기",   M2,  "results.SS-REAL.d2.SPEC6.rel_median_pct",  lambda v: "+"+pct(v)),
 ("M2 SS-BF4 SPEC12 승수",   M2,  "results.SS-BF4.d2.SPEC12.better",   lambda v: cnt(v,24)),
 ("M2 SS-BF4 SPEC12 크기",   M2,  "results.SS-BF4.d2.SPEC12.rel_median_pct",  lambda v: "+"+pct(v)),
 ("M2 SS-BF4 SPEC6 크기",    M2,  "results.SS-BF4.d2.SPEC6.rel_median_pct",   lambda v: "+"+pct(v)),
 ("M2 SS-SHORT SPEC12 승수", M2,  "results.SS-SHORT.d2.SPEC12.better", lambda v: cnt(v,24)),
 ("M2 SS-SHORT SPEC6 승수",  M2,  "results.SS-SHORT.d2.SPEC6.better",  lambda v: cnt(v,24)),
 ("M2 MDE(SS-REAL)",         M2,  "results.SS-REAL.mde.pct_below_confirm",    lambda v: pct(v)),
 ("M2 skill SS-REAL",        M2,  "results.SS-REAL.stats.SPEC25.skill",       lambda v: pct(v*100)),
 ("M2 skill SS-SHORT",       M2,  "results.SS-SHORT.stats.SPEC25.skill",      lambda v: pct(v*100)),
 ("M2 skill SS-BF4",         M2,  "results.SS-BF4.stats.SPEC25.skill",        lambda v: pct(v*100)),
 ("M2 SS-REAL vs 제로 크기",  M2,  "results.SS-REAL.ship_vs_zero.rel_median_pct", lambda v: pct(v,1)),
 # ── 2026-08-28 추가: M-4(용량) · M-5b(창) · M-8b(사다리) ──────────────────
 # ⛔손복사 금지. 기대 문자열은 전부 verdict 에서 계산한다.
 ("M-4 실측 K-MAX 승수",     M4,  "primary.better",                    lambda v: cnt(v,12)),
 ("M-4 실측 K-MAX 크기",     M4,  "primary.rel_median_pct",            lambda v: f"{v:.0f}"),
 ("M-4 합성 K-MAX 승수",     M4,  "syn.K-MAX.better",                  lambda v: cnt(v,48)),
 ("M-4 K-8 승수(탐색적)",     M4,  "real.K-8.better",                   lambda v: cnt(v,12)),
 ("M-4 기준선 vs 제로",      M4,  "real.skill.SHIP-K.beats_zero",      lambda v: cnt(v,12)),
 ("M-4 K-MAX vs 제로",      M4,  "real.skill.K-MAX.beats_zero",       lambda v: cnt(v,12)),
 ("M-4 게이트",              M4,  "gates",                             lambda v: cnt(len(v),len(v))),
 ("M-5b PROP 승수",          M5B, "primary.better",                    lambda v: cnt(v,12)),
 ("M-5b PROP 크기",          M5B, "primary.rel_median_pct",            lambda v: pct(v,1)),
 ("M-5b 판별자 승수",         M5B, "discriminator.better",              lambda v: cnt(v,12)),
 ("M-5b 판별자 크기",         M5B, "discriminator.rel_median_pct",      lambda v: pct(v,1)),
 ("M-5b 용량반응 TP6",        M5B, "dose_response.0.better",            lambda v: cnt(v,12)),
 ("M-5b 용량반응 TP4",        M5B, "dose_response.1.better",            lambda v: cnt(v,12)),
 ("M-5b SEC8 판별자 크기",    M5B, "extra.sec8_ship.rel_median_pct",    lambda v: pct(v,1)),
 ("M-5b ORIG2 승수",         M5B, "extra.orig2_prop.better",           lambda v: cnt(v,12)),
 ("M-5b ORIG2 크기",         M5B, "extra.orig2_prop.rel_median_pct",   lambda v: "+"+pct(v,1)),
 ("M-5b 밀도 ANCHOR",        M5B, "density_disclosure.0.dens_per_s",   lambda v: f"{v:.1f}"),
 ("M-5b 밀도 TP2",           M5B, "density_disclosure.3.dens_per_s",   lambda v: f"{v:.1f}"),
 ("M-5b 게이트",             M5B, "gates",                             lambda v: cnt(len(v),len(v))),
 ("M-8b 끝점 승수",          M8B, "endpoint.worse",                    lambda v: cnt(v,12)),
 ("M-8b 끝점 크기",          M8B, "endpoint.rel_median_pct",           lambda v: pct(v,1)),
 ("M-8b PRIMARY 승수",       M8B, "primary.better",                    lambda v: cnt(v,12)),
 ("M-8b PRIMARY 크기",       M8B, "primary.rel_median_pct",            lambda v: pct(v)),
 ("M-8b MDE q80(단일검정)",    M8B, "thresholds.q80_primary",            lambda v: f"{v:.3f}"),
 # ── 2026-08-29 추가 (AMENDMENT_2): 가족 MDE 와 가법성 두 수. ⛔손복사 금지 —
 #   기대 문자열은 verdict 에서 계산한다. 재유도 단일출처는
 #   tmp/M8b_LADDER_2026-08-28/factorization_check.py (동결 /chain·/endpoint 에서 다시 만든다).
 ("M-8b MDE q80(가족 Holm)",   M8B, "thresholds.q80_family",             lambda v: f"{v:.3f}"),
 ("M-8b 가족 최소 승수",        M8B, "thresholds.holm_first_min_wins",    lambda v: cnt(v,12)),
 ("M-8b ΣΔ_k",               M8B, "amendment_2.factorization.sum_delta_log",      lambda v: f"{v:.6f}"),
 ("M-8b 끝점 Δlog",           M8B, "amendment_2.factorization.endpoint_delta_log", lambda v: f"{v:.6f}"),
 ("M-8b 승수 곱",             M8B, "amendment_2.factorization.mult_sum",           lambda v: f"{v:.5f}"),
 ("M-8b 끝점 승수",           M8B, "amendment_2.factorization.mult_endpoint",      lambda v: f"{v:.5f}"),
 ("M-8b 가법성 격차 %",        M8B, "amendment_2.factorization.gap_pct",            lambda v: f"{v:.3f}"),
 ("M-8b L1 승수",            M8B, "chain.0.worse",                     lambda v: cnt(v,12)),
 ("M-8b L2 승수",            M8B, "chain.1.worse",                     lambda v: cnt(v,12)),
 ("M-8b L6 승수",            M8B, "chain.5.worse",                     lambda v: cnt(v,12)),
 ("M-8b BANK 절대 중앙",      M8B, "magnitude.per_rung.0.abs_nrmse_median",  lambda v: f"{v:.3f}"),
 ("M-8b PROP 절대 중앙",      M8B, "magnitude.per_rung.8.abs_nrmse_median",  lambda v: f"{v:.3f}"),
 ("M-8b BANK vs 참제로",      M8B, "magnitude.per_rung.0.vs_true_zero_median", lambda v: f"{v:.3f}"),
 ("M-8b PROP vs 참제로",      M8B, "magnitude.per_rung.8.vs_true_zero_median", lambda v: f"{v:.3f}"),
 ("M-8b 게이트",             M8B, "gates",                             lambda v: cnt(len(v),len(v))),
 # ── 2026-08-29 추가: M-1(비선형 × 실측 해상상태) · M-3(학습 지지집합 구조 좌표) ──────
 # ⛔손복사 금지. 기대 문자열은 전부 verdict 에서 계산한다.
 # ⚠M-3 의 `DEP12` 는 **처치 미집행(판정 무효)** 이므로 그 셀의 수는 **핀을 걸지 않는다** —
 #   논문에 인용되어서도 안 되는 수다(AMENDMENT_2 §8.3).
 ("M-1 LIN 승수",            M1,  "arms.LIN.ship_vs_spec.better",      lambda v: cnt(v,24)),
 ("M-1 LIN 크기",            M1,  "arms.LIN.ship_vs_spec.rel_median_pct", lambda v: pct(v)),
 ("M-1 HOS 승수",            M1,  "arms.HOS.ship_vs_spec.better",      lambda v: cnt(v,24)),
 ("M-1 HOS 크기",            M1,  "arms.HOS.ship_vs_spec.rel_median_pct", lambda v: pct(v)),
 ("M-1 LIN 음성대조",         M1,  "arms.LIN.spec_vs_zero.better",      lambda v: cnt(v,24)),
 ("M-1 HOS 음성대조",         M1,  "arms.HOS.spec_vs_zero.better",      lambda v: cnt(v,24)),
 ("M-1 LIN 기준선 기술",       M1,  "arms.LIN.spec_vs_zero.rel_median_pct", lambda v: pct(v)),
 ("M-1 HOS 기준선 기술",       M1,  "arms.HOS.spec_vs_zero.rel_median_pct", lambda v: pct(v)),
 ("M-1 LIN vs 제로 크기",     M1,  "arms.LIN.ship_vs_zero.rel_median_pct", lambda v: pct(v)),
 ("M-1 HOS vs 제로 크기",     M1,  "arms.HOS.ship_vs_zero.rel_median_pct", lambda v: pct(v)),
 ("M-1 짝대비 승수(탐색적)",    M1,  "paired_contrast.hos_worse",         lambda v: cnt(v,24)),
 ("M-1 짝대비 크기(탐색적)",    M1,  "paired_contrast.rel_median_pct",    lambda v: "+"+pct(v)),
 ("M-1 확증 문턱 p",          M1,  "thresholds.p_confirm",              lambda v: f"{v:.6f}"),
 ("M-1 게이트",              M1,  "gates_pass",                        lambda v: cnt(v,36)),
 ("M-3 DIR180 승수",         M3,  "cells.DIR180.ship_vs_zero.better",  lambda v: cnt(v,16)),
 ("M-3 DIR180 크기",         M3,  "cells.DIR180.ship_vs_zero.rel_median_pct", lambda v: pct(v)),
 ("M-3 DIR180 음성대조",      M3,  "cells.DIR180.spec_vs_zero.better",  lambda v: cnt(v,16)),
 ("M-3 SPR5 승수",           M3,  "cells.SPR5.ship_vs_spec.better",    lambda v: cnt(v,16)),
 ("M-3 SPR5 p",              M3,  "cells.SPR5.ship_vs_spec.p_one_sided", lambda v: f"{v:.2f}"),
 ("M-3 SPR5 음성대조",        M3,  "cells.SPR5.spec_vs_zero.better",    lambda v: cnt(v,16)),
 ("M-3 SPR5 vs 제로",        M3,  "cells.SPR5.ship_vs_zero.better",    lambda v: cnt(v,16)),
 ("M-3 확증 문턱 p",          M3,  "thresholds.p_confirm",              lambda v: f"{v:.6f}"),
 ("M-3 게이트",              M3,  "gates_pass",                        lambda v: cnt(v,30)),
 # ── 2026-08-30 추가 (M-7): 감독 축(E1) + 분포 이동 검사(E2). ⛔손복사 금지 —
 #   기대 문자열은 전부 verdict 에서 계산한다.
 #   ⚠E3(가파른·비선형)는 **바 없는 보고 전용** 이므로 핀을 걸지 않는다 —
 #     그 수는 논문이 확증으로 인용해서도 안 된다(PREREG §3 E3: HARD 팔에게 in-family).
 #   ⚠학습 시드 편차(9.14% 등)는 verdict 에 없다(레코드별 상대 크기는 판정 대상이 아니었다)
 #     ⟹ 그 수의 단일출처는 tmp/M7_SUPERVISION_2026-08-28/result_recheck.py 이고,
 #     그 스크립트가 동결 jsonl 에서 다시 만들어 **논문 본문과 직접 대조** 한다.
 ("M-7 손실 주효과 승수",     M7A, "primary.contrasts.T-1_loss.wins_negative",   lambda v: cnt(v,12)),
 ("M-7 손실 주효과 p",        M7A, "primary.contrasts.T-1_loss.p",               lambda v: f"{v:.3f}"),
 ("M-7 분포 주효과 승수",     M7A, "primary.contrasts.T-2_data.wins_negative",   lambda v: cnt(v,12)),
 ("M-7 분포 주효과 p",        M7A, "primary.contrasts.T-2_data.p",               lambda v: f"{v:.4f}"),
 ("M-7 교호 승수",           M7A, "primary.contrasts.T-3_interaction.wins_negative", lambda v: cnt(v,12)),
 ("M-7 음성대조 승수",        M7A, "primary.neg_control.wins",                   lambda v: cnt(v,12)),
 ("M-7 음성대조 p",          M7A, "primary.neg_control.p",                      lambda v: f"{v:.3f}"),
 ("M-7 중단 문턱",           M7A, "primary.neg_control.max_side",               lambda v: str(v)),
 ("M-7 MDE 최소 승수",       M7A, "primary.mde.min_wins",                       lambda v: cnt(v,12)),
 ("M-7 MDE p(11/12)",        M7A, "primary.mde.p_at_min",                       lambda v: f"{v:.6f}"),
 ("M-7 MDE p(10/12)",        M7A, "primary.mde.p_at_min_minus_1",               lambda v: f"{v:.6f}"),
 ("M-7 실측기술 BASE",        M7A, "primary.skill_vs_true_zero.BASE.beats_true_zero",   lambda v: cnt(v,12)),
 ("M-7 실측기술 BASE-B",      M7A, "primary.skill_vs_true_zero.BASE-B.beats_true_zero", lambda v: cnt(v,12)),
 ("M-7 실측기술 BANK",        M7A, "primary.skill_vs_true_zero.BANK.beats_true_zero",   lambda v: cnt(v,12)),
 ("M-7 E2 BASE 승수",        M7B, "s1.detail.per_cell.BASE.better",             lambda v: cnt(v,48)),
 ("M-7 E2 LOSS 승수",        M7B, "s1.detail.per_cell.LOSS.better",             lambda v: cnt(v,48)),
 ("M-7 E2 DATA 승수",        M7B, "s1.detail.per_cell.DATA.better",             lambda v: cnt(v,48)),
 ("M-7 E2 BOTH 승수",        M7B, "s1.detail.per_cell.BOTH.better",             lambda v: cnt(v,48)),
 ("M-7 E2 BASE-B 승수",      M7B, "s1.detail.per_cell.BASE-B.better",           lambda v: cnt(v,48)),
 ("M-7 E2 음성대조",         M7B, "s1.detail.neg_control.better",               lambda v: cnt(v,48)),
 ("M-7 R1 기준(SHIPPED)",    M7B, "s1.r1_hard_bar.shipped_mean",                lambda v: f"{v:.4f}"),
 ("M-7 R1 바",              M7B, "s1.r1_hard_bar.per_cell.BASE.bar",           lambda v: f"{v:.4f}"),
 ("M-7 E2 평균 BASE",        M7B, "s1.r1_hard_bar.per_cell.BASE.mean",          lambda v: f"{v:.4f}"),
 ("M-7 E2 평균 LOSS",        M7B, "s1.r1_hard_bar.per_cell.LOSS.mean",          lambda v: f"{v:.4f}"),
 ("M-7 E2 평균 DATA",        M7B, "s1.r1_hard_bar.per_cell.DATA.mean",          lambda v: f"{v:.4f}"),
 ("M-7 E2 평균 BOTH",        M7B, "s1.r1_hard_bar.per_cell.BOTH.mean",          lambda v: f"{v:.4f}"),
 ("M-7 E2 평균 BASE-B",      M7B, "s1.r1_hard_bar.per_cell.BASE-B.mean",        lambda v: f"{v:.4f}"),
 ("M-7 HARD k·Hs 중앙",      M7,  "manifest.hard_kHs_median",                   lambda v: f"{v:.4f}"),
 ("M-7 현행 k·Hs 중앙",      M7,  "manifest.base_kHs_median",                   lambda v: f"{v:.4f}"),
 ("M-7 미세조정 스텝",        M7,  "manifest.lam0_finetune_steps",               lambda v: f"{v:,}"),
 ("M-7 프로브 n",            M7,  "manifest.hard_probe.n_scored",               lambda v: str(v)),
 ("M-7 프로브 p10",          M7,  "manifest.hard_probe.p10",                    lambda v: f"{v:.3f}"),
 ("M-7 프로브 최소",          M7,  "manifest.hard_probe.min",                    lambda v: f"{v:.3f}"),
 ("M-7 게이트",              M7,  "gates_pass",                                 lambda v: cnt(v,16)),
]

bad = []
for label, rel, path, fmt in CHECKS:
    try:
        exp = fmt(jget(rel, path))
    except Exception as e:
        bad.append((label, f"verdict 경로 실패 {rel}:{path} {e}")); continue
    needle = exp.lstrip('+')
    where = [i for i, b in BODY if needle in b]
    if where:
        print(f"✅ {label:26s} {exp:>10s}  ← {rel.split('/')[-1]}:{path}   줄 {where[:6]}")
    else:
        print(f"⛔ {label:26s} {exp:>10s}  ← {rel.split('/')[-1]}:{path}   ★논문에서 못 찾음")
        bad.append((label, f"기대 '{exp}' 가 본문에 없다 ({rel}:{path})"))

print(f"\n대조 {len(CHECKS)}건 · 불일치 {len(bad)}건")
for l, m in bad:
    print(f"   ⛔ {l}: {m}")
sys.exit(1 if bad else 0)
