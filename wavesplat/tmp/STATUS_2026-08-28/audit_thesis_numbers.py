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
 ("M-8b 끝점 승수곱",         M8B, "amendment_2.factorization.mult_endpoint",      lambda v: f"{v:.5f}"),
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

# ═══════════════════════════════════════════════════════════════════════════
# ★★★2026-08-30 — **앵커 동결**. 그전까지 판정은 `needle in b` (본문 어딘가에
#   그 문자열이 부분문자열로 있는가) 하나였다. 돌연변이 시험이 그것을 반쯤 무력화했다:
#
#     verdict 값을 마지막 자리 ±1 로 흔들어도 그대로 통과 = **71/148 = 48.0%**
#     (예: `M-1 LIN 승수 22/24` 를 21/24 로 흔들면 **같은 문단의 HOS 21/24** 에 붙어 통과)
#
#   ⟹ ★***이 검사기가 지키겠다고 선언한 시나리오("verdict 가 바뀌면 먼저 깨진다")가
#      절반의 핀에서 성립하지 않았다.*** 사유는 4번 검사기와 **같은 부류** —
#      ***대조가 위치를 모른다.*** 9,384줄 어디든 같은 수가 있으면 통과한다.
#
# ★수리 = **핀마다 "그 수가 사는 문장의 골격 + 슬롯" 을 동결**한다(`ANCHORS`).
#   골격 = 그 줄의 모든 수를 `#` 로 가린 것. 슬롯 = 그 줄에서 몇 번째 수인가.
#   골격이 일치하면 슬롯 수도 반드시 일치하므로(가린 개수가 곧 슬롯 개수) 슬롯 비교는 안전하다.
#   ⟹ verdict 가 바뀌면 **동결된 그 문장의 그 자리**에 옛 값이 남아 있으므로 **잡힌다**.
#   실측(2026-08-30): 돌연변이 죽임 **146/148 = 98.6%**, 오늘 원고 기준 **오탐 0건**.
#   (생존 2건은 맨숫자 `29`·`9` 로, 문장 골격까지 같은 자리에 이웃값이 실재한다.)
#
# ⚠**래칫 아님 · 종료코드는 오늘보다 엄해지지 않는다.** 세 갈래로 나눈다:
#   ⛔FAIL  — 동결 골격이 본문에 **있는데** 그 슬롯 값이 기대와 다르다 (= 진짜 불일치)
#             · 또는 값이 본문 어디에도 없다 (종전 규칙 그대로)
#   ⚠WARN  — 동결 골격이 **사라졌다**(문장을 개작했다) ⟹ 종전 규칙으로 판정하고 경고만.
#             ★사유: 원고가 여러 세션에서 동시에 개작되는 동안 빨간불을 켜 두면
#             **경고가 항상 켜져 있어 신호가 죽는다**(저장소가 반복해 대가를 치른 부류).
#   ⚠미동결 — 앵커가 없다. ★저장소 규율상 ***"동결한 적이 없다" 가 "동결 뒤 움직였다" 보다
#             나쁘다***(기준선이 없으면 영원히 OK 를 찍는다) ⟹ 반드시 세어서 보고한다.
#   재동결 = `python3 …/audit_thesis_numbers.py --refreeze` (문장을 의도적으로 고친 뒤).
# ═══════════════════════════════════════════════════════════════════════════
# ---8<--- CHECKS_TABLE_END : 여기까지가 '기대값 계산'이고 아래가 '판정'이다.
#           ⛔이 줄을 지우지 마라 — tmp/_audit/check_thesis_numbers.py 가 이 표만
#           떼어 쓰려고 여기서 자른다(2026-08-30, 그 marker 가 깨져 조용히 죽은 적이 있다).
ANCHOR_FILE = ROOT / "tmp/STATUS_2026-08-28/thesis_number_anchors.json"
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?')

def _norm(t):    return re.sub(r'[+\-−\s,]', '', t)
def _skel(s):    return NUM.sub('#', s)
def _slots(s):   return [_norm(m.group(0)) for m in NUM.finditer(s)]

def _anchors_for(needle):
    """본문에서 needle 이 **온전한 수 토큰**으로 앉은 자리 전부를 동결한다.
    앵커 = (문장 골격, 핀 슬롯, **그 줄의 나머지 수들**).
    ⚠**나머지 수까지 넣는 이유**: 골격만으로는 표 행처럼 문면이 같은 딴 문장과 충돌한다
      (실측: 골격만 쓰면 9개 핀이 엉뚱한 줄과 만났다 → 나머지 수를 넣으면 **0건**).
      충돌이 사라지므로 아래 판정을 ***"동결된 앵커 전부가 그 값을 갖고 있어야 한다"***
      로 **엄격화**할 수 있고, 그래야 **원고 쪽 손편집**(verdict 는 그대로인데 논문의 수만
      바뀐 경우)까지 잡힌다. 느슨한 규칙(하나라도 맞으면 통과)은 그것을 놓쳤다."""
    out = []
    for i, b in BODY:
        sl = _slots(b)
        for j, t in enumerate(sl):
            if t == needle:
                out.append({"skeleton": _skel(b), "slot": j, "line_at_freeze": i,
                            "others": [None if k == j else v for k, v in enumerate(sl)]})
    return out


def _matched(anch):
    """동결된 그 문장이 **지금** 어디에 있고, 핀 자리에 무엇이 들어 있는가."""
    res = []
    for i, b in BODY:
        sk, sl = _skel(b), _slots(b)
        for a in anch:
            others = a.get("others")
            if a["skeleton"] != sk:
                continue
            if others is not None:
                if len(sl) != len(others):
                    continue
                if not all(o is None or o == sl[k] for k, o in enumerate(others)):
                    continue
            elif a["slot"] >= len(sl):
                continue
            res.append((i, sl[a["slot"]]))
    return res

if "--refreeze" in sys.argv:
    frozen = {}
    for label, rel, path, fmt in CHECKS:
        try:
            a = _anchors_for(_norm(fmt(jget(rel, path))))
        except Exception:
            continue
        if a:
            frozen[f"{label}|{rel}|{path}"] = a
    ANCHOR_FILE.write_text(json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"★재동결 완료: {len(frozen)}/{len(CHECKS)} 핀 → {ANCHOR_FILE}")
    sys.exit(0)

try:
    ANCHORS = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
except Exception:
    ANCHORS = {}
    print(f"⚠앵커 파일이 없다({ANCHOR_FILE.name}) — 전 핀이 **미동결**로 돈다. "
          f"`--refreeze` 로 만들어라.")

bad, moved, unfrozen, split = [], [], [], []
for label, rel, path, fmt in CHECKS:
    try:
        exp = fmt(jget(rel, path))
    except Exception as e:
        bad.append((label, f"verdict 경로 실패 {rel}:{path} {e}")); continue
    needle = _norm(exp)
    tok_where = [i for i, b in BODY if needle in _slots(b)]
    sub_where = [i for i, b in BODY if exp.lstrip('+') in b]
    where = tok_where or sub_where
    src = f"{rel.split('/')[-1]}:{path}"
    anch = ANCHORS.get(f"{label}|{rel}|{path}")

    if anch:
        seen = _matched(anch)      # 동결 문장이 실재하는 지점에서 읽은 핀 자리 값
        if not seen:
            moved.append(label)
            tag = "⚠앵커이동" if where else "⛔"
            print(f"{tag} {label:26s} {exp:>10s}  ← {src}   "
                  f"{'동결 문장이 개작됐다 (종전 규칙으로 판정)  줄 ' + str(where[:6]) if where else '★논문에서 못 찾음'}")
            if not where:
                bad.append((label, f"기대 '{exp}' 가 본문에 없다 ({src})"))
            continue
        hit = [i for i, v in seen if v == needle]
        off = [(i, v) for i, v in seen if v != needle]
        # ★엄격 규칙: **동결 앵커 전부**가 그 값을 갖고 있어야 한다.
        #   (느슨한 "하나라도 맞으면 통과" 는 원고 쪽 손편집을 놓쳤다 — 같은 값이 여러 문장에
        #    살면 한 곳을 틀리게 고쳐도 다른 곳이 통과시켜 준다. 실측 오탐 0건이라 엄격화했다.)
        if off:
            print(f"⛔ {label:26s} {exp:>10s}  ← {src}   "
                  f"★**동결 앵커의 값이 다르다** {off[:4]}")
            bad.append((label, f"앵커 불일치: 기대 '{exp}' 인데 동결 문장에는 {off[:4]} ({src})"))
        else:
            print(f"✅ {label:26s} {exp:>10s}  ← {src}   줄 {hit[:6]}  (앵커 {len(seen)}곳 전부 일치)")
    else:
        unfrozen.append(label)
        if where:
            print(f"⚠미동결 {label:24s} {exp:>10s}  ← {src}   줄 {where[:6]}")
        else:
            print(f"⛔ {label:26s} {exp:>10s}  ← {src}   ★논문에서 못 찾음")
            bad.append((label, f"기대 '{exp}' 가 본문에 없다 ({src})"))

print(f"\n대조 {len(CHECKS)}건 · ⛔불일치 {len(bad)}건 · ⚠앵커이동 {len(moved)}건 "
      f"· ⚠미동결 {len(unfrozen)}건 · ⚠문장 내 값 갈림 {len(split)}건")
for l, m in bad:
    print(f"   ⛔ {l}: {m}")
if moved:
    print(f"   ⚠앵커이동(문장 개작 ⟹ 확인 후 --refreeze): {moved}")
if unfrozen:
    print(f"   ⚠미동결(★기준선이 없으면 영원히 통과한다): {unfrozen}")
sys.exit(1 if bad else 0)
