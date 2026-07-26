# EVIDENCE MAP — machine-generated 2026-07-26

> **Do not hand-edit.** Regenerate with
>
> ```
> python3 tmp/obsaxis_2026-07-26/gen_evidence_map.py > docs/EVIDENCE_MAP_GENERATED_2026-07-26.md
> ```
>
> Every number is read from a result file and every verdict is computed from the frozen
> criterion in `gen_evidence_map.py`. This exists because the hand-typed map carried two
> material errors (persona debate 2026-07-26, E1 "repairs 3/3" and E2 the mean-only real
> line); both are now emitted by code and cannot drift.

## 1. Tracker double phase-advance was a bug; fixing it improves every episode

- **measured:** 0.1990 -> 0.1145 (-42.5%), improved 100/100, Wilcoxon p=3.9e-18
- **frozen criterion:** improve >= 90/100 AND p < 0.001
- **verdict:** **MEASURED - PASS**
- **source:** `tmp/rerun_2026-07-25/batchB_ws.jsonl + batchC_ws_phasefix.jsonl`

## 2. Learned set-proposal init beats the analytic init on crowded synthetic seas

- **measured:** 0.2525 -> 0.1809 (medDelta -0.0616), 36/40, p=5.0e-11
- **frozen criterion:** pre-registered P1 AND P2 (tmp/wp_integration_2026-07-25/PREREG.md)
- **verdict:** **MEASURED - PASS**
- **source:** `tmp/wp_integration_2026-07-25/results.json`
- **caveat (auto):** component retention 58.6% FAILED its own bar (tracker erosion)

## 3. A GT-free router beats both pure arms on the 100-episode benchmark

- **measured:** pure analytic 0.1145 / pure learned 0.1011 / auto 0.0979 (n_selected >= 8)
- **frozen criterion:** auto <= min(pure arms)
- **verdict:** **MEASURED - PASS**
- **source:** `tmp/unified_2026-07-26/switch_results.json`
- **caveat (auto):** IN-SAMPLE: the threshold and the guard ratio were set on these same 100 episodes

## 4. The beamform guard removes the sparse misroutes without touching crowded routing

- **measured:** auto 0.0979 -> guard 0.0875; sparse 0.0249 (bar 0.0254); crowded flips 0
- **frozen criterion:** sparse <= pure_v2 + 0.005 AND 0 crowded flips AND aggregate improves
- **verdict:** **MEASURED - PASS**
- **source:** `tmp/unified_guard_2026-07-26/task1_results.json`

## 5. The guard closes most of the gap between the router and an oracle switch

- **measured:** deployed 0.0875 vs lane-consistent oracle 0.0806: 60.2% of the auto->oracle gap closed (old min-of-medians oracle 0.0818 would have said 64.6%)
- **frozen criterion:** deployed and oracle aggregated identically (per-lane, then median, then mean)
- **verdict:** **MEASURED - QUOTABLE (E3 fixed)**
- **source:** `tmp/obsaxis_2026-07-26/oracle_consistent.json`
- **caveat (auto):** the old crowded-slice paradox (deployed below its own oracle) is gone: 6 -> 0 episodes; the ratio is an in-sample point estimate with no CI

## 6. On real stereo seas the learned arm LOSES accuracy (the programme predicted this)

- **measured:** n=27 normal windows, median Delta=-0.0571 (hand-learned), learned better on 6/27, two-sided Wilcoxon p=0.005
- **frozen criterion:** pre-registered cell 1 (LOSS on normal-27) - tmp/wp_real_2026-07-25/PREREG.md
- **verdict:** **MEASURED - CONFIRMED LOSS**
- **source:** `tmp/wp_real_2026-07-25/wp_real.json`
- **caveat (auto):** analysis unit is the WINDOW; windows within a record are not independent (n_records=5)

## 7. The learned arm repairs the analytic arm's collapse windows

- **measured:** learned arm below the line: 2/3; end-to-end (detector o fallback): 1/3; large error reduction (Delta >= 0.5): 2/3
- **frozen criterion:** FROZEN: "fixed" = learned nRMSE < 1.0 (PREREG sec.3(a))
- **verdict:** **MEASURED - 2/3, NOT 3/3**
- **source:** `tmp/unified_2026-07-26/fallback_results.json`
- **caveat (auto):** repaired fields land at ~0.96-1.07, i.e. at a no-skill predictor: this is bounded degradation, not restored skill. Report as an operating characteristic, never as nRMSE gain. AA14_w0 2.31->1.07 [NOT FIXED]; AA15a_w3 1.12->0.79; LJ18g_w3 2.93->0.96

## 8. The deployed detector-plus-fallback chain improves the 30-window real aggregate

- **measured:** MEAN 0.8815 -> 0.7747; MEDIAN 0.8091 -> 0.8091
- **frozen criterion:** a central-tendency win must move the MEDIAN (do-not-cite rule: no mean-based real win)
- **verdict:** **MEASURED - NOT A GENERAL WIN**
- **source:** `tmp/unified_2026-07-26/fallback_results.json`
- **caveat (auto):** AUTO-CAVEAT: the median is unchanged (0.8091 -> 0.8091); the entire mean delta comes from the 2 window(s) the detector touched. Do not quote the mean without the median.

## 9. The collapse detector does not false-alarm on normal real windows

- **measured:** 0/27 false fires at tau=1.5; separation margin = min(amp|fired) 2.402 - max(amp|not fired) 1.196 = +1.206
- **frozen criterion:** R1 (P3, strictest): 0 false fires = PASS, 1 = PARTIAL, >=2 = FAIL; R2: margin > 0
- **verdict:** **MEASURED - PASS**
- **source:** `tmp/unified_2026-07-26/fallback_results.json`
- **caveat (auto):** tau was set IN-SAMPLE on these 27 windows; specificity is unproven until the 8 unused Bergamasco records are scored (W7). Report the margin, not a p-value.

## 10. The regime boundary is detectable from the observations, not just from the simulator knob

- **measured:** Spearman |rho| of the per-episode delta: bf_peak_ratio 0.257 (p=9.9e-03) vs n_components 0.514 (p=4.4e-08), n=100
- **frozen criterion:** PRE-STATED (PRESTATE.md): |rho_bf| >= |rho_ncomp| - 0.05 => boundary is not a relabeling of the answer key
- **verdict:** **MEASURED - FAIL: n_components dominates; the phase diagram is drawn on a coordinate the runtime cannot read - P3 is right and Ch.5 says so**
- **source:** `tmp/obsaxis_2026-07-26/obsaxis_stats.json`
- **caveat (auto):** the 30 real windows cluster at bf_peak_ratio 0.51-0.95 (IQR) where only 6/100 synthetic episodes live; 0/30 fall below the 0.05 guard, so the synthetic guard never fires on real data

---

_10 claims emitted. Criteria are frozen in code; editing a number here without editing its source file is a no-op that the next regeneration deletes._
