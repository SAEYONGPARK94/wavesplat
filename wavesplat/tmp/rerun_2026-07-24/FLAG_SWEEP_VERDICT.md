# Flag sweep verdict (2026-07-24)

Pre-registered rule (fixed in `flagjobs.py` before the numbers): an arm counts as an
improvement only if the paired per-episode median moves the same way on >= 4 of 6 episodes
AND the mean move exceeds the default arm's mean per-episode seed sd (measured 0.0306).

| arm | mean Δ | better on | verdict |
|---|---|---|---|
| deterministic | −0.0060 | 5/6 | not resolvable (|Δ| < 0.031); **but CV → 0** |
| min_dk_005 | +0.0041 | 3/6 | not resolvable; hurts BF3_5comp (0.170→0.221) |
| recompute_residual | +0.0153 | 4/6 **worse** | **does not help** — hurts BF3_5comp (0.170→0.242) |
| scale_inv_grow | −0.0010 | 2/6 | null (grow trigger rarely binding on this set) |
| scale_inv_l1 | +0.0071 | 2/6 | not resolvable, slightly worse |

## Decisions

**No flag is adopted.** All four code changes stay at their shipped defaults. They remain
available as documented module-level options, but this sweep does not justify turning any on.

Two findings worth keeping:

1. **`recompute_residual` is not the fix.** It was implemented as the "principled" repair for
   the `min_dk` static-power-map problem (recompute the beamform map on the residual each
   greedy step so the suppression radius stops being load-bearing). Measured, it degrades the
   mid-density regime *more* than simply lowering the radius (BF3_5comp 0.170 → 0.242). So the
   `min_dk` issue is a genuine regime-dependent hard trade: neither a smaller fixed radius nor
   residual-recomputation helps uniformly. The paper's `min_dk` caveat (regime-dependent
   trade, adaptive-radius repair untested) is therefore correctly hedged and should not be
   upgraded to claim a working fix.

2. **`deterministic` is a reproducibility/accuracy trade, not a free win.** It removes both
   random-phase initialization sites, collapsing run-to-run CV to ~0 (sd 0.000 on 4/6
   episodes), and is 5/6 in the improving direction — but the mean move is inside seed noise
   and split by regime (helps crowded BF4_15comp 0.291→0.271, hurts BF3_5comp 0.170→0.205).
   This matches the estimator-variance finding (multi-restart + median is the right reporting
   fix, not forcing determinism). Not adopted as default; documented for anyone who needs
   bit-reproducibility at a small, regime-dependent accuracy cost.

Consistent with [[project_scale_constants_2026-07-23]] and [[project_estimator_variance_2026-07-23]].
