# Verification — learned vs hand headline numbers

Date: 2026-07-12. All values independently re-derived from the stored JSONs by `scripts/d3_appendix_figs.py::verify`.

## Claim 1 — 8-record nowcast (unobserved region)
Path: `sa_multirecord_mmse{,_net}.json`, per record mean over windows of `conditions.A.mean.common_unobs.corr_bl`.

| record | n_win | hand | learned(net) | delta |
|---|---|---|---|---|
| BS11a | 6 | 0.4740 | 0.5789 | +0.1049 |
| AA14 | 6 | 0.5050 | 0.5824 | +0.0774 |
| AA15a | 6 | 0.4777 | 0.5178 | +0.0401 |
| BS11b | 6 | 0.2659 | 0.3448 | +0.0788 |
| BS13a | 6 | 0.4603 | 0.6259 | +0.1657 |
| BS13b | 6 | 0.3485 | 0.5995 | +0.2511 |
| AA15b | 6 | 0.3056 | 0.6131 | +0.3075 |
| YS17 | 5 | 0.3333 | 0.4454 | +0.1121 |
| **mean(8 rec)** | 47 | **0.3963** | **0.5385** | **+0.1422** |

- re-derived: hand **0.396** -> learned **0.538**, +35.9% rel, sign **8/8**
- documented: hand 0.396 -> learned 0.538, +36%, 8/8
- **match: YES** (hand 0.396 vs 0.396; learned 0.538 vs 0.538; +36% vs +36%)
- cluster boot (contrast A): mean +0.1428, n_win 47, sign 8/8, CI [+0.0874, +0.2062], survives True

## Claim 2 — 60-window whole-field corr
Path: `field_wide_eval.json`, fields `field_hand`, `field_learned`, `per_lead_*`.

- whole-field mean: hand **0.5497** -> learned **0.5832**, delta **+0.0334**
- learned > hand: **43/60**; record sign **5/5**
- documented: hand +0.550, learned +0.583, Δ+0.033, 43/60, 5/5
- **match: YES**

| lead (s) | hand | learned | delta |
|---|---|---|---|
| 0 | 0.6449 | 0.6718 | +0.0269 |
| 2 | 0.5953 | 0.6246 | +0.0293 |
| 5 | 0.5283 | 0.5643 | +0.0360 |
| 8 | 0.4271 | 0.4637 | +0.0365 |

- learned wins at **4/4** leads (all four)
- cluster boot (contrast E): mean +0.0334, CI [+0.0180, +0.0483], t_win 3.22, sign 5/5
- documented field CI [+0.018, +0.048]; re-read [+0.0180, +0.0483] -> match YES

## Claim 3 — 60-window encounter-track corr
Path: `gate_promotion_eval.json`, fields `hand`, `bw_s0_head` (learned = net + s2-head).

- track mean: hand **0.5320** -> learned **0.6195**
- delta all 60: **+0.0875**; learned>hand 38/60; record sign **5/5**
- failure windows (hand<0.5, n=25): delta **+0.2266** (hand 0.145 -> learned 0.372)
- documented: Δ+0.088 all / +0.227 failure windows
- **match: YES**
- cluster boot (B all60): mean +0.0875 CI [+0.0523, +0.1167]; (B bad25): mean +0.2266 CI [+0.1417, +0.2970]

## Discrepancy summary
- None. All three headline claims and the two cited cluster-bootstrap CIs reproduce to display precision.

## Rounding notes
- Documented values are 3-dp rounded; re-derived means (hand nowcast 0.3963, learned 0.5385; field hand 0.5497, learned 0.5832; track all 0.0875, bad 0.2266) round to the documented figures. No divergent rounding found.
