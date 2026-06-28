# Synthetic aperture (window length) — the continuous "win" is a SHORT-WINDOW artifact (2026-06-28)

`d54_window_aperture.py`. Long HOS window (5·Tp); reconstruct (F1 dense-continuous & RELAX
discrete) using progressively longer sub-windows of obs.

## Result — F1/DPS (continuous) flat; RELAX (discrete) improves and OVERTAKES
| window | F1 (continuous) | RELAX (discrete) |
|---|---|---|
| 1.0·Tp | 0.230 | 0.271 |
| 2.2·Tp | 0.224 | 0.270 |
| 3.5·Tp | 0.228 | 0.228 |
| 5.0·Tp | **0.236** | **0.221** |

(BF5 is dramatic: RELAX 0.251→0.170 as window 1→5·Tp; BF4 noisier.)

## Finding
- The synthetic aperture (D_eff = L + c_g·T_obs) is a **real information lever** — but it
  helps the **adaptive discrete method (RELAX)**, NOT the fixed continuous representations
  (F1) or the learned prior (DPS), which are flat in window length (~0.234, like d49's
  flat-in-obs).
- With enough aperture (≈5·Tp), **RELAX (0.221) BEATS the continuous methods (F1 0.236,
  DPS ~0.234)** — the discrete method resolves the continuous spectrum's components once the
  aperture is large enough.
- → **The "continuous representation / learning wins on realistic continuous spectra" result
  was a SHORT-WINDOW (2.2·Tp) artifact.** Give the classical adaptive estimator more time and
  it overtakes again.

## Complete, unified honest conclusion (all regimes)
- The problem is **information-limited** everywhere (CRB).
- **More information (synthetic aperture / time) is the universal lever**, and the **classical
  adaptive estimator (RELAX) exploits every lever best** (aperture, density).
- **Learning has NO durable accuracy advantage** anywhere: its apparent edges —
  continuous-spectrum (short window), observation-scarcity, active-sensing — are
  window/regime artifacts or niches that vanish with more information, OR are matched by a
  non-learned continuous method (F1).
- **Learning's only durable value = calibrated UQ** (a posterior the deterministic RELAX
  lacks), plus turnkey operation.

## Paper implication
The strongest, un-defeatable framing: a **CRB information-limit theory** with an exhaustive
demonstration that *more information (aperture), exploited by an adaptive classical estimator,
is what matters* — and that **no learning approach beats it on accuracy** (5+ methods, all
regimes), the field's discrete benchmark having masked the limit. Learning contributes
calibrated uncertainty, not accuracy.
