# F2 — What detection law does the real Falcon use?

2026-07-10. Question: does the real Seyond/Innovusion Falcon (FK24) declare a
return by **thresholding received power**, or by a **smooth multiplicative**
return law like the simulator's `P = η·Fresnel(θ)·cosθ / (d/125)²`? The answer
decides whether the LiDAR return **mask** carries ~2.5% of the amplitude
information (the in-simulator measurement, density-invariant under a
multiplicative law) or ~17% at the deployed sparse rate (red-team, threshold
law) — i.e. whether a selection-aware learning chapter is a speed chapter or an
accuracy chapter.

Sources read: `Falcon_lidar/Seyond Falcon LiDAR User Manual (including FK24)
240528.pdf` (195 pp, text extracted to `tmp/falcon_manual_text.txt`); web
literature and Seyond/Innovusion public docs and patents (URLs below).

---

## Verdict

**Threshold-on-power. High confidence.** The real Falcon declares a return when
a per-pulse received-signal statistic crosses a detection threshold; it does not
implement anything resembling the simulator's smooth multiplicative return
probability. This rests on **direct manual artifacts** plus **universal ToF
detection practice**. The manual is **silent on the photodetector and on any
numeric detection-probability / false-alarm / minimum-detectable-signal
figure**, so the *exact statistic* (matched filter + fixed vs. CFAR threshold)
and the *exact speckle law* (exponential vs. partially de-speckled) are
**inference from detection physics**, labelled as such below, at medium-high
confidence.

### Strongest single piece of evidence
The specification table (manual p.13, Table 2) lists **"Detection range (10%
reflectivity) 250 m"** alongside **"Maximum detection range 500 m"** and
**"Detection range (blind area) 1.5 m"**. A detection range *quoted at a stated
target reflectivity* is definitionally the range at which the return from that
target crosses the receiver's detection threshold — this is threshold language,
not multiplicative-law language. A smooth multiplicative return-probability
sensor has no such thing as a hard "detection range at 10%."

### Corroborating direct manual evidence
- **Per-return `confidence_level`** in the point-cloud packet (p.105): a graded
  detection confidence per point, `0 lowest → 3 highest`. A four-level
  per-return confidence is the signature of a threshold-margin / SNR decision,
  not of a continuous multiplicative draw.
- **Multi-return by strength** (p.105 `multi_return_mode`: `single` /
  `strongest + 2nd strongest` / `strongest + furthest`; p.37/p.84 return-mode
  UI; `is_2nd_return` flag p.106): the receiver detects candidate echoes and
  **ranks/selects them by strength** — peak-over-threshold detection.
- **ToF description** (p.10 §1.2): distance is measured "between the emission of
  the light pulse and **the detection of the reflected light**" by the "optical
  sensor" — a discrete detection *event*, not an analog occupancy.
- **Intensity vs. reflectivity** (p.37, p.59–60): `intensity` is "the intensity
  of the echo read directly by LiDAR … varies with object distance, object
  reflectivity, beam angle"; `reflectivity` is that intensity "rectified with
  the object distance, beam angle." A return therefore carries a measured echo
  amplitude that a detector compares against range-dependent expectations.
- **Internal intensity fault** `INNO_LIDAR_IN_FAULT_REFINTENSITY=55` "abnormal
  light intensity" (p.156): the device monitors received intensity as a
  first-class quantity.

### What the manual does NOT say (verdict rests partly on inference here)
- No detector type is named. The words APD, SPAD, SiPM, avalanche, photodiode,
  false-alarm, detection-probability, SNR, and minimum-detectable-signal **do
  not occur** anywhere in 195 pages (grep-verified). The receiver is called only
  "optical sensor" (p.10) and "four channel sensors" (p.135,
  `get_detector_temps`). Falcon is stated to be a **1550 nm semi-solid-state**
  ToF LiDAR (p.11, p.13), Class-1 (IEC-60825), 10 FPS, 1500 lines/s, ±5 cm
  Lambertian accuracy, 2 cm (1σ) precision (p.13). 1550 nm at automotive/marine
  range effectively rules out Geiger-mode SPAD arrays (those dominate at
  905 nm); 1550 nm long-range ToF is near-universally **linear-mode InGaAs APD
  + matched-filter/threshold timing** — but this is inference, not a manual
  statement.

## Detection-physics inference (labelled; medium-high confidence)
1. **ToF LiDAR declares returns by threshold.** Time-of-flight receivers
   matched-filter the digitized pulse and declare a return where the filtered
   statistic exceeds a threshold; the threshold is commonly held at a constant
   false-alarm rate (CFAR), and APD reverse-bias/gain is adjusted to hold that
   rate. False-alarm rate is literally "the probability that system noise
   exceeds the detection threshold." This is textbook and appears across the
   LiDAR-detection literature and patents (see Sources).
2. **Innovusion/Seyond's own patents describe threshold + adaptive-gain
   detection** (time-varying / multi-gain detectors, CFAR in pulsed LiDAR) —
   consistent with, though not proof of, the FK24 receiver.
3. **Rough-water 1550 nm coherent returns are speckled.** Coherent illumination
   of an optically rough surface produces fully developed speckle whose
   single-look intensity is **exponentially distributed** (Goodman). Hence a
   per-hit received power `A(θ,d)·X`, `X ~ Exp(1)`, and `return ⟺ A·X > T`, so
   `p = exp(−T/A)` — **exactly the red-team model**. Caveat: multi-pulse
   averaging / multiple looks partially de-speckle (Gamma(m,·), m looks),
   softening the exponent; the effective exponent must be calibrated, not
   assumed.

---

## What this implies

1. **The in-simulator conclusion stands as an in-simulator statement.** Under
   the simulator's smooth multiplicative law, mask/amplitude ≈ 2.5% and is
   density-invariant to O(p²). That is a true property **of that law's family**
   — it is not a property of threshold detection, and it was never evidence
   about the real sensor.

2. **The multiplicative law understates mask information at sparse operation.**
   The invariance is a *feature of multiplicativity*: scaling the return
   probability by a constant factor cancels in the mask/amplitude ratio. A
   threshold law breaks that cancellation. From the red-team identity, with
   `p = exp(−T/A)`:

   ```
   d ln p / ds = (T/A) · d ln A/ds = (−ln p) · d ln A/ds
   ```

   The mask bit's log-sensitivity to the surface state `s` scales with
   `(−ln p)`, so **sparser sensing makes the mask MORE informative**, the
   opposite of the invariant multiplicative case. Reproduced here
   (`tmp/redteam_threshold.py`, CPU), same radiometric shape `A ∝
   Fresnel(θ)cosθ/(d/125)²` as the simulator:

   | return rate | I_mask   | I_amp    | mask/amp |
   |-------------|----------|----------|----------|
   | 0.001       | 2.79e-04 | 1.63e-03 | **17.15%** |
   | 0.01        | 1.41e-03 | 1.63e-02 | 8.67%    |
   | 0.1         | 5.34e-03 | 1.63e-01 | 3.28%    |
   | 0.3         | 1.41e-02 | 4.89e-01 | 2.88%    |

   Grazing-incidence water returns operate at the **sparse** end; at the sparse
   deployed rate the mask reopens to ~17% (an order of magnitude above the
   2.5% multiplicative figure), and only collapses back toward ~2.5% in the
   dense limit.

3. **The deployment claim must switch families.** "Mask ≈ negligible" is safe
   *inside the multiplicative simulator* and unsafe *as a claim about the real
   Falcon*. If the real detector is threshold-on-power (verdict above), the mask
   channel is a genuine, exploitable accuracy channel at sparse operation, and
   **selection-aware amortized inference becomes an accuracy chapter, not a
   speed chapter.** This is a family switch in the forward model, and the thesis
   should say so explicitly rather than carry the 2.5% number into deployment
   language.

---

## The exact simulator change that implements the threshold law

New raydrop model (create `wavesim/raydrop_threshold.py`; do **not** edit the
existing `raydrop_uniform.py` / `raydrop_slope.py`), mirroring the current
radiometric shape but replacing the smooth return probability with a thresholded
speckle draw. Per hit at incidence angle `θ` and range `d`:

```
A(θ, d) = Fresnel(θ) · cos(θ) / (d/125)²          # same shape as today
X ~ Exp(1)                                         # fully developed speckle (1 look)
# optional partial de-speckling: X ~ Gamma(m, 1/m) for m looks (multi-pulse avg)
return  ⟺  A · X > T                               # threshold detection
```

- `Fresnel(θ)` reuse `wavesim.raydrop_uniform.fresnel_reflectance`
  (`N_WATER_1550NM`, p.27 of that file).
- **Calibrate `T`** (equivalently a laser-power scale) so the **mean return
  rate equals the deployed `n_pts` per frame** — the same knob the red-team
  bisects. The rate knob is `T`/power, NOT a multiplicative factor on `p`; that
  is the whole point.
- Reported amplitude in "intensity" mode should be `A·X` (the echo the LiDAR
  reads directly, per manual p.37), consistent with the sensor semantics.

## Pre-registered re-run (falsifiable)

Rerun the information-decomposition logic of `scripts/d3_s2c_coherent.py`
(mask-vs-amplitude information about the surface field) under the **threshold**
raydrop, at the **deployed return rate**, holding everything else fixed.

- **Pre-registered expectation (red-team right):** mask/amplitude ≈ **~17%** at
  the deployed sparse rate — an order-of-magnitude reopening vs. the 2.5%
  multiplicative figure — with the ratio rising monotonically as the rate falls.
- **Pre-registered null (red-team wrong / de-speckling dominates):** if strong
  multi-look averaging pushes the effective law back toward multiplicative, the
  ratio stays ~2.5% and the mask stays closed. This is the falsifier; report it
  either way.
- Sensitivity: sweep `m` (looks) to bound how much realistic pulse-averaging
  de-speckles the exponent, and confirm the qualitative rate-dependence
  `mask/amp ↑ as rate ↓` is preserved across `m`.

---

## Confidence summary
- Threshold-on-power detection: **high** (direct manual artifacts + universal
  ToF practice).
- Linear-mode InGaAs-APD + matched-filter/CFAR timing specifically: **inference,
  medium-high** (manual silent on detector; 1550 nm long-range ToF norm).
- Exponential (single-look) speckle exactly: **inference, medium** (standard for
  rough-surface coherent 1550 nm, but Falcon's pulse processing may partially
  de-speckle; calibrate `m` and `T`, don't assume).
- Net: the mask channel **reopens under the real detector's family**; the 17%
  figure is the pre-registered target, to be confirmed by the re-run above
  before any deployment-facing claim is made.

## Sources
Manual: `Falcon_lidar/Seyond Falcon LiDAR User Manual (including FK24) 240528.pdf`
pp.10–14 (product/spec), p.37/59–60/84 (intensity vs reflectivity, return modes),
p.105–106 (`confidence_level`, `multi_return_mode`, `is_2nd_return`, `refl`),
p.135 (four channel sensors), p.156 (`REFINTENSITY` fault).

- Falcon product page (250 m @ 10%, 500 m, 1550 nm): https://www.seyond.com/products/falcon-k1/
- Falcon K1 User Manual (public): https://www.seyond.com/wp-content/uploads/2024/12/Seyond-Falcon-K1-LiDAR_User-Manual_V2.1.2_EN_PublicRH_20240314.pdf
- Falcon Prime system doc (public): https://mobilespector.com/wp-content/uploads/2024/05/Seyond-Falcon-Prime-LiDAR-System_V1.0.2_English_public_20221214.pdf
- Threshold/CFAR detection in ToF LiDAR (FAR = P(noise>threshold), adaptive APD gain): https://www.mdpi.com/2076-3417/13/6/3772 ; https://www.researchgate.net/publication/309262075_TOF-LIDAR_signal_processing_using_the_CFAR_detector
- CFAR / matched-filter detection patents (pulsed LiDAR): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11041944 ; https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11442145 ; https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11867836
- Innovusion detector-gain / multi-gain patents: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10976417 ; https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10627495
- Speckle statistics of rough-surface coherent 1550 nm returns (exponential single-look intensity): https://sciencedirect.com/science/article/abs/pii/S0030402621020106
- Red-team reproduction: `tmp/redteam_threshold.py` (this repo).
