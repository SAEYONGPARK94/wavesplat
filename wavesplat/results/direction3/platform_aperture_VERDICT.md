> ⛔**2026-08-13 감사 — 이 판정은 철회됨.** `CRB ×4.4 sharper` 는 **단일 시드**(`d72:50`, seed=1;
> `:88` 은 시드를 전달하지 않는다)에서 나온 값이고, 복제하면 **2.8배**(8시드)·**3.22배**(30시드)
> 이며 판정규칙은 **11/40 에서만 발화**한다 ⟹ **부호는 생존, 크기는 사망**. 제목의 `CONFIRMED`
> 는 유효하지 않다. 더 나아가 **실측 관측기하 위의 field-CRB 에서는 정지가 이긴다**
> (정지 0.00986 vs cross 0.01615 = **0/12**, along 0.02631 = 0/12) ⟹ 이 문서가 전제하는
> "이동이 정보를 만든다" 는 **부호까지 뒤집혔다**.
> ⛔**또한 이 팔(`d72`)은 LiDAR 를 0회 호출한다** — 광선·차폐·raydrop·FOV·range 전무,
> `wavesim` 임포트 0회, 관측은 상자 균일난수 + 해석식 `Σ A·cos` + N(0, 0.02).
> 추정기도 출하 `SequentialUpdater` 가 아니다 ⟹ **출하 시스템의 증거가 아니다.**
> 근거: `tmp/AUDIT_recheck_2026-08-13/A3_citation_bans.md`, `reference_citation_bans.md`
>
> 본문 수치는 이력 보존을 위해 남긴다 — **인용하지 말 것.**

# VERDICT — Platform-motion synthetic aperture resolves the directional bottleneck (2026-06-29)
# ⛔ RETRACTED 2026-08-13 (제목의 "CONFIRMED" 무효)

The make-or-break test for the doctoral POSITIVE. Our CRB (d1) showed the directional (cross-range)
identifiability is the binding bottleneck — wave-motion synthetic aperture (c_g·T_obs) sharpens |k|/ω
but is flat in direction. Hypothesis: PLATFORM motion (the ship moving) supplies the missing
cross-range aperture. Tested by feeding moving observation positions (x=x0+vx·t, y=y0+vy·t) into the
existing FIM. `scripts/d72_platform_aperture.py`.

## Result — ⛔CONFIRMED **철회됨 (2026-08-13 감사)**
Realistic multidirectional sea (8 components, ±25° spread), footprint 50–60 m, T_obs 20 s, wave
c_g = 5.2 m/s. Median directional CRB vs platform speed:

| platform speed | 0 | 3 | 5 | 7 | 10 m/s |
|---|---|---|---|---|---|
| **cross-track (⊥ wave)** | 9.6° | 7.5° | **3.9°** | 2.7° | **2.2°** |
| along-track (∥ wave) | 9.6° | 23.3° | 23.5° | 27.1° | 19.0° |

- ⛔**RETRACTED (2026-08-13 감사)** — "Cross-track platform motion collapses the directional CRB:
  9.6° → 2.2° (**×4.4 sharper**) at 10 m/s, ×2.5 at 5 m/s."
  단일 시드값이다. 복제 = **2.8배**(8시드) / **3.22배**(30시드), 판정규칙 11/40 발화.
  허용되는 서술 = **"부호 생존, 크기 사망"**. 근거: `reference_citation_bans.md`
- **Along-track motion HURTS** (9.6° → ~23°): at v ≈ c_g the ship "surfs" the wave → the field looks
  frozen → Doppler/temporal diversity is lost → identifiability degrades. (A bonus physical insight:
  the WORST trajectory is surfing the dominant wave.)
- (Single isolated component has NO directional bottleneck — CRB ~0.3° at any speed — so that sanity
  panel is uninformative; the bottleneck is about SEPARATING nearby components, where the effect is
  strong.)

## Why this is the doctoral positive
It turns the entire negative arc into the SETUP for a positive thesis:
> *Phase-resolved ship-LiDAR wave estimation from a STATIONARY sensor is information-limited, and the
> binding gap is directional (CRB). The ship's own motion is the cure: CROSS-TRACK platform-motion
> synthetic aperture supplies the cross-range aperture needed for directional identifiability
> (⛔`CRB ×4.4 sharper` — **RETRACTED 2026-08-13, 단일 시드**), while along-track surfing is worst. We turn the CRB into a TRAJECTORY-DESIGN
> tool: optimal vessel headings/maneuvers that minimize the directional CRB and unlock reconstruction
> the stationary limit forbids.*
The CRB theory (C1) becomes the ENGINE (design tool), not the conclusion. Positive, novel-framed,
doctoral-scope (theory → optimal trajectory design → achievability → validation).

## Novelty (lit-check, honest)
Synthetic aperture from platform motion is well-known in SAR / synthetic-aperture sonar. BUT: for
OCEAN WAVES, SAR measures directional/power SPECTRA (statistical), NOT phase-resolved fields; SAS
images the seafloor, not the surface wave field. No work found framing platform-motion synthetic
aperture as the **directional-identifiability cure for phase-resolved ship-borne wave reconstruction**,
with CRB-guided trajectory optimization. → Novelty is **application + analysis + trajectory design**,
not the synthetic-aperture concept. Defensible doctoral novelty (verify with a deeper lit review).

## Next steps (to make it a thesis, not just a CRB curve)
1. **Achievability:** show a DEPLOYABLE estimator (RELAX / posterior) actually reconstructs better with
   cross-track motion on HOS seas — the CRB says it's possible; demonstrate it end-to-end.
2. **Optimal trajectory design:** minimize the directional CRB over headings/maneuvers (incl. realistic
   ship-speed/turn-rate constraints); compare straight cross-track vs zig-zag vs circular.
3. **HOS + real-data validation** (D1) eventually.

## Caveats
- This is the CRB (a LOWER BOUND / possibility result) on a synthetic multidirectional sea — not yet an
  achieved reconstruction nor HOS. Achievability (step 1) is the immediate must-do before claiming the
  thesis. The mechanism (cross-range aperture) is general and analytically grounded (∂ψ/∂ky = y0+vy·t).
