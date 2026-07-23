# "Weak short-λ components are unrecoverable by all methods" — **OVERTURNED** — 2026-07-23

Script `tmp/shortlambda_revalidate.py`, log `tmp/shortlambda_revalidate.log`, raw
`tmp/shortlambda_revalidate_results.json`. Decision rule pre-registered in the script header
**before** any numbers.

## The claim under test

From the ANM feasibility gate (`tmp/NOVELTY_ROUND3_execution_2026-07-23.md`, run this
morning): *"Weak short-λ components are unrecoverable by ALL methods at ALL apertures … ANM
does not supply the missing achievability half — the crowded/weak regime is the information
wall, exactly as the record says."* This sentence is the **empirical leg** of the project's
information-wall argument.

## Why it was suspect

It was measured on `data/`, where `wavesim/mesh_adequacy.py` fails **0 of 102** episodes and
raycast mesh spacing is dx 6.15 / dy 6.90 m — content shorter than ~14 m is not
representable at all. If the component was never rendered into the truth field, "no method
recovers it" is circular.

## Setup

`data_finemesh_2026-07-23/` (200×167, dx = 1.206 m) is a **paired** replacement: same seeds,
and the wave components bit-match the shipped files (verified **100/100**). Same estimator
(the ANM gate's own greedy + joint-refine from `tmp/blasso_feasibility.py`), same subsample,
same seed — only the rendering differs. Determinism asserted, not assumed: re-running one
episode twice gave identical output to the last digit.

## Result — the pre-registered "rendering artifact" branch fired

| episode | components recovered | weakest comp | shortest-λ comp | field nRMSE |
|---|---|---|---|---|
| BF3 2comp | 2 → **2** / 2 | HIT → HIT | HIT → HIT | 0.0488 → **0.0131** (−73 %) |
| BF3 5comp s0 | 3 → **5** / 5 | **MISS → HIT** | **MISS → HIT** | 0.0752 → **0.0193** (−74 %) |
| BF3 5comp s1 | 4 → **5** / 5 | **MISS → HIT** | **MISS → HIT** | 0.0764 → **0.0172** (−77 %) |
| BF4 15comp | 11 → **15** / 15 | HIT → HIT | **MISS → HIT** | 0.1145 → **0.1018** (−11 %) |
| **total** | **20/27 → 27/27** | | 1/4 → **4/4** | |

**Every episode reaches full component recovery on the adequate mesh.** The specific component
the gate called unrecoverable (BF3 5comp: A = 0.028, λ = 6.2 m) is recovered on both seeds.
It was not missed — on a 6.90 m mesh, a 6.2 m wave **was not there to find**.

## What is overturned, and what is not

- ❌ **RETRACTED: "weak short-λ components are unrecoverable by all methods at all
  apertures."** MEASURED, replicated on 2 independent seeds plus 2 further episodes. The
  empirical leg of the information-wall argument must be re-derived on adequate data
  wherever it rested on this.
- ✅ **The ANM gate's actual conclusion survives**: SFW ≈ GREEDY+JOINT was a *relative*
  comparison with both arms on identical data. The gate's NEGATIVE verdict on atomic-norm as
  a novelty lever is unaffected.
- ✅ **The CRB theory is untouched** — it is estimator-independent and was never derived from
  this measurement.
- ⚠ **The crowded regime is a different story, and the wall may well be real there.** BF4
  15comp reaches full recovery (11 → 15/15) yet field error falls only **−11 %**
  (0.1145 → 0.1018), against −73…−77 % on the sparse episodes. Recovering every component
  does **not** proportionally reduce error when the scene is crowded. That is the signature of
  a genuine conditioning/information limit, and this experiment does **not** overturn it.

## A second observation worth its own follow-up

The same rendering change that cuts this estimator's error by up to 77 % moved the WaveSplat
tracker's aggregate **not at all** in the 12-episode pilot (paired t = −0.44,
`tmp/MESH_PILOT_RESULT_2026-07-23.md`). Two estimators, same data change, opposite
sensitivity. Either the tracker is not limited by what the mesh was destroying, or it has a
bottleneck that masks the improvement. Unexplained; worth a targeted comparison.

## Limits

4 episodes, synthetic linear seas, single deterministic estimator (the numpy greedy/joint
pipeline — **not** the torch tracker whose bimodal CV-53 % variance is documented in
`tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md`, hence no multi-seed needed here). No HOS.
Component matching uses the gate's own tolerance (|Δk| < 0.06 and |Δdir| < 15°).
