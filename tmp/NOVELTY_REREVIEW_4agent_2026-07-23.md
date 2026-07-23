# WaveSplat Novelty Re-Review — 4 parallel de-anchored agents — 2026-07-23

**Prompt (user, Discord):** "Novelty를 강화할 다른 방법은 없는지 우리의 방법론을 처음부터 검토해줘. 여러 에이전트를 동시에. 지금까지의 작업은 잊고."

**Method:** 4 independent `general-purpose` agents, orthogonal lenses, each told to (a) read the actual code + `wavesplat.tex`, (b) approach as if the contribution had not been chosen, (c) WebSearch-verify every "novel/first/only" claim before stating it. None was fed the project's prior conclusions (de-anchoring by design). This memo reconciles their four reports against the project record.

---

## TL;DR

1. **Independent 4-way convergence.** From four different starting points the agents converged on the same axis: the durable contribution is **not "a better estimator"** (the "beats-ETKF-30%" headline is baseline-specific and sits in an active estimator race) but a **sensor-anchored information / identifiability characterization** of shipboard-LiDAR wave sensing, with the estimator repositioned as *evidence of reachability*. → This **re-validates the current thesis spine** (CRB-anchored operability envelope) from scratch, by agents that were not told it existed.

2. **The one genuinely-NEW lever** (not in the project's ~40-architecture tried-list): **dispersion-manifold gridless atomic-norm / BLASSO super-resolution**. Two agents (method, cross-domain) reached it independently. It is the single move that:
   - gives a genuinely-new *algorithm* carrying a *recovery theorem* — i.e. it supplies the **achievability half** the CRB story currently lacks (GUARD2 has been unmet);
   - matches the project's own empirical success pattern ("physics-normalized prior + closed-form/convex solve wins; raw→point-estimate mean-collapses");
   - absorbs three heuristic components (beamform-init, CLEAN/RELAX growth, adaptive-N) into one principled object;
   - lets you **rigorously partition the algorithm-wall from the information-wall**.

3. **The strongest synthesis (better than any single agent produced): a two-regime, two-sided characterization.** ANM recovery certificate (achievability) in the **well-separated / sparse** regime ⊕ misspecified-Gaussian CRB (converse) in the **crowded / broadband** regime; the crossover between them *is* the operability envelope. This upgrades the spine from "converse-only" (its current weakness) toward a genuinely two-sided result — which is what separates "theorem-grade" from "re-deriving Fucile 2018".

4. **Honest brake:** the whitespace is **narrow**, flanked by Fucile 2018 ("Predictability Region"), Qi 2018 (predictable zone), Ehlers PoF 2025 (training-free PINO). ANM is a strong **methods** contribution, **not** a paradigm shift, and it does **not** move the information wall — it *maps* it more sharply. Commit only after a cheap feasibility probe (below).

---

## What each agent concluded

- **Problem-reframing.** Strongest = *sensor-conditioned identifiability/CRB map fused with the honest-negative stance* (R1+R5): "what can a given forward-looking shipboard LiDAR estimate — per-parameter — and WaveSplat as the estimator that reaches the bound." Warns: narrow whitespace, must position explicitly vs **Fucile, Bulian & Lugni (Ocean Eng 2018, "Predictability Region"/LEPrE)** and **Qi/Wu/Liu/Yue (Wave Motion 2018, predictable zone)**; must show **achievability** (currently unproven) and a real-data anchor; sensor-niche is aging (**"Wave characteristics from LiDAR point cloud", Ocean Eng 2025**). Rejected as spent: grid-free-as-novelty, standalone synthetic aperture, UQ-headline, control-value (project's own C2 demo = NO-WIN), observation-point selection.

- **Method / algorithm.** Every component is a known technique in a new setting: dispersion-shell beamformer (Young/Nieto-Borge radar inversion, 1985–2004) → CLEAN/RELAX (Högbom 1974; Li–Stoica 1996) → damped-Gauss-Newton polish → greedy order-selection (weaker than RJMCMC/MDL/SBL) → freeze-frequency phase tracking (McAulay–Quatieri 1986) → heuristic WLS. **No new algorithm — but the problem invites one:** ① manifold-constrained gridless atomic-norm line-spectral estimation on the dispersion shell + recovery/CRB-attainment guarantee; ② the moving-platform CRB; ③ gridless joint current+dispersion (narrower).

- **Cross-domain transplant.** Same #1 = dispersion-manifold gridless/atomic-norm super-resolution (Candès–Fernández-Granda 2014; Tang–Bhaskar–Shah–Recht 2013; grid-free compressive beamforming Xenaki–Gerstoft JASA 2015), solved by **sliding-Frank-Wolfe / continuous BLASSO** (Boyd 2017; Denoyelle 2019) because the moving-platform, dispersion-warped sampling breaks the Toeplitz/Vandermonde SDP structure. #2 = pair it with two-sided resolution theory (achievability ⊕ CRB). **Rejects the learned-prior transplants** (diffusion, neural process, PINN/PINO/FNO) as both pre-empted *and* already tested-to-not-win inside the repo, and confirms the **"Gaussian Splatting" name is cosmetic — do not build on it** (a delocalized oscillatory field has no good localized-Gaussian basis). Key pre-emptor to differentiate against: **Ehlers et al., Physics of Fluids 2025 / arXiv 2508.03315** (training-free PINO, buoy+radar, *no* LiDAR, *no* moving platform, *no* off-grid guarantee).

- **Theory / evaluation.** No *new* theorem is available — DOA-CRB (Stoica–Nehorai 1989), constrained-CRB (Gorman–Hero 1990; Stoica–Ng 1998), moving-array CRB, direction-ambiguity resolution (SAR cross-spectra + dispersion) and predictable-zone geometry are all classical/pre-empted. Maximally defensible = a **converse-only, Gaussian, explicitly-misspecified** CRB *instantiation* for this sensor, reported as an efficiency/operability envelope. **Critical warning (the exact past failure mode):** a two-sided "information *wall*" is **not** rigorously attainable by CRB alone — multitone estimation is a threshold/non-convex problem so the CRB is unmet in the crowded regime, and the linear-estimator-vs-Gerstner-truth mismatch demands a **misspecified CRB (Richmond–Horowitz 2015)**. Also flags one genuine open gap: **calibrated UQ** (the repo's prior UQ was uncalibrated, z-std ≈ 2.4) — though partly pre-empted by Wu et al. (Applied Energy 2022).

---

## Reconciling the one real tension: achievability

Agent 3 said ANM *gives* min-separation achievability guarantees; Agent 4 said achievability is *not* attainable. **Both are right, scoped by regime** — and that scoping is the deliverable:

- **Well-separated / sparse regime** (few, angularly-separated components): the ANM dual-certificate gives a genuine **achievability** result — an off-grid recovery guarantee under a minimum-separation condition. This is exactly the *under-determined-but-sparse* regime the project record repeatedly identifies as the one place the algorithm still has headroom.
- **Crowded / broadband regime** (N≳15, closely-spaced): the separation condition fails *and* the CRB is unattainable → only the **converse** (misspecified-Gaussian CRB) is rigorous, and the project's own record already says no estimator wins there (information wall).
- **The crossover between these two regimes is the operability envelope.** So the two agents' "contradiction" is actually the structure of the result: use the achievability tool where it is valid, the converse tool where it is valid, and report the boundary.

**Bonus reconciliation with the record:** the project previously found the "aperture premium" largely evaporates for the *heuristic* estimator ("most of the aperture wall was the algorithm wall"). A provably near-optimal ANM estimator is precisely the instrument to settle that cleanly — how much of the residual gap is algorithmic (recoverable by a better estimator) vs informational (a true wall). That sharpens the whole thesis rather than competing with it.

---

## Pre-emption / positioning map (must cite in Related Work)

| Prior art | What it pre-empts | Your differentiator |
|---|---|---|
| Ehlers et al., *Phys. Fluids* 2025 (arXiv 2508.03315) | training-free real-time phase-resolved reconstruction (buoy+radar, PINO) | LiDAR + **moving platform** + **off-grid recovery guarantee** |
| Fucile, Bulian & Lugni, *Ocean Eng* 2018 (Predictability Region/LEPrE) | operability/trust-map from a *known* spectrum | **inverse** identifiability of spectral params *from the sensor geometry*; noise-aware CRB |
| Qi/Wu/Liu/Yue, *Wave Motion* 2018; Naaijen | deterministic predictable *zone* (where, from group speeds) | per-parameter, **noise-aware** information bound (not deterministic support) |
| Desmars, *JFM* 2023; Ehlers, *Ocean Eng* 2023/24; LETKF | the estimator race → don't headline "beats X" | reposition estimator as *achievability evidence* |
| *Ocean Eng* 2025, "wave characteristics from LiDAR point cloud" | "shipboard LiDAR is new" | novelty is the **identifiability characterization**, never the sensor |
| on-grid CS directional spectrum (arXiv 2605.25074, 2026) | CS recovery of the *spectrum* | **gridless, phase-resolved field** (strictly dominates on-grid) |

---

## Do-not-revisit (rejected across agents + record)

Learned priors as headline (diffusion / neural process / PINN / PINO / FNO — pre-empted **and** internally tested-to-not-win in the informative regime); the **Gaussian-Splatting analogy** (cosmetic; bad basis); control-decision value (C2 demo = NO-WIN, Δ≈+0.01); UQ-calibrated forecast as *primary* novelty (Fucile 2018, Wu 2022); grid-free / dispersion-constraint as *novelty* (radar inversion 1985–2004); standalone synthetic-aperture (internally falsified); five-metric panel as a *methodology* contribution (= Nash–Sutcliffe/Murphy skill).

---

## Recommended next step — a cheap FEASIBILITY GATE before committing

Do **not** rewrite the thesis on this yet. Run one bounded probe (~days, not weeks):

1. Implement 2-D BLASSO / sliding-Frank-Wolfe over the dispersion manifold; run on a handful of **well-separated** synthetic episodes.
   - (a) Does it recover off-grid components the heuristic beamform-init misses?
   - (b) Where separation holds, does its error approach the (Gaussian) CRB?
2. If **yes** → ANM becomes the new methodological backbone; it supplies the achievability half of a two-sided characterization. Then: map the ANM-valid regime against the operability envelope, write the **misspecified** CRB converse, and add the explicit "delta vs Fucile 2018 / Qi 2018 / Ehlers 2025" paragraph.
3. If **no** (solver won't converge on the warped manifold, or gains don't beat the heuristic) → fall back to the **converse-only CRB + operability-envelope** spine, unchanged. No sunk cost in the thesis.

**One-line verdict:** the review does not overturn the current (modest, solid) direction — it *strengthens* it by supplying the missing achievability half via an untried, on-pattern, theorem-bearing method (gridless ANM on the dispersion manifold), gated behind a cheap feasibility check and positioned honestly against Fucile/Qi/Ehlers.
