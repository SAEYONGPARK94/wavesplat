# R1 — P1 (harsh ML reviewer) — opening position

PERSONA P1 — OPENING POSITION (Round 1)

---

**THE STEELMAN, STATED HONESTLY BEFORE I ATTACK IT**

The weak version of this program is "a router that picks between two initializers." I will not argue against that, because it is not the strongest reading and the candidate does not appear to believe it either.

The strong version is this: *the object of study is not an estimator, it is a boundary.* A grid-free spectral estimator has two candidate priors — an analytic beamform-peak proposal and a learned conditional-diffusion set-proposal. The claim is (i) there exists a sharp crossover between them, (ii) the crossover is pre-registered rather than discovered post hoc, (iii) the mechanism is measured *on both sides* — the analytic arm fails crowded because greedy peak extraction lands in a bad basin, the learned arm fails sparse because the proposal network does not emit atoms near the true swell at all (generation-side OOD, not a post-processing deficit), (iv) the boundary is decidable from observations alone, which is demonstrated by building the switch and showing it beats both pure arms, and (v) four independent attempts to erase the boundary by retraining are sealed as honest negatives with localized causes. Under this reading the router is not the contribution — the router is the *instrument that makes the boundary claim falsifiable*. That is a defensible scientific posture, and it is rarer than a leaderboard win.

I take that seriously. Now the objections.

---

**MEASURED (I stipulate these; they are pre-registered, seeded, median-of-5, and re-derivable)**

- Tracker double phase-advance fix: 0.1990 → 0.1145 mean nRMSE, 100/100 episodes, p=3.9e-18.
- Learned set-proposal, crowded 40ep, init level: −45%, 39/40, p=1.8e-12. Integrated through the tracker: 0.2525 → 0.1809, 36/40, p=5e-11. Ablation localizes the mechanism to learned kx/ky placement.
- Clean crossover: sparse 40ep pure-v2 0.0204 vs pure-learned 0.0534. Learned-only sparse parity failed twice under pre-registration; cause measured as proposal-generation OOD on BF2_1comp.
- GT-free router 0.0979, +guard 0.0875, vs pure arms 0.1145 / 0.1011, oracle 0.0818. Threshold-insensitive over {6,8,10}.
- Real stereo, 27 non-degenerate windows: learned **loses** accuracy, 6/27, p=0.005. Learned repairs 3/3 analytic collapse windows and self-collapses 0/30. Deployed fallback moves the 30-window mean 0.881 → 0.775 with 0/27 false alarms.
- Three retraining axes null/FAIL: e2e fine-tune (medΔ −0.0007, p=0.30, leak localized to an observation-distribution gap), fan-geometry, learned-only parity.

**ARGUED (interpretation, not measurement)**

"Regime boundary between physics and learning." "Modular = product." "The operator does not choose the method." "The wall is representational, not informational." "Learning's real-data contribution is collapse insurance." Every one of these is a sentence about the world inferred from measurements taken inside one estimator family, on one generator, with one network.

---

**OBJECTIONS**

**1. The regime axis does not exist in the ocean.** The entire boundary is drawn on `n_components` — a knob in a synthetic linear generator, with GT that is pure-linear by the candidate's own constraint. "Crowded" means 15–30 cosine components. That is not a sea state; it is a sampling parameter. The program's own earlier measurement confirms the problem: real records are *all* broadband (entropy 0.82–0.94, within-record σ 0.01–0.03), so the axis on which the thesis's central figure is plotted cannot be varied in nature. A phase diagram whose x-axis is unmeasurable outside the simulator is a tuning curve with a physics vocabulary.

**2. The router is measured to route *wrong* on the only real data.** The switch fires learned when the proposal returns `n_selected ≥ 6`. The real-data diagnosis reports proposal `n_eff ≈ 24` on essentially every real window — i.e. the router classifies all 27 normal windows as crowded and would route to learned, which loses at p=0.005 on exactly those windows. Correspondingly, the real deployment does **not** run `init_method='auto'`; it runs hand-everywhere plus an amplitude-ratio fallback. So the headline "one estimator, one config" is true on the synthetic benchmark and false across the thesis's own two domains, where two different configurations are silently used. This is the single most damaging fact in the packet and it is derived entirely from the candidate's own files.

**3. The largest rung on the ladder is a self-inflicted bug.** 0.199 → 0.1145 is a 42.5% improvement obtained by repairing a double phase advance in the candidate's own tracker. The candidate already accepted the general form of this argument on 2026-07-24 — "the tracker is our own invention, so beating it is fixing our artifact, not a field contribution." Applied consistently, that removes the first and biggest rung from the program journey figure, and raises a separate question: the published OE benchmark numbers were measured on a broken tracker, and the fix is *still not the shipped default*.

**4. The deployed constants are not visibly held out.** Threshold insensitivity over {6,8,10} is a real defense for the router. It is not a defense for `auto_guard_ratio = 0.05`, which was introduced *after* the router's misroutes were observed and which recovers exactly the gap the router opened; nor for the real-data detector `amp_ratio > 1.5`, which the record explicitly states was designed in-sample on the same 30 windows on which "0/27 false alarms" is reported. In-sample specificity is a fit, not a specificity.

**5. The one surviving real claim has no control.** "Collapse insurance" rests on n=3. The diagnosed cause is a greedy basin failure. The obvious non-learned escapes — multi-restart greedy, second-peak init, relaxed `min_dk` — are not run. Until they are, "learning repairs collapses" is indistinguishable from "any different initialization escapes a bad basin," which is not a learning result.

**6. The framing is a domain instance of known theory.** "Learning's value concentrates where data does not constrain the field" is the null-space result the candidate's own file already located (Schwab/Antholzer/Haltmeier IP 2019; Deep Decomposition Learning ECCV 2020), and Ehlers 2023 already performs shadowed-vs-visible error decomposition. The one gate this program's own reviewers called existential — a neural-operator baseline on the same fair bench — remains unrun. Beating your own analytic prior is not a comparison an ML venue accepts.

**LIT-CHECK-NEEDED** (I ran none; exact queries I would run):
- `learned initialization non-convex sparse recovery neural proposal greedy matching pursuit`
- `per-instance algorithm selection classical vs learned solver inverse problems router`
- `conditional diffusion set prediction off-grid line spectral estimation`
- `learned failure detection safe fallback classical solver deployment inverse problem`
- `Ehlers 2023 wave reconstruction shadowed visible error decomposition FNO`

---

**VERDICTS**

**Q1 (novelty): MAJOR REVISION, trending REJECT as currently framed.** As "unified estimator with GT-free auto-switch": reject — that is systems glue, the router is two scalar heuristics, and per-instance algorithm selection is a mature field. As "pre-registered, two-sided-mechanism regime crossover with four sealed falsifications and an honest real-data negative": borderline-acceptable at a workshop / clearly publishable in ocean engineering, *conditional* on the neural-operator baseline and on the boundary surviving an arm swap. The genuinely unoccupied element, if anything, is the **negative side**: proposal-generation OOD localized as the reason a learned prior cannot be post-processed into parity. Mechanistic accounts of *why learned priors fail* are scarcer than accounts of why they win.

**Q2 (PhD): CONDITIONAL PASS — as an ocean-engineering / measurement-science doctorate with a falsification-methodology spine. FAIL as an ML thesis.** The volume of pre-registered self-refutation, the symmetric audits that overturned the candidate's own positive results, and one real dataset carried through to an honest loss constitute doctoral-level scientific conduct. They do not constitute a machine-learning contribution.

---

**TOP 3 CRUXES** (attack these; my verdict flips on them)

**CRUX 1 — Is the boundary a property of the problem, or of this one method pair?** Swap either arm: the analytic side for RELAX/CLEAN or the already-implemented sliding-Frank-Wolfe BLASSO (`tmp/blasso_feasibility.py`), the learned side for the existing DETR set-predictor. If the crossover reproduces at a comparable complexity with *either* arm swapped, I upgrade Q1 to accept-as-diagnostic and Q2 to clear pass. If the crossover is specific to cond-diffusion-vs-greedy-beamform, the regime map is a tuning curve and the thesis has no general object.

**CRUX 2 — Are the three deployed constants held out?** Were `n_selected ≥ 6`, `auto_guard_ratio = 0.05`, and `amp_ratio > 1.5` selected on the same 100 episodes / 30 windows on which 0.0875 and 0/27 are reported? Freeze all three, run fresh seeds and fresh real windows. If 0.0875 (or anything ≤ 0.1011) and zero false alarms replicate out-of-sample, objection 4 dies entirely. If not, 0.0875 is not a number and the deployment card is unpublishable.

**CRUX 3 — Does the router work on real data at all?** Run `init_method='auto'`, unmodified, end-to-end on the 30 real windows against hand-everywhere. Own diagnosis (`n_eff ≈ 24`) predicts it routes learned nearly everywhere and therefore loses. If it wins, "one estimator, one config" survives contact with reality and objection 2 dies. If it loses — or if it is never run — then the thesis's headline system is synthetic-only, the real chapter is a fallback heuristic with n=3 and no control, and the honest title is "a study of when a learned proposal helps a synthetic benchmark," which is a solid paper and a thin dissertation.
