# Novelty review + persona debate — resuming the research (2026-06-29)

Triggered by user: "novelty 검토와 페르소나 토론으로 연구를 다시 진행해보자."
Grounded in a literature reality-check (DSWP predictable zone, phase-resolved UQ, ML
reconstruction) run 2026-06-29, on top of the d30–d61 exhaustive map.

---

## 0. Literature reality-check (what is ALREADY done in the field)

| topic | prior art found | implication for our novelty |
|---|---|---|
| Predictable zone | **geometric DPZ from group velocity** is canonical (Naaijen/Belmont; Lee–Nam–Liu–Yang 2024, marine-radar predictability) | a predictable zone per se is NOT new; the *geometric c_g cone* is the standard |
| UQ for phase-resolved forecast | 2018 probabilistic prediction-error; **2022 variational-Bayesian-ML UQ** (Appl. Energy); **2025 uncertainty-guided prediction horizon under data sparsity** (JMSE, τ-trimming) | "calibrated UQ" alone is **crowded, NOT novel** |
| ML reconstruction from sparse remote sensing | Ehlers et al. 2023 (ML phase-resolved reconstruction of nonlinear surfaces from sparse data) | amortized neural reconstruction is **NOT novel as a method** |
| Data assimilation forecast | EnKF/ETKF phase-resolved forecast (arXiv 2020; 2025 joint wave+ship-motion DA) | our ETKF baseline is standard; DA forecasting is established |
| LiDAR wave reconstruction | flash-LIDAR (2014), LiDAR point-cloud (2024) | LiDAR sensing itself is not the contribution |

**No Cramér–Rao / Fisher-information treatment of the predictable zone appeared.** That is the
one genuinely open wedge.

---

## 1. Honest novelty verdict (ranked)

**✅ STRONG — the only defensible cores**
1. **Information-theoretic (CRB/FIM) predictable zone.** The field's predictable zone is built
   *geometrically* from group velocity. We claim the binding constraint under sparse, moving
   ship-LiDAR is **Fisher-information identifiability**, and the true predictable boundary is set
   by FIM rank/conditioning, **not** the c_g cone. If this holds and is validated, it *refines /
   corrects* a canonical result — that is a real, citable theoretical contribution.
   (Dir-I: `d1_crb.py`, `d1_predictable_zone.py`; T_wall; D_eff = L + c_g·T_obs.)
2. **Discrete-benchmark-bias critique (meta/methodology).** The field evaluates phase-resolved
   estimators on discrete-component synthetic seas, which structurally favor discrete estimators
   (RELAX) and *mask* the information limit. On realistic continuous/HOS seas the ranking flips.
   Provocative, defensible, and re-usable by others.

**⚠️ WEAK as standalone novelty (crowded) — must be REPOSITIONED, not headlined**
3. **Fast amortized neural posterior.** As a *method* it is incremental (ML sparse reconstruction
   + Gaussian-NLL UQ both exist). It only becomes novel if its **UQ is calibrated to the CRB
   predictable zone** (theory-grounded, not post-hoc temperature scaling), i.e. it *operationalizes*
   core #1. By itself "fast + UQ" loses to 2022/2025 prior art.
4. **Calibrated UQ / nowcast+forecast package.** Crowded (2018/2022/2025). Engineering, not novelty.

**❌ NOT a contribution by itself**
5. The exhaustive negative results (learning can't beat classical). Rigorous and *necessary as
   support*, but a negative-results sprawl is not a headline at a good venue. It is the evidence
   layer for core #1, nothing more.

---

## 2. Persona debate

**P1 — Reviewer 2 (skeptic, Ocean Eng / IEEE JOE).**
"Amortized posterior? ML phase-resolved reconstruction from sparse data is Ehlers 2023. UQ for
wave forecast is 2018, 2022, *and a 2025 data-sparsity paper*. You are repackaging. And a
28-experiment 'we tried everything and learning lost' is not a paper. **Give me one theorem or
one corrected canonical result, or reject.**" → Concedes: *if* the FIM predictable zone genuinely
differs from the geometric DPZ and forecast skill follows FIM, that is publishable.

**P2 — Ocean-eng domain expert (DSWP).**
"The group-velocity predictable zone is canonical. You are claiming the real boundary is FIM
identifiability. That is contentious — **good**, if you validate it and frame it right: not 'DPZ
is wrong' but 'geometric DPZ is necessary-not-sufficient; under sparse moving sensing,
identifiability binds first.' But: (a) show it on HOS/real data, (b) reviewers here demand
experimental validation — pure synthetic won't survive; the SNU towing-tank plan must become
real. The CRB-calibrated UQ tied to predictability is genuinely useful for control."

**P3 — ML theorist.**
"Stop selling the architecture; sell the principle: **amortized inference that provably approaches
the CRB, with uncertainty calibrated to Fisher information.** That's the ML story. Your current UQ
is post-hoc temperature scaling (s≈1) — reviewers want σ *derived* from the FIM and then validated
by coverage, not fit. Predict posterior covariance from CRB, show empirical coverage matches with
no tuning → theory-grounded UQ. That elevates 'vanilla NN' to 'first CRB-calibrated amortized
posterior for this problem.'"

**P4 — Advisor.**
"I wanted learning central and winning. The honest map says it doesn't win on accuracy. So pivot
learning from 'accuracy winner' to **'the only object that delivers a CRB-calibrated posterior in
real time.'** Learning-central AND honest. Kill the sprawl: the 28 experiments collapse to one
'levers & negatives' figure. One-sentence thesis:
> *The predictable zone of phase-resolved sea estimation is set by Fisher-information
> identifiability, not geometry; we derive it (CRB), show every classical and learned estimator is
> capped by it, and deliver a fast amortized posterior whose uncertainty is calibrated to this zone
> for real-time forecasting.*"

---

## 3. Convergence — the sharpened thesis

Drop "fast + UQ + learning beats X" as the headline (crowded, and false on accuracy). Headline the
**theory**, use learning to *operationalize* it:

> **Information-theoretic predictability of phase-resolved seas, and a CRB-calibrated amortized
> posterior for real-time forecasting.**

- **Core (novel):** FIM/CRB predictable zone refines the canonical geometric (group-velocity) DPZ.
- **Operationalization (learning-central, now non-trivial):** an amortized posterior whose UQ is
  *calibrated to the CRB mask* — uncertainty that knows the physics of predictability.
- **Methodology (novel):** discrete-benchmark bias; realistic continuous/HOS evaluation.
- **Evidence layer:** the negatives + levers become one figure showing everything is CRB-capped.

---

## 4. Next experiments — to ESTABLISH novelty (not re-confirm negatives)

1. **Make-or-break: FIM predictable zone vs geometric DPZ, head-to-head (HOS).**
   Compute both the group-velocity geometric zone and the FIM-identifiability zone for the same
   episode; measure where forecast skill actually collapses. *If skill follows FIM and FIM ≠
   geometry → the paper's core. If they coincide → novelty collapses to "re-derived DPZ via CRB"
   (weaker, must know now).* This is the single most important experiment to run first.
2. **CRB-grounded UQ (vs post-hoc).** Predict posterior σ from the FIM/CRB and test empirical
   coverage with no temperature fit; compare to current s≈1 scaling. Calibrated-from-theory σ =
   the ML novelty that survives Reviewer 2.
3. **Explicit positioning vs 2018/2022/2025 UQ-forecast prior art** + a concrete real-data /
   towing-tank validation plan (P2 requirement). Build the comparison table now so the paper's
   delta is unambiguous.

**Secondary / speculative learning-win candidate:** constrained active sensing. Free-aim sensing
was uniform-optimal (d42/d43, not winnable), BUT realistic LiDAR has fixed FOV + scan-rate + ship
motion; neural acquisition already showed a *transferable* −16…−34% (d40) before being set aside.
Under hardware constraints this is the one place learning plausibly wins on a real objective —
reconsider only if the posterior story reads too incremental.

---

## 5. Open decision for the user
- Run experiment #1 (FIM-zone vs geometric-DPZ) first? It is make-or-break and decides whether the
  theory headline is safe.
- Venue: Ocean Eng / IEEE JOE (theory + validation, learning secondary) vs an ML venue
  (amortized-CRB-posterior central)? This changes how hard to push "learning."
- Revive constrained active sensing as a second learning-central thread, or keep it in reserve?
