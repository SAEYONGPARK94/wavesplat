# GP K_disp Full Evaluation (2026-06-09)
## 30 Episodes: 6 categories × 5 seeds (seed 0–4)

---

## Results Summary

### Final Averages (30 episodes)

| Method | nRMSE | VR | Pearson | Skill | Notes |
|--------|-------|-----|---------|-------|-------|
| WaveSplat MAP | 0.259 | 0.588 | 0.293 | -0.095 | Phase 0 baseline (6ep seed0) |
| WavePINO S4 | 0.222 | 0.292 | 0.498 | 0.264 | Neural baseline (6ep seed0) |
| GP K_disp [oracle] | **0.126** | 0.601 | **0.802** | **0.665** | Theoretical upper bound (30ep) |
| GP K_disp [ws_multistart] | 0.228 | **0.583** | 0.414 | 0.039 | Best practical (30ep) |

### Per-Category Breakdown: ws_multistart (30 episodes)

| Category | nRMSE | VR | Pearson | Skill | n |
|----------|-------|----|---------|-------|---|
| BF3_5comp  | 0.158 | 0.382 | 0.795 | 0.599 | 5 |
| BF4_5comp  | 0.139 | 0.712 | 0.831 | 0.665 | 5 |
| BF4_15comp | 0.250 | 0.614 | 0.383 | -0.058 | 5 |
| BF4_30comp | 0.277 | 0.539 | 0.097 | -0.355 | 5 |
| BF5_15comp | 0.275 | 0.644 | 0.244 | -0.222 | 5 |
| BF5_30comp | 0.268 | 0.610 | 0.135 | -0.394 | 5 |
| **AVERAGE** | **0.228** | **0.583** | **0.414** | **0.039** | 30 |

### Per-Category Breakdown: oracle (30 episodes)

| Category | nRMSE | VR | Pearson | Skill | n |
|----------|-------|----|---------|-------|---|
| BF3_5comp  | 0.082 | 0.708 | 0.952 | 0.890 | 5 |
| BF4_5comp  | 0.052 | 0.845 | 0.978 | 0.951 | 5 |
| BF4_15comp | 0.122 | 0.706 | 0.863 | 0.740 | 5 |
| BF4_30comp | 0.182 | 0.376 | 0.637 | 0.406 | 5 |
| BF5_15comp | 0.133 | 0.624 | 0.842 | 0.699 | 5 |
| BF5_30comp | 0.185 | 0.344 | 0.539 | 0.305 | 5 |
| **AVERAGE** | **0.126** | **0.601** | **0.802** | **0.665** | 30 |

---

## Key Findings (30-Episode Analysis)

### 1. Pilot VR=1.006 was a seed0 artifact
- Pilot (6ep, seed0 only): nRMSE=0.242, **VR=1.006**, Pearson=0.536
- Full eval (30ep, seeds 0–4): nRMSE=0.228, **VR=0.583**, Pearson=0.414
- Seed0 complex sea happened to have over-amplitude (VR>1) that averaged to 1.006
- Other seeds have under-amplitude (VR<1), pulling 30ep avg to 0.583
- **Corrected claim: VR=0.583 (2× WavePINO's 0.292) — still significant**

### 2. nRMSE improved slightly (30ep better than 6ep pilot)
- Pilot nRMSE=0.242 → 30ep nRMSE=0.228 (6% improvement)
- Explanation: seed0 includes some difficult outlier episodes; 30ep average is more stable

### 3. Simple sea vs complex sea divergence
- Simple (5 comp): nRMSE≈0.14–0.16, Pearson≈0.81–0.83 — excellent
- Complex (15–30 comp): nRMSE≈0.25–0.28, Pearson≈0.10–0.38 — poor
- Root cause: WaveSplat direction estimation degrades for 15–30 components

### 4. Oracle-practical gap by complexity
| Category | Oracle nRMSE | ws_multi nRMSE | Gap | Oracle Pearson | ws_multi Pearson |
|----------|-------------|----------------|-----|----------------|-----------------|
| BF3_5c  | 0.082 | 0.158 | +0.076 | 0.952 | 0.795 |
| BF4_5c  | 0.052 | 0.139 | +0.087 | 0.978 | 0.831 |
| BF4_15c | 0.122 | 0.250 | +0.128 | 0.863 | 0.383 |
| BF4_30c | 0.182 | 0.277 | +0.095 | 0.637 | 0.097 |
| BF5_15c | 0.133 | 0.275 | +0.142 | 0.842 | 0.244 |
| BF5_30c | 0.185 | 0.268 | +0.083 | 0.539 | 0.135 |

The Pearson gap is the critical limitation: complex sea direction estimation fails to capture wave structure.

---

## Comparison vs WavePINO

Note: WavePINO evaluated on 6 episodes (seed0 only). GP evaluated on 30 episodes.

| Metric | GP ws_multi (30ep) | WavePINO S4 (6ep) | Winner |
|--------|-------------------|-------------------|--------|
| nRMSE | 0.228 | **0.222** | WavePINO (+3%) |
| VR | **0.583** | 0.292 | **GP (2.0×)** |
| Pearson | 0.414 | **0.498** | WavePINO |
| Skill | 0.039 | **0.264** | WavePINO |
| Amplitude | better | severe hedging | **GP** |
| UQ | full posterior | none | **GP** |

For a fair comparison, GP seed0-only (6ep) results:
- nRMSE≈0.256, VR≈1.032, Pearson≈0.526 (seed0 favorable, not representative)

---

## Revised Thesis Framing

### What stands (confirmed over 30 episodes):
1. **Oracle GP is exceptional**: nRMSE=0.126, Pearson=0.802 — proves K_disp framework
2. **VR advantage**: ws_multistart VR=0.583 vs WavePINO VR=0.292 (2× improvement)
3. **Simple sea GP is excellent**: nRMSE≈0.14–0.16, Pearson≈0.80–0.83

### What changed from 6ep pilot:
1. VR=1.006 → 0.583 (seed0 bias corrected; claim is now "2× improvement" not "near-perfect")
2. Pearson=0.536 → 0.414 (complex sea fails for seeds 1–4)
3. nRMSE=0.242 → 0.228 (slightly better over full evaluation)

### Research problem (unchanged):
- Oracle (0.126) → practical (0.228) gap = K_disp estimation quality
- For complex sea (15–30 comp): direction estimation is the bottleneck
- Better direction estimation → close the gap → thesis future work

---

## Result Files
- `results/gp_disp_full_eval.json` — full 30-episode JSON (oracle + ws_multistart)
- `results/gp_disp_pilot.json` — 6-episode pilot (all methods)
- `scripts/gp_disp_pilot.py` — evaluation script
