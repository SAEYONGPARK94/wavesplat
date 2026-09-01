# Cover letter draft — Ocean Engineering (encounter forecasting paper)
(2026-07-06 초안. 투고 시 elsarticle 커버레터 양식으로 변환.)

Dear Editors,

We submit for your consideration the manuscript "Control-compatible
real-time wave encounter forecasting with a given forward-looking
LiDAR: information limits, streaming equivalence, and self-diagnosed
reliability" for publication in Ocean Engineering.

Forward-looking wave sensing for vessel control is an established
concept — patents, commercial radar systems, and LiDAR deterministic
wave prediction all exist. What this manuscript contributes is the
question those works leave open: for a specific, commercially
available sensor on a specific fast vessel, where does the information
in the observation end, can a real-time estimator reach that boundary,
and can the system detect online when its forecast fails?

Three results: (1) an encounter-frame application of predictable-zone
theory that yields a quantified design verdict — the feedforward
preview band of a 22-kt foiling vessel is served at correlation 0.91
(95 % CI 0.88–0.94), with a ladder of falsification experiments
(true-spectrum oracle, budget, learning scale, sampling structure,
nonlinearity) showing the estimator saturated within the examined
class; (2) an exact equivalence between the batch estimator and a
streaming recursive filter, with measured latency of 18 ms per 10-Hz
frame — a factor five inside the control budget — and zero-shot
transfer to five real-sea stereo sites (pooled correlation 0.70);
(3) a self-diagnosis study reported with unusual candor: a
site-adaptive reliability gate that raises conditional skill from 0.49
to 0.68 at 50 % availability in exploratory analysis, together with
two pre-registered confirmation attempts (protocols git-timestamped
before execution) that reproduce its direction but not its
significance — we report the gate as preliminary and publish the
trail.

Every headline number in the manuscript re-derives from released
result files via a single verification script (31/31), and the
validation data are the open CC-BY stereo records of Guimarães et al.
(2020). We believe the combination of a quantified design verdict,
measured real-time performance, multi-site zero-shot validation, and
pre-registered negative results will be of interest to Ocean
Engineering's readership.

This manuscript is distinct from our submitted method paper
("WaveSplat: grid-free spectral wave field estimation from sparse
shipboard LiDAR"): that work proposes a nowcast estimation method and
benchmarks it against assimilation baselines; the present work asks a
system-level forecasting question with a different estimator, and
cites the former as prior work.

[Suggested reviewers / declarations / etc.]
