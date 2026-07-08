"""K1 analysis — does the lead-decay break scale with memory W? (H6 test)

Reads falcon_real_operator_YS17_Wsweep.json (W in {5,10,20,40,60}) and prints
the pooled per-lead nRMSE curve per W (fixed normalizer), plus the first lead
where the curve crosses thresholds. H6 (window phase horizon, delta_omega ~
2*pi/W) predicts the crossing lead grows roughly linearly with W; H6 is
rejected if the curves collapse onto one another regardless of W.
"""
import json
import numpy as np

d = json.load(open(
    "results/direction3/falcon_real_operator_YS17_Wsweep.json"))
leads = np.asarray(d["leads"], float)
wins = d["windows"]
gall = np.concatenate([np.asarray(w["gt"], float)[np.isfinite(w["gt"])]
                       for w in wins])
sg = np.std(gall)

print(f"YS17 W-sweep: {len(wins)} windows, sigma_g={sg:.3f} m")
print(f"{'W':>4} | per-lead nRMSE (fixed normalizer)")
curves = {}
for W in d["w_mems"]:
    Wk = str(float(W))
    row = []
    for li in range(len(leads)):
        es = []
        for w in wins:
            gt = np.asarray(w["gt"], float)
            pr = np.asarray(w["by_W"][Wk]["pred"], float)
            n = min(len(gt), len(pr))
            if li < n and np.isfinite(gt[li]) and np.isfinite(pr[li]):
                es.append(pr[li] - gt[li])
        row.append(np.sqrt(np.mean(np.square(es))) / sg if len(es) >= 4
                   else np.nan)
    curves[W] = np.array(row)
    s = " ".join(f"{v:.2f}" if np.isfinite(v) else " -- " for v in row)
    print(f"{W:4.0f} | {s}")

print("\nlead axis:", " ".join(f"{l:4.0f}" for l in leads))
print("\nfirst lead crossing threshold (2-lead smoothed):")
print(f"{'W':>4} | tau(>0.8) tau(>1.0) | band nRMSE 0-2s / 0-5s / 0-10s / 10-20s")
for W in d["w_mems"]:
    c = curves[W]
    sm = np.convolve(np.nan_to_num(c, nan=2.0), [0.5, 0.5], mode="same")
    def cross(th):
        for li in range(1, len(leads)):
            if sm[li] > th:
                return leads[li]
        return np.inf
    Wk = str(float(W))
    bands = []
    for lo, hi in [(0, 2), (0, 5), (0, 10), (10, 20)]:
        es, gs = [], []
        for w in wins:
            gt = np.asarray(w["gt"], float)
            pr = np.asarray(w["by_W"][Wk]["pred"], float)
            n = min(len(gt), len(pr), len(leads))
            m = (leads[:n] >= lo) & (leads[:n] <= hi) & \
                np.isfinite(gt[:n]) & np.isfinite(pr[:n])
            es.append((pr[:n] - gt[:n])[m]); gs.append(gt[:n][m])
        e = np.concatenate(es); g = np.concatenate(gs)
        bands.append(np.sqrt(np.mean(e ** 2)) / np.std(g))
    print(f"{W:4.0f} | {cross(0.8):8.1f} {cross(1.0):8.1f} | "
          + " / ".join(f"{b:.2f}" for b in bands))
