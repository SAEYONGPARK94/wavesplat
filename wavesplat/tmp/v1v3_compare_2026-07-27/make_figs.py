"""v1 vs v3 mixed-retrain checkpoint comparison figure (2026-07-27).

Data sources:
  - tmp/mixed_retrain_2026-07-26/full40_exploratory.json  (v1, v3 per-episode, fresh 40)
  - tmp/heldout40_2026-07-26/arm_runs.jsonl               (old learned, multi-seed)
  - tmp/heldout40_2026-07-26/v2_arm.jsonl                 (analytic v2_konly)
  - RESULT.md gate table (frozen numbers, hardcoded below)
"""
import json, os
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/vexit/wavesplat/wavesplat/tmp"
OUT = os.path.join(BASE, "v1v3_compare_2026-07-27")

# Validated categorical palette (dataviz skill, light mode)
C_OLD = "#2a78d6"   # slot 1 blue  — old net
C_V1 = "#eb6834"    # slot 2 orange — v1
C_V3 = "#1baf7a"    # slot 3 aqua  — v3
C_REF = "#8a8983"   # neutral gray — analytic reference
INK = "#1a1a19"; INK2 = "#5f5e56"
GRID = dict(color="#e6e4dc", lw=0.8)

def slice_of(ep):
    import re
    n = int(re.search(r"_(\d+)comp_", ep).group(1))
    if n <= 2: return "sparse"
    if n == 5: return "moderate"
    return "crowded"

full40 = json.load(open(f"{BASE}/mixed_retrain_2026-07-26/full40_exploratory.json"))
v1, v3 = full40["v1"], full40["v3"]
eps = sorted(v1.keys())
assert len(eps) == 40

def per_ep_median(path, key="avg_nRMSE"):
    acc = {}
    for line in open(path):
        r = json.loads(line)
        acc.setdefault(r["episode"], []).append(r[key])
    return {e: median(v) for e, v in acc.items()}

old = per_ep_median(f"{BASE}/heldout40_2026-07-26/arm_runs.jsonl")
v2a = per_ep_median(f"{BASE}/heldout40_2026-07-26/v2_arm.jsonl", key="nRMSE")

def slice_mean(d, sl):
    vals = [d[e] for e in eps if (slice_of(e) == sl if sl else True)]
    return float(np.mean(vals))

slices = [("all", None), ("sparse", "sparse"), ("moderate", "moderate"), ("crowded", "crowded")]
arms = [("analytic v2", v2a, C_REF), ("old net", old, C_OLD), ("v1", v1, C_V1), ("v3", v3, C_V3)]
print("sanity (mean):")
for name, d, _ in arms:
    print(f"  {name:12s} " + "  ".join(f"{sl}={slice_mean(d, key):.5f}" for sl, key in slices))

fig = plt.figure(figsize=(16, 11), facecolor="white")
fig.suptitle("Mixed-curriculum retrain — checkpoint v1 vs v3 (fresh held-out 40, pure-learned arm)",
             fontsize=15, fontweight="bold", color=INK, y=0.985)
fig.text(0.5, 0.955, "v3 = only FULL PASS of frozen gates (safety bars) · v1 = best mean accuracy but fails cardinality bar G2a",
         ha="center", fontsize=10.5, color=INK2)

gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.92, bottom=0.06, hspace=0.38, wspace=0.22)

# ---------- A: slice-wise mean nRMSE ----------
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(slices)); w = 0.19
for i, (name, d, c) in enumerate(arms):
    vals = [slice_mean(d, key) for _, key in slices]
    b = ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=c, label=name)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.003 + (0.009 if i % 2 else 0), f"{v:.3f}",
                ha="center", fontsize=6.8, color=INK2)
ax.set_xticks(x, [f"{sl}\n(n={n})" for (sl, _), n in zip(slices, [40, 16, 8, 16])], fontsize=10)
ax.set_ylabel("mean nRMSE  (lower = better)", fontsize=10)
ax.set_title("A · Accuracy by regime — the trade-off", fontsize=12, fontweight="bold", color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_ylim(0, 0.26)

# ---------- B: paired per-episode scatter ----------
ax = fig.add_subplot(gs[0, 1])
reg_c = {"sparse": C_OLD, "moderate": C_V1, "crowded": C_V3}
lims = (0.002, 0.45)
for d, mk, lbl in [(v1, "o", "v1"), (v3, "^", "v3")]:
    for e in eps:
        ax.scatter(old[e], d[e], s=42 if mk == "o" else 46, marker=mk,
                   facecolor=reg_c[slice_of(e)], edgecolor="white", linewidth=1.2,
                   alpha=0.9 if mk == "o" else 0.75, zorder=3)
ax.plot(lims, lims, color=INK2, lw=1, ls="--", zorder=2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("old net nRMSE (per episode)", fontsize=10)
ax.set_ylabel("new checkpoint nRMSE", fontsize=10)
ax.set_title("B · Per-episode paired view — gains are catastrophe removal", fontsize=12,
             fontweight="bold", color=INK, loc="left")
ax.text(0.03, 0.93, "below diagonal = new better", transform=ax.transAxes, fontsize=8.5, color=INK2)
cat = "episode_BF2_1comp_seed888_slope.h5"
ax.annotate("BF2 1-comp catastrophe\nfixed: 0.329 → 0.017",
            xy=(old[cat], v1[cat]), xytext=(0.40, 0.045), ha="right", fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9, shrinkB=6))
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="#c3c2b7", mec="white", label="v1 (circle)"),
    Line2D([], [], marker="^", ls="", mfc="#c3c2b7", mec="white", label="v3 (triangle)"),
    Line2D([], [], marker="s", ls="", mfc=C_OLD, mec="white", label="sparse"),
    Line2D([], [], marker="s", ls="", mfc=C_V1, mec="white", label="moderate"),
    Line2D([], [], marker="s", ls="", mfc=C_V3, mec="white", label="crowded"),
], frameon=False, fontsize=8.5, loc="lower right", ncol=2)

# ---------- C: gate ladder ----------
gates = [
    ("G1 crowded mean\n(bar ≤ 0.17878)", [0.16052, 0.16379, 0.17491], 0.17878, 0.16878, "%.4f"),
    ("G2a max n_selected\n(bar ≤ 6)", [12, 8, 6], 6, 30, "%.0f"),
    ("G2b 1-comp mean\n(bar ≤ 0.01855)", [0.01128, 0.01092, 0.01047], 0.01855, 0.03567, "%.4f"),
]
sub = gs[1, 0].subgridspec(1, 3, wspace=0.45)
round_cols = [C_V1, "#c3c2b7", C_V3]
for gi, (name, vals, bar, oldv, fmt) in enumerate(gates):
    ax = fig.add_subplot(sub[0, gi])
    xs = np.arange(3)
    bs = ax.bar(xs, vals, 0.62, color=round_cols)
    ax.axhline(bar, color="#e34948", lw=1.4, ls="--")
    ax.axhline(oldv, color=INK2, lw=1.2, ls=":")
    top = max(max(vals), bar, oldv)
    ax.set_ylim(0, top * 1.28)
    for r, v in zip(bs, vals):
        passed = v <= bar + 1e-12
        ax.text(r.get_x() + r.get_width() / 2, v + top * 0.03, (fmt % v), ha="center", fontsize=7.5, color=INK2)
        ax.text(r.get_x() + r.get_width() / 2, top * 1.18, "PASS" if passed else "FAIL", ha="center",
                fontsize=8, fontweight="bold", color="#008300" if passed else "#e34948")
    ax.set_xticks(xs, ["r1\n(v1)", "r2", "r3\n(v3)"], fontsize=8.5)
    ax.set_title(name, fontsize=9, color=INK)
    if gi == 0:
        ax.text(-0.42, bar * 1.01, "frozen bar", fontsize=7.5, color="#e34948", va="bottom", ha="left")
        ax.text(-0.42, oldv * 0.99, "old net", fontsize=7.5, color=INK2, va="top", ha="left")
fig.text(0.06, 0.475, "C · Frozen gates by round — sparse cardinality was bought with crowded accuracy",
         fontsize=12, fontweight="bold", color=INK)

# ---------- D: router degeneracy + paired-median verdicts ----------
ax = fig.add_subplot(gs[1, 1])
nets = ["old net", "v1", "v3"]; routed = [4, 3, 0]
cols = [C_OLD, C_V1, C_V3]
b = ax.barh([2, 1, 0], routed, 0.55, color=cols)
for y, v in zip([2, 1, 0], routed):
    ax.text(v + 0.25, y, f"{v}/16", va="center", fontsize=10, color=INK)
ax.set_yticks([2, 1, 0], nets, fontsize=10)
ax.set_xlim(0, 16); ax.set_xlabel("sparse episodes the deployed router sends to the learned arm", fontsize=9.5)
ax.set_title("D · Router degeneracy — and the paired-median caveat", fontsize=12,
             fontweight="bold", color=INK, loc="left")
ax.text(0.98, 0.60,
        "v3's learned arm now BEATS analytic on sparse\n(0.0105 vs 0.0135) — but the router sends 0/16 there.\n"
        "Router was tuned to the old net's pathology.\n\n"
        "Paired median vs old net (all 40):\n"
        "  v1: p = 0.54 (no change)\n"
        "  v3: p = 0.029 (slightly WORSE)\n"
        "Mean gains = catastrophe removal,\nnot a broad accuracy shift.",
        transform=ax.transAxes, fontsize=9, color=INK, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#faf9f5", ec="#e6e4dc"))

for ax in fig.get_axes():
    ax.set_facecolor("white")
    ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8.5)

out = os.path.join(OUT, "v1_v3_comparison.png")
fig.savefig(out, dpi=150, facecolor="white")
print("saved", out)

# Paired sanity: recompute sign counts vs old
for name, d in [("v1", v1), ("v3", v3)]:
    diffs = [d[e] - old[e] for e in eps]
    better = sum(1 for x in diffs if x < 0)
    print(f"{name} vs old: better on {better}/40, medianΔ {median(diffs):+.5f}")
